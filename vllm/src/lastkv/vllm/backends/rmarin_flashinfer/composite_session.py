"""Framework-neutral RMarin composite KV session.

The serving-framework adapters intentionally remain thin.  This module owns
the state transition that both vLLM and SGLang need:

* one canonical, full-history, raw top-layer K/V store;
* one ``2 * chunk_size`` ring store for every target layer;
* a deferred boundary commit which turns the just-completed raw local chunk
  into the learned source/local blend;
* exact, disjoint source/local materializations for FlashInfer.

The boundary commit is deliberately performed at the *next* forward's chunk
boundary, not when the final layer first writes the boundary token.  The
current attention call must still see same-layer raw K/V for its current
chunk.  Committing before that call is a subtle semantic error.

There are no eager torch or FlashInfer imports.  ``TorchRMarinTensorOps`` and
``TorchFlashInferMaterializer`` import their GPU dependencies only when
instantiated or called.  Tests can inject dependency-free implementations of
the two small protocols below.

The P0 implementation materializes target-layer source values and the exact
local range into scratch NHD pages.  It is correct and directly executable,
but it is not the final fused gather/transform kernel.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
import math
from typing import Any, Protocol, runtime_checkable

from .access_plan import RMarinAccessPlan, TokenRange
from .bindings import (
    CallbackRMarinFrameworkBindings,
    RMarinLayerResources,
    RMarinRequestSpan,
)
from .runtime_contract import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CHUNK_SIZE,
    RMarinBackendContractError,
    RMarinRuntimeConstraints,
    require_supported_chunk_size,
)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RMarinBackendContractError(f"{name} must be a positive integer")
    return value


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RMarinBackendContractError(f"{name} must be a non-negative integer")
    return value


def _request_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RMarinBackendContractError("request_id must be a non-empty string")
    return value


@dataclass(frozen=True)
class RMarinTensorDescriptor:
    """Minimal tensor identity needed to reject mixed cache storage."""

    shape: tuple[int, ...]
    dtype: str
    device: str

    def __post_init__(self) -> None:
        try:
            shape = tuple(int(item) for item in self.shape)
        except (TypeError, ValueError) as exc:
            raise RMarinBackendContractError("tensor shape is invalid") from exc
        if len(shape) != 3 or any(item < 1 for item in shape):
            raise RMarinBackendContractError(
                "RMarin K/V tensors must have [tokens, kv_heads, head_dim] shape"
            )
        if not isinstance(self.dtype, str) or not self.dtype:
            raise RMarinBackendContractError("tensor dtype is missing")
        if not isinstance(self.device, str) or not self.device:
            raise RMarinBackendContractError("tensor device is missing")
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True)
class RMarinLayerParameters:
    """Per-target-layer recurrent cache parameters.

    ``alpha_logits`` is one value per KV head.  ``value_scales`` is also
    per-head for ``v`` and ``vnorm``.  ``value_norm_weights`` is one value per
    head-dimension channel for ``vnormqk``.
    """

    alpha_logits: Any
    source_mode: str = "none"
    value_scales: Any | None = None
    value_norm_weights: Any | None = None
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.source_mode not in {"none", "v", "vnorm", "vnormqk"}:
            raise RMarinBackendContractError(
                "source_mode must be none, v, vnorm, or vnormqk"
            )
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float)):
            raise RMarinBackendContractError("epsilon must be finite and positive")
        epsilon = float(self.epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise RMarinBackendContractError("epsilon must be finite and positive")
        if self.source_mode == "none" and (
            self.value_scales is not None or self.value_norm_weights is not None
        ):
            raise RMarinBackendContractError(
                "source_mode=none may not carry source scale parameters"
            )
        if self.source_mode in {"v", "vnorm"} and (
            self.value_scales is None or self.value_norm_weights is not None
        ):
            raise RMarinBackendContractError(
                f"source_mode={self.source_mode} requires only value_scales"
            )
        if self.source_mode == "vnormqk" and (
            self.value_norm_weights is None or self.value_scales is not None
        ):
            raise RMarinBackendContractError(
                "source_mode=vnormqk requires only value_norm_weights"
            )
        object.__setattr__(self, "epsilon", epsilon)


@dataclass
class RMarinKVStore:
    """One mutable token-major K/V allocation."""

    key: Any
    value: Any
    capacity: int

    def __post_init__(self) -> None:
        _positive_int("capacity", self.capacity)
        if self.key is self.value:
            raise RMarinBackendContractError(
                "key and value must use distinct tensor allocations"
            )

    @property
    def kernel_tuple(self) -> tuple[Any, Any]:
        return self.key, self.value


@dataclass(frozen=True)
class RMarinExactPageTable:
    """Auditable page table for one materialized logical range."""

    logical_range: TokenRange
    page_size: int
    page_indices: tuple[int, ...]
    last_page_length: int
    query_tokens: int
    causal: bool

    def __post_init__(self) -> None:
        if not isinstance(self.logical_range, TokenRange) or self.logical_range.empty:
            raise RMarinBackendContractError(
                "FlashInfer page table requires a non-empty logical range"
            )
        page_size = _positive_int("page_size", self.page_size)
        query_tokens = _positive_int("query_tokens", self.query_tokens)
        pages = (self.logical_range.length + page_size - 1) // page_size
        expected_indices = tuple(range(pages))
        if self.page_indices != expected_indices:
            raise RMarinBackendContractError(
                "scratch page indices must be a contiguous zero-based range"
            )
        expected_last = self.logical_range.length - (pages - 1) * page_size
        if self.last_page_length != expected_last:
            raise RMarinBackendContractError(
                "last_page_length differs from the exact logical range"
            )
        if query_tokens > self.logical_range.length:
            raise RMarinBackendContractError(
                "query tokens cannot exceed the materialized K/V range"
            )
        if not isinstance(self.causal, bool):
            raise RMarinBackendContractError("causal must be a bool")

    @property
    def token_count(self) -> int:
        return self.logical_range.length


class RMarinPlannedFlashInferWrapper:
    """Delegating wrapper which retains and enforces its exact plan receipt."""

    def __init__(
        self,
        wrapper: Any,
        page_table: RMarinExactPageTable,
    ) -> None:
        if wrapper is None:
            raise RMarinBackendContractError("FlashInfer wrapper is missing")
        if not isinstance(page_table, RMarinExactPageTable):
            raise RMarinBackendContractError("page_table is invalid")
        self.wrapper = wrapper
        self.page_table = page_table

    @staticmethod
    def _validate_run_kwargs(kwargs: Mapping[str, Any]) -> None:
        forbidden = {"causal", "window_left", "pos_encoding_mode"} & set(kwargs)
        if forbidden:
            raise RMarinBackendContractError(
                "wrapper-plan arguments may not be passed to run_return_lse: "
                + ", ".join(sorted(forbidden))
            )

    def run_return_lse(self, query: Any, kv_cache: Any, **kwargs: Any) -> Any:
        self._validate_run_kwargs(kwargs)
        method = getattr(self.wrapper, "run_return_lse", None)
        if not callable(method):
            method = getattr(self.wrapper, "forward_return_lse", None)
        if not callable(method):
            raise RMarinBackendContractError(
                "planned FlashInfer wrapper has no return-LSE method"
            )
        return method(query, kv_cache, **kwargs)

    def forward_return_lse(self, query: Any, kv_cache: Any, **kwargs: Any) -> Any:
        return self.run_return_lse(query, kv_cache, **kwargs)


@dataclass(frozen=True)
class RMarinMaterializedPass:
    """Kernel-ready paged K/V plus its pre-planned wrapper."""

    wrapper: RMarinPlannedFlashInferWrapper
    kv_cache: Any
    page_table: RMarinExactPageTable

    def __post_init__(self) -> None:
        if not isinstance(self.wrapper, RMarinPlannedFlashInferWrapper):
            raise RMarinBackendContractError("wrapper must retain an exact page table")
        if self.kv_cache is None:
            raise RMarinBackendContractError("materialized K/V cache is missing")
        if self.wrapper.page_table != self.page_table:
            raise RMarinBackendContractError(
                "wrapper and materialized pass page tables differ"
            )


@dataclass(frozen=True)
class RMarinCompositeMemoryBudget:
    """Conservative BF16 cache/workspace/scratch memory accounting."""

    persistent_kv_bytes: int
    flashinfer_workspace_bytes: int
    attention_scratch_upper_bound_bytes: int
    boundary_scratch_upper_bound_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "persistent_kv_bytes",
            "flashinfer_workspace_bytes",
            "attention_scratch_upper_bound_bytes",
            "boundary_scratch_upper_bound_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RMarinBackendContractError(
                    f"{name} must be a non-negative integer"
                )

    @property
    def preprofile_reserved_bytes(self) -> int:
        return self.persistent_kv_bytes + self.flashinfer_workspace_bytes

    @property
    def dynamic_scratch_upper_bound_bytes(self) -> int:
        return max(
            self.attention_scratch_upper_bound_bytes,
            self.boundary_scratch_upper_bound_bytes,
        )

    @property
    def reserved_plus_peak_bytes(self) -> int:
        return self.preprofile_reserved_bytes + self.dynamic_scratch_upper_bound_bytes


@runtime_checkable
class RMarinTensorOps(Protocol):
    """Tensor/storage primitives used by :class:`RMarinCompositeKVSession`."""

    def describe(self, value: Any) -> RMarinTensorDescriptor:
        """Return shape, dtype, and device without changing the tensor."""

    def allocate(self, capacity: int, like: Any) -> Any:
        """Allocate one token-major tensor like ``like``."""

    def write(self, destination: Any, start: int, source: Any) -> None:
        """Copy a contiguous source into destination at a physical offset."""

    def gather(self, source: Any, indices: Sequence[int]) -> Any:
        """Return a distinct token-major scratch in the requested order."""

    def transform_source(
        self,
        key: Any,
        value: Any,
        parameters: RMarinLayerParameters,
    ) -> tuple[Any, Any]:
        """Return target-layer transformed source scratch."""

    def blend_recent(
        self,
        local_key: Any,
        local_value: Any,
        raw_source_key: Any,
        raw_source_value: Any,
        parameters: RMarinLayerParameters,
    ) -> tuple[Any, Any]:
        """Return materialized recent effective K/V."""


@runtime_checkable
class RMarinFlashInferMaterializer(Protocol):
    """Build exact scratch pages and a planned return-LSE wrapper."""

    def reserve(self, like: Any) -> None:
        """Allocate both source/local workspaces before framework profiling."""

    def materialize(
        self,
        *,
        role: str,
        query: Any,
        key: Any,
        value: Any,
        logical_range: TokenRange,
        page_size: int,
        causal: bool,
        sm_scale: float | None,
    ) -> RMarinMaterializedPass:
        """Materialize and plan one exact source or local pass."""


class RMarinCompositeKVSession:
    """Single-request, TP/PP=1 composite RMarin cache state machine."""

    def __init__(
        self,
        *,
        request_id: str,
        max_sequence_tokens: int,
        num_layers: int,
        source_layer_index: int,
        layer_parameters: Sequence[RMarinLayerParameters],
        tensor_ops: RMarinTensorOps,
        materializer: RMarinFlashInferMaterializer,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        page_size: int = DEFAULT_BLOCK_SIZE,
        sm_scale: float | None = None,
    ) -> None:
        self.request_id = _request_id(request_id)
        self.max_sequence_tokens = _positive_int(
            "max_sequence_tokens", max_sequence_tokens
        )
        self.num_layers = _positive_int("num_layers", num_layers)
        self.source_layer_index = _non_negative_int(
            "source_layer_index", source_layer_index
        )
        if self.source_layer_index != self.num_layers - 1:
            raise RMarinBackendContractError(
                "RMarin shared source must be the final transformer layer"
            )
        self.chunk_size = require_supported_chunk_size(chunk_size)
        self.page_size = _positive_int("page_size", page_size)
        if self.page_size != DEFAULT_BLOCK_SIZE or self.chunk_size % self.page_size:
            raise RMarinBackendContractError(
                f"RMarin P0 requires page_size={DEFAULT_BLOCK_SIZE}"
            )
        try:
            parameters = tuple(layer_parameters)
        except TypeError as exc:
            raise RMarinBackendContractError(
                "layer_parameters must be a sequence"
            ) from exc
        if len(parameters) != self.num_layers or any(
            not isinstance(item, RMarinLayerParameters) for item in parameters
        ):
            raise RMarinBackendContractError(
                "layer_parameters must contain one RMarinLayerParameters per layer"
            )
        if not isinstance(tensor_ops, RMarinTensorOps):
            raise RMarinBackendContractError(
                "tensor_ops does not implement the RMarinTensorOps protocol"
            )
        if not isinstance(materializer, RMarinFlashInferMaterializer):
            raise RMarinBackendContractError(
                "materializer does not implement the RMarinFlashInferMaterializer "
                "protocol"
            )
        if sm_scale is not None:
            if isinstance(sm_scale, bool) or not isinstance(sm_scale, (int, float)):
                raise RMarinBackendContractError("sm_scale must be finite and positive")
            sm_scale = float(sm_scale)
            if not math.isfinite(sm_scale) or sm_scale <= 0:
                raise RMarinBackendContractError("sm_scale must be finite and positive")
        self.layer_parameters = parameters
        self.tensor_ops = tensor_ops
        self.materializer = materializer
        self.sm_scale = sm_scale
        self._source: RMarinKVStore | None = None
        self._locals: list[RMarinKVStore | None] = [None] * self.num_layers
        self._local_tags: list[list[int | None]] = [
            [None] * (2 * self.chunk_size) for _ in range(self.num_layers)
        ]
        self._layer_next: list[int] = [0] * self.num_layers
        self._source_next = 0
        self._committed_boundary = 0
        self._active_span: RMarinRequestSpan | None = None
        self._active_layers: set[int] = set()
        self._signature: RMarinTensorDescriptor | None = None
        self._poison_reason: str | None = None
        self._persistent_reserved = False

    @property
    def poisoned(self) -> bool:
        return self._poison_reason is not None

    @property
    def poison_reason(self) -> str | None:
        return self._poison_reason

    @property
    def persistent_reserved(self) -> bool:
        return self._persistent_reserved

    @property
    def canonical_source_store(self) -> RMarinKVStore | None:
        return self._source

    @property
    def local_stores(self) -> tuple[RMarinKVStore | None, ...]:
        return tuple(self._locals)

    def memory_budget(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        bytes_per_element: int = 2,
    ) -> RMarinCompositeMemoryBudget:
        """Return the P0 persistent and conservative dynamic byte budget.

        The attention bound accounts for local gather+pack and raw
        gather+transform+pack.  The boundary bound keeps all layer replacement
        chunks until the atomic commit and adds conservative transform
        temporaries.  FlashInfer output/logsumexp buffers are framework/kernel
        concerns and are not included here.
        """

        heads = _positive_int("num_kv_heads", num_kv_heads)
        width = _positive_int("head_dim", head_dim)
        element = _positive_int("bytes_per_element", bytes_per_element)
        kv_token_bytes = 2 * heads * width * element
        persistent_slots = self.max_sequence_tokens + (
            self.num_layers * 2 * self.chunk_size
        )
        maximum_position = self.max_sequence_tokens - 1
        current_start = (maximum_position // self.chunk_size) * self.chunk_size
        maximum_older = max(0, current_start - self.chunk_size)
        maximum_local = min(self.max_sequence_tokens, 2 * self.chunk_size)
        attention_slots = 3 * maximum_older + 2 * maximum_local
        boundary_slots = (self.num_layers + 6) * self.chunk_size
        workspace_bytes = getattr(
            self.materializer,
            "reserved_workspace_bytes",
            0,
        )
        if callable(workspace_bytes):
            workspace_bytes = workspace_bytes()
        if (
            isinstance(workspace_bytes, bool)
            or not isinstance(workspace_bytes, int)
            or workspace_bytes < 0
        ):
            raise RMarinBackendContractError(
                "materializer reserved_workspace_bytes is invalid"
            )
        return RMarinCompositeMemoryBudget(
            persistent_kv_bytes=persistent_slots * kv_token_bytes,
            flashinfer_workspace_bytes=workspace_bytes,
            attention_scratch_upper_bound_bytes=attention_slots * kv_token_bytes,
            boundary_scratch_upper_bound_bytes=boundary_slots * kv_token_bytes,
        )

    @property
    def completed_tokens(self) -> int:
        if len(set(self._layer_next)) != 1 or self._source_next != self._layer_next[0]:
            raise RMarinBackendContractError(
                "session layer/source positions are not synchronized"
            )
        return self._source_next

    def _require_healthy(self) -> None:
        if self._poison_reason is not None:
            raise RMarinBackendContractError(
                "RMarin composite session is poisoned: " + self._poison_reason
            )

    def _poison(self, exc: BaseException) -> None:
        self._poison_reason = f"{type(exc).__name__}: {exc}"

    def reset_request(self, request_id: str) -> None:
        """Clear logical ownership and begin a new request.

        Profile-visible reserved allocations remain alive and are safely reused;
        stale contents are inaccessible because tags and valid prefixes reset.
        """

        self.request_id = _request_id(request_id)
        if not self._persistent_reserved:
            self._source = None
            self._locals = [None] * self.num_layers
        self._local_tags = [
            [None] * (2 * self.chunk_size) for _ in range(self.num_layers)
        ]
        self._layer_next = [0] * self.num_layers
        self._source_next = 0
        self._committed_boundary = 0
        self._active_span = None
        self._active_layers = set()
        if not self._persistent_reserved:
            self._signature = None
        self._poison_reason = None

    def reserve(self, key_template: Any, value_template: Any) -> None:
        """Preallocate persistent K/V and both FlashInfer workspaces.

        This must run during model construction, before vLLM/SGLang memory
        profiling.  It is idempotent for an identical tensor signature and
        does not mark any token initialized.
        """

        self._require_healthy()
        if (
            self._active_span is not None
            or any(self._layer_next)
            or self._source_next
            or self._committed_boundary
        ):
            raise RMarinBackendContractError(
                "persistent cache reservation must precede request execution"
            )
        try:
            self._validate_pair(key_template, value_template, 1)
            if self._persistent_reserved:
                return
            if self._source is None:
                self._source = self._allocate_store(
                    self.max_sequence_tokens,
                    key_template,
                    value_template,
                )
            for layer_index, store in enumerate(self._locals):
                if store is None:
                    self._locals[layer_index] = self._allocate_store(
                        2 * self.chunk_size,
                        key_template,
                        value_template,
                    )
            self.materializer.reserve(key_template)
            self._persistent_reserved = True
        except Exception as exc:
            self._poison(exc)
            raise

    def preallocate_persistent(self, key_template: Any, value_template: Any) -> None:
        """Compatibility alias for :meth:`reserve`."""

        self.reserve(key_template, value_template)

    def _validate_pair(self, key: Any, value: Any, tokens: int) -> None:
        key_descriptor = self.tensor_ops.describe(key)
        value_descriptor = self.tensor_ops.describe(value)
        if key_descriptor != value_descriptor:
            raise RMarinBackendContractError(
                "key and value tensor shape/dtype/device must match"
            )
        if key_descriptor.shape[0] != tokens:
            raise RMarinBackendContractError(
                "K/V token count differs from the prepared request span"
            )
        if key_descriptor.dtype not in {"bfloat16", "bf16", "torch.bfloat16"}:
            raise RMarinBackendContractError("RMarin P0 requires BF16 K/V tensors")
        tail_signature = RMarinTensorDescriptor(
            shape=(1, key_descriptor.shape[1], key_descriptor.shape[2]),
            dtype=key_descriptor.dtype,
            device=key_descriptor.device,
        )
        if self._signature is None:
            self._signature = tail_signature
        elif self._signature != tail_signature:
            raise RMarinBackendContractError(
                "RMarin session cannot mix KV head shape, dtype, or device"
            )
        return key_descriptor

    def _allocate_store(self, capacity: int, key: Any, value: Any) -> RMarinKVStore:
        key_store = self.tensor_ops.allocate(capacity, key)
        value_store = self.tensor_ops.allocate(capacity, value)
        if key_store is value_store:
            raise RMarinBackendContractError(
                "tensor_ops returned aliased key/value allocations"
            )
        return RMarinKVStore(key=key_store, value=value_store, capacity=capacity)

    def _write_ring(
        self,
        *,
        layer_index: int,
        logical_start: int,
        key: Any,
        value: Any,
    ) -> None:
        store = self._locals[layer_index]
        if store is None:
            store = self._allocate_store(2 * self.chunk_size, key, value)
            self._locals[layer_index] = store
        tokens = self.tensor_ops.describe(key).shape[0]
        logical_positions = tuple(range(logical_start, logical_start + tokens))
        first_physical = logical_start % store.capacity
        first_count = min(tokens, store.capacity - first_physical)
        if first_count:
            first_indices = tuple(range(first_count))
            first_key = self.tensor_ops.gather(key, first_indices)
            first_value = self.tensor_ops.gather(value, first_indices)
            self.tensor_ops.write(store.key, first_physical, first_key)
            self.tensor_ops.write(store.value, first_physical, first_value)
        remaining = tokens - first_count
        if remaining:
            remaining_indices = tuple(range(first_count, tokens))
            remaining_key = self.tensor_ops.gather(key, remaining_indices)
            remaining_value = self.tensor_ops.gather(value, remaining_indices)
            self.tensor_ops.write(store.key, 0, remaining_key)
            self.tensor_ops.write(store.value, 0, remaining_value)
        for logical in logical_positions:
            self._local_tags[layer_index][logical % store.capacity] = logical

    def _write_source(
        self,
        *,
        logical_start: int,
        key: Any,
        value: Any,
    ) -> None:
        if self._source is None:
            self._source = self._allocate_store(
                self.max_sequence_tokens,
                key,
                value,
            )
        tokens = self.tensor_ops.describe(key).shape[0]
        end = logical_start + tokens
        if end > self._source.capacity:
            raise RMarinBackendContractError(
                "raw top-layer source write exceeds max_sequence_tokens"
            )
        self.tensor_ops.write(self._source.key, logical_start, key)
        self.tensor_ops.write(self._source.value, logical_start, value)

    def _gather_ring(
        self,
        layer_index: int,
        token_range: TokenRange,
    ) -> tuple[Any, Any]:
        store = self._locals[layer_index]
        if store is None:
            raise RMarinBackendContractError("layer-local cache is uninitialized")
        logical = tuple(range(token_range.start, token_range.end))
        missing = [
            position
            for position in logical
            if self._local_tags[layer_index][position % store.capacity] != position
        ]
        if missing:
            raise RMarinBackendContractError(
                "layer-local ring has stale or missing logical positions: "
                + ",".join(str(item) for item in missing[:8])
            )
        physical = tuple(position % store.capacity for position in logical)
        return (
            self.tensor_ops.gather(store.key, physical),
            self.tensor_ops.gather(store.value, physical),
        )

    def _gather_source(self, token_range: TokenRange) -> tuple[Any, Any]:
        if self._source is None or token_range.end > self._source_next:
            raise RMarinBackendContractError(
                "canonical raw source range is uninitialized"
            )
        indices = tuple(range(token_range.start, token_range.end))
        return (
            self.tensor_ops.gather(self._source.key, indices),
            self.tensor_ops.gather(self._source.value, indices),
        )

    def _commit_previous_boundary(self, end: int) -> None:
        if end == self._committed_boundary:
            return
        if end != self._committed_boundary + self.chunk_size:
            raise RMarinBackendContractError(
                "boundary commits must be contiguous and exactly one chunk"
            )
        if end % self.chunk_size or end > self._source_next:
            raise RMarinBackendContractError(
                "cannot commit an incomplete recurrent chunk"
            )
        if any(position < end for position in self._layer_next):
            raise RMarinBackendContractError(
                "cannot commit before every layer has written the completed chunk"
            )
        recent = TokenRange(end - self.chunk_size, end)
        raw_key, raw_value = self._gather_source(recent)
        replacements: list[tuple[Any, Any]] = []
        for layer_index, parameters in enumerate(self.layer_parameters):
            local_key, local_value = self._gather_ring(layer_index, recent)
            replacements.append(
                self.tensor_ops.blend_recent(
                    local_key,
                    local_value,
                    raw_key,
                    raw_value,
                    parameters,
                )
            )
        # Every transform above succeeded before the first persistent write.
        # A device failure during the following writes poisons the session.
        for layer_index, (key, value) in enumerate(replacements):
            self._write_ring(
                layer_index=layer_index,
                logical_start=recent.start,
                key=key,
                value=value,
            )
        self._committed_boundary = end

    def prepare_forward(self, span: RMarinRequestSpan) -> None:
        """Validate one boundary-safe span and commit the prior chunk if needed."""

        self._require_healthy()
        if not isinstance(span, RMarinRequestSpan):
            raise RMarinBackendContractError("span must be RMarinRequestSpan")
        if span.chunk_size != self.chunk_size:
            raise RMarinBackendContractError(
                "request span chunk size differs from the composite session"
            )
        if span.num_computed_tokens + span.num_query_tokens > self.max_sequence_tokens:
            raise RMarinBackendContractError("request span exceeds max_sequence_tokens")
        if span.num_computed_tokens == 0 and (
            span.request_id != self.request_id
            or any(self._layer_next)
            or self._source_next
        ):
            self.reset_request(span.request_id)
        if span.request_id != self.request_id:
            raise RMarinBackendContractError(
                "non-zero request span cannot replace active request state"
            )
        if (
            self._active_span is not None
            and len(self._active_layers) != self.num_layers
        ):
            raise RMarinBackendContractError(
                "previous forward did not visit every transformer layer"
            )
        start = span.num_computed_tokens
        if any(position != start for position in self._layer_next):
            raise RMarinBackendContractError(
                "request span is not contiguous with every layer-local cache"
            )
        if self._source_next != start:
            raise RMarinBackendContractError(
                "request span is not contiguous with canonical raw source cache"
            )
        try:
            if start and start % self.chunk_size == 0:
                self._commit_previous_boundary(start)
        except Exception as exc:
            self._poison(exc)
            raise
        self._active_span = span
        self._active_layers = set()

    def prepare_layer(
        self,
        *,
        layer_index: int,
        query: Any,
        key: Any,
        value: Any,
        request_span: RMarinRequestSpan,
        access_plan: RMarinAccessPlan,
        save_kv_cache: bool = True,
    ) -> RMarinLayerResources:
        """Write one layer and build exact source/local FlashInfer resources."""

        self._require_healthy()
        index = _non_negative_int("layer_index", layer_index)
        if index >= self.num_layers:
            raise RMarinBackendContractError("layer_index is out of range")
        if self._active_span != request_span:
            raise RMarinBackendContractError(
                "prepare_layer span differs from prepare_forward"
            )
        if index != len(self._active_layers) or index in self._active_layers:
            raise RMarinBackendContractError(
                "RMarin P0 requires each layer exactly once in increasing order"
            )
        if save_kv_cache is not True:
            raise RMarinBackendContractError(
                "RMarin attention cannot run with KV cache writes disabled"
            )
        if not isinstance(access_plan, RMarinAccessPlan):
            raise RMarinBackendContractError("access_plan is invalid")
        if (
            access_plan.layer_spec.layer_index != index
            or access_plan.layer_spec.source_layer_index != self.source_layer_index
            or access_plan.request_state.request_id != self.request_id
            or access_plan.request_state.query_position
            != request_span.final_query_position
        ):
            raise RMarinBackendContractError(
                "access plan does not match the active request/layer/span"
            )
        start = request_span.num_computed_tokens
        if self._layer_next[index] != start:
            raise RMarinBackendContractError(
                "layer write is not contiguous with its local ring"
            )
        try:
            self._validate_pair(key, value, request_span.num_query_tokens)
            query_descriptor = self.tensor_ops.describe(query)
            if (
                query_descriptor.shape[0] != request_span.num_query_tokens
                or query_descriptor.shape[2] != self.tensor_ops.describe(key).shape[2]
                or query_descriptor.dtype != self.tensor_ops.describe(key).dtype
                or query_descriptor.device != self.tensor_ops.describe(key).device
            ):
                raise RMarinBackendContractError(
                    "query and K/V token/head-dimension/dtype/device are incompatible"
                )
            self._write_ring(
                layer_index=index,
                logical_start=start,
                key=key,
                value=value,
            )
            end = start + request_span.num_query_tokens
            self._layer_next[index] = end
            if index == self.source_layer_index:
                if self._source_next != start:
                    raise RMarinBackendContractError(
                        "raw source write is not contiguous"
                    )
                self._write_source(
                    logical_start=start,
                    key=key,
                    value=value,
                )
                self._source_next = end

            local_key, local_value = self._gather_ring(
                index,
                access_plan.local_pass.token_range,
            )
            local_pass = self.materializer.materialize(
                role="local",
                query=query,
                key=local_key,
                value=local_value,
                logical_range=access_plan.local_pass.token_range,
                page_size=self.page_size,
                causal=True,
                sm_scale=self.sm_scale,
            )

            source_pass: RMarinMaterializedPass | None = None
            if not access_plan.older.empty:
                raw_key, raw_value = self._gather_source(access_plan.older)
                source_key, source_value = self.tensor_ops.transform_source(
                    raw_key,
                    raw_value,
                    self.layer_parameters[index],
                )
                source_pass = self.materializer.materialize(
                    role="source",
                    query=query,
                    key=source_key,
                    value=source_value,
                    logical_range=access_plan.older,
                    page_size=self.page_size,
                    causal=False,
                    sm_scale=self.sm_scale,
                )
            self._active_layers.add(index)
        except Exception as exc:
            self._poison(exc)
            raise

        canonical = self._source.kernel_tuple if source_pass is not None else None
        return RMarinLayerResources(
            local_wrapper=local_pass.wrapper,
            local_kv=local_pass.kv_cache,
            source_wrapper=source_pass.wrapper if source_pass is not None else None,
            source_kv=source_pass.kv_cache if source_pass is not None else None,
            canonical_source_kv=canonical,
            source_kwargs={},
            local_kwargs={},
            source_wrapper_noncausal=True,
            local_wrapper_causal=True,
            source_kernel_representation="materialized_target",
            source_transform_verified=True,
            canonical_source_is_raw=True,
            recent_blend_materialized=True,
            boundary_safe=True,
            final_layer_representations_distinct=True,
        )

    def build_callback_bindings(
        self,
        *,
        runtime_constraints: RMarinRuntimeConstraints,
        request_span_provider: Callable[[Any], RMarinRequestSpan],
        layer_index_provider: Callable[[Any], int],
        runtime_validator: Callable[[Any], None] = lambda _: None,
    ) -> CallbackRMarinFrameworkBindings:
        """Attach this session to the existing framework callback bridge."""

        if not isinstance(runtime_constraints, RMarinRuntimeConstraints):
            raise RMarinBackendContractError(
                "runtime_constraints must be RMarinRuntimeConstraints"
            )
        if runtime_constraints.chunk_size != self.chunk_size:
            raise RMarinBackendContractError("runtime and session chunk sizes differ")

        def resources(**kwargs: Any) -> RMarinLayerResources:
            return self.prepare_layer(
                layer_index=layer_index_provider(kwargs["layer"]),
                query=kwargs["query"],
                key=kwargs["key"],
                value=kwargs["value"],
                request_span=kwargs["request_span"],
                access_plan=kwargs["access_plan"],
                save_kv_cache=kwargs["save_kv_cache"],
            )

        def prepare(_backend: Any, forward_batch: Any) -> None:
            self.prepare_forward(request_span_provider(forward_batch))

        return CallbackRMarinFrameworkBindings(
            num_layers=self.num_layers,
            source_layer_index=self.source_layer_index,
            runtime_constraints=runtime_constraints,
            request_span_provider=request_span_provider,
            layer_resources_provider=resources,
            layer_index_provider=layer_index_provider,
            runtime_validator=runtime_validator,
            forward_preparer=prepare,
        )


class TorchRMarinTensorOps:
    """Torch implementation of persistent stores and recurrent transforms."""

    def __init__(self) -> None:
        try:
            self.torch = import_module("torch")
        except ImportError as exc:
            raise RMarinBackendContractError(
                "TorchRMarinTensorOps requires torch"
            ) from exc
        # SPEED-P1: validated parameter vectors are constant model weights;
        # cache the validated device tensor per (object, device, dtype) so the
        # per-layer H2D copy and isfinite().item() sync run once, not per step.
        self._parameter_cache: dict[tuple[Any, ...], tuple[Any, Any]] = {}

    def _tensor(self, value: Any, *, name: str) -> Any:
        if not isinstance(value, self.torch.Tensor):
            raise RMarinBackendContractError(f"{name} must be a torch.Tensor")
        return value

    def describe(self, value: Any) -> RMarinTensorDescriptor:
        tensor = self._tensor(value, name="value")
        return RMarinTensorDescriptor(
            shape=tuple(int(item) for item in tensor.shape),
            dtype=str(tensor.dtype),
            device=str(tensor.device),
        )

    def allocate(self, capacity: int, like: Any) -> Any:
        tensor = self._tensor(like, name="like")
        return self.torch.empty(
            (capacity, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )

    def write(self, destination: Any, start: int, source: Any) -> None:
        destination = self._tensor(destination, name="destination")
        source = self._tensor(source, name="source")
        end = start + source.shape[0]
        if start < 0 or end > destination.shape[0]:
            raise RMarinBackendContractError("tensor write exceeds allocation")
        with self.torch.no_grad():
            destination[start:end].copy_(source)

    def gather(self, source: Any, indices: Sequence[int]) -> Any:
        source = self._tensor(source, name="source")
        frozen = tuple(int(item) for item in indices)
        if not frozen:
            raise RMarinBackendContractError("tensor gather may not be empty")
        if min(frozen) < 0 or max(frozen) >= source.shape[0]:
            raise RMarinBackendContractError("tensor gather index is out of range")
        index = self.torch.tensor(
            frozen,
            dtype=self.torch.long,
            device=source.device,
        )
        return source.index_select(0, index)

    def _parameter(
        self,
        value: Any,
        *,
        like: Any,
        length: int,
        name: str,
    ) -> Any:
        cache_key = (id(value), str(like.device), str(like.dtype), length, name)
        cached = self._parameter_cache.get(cache_key)
        if cached is not None and cached[0] is value:
            return cached[1]
        if isinstance(value, self.torch.Tensor):
            parameter = (
                value.detach()
                .reshape(-1)
                .to(
                    device=like.device,
                    dtype=like.dtype,
                )
            )
        else:
            try:
                parameter = self.torch.tensor(
                    tuple(value),
                    device=like.device,
                    dtype=like.dtype,
                ).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise RMarinBackendContractError(
                    f"{name} must be a finite vector"
                ) from exc
        if parameter.numel() != length:
            raise RMarinBackendContractError(
                f"{name} length must be {length}, observed {parameter.numel()}"
            )
        if not bool(self.torch.isfinite(parameter).all().item()):
            raise RMarinBackendContractError(f"{name} contains non-finite values")
        self._parameter_cache[cache_key] = (value, parameter)
        return parameter

    def transform_source(
        self,
        key: Any,
        value: Any,
        parameters: RMarinLayerParameters,
    ) -> tuple[Any, Any]:
        key = self._tensor(key, name="key")
        value = self._tensor(value, name="value")
        heads = int(value.shape[1])
        width = int(value.shape[2])
        mode = parameters.source_mode
        if mode == "none":
            return key.clone(), value.clone()
        if mode == "vnormqk":
            weight = self._parameter(
                parameters.value_norm_weights,
                like=value,
                length=width,
                name="value_norm_weights",
            ).view(1, 1, width)
            variance = value.float().pow(2).mean(dim=-1, keepdim=True)
            normalized = (
                value.float() * self.torch.rsqrt(variance + parameters.epsilon)
            ).to(value.dtype)
            return key.clone(), normalized * weight
        scale = self._parameter(
            parameters.value_scales,
            like=value,
            length=heads,
            name="value_scales",
        ).view(1, heads, 1)
        if mode == "v":
            transformed = value * scale
        elif mode == "vnorm":
            variance = value.float().pow(2).mean(dim=-1, keepdim=True)
            normalized = (
                value.float() * self.torch.rsqrt(variance + parameters.epsilon)
            ).to(value.dtype)
            transformed = normalized * scale
        else:  # guarded by RMarinLayerParameters
            raise RMarinBackendContractError(f"unsupported source mode {mode!r}")
        return key.clone(), transformed

    def blend_recent(
        self,
        local_key: Any,
        local_value: Any,
        raw_source_key: Any,
        raw_source_value: Any,
        parameters: RMarinLayerParameters,
    ) -> tuple[Any, Any]:
        local_key = self._tensor(local_key, name="local_key")
        local_value = self._tensor(local_value, name="local_value")
        source_key, source_value = self.transform_source(
            raw_source_key,
            raw_source_value,
            parameters,
        )
        if (
            local_key.shape != source_key.shape
            or local_value.shape != source_value.shape
        ):
            raise RMarinBackendContractError(
                "local and raw-source chunks have different shapes"
            )
        heads = int(local_key.shape[1])
        alpha = (
            self._parameter(
                parameters.alpha_logits,
                like=local_key,
                length=heads,
                name="alpha_logits",
            )
            .sigmoid()
            .view(1, heads, 1)
        )
        return (
            alpha * source_key.to(local_key.dtype) + (1 - alpha) * local_key,
            alpha * source_value.to(local_value.dtype) + (1 - alpha) * local_value,
        )


class TorchFlashInferMaterializer:
    """Scratch NHD page materializer for FlashInfer 0.6.x.

    Two workspaces (source/local) are reused serially across layers.  This
    keeps P0 memory bounded while preserving independent simultaneous source
    and local plans for one attention call.
    """

    def __init__(
        self,
        *,
        workspace_bytes: int = 128 * 1024 * 1024,
        backend: str = "fa2",
    ) -> None:
        self.workspace_bytes = _positive_int("workspace_bytes", workspace_bytes)
        if not isinstance(backend, str) or not backend:
            raise RMarinBackendContractError(
                "FlashInfer backend must be a non-empty string"
            )
        self.backend = backend
        try:
            self.torch = import_module("torch")
            self.flashinfer = import_module("flashinfer")
        except ImportError as exc:
            raise RMarinBackendContractError(
                "TorchFlashInferMaterializer requires torch and flashinfer"
            ) from exc
        self._workspaces: dict[tuple[str, str], Any] = {}
        self._wrappers: dict[tuple[str, str, str], Any] = {}

    @property
    def reserved_workspace_bytes(self) -> int:
        return 2 * self.workspace_bytes

    def reserve(self, like: Any) -> None:
        """Allocate independent source/local workspaces on ``like``'s device."""

        if isinstance(like, self.torch.Tensor):
            device = like.device
        else:
            try:
                device = self.torch.device(like)
            except (TypeError, ValueError) as exc:
                raise RMarinBackendContractError(
                    "workspace reservation requires a torch tensor or device"
                ) from exc
        self._workspace("source", device)
        self._workspace("local", device)

    def _workspace(self, role: str, device: Any) -> Any:
        key = (role, str(device))
        workspace = self._workspaces.get(key)
        if workspace is None:
            workspace = self.torch.empty(
                self.workspace_bytes,
                dtype=self.torch.uint8,
                device=device,
            )
            self._workspaces[key] = workspace
        return workspace

    def _wrapper(self, role: str, phase: str, device: Any) -> Any:
        key = (role, phase, str(device))
        wrapper = self._wrappers.get(key)
        if wrapper is not None:
            return wrapper
        workspace = self._workspace(role, device)
        if phase == "decode":
            factory = getattr(
                self.flashinfer,
                "BatchDecodeWithPagedKVCacheWrapper",
                None,
            )
            if not callable(factory):
                raise RMarinBackendContractError(
                    "FlashInfer decode paged wrapper is unavailable"
                )
            wrapper = factory(
                workspace,
                "NHD",
                use_tensor_cores=True,
                backend=self.backend,
            )
        else:
            factory = getattr(
                self.flashinfer,
                "BatchPrefillWithPagedKVCacheWrapper",
                None,
            )
            if not callable(factory):
                raise RMarinBackendContractError(
                    "FlashInfer prefill paged wrapper is unavailable"
                )
            wrapper = factory(workspace, "NHD", backend=self.backend)
        self._wrappers[key] = wrapper
        return wrapper

    def _pack_nhd(self, key: Any, value: Any, page_size: int) -> Any:
        if not isinstance(key, self.torch.Tensor) or not isinstance(
            value, self.torch.Tensor
        ):
            raise RMarinBackendContractError(
                "FlashInfer materialization requires torch tensors"
            )
        if key.shape != value.shape or key.ndim != 3:
            raise RMarinBackendContractError(
                "materialized K/V must have matching [tokens, heads, dim] shape"
            )
        tokens, heads, width = (int(item) for item in key.shape)
        pages = (tokens + page_size - 1) // page_size
        cache = self.torch.zeros(
            (pages, 2, page_size, heads, width),
            dtype=key.dtype,
            device=key.device,
        )
        # ``cache[:, kv_index]`` is not contiguous: consecutive pages are
        # separated by the other K/V plane. A flattening view is therefore
        # invalid, and reshape would silently write into a copy. Copy complete
        # pages and the final tail directly into the strided destination.
        complete_pages, tail_tokens = divmod(tokens, page_size)
        complete_tokens = complete_pages * page_size
        if complete_pages:
            cache[:complete_pages, 0].copy_(
                key[:complete_tokens].reshape(
                    complete_pages,
                    page_size,
                    heads,
                    width,
                )
            )
            cache[:complete_pages, 1].copy_(
                value[:complete_tokens].reshape(
                    complete_pages,
                    page_size,
                    heads,
                    width,
                )
            )
        if tail_tokens:
            cache[complete_pages, 0, :tail_tokens].copy_(
                key[complete_tokens:]
            )
            cache[complete_pages, 1, :tail_tokens].copy_(
                value[complete_tokens:]
            )
        return cache

    def materialize(
        self,
        *,
        role: str,
        query: Any,
        key: Any,
        value: Any,
        logical_range: TokenRange,
        page_size: int,
        causal: bool,
        sm_scale: float | None,
    ) -> RMarinMaterializedPass:
        if role not in {"source", "local"}:
            raise RMarinBackendContractError("role must be source or local")
        if causal is not (role == "local"):
            raise RMarinBackendContractError(
                "source must be non-causal and local must be causal"
            )
        if not isinstance(query, self.torch.Tensor) or query.ndim != 3:
            raise RMarinBackendContractError(
                "query must be a [tokens, heads, head_dim] torch tensor"
            )
        if not isinstance(logical_range, TokenRange) or logical_range.empty:
            raise RMarinBackendContractError("logical_range must be non-empty")
        page_size = _positive_int("page_size", page_size)
        query_tokens = int(query.shape[0])
        kv_tokens = int(key.shape[0])
        if kv_tokens != logical_range.length:
            raise RMarinBackendContractError(
                "materialized K/V length differs from logical range"
            )
        if key.dtype != value.dtype or key.device != value.device:
            raise RMarinBackendContractError("materialized K/V dtype/device differ")
        pages = (kv_tokens + page_size - 1) // page_size
        page_table = RMarinExactPageTable(
            logical_range=logical_range,
            page_size=page_size,
            page_indices=tuple(range(pages)),
            last_page_length=kv_tokens - (pages - 1) * page_size,
            query_tokens=query_tokens,
            causal=causal,
        )
        cache = self._pack_nhd(key, value, page_size)
        device = key.device
        kv_indptr = self.torch.tensor(
            [0, pages],
            dtype=self.torch.int32,
            device=device,
        )
        page_indices = self.torch.arange(
            pages,
            dtype=self.torch.int32,
            device=device,
        )
        last_page_length = self.torch.tensor(
            [page_table.last_page_length],
            dtype=self.torch.int32,
            device=device,
        )
        query_heads = int(query.shape[1])
        kv_heads = int(key.shape[1])
        head_dim = int(key.shape[2])
        if int(query.shape[2]) != head_dim or query_heads % kv_heads:
            raise RMarinBackendContractError(
                "FlashInfer query/KV heads or head dimensions are incompatible"
            )
        plan_kwargs: dict[str, Any] = {
            "q_data_type": query.dtype,
            "kv_data_type": key.dtype,
        }
        if sm_scale is not None:
            plan_kwargs["sm_scale"] = sm_scale
        phase = "decode" if query_tokens == 1 else "prefill"
        wrapper = self._wrapper(role, phase, device)
        if phase == "decode":
            wrapper.plan(
                kv_indptr,
                page_indices,
                last_page_length,
                query_heads,
                kv_heads,
                head_dim,
                page_size,
                **plan_kwargs,
            )
        else:
            qo_indptr = self.torch.tensor(
                [0, query_tokens],
                dtype=self.torch.int32,
                device=device,
            )
            wrapper.plan(
                qo_indptr,
                kv_indptr,
                page_indices,
                last_page_length,
                query_heads,
                kv_heads,
                head_dim,
                page_size,
                causal=causal,
                **plan_kwargs,
            )
        planned = RMarinPlannedFlashInferWrapper(wrapper, page_table)
        return RMarinMaterializedPass(
            wrapper=planned,
            kv_cache=cache,
            page_table=page_table,
        )


__all__ = [
    "RMarinCompositeMemoryBudget",
    "RMarinCompositeKVSession",
    "RMarinExactPageTable",
    "RMarinFlashInferMaterializer",
    "RMarinKVStore",
    "RMarinLayerParameters",
    "RMarinMaterializedPass",
    "RMarinPlannedFlashInferWrapper",
    "RMarinTensorDescriptor",
    "RMarinTensorOps",
    "TorchFlashInferMaterializer",
    "TorchRMarinTensorOps",
]
