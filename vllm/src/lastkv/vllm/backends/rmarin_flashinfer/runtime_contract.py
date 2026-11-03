"""Fail-closed runtime contract shared by the RMarin framework adapters.

The memory-optimal RMarin layout is not an ordinary sliding-window cache:

* one full-history cache contains *raw* K/V emitted by the source (top) layer;
* every target layer owns a bounded local cache containing the materialized
  recent blend and the current causal chunk;
* old-source and local attention are evaluated as disjoint FlashInfer states
  and merged with their log-sum-exp values.

This module contains no torch, vLLM, SGLang, or FlashInfer imports.  Login-node
probes and unit tests can therefore validate the contract before importing a
GPU runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import importlib.metadata
from typing import Any, Protocol, runtime_checkable

from .access_plan import RMarinAccessPlan


BACKEND_ID = "rmarin_flashinfer_shared_source_p0"
VLLM_VERSION = "0.25.0"
SGLANG_VERSION = "0.5.15"
VLLM_FLASHINFER_VERSION = "0.6.13"
SGLANG_FLASHINFER_VERSION = "0.6.12"
DEFAULT_CHUNK_SIZE = 512
SUPPORTED_CHUNK_SIZES = frozenset({256, DEFAULT_CHUNK_SIZE, 1024, 2048})
DEFAULT_BLOCK_SIZE = 16
MAX_ACTIVE_SEQUENCES = 32


class RMarinBackendContractError(RuntimeError):
    """Raised when a runtime could silently violate recurrent semantics."""


def _public_version(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise RMarinBackendContractError("runtime package version is missing")
    return value.split("+", 1)[0]


def require_exact_package_version(
    distribution: str,
    expected: str,
    *,
    version_getter: Any = importlib.metadata.version,
) -> str:
    """Require one exact public package version and return the observed value."""

    try:
        observed = version_getter(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RMarinBackendContractError(
            f"{distribution} must be installed for the RMarin backend"
        ) from exc
    if _public_version(observed) != expected:
        raise RMarinBackendContractError(
            f"RMarin P0 requires {distribution} {expected}, observed {observed!r}"
        )
    return observed


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RMarinBackendContractError(f"{name} must be a positive integer")
    return value


def require_supported_chunk_size(
    value: object,
    *,
    name: str = "chunk_size",
) -> int:
    """Return one audited recurrent chunk size or fail closed."""

    chunk = _positive_int(name, value)
    if chunk not in SUPPORTED_CHUNK_SIZES:
        choices = ", ".join(str(item) for item in sorted(SUPPORTED_CHUNK_SIZES))
        raise RMarinBackendContractError(
            f"{name} must be one of {{{choices}}}, observed {chunk}"
        )
    return chunk


@dataclass(frozen=True)
class RMarinRuntimeConstraints:
    """The deliberately narrow correctness envelope for the first backend."""

    chunk_size: int = DEFAULT_CHUNK_SIZE
    block_size: int = DEFAULT_BLOCK_SIZE
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    max_active_sequences: int = 1
    enforce_eager: bool = True
    prefix_cache: bool = False
    speculative_decoding: bool = False
    beam_search: bool = False
    copy_on_write: bool = False
    cuda_graph: bool = False
    kv_offload_or_connector: bool = False

    def __post_init__(self) -> None:
        chunk = require_supported_chunk_size(self.chunk_size)
        block = _positive_int("block_size", self.block_size)
        if block != DEFAULT_BLOCK_SIZE or chunk % block:
            raise RMarinBackendContractError(
                f"RMarin P0 requires block_size={DEFAULT_BLOCK_SIZE} "
                "dividing chunk_size"
            )
        if self.dtype not in {"bfloat16", "bf16"}:
            raise RMarinBackendContractError("RMarin P0 requires BF16 activations")
        if self.kv_cache_dtype not in {"auto", "bfloat16", "bf16"}:
            raise RMarinBackendContractError(
                "RMarin P0 forbids quantized KV cache dtypes"
            )
        parallel = {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "context_parallel_size": self.context_parallel_size,
        }
        for name, value in parallel.items():
            if _positive_int(name, value) != 1:
                raise RMarinBackendContractError(
                    "RMarin P0 requires TP=1, PP=1, and CP/DCP=1"
                )
        from . import fastmode as _fm

        if (
            _positive_int("max_active_sequences", self.max_active_sequences)
            > max(MAX_ACTIVE_SEQUENCES, _fm.MAX_SEQS)
        ):
            raise RMarinBackendContractError(
                "RMarin batched mode permits at most "
                f"{MAX_ACTIVE_SEQUENCES} active sequences"
            )
        from . import fastmode

        if not self.enforce_eager or self.cuda_graph:
            if not fastmode.ALLOW_CUDAGRAPH:
                raise RMarinBackendContractError(
                    "RMarin P0 requires eager execution with CUDA graphs disabled"
                )
        unsupported = {
            # PAGED_SOURCE drives vLLM's prefix cache itself (block retention for
            # session resume; the chunk-aligned scheduler clamps every hit to a
            # ring-snapshot boundary or to zero), so the cache is allowed there.
            "prefix/radix cache": self.prefix_cache and not fastmode.PAGED_SOURCE,
            "speculative decoding": self.speculative_decoding,
            "beam search": self.beam_search,
            "copy-on-write cache": self.copy_on_write,
            "KV offload/connectors": self.kv_offload_or_connector,
        }
        active = [name for name, enabled in unsupported.items() if enabled]
        if active:
            raise RMarinBackendContractError(
                "RMarin P0 forbids " + ", ".join(active)
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RMarinRuntimeConstraints":
        if not isinstance(value, Mapping):
            raise RMarinBackendContractError("runtime constraints must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise RMarinBackendContractError(
                "unknown RMarin runtime constraints: " + ", ".join(unknown)
            )
        return cls(**dict(value))


@dataclass(frozen=True)
class RMarinLayerExecution:
    """One layer's already-planned source/local FlashInfer invocation.

    Framework-specific cache managers build this object.  The common executor
    checks it again immediately before launching either kernel.
    """

    request_ids: tuple[str, ...]
    layer_index: int
    source_layer_index: int
    query_tokens: int
    source_tokens: int
    local_tokens: int
    local_wrapper: Any
    local_kv: Any
    access_plan: RMarinAccessPlan
    source_wrapper: Any | None = None
    source_kv: Any | None = None
    canonical_source_kv: Any | None = None
    source_kwargs: Mapping[str, Any] = field(default_factory=dict)
    local_kwargs: Mapping[str, Any] = field(default_factory=dict)
    source_wrapper_noncausal: bool = True
    local_wrapper_causal: bool = True
    source_kernel_representation: str = "raw_shared"
    source_transform_verified: bool = True
    canonical_source_is_raw: bool = True
    recent_blend_materialized: bool = True
    boundary_safe: bool = True
    final_layer_representations_distinct: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_ids, tuple)
            or not self.request_ids
            or any(not isinstance(item, str) or not item for item in self.request_ids)
        ):
            raise RMarinBackendContractError(
                "request_ids must be a non-empty tuple of non-empty strings"
            )
        if len(set(self.request_ids)) != len(self.request_ids):
            raise RMarinBackendContractError("duplicate request_ids in layer execution")
        if not isinstance(self.access_plan, RMarinAccessPlan):
            raise RMarinBackendContractError(
                "layer execution must carry a validated RMarinAccessPlan"
            )
        planned_request = self.access_plan.request_state.request_id
        if self.request_ids != (planned_request,):
            raise RMarinBackendContractError(
                "P0 layer execution request differs from its access plan"
            )
        for name in (
            "layer_index",
            "source_layer_index",
            "query_tokens",
            "local_tokens",
        ):
            value = getattr(self, name)
            minimum = 0 if name.endswith("index") else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise RMarinBackendContractError(
                    f"{name} must be an integer >= {minimum}"
                )
        if (
            self.layer_index != self.access_plan.layer_spec.layer_index
            or self.source_layer_index
            != self.access_plan.layer_spec.source_layer_index
        ):
            raise RMarinBackendContractError(
                "layer execution indices differ from its access plan"
            )
        if (
            isinstance(self.source_tokens, bool)
            or not isinstance(self.source_tokens, int)
            or self.source_tokens < 0
        ):
            raise RMarinBackendContractError(
                "source_tokens must be a non-negative integer"
            )
        expected_source_tokens = self.access_plan.older.length
        expected_local_tokens = self.access_plan.local_pass.token_range.length
        if self.source_tokens != expected_source_tokens:
            raise RMarinBackendContractError(
                "source_tokens differs from the planned older-source range"
            )
        if self.local_tokens != expected_local_tokens:
            raise RMarinBackendContractError(
                "local_tokens differs from the planned recent/current range"
            )
        if self.query_tokens > self.access_plan.current.length:
            raise RMarinBackendContractError(
                "query_tokens exceeds the planned current causal range"
            )
        if self.access_plan.requires_lse_merge != (self.source_tokens > 0):
            raise RMarinBackendContractError(
                "source execution differs from the plan's LSE merge contract"
            )
        if self.local_wrapper is None or self.local_kv is None:
            raise RMarinBackendContractError(
                "every RMarin attention call requires a local cache"
            )
        has_source_objects = (
            self.source_wrapper is not None
            and self.source_kv is not None
            and self.canonical_source_kv is not None
        )
        if self.source_tokens > 0 and not has_source_objects:
            raise RMarinBackendContractError(
                "old-source tokens require a source wrapper, kernel-ready source "
                "cache, and canonical raw top-layer cache"
            )
        if self.source_tokens == 0 and (
            self.source_wrapper is not None
            or self.source_kv is not None
            or self.canonical_source_kv is not None
        ):
            raise RMarinBackendContractError(
                "empty old-source range may not carry partial source bindings"
            )
        if self.source_kernel_representation not in {
            "raw_shared",
            "materialized_target",
        }:
            raise RMarinBackendContractError(
                "source_kernel_representation must be raw_shared or "
                "materialized_target"
            )
        if self.source_tokens and self.canonical_source_is_raw is not True:
            raise RMarinBackendContractError(
                "the canonical shared source cache must contain raw top-layer KV"
            )
        if self.source_tokens and self.source_transform_verified is not True:
            raise RMarinBackendContractError(
                "kernel-ready source values must carry a verified target-layer "
                "transform receipt"
            )
        if not self.recent_blend_materialized:
            raise RMarinBackendContractError(
                "recent source/local K/V blend must be materialized before attention"
            )
        if not self.boundary_safe:
            raise RMarinBackendContractError(
                "scheduled tokens cross an RMarin chunk boundary"
            )
        if (
            self.layer_index == self.source_layer_index
            and not self.final_layer_representations_distinct
        ):
            raise RMarinBackendContractError(
                "the source layer must preserve distinct raw-source and effective-local KV"
            )
        if not isinstance(self.source_kwargs, Mapping) or not isinstance(
            self.local_kwargs, Mapping
        ):
            raise RMarinBackendContractError(
                "source_kwargs and local_kwargs must be mappings"
            )
        forbidden_plan_keys = {"causal", "window_left", "pos_encoding_mode"}
        passed_plan_keys = sorted(
            forbidden_plan_keys
            & (set(self.source_kwargs) | set(self.local_kwargs))
        )
        if passed_plan_keys:
            raise RMarinBackendContractError(
                "FlashInfer wrapper-plan arguments may not be passed to the "
                "return-LSE execution call: " + ", ".join(passed_plan_keys)
            )
        if self.source_tokens > 0 and self.source_wrapper_noncausal is not True:
            raise RMarinBackendContractError(
                "old-source wrapper must be planned non-causal"
            )
        if self.local_wrapper_causal is not True:
            raise RMarinBackendContractError(
                "local wrapper must be planned causal"
            )


@runtime_checkable
class RMarinFrameworkBindings(Protocol):
    """Framework-owned model/cache/scheduler bridge required by the backend."""

    chunk_size: int
    source_layer_index: int

    def validate_runtime(self, model_runner: Any) -> None:
        """Validate allocator, model weights, and scheduler settings."""

    def prepare_forward(self, backend: Any, forward_batch: Any) -> None:
        """Prepare source/local page tables without crossing a chunk boundary."""

    def prepare_layer(
        self,
        *,
        backend: Any,
        layer: Any,
        forward_batch: Any,
        query: Any,
        key: Any,
        value: Any,
        save_kv_cache: bool,
    ) -> RMarinLayerExecution:
        """Write raw/local KV and return the disjoint attention bindings."""


__all__ = [
    "BACKEND_ID",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_CHUNK_SIZE",
    "MAX_ACTIVE_SEQUENCES",
    "RMarinBackendContractError",
    "RMarinFrameworkBindings",
    "RMarinLayerExecution",
    "RMarinRuntimeConstraints",
    "SGLANG_FLASHINFER_VERSION",
    "SGLANG_VERSION",
    "SUPPORTED_CHUNK_SIZES",
    "VLLM_FLASHINFER_VERSION",
    "VLLM_VERSION",
    "require_exact_package_version",
    "require_supported_chunk_size",
]
