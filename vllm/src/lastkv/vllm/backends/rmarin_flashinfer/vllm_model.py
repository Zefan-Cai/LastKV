"""Out-of-tree vLLM 0.25 model bridge for the RMarin FlashInfer backend.

The legacy recurrent-vLLM compatibility model keeps a full paged KV cache for
every transformer layer.  This module is deliberately independent from that
implementation.  It constructs ordinary Llama projections and MLPs, but:

* forces the audited ``CUSTOM`` RMarin attention backend;
* gives vLLM only a one-block sliding-window placeholder per layer;
* binds Q/K norm and recurrent checkpoint parameters to one composite session;
* stages the exact single-request span from model ``positions``; and
* leaves all persistent raw-source and bounded local KV storage to the
  composite session.

Heavy torch/vLLM/FlashInfer imports are resolved only when vLLM asks for the
registered model class.  Static probes and report generation can import this
module on CPU-only hosts.
"""

from __future__ import annotations

import os

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from .batching import (
    RMarinBatchBindings,
    RMarinBatchMaterializer,
    RMarinBatchSpanTracker,
    RMarinSessionPool,
    install_runner_capture,
)
from .bindings import RMarinRequestSpan
from .runtime_contract import (
    RMarinBackendContractError,
    RMarinRuntimeConstraints,
    require_supported_chunk_size,
)
from .vllm_backend import (
    MODEL_ADAPTER_BINDINGS_ATTRIBUTE,
    build_vllm_backend_types,
    validate_vllm_runtime_config,
)


RECURRENT_ARCHITECTURE = "CustomRecurrentMarinAKV254ReuseKVForCausalLM"
RMARIN_MODEL_CLASS = "RMarinFlashInferForCausalLM"
RMARIN_MODEL_IMPORT = (
    f"lastkv.vllm.backends.rmarin_flashinfer.vllm_model:{RMARIN_MODEL_CLASS}"
)
ACTIVE_REQUEST_ID = "vllm-rmarin-p0-active-request"
DEFAULT_FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024
DEFAULT_DYNAMIC_SAFETY_BYTES = 256 * 1024 * 1024
SESSION_POOL_SLOTS_ENV = "RMARIN_SESSION_POOL_SLOTS"
_RECURRENT_CHECKPOINT_SUFFIXES = (
    "prev_chunk_alphas",
    "q_norm.weight",
    "k_norm.weight",
    "deep_v_scale",
    "deep_v_norm.weight",
)


class RMarinVllmModelError(RMarinBackendContractError):
    """The model/config/session bridge cannot preserve RMarin semantics."""


def _session_pool_slots(vllm_config: Any) -> int:
    """Return a state-pool size that may exceed scheduler concurrency.

    vLLM can temporarily omit a live structured-output request from a model
    forward.  The recurrent backend must retain that request's state while it
    is omitted, so an explicitly larger pool provides spill slots without
    changing scheduler concurrency or generation semantics.
    """

    scheduled = int(vllm_config.scheduler_config.max_num_seqs)
    raw = os.environ.get(SESSION_POOL_SLOTS_ENV)
    if raw is None:
        return scheduled
    try:
        slots = int(raw)
    except ValueError as exc:
        raise RMarinVllmModelError(
            f"{SESSION_POOL_SLOTS_ENV} must be an integer"
        ) from exc
    if slots < scheduled:
        raise RMarinVllmModelError(
            f"{SESSION_POOL_SLOTS_ENV}={slots} must be at least "
            f"scheduler max_num_seqs={scheduled}"
        )
    from . import fastmode as _fm

    if slots > max(32, _fm.MAX_SEQS):
        raise RMarinVllmModelError(
            f"{SESSION_POOL_SLOTS_ENV}={slots} exceeds backend limit {max(32, _fm.MAX_SEQS)}"
        )
    return slots


@dataclass(frozen=True)
class RMarinVllmMemoryBudget:
    """Conservative GPU bytes outside vLLM's stock KV allocation.

    Persistent stores and FlashInfer workspaces are reserved during model
    construction, before vLLM profiles non-KV memory.  Materialization scratch
    remains dynamic, so it must fit in the fraction excluded by
    ``gpu_memory_utilization``.
    """

    max_sequence_tokens: int
    num_layers: int
    chunk_size: int
    num_kv_heads: int
    head_dim: int
    workspace_bytes_per_role: int = DEFAULT_FLASHINFER_WORKSPACE_BYTES
    element_bytes: int = 2

    def __post_init__(self) -> None:
        for name in (
            "max_sequence_tokens",
            "num_layers",
            "chunk_size",
            "num_kv_heads",
            "head_dim",
            "workspace_bytes_per_role",
            "element_bytes",
        ):
            _integer(name, getattr(self, name), minimum=1)
        try:
            require_supported_chunk_size(self.chunk_size)
        except RMarinBackendContractError as exc:
            raise RMarinVllmModelError(str(exc)) from exc
        if self.max_sequence_tokens < self.chunk_size:
            raise RMarinVllmModelError(
                "max_sequence_tokens must cover at least one recurrent chunk"
            )

    @property
    def bytes_per_k_or_v_token(self) -> int:
        return self.num_kv_heads * self.head_dim * self.element_bytes

    @property
    def persistent_source_bytes(self) -> int:
        return 2 * self.max_sequence_tokens * self.bytes_per_k_or_v_token

    @property
    def persistent_local_bytes(self) -> int:
        return 2 * self.chunk_size * self.num_layers * 2 * self.bytes_per_k_or_v_token

    @property
    def persistent_workspace_bytes(self) -> int:
        # Source and local plans must coexist until LSE merge.
        return 2 * self.workspace_bytes_per_role

    @property
    def persistent_reserved_bytes(self) -> int:
        return (
            self.persistent_source_bytes
            + self.persistent_local_bytes
            + self.persistent_workspace_bytes
        )

    @property
    def attention_scratch_upper_bound_bytes(self) -> int:
        # At the longest sequence, source materialization retains raw gather,
        # transformed K/V, and packed K/V (6 tensors).  Local materialization
        # retains gathered and packed K/V (4 tensors).  Page padding is bounded
        # by the 16-token block and covered by the explicit safety margin.
        maximum_position = self.max_sequence_tokens - 1
        current_start = (maximum_position // self.chunk_size) * self.chunk_size
        older_tokens = max(
            0,
            current_start - self.chunk_size,
        )
        local_tokens = min(
            self.max_sequence_tokens,
            2 * self.chunk_size,
        )
        return (6 * older_tokens + 4 * local_tokens) * self.bytes_per_k_or_v_token

    @property
    def boundary_scratch_upper_bound_bytes(self) -> int:
        # Atomic boundary commit retains one replacement K/V chunk per layer
        # plus conservative source/transform temporaries.
        return 2 * (self.num_layers + 6) * self.chunk_size * self.bytes_per_k_or_v_token

    @property
    def dynamic_scratch_peak_bytes(self) -> int:
        return max(
            self.attention_scratch_upper_bound_bytes,
            self.boundary_scratch_upper_bound_bytes,
        )

    def require_dynamic_headroom(
        self,
        *,
        total_gpu_bytes: int,
        gpu_memory_utilization: float,
        safety_bytes: int = DEFAULT_DYNAMIC_SAFETY_BYTES,
    ) -> int:
        total = _integer("total_gpu_bytes", total_gpu_bytes, minimum=1)
        safety = _integer("safety_bytes", safety_bytes, minimum=1)
        utilization = _finite_float(
            "gpu_memory_utilization",
            gpu_memory_utilization,
        )
        if utilization <= 0 or utilization >= 1:
            raise RMarinVllmModelError(
                "gpu_memory_utilization must be strictly between zero and one"
            )
        available = int(total * (1 - utilization))
        required = self.dynamic_scratch_peak_bytes + safety
        if available < required:
            raise RMarinVllmModelError(
                "gpu_memory_utilization leaves insufficient RMarin dynamic "
                f"headroom: available={available} required={required}; lower "
                "gpu_memory_utilization"
            )
        return available

    def receipt(self) -> dict[str, int]:
        return {
            "persistent_source_bytes": self.persistent_source_bytes,
            "persistent_local_bytes": self.persistent_local_bytes,
            "persistent_workspace_bytes": self.persistent_workspace_bytes,
            "persistent_reserved_bytes": self.persistent_reserved_bytes,
            "attention_scratch_upper_bound_bytes": (
                self.attention_scratch_upper_bound_bytes
            ),
            "boundary_scratch_upper_bound_bytes": (
                self.boundary_scratch_upper_bound_bytes
            ),
            "dynamic_scratch_peak_bytes": self.dynamic_scratch_peak_bytes,
        }


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RMarinVllmModelError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RMarinVllmModelError(f"{name} must be a finite number")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise RMarinVllmModelError(f"{name} must be a finite number")
    return result


def _layer_scale_values(
    config: Any,
    *,
    layer_index: int,
    num_kv_heads: int,
) -> tuple[float, ...]:
    raw = getattr(config, "deep_v_scale_init", 1.0)
    if isinstance(raw, (tuple, list)):
        if len(raw) != int(config.num_hidden_layers):
            raise RMarinVllmModelError(
                "deep_v_scale_init must contain one entry per hidden layer"
            )
        raw = raw[layer_index]
    if isinstance(raw, (tuple, list)):
        if len(raw) != num_kv_heads:
            raise RMarinVllmModelError(
                f"layer {layer_index} deep_v_scale_init has {len(raw)} "
                f"values, expected {num_kv_heads}"
            )
        return tuple(
            _finite_float(
                f"deep_v_scale_init[{layer_index}][{head_index}]",
                value,
            )
            for head_index, value in enumerate(raw)
        )
    scalar = _finite_float(
        f"deep_v_scale_init[{layer_index}]",
        raw,
    )
    return (scalar,) * num_kv_heads


@dataclass(frozen=True)
class RMarinCheckpointLayerSpec:
    """Checkpoint-visible recurrent parameters for one target layer."""

    layer_index: int
    num_kv_heads: int
    head_dim: int
    use_qk_norm: bool
    source_mode: str
    deep_scale_frozen: bool
    scale_init: tuple[float, ...]
    rms_norm_eps: float

    def __post_init__(self) -> None:
        _integer("layer_index", self.layer_index)
        heads = _integer("num_kv_heads", self.num_kv_heads, minimum=1)
        _integer("head_dim", self.head_dim, minimum=1)
        if type(self.use_qk_norm) is not bool:
            raise RMarinVllmModelError("use_qk_norm must be boolean")
        if self.source_mode not in {"none", "v", "vnorm", "vnormqk"}:
            raise RMarinVllmModelError("unsupported RMarin source transform")
        if type(self.deep_scale_frozen) is not bool:
            raise RMarinVllmModelError("deep_scale_frozen must be boolean")
        if len(self.scale_init) != heads:
            raise RMarinVllmModelError("scale_init must contain one value per KV head")
        for index, value in enumerate(self.scale_init):
            _finite_float(f"scale_init[{index}]", value)
        epsilon = _finite_float("rms_norm_eps", self.rms_norm_eps)
        if epsilon <= 0:
            raise RMarinVllmModelError("rms_norm_eps must be positive")

    @property
    def checkpoint_suffixes(self) -> tuple[str, ...]:
        suffixes = ["prev_chunk_alphas"]
        if self.use_qk_norm:
            suffixes.extend(("q_norm.weight", "k_norm.weight"))
        if self.source_mode == "vnormqk":
            suffixes.append("deep_v_norm.weight")
        elif self.source_mode in {"v", "vnorm"} and not self.deep_scale_frozen:
            suffixes.append("deep_v_scale")
        return tuple(suffixes)


def validate_rmarin_hf_config(config: Any) -> tuple[RMarinCheckpointLayerSpec, ...]:
    """Validate the exact last-layer-source RMarin model contract."""

    layer_count = _integer(
        "num_hidden_layers",
        getattr(config, "num_hidden_layers", None),
        minimum=1,
    )
    num_heads = _integer(
        "num_attention_heads",
        getattr(config, "num_attention_heads", None),
        minimum=1,
    )
    kv_heads = _integer(
        "num_key_value_heads",
        getattr(config, "num_key_value_heads", None),
        minimum=1,
    )
    hidden_size = _integer(
        "hidden_size",
        getattr(config, "hidden_size", None),
        minimum=1,
    )
    head_dim = _integer(
        "head_dim",
        getattr(config, "head_dim", hidden_size // num_heads),
        minimum=1,
    )
    if hidden_size != num_heads * head_dim:
        raise RMarinVllmModelError(
            "hidden_size must equal num_attention_heads * head_dim"
        )
    if num_heads % kv_heads:
        raise RMarinVllmModelError(
            "num_attention_heads must be divisible by num_key_value_heads"
        )
    if bool(getattr(config, "use_nope", False)):
        raise RMarinVllmModelError("RMarin FlashInfer P0 does not support use_nope")
    try:
        require_supported_chunk_size(getattr(config, "chunk_size", None))
    except RMarinBackendContractError as exc:
        raise RMarinVllmModelError(str(exc)) from exc

    # Existing recurrent exports retain ``v_gap`` in config.  This backend is
    # intentionally narrower: every target layer must resolve to the final
    # transformer layer, exactly matching RMarin's shared source cache.
    v_gap = getattr(config, "v_gap", None)
    if isinstance(v_gap, bool) or not isinstance(v_gap, int):
        raise RMarinVllmModelError("RMarin config requires integer v_gap")
    source_map = tuple(
        max(min(layer_index + v_gap, layer_count - 1), 0)
        for layer_index in range(layer_count)
    )
    if set(source_map) != {layer_count - 1}:
        raise RMarinVllmModelError(
            "RMarin FlashInfer requires every v_gap mapping to resolve to the "
            "final transformer layer"
        )

    use_qk_norm = bool(getattr(config, "use_qk_norm", False))
    deep_rescale = bool(getattr(config, "deep_kv_rescale", False))
    deep_mode = str(getattr(config, "deep_rescale_mode", "v"))
    if deep_mode not in {"v", "vnorm", "vnormqk"}:
        raise RMarinVllmModelError("deep_rescale_mode must be v, vnorm, or vnormqk")
    source_mode = deep_mode if deep_rescale else "none"
    deep_frozen = bool(getattr(config, "deep_scale_frozen", False))
    rms_norm_eps = _finite_float(
        "rms_norm_eps",
        getattr(config, "rms_norm_eps", None),
    )
    return tuple(
        RMarinCheckpointLayerSpec(
            layer_index=layer_index,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            use_qk_norm=use_qk_norm,
            source_mode=source_mode,
            deep_scale_frozen=deep_frozen,
            scale_init=_layer_scale_values(
                config,
                layer_index=layer_index,
                num_kv_heads=kv_heads,
            ),
            rms_norm_eps=rms_norm_eps,
        )
        for layer_index in range(layer_count)
    )


def expected_rmarin_checkpoint_keys(config: Any) -> frozenset[str]:
    """Return every recurrent key that the prepared export must contain."""

    return frozenset(
        f"model.layers.{spec.layer_index}.self_attn.{suffix}"
        for spec in validate_rmarin_hf_config(config)
        for suffix in spec.checkpoint_suffixes
    )


def _is_recurrent_checkpoint_key(name: str) -> bool:
    return ".self_attn." in name and name.endswith(_RECURRENT_CHECKPOINT_SUFFIXES)


class RMarinPositionSpanTracker:
    """Stage one boundary-safe P0 span without activating dummy forwards."""

    def __init__(
        self,
        *,
        chunk_size: int,
        request_id: str = ACTIVE_REQUEST_ID,
    ) -> None:
        try:
            self.chunk_size = require_supported_chunk_size(chunk_size)
        except RMarinBackendContractError as exc:
            raise RMarinVllmModelError(str(exc)) from exc
        if not isinstance(request_id, str) or not request_id:
            raise RMarinVllmModelError("request_id must be non-empty")
        self.request_id_prefix = request_id
        self._request_epoch = 0
        self._active_request_id: str | None = None
        self._generation = 0
        self._activated_generation = 0
        self._span: RMarinRequestSpan | None = None
        self._session: Any | None = None

    def bind_session(self, session: Any) -> None:
        if session is None or not callable(getattr(session, "prepare_forward", None)):
            raise RMarinVllmModelError(
                "composite session must expose prepare_forward(span)"
            )
        if self._session is not None and self._session is not session:
            raise RMarinVllmModelError(
                "position tracker cannot be rebound to another composite session"
            )
        self._session = session

    @property
    def span(self) -> RMarinRequestSpan:
        if self._span is None:
            raise RMarinVllmModelError(
                "RMarin attention ran before model positions were staged"
            )
        return self._span

    def stage_positions(self, positions: Any) -> RMarinRequestSpan:
        ndim = getattr(positions, "ndim", None)
        if ndim != 1:
            raise RMarinVllmModelError("RMarin positions must have shape [tokens]")
        values = positions.detach().to(device="cpu").tolist()
        if not isinstance(values, list) or not values:
            raise RMarinVllmModelError("RMarin forward requires at least one token")
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise RMarinVllmModelError("RMarin positions must contain integers")
        start = values[0]
        expected = list(range(start, start + len(values)))
        if values != expected:
            raise RMarinVllmModelError(
                "RMarin P0 requires one contiguous, ordered request span"
            )
        if start == 0:
            # P0 permits one active sequence and disables prefix/speculative
            # state, so position zero is the only legal ownership transition.
            # Give every new or recomputed sequence an explicit epoch instead
            # of letting two requests share one synthetic cache owner.
            self._request_epoch += 1
            self._active_request_id = (
                f"{self.request_id_prefix}:epoch-{self._request_epoch}"
            )
        elif self._active_request_id is None:
            raise RMarinVllmModelError(
                "RMarin non-zero positions arrived before a position-zero "
                "request ownership transition"
            )
        span = RMarinRequestSpan(
            request_id=self._active_request_id,
            num_computed_tokens=start,
            num_query_tokens=len(values),
            chunk_size=self.chunk_size,
        )
        self._generation += 1
        self._span = span
        return span

    def provide_span(self, attention_metadata: Any) -> RMarinRequestSpan:
        span = self.span
        token_count = getattr(attention_metadata, "num_actual_tokens", None)
        if token_count != span.num_query_tokens:
            raise RMarinVllmModelError(
                "vLLM attention metadata token count differs from model positions"
            )
        for name in (
            "num_decodes",
            "num_decode_tokens",
            "num_prefills",
            "num_prefill_tokens",
        ):
            value = getattr(attention_metadata, name, None)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RMarinVllmModelError(
                    f"vLLM attention metadata {name} must be a non-negative integer"
                )
        if getattr(attention_metadata, "causal", None) is not True:
            raise RMarinVllmModelError(
                "RMarin requires causal vLLM attention metadata"
            )
        if getattr(attention_metadata, "use_cascade", None) is not False:
            raise RMarinVllmModelError(
                "RMarin owns the source/local LSE merge and rejects vLLM cascade"
            )
        decode_tokens = attention_metadata.num_decode_tokens
        prefill_tokens = attention_metadata.num_prefill_tokens
        if decode_tokens + prefill_tokens != span.num_query_tokens:
            raise RMarinVllmModelError(
                "vLLM decode/prefill token counts differ from model positions"
            )
        decode_active = decode_tokens > 0
        prefill_active = prefill_tokens > 0
        if decode_active and prefill_active:
            raise RMarinVllmModelError(
                "RMarin P0 rejects mixed decode/prefill attention metadata"
            )
        if not decode_active and not prefill_active:
            raise RMarinVllmModelError(
                "vLLM attention metadata contains no active request phase"
            )
        if decode_active:
            if (
                attention_metadata.num_decodes != 1
                or attention_metadata.num_prefills != 0
                or getattr(attention_metadata, "decode", None) is None
                or getattr(attention_metadata, "prefill", None) is not None
            ):
                raise RMarinVllmModelError(
                    "vLLM decode metadata violates the single-request P0 contract"
                )
        elif (
            attention_metadata.num_prefills != 1
            or attention_metadata.num_decodes != 0
            or getattr(attention_metadata, "prefill", None) is None
            or getattr(attention_metadata, "decode", None) is not None
        ):
            raise RMarinVllmModelError(
                "vLLM prefill metadata violates the single-request P0 contract"
            )
        slot_mapping = getattr(attention_metadata, "slot_mapping", None)
        slot_shape = getattr(slot_mapping, "shape", None)
        try:
            slot_tokens = int(slot_shape[0])
        except (IndexError, TypeError, ValueError) as exc:
            raise RMarinVllmModelError(
                "vLLM attention metadata slot_mapping must have shape [tokens]"
            ) from exc
        if len(slot_shape) != 1 or slot_tokens != span.num_query_tokens:
            raise RMarinVllmModelError(
                "vLLM slot_mapping length differs from model positions"
            )
        # vLLM's thin adapter invokes prepare_layer directly, unlike SGLang's
        # model-runner hook.  Activate the staged session exactly once when
        # real attention metadata first appears.  Dummy/profile forwards have
        # attn_metadata=None and are returned before this provider is called.
        if self._activated_generation != self._generation:
            if self._session is None:
                raise RMarinVllmModelError(
                    "RMarin position tracker has no composite session"
                )
            self._session.prepare_forward(span)
            self._activated_generation = self._generation
        return span


SessionFactory = Callable[..., Any]


def _default_session_factory(
    *,
    vllm_config: Any,
    layer_parameters: Sequence[Any],
    importer: Callable[[str], Any],
) -> Any:
    session_module = importer(
        "lastkv.vllm.backends.rmarin_flashinfer.composite_session"
    )
    tensor_ops = session_module.TorchRMarinTensorOps()
    from . import fastmode as _fm

    materializer = RMarinBatchMaterializer(
        workspace_bytes=DEFAULT_FLASHINFER_WORKSPACE_BYTES,
        backend=_fm.PREFILL_BACKEND,
    )
    source_tokens = 0
    if _fm.PAGED_SOURCE:
        source_tokens = _paged_source_budget_tokens(vllm_config, importer, num_slots=_session_pool_slots(vllm_config))
    return RMarinSessionPool(
        num_slots=_session_pool_slots(vllm_config),
        max_sequence_tokens=int(vllm_config.model_config.max_model_len),
        source_tokens=source_tokens,
        num_layers=len(layer_parameters),
        source_layer_index=len(layer_parameters) - 1,
        layer_parameters=tuple(layer_parameters),
        tensor_ops=tensor_ops,
        materializer=materializer,
        chunk_size=require_supported_chunk_size(
            getattr(vllm_config.model_config.hf_config, "chunk_size", None)
        ),
        page_size=int(vllm_config.cache_config.block_size),
    )


def _paged_source_budget_tokens(vllm_config: Any, importer: Callable[[str], Any], *, num_slots: int) -> int:
    """Source-page budget (tokens) for PAGED_SOURCE.

    Explicit ``RMARIN_SOURCE_TOKENS`` wins.  Otherwise take the GPU memory
    that is free right now (weights are already resident), subtract the
    per-slot rings, the per-layer block tables, vLLM's dynamic headroom
    (``total * (1 - gpu_memory_utilization)``) and a safety margin, and
    convert at K+V bytes/token plus the placeholder spec's bytes/token.
    """

    from . import fastmode as _fm

    torch = importer("torch")
    page = int(vllm_config.cache_config.block_size)
    max_len = int(vllm_config.model_config.max_model_len)
    hf = vllm_config.model_config.hf_config
    specs = validate_rmarin_hf_config(hf)
    layers = len(specs)
    heads = int(specs[0].num_kv_heads)
    width = int(specs[0].head_dim)
    chunk = require_supported_chunk_size(getattr(hf, "chunk_size", None))
    elem = torch.tensor([], dtype=vllm_config.model_config.dtype).element_size()
    if _fm.SOURCE_TOKENS > 0:
        tokens = int(_fm.SOURCE_TOKENS)
    else:
        free, total = torch.cuda.mem_get_info()
        util = float(vllm_config.cache_config.gpu_memory_utilization)
        ring_pages = 2 * chunk // page
        ring_bytes = layers * 2 * (ring_pages * num_slots + 1) * page * heads * width * elem
        bt_bytes = layers * num_slots * (-(-max_len // page) + ring_pages) * 4
        from .resume import resume_enabled as _resume_enabled

        snap_bytes = (
            _fm.snapshot_buffers(num_slots) * 2 * layers * chunk * heads * width * elem if _resume_enabled() else 0
        )
        ring_bytes += snap_bytes
        # dynamic headroom: vLLM's share above gpu_memory_utilization, a fixed
        # margin (RMARIN_SOURCE_HEADROOM_GIB, default 4) and ~24 MiB per sequence
        # for sampler/activation buffers that grow with the batch (128 x 24k
        # resident requests OOMed with a 2 GiB fixed margin).
        margin_gib = float(os.environ.get("RMARIN_SOURCE_HEADROOM_GIB", "4"))
        headroom = int(total * (1.0 - util)) + int(margin_gib * 2**30) + num_slots * 24 * 2**20
        per_token = 2 * heads * width * elem + 2 * 1 * int(_fm.PLACEHOLDER_HEAD_SIZE) * elem
        budget = int(free) - ring_bytes - bt_bytes - headroom
        tokens = budget // per_token
        print({"rmarin_paged_source_budget": {"free_gib": round(free / 2**30, 2), "ring_gib": round(ring_bytes / 2**30, 2),
                                              "snapshot_gib": round(snap_bytes / 2**30, 2),
                                              "bt_gib": round(bt_bytes / 2**30, 2), "headroom_gib": round(headroom / 2**30, 2),
                                              "source_tokens": int(tokens)}}, flush=True)
    tokens = (tokens // page) * page
    if tokens < max_len:
        raise RMarinVllmModelError(
            f"paged source budget {tokens} tokens is below max_model_len {max_len}; lower max_num_seqs or raise gpu_memory_utilization"
        )
    return int(tokens)


_CONSTRAINT_CACHE: dict[int, list[Any]] = {}


def _same_runtime_constraints(
    expected: RMarinRuntimeConstraints,
    vllm_config: Any,
) -> None:
    # SPEED-P4: the config object is immutable during serving; re-derive the
    # constraints once per 1024 calls instead of 32x per step.
    entry = _CONSTRAINT_CACHE.get(id(vllm_config))
    if entry is not None and entry[0] is vllm_config and entry[2] < 1024:
        entry[2] += 1
        observed = entry[1]
    else:
        observed = validate_vllm_runtime_config(vllm_config)
        _CONSTRAINT_CACHE[id(vllm_config)] = [vllm_config, observed, 0]
    if observed != expected:
        raise RMarinVllmModelError(
            "vLLM runtime config changed after RMarin model construction"
        )


_RUNTIME_TYPES: tuple[type[Any], ...] | None = None


def build_rmarin_vllm_model_types(
    *,
    importer: Callable[[str], Any] = import_module,
    version_getter: Callable[[str], str] | None = None,
    session_factory: SessionFactory | None = None,
    attention_backend_type: type[Any] | None = None,
) -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    """Build model classes lazily against the pinned vLLM 0.25 runtime."""

    torch = importer("torch")
    nn = importer("torch.nn")
    from . import fastmode as _fastmode
    from . import fastpath as _fastpath

    _FUSED_NORM = _fastmode.FUSED_NORM
    _ops = importer("vllm._custom_ops") if _FUSED_NORM else None
    print({"rmarin_fastmode": _fastmode.summary()}, flush=True)
    llama = importer("vllm.model_executor.models.llama")
    attention_module = importer("vllm.model_executor.layers.attention")
    utils = importer("vllm.model_executor.models.utils")
    attention_type_module = importer("vllm.v1.attention.backend")
    forward_context_module = importer("vllm.forward_context")
    session_module = importer(
        "lastkv.vllm.backends.rmarin_flashinfer.composite_session"
    )
    if attention_backend_type is None:
        attention_backend_type = build_vllm_backend_types(
            importer=importer,
            version_getter=version_getter,
        )[0]
    factory = session_factory or _default_session_factory

    class TrainingEagerRMSNorm(nn.Module):
        """Checkpoint-compatible eager RMSNorm used by recurrent training."""

        def __init__(self, hidden_size: int, *, eps: float) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = float(eps)

        def forward(
            self,
            hidden_states: Any,
            residual: Any | None = None,
        ) -> Any:
            if _FUSED_NORM:
                # SPEED: one CUDA kernel instead of ~8 eager ops.  vLLM's
                # fused_add_rms_norm updates `hidden_states` (normed) and
                # `residual` (sum) in place, matching the eager contract.
                if residual is not None:
                    _ops.fused_add_rms_norm(hidden_states, residual, self.weight, self.eps)
                    return hidden_states, residual
                width = hidden_states.shape[-1]
                flat = hidden_states.reshape(-1, width)
                if not flat.is_contiguous():
                    flat = flat.contiguous()
                out = torch.empty_like(flat)
                _ops.rms_norm(out, flat, self.weight, self.eps)
                return out.view(hidden_states.shape)
            if residual is not None:
                residual = residual + hidden_states
                hidden_states = residual
            dtype = hidden_states.dtype
            normalized = hidden_states.float()
            variance = normalized.pow(2).mean(dim=-1, keepdim=True)
            normalized = normalized * torch.rsqrt(variance + self.eps)
            output = self.weight.to(dtype=dtype) * normalized.to(dtype)
            if residual is None:
                return output
            return output, residual

    class RMarinFlashInferAttention(nn.Module):
        """Llama projections with RMarin parameters and CUSTOM attention."""

        def __init__(
            self,
            config: Any,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            max_position_embeddings: int = 8192,
            quant_config: Any | None = None,
            bias: bool = False,
            bias_o_proj: bool = False,
            cache_config: Any | None = None,
            prefix: str = "",
            attn_type: str = attention_type_module.AttentionType.DECODER,
        ) -> None:
            super().__init__()
            if cache_config is None:
                raise RMarinVllmModelError(
                    "RMarin requires an explicit vLLM cache config"
                )
            if quant_config is not None:
                raise RMarinVllmModelError(
                    "RMarin FlashInfer P0 does not support model quantization"
                )
            self.layer_index = utils.extract_layer_index(prefix)
            layer_specs = validate_rmarin_hf_config(config)
            self.rmarin_checkpoint_spec = layer_specs[self.layer_index]
            self.hidden_size = hidden_size
            self.total_num_heads = num_heads
            self.total_num_kv_heads = num_kv_heads
            self.num_heads = num_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = self.rmarin_checkpoint_spec.head_dim
            self.q_size = self.num_heads * self.head_dim
            self.kv_size = self.num_kv_heads * self.head_dim
            self.scaling = self.head_dim**-0.5
            self.max_position_embeddings = max_position_embeddings

            self.qkv_proj = llama.QKVParallelLinear(
                hidden_size=hidden_size,
                head_size=self.head_dim,
                total_num_heads=self.total_num_heads,
                total_num_kv_heads=self.total_num_kv_heads,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.o_proj = llama.RowParallelLinear(
                input_size=self.total_num_heads * self.head_dim,
                output_size=hidden_size,
                bias=bias_o_proj,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )
            # RMARIN_ROPE_MAX_POSITION (2026-08-05): the cos/sin cache was
            # sized by config mpe (8192); absolute positions beyond it read out
            # of bounds -> garbage rope, collapsing everything past 8k (PTL NLL
            # 7+, RULER 0.0 at 16k+). Extending max_position keeps identical
            # values for in-range positions (cache rows depend only on pos).
            _rope_max = max(
                max_position_embeddings,
                int(os.environ.get("RMARIN_ROPE_MAX_POSITION", "0") or 0),
            )
            self.rotary_emb = llama.get_rope(
                self.head_dim,
                max_position=_rope_max,
                rope_parameters=getattr(config, "rope_parameters", None),
                is_neox_style=True,
            )

            spec = self.rmarin_checkpoint_spec
            self.prev_chunk_alphas = nn.Parameter(torch.zeros(self.num_kv_heads))
            if spec.use_qk_norm:
                self.q_norm = TrainingEagerRMSNorm(
                    self.head_dim,
                    eps=spec.rms_norm_eps,
                )
                self.k_norm = TrainingEagerRMSNorm(
                    self.head_dim,
                    eps=spec.rms_norm_eps,
                )
            else:
                self.q_norm = nn.Identity()
                self.k_norm = nn.Identity()
            scale = torch.tensor(spec.scale_init, dtype=torch.float32)
            if spec.source_mode == "vnormqk":
                self.deep_v_norm = TrainingEagerRMSNorm(
                    self.head_dim,
                    eps=spec.rms_norm_eps,
                )
                with torch.no_grad():
                    self.deep_v_norm.weight.fill_(
                        sum(spec.scale_init) / len(spec.scale_init)
                    )
            elif spec.source_mode in {"v", "vnorm"} and not spec.deep_scale_frozen:
                self.deep_v_scale = nn.Parameter(scale)
            else:
                self.register_buffer(
                    "deep_v_scale",
                    scale,
                    persistent=False,
                )

            # This cache exists solely because vLLM requires Attention layers
            # to contribute a KV spec and metadata binding.  Layer zero owns
            # one block-window placeholder group; every later layer uses
            # vLLM's public KV-sharing binding to that group.  The scheduler
            # may provision enough placeholder blocks for one scheduled span,
            # but it cannot reserve a full-history cache per layer.  The
            # composite session owns the actual full source and 2C locals.
            placeholder_window = int(cache_config.block_size)
            if _fastmode.PAGED_SOURCE:
                # full-attention placeholder: vLLM allocates one block per 16
                # tokens of every request and those block ids ARE our source
                # pages (see fastmode.PAGED_SOURCE); the spec itself is tiny.
                placeholder_window = None
            target_layer_name = None
            if self.layer_index:
                marker = f".layers.{self.layer_index}."
                if marker not in prefix:
                    raise RMarinVllmModelError(
                        "cannot derive the layer-zero placeholder cache name"
                    )
                target_layer_name = prefix.replace(marker, ".layers.0.", 1) + ".attn"
            self.attn = attention_module.Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                per_layer_sliding_window=placeholder_window,
                attn_type=attn_type,
                attn_backend=attention_backend_type,
                kv_sharing_target_layer_name=target_layer_name,
                prefix=f"{prefix}.attn",
            )
            self.attn.rmarin_layer_index = self.layer_index
            self.attn.rmarin_stock_placeholder_tokens = placeholder_window
            self.attn.rmarin_stock_placeholder_owner = self.layer_index == 0
            if _fastmode.PAGED_SOURCE and self.layer_index == 0:
                kv_iface = importer("vllm.v1.kv_cache_interface")
                _attn = self.attn
                _spec_type = kv_iface.FullAttentionSpec

                def _tiny_spec(vllm_config: Any, _attn: Any = _attn, _spec_type: Any = _spec_type) -> Any:
                    return _spec_type(
                        block_size=int(vllm_config.cache_config.block_size),
                        num_kv_heads=1,
                        head_size=int(_fastmode.PLACEHOLDER_HEAD_SIZE),
                        dtype=_attn.kv_cache_torch_dtype,
                    )

                self.attn.get_kv_cache_spec = _tiny_spec
            if self.attn.attn_backend.get_name() != "CUSTOM":
                raise RMarinVllmModelError(
                    "RMarin model did not construct the CUSTOM attention backend"
                )

        def recurrent_session_parameters(self) -> Any:
            spec = self.rmarin_checkpoint_spec
            value_scales = (
                getattr(self, "deep_v_scale", None)
                if spec.source_mode in {"v", "vnorm"}
                else None
            )
            value_norm_weights = (
                self.deep_v_norm.weight if spec.source_mode == "vnormqk" else None
            )
            return session_module.RMarinLayerParameters(
                alpha_logits=self.prev_chunk_alphas,
                source_mode=spec.source_mode,
                value_scales=value_scales,
                value_norm_weights=value_norm_weights,
                epsilon=(spec.rms_norm_eps if spec.source_mode == "vnormqk" else 1e-6),
            )

        def rmarin_fold_value_scale(self) -> None:
            """MERGED_ATTN: fold the per-KV-head value scale into the weights.

            W_v rows of head h are divided by s[h] and the o_proj input columns
            of its query heads are multiplied by s[h], so this layer's V
            projection yields V/s (what the ring stores) while the attention
            output times o_proj is unchanged: W_o (s * attn(.., V/s)).
            """

            spec = self.rmarin_checkpoint_spec
            if spec.source_mode != "v":
                raise RMarinVllmModelError("merged attention requires source_mode=v")
            if getattr(self.qkv_proj, "bias", None) is not None or getattr(self.o_proj, "bias", None) is not None:
                raise RMarinVllmModelError("merged attention weight folding assumes bias-free projections")
            with torch.no_grad():
                scale = self.deep_v_scale.detach().to(torch.float32).reshape(-1)
                if int(scale.numel()) != self.num_kv_heads:
                    raise RMarinVllmModelError("deep_v_scale must have one entry per KV head")
                if not bool(torch.isfinite(scale).all()) or bool((scale == 0).any()):
                    raise RMarinVllmModelError("deep_v_scale must be finite and non-zero to fold")
                w = self.qkv_proj.weight
                if tuple(w.shape) != (self.q_size + 2 * self.kv_size, self.hidden_size):
                    raise RMarinVllmModelError("unexpected qkv_proj weight layout (TP=1 expected)")
                v_rows = w[self.q_size + self.kv_size :].view(self.num_kv_heads, self.head_dim, -1)
                v_rows.copy_((v_rows.to(torch.float32) / scale.view(-1, 1, 1)).to(w.dtype))
                wo = self.o_proj.weight
                if tuple(wo.shape) != (self.hidden_size, self.q_size):
                    raise RMarinVllmModelError("unexpected o_proj weight layout (TP=1 expected)")
                group = self.num_heads // self.num_kv_heads
                cols = wo.view(self.hidden_size, self.num_kv_heads, group * self.head_dim)
                cols.copy_((cols.to(torch.float32) * scale.view(1, -1, 1)).to(wo.dtype))

        def forward(self, positions: Any, hidden_states: Any) -> Any:
            qkv, _ = self.qkv_proj(hidden_states)
            query, key, value = qkv.split(
                [self.q_size, self.kv_size, self.kv_size],
                dim=-1,
            )
            fast = _fastpath.current()
            if fast is not None and fast.fused_qkv and fast.active_generation is not None:
                # L5: rotary + q/k RMSNorm happen inside the fast path's fused
                # per-layer kernel (bit-identical); hand it raw q/k/v + positions.
                fast.positions = positions
                attention_output = self.attn(query, key, value)
                output, _ = self.o_proj(attention_output)
                return output
            query, key = self.rotary_emb(positions, query, key)
            query = self.q_norm(query.view(-1, self.num_heads, self.head_dim)).reshape(
                -1, self.q_size
            )
            key = self.k_norm(key.view(-1, self.num_kv_heads, self.head_dim)).reshape(
                -1, self.kv_size
            )
            # Attention reshapes q/k/v and its output buffer to [T,H,D]
            # before invoking the custom implementation.
            attention_output = self.attn(query, key, value)
            output, _ = self.o_proj(attention_output)
            return output

    class RMarinFlashInferDecoderLayer(llama.LlamaDecoderLayer):
        def __init__(self, vllm_config: Any, prefix: str = "") -> None:
            super().__init__(
                vllm_config=vllm_config,
                prefix=prefix,
                attn_layer_type=RMarinFlashInferAttention,
            )
            config = vllm_config.model_config.hf_config
            self.input_layernorm = TrainingEagerRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = TrainingEagerRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )

    class RMarinFlashInferModel(llama.LlamaModel):
        def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
            constraints = validate_vllm_runtime_config(vllm_config)
            specs = validate_rmarin_hf_config(vllm_config.model_config.hf_config)
            if vllm_config.quant_config is not None:
                raise RMarinVllmModelError(
                    "RMarin FlashInfer P0 does not support model quantization"
                )
            super().__init__(
                vllm_config=vllm_config,
                prefix=prefix,
                layer_type=RMarinFlashInferDecoderLayer,
            )
            config = vllm_config.model_config.hf_config
            self.norm = TrainingEagerRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            attentions = tuple(
                self.layers[index].self_attn for index in range(len(specs))
            )
            layer_parameters = tuple(
                attention.recurrent_session_parameters() for attention in attentions
            )
            self.rmarin_composite_session = factory(
                vllm_config=vllm_config,
                layer_parameters=layer_parameters,
                importer=importer,
            )
            memory_budget = RMarinVllmMemoryBudget(
                max_sequence_tokens=int(vllm_config.model_config.max_model_len),
                num_layers=len(attentions),
                chunk_size=constraints.chunk_size,
                num_kv_heads=specs[0].num_kv_heads,
                head_dim=specs[0].head_dim,
            )
            first_parameter = attentions[0].prev_chunk_alphas
            device = first_parameter.device
            cuda = getattr(torch, "cuda", None)
            if (
                cuda is None
                or not callable(getattr(cuda, "is_available", None))
                or not cuda.is_available()
                or not str(device).startswith("cuda")
            ):
                raise RMarinVllmModelError(
                    "RMarin FlashInfer P0 requires CUDA model construction"
                )
            pool = self.rmarin_composite_session
            num_slots = int(getattr(pool, "num_slots", 0))
            scheduler_slots = int(vllm_config.scheduler_config.max_num_seqs)
            if num_slots < scheduler_slots:
                raise RMarinVllmModelError(
                    "session pool has fewer slots than scheduler max_num_seqs"
                )
            total_gpu_bytes = int(cuda.get_device_properties(device).total_memory)
            gpu_memory_utilization = getattr(
                vllm_config.cache_config,
                "gpu_memory_utilization",
                None,
            )
            utilization = float(gpu_memory_utilization)
            if not 0 < utilization < 1:
                raise RMarinVllmModelError(
                    "gpu_memory_utilization must be strictly between zero and one"
                )
            available = int(total_gpu_bytes * (1 - utilization))
            # The fast path reads rings/slabs zero-copy: its dynamic scratch is
            # per-step (FA workspace, merge outputs), not per slot.
            pooled_dynamic_peak = (
                memory_budget.dynamic_scratch_peak_bytes
                if _fastmode.FAST
                else num_slots * memory_budget.dynamic_scratch_peak_bytes
            )
            required = pooled_dynamic_peak + DEFAULT_DYNAMIC_SAFETY_BYTES
            if available < required:
                raise RMarinVllmModelError(
                    "gpu_memory_utilization leaves insufficient RMarin batched "
                    f"dynamic headroom: available={available} required={required}; "
                    "lower gpu_memory_utilization"
                )
            dynamic_headroom = available
            template_key = torch.empty(
                (1, specs[0].num_kv_heads, specs[0].head_dim),
                dtype=vllm_config.model_config.dtype,
                device=device,
            )
            template_value = torch.empty_like(template_key)
            reserve = getattr(pool, "reserve", None)
            if not callable(reserve):
                raise RMarinVllmModelError(
                    "session pool cannot reserve memory before vLLM profiling"
                )
            reserve(template_key, template_value)
            session_memory_budget = pool.memory_budget(
                num_kv_heads=specs[0].num_kv_heads,
                head_dim=specs[0].head_dim,
                bytes_per_element=2,
            )
            if (
                session_memory_budget.preprofile_reserved_bytes
                != memory_budget.persistent_reserved_bytes
                or session_memory_budget.dynamic_scratch_upper_bound_bytes
                != memory_budget.dynamic_scratch_peak_bytes
            ):
                raise RMarinVllmModelError(
                    "model and composite-session GPU memory budgets differ"
                )
            self.rmarin_memory_budget = memory_budget
            self.rmarin_memory_receipt = {
                **memory_budget.receipt(),
                "total_gpu_bytes": total_gpu_bytes,
                "gpu_memory_utilization": utilization,
                "pool_slots": num_slots,
                "pooled_dynamic_scratch_peak_bytes": pooled_dynamic_peak,
                "dynamic_headroom_bytes": dynamic_headroom,
                "dynamic_safety_bytes": DEFAULT_DYNAMIC_SAFETY_BYTES,
                "preprofile_reservation_complete": True,
                "fused_onread": bool(getattr(pool, "fused_onread", False)),
                "fused_slab_bytes": int(getattr(pool, "fused_slab_bytes", 0)),
            }
            install_runner_capture()
            if getattr(pool, "paged_source", False):
                # vLLM's block ids index our source pool: pin its block count to ours
                vllm_config.cache_config.num_gpu_blocks_override = int(pool.source_pages)
                print({"rmarin_paged_source": {"source_pages": int(pool.source_pages), "source_tokens": int(pool.source_tokens),
                                               "ring_slots": int(pool.num_slots), "pool_gib": round(pool.fused_slab_bytes / 2**30, 2),
                                               "prefix_caching": bool(vllm_config.cache_config.enable_prefix_caching)}}, flush=True)
            self.rmarin_position_tracker = RMarinBatchSpanTracker(
                chunk_size=constraints.chunk_size,
            )
            if _fastmode.FDP and getattr(pool, "_ring_k", None) is not None:
                from . import fastpath as _fastpath

                capture_sizes = list(
                    getattr(vllm_config.compilation_config, "cudagraph_capture_sizes", None) or []
                )
                max_batch = max(
                    int(vllm_config.scheduler_config.max_num_seqs),
                    max(capture_sizes) if capture_sizes else 0,
                )
                self.rmarin_fastpath = _fastpath.install(
                    pool, self.rmarin_position_tracker, max_batch=max_batch, sm_scale=pool.sm_scale,
                    max_tokens=int(vllm_config.scheduler_config.max_num_batched_tokens) + max_batch,
                )
                print({"rmarin_fastpath": "installed", "max_batch": max_batch, "capture_sizes": capture_sizes}, flush=True)
                if _fastmode.FUSED_QKV and getattr(pool, "merged", False):
                    if all(
                        isinstance(attention.q_norm, TrainingEagerRMSNorm)
                        and isinstance(attention.k_norm, TrainingEagerRMSNorm)
                        for attention in attentions
                    ):
                        self.rmarin_fastpath.register_fused_qkv(
                            [(attention.q_norm, attention.k_norm, attention.rotary_emb) for attention in attentions]
                        )
                    else:
                        print({"rmarin_fused_qkv": "disabled (model has no q/k RMSNorm)"}, flush=True)
            bindings = RMarinBatchBindings(
                num_layers=len(attentions),
                source_layer_index=len(attentions) - 1,
                runtime_constraints=constraints,
                tracker=self.rmarin_position_tracker,
                pool=pool,
                layer_index_provider=lambda layer: int(layer.rmarin_layer_index),
                runtime_validator=lambda active_config: _same_runtime_constraints(
                    constraints,
                    active_config,
                ),
            )
            for attention in attentions:
                setattr(
                    attention.attn,
                    MODEL_ADAPTER_BINDINGS_ATTRIBUTE,
                    bindings,
                )
            self.rmarin_attention_bindings = bindings
            self.rmarin_runtime_constraints = constraints
            self.rmarin_attentions = attentions
            self.rmarin_scales_folded = False

        def rmarin_fold_value_scales(self) -> None:
            if self.rmarin_scales_folded:
                return
            for attention in self.rmarin_attentions:
                attention.rmarin_fold_value_scale()
            self.rmarin_scales_folded = True
            print({"rmarin_merged_attn": "value scales folded into qkv_proj/o_proj", "layers": len(self.rmarin_attentions)}, flush=True)

        def forward(
            self,
            input_ids: Any | None,
            positions: Any,
            intermediate_tensors: Any | None,
            inputs_embeds: Any | None = None,
            **extra_layer_kwargs: Any,
        ) -> Any:
            # vLLM's pinned profile path deliberately passes metadata=None and
            # leaves its reusable positions buffer zero-filled.  Stock
            # FlashInfer treats that exact state as a skip-attention dummy
            # forward.  Do not stage request ownership from those synthetic
            # positions; every non-dummy path remains fail-closed below.
            forward_context = forward_context_module.get_forward_context()
            fast = _fastpath.current()
            fast_active = fast is not None and fast.active_generation is not None
            if getattr(forward_context, "attn_metadata", None) is not None and not fast_active:
                self.rmarin_position_tracker.stage_positions(positions)
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                **extra_layer_kwargs,
            )

    class RMarinFlashInferForCausalLM(llama.LlamaForCausalLM):
        def _init_model(
            self,
            vllm_config: Any,
            prefix: str = "",
            layer_type: type[Any] = RMarinFlashInferDecoderLayer,
        ) -> Any:
            del layer_type
            return RMarinFlashInferModel(
                vllm_config=vllm_config,
                prefix=prefix,
            )

        def load_weights(
            self,
            weights: Iterable[tuple[str, Any]],
        ) -> set[str]:
            # The stock AutoWeightsLoader sees recurrent parameters at their
            # original checkpoint paths under model.layers.N.self_attn. Track
            # both the streaming checkpoint input and the loader receipt so a
            # missing, duplicate, or config-incompatible recurrent tensor
            # cannot leave a partially initialized model serving requests.
            expected = expected_rmarin_checkpoint_keys(self.config)
            observed: set[str] = set()
            duplicates: set[str] = set()

            def tracked_weights() -> Iterable[tuple[str, Any]]:
                for name, tensor in weights:
                    if _is_recurrent_checkpoint_key(name):
                        if name in observed:
                            duplicates.add(name)
                        observed.add(name)
                    yield name, tensor

            loaded = super().load_weights(tracked_weights())
            unexpected = observed - expected
            missing_checkpoint = expected - observed
            missing_loaded = expected - loaded
            if duplicates or unexpected or missing_checkpoint or missing_loaded:
                details = []
                for label, values in (
                    ("duplicate", duplicates),
                    ("unexpected", unexpected),
                    ("missing_checkpoint", missing_checkpoint),
                    ("missing_loaded", missing_loaded),
                ):
                    if values:
                        details.append(f"{label}={','.join(sorted(values))}")
                raise RMarinVllmModelError(
                    "RMarin recurrent checkpoint binding failed: " + "; ".join(details)
                )
            self.rmarin_weight_receipt = {
                "expected_recurrent_tensors": len(expected),
                "observed_recurrent_tensors": len(observed),
                "loaded_recurrent_tensors": len(expected & loaded),
                "exact_recurrent_binding": True,
            }
            pool = self.model.rmarin_composite_session
            if hasattr(pool, "invalidate_parameter_caches"):
                pool.invalidate_parameter_caches()
            if getattr(pool, "ring_v_scaled", False) and _fastmode.FOLD_SCALES:
                self.model.rmarin_fold_value_scales()
            return loaded

    classes = {
        "TrainingEagerRMSNorm": TrainingEagerRMSNorm,
        "RMarinFlashInferAttention": RMarinFlashInferAttention,
        "RMarinFlashInferDecoderLayer": RMarinFlashInferDecoderLayer,
        "RMarinFlashInferModel": RMarinFlashInferModel,
        RMARIN_MODEL_CLASS: RMarinFlashInferForCausalLM,
    }
    for runtime_name, runtime_class in classes.items():
        runtime_class.__name__ = runtime_name
        runtime_class.__qualname__ = runtime_name
        runtime_class.__module__ = __name__
    return (
        RMarinFlashInferAttention,
        RMarinFlashInferDecoderLayer,
        RMarinFlashInferModel,
        RMarinFlashInferForCausalLM,
    )


def _runtime_types() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    global _RUNTIME_TYPES
    if _RUNTIME_TYPES is None:
        _RUNTIME_TYPES = build_rmarin_vllm_model_types()
    return _RUNTIME_TYPES


def __getattr__(name: str) -> Any:
    names = {
        "RMarinFlashInferAttention": 0,
        "RMarinFlashInferDecoderLayer": 1,
        "RMarinFlashInferModel": 2,
        RMARIN_MODEL_CLASS: 3,
    }
    index = names.get(name)
    if index is None:
        raise AttributeError(name)
    return _runtime_types()[index]


__all__ = [
    "ACTIVE_REQUEST_ID",
    "RECURRENT_ARCHITECTURE",
    "RMARIN_MODEL_CLASS",
    "RMARIN_MODEL_IMPORT",
    "RMarinCheckpointLayerSpec",
    "RMarinPositionSpanTracker",
    "RMarinVllmMemoryBudget",
    "RMarinVllmModelError",
    "build_rmarin_vllm_model_types",
    "expected_rmarin_checkpoint_keys",
    "validate_rmarin_hf_config",
]
