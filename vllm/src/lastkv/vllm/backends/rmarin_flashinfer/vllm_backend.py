"""Thin, fail-closed vLLM 0.25 adapter for RMarin FlashInfer attention.

This module does not import torch, vLLM, or FlashInfer at import time.  vLLM
resolves the backend class lazily after the general plugin registers the
``CUSTOM`` backend.

The adapter deliberately does not pretend that an attention backend can load
RMarin checkpoint weights.  A model/cache bridge attached to every attention
layer must:

* apply the configured Q/K norm (or bind the explicit identity path);
* load and apply ``prev_chunk_alphas``;
* load and apply the source V scale/normalization;
* preserve raw top-layer source KV separately from effective local KV;
* maintain boundary-safe, per-request source/local page tables; and
* return a shared :class:`RMarinLayerExecution` containing real FlashInfer
  return-LSE wrappers.

The common executor issues the source/local calls and combines them with
``flashinfer.cascade.merge_state``.  Missing or stale bridge state fails
closed instead of falling back to standard Transformer attention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
import inspect
import os
from typing import Any

from .batching import (
    SCHEDULER_CLASS_PATH,
    RMarinBatchLayerExecution,
    execute_rmarin_flashinfer_batch,
)
from .runtime_contract import (
    DEFAULT_BLOCK_SIZE,
    RMarinBackendContractError,
    RMarinFrameworkBindings,
    RMarinRuntimeConstraints,
    VLLM_FLASHINFER_VERSION,
    VLLM_VERSION,
)


PINNED_VLLM_VERSION = VLLM_VERSION
PINNED_FLASHINFER_VERSION = VLLM_FLASHINFER_VERSION
BACKEND_CLASS_PATH = (
    "lastkv.vllm.backends.rmarin_flashinfer.vllm_backend.RMarinFlashInferBackend"
)
MODEL_ADAPTER_BINDINGS_ATTRIBUTE = "rmarin_attention_bindings"

_REGISTRY_MODULE = "vllm.v1.attention.backends.registry"
_FLASHINFER_BACKEND_MODULE = "vllm.v1.attention.backends.flashinfer"
_FLASHINFER_CASCADE_MODULE = "flashinfer.cascade"

_EXPECTED_REGISTER_PARAMETERS = ("backend", "class_path", "is_mamba")
_EXPECTED_IMPL_INIT_PARAMETERS = (
    "self",
    "num_heads",
    "head_size",
    "scale",
    "num_kv_heads",
    "alibi_slopes",
    "sliding_window",
    "kv_cache_dtype",
    "logits_soft_cap",
    "attn_type",
    "kv_sharing_target_layer_name",
    "sinks",
)
_EXPECTED_IMPL_FORWARD_PARAMETERS = (
    "self",
    "layer",
    "query",
    "key",
    "value",
    "kv_cache",
    "attn_metadata",
    "output",
    "output_scale",
    "output_block_scale",
)
_EXPECTED_MERGE_PARAMETERS = ("v_a", "s_a", "v_b", "s_b")
_MODEL_BINDING_MARKERS = (
    "qk_norm_bound",
    "alpha_bound",
    "source_scaling_bound",
    "top_layer_dual_cache_bound",
)


class RMarinVllmCompatibilityError(RMarinBackendContractError):
    """The installed runtime differs from the pinned, audited API."""


def _public_version(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RMarinVllmCompatibilityError(f"{label} version is missing")
    return value.split("+", 1)[0]


def _parameter_names(value: Any, *, label: str) -> tuple[str, ...]:
    try:
        return tuple(inspect.signature(value).parameters)
    except (TypeError, ValueError) as error:
        raise RMarinVllmCompatibilityError(
            f"cannot inspect the pinned {label} API; rebuild the RMarin "
            "integration against the deployed runtime"
        ) from error


def _require_signature(
    value: Any,
    expected: tuple[str, ...],
    *,
    label: str,
) -> None:
    observed = _parameter_names(value, label=label)
    if observed != expected:
        raise RMarinVllmCompatibilityError(
            f"{label} signature changed: expected {expected}, observed "
            f"{observed}. Re-pin and validate the RMarin backend before use."
        )


@dataclass(frozen=True)
class VllmRuntimeApiEvidence:
    """Audited runtime identity and callable signatures."""

    vllm_version: str
    flashinfer_version: str
    register_backend_parameters: tuple[str, ...]
    impl_init_parameters: tuple[str, ...]
    impl_forward_parameters: tuple[str, ...]
    merge_state_parameters: tuple[str, ...]


def validate_vllm_runtime_api(
    *,
    importer: Callable[[str], Any] = import_module,
    version_getter: Callable[[str], str] | None = None,
) -> VllmRuntimeApiEvidence:
    """Validate the exact vLLM/FlashInfer API used by this adapter."""

    if version_getter is None:
        from importlib.metadata import version

        version_getter = version

    vllm_version = version_getter("vllm")
    flashinfer_version = version_getter("flashinfer-python")
    if _public_version(vllm_version, label="vLLM") != PINNED_VLLM_VERSION:
        raise RMarinVllmCompatibilityError(
            "RMarin FlashInfer is pinned to vLLM "
            f"{PINNED_VLLM_VERSION}, observed {vllm_version!r}"
        )
    if (
        _public_version(flashinfer_version, label="FlashInfer")
        != PINNED_FLASHINFER_VERSION
    ):
        raise RMarinVllmCompatibilityError(
            "RMarin FlashInfer is pinned to flashinfer-python "
            f"{PINNED_FLASHINFER_VERSION}, observed {flashinfer_version!r}"
        )

    registry = importer(_REGISTRY_MODULE)
    backend_module = importer(_FLASHINFER_BACKEND_MODULE)
    cascade = importer(_FLASHINFER_CASCADE_MODULE)
    try:
        register_backend = registry.register_backend
        custom = registry.AttentionBackendEnum.CUSTOM
        backend_class = backend_module.FlashInferBackend
        impl_class = backend_module.FlashInferImpl
        merge_state = cascade.merge_state
    except AttributeError as error:
        raise RMarinVllmCompatibilityError(
            "vLLM 0.25/FlashInfer API is missing a pinned RMarin symbol; "
            "rebuild the integration against the deployed runtime"
        ) from error

    _require_signature(
        register_backend,
        _EXPECTED_REGISTER_PARAMETERS,
        label="vLLM register_backend",
    )
    _require_signature(
        impl_class.__init__,
        _EXPECTED_IMPL_INIT_PARAMETERS,
        label="vLLM FlashInferImpl.__init__",
    )
    _require_signature(
        impl_class.forward,
        _EXPECTED_IMPL_FORWARD_PARAMETERS,
        label="vLLM FlashInferImpl.forward",
    )
    _require_signature(
        merge_state,
        _EXPECTED_MERGE_PARAMETERS,
        label="FlashInfer merge_state",
    )
    for name in (
        "get_builder_cls",
        "get_impl_cls",
        "get_kv_cache_shape",
        "get_kv_cache_stride_order",
    ):
        if not callable(getattr(backend_class, name, None)):
            raise RMarinVllmCompatibilityError(
                f"vLLM FlashInferBackend.{name} is unavailable; re-pin and "
                "validate the RMarin backend"
            )
    if custom is None:
        raise RMarinVllmCompatibilityError(
            "vLLM AttentionBackendEnum.CUSTOM is unavailable"
        )

    return VllmRuntimeApiEvidence(
        vllm_version=vllm_version,
        flashinfer_version=flashinfer_version,
        register_backend_parameters=_parameter_names(
            register_backend,
            label="vLLM register_backend",
        ),
        impl_init_parameters=_parameter_names(
            impl_class.__init__,
            label="vLLM FlashInferImpl.__init__",
        ),
        impl_forward_parameters=_parameter_names(
            impl_class.forward,
            label="vLLM FlashInferImpl.forward",
        ),
        merge_state_parameters=_parameter_names(
            merge_state,
            label="FlashInfer merge_state",
        ),
    )


def rmarin_prefill_factor() -> int:
    """Scheduled-prefill budget multiplier (env RMARIN_PREFILL_FACTOR).

    The chunk-aligned scheduler always caps every request's span at its own
    next recurrent chunk boundary, so raising the global batched-token
    budget only allows several requests' prefill chunks to share one
    forward; per-request semantics are unchanged.  Default 1 preserves the
    audited W4 envelope exactly.
    """

    raw = os.environ.get("RMARIN_PREFILL_FACTOR", "1")
    try:
        factor = int(raw)
    except ValueError as exc:
        raise RMarinBackendContractError(
            f"RMARIN_PREFILL_FACTOR must be an integer, observed {raw!r}"
        ) from exc
    if not 1 <= factor <= 8:
        raise RMarinBackendContractError(
            f"RMARIN_PREFILL_FACTOR must be in [1, 8], observed {factor}"
        )
    return factor


def _lookup(value: Any, *path: str, default: Any = None) -> Any:
    current = value
    for part in path:
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
        else:
            current = getattr(current, part, default)
        if current is default:
            return default
    return current


def _dtype_name(value: Any) -> str:
    text = str(value).lower()
    if text.startswith("torch."):
        text = text.removeprefix("torch.")
    return text


def validate_vllm_runtime_config(vllm_config: Any) -> RMarinRuntimeConstraints:
    """Map vLLM config into the shared P0 correctness envelope."""

    model_config = _lookup(vllm_config, "model_config")
    hf_config = _lookup(model_config, "hf_config")
    chunk_size = _lookup(hf_config, "chunk_size")
    if chunk_size is None:
        raise RMarinBackendContractError(
            "vLLM RMarin model config has no recurrent chunk_size"
        )
    max_num_seqs = _lookup(vllm_config, "scheduler_config", "max_num_seqs")
    if isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int):
        raise RMarinBackendContractError(
            "vLLM scheduler_config.max_num_seqs is missing"
        )
    max_num_batched_tokens = _lookup(
        vllm_config,
        "scheduler_config",
        "max_num_batched_tokens",
    )
    prefill_factor = rmarin_prefill_factor()
    expected_batched_tokens = prefill_factor * chunk_size + max_num_seqs
    if max_num_batched_tokens != expected_batched_tokens:
        raise RMarinBackendContractError(
            "RMarin batched mode requires scheduler_config.max_num_batched_tokens "
            "to equal RMARIN_PREFILL_FACTOR * chunk_size + max_num_seqs = "
            f"{expected_batched_tokens}; observed {max_num_batched_tokens}"
        )
    long_prefill_threshold = _lookup(
        vllm_config,
        "scheduler_config",
        "long_prefill_token_threshold",
    )
    if long_prefill_threshold != chunk_size:
        raise RMarinBackendContractError(
            "RMarin batched mode requires "
            f"long_prefill_token_threshold == chunk_size ({chunk_size}); "
            f"observed {long_prefill_threshold}"
        )
    scheduler_cls = _lookup(vllm_config, "scheduler_config", "scheduler_cls")
    scheduler_path = (
        scheduler_cls
        if isinstance(scheduler_cls, str)
        else (
            f"{getattr(scheduler_cls, '__module__', '')}."
            f"{getattr(scheduler_cls, '__qualname__', '')}"
        )
    )
    if scheduler_path != SCHEDULER_CLASS_PATH:
        raise RMarinBackendContractError(
            "RMarin batched mode requires scheduler_cls="
            f"{SCHEDULER_CLASS_PATH!r} so scheduled spans never cross a "
            f"recurrent chunk boundary; observed {scheduler_cls!r}"
        )
    block_size = _lookup(
        vllm_config,
        "cache_config",
        "block_size",
        default=DEFAULT_BLOCK_SIZE,
    )
    dcp = _lookup(
        vllm_config,
        "parallel_config",
        "decode_context_parallel_size",
        default=1,
    )
    if dcp is None:
        dcp = 1
    cudagraph_mode = _lookup(
        vllm_config,
        "compilation_config",
        "cudagraph_mode",
        default=None,
    )
    cuda_graph = False
    if cudagraph_mode is not None:
        cuda_graph = str(cudagraph_mode).lower() not in {
            "none",
            "cudagraphmode.none",
        }

    constraints = RMarinRuntimeConstraints(
        chunk_size=chunk_size,
        block_size=block_size,
        dtype=_dtype_name(_lookup(model_config, "dtype")),
        kv_cache_dtype=_dtype_name(
            _lookup(vllm_config, "cache_config", "cache_dtype", default="auto")
        ),
        tensor_parallel_size=_lookup(
            vllm_config,
            "parallel_config",
            "tensor_parallel_size",
        ),
        pipeline_parallel_size=_lookup(
            vllm_config,
            "parallel_config",
            "pipeline_parallel_size",
        ),
        context_parallel_size=dcp,
        max_active_sequences=_lookup(
            vllm_config,
            "scheduler_config",
            "max_num_seqs",
        ),
        enforce_eager=_lookup(model_config, "enforce_eager") is True,
        prefix_cache=bool(
            _lookup(
                vllm_config,
                "cache_config",
                "enable_prefix_caching",
                default=False,
            )
        ),
        speculative_decoding=_lookup(vllm_config, "speculative_config") is not None,
        beam_search=bool(
            _lookup(
                vllm_config, "scheduler_config", "enable_beam_search", default=False
            )
        ),
        copy_on_write=bool(
            _lookup(vllm_config, "cache_config", "enable_copy_on_write", default=False)
        ),
        cuda_graph=cuda_graph,
        kv_offload_or_connector=bool(
            _is_foreign_kv_connector(_lookup(vllm_config, "kv_transfer_config", default=None))
            or _lookup(vllm_config, "kv_offloading_config", default=None)
        ),
    )
    return constraints


def _is_foreign_kv_connector(kv_transfer_config: Any) -> bool:
    """Only the RMarin session-resume connector is permitted (L2 prototype).

    It never moves KV bytes through vLLM's paged cache; it only reports
    externally-computed tokens for prefixes whose recurrent state is resident
    in a session slot.  Any other connector/offload keeps the P0 prohibition.
    """

    if kv_transfer_config is None:
        return False
    name = _lookup(kv_transfer_config, "kv_connector", default=None)
    module = _lookup(kv_transfer_config, "kv_connector_module_path", default=None)
    return not (
        name == "RMarinResumeConnector"
        and str(module or "").endswith("rmarin_flashinfer.resume")
    )


def _require_model_bindings(layer: Any) -> RMarinFrameworkBindings:
    bindings = getattr(layer, MODEL_ADAPTER_BINDINGS_ATTRIBUTE, None)
    # SPEED-P4: marker validation is pure; memoize per layer+bindings object.
    checked = getattr(layer, "_rmarin_bindings_checked", None)
    if checked is not None and checked is bindings:
        return bindings
    if not isinstance(bindings, RMarinFrameworkBindings):
        raise RMarinBackendContractError(
            "vLLM RMarin requires layer.rmarin_attention_bindings; the model "
            "adapter must bind recurrent weights, Q/K norm, source scaling, "
            "dual caches, and boundary-safe page tables"
        )
    missing = [
        name
        for name in _MODEL_BINDING_MARKERS
        if getattr(bindings, name, None) is not True
    ]
    if missing:
        raise RMarinBackendContractError(
            "vLLM RMarin model-adapter contract is incomplete: " + ", ".join(missing)
        )
    try:
        layer._rmarin_bindings_checked = bindings
    except Exception:
        pass
    return bindings


_RUNTIME_TYPES: tuple[type[Any], type[Any]] | None = None


class RMarinLeanMetadata:
    """Minimal per-step attention metadata consumed by the RMarin impl."""

    __slots__ = (
        "num_actual_tokens",
        "slot_mapping",
        "causal",
        "use_cascade",
        "num_decode_tokens",
        "num_prefill_tokens",
        "num_decodes",
        "num_prefills",
        "num_reqs",
        "query_start_loc_cpu",
    )

    def __init__(self, *, num_actual_tokens, slot_mapping, num_decode_tokens, num_prefill_tokens, num_decodes, num_prefills, num_reqs, query_start_loc_cpu):
        self.num_actual_tokens = num_actual_tokens
        self.slot_mapping = slot_mapping
        self.causal = True
        self.use_cascade = False
        self.num_decode_tokens = num_decode_tokens
        self.num_prefill_tokens = num_prefill_tokens
        self.num_decodes = num_decodes
        self.num_prefills = num_prefills
        self.num_reqs = num_reqs
        self.query_start_loc_cpu = query_start_loc_cpu


_LEAN_BUILDER_TYPE: type[Any] | None = None


def build_lean_metadata_builder_type(importer: Callable[[str], Any] = import_module) -> type[Any]:
    """SPEED: skip the stock FlashInfer per-step wrapper planning.

    The RMarin impl ignores vLLM's paged placeholder cache; it only needs the
    token accounting and slot_mapping.  The stock builder would otherwise plan
    FlashInfer decode/prefill wrappers for the placeholder every step.
    """

    global _LEAN_BUILDER_TYPE
    if _LEAN_BUILDER_TYPE is not None:
        return _LEAN_BUILDER_TYPE
    backend_api = importer("vllm.v1.attention.backend")
    base = backend_api.AttentionMetadataBuilder
    cg = backend_api.AttentionCGSupport
    from . import fastmode

    class RMarinLeanMetadataBuilder(base):
        _cudagraph_support = (
            cg.UNIFORM_SINGLE_TOKEN_DECODE if fastmode.ALLOW_CUDAGRAPH else cg.NEVER
        )
        reorder_batch_threshold = 1

        def __init__(self, kv_cache_spec: Any, layer_names: Any, vllm_config: Any, device: Any) -> None:
            try:
                super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            except TypeError:
                pass
            self.kv_cache_spec = kv_cache_spec
            self.layer_names = layer_names
            self.vllm_config = vllm_config
            self.device = device

        def _fast_prologue(self, common_attn_metadata: Any, num_reqs: int, uniform_decode: bool) -> None:
            from . import fastpath

            fast = fastpath.current()
            if fast is None:
                return
            model_config = self.vllm_config.model_config
            query_heads = int(model_config.hf_config.num_attention_heads)
            query_dtype = model_config.dtype
            from .batching import _CAPTURE

            if fastpath.STATE.dummy_run or getattr(self, "_capture_mode", False) or _CAPTURE.value is None:
                fast.prepare_dummy(num_reqs, query_heads, query_dtype)
                return
            fast.active_generation = None
            if not uniform_decode and not fast.prefill_enabled:
                return
            staged = fast.tracker.stage_from_capture()
            if len(staged.spans) > num_reqs or staged.total_tokens > int(common_attn_metadata.num_actual_tokens):
                # stale capture: this build is not for a real scheduled step
                fast.prepare_dummy(num_reqs, query_heads, query_dtype)
                return
            if not fast.prepare_real(
                staged, num_reqs, query_heads, query_dtype,
                block_table=getattr(common_attn_metadata, "block_table_tensor", None),
                slot_mapping=getattr(common_attn_metadata, "slot_mapping", None),
            ):
                fastpath.STATE.stats["ineligible"] += 1

        def build(self, common_prefix_len: int, common_attn_metadata: Any, fast_build: bool = False) -> RMarinLeanMetadata:
            q = common_attn_metadata.query_start_loc_cpu
            num_reqs = int(common_attn_metadata.num_reqs)
            lens = (q[1 : num_reqs + 1] - q[:num_reqs]).tolist()
            decodes = sum(1 for item in lens if item == 1)
            total = int(common_attn_metadata.num_actual_tokens)
            self._fast_prologue(common_attn_metadata, num_reqs, int(common_attn_metadata.max_query_len) == 1)
            return RMarinLeanMetadata(
                num_actual_tokens=total,
                slot_mapping=common_attn_metadata.slot_mapping,
                num_decode_tokens=decodes,
                num_prefill_tokens=total - decodes,
                num_decodes=decodes,
                num_prefills=num_reqs - decodes,
                num_reqs=num_reqs,
                query_start_loc_cpu=q,
            )

        def build_for_cudagraph_capture(self, common_attn_metadata: Any) -> RMarinLeanMetadata:
            self._capture_mode = True
            try:
                return self.build(0, common_attn_metadata)
            finally:
                self._capture_mode = False

    RMarinLeanMetadataBuilder.__name__ = "RMarinLeanMetadataBuilder"
    RMarinLeanMetadataBuilder.__qualname__ = "RMarinLeanMetadataBuilder"
    RMarinLeanMetadataBuilder.__module__ = __name__
    _LEAN_BUILDER_TYPE = RMarinLeanMetadataBuilder
    return RMarinLeanMetadataBuilder


def build_vllm_backend_types(
    *,
    importer: Callable[[str], Any] = import_module,
    version_getter: Callable[[str], str] | None = None,
) -> tuple[type[Any], type[Any]]:
    """Construct ``(backend, impl)`` after validating the pinned runtime."""

    from . import fastpath as _fastpath

    validate_vllm_runtime_api(
        importer=importer,
        version_getter=version_getter,
    )
    backend_module = importer(_FLASHINFER_BACKEND_MODULE)
    cascade = importer(_FLASHINFER_CASCADE_MODULE)
    torch = importer("torch")
    base_backend = backend_module.FlashInferBackend
    base_impl = backend_module.FlashInferImpl
    get_current_vllm_config = getattr(
        backend_module,
        "get_current_vllm_config_or_none",
        None,
    )
    if not callable(get_current_vllm_config):
        raise RMarinVllmCompatibilityError(
            "vLLM FlashInfer backend does not expose the pinned current-config "
            "accessor needed for fail-closed RMarin runtime validation"
        )
    merge_state = cascade.merge_state

    class RMarinFlashInferImpl(base_impl):
        """vLLM AttentionImpl backed by the shared RMarin executor."""

        can_return_lse_for_decode = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            vllm_config = get_current_vllm_config()
            if vllm_config is None:
                raise RMarinBackendContractError(
                    "RMarin attention was constructed outside vLLM's config context"
                )
            self.rmarin_runtime_constraints = validate_vllm_runtime_config(vllm_config)
            self.rmarin_vllm_config = vllm_config

        def forward(
            self,
            layer: Any,
            query: Any,
            key: Any,
            value: Any,
            kv_cache: Any,
            attn_metadata: Any,
            output: Any,
            output_scale: Any | None = None,
            output_block_scale: Any | None = None,
        ) -> Any:
            del kv_cache  # Composite source/local caches belong to the bridge.
            if attn_metadata is None:
                return output.fill_(0)
            if output_scale is not None or output_block_scale is not None:
                raise RMarinBackendContractError(
                    "RMarin P0 disables fused output quantization"
                )
            fast = _fastpath.current()
            if fast is not None and fast.active_generation is not None:
                return fast.run_layer(
                    int(layer.rmarin_layer_index),
                    query.view(-1, self.num_heads, self.head_size),
                    key,
                    value,
                    output,
                )
            bindings = _require_model_bindings(layer)
            bindings.validate_runtime(self.rmarin_vllm_config)
            num_actual_tokens = getattr(attn_metadata, "num_actual_tokens", None)
            if (
                isinstance(num_actual_tokens, bool)
                or not isinstance(num_actual_tokens, int)
                or num_actual_tokens < 1
            ):
                raise RMarinBackendContractError(
                    "vLLM attention metadata num_actual_tokens is invalid"
                )
            query_view = query[:num_actual_tokens].view(
                -1, self.num_heads, self.head_size
            )
            key = key[:num_actual_tokens]
            value = value[:num_actual_tokens]
            execution = bindings.prepare_layer(
                backend=self,
                layer=layer,
                forward_batch=attn_metadata,
                query=query_view,
                key=key,
                value=value,
                save_kv_cache=True,
            )
            if not isinstance(execution, RMarinBatchLayerExecution):
                raise RMarinBackendContractError(
                    "vLLM RMarin model adapter must return RMarinBatchLayerExecution"
                )
            if len(execution.request_ids) < 1:
                raise RMarinBackendContractError(
                    "RMarin batched execution carries no requests"
                )
            merged_output = execute_rmarin_flashinfer_batch(
                query_view,
                execution,
                merger=merge_state,
            )
            output_actual = output[:num_actual_tokens]
            output_shape = tuple(int(item) for item in output_actual.shape)
            merged_shape = tuple(int(item) for item in merged_output.shape)
            if output_shape != merged_shape:
                raise RMarinBackendContractError(
                    "RMarin merged output shape differs from vLLM output buffer: "
                    f"{merged_shape} != {output_shape}"
                )
            # vLLM reshapes the output buffer to [tokens, heads, head_dim]
            # before invoking AttentionImpl.forward and flattens it again in
            # the outer Attention layer.  Preserve that runtime contract.
            output_actual.copy_(merged_output)
            if int(output.shape[0]) > num_actual_tokens:
                output[num_actual_tokens:].fill_(0)
            return output

    class RMarinFlashInferBackend(base_backend):
        """Custom vLLM backend using disjoint source/local FlashInfer states."""

        supported_dtypes = [torch.bfloat16]
        supported_kv_cache_dtypes = ["auto", "bfloat16"]
        # The model bridge writes both raw-source and effective-local caches.
        forward_includes_kv_cache_update = True

        @staticmethod
        def get_name() -> str:
            # vLLM maps this value back through AttentionBackendEnum[name].
            return "CUSTOM"

        @staticmethod
        def get_impl_cls() -> type[Any]:
            return RMarinFlashInferImpl

        @staticmethod
        def get_builder_cls() -> type[Any]:
            from . import fastmode

            if fastmode.LEAN_METADATA:
                return build_lean_metadata_builder_type(importer)
            return base_backend.get_builder_cls()

        @classmethod
        def supports_non_causal(cls) -> bool:
            return False

        @classmethod
        def supports_sink(cls) -> bool:
            return False

    RMarinFlashInferImpl.__name__ = "RMarinFlashInferImpl"
    RMarinFlashInferImpl.__qualname__ = "RMarinFlashInferImpl"
    RMarinFlashInferImpl.__module__ = __name__
    RMarinFlashInferBackend.__name__ = "RMarinFlashInferBackend"
    RMarinFlashInferBackend.__qualname__ = "RMarinFlashInferBackend"
    RMarinFlashInferBackend.__module__ = __name__
    return RMarinFlashInferBackend, RMarinFlashInferImpl


def _runtime_types() -> tuple[type[Any], type[Any]]:
    global _RUNTIME_TYPES
    if _RUNTIME_TYPES is None:
        _RUNTIME_TYPES = build_vllm_backend_types()
    return _RUNTIME_TYPES


def __getattr__(name: str) -> Any:
    """Resolve class-path targets without importing GPU packages eagerly."""

    if name == "RMarinFlashInferBackend":
        return _runtime_types()[0]
    if name == "RMarinFlashInferImpl":
        return _runtime_types()[1]
    raise AttributeError(name)


__all__ = [
    "BACKEND_CLASS_PATH",
    "MODEL_ADAPTER_BINDINGS_ATTRIBUTE",
    "PINNED_FLASHINFER_VERSION",
    "PINNED_VLLM_VERSION",
    "RMarinVllmCompatibilityError",
    "VllmRuntimeApiEvidence",
    "build_vllm_backend_types",
    "validate_vllm_runtime_api",
    "validate_vllm_runtime_config",
]
