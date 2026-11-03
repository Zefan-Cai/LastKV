"""Fail-closed compatibility helpers for the two recurrent-vLLM runtimes.

This module is deliberately dependency free.  Importing it must not import
torch or vLLM because the static compatibility probe also runs on login nodes.
Runtime modules resolve their vLLM symbols only after this module has selected
an exact, supported API profile.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
import importlib.metadata
from typing import Any


VLLM_0_11_0 = "0.11.0"
VLLM_0_25_0 = "0.25.0"
SUPPORTED_VLLM_VERSIONS = (VLLM_0_11_0, VLLM_0_25_0)

KV_LAYOUT_KV_BLOCKS = "kv_blocks"
KV_LAYOUT_BLOCKS_KV = "blocks_kv"


@dataclass(frozen=True)
class VllmRuntimeProfile:
    """Exact API choices bound to one supported vLLM release."""

    vllm_version: str
    torch_minor: str | None
    attention_module: str
    attention_type_module: str
    decoder_layer_strategy: str
    raw_source_write_strategy: str
    attention_context_strategy: str
    flash_attn_cache_layout: str
    engine_attention_strategy: str
    linear_primitive: str


_PROFILES = {
    VLLM_0_11_0: VllmRuntimeProfile(
        vllm_version=VLLM_0_11_0,
        torch_minor="2.8",
        attention_module="vllm.attention",
        attention_type_module="vllm.attention",
        decoder_layer_strategy="manual_llama_decoder",
        raw_source_write_strategy="attention_forward_discard",
        attention_context_strategy="forward_context",
        flash_attn_cache_layout=KV_LAYOUT_KV_BLOCKS,
        engine_attention_strategy="environment",
        linear_primitive="torch.nn.functional.linear",
    ),
    VLLM_0_25_0: VllmRuntimeProfile(
        vllm_version=VLLM_0_25_0,
        torch_minor=None,
        attention_module="vllm.model_executor.layers.attention",
        attention_type_module="vllm.v1.attention.backend",
        decoder_layer_strategy="injected_attention_type",
        raw_source_write_strategy="unified_kv_cache_update",
        attention_context_strategy="attention_context",
        flash_attn_cache_layout=KV_LAYOUT_BLOCKS_KV,
        engine_attention_strategy="constructor",
        linear_primitive="vllm.linear_batch_invariant",
    ),
}

_VLLM_0_11_DUMMY_FORWARD_ACTIVE: ContextVar[bool] = ContextVar(
    "recurrent_vllm_0_11_dummy_forward_active",
    default=False,
)
_VLLM_0_11_DUMMY_GUARD_MARKER = "_recurrent_vllm_0_11_dummy_forward_guard"


def _public_version(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} version is missing")
    # CUDA wheels commonly carry a local suffix (for example ``+cu128``).
    # The public version remains the API contract; pre/post/dev releases are
    # intentionally not coerced to a final release.
    return value.split("+", 1)[0]


def require_runtime_profile(
    *,
    vllm_version: str,
    torch_version: str,
) -> VllmRuntimeProfile:
    """Return the exact supported API profile or fail closed.

    vLLM 0.11.0 is the bounded torch-2.8 compatibility target.  vLLM 0.25.0
    keeps its existing runtime contract; its torch release remains governed by
    the pinned vLLM environment rather than broadened here.
    """

    public_vllm = _public_version(vllm_version, "vLLM")
    public_torch = _public_version(torch_version, "torch")
    profile = _PROFILES.get(public_vllm)
    if profile is None:
        raise RuntimeError(
            "recurrent P0 supports exactly vLLM "
            f"{', '.join(SUPPORTED_VLLM_VERSIONS)}, observed {vllm_version!r}"
        )
    if profile.torch_minor is not None:
        prefix = profile.torch_minor + "."
        if not public_torch.startswith(prefix):
            raise RuntimeError(
                f"vLLM {profile.vllm_version} recurrent P0 requires "
                f"torch {profile.torch_minor}.x, observed {torch_version!r}"
            )
    return profile


def installed_runtime_profile() -> VllmRuntimeProfile:
    """Select the installed runtime without importing either heavy package."""

    try:
        vllm_version = importlib.metadata.version("vllm")
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "vLLM and torch must be installed before loading the recurrent model"
        ) from error
    return require_runtime_profile(
        vllm_version=vllm_version,
        torch_version=torch_version,
    )


def resolve_attention_symbols(
    profile: VllmRuntimeProfile,
    *,
    importer: Callable[[str], Any] = import_module,
) -> tuple[type[Any], type[Any]]:
    """Resolve ``Attention`` and ``AttentionType`` from versioned locations."""

    attention_module = importer(profile.attention_module)
    attention_type_module = importer(profile.attention_type_module)
    try:
        attention = attention_module.Attention
        attention_type = attention_type_module.AttentionType
    except AttributeError as error:
        raise RuntimeError(
            f"vLLM {profile.vllm_version} attention API differs from its "
            "recurrent compatibility profile"
        ) from error
    return attention, attention_type


def resolve_forward_context_binding(
    *,
    forward_context: Any,
    attention: Any,
    layer_name: str,
) -> tuple[Any, Any]:
    """Resolve vLLM 0.11 metadata and KV cache through ``ForwardContext``.

    vLLM 0.11 stores one cache tensor per virtual engine on each Attention
    instance.  Recurrent P0 is PP=1, but the index is still read from the
    active context and validated rather than assumed to be zero.
    """

    metadata_by_layer = getattr(forward_context, "attn_metadata", None)
    if metadata_by_layer is None:
        metadata = None
    elif isinstance(metadata_by_layer, Mapping):
        if layer_name not in metadata_by_layer:
            raise RuntimeError(
                f"vLLM ForwardContext has no metadata for {layer_name!r}"
            )
        metadata = metadata_by_layer[layer_name]
    else:
        raise RuntimeError(
            "vLLM 0.11 recurrent P0 requires per-layer attention metadata"
        )

    virtual_engine = getattr(forward_context, "virtual_engine", None)
    caches = getattr(attention, "kv_cache", None)
    if (
        isinstance(virtual_engine, bool)
        or not isinstance(virtual_engine, int)
        or virtual_engine < 0
        or not isinstance(caches, Sequence)
        or virtual_engine >= len(caches)
    ):
        raise RuntimeError("vLLM ForwardContext KV cache binding is invalid")
    return metadata, caches[virtual_engine]


def vllm_0_11_dummy_forward_active() -> bool:
    """Return whether the current call is inside vLLM 0.11's `_dummy_run`.

    vLLM 0.11 binds real KV cache tensors before its memory-profile dummy
    forward, but deliberately supplies ``attn_metadata=None`` for that call.
    Real V1 request forwards instead supply a per-layer metadata mapping.  The
    explicit scope lets cache rewriting skip only the documented dummy path;
    missing metadata outside this scope remains a fatal contract violation.
    """

    return _VLLM_0_11_DUMMY_FORWARD_ACTIVE.get()


def install_vllm_0_11_dummy_forward_guard(
    profile: VllmRuntimeProfile,
    *,
    runner_class: type[Any] | None = None,
) -> None:
    """Mark the exact vLLM 0.11 GPU runner dummy/profile call scope."""

    if profile.vllm_version != VLLM_0_11_0:
        return
    if runner_class is None:
        module = import_module("vllm.v1.worker.gpu_model_runner")
        runner_class = module.GPUModelRunner
    original = runner_class._dummy_run
    if getattr(original, _VLLM_0_11_DUMMY_GUARD_MARKER, False):
        return

    @wraps(original)
    def guarded_dummy_run(*args: Any, **kwargs: Any) -> Any:
        token = _VLLM_0_11_DUMMY_FORWARD_ACTIVE.set(True)
        try:
            return original(*args, **kwargs)
        finally:
            _VLLM_0_11_DUMMY_FORWARD_ACTIVE.reset(token)

    setattr(guarded_dummy_run, _VLLM_0_11_DUMMY_GUARD_MARKER, True)
    runner_class._dummy_run = guarded_dummy_run


def configure_vllm_0_11_attention_environment(
    profile: VllmRuntimeProfile,
    environment: dict[str, str],
) -> None:
    """Bind vLLM 0.11 to FlashAttention 2 before importing vLLM."""

    if profile.engine_attention_strategy != "environment":
        return
    required = {
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "VLLM_FLASH_ATTN_VERSION": "2",
    }
    for name, value in required.items():
        configured = environment.get(name)
        if configured not in {None, value}:
            raise RuntimeError(
                f"recurrent P0 requires {name}={value}, observed {configured!r}"
            )
        environment[name] = value


def versioned_engine_kwargs(
    profile: VllmRuntimeProfile,
    *,
    attention_backend: str,
    enable_flashinfer_autotune: bool,
) -> dict[str, Any]:
    """Return only attention kwargs accepted by the selected LLM constructor."""

    if attention_backend != "FLASH_ATTN":
        raise ValueError("recurrent P0 requires FLASH_ATTN")
    if enable_flashinfer_autotune:
        raise ValueError("recurrent P0 disables FlashInfer autotune")
    if profile.engine_attention_strategy == "constructor":
        return {
            "attention_backend": attention_backend,
            "enable_flashinfer_autotune": False,
            "attention_config": {"flash_attn_version": 2},
        }
    if profile.engine_attention_strategy == "environment":
        return {}
    raise RuntimeError("unknown recurrent vLLM attention binding strategy")


__all__ = [
    "KV_LAYOUT_BLOCKS_KV",
    "KV_LAYOUT_KV_BLOCKS",
    "SUPPORTED_VLLM_VERSIONS",
    "VLLM_0_11_0",
    "VLLM_0_25_0",
    "VllmRuntimeProfile",
    "configure_vllm_0_11_attention_environment",
    "install_vllm_0_11_dummy_forward_guard",
    "installed_runtime_profile",
    "require_runtime_profile",
    "resolve_attention_symbols",
    "resolve_forward_context_binding",
    "vllm_0_11_dummy_forward_active",
    "versioned_engine_kwargs",
]
