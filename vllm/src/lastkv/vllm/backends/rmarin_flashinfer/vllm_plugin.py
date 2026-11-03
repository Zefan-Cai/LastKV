"""vLLM general-plugin entry point for the RMarin custom attention backend."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
import os
from typing import Any

from .vllm_backend import (
    BACKEND_CLASS_PATH,
    RMarinVllmCompatibilityError,
    VllmRuntimeApiEvidence,
    validate_vllm_runtime_api,
)
from .vllm_model_registry import register_rmarin_vllm_model


def register_rmarin_attention_backend(
    *,
    importer: Callable[[str], Any] = import_module,
    version_getter: Callable[[str], str] | None = None,
) -> VllmRuntimeApiEvidence:
    """Register ``AttentionBackendEnum.CUSTOM`` through vLLM's public API.

    The function is idempotent for this exact class path and rejects a
    different CUSTOM backend.  It is suitable for a ``vllm.general_plugins``
    entry point; packaging wiring is kept separate from the runtime module.
    """

    evidence = validate_vllm_runtime_api(
        importer=importer,
        version_getter=version_getter,
    )
    registry = importer("vllm.v1.attention.backends.registry")
    custom = registry.AttentionBackendEnum.CUSTOM

    is_overridden = getattr(custom, "is_overridden", None)
    get_path = getattr(custom, "get_path", None)
    if callable(is_overridden) and is_overridden():
        if not callable(get_path):
            raise RMarinVllmCompatibilityError(
                "vLLM CUSTOM backend is already registered but its class path "
                "cannot be inspected"
            )
        observed = get_path()
        if observed != BACKEND_CLASS_PATH:
            raise RMarinVllmCompatibilityError(
                "vLLM CUSTOM attention backend is already registered as "
                f"{observed!r}; RMarin requires {BACKEND_CLASS_PATH!r}"
            )
        return evidence

    registry.register_backend(
        custom,
        BACKEND_CLASS_PATH,
        is_mamba=False,
    )
    if callable(get_path):
        observed = get_path()
        if observed != BACKEND_CLASS_PATH:
            raise RMarinVllmCompatibilityError(
                "vLLM did not retain the registered RMarin backend class path"
            )
    return evidence


def vllm_general_plugin() -> None:
    """Register both the custom attention backend and lazy RMarin model.

    The package intentionally exposes both the established recurrent-vLLM
    compatibility model and this independent RMarin FlashInfer model under the
    prepared export's architecture name.  vLLM's registry permits silent
    replacement, so selection must be explicit and exclusive.
    """

    configured = os.environ.get("VLLM_PLUGINS")
    selected = (
        {item.strip() for item in configured.split(",") if item.strip()}
        if configured is not None
        else set()
    )
    if selected != {"lastkv"}:
        raise RMarinVllmCompatibilityError(
            "RMarin FlashInfer requires the exclusive plugin selection "
            "VLLM_PLUGINS=lastkv; observed "
            f"{configured!r}. This prevents silent ModelRegistry ownership "
            "replacement by the legacy recurrent_marin plugin."
        )
    register_rmarin_attention_backend()
    # ModelRegistry receives only an import string.  This does not import
    # torch, the model implementation, or initialize CUDA in the plugin
    # loader/process which inspects the prepared export's architecture.
    register_rmarin_vllm_model()


__all__ = [
    "register_rmarin_attention_backend",
    "vllm_general_plugin",
]
