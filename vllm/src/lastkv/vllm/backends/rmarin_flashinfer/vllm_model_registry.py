"""Explicit, lazy vLLM model registration for RMarin exports."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from .vllm_model import (
    RECURRENT_ARCHITECTURE,
    RMARIN_MODEL_IMPORT,
)
from .runtime_contract import RMarinBackendContractError


def register_rmarin_vllm_model(
    *,
    importer: Callable[[str], Any] = import_module,
) -> None:
    """Register the RMarin model without importing its GPU implementation."""

    transformers = importer("transformers")
    vllm = importer("vllm")
    config_module = importer(
        "lastkv.vllm.backends.recurrent_vllm.transformers_config"
    )
    transformers.AutoConfig.register(
        config_module.RecurrentMarinConfig.model_type,
        config_module.RecurrentMarinConfig,
        exist_ok=True,
    )
    vllm.ModelRegistry.register_model(
        RECURRENT_ARCHITECTURE,
        RMARIN_MODEL_IMPORT,
    )
    registered = getattr(vllm.ModelRegistry, "models", {}).get(
        RECURRENT_ARCHITECTURE
    )
    module_name, separator, class_name = RMARIN_MODEL_IMPORT.partition(":")
    if (
        not separator
        or getattr(registered, "module_name", None) != module_name
        or getattr(registered, "class_name", None) != class_name
    ):
        raise RMarinBackendContractError(
            "vLLM ModelRegistry did not retain the RMarin composite model owner"
        )


__all__ = ["register_rmarin_vllm_model"]
