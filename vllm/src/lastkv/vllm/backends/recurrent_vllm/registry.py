"""Lazy out-of-tree model registration for vLLM."""

from __future__ import annotations

from .contract import RECURRENT_ARCHITECTURE


MODEL_IMPORT = (
    "lastkv.vllm.backends.recurrent_vllm.model:"
    "RecurrentMarinAKV254ReuseKVForCausalLM"
)


def assert_recurrent_model_owner() -> None:
    """Fail if another general plugin replaced the legacy model owner."""

    from vllm import ModelRegistry

    registered = getattr(ModelRegistry, "models", {}).get(RECURRENT_ARCHITECTURE)
    module_name, separator, class_name = MODEL_IMPORT.partition(":")
    if (
        not separator
        or getattr(registered, "module_name", None) != module_name
        or getattr(registered, "class_name", None) != class_name
    ):
        raise RuntimeError(
            "vLLM ModelRegistry does not retain the recurrent_marin model "
            "owner; set VLLM_PLUGINS=recurrent_marin exclusively"
        )


def register_recurrent_model() -> None:
    from transformers import AutoConfig
    from vllm import ModelRegistry

    from .transformers_config import RecurrentMarinConfig

    # Prepared exports contain the training model_type, which is intentionally
    # not part of upstream Transformers.  Register only its config schema; the
    # model implementation itself stays native vLLM via ModelRegistry below.
    AutoConfig.register(
        RecurrentMarinConfig.model_type,
        RecurrentMarinConfig,
        exist_ok=True,
    )
    ModelRegistry.register_model(RECURRENT_ARCHITECTURE, MODEL_IMPORT)
    assert_recurrent_model_owner()


__all__ = [
    "MODEL_IMPORT",
    "assert_recurrent_model_owner",
    "register_recurrent_model",
]
