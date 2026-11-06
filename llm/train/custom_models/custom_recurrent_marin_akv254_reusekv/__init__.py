from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_custom_recurrent_marin_akv254_reusekv import (
    CustomRecurrentMarinAKV254ReuseKVConfig,
)
from .modeling_custom_recurrent_marin_akv254_reusekv import (
    CustomRecurrentMarinAKV254ReuseKVForCausalLM,
    CustomRecurrentMarinAKV254ReuseKVModel,
)

AutoConfig.register(
    CustomRecurrentMarinAKV254ReuseKVConfig.model_type,
    CustomRecurrentMarinAKV254ReuseKVConfig,
    exist_ok=True,
)
AutoModel.register(
    CustomRecurrentMarinAKV254ReuseKVConfig,
    CustomRecurrentMarinAKV254ReuseKVModel,
    exist_ok=True,
)
AutoModelForCausalLM.register(
    CustomRecurrentMarinAKV254ReuseKVConfig,
    CustomRecurrentMarinAKV254ReuseKVForCausalLM,
    exist_ok=True,
)

__all__ = [
    "CustomRecurrentMarinAKV254ReuseKVConfig",
    "CustomRecurrentMarinAKV254ReuseKVForCausalLM",
    "CustomRecurrentMarinAKV254ReuseKVModel",
]
