from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .config_custom_marin import CustomMarinConfig
from .modeling_custom_marin import CustomMarinForCausalLM, CustomMarinModel

AutoConfig.register(CustomMarinConfig.model_type, CustomMarinConfig, exist_ok=True)
AutoModel.register(CustomMarinConfig, CustomMarinModel, exist_ok=True)
AutoModelForCausalLM.register(CustomMarinConfig, CustomMarinForCausalLM, exist_ok=True)

__all__ = ["CustomMarinConfig", "CustomMarinForCausalLM", "CustomMarinModel"]
