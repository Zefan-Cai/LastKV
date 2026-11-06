from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_recurrent_transformer import RecurrentTransformerConfig
from .modeling_recurrent_transformer import RecurrentTransformerForCausalLM, RecurrentTransformerModel

AutoConfig.register(RecurrentTransformerConfig.model_type, RecurrentTransformerConfig, exist_ok=True)
AutoModel.register(RecurrentTransformerConfig, RecurrentTransformerModel, exist_ok=True)
AutoModelForCausalLM.register(RecurrentTransformerConfig, RecurrentTransformerForCausalLM, exist_ok=True)


RecurrentTransformerAggregateKVV254ShareKVConfig = RecurrentTransformerConfig
RecurrentTransformerAggregateKVV254ShareKVModel = RecurrentTransformerModel
RecurrentTransformerAggregateKVV254ShareKVForCausalLM = RecurrentTransformerForCausalLM

__all__ = [
    "RecurrentTransformerConfig",
    "RecurrentTransformerForCausalLM",
    "RecurrentTransformerModel",
    "RecurrentTransformerAggregateKVV254ShareKVConfig",
    "RecurrentTransformerAggregateKVV254ShareKVModel",
    "RecurrentTransformerAggregateKVV254ShareKVForCausalLM",
]
