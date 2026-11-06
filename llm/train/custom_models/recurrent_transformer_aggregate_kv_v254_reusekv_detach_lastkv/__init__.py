from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_recurrent_transformer import RecurrentTransformerConfig
from .modeling_recurrent_transformer import RecurrentTransformerForCausalLM, RecurrentTransformerModel

AutoConfig.register(RecurrentTransformerConfig.model_type, RecurrentTransformerConfig, exist_ok=True)
AutoModel.register(RecurrentTransformerConfig, RecurrentTransformerModel, exist_ok=True)
AutoModelForCausalLM.register(RecurrentTransformerConfig, RecurrentTransformerForCausalLM, exist_ok=True)


RecurrentTransformerAggregateKVV254ReuseKVDetachLastKVConfig = RecurrentTransformerConfig
RecurrentTransformerAggregateKVV254ReuseKVDetachLastKVModel = RecurrentTransformerModel
RecurrentTransformerAggregateKVV254ReuseKVDetachLastKVForCausalLM = RecurrentTransformerForCausalLM

__all__ = [
    "RecurrentTransformerConfig",
    "RecurrentTransformerForCausalLM",
    "RecurrentTransformerModel",
    "RecurrentTransformerAggregateKVV254ReuseKVDetachLastKVConfig",
    "RecurrentTransformerAggregateKVV254ReuseKVDetachLastKVModel",
    "RecurrentTransformerAggregateKVV254ReuseKVDetachLastKVForCausalLM",
]
