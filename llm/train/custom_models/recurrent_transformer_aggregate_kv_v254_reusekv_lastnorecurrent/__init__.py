from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_recurrent_transformer import RecurrentTransformerConfig
from .modeling_recurrent_transformer import RecurrentTransformerForCausalLM, RecurrentTransformerModel

AutoConfig.register(RecurrentTransformerConfig.model_type, RecurrentTransformerConfig, exist_ok=True)
AutoModel.register(RecurrentTransformerConfig, RecurrentTransformerModel, exist_ok=True)
AutoModelForCausalLM.register(RecurrentTransformerConfig, RecurrentTransformerForCausalLM, exist_ok=True)


RecurrentTransformerAggregateKVV254ReuseKVLastNoRecurrentConfig = RecurrentTransformerConfig
RecurrentTransformerAggregateKVV254ReuseKVLastNoRecurrentModel = RecurrentTransformerModel
RecurrentTransformerAggregateKVV254ReuseKVLastNoRecurrentForCausalLM = RecurrentTransformerForCausalLM

__all__ = [
    "RecurrentTransformerConfig",
    "RecurrentTransformerForCausalLM",
    "RecurrentTransformerModel",
    "RecurrentTransformerAggregateKVV254ReuseKVLastNoRecurrentConfig",
    "RecurrentTransformerAggregateKVV254ReuseKVLastNoRecurrentModel",
    "RecurrentTransformerAggregateKVV254ReuseKVLastNoRecurrentForCausalLM",
]
