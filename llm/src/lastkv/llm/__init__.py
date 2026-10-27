"""Explicitly import this module to enable LastKV language-model loading."""

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .configuration import CustomRecurrentMarinAKV254ReuseKVConfig
from .modeling import CustomRecurrentMarinAKV254ReuseKVForCausalLM, LastKVCache

AutoConfig.register(CustomRecurrentMarinAKV254ReuseKVConfig.model_type, CustomRecurrentMarinAKV254ReuseKVConfig, exist_ok=True)
AutoModelForCausalLM.register(CustomRecurrentMarinAKV254ReuseKVConfig, CustomRecurrentMarinAKV254ReuseKVForCausalLM, exist_ok=True)


def load_model(model_name_or_path="ZefanCai/LastKV-LLM-8B-Base", *, revision=None,
               device="cpu", dtype=torch.float32, attention_backend="sdpa"):
    """Strictly load the original HF tensor names and enable cached inference."""
    if attention_backend not in ("sdpa", "eager"):
        raise ValueError("attention_backend must be 'sdpa' or 'eager'")
    config = CustomRecurrentMarinAKV254ReuseKVConfig.from_pretrained(model_name_or_path, revision=revision)
    config._attn_implementation = attention_backend
    model = CustomRecurrentMarinAKV254ReuseKVForCausalLM.from_pretrained(
        model_name_or_path, revision=revision, config=config, dtype=dtype,
        attn_implementation=attention_backend, use_safetensors=True,
    )
    model.config.use_cache = True
    model.generation_config.use_cache = True
    return model.to(device).eval()


__all__ = ["CustomRecurrentMarinAKV254ReuseKVConfig", "CustomRecurrentMarinAKV254ReuseKVForCausalLM", "LastKVCache", "load_model"]
