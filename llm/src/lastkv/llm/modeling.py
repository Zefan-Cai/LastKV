"""LastKV inference with one shared final-layer history.

Adapted from Zefan-Cai/llm at 3d914d3217c2f3ea1513852e621c146099ece394
and 580a6c2a46b68a0312d5e7df8ae1d30f8734c812. The source framework is
fla-org/flame, Copyright (c) 2023-2025 Songlin Yang, Yu Zhang (MIT).
Llama components are provided by Hugging Face Transformers (Apache-2.0).

The training sources recompute the prefix for generation. This implementation
retains the same chunk recurrence while caching completed chunks and the
per-layer current/previous chunk. No training kernels or optimizer code is used.
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from .configuration import CustomRecurrentMarinAKV254ReuseKVConfig


def _append(previous, current):
    if previous is None:
        return current
    return tuple(torch.cat((old, new), dim=2) for old, new in zip(previous, current))


@dataclass
class LastKVCache:
    """Completed shared history plus bounded, per-layer chunk state.

    Pass only new tokens to ``forward`` with this cache. Cache boundaries are
    measured from the start of the original prompt, not from individual calls.
    """

    num_layers: int
    chunk_size: int
    seen_tokens: int = 0
    deep: tuple | None = None
    previous: list = field(default_factory=list)
    current: list = field(default_factory=list)

    def __post_init__(self):
        self.previous = [None] * self.num_layers
        self.current = [None] * self.num_layers

    def get_seq_length(self, layer_idx=0):
        return self.seen_tokens

    def reorder_cache(self, beam_idx):
        def select(kv):
            if kv is None:
                return None
            return tuple(t.index_select(0, beam_idx.to(t.device)) for t in kv)

        self.deep = select(self.deep)
        self.previous = [select(kv) for kv in self.previous]
        self.current = [select(kv) for kv in self.current]


class LastKVAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.prev_chunk_alphas = nn.Parameter(torch.zeros(self.num_key_value_heads))
        if config.deep_kv_rescale:
            initial = config.deep_v_scale_init
            value = initial[layer_idx] if isinstance(initial, (list, tuple)) else initial
            values = value if isinstance(value, (list, tuple)) else [value] * self.num_key_value_heads
            if len(values) != self.num_key_value_heads:
                raise ValueError("deep_v_scale_init must match the number of KV heads")
            self.deep_v_scale = nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else nn.Identity()

    def _scale(self, value):
        if not self.config.deep_kv_rescale:
            return value
        return value * self.deep_v_scale.to(value.dtype).view(1, -1, 1, 1)

    def forward(self, hidden_states, position_embeddings, cache):
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # The published implementation normalizes Q/K AFTER applying RoPE.
        if not self.config.use_nope:
            query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        query, key = self.q_norm(query), self.k_norm(key)
        current = _append(cache.current[self.layer_idx], (key, value))
        cache.current[self.layer_idx] = current
        keys, values = [], []
        if cache.deep is not None:
            deep_k, deep_v = cache.deep
            old_length = deep_k.shape[2] - self.config.chunk_size
            if old_length:
                keys.append(deep_k[:, :, :old_length])
                values.append(self._scale(deep_v[:, :, :old_length]))
            previous_k, previous_v = cache.previous[self.layer_idx]
            alpha = self.prev_chunk_alphas.sigmoid().to(key.dtype).view(1, -1, 1, 1)
            keys.append(alpha * deep_k[:, :, old_length:] + (1 - alpha) * previous_k)
            values.append(alpha * self._scale(deep_v[:, :, old_length:]) + (1 - alpha) * previous_v)
        keys.append(current[0])
        values.append(current[1])
        full_k, full_v = torch.cat(keys, dim=2), torch.cat(values, dim=2)
        key_length = full_k.shape[2]
        # PyTorch's non-square is_causal mask is upper-left aligned. Cached
        # decoding needs bottom-right alignment, including partial chunks.
        mask = None
        causal = length == key_length
        if not causal and length > 1:
            query_positions = torch.arange(length, device=query.device) + key_length - length
            mask = torch.arange(key_length, device=query.device)[None, :] <= query_positions[:, None]
        if self.config._attn_implementation == "eager":
            groups = self.num_heads // self.num_key_value_heads
            k = full_k.repeat_interleave(groups, dim=1)
            v = full_v.repeat_interleave(groups, dim=1)
            scores = query @ k.transpose(-1, -2) * self.head_dim ** -0.5
            if causal:
                mask = torch.ones(length, length, device=query.device, dtype=torch.bool).tril()
            if mask is not None:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            attended = probabilities @ v
        else:
            attended = F.scaled_dot_product_attention(
                query, full_k, full_v, attn_mask=mask, is_causal=causal,
                dropout_p=0.0, enable_gqa=self.num_heads != self.num_key_value_heads,
            )
        attended = attended.transpose(1, 2).reshape(batch, length, self.num_heads * self.head_dim)
        return self.o_proj(attended)


class LastKVDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = LastKVAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, cache):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states), position_embeddings, cache)
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class CustomRecurrentMarinAKV254ReuseKVModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(LastKVDecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                position_ids=None, past_key_values=None, use_cache=True):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds")
        if attention_mask is not None and not bool(torch.all(attention_mask != 0)):
            raise ValueError("LastKV supports unpadded sequences; generate each padded prompt separately")
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, length, _ = hidden.shape
        if not length:
            raise ValueError("At least one new token is required")
        if past_key_values is not None and not isinstance(past_key_values, LastKVCache):
            raise TypeError("Use a LastKVCache returned by this model, not a standard Llama cache")
        cache = past_key_values or LastKVCache(len(self.layers), self.config.chunk_size)
        if cache.chunk_size != self.config.chunk_size or cache.num_layers != len(self.layers):
            raise ValueError("Cache architecture does not match this model")
        if position_ids is None:
            position_ids = torch.arange(cache.seen_tokens, cache.seen_tokens + length, device=hidden.device)[None, :]
        elif position_ids.ndim == 1:
            position_ids = position_ids[None, :]
        if position_ids.shape[-1] != length or position_ids.shape[0] not in (1, batch):
            raise ValueError("position_ids must describe the new input tokens")
        outputs = []
        start = 0
        while start < length:
            take = min(length - start, self.config.chunk_size - cache.seen_tokens % self.config.chunk_size)
            chunk = hidden[:, start:start + take]
            positions = self.rotary_emb(chunk, position_ids[:, start:start + take])
            for layer in self.layers:
                chunk = layer(chunk, positions, cache)
            outputs.append(chunk)
            cache.seen_tokens += take
            if cache.seen_tokens % self.config.chunk_size == 0:
                cache.deep = _append(cache.deep, cache.current[-1])
                cache.previous = cache.current
                cache.current = [None] * len(self.layers)
            start += take
        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(torch.cat(outputs, dim=1)),
            past_key_values=cache if use_cache else None,
        )


class CustomRecurrentMarinAKV254ReuseKVForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = CustomRecurrentMarinAKV254ReuseKVConfig
    base_model_prefix = "model"
    _supports_sdpa = True
    _no_split_modules = ["LastKVDecoderLayer"]
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomRecurrentMarinAKV254ReuseKVModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LlamaRMSNorm):
            nn.init.ones_(module.weight)

    @classmethod
    def _supports_default_dynamic_cache(cls):
        return False

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        wants_info = kwargs.pop("output_loading_info", False)
        model, info = super().from_pretrained(*args, output_loading_info=True, **kwargs)
        problems = {key: info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
        if problems:
            raise ValueError(f"Checkpoint is not a strict LastKV model match: {problems}")
        return (model, info) if wants_info else model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, inputs_embeds=None,
                                      use_cache=True, position_ids=None, **kwargs):
        if past_key_values is not None:
            seen = past_key_values.get_seq_length()
            input_ids = input_ids[:, seen:] if input_ids.shape[1] > seen else input_ids[:, -1:]
            inputs_embeds = None
            if position_ids is not None:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        inputs = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None else {"input_ids": input_ids}
        return {**inputs, "past_key_values": past_key_values, "attention_mask": attention_mask,
                "position_ids": position_ids, "use_cache": use_cache, "logits_to_keep": 1}

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                return_dict=None, logits_to_keep=0, labels=None,
                output_attentions=False, output_hidden_states=False, **kwargs):
        if labels is not None or output_attentions or output_hidden_states:
            raise ValueError("This release provides inference logits and a KV cache only")
        if kwargs.get("cu_seqlens") is not None:
            raise ValueError("Packed-document inputs are unsupported; process each document separately")
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        use_cache = self.config.use_cache if use_cache is None else use_cache
        outputs = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds,
                             attention_mask=attention_mask, position_ids=position_ids,
                             past_key_values=past_key_values, use_cache=use_cache)
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, indices])
        if return_dict is False:
            return (logits, outputs.past_key_values)
        return CausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values)
