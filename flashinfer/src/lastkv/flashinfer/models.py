"""Llama-shaped decoder (Transformer control) and LastKV decoder on FlashInfer kernels.

Module shape follows R-KV/FlashInfer ``rkv/models.py``: fused qkv / gate_up
GEMMs, FlashInfer ``rmsnorm`` / ``fused_add_rmsnorm`` / ``silu_and_mul`` /
``apply_rope_with_cos_sin_cache_inplace``. Attention is injected by the engine
as a per-layer callable ``attention_fn(layer_idx, q, k, v) -> o`` (all
``[num_tokens, heads, head_dim]``), so one forward serves prefill and decode.

LastKV differences versus the Llama control (``lastkv/llm/modeling.py``):

* q/k RMSNorm is applied AFTER RoPE (the R-KV Qwen3 path does norm first);
* every layer carries ``prev_chunk_alphas`` ``[num_kv_heads]`` (blend logits
  for the previous chunk) and ``deep_v_scale`` ``[num_kv_heads]`` (per-head
  scale of the shared final-layer V). Both are stored in the checkpoint dtype
  (bf16) and consumed by the engine, not by the forward pass here.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from flashinfer.activation import silu_and_mul
from flashinfer.norm import fused_add_rmsnorm, rmsnorm
from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace

from .config import ModelConfig

AttentionFn = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _llama3_scaled_inv_freq(inv_freq: torch.Tensor, scaling: dict) -> torch.Tensor:
    factor = float(scaling["factor"])
    low_freq_factor = float(scaling["low_freq_factor"])
    high_freq_factor = float(scaling["high_freq_factor"])
    original_max_pos = float(scaling["original_max_position_embeddings"])
    low_freq_wavelen = original_max_pos / low_freq_factor
    high_freq_wavelen = original_max_pos / high_freq_factor
    wavelen = 2 * torch.pi / inv_freq
    smooth = (original_max_pos / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * inv_freq / factor + smooth * inv_freq
    return torch.where(
        wavelen > low_freq_wavelen,
        inv_freq / factor,
        torch.where(wavelen < high_freq_wavelen, inv_freq, smoothed),
    )


def build_cos_sin_cache(config: ModelConfig, max_position: int, device: torch.device) -> torch.Tensor:
    """FlashInfer ``cos_sin_cache``: fp32 ``[max_position, head_dim]``, cos || sin."""
    head_dim = config.head_dim
    inv_freq = 1.0 / config.rope_theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim
    )
    scaling = config.rope_scaling or {}
    rope_type = scaling.get("rope_type", "default")
    if rope_type == "llama3":
        inv_freq = _llama3_scaled_inv_freq(inv_freq, scaling)
    elif rope_type != "default":
        raise NotImplementedError(f"rope_type {rope_type!r} not supported")
    positions = torch.arange(max_position, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, device, dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float, device, dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(size, device=device, dtype=dtype), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return rmsnorm(x, self.weight, self.eps)
        # per-head norm over [tokens, heads, head_dim]
        return rmsnorm(x.reshape(-1, x.shape[-1]), self.weight, self.eps).view_as(x)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, device, dtype) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.family = config.family
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_q_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.use_rope = not (config.family == "lastkv" and config.use_nope)
        self.qkv_proj = Linear(
            config.hidden_size, self.q_size + 2 * self.kv_size, config.attention_bias, device, dtype
        )
        self.o_proj = Linear(self.q_size, config.hidden_size, config.attention_bias, device, dtype)
        if config.family == "lastkv" and config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps, device, dtype)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps, device, dtype)
        else:
            self.q_norm = None
            self.k_norm = None
        if config.family == "lastkv":
            # Consumed by the engine (blend + shared-V scale); kept in the
            # checkpoint dtype so sigmoid()/scaling round like the HF model.
            self.prev_chunk_alphas = nn.Parameter(
                torch.empty(self.num_kv_heads, device=device, dtype=dtype), requires_grad=False
            )
            if config.deep_kv_rescale:
                self.deep_v_scale = nn.Parameter(
                    torch.empty(self.num_kv_heads, device=device, dtype=dtype), requires_grad=False
                )
            else:
                self.register_parameter("deep_v_scale", None)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        attention_fn: AttentionFn,
    ) -> torch.Tensor:
        num_tokens = hidden.shape[0]
        qkv = self.qkv_proj(hidden)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = q.reshape(num_tokens, self.num_q_heads, self.head_dim).contiguous()
        k = k.reshape(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        v = v.reshape(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        if self.use_rope:
            apply_rope_with_cos_sin_cache_inplace(positions, q, k, self.head_dim, cos_sin_cache, is_neox=True)
        if self.q_norm is not None:
            # LastKV: normalize AFTER RoPE (modeling.py: apply_rotary_pos_emb, then q_norm/k_norm).
            q = self.q_norm(q)
            k = self.k_norm(k)
        o = attention_fn(self.layer_idx, q, k, v)
        return self.o_proj(o.view(num_tokens, -1))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig, device, dtype) -> None:
        super().__init__()
        self.gate_up_proj = Linear(config.hidden_size, 2 * config.intermediate_size, config.mlp_bias, device, dtype)
        self.down_proj = Linear(config.intermediate_size, config.hidden_size, config.mlp_bias, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu_and_mul(self.gate_up_proj(x)))


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, device, dtype) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx, device, dtype)
        self.mlp = MLP(config, device, dtype)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, device, dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, device, dtype)

    def forward(self, hidden, residual, positions, cos_sin_cache, attention_fn):
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            fused_add_rmsnorm(hidden, residual, self.input_layernorm.weight, self.input_layernorm.eps)
        hidden = self.self_attn(hidden, positions, cos_sin_cache, attention_fn)
        fused_add_rmsnorm(hidden, residual, self.post_attention_layernorm.weight, self.post_attention_layernorm.eps)
        hidden = self.mlp(hidden)
        return hidden, residual


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device, dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), requires_grad=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


class Model(nn.Module):
    def __init__(self, config: ModelConfig, max_position: int, device, dtype) -> None:
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, device, dtype)
        self.layers = nn.ModuleList(DecoderLayer(config, i, device, dtype) for i in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device, dtype)
        self.register_buffer("cos_sin_cache", build_cos_sin_cache(config, max_position, device), persistent=False)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, attention_fn: AttentionFn) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, positions, self.cos_sin_cache, attention_fn)
        fused_add_rmsnorm(hidden, residual, self.norm.weight, self.norm.eps)
        return hidden


class CausalLM(nn.Module):
    """Decoder-only causal LM (Llama control or LastKV) over a ``ModelConfig``."""

    def __init__(self, config: ModelConfig, max_position: int, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.model = Model(config, max_position, device, dtype)
        self.lm_head = None if config.tie_word_embeddings else Linear(
            config.hidden_size, config.vocab_size, False, device, dtype
        )
        packed: dict[str, tuple[str, int]] = {}
        for i, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            attn_base = f"model.layers.{i}.self_attn."
            mlp_base = f"model.layers.{i}.mlp."
            for suffix in (("weight", "bias") if config.attention_bias else ("weight",)):
                target = f"{attn_base}qkv_proj.{suffix}"
                packed[f"{attn_base}q_proj.{suffix}"] = (target, 0)
                packed[f"{attn_base}k_proj.{suffix}"] = (target, attn.q_size)
                packed[f"{attn_base}v_proj.{suffix}"] = (target, attn.q_size + attn.kv_size)
            for suffix in (("weight", "bias") if config.mlp_bias else ("weight",)):
                target = f"{mlp_base}gate_up_proj.{suffix}"
                packed[f"{mlp_base}gate_proj.{suffix}"] = (target, 0)
                packed[f"{mlp_base}up_proj.{suffix}"] = (target, config.intermediate_size)
        self.packed_map = packed

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, attention_fn: AttentionFn) -> torch.Tensor:
        return self.model(input_ids, positions, attention_fn)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.model.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(hidden, weight)

    # LastKV recurrence parameters stacked over layers (engine helpers).
    def stacked_alphas(self) -> torch.Tensor:
        return torch.stack([layer.self_attn.prev_chunk_alphas for layer in self.model.layers])

    def stacked_deep_v_scale(self) -> torch.Tensor | None:
        first = self.model.layers[0].self_attn.deep_v_scale
        if first is None:
            return None
        return torch.stack([layer.self_attn.deep_v_scale for layer in self.model.layers])
