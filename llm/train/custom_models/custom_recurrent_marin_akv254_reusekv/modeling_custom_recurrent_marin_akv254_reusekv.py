"""Marin-8B (Llama-3.1) with AKV254 reuse-KV chunked-recurrent attention.

Weight layout is 1:1 with the marin-8b-base-phoenix / HF Llama checkpoint
(embed_tokens, layers.N.self_attn.{q,k,v,o}_proj, layers.N.mlp, norms,
lm_head); the only new parameter is `prev_chunk_alphas` per attention (init 0,
tolerated as missing on load).

The chunk-loop scheduling follows
recurrent_transformer_gated_attention_aggregate_kv_v254_reusekv/
modeling_recurrent_transformer.py:

- Every layer-chunk call is wrapped in non-reentrant
  ``torch.utils.checkpoint`` during training. The backward recompute replays
  the chunk, so the KV appended into the preallocated storages (CacheAppend
  writes are idempotent per-slice) is reproduced exactly — this is what keeps
  the skip-layer (v_gap-mapped) KV consistent under recomputation.
- Only the first chunk goes through ``Module.__call__`` so FSDP2's
  pre/post-forward hooks run (unshard). Later chunks call ``layer.forward``
  directly, skipping the hooks; this requires
  ``fsdp_reshard_after_forward = "never"`` so params are still unsharded.
  Run with flame ``activation_checkpoint.mode = "none"`` — the model already
  checkpoints internally; stacking flame's AC wrapper would recompute twice.
"""

from __future__ import annotations

import os

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..chunked_ce import chunked_linear_cross_entropy
from ..liger_flce import liger_linear_cross_entropy
from ..qk_norm import eager_rmsnorm
from transformers.models.llama.modeling_llama import (
    GenerationMixin,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm as _LlamaRMSNorm,
    LlamaRotaryEmbedding as _LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.utils.generic import check_model_inputs

from ..llama_rope_utils import reset_llama_rotary_embedding
from .configuration_custom_recurrent_marin_akv254_reusekv import (
    CustomRecurrentMarinAKV254ReuseKVConfig,
)

try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        flash_attn_func = None


class LlamaRMSNorm(_LlamaRMSNorm):
    def reset_parameters(self):
        nn.init.ones_(self.weight)


class DeepVNorm(_LlamaRMSNorm):
    """qk-norm treatment for the incoming deep V: RMSNorm over head_dim with a
    learnable [head_dim] weight, initialized to the receiving layer's measured
    RMS (constant fill) so deep V arrives in-distribution at step 0 while the
    per-channel weight gives diagonal-adapter freedom."""

    def __init__(self, hidden_size, eps, fill):
        super().__init__(hidden_size, eps=eps)
        self._fill = float(fill)
        with torch.no_grad():
            self.weight.fill_(self._fill)

    def reset_parameters(self):
        nn.init.constant_(self.weight, self._fill)


class LlamaRotaryEmbedding(_LlamaRotaryEmbedding):
    def reset_parameters(self):
        reset_llama_rotary_embedding(self)


_MODE_ROUTED_FUSED_RMS_NORM = None


def _mode_routed_fused_rms_norm_cls():
    """fla fused Triton RMSNorm for TRAINING; eval routes to eager_rmsnorm with
    the same weight — the fla kernel hangs deterministically in the 448-rank
    benchmark-eval path (see qk_norm.py). Weight name stays 'weight', so
    checkpoints are byte-compatible with LlamaRMSNorm. Mirrors the custom_marin
    arm exactly."""
    global _MODE_ROUTED_FUSED_RMS_NORM
    if _MODE_ROUTED_FUSED_RMS_NORM is None:
        from fla.modules import RMSNorm as FlaRMSNorm

        class ModeRoutedFusedRMSNorm(FlaRMSNorm):
            def forward(self, x):
                if self.training:
                    return super().forward(x)
                return eager_rmsnorm(x, self.weight, self.eps)

        _MODE_ROUTED_FUSED_RMS_NORM = ModeRoutedFusedRMSNorm
    return _MODE_ROUTED_FUSED_RMS_NORM


def build_marin_rms_norm(config, hidden_size: int) -> nn.Module:
    # fuse_norm off (default) returns the local LlamaRMSNorm subclass so
    # reset_parameters + _init_weights isinstance still apply unchanged.
    if getattr(config, "fuse_norm", False):
        return _mode_routed_fused_rms_norm_cls()(hidden_size, eps=config.rms_norm_eps)
    return LlamaRMSNorm(hidden_size, eps=config.rms_norm_eps)


class CacheAppend(torch.autograd.Function):
    """Append along sequence dim in preallocated [B, H, L, D] storage while
    preserving gradients to the active prefix and the appended tensor."""

    @staticmethod
    def forward(ctx, storage, active_cache, x):
        bs = active_cache.size(0)
        start = active_cache.size(2)
        end = start + x.size(2)
        storage.data[:bs, :, start:end] = x.to(storage.dtype)
        ctx.bs = bs
        ctx.start = start
        ctx.end = end
        return storage[:bs, :, :end]

    @staticmethod
    def backward(ctx, grad_output):
        bs = ctx.bs
        start = ctx.start
        end = ctx.end
        return None, grad_output[:bs, :, :start], grad_output[:bs, :, start:end]


cache_append = CacheAppend.apply


def _validate_no_padding_attention_mask(attention_mask: Optional[torch.Tensor]) -> None:
    if attention_mask is None:
        return
    if not bool(torch.all(attention_mask != 0).item()):
        raise ValueError("custom_recurrent_marin_akv254 does not support padded attention_mask.")


class AKV254PastKV:
    def __init__(
        self,
        deep_kv_cache: tuple[torch.Tensor, torch.Tensor],
        prev_local_kv_cache: tuple[torch.Tensor | None, torch.Tensor | None],
        prev_deep_kv_cache: tuple[torch.Tensor | None, torch.Tensor | None],
    ):
        self.deep_kv_cache = deep_kv_cache
        self.prev_local_kv_cache = prev_local_kv_cache
        self.prev_deep_kv_cache = prev_deep_kv_cache


class AKV254MarinAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_nope = getattr(config, "use_nope", False)

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.prev_chunk_alphas = nn.Parameter(torch.zeros(self.num_key_value_heads))

        # deep-KV entry rescale (see config docstring). Created only when
        # enabled so v1 checkpoints/keys are untouched with the flag off.
        self._deep_rescale = bool(getattr(config, "deep_kv_rescale", False))
        self._deep_mode = getattr(config, "deep_rescale_mode", "kv")
        self._deep_frozen = bool(getattr(config, "deep_scale_frozen", False))
        if self._deep_rescale:
            init = getattr(config, "deep_v_scale_init", 1.0)
            entry = init[layer_idx] if isinstance(init, (list, tuple)) else init
            # per-layer entry may be a scalar (broadcast) or a per-kv-head list
            if isinstance(entry, (list, tuple)):
                if len(entry) != self.num_key_value_heads:
                    raise ValueError("per-head deep_v_scale_init entry length != num_key_value_heads")
                vals = [float(x) for x in entry]
            else:
                vals = [float(entry)] * self.num_key_value_heads
            self._deep_v_init_vals = vals
            v0 = sum(vals) / len(vals)   # scalar summary (vnormqk fill)
            self._deep_v_init = v0
            # "kv"/"v": multiplicative scale on deep V (init = RMS ratio).
            # "vnorm": per-token RMSNorm on deep V x this gain (init = abs RMS).
            # frozen: constants only — no learnable knob to collapse (the
            # per-layer scalar broadcasts; no tensor state, meta-init safe).
            if self._deep_mode == "vnormqk":
                self.deep_v_norm = DeepVNorm(self.head_dim, config.rms_norm_eps, v0)
            elif not self._deep_frozen:
                self.deep_v_scale = nn.Parameter(torch.tensor(vals, dtype=torch.float32))
                if self._deep_mode == "kv":
                    self.deep_k_scale = nn.Parameter(torch.ones(self.num_key_value_heads))

        self._fla_qk = False
        self._qk_eps = config.rms_norm_eps
        if getattr(config, "use_qk_norm", False):
            _norm_cls = LlamaRMSNorm
            if getattr(config, "use_fla_qk_norm", False):
                from fla.modules import RMSNorm as _norm_cls  # noqa: N813
                self._fla_qk = True
            self.q_norm = _norm_cls(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = _norm_cls(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self._fwd_k: Optional[torch.Tensor] = None
        self._fwd_v: Optional[torch.Tensor] = None
        # Plain tensors (not buffers): FSDP must not shard them and they are
        # recreated on device/dtype moves via _apply (meta-init safe).
        self.k_storage = torch.empty(0, self.num_key_value_heads, 0, self.head_dim)
        self.v_storage = torch.empty(0, self.num_key_value_heads, 0, self.head_dim)

    def _apply(self, fn):
        super()._apply(fn)
        reference = self.q_proj.weight
        self.k_storage = torch.empty(
            0, self.num_key_value_heads, 0, self.head_dim,
            device=reference.device, dtype=reference.dtype,
        )
        self.v_storage = torch.empty(
            0, self.num_key_value_heads, 0, self.head_dim,
            device=reference.device, dtype=reference.dtype,
        )
        return self

    def reset_parameters(self):
        nn.init.zeros_(self.prev_chunk_alphas)
        if self._deep_rescale and self._deep_mode == "vnormqk":
            self.deep_v_norm.reset_parameters()
        elif self._deep_rescale and not self._deep_frozen:
            with torch.no_grad():
                src = torch.tensor(self._deep_v_init_vals, dtype=self.deep_v_scale.dtype)
                try:
                    from torch.distributed.tensor import DTensor, distribute_tensor
                    if isinstance(self.deep_v_scale.data, DTensor):
                        src = distribute_tensor(
                            src.to(self.deep_v_scale.device),
                            self.deep_v_scale.data.device_mesh,
                            self.deep_v_scale.data.placements,
                        )
                except ImportError:
                    pass
                self.deep_v_scale.data.copy_(src)
            if self._deep_mode == "kv":
                nn.init.ones_(self.deep_k_scale)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict.setdefault(prefix + "prev_chunk_alphas", torch.zeros_like(self.prev_chunk_alphas))
        if self._deep_rescale and self._deep_mode == "vnormqk":
            state_dict.setdefault(prefix + "deep_v_norm.weight", torch.full_like(self.deep_v_norm.weight, self.deep_v_norm._fill))
        elif self._deep_rescale and not self._deep_frozen:
            state_dict.setdefault(prefix + "deep_v_scale", torch.tensor(self._deep_v_init_vals, dtype=self.deep_v_scale.dtype))
            if self._deep_mode == "kv":
                state_dict.setdefault(prefix + "deep_k_scale", torch.ones_like(self.deep_k_scale))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


    def _norm_q(self, x):
        if self._fla_qk and not self.training and not isinstance(self.q_norm, nn.Identity):
            return eager_rmsnorm(x, self.q_norm.weight, self._qk_eps)
        return self.q_norm(x)

    def _norm_k(self, x):
        if self._fla_qk and not self.training and not isinstance(self.k_norm, nn.Identity):
            return eager_rmsnorm(x, self.k_norm.weight, self._qk_eps)
        return self.k_norm(x)

    def _prev_chunk_alpha(self, dtype: torch.dtype) -> torch.Tensor:
        return self.prev_chunk_alphas.sigmoid().to(dtype=dtype).view(1, self.num_key_value_heads, 1, 1)

    def scale_deep_kv(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rescale INCOMING deep (layer-31) KV into this layer's native range."""
        if not self._deep_rescale:
            return k, v
        if self._deep_mode == "vnormqk":
            return k, self.deep_v_norm(v)
        if self._deep_frozen:
            t = getattr(self, "_deep_v_vec_t", None)
            if t is None or t.device != v.device or t.dtype != v.dtype:
                t = torch.tensor(self._deep_v_init_vals, device=v.device, dtype=v.dtype).view(1, self.num_key_value_heads, 1, 1)
                self._deep_v_vec_t = t
            if self._deep_mode == "vnorm":
                var = v.float().pow(2).mean(-1, keepdim=True)
                v = (v.float() * torch.rsqrt(var + 1e-6)).to(v.dtype) * t
            else:
                v = v * t
            return k, v
        vs = self.deep_v_scale.to(v.dtype).view(1, self.num_key_value_heads, 1, 1)
        if self._deep_mode == "vnorm":
            # qk-norm-style: normalize each head-dim vector of the incoming
            # deep V to unit RMS, then apply the learnable per-head gain.
            var = v.float().pow(2).mean(-1, keepdim=True)
            v = (v.float() * torch.rsqrt(var + 1e-6)).to(v.dtype) * vs
        else:
            v = v * vs
        if self._deep_mode == "kv":
            ks = self.deep_k_scale.to(k.dtype).view(1, self.num_key_value_heads, 1, 1)
            k = k * ks
        return k, v

    def _blend_prev_kv(
        self,
        local_k: torch.Tensor,
        local_v: torch.Tensor,
        source_k: torch.Tensor,
        source_v: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, "_fused_blend", False) and self._deep_mode not in ("vnorm", "vnormqk"):
            # one-kernel fused sigmoid-blend + per-head deep scale (fp32 inside)
            from ..custom_recurrent_marin_akv254_reusekv.fused_blend import fused_blend_scale
            vs = ks = None
            if self._deep_rescale:
                if self._deep_frozen:
                    t = getattr(self, "_deep_v_vec_t", None)
                    if t is None or t.device != source_v.device:
                        t = torch.tensor(self._deep_v_init_vals, device=source_v.device, dtype=torch.float32).view(-1)
                        self._deep_v_vec_t = t
                    vs = t.view(-1)
                else:
                    vs = self.deep_v_scale
                    if self._deep_mode == "kv":
                        ks = self.deep_k_scale
            return fused_blend_scale(source_k, source_v, local_k, local_v, self.prev_chunk_alphas, vs, ks, dtype)
        source_k, source_v = self.scale_deep_kv(source_k, source_v)
        alpha = self._prev_chunk_alpha(dtype)
        return (
            alpha * source_k.to(dtype) + (1 - alpha) * local_k.to(dtype),
            alpha * source_v.to(dtype) + (1 - alpha) * local_v.to(dtype),
        )

    def _shape_q(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        return hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        return hidden_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states = self._shape_q(self.q_proj(hidden_states))
        key_states = self._shape_kv(self.k_proj(hidden_states))
        value_states = self._shape_kv(self.v_proj(hidden_states))

        if not self.use_nope:
            cos, sin = position_embeddings
            if getattr(self, "_fused_rope", False):
                from liger_kernel.transformers.rope import liger_rotary_pos_emb
                query_states, key_states = liger_rotary_pos_emb(query_states, key_states, cos, sin)
            else:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = self._norm_q(query_states)
        key_states = self._norm_k(key_states)
        return query_states, key_states, value_states

    def project_kv_from_normalized(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_states = self._shape_kv(self.k_proj(hidden_states))
        value_states = self._shape_kv(self.v_proj(hidden_states))
        if not self.use_nope:
            cos, sin = position_embeddings
            empty_query = torch.empty_like(key_states)
            _, key_states = apply_rotary_pos_emb(empty_query, key_states, cos, sin)
        return self._norm_k(key_states), value_states

    def _ensure_storage(self, batch_size: int, needed_len: int, device: torch.device, dtype: torch.dtype):
        b, h, l, d = self.k_storage.shape
        batch_size = max(batch_size, b)
        needed_len = max(needed_len, l)
        if b < batch_size or l < needed_len or self.k_storage.device != device or self.k_storage.dtype != dtype:
            self.k_storage = self.k_storage.new_empty(
                batch_size,
                h,
                needed_len,
                d,
                device=device,
                dtype=dtype,
            )
            self.v_storage = self.v_storage.new_empty(
                batch_size,
                h,
                needed_len,
                d,
                device=device,
                dtype=dtype,
            )

    def cache_init(self, batch_size: int, max_len: int, device: torch.device, dtype: torch.dtype):
        self._ensure_storage(batch_size, max_len, device=device, dtype=dtype)
        return self.k_storage[:batch_size, :, :0], self.v_storage[:batch_size, :, :0]

    def append_cache(
        self,
        cache: tuple[torch.Tensor, torch.Tensor],
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return cache_append(self.k_storage, cache[0], new_k), cache_append(self.v_storage, cache[1], new_v)

    def _compose_full_kv(
        self,
        past_kv: AKV254PastKV,
        current_k: torch.Tensor,
        current_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_k, full_v = past_kv.deep_kv_cache
        prev_local_k, prev_local_v = past_kv.prev_local_kv_cache
        prev_deep_k, prev_deep_v = past_kv.prev_deep_kv_cache
        if getattr(self, "_shared_deep_cache", False):
            # Shared-cache mode: the deep prefix holds RAW source-layer KV (one
            # storage for all layers). Apply this layer's rescale at READ time —
            # numerically identical to the write-time scaling (same bf16 ops on
            # the same operands) — and assemble with cat instead of in-storage
            # appends, so the read path never writes shared state (checkpoint
            # recompute becomes pure).
            parts_k = []
            parts_v = []
            if full_k.size(2) > 0:
                sk, sv = self.scale_deep_kv(full_k, full_v)
                parts_k.append(sk)
                parts_v.append(sv)
            if prev_local_k is not None and prev_deep_k is not None:
                blended_k, blended_v = self._blend_prev_kv(
                    prev_local_k,
                    prev_local_v,
                    prev_deep_k,
                    prev_deep_v,
                    current_k.dtype,
                )
                parts_k.append(blended_k)
                parts_v.append(blended_v)
            parts_k.append(current_k)
            parts_v.append(current_v)
            if len(parts_k) == 1:
                return current_k, current_v
            return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)
        if prev_local_k is not None and prev_deep_k is not None:
            blended_k, blended_v = self._blend_prev_kv(
                prev_local_k,
                prev_local_v,
                prev_deep_k,
                prev_deep_v,
                current_k.dtype,
            )
            full_k, full_v = self.append_cache((full_k, full_v), blended_k, blended_v)
        return self.append_cache((full_k, full_v), current_k, current_v)

    @staticmethod
    def _bottom_right_causal_mask(q_len: int, k_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        q_positions = torch.arange(q_len, device=device).view(q_len, 1) + max(k_len - q_len, 0)
        k_positions = torch.arange(k_len, device=device).view(1, k_len)
        invalid = k_positions > q_positions
        mask = torch.zeros((q_len, k_len), device=device, dtype=dtype)
        mask = mask.masked_fill(invalid, torch.finfo(dtype).min)
        return mask.view(1, 1, q_len, k_len)

    @staticmethod
    def _varlen_bottom_right_mask(
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        q_len: int,
        k_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Same mask flash-attn varlen applies: bottom-right causal *within*
        each (q, k) document pair, everything across documents masked out."""
        neg = torch.finfo(dtype).min
        mask = torch.full((q_len, k_len), neg, device=device, dtype=dtype)
        q_off = cu_q.tolist()
        k_off = cu_k.tolist()
        for i in range(len(q_off) - 1):
            qs, qe = q_off[i], q_off[i + 1]
            ks, ke = k_off[i], k_off[i + 1]
            if qe <= qs or ke <= ks:
                continue
            q_pos = torch.arange(qe - qs, device=device).view(-1, 1) + max((ke - ks) - (qe - qs), 0)
            k_pos = torch.arange(ke - ks, device=device).view(1, -1)
            block = torch.zeros((qe - qs, ke - ks), device=device, dtype=dtype)
            mask[qs:qe, ks:ke] = block.masked_fill(k_pos > q_pos, neg)
        return mask.view(1, 1, q_len, k_len)

    def _eager_bottom_right_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        dropout_p: float,
        varlen: Optional[tuple[torch.Tensor, torch.Tensor, int, int]] = None,
    ) -> torch.Tensor:
        if key_states.size(1) != query_states.size(1):
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if varlen is not None:
            cu_q, cu_k, _, _ = varlen
            causal_mask = self._varlen_bottom_right_mask(
                cu_q, cu_k,
                query_states.size(2), key_states.size(2),
                query_states.device, query_states.dtype,
            )
        else:
            causal_mask = self._bottom_right_causal_mask(
                query_states.size(2),
                key_states.size(2),
                query_states.device,
                query_states.dtype,
            )
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
        return torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()

    def _direct_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        dropout_p: float,
        varlen: Optional[tuple[torch.Tensor, torch.Tensor, int, int]] = None,
    ) -> torch.Tensor:
        attn_impl = getattr(self.config, "_attn_implementation", "eager")
        use_flash = (
            flash_attn_func is not None
            and query_states.is_cuda
            and attn_impl in {"flash_attention_2", "flash_attention_3"}
        )
        if not use_flash:
            return self._eager_bottom_right_attention(
                query_states, key_states, value_states, dropout_p, varlen=varlen
            )
        if varlen is not None:
            return self._varlen_attention(query_states, key_states, value_states, dropout_p, varlen)

        q = query_states.transpose(1, 2).contiguous()
        k = key_states.transpose(1, 2).contiguous()
        v = value_states.transpose(1, 2).contiguous()
        # flash kernels only accept fp16/bf16; fp32 activations (e.g. DDP
        # autocast leaves embeddings fp32) get round-tripped through bf16
        orig_dtype = q.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        try:
            attn_output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=self.scaling,
                causal=True,
            )
        except TypeError:
            attn_output = flash_attn_func(q, k, v, causal=True)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        if attn_output.dtype != orig_dtype:
            attn_output = attn_output.to(orig_dtype)
        return attn_output

    def _varlen_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        dropout_p: float,
        varlen: tuple[torch.Tensor, torch.Tensor, int, int],
    ) -> torch.Tensor:
        """Document-isolated attention for a packed sequence.

        The caller has already trimmed the composed KV so that index 0 is the
        first token of the earliest document present in this chunk; cu_q/cu_k
        are offsets into the chunk and into that trimmed key window. flash-attn
        varlen aligns each (q, k) pair bottom-right, which is exactly what a
        chunk of a document attending its own prefix needs.
        """
        # Mirror the module-level preference: FA3 (flash_attn_interface) when it
        # is installed, else FA2. On Hopper the varlen kernel is where most of
        # this model's time goes, so the FA3 path is worth having.
        try:
            from flash_attn_interface import flash_attn_varlen_func
        except ImportError:
            from flash_attn import flash_attn_varlen_func

        cu_q, cu_k, max_q, max_k = varlen
        # [B, H, S, D] -> [S, H, D]; varlen requires batch 1 (packed layout)
        q = query_states.squeeze(0).transpose(0, 1).contiguous()
        k = key_states.squeeze(0).transpose(0, 1).contiguous()
        v = value_states.squeeze(0).transpose(0, 1).contiguous()
        orig_dtype = q.dtype
        if orig_dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        kw = dict(
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=self.scaling,
            causal=True,
        )
        try:
            # FA3 dropped dropout entirely; only ask for it when it is actually
            # wanted, and fall back if the installed kernel rejects the kwarg.
            out = flash_attn_varlen_func(q, k, v, dropout_p=dropout_p, **kw) if dropout_p else \
                  flash_attn_varlen_func(q, k, v, **kw)
        except TypeError:
            out = flash_attn_varlen_func(q, k, v, **kw)
        if isinstance(out, tuple):
            out = out[0]
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out.unsqueeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        akv_past_kv: AKV254PastKV,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        query_states, key_states, value_states = self._project_qkv(hidden_states, position_embeddings)
        self._fwd_k = key_states
        self._fwd_v = value_states

        key_states, value_states = self._compose_full_kv(akv_past_kv, key_states, value_states)

        # Document isolation: drop the part of the composed prefix that belongs
        # to documents which ended before this chunk began. Packing is
        # sequential, so that part is always a contiguous head of the key
        # window and this is a pure slice — it also shrinks the deep-KV read.
        varlen = kwargs.get("akv_varlen")
        if varlen is not None:
            key_trim, cu_q, cu_k, max_q, max_k = varlen
            if key_trim:
                key_states = key_states[:, :, key_trim:, :]
                value_states = value_states[:, :, key_trim:, :]
            varlen = (cu_q, cu_k, max_q, max_k)

        dropout_p = self.attention_dropout if self.training else 0.0
        attn_output = self._direct_attention(
            query_states,
            key_states,
            value_states,
            dropout_p,
            varlen=varlen,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], self.num_heads * self.head_dim)
        return self.o_proj(attn_output), (self._fwd_k, self._fwd_v)


class AKV254MarinDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, attention_cls: type[AKV254MarinAttention]):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = attention_cls(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = build_marin_rms_norm(config, config.hidden_size)
        self.post_attention_layernorm = build_marin_rms_norm(config, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        akv_past_kv: AKV254PastKV,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, curr_kv = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            akv_past_kv=akv_past_kv,
            **kwargs,
        )
        if getattr(self, "_fused_addnorm", False) and self.training:
            # fla fused residual-add + RMSNorm (post-LN); returns (normed, sum)
            from fla.modules import RMSNorm as _FlaRMSNorm
            hidden_states, residual = _FlaRMSNorm.forward(
                self.post_attention_layernorm, hidden_states, residual=residual, prenorm=True
            )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
        if getattr(self, "_fused_swiglu", False):
            from liger_kernel.ops.swiglu import LigerSiLUMulFunction
            hidden_states = self.mlp.down_proj(
                LigerSiLUMulFunction.apply(self.mlp.gate_proj(hidden_states), self.mlp.up_proj(hidden_states))
            )
        else:
            hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, curr_kv

    def project_kv_from_normalized(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.self_attn.project_kv_from_normalized(hidden_states, position_embeddings)


class AKV254MarinPreTrainedModel(LlamaPreTrainedModel):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AKV254MarinDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LlamaRMSNorm):
            module.reset_parameters()
        elif isinstance(module, LlamaRotaryEmbedding):
            # Under flame's meta-device construction, computing inv_freq here
            # would materialize real tensors out of meta ("Cannot copy out of
            # meta tensor"). Skip — the post-to_empty() post_init() rerun and
            # flame's reinit_rope_buffers() recompute it on the real device.
            if module.inv_freq.device.type != "meta":
                module.reset_parameters()
        elif hasattr(module, "reset_parameters"):
            # reference fallback (ref_modeling.py): covers fresh-param modules
            # the isinstance checks miss — fla RMSNorm (use_fla_qk_norm /
            # fuse_norm), DeepVNorm, and the attention module itself (alphas +
            # deep scales) — so HF-style _init_weights materialization is safe
            # even without flame's per-module reset_parameters sweep.
            module.reset_parameters()
        if hasattr(module, "prev_chunk_alphas"):
            nn.init.zeros_(module.prev_chunk_alphas)


class CustomRecurrentMarinAKV254ReuseKVModel(AKV254MarinPreTrainedModel):
    config_class = CustomRecurrentMarinAKV254ReuseKVConfig
    attention_cls: type[AKV254MarinAttention] = AKV254MarinAttention
    akv254_kv_mode = "reusekv"

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        _dvs = getattr(config, "deep_v_scale_init", 1.0)
        if getattr(config, "deep_kv_rescale", False) and isinstance(_dvs, (list, tuple)) \
                and len(_dvs) != config.num_hidden_layers:
            raise ValueError(
                f"deep_v_scale_init has {len(_dvs)} entries but the model has "
                f"{config.num_hidden_layers} layers")
        self.layers = nn.ModuleList(
            [AKV254MarinDecoderLayer(config, layer_idx, self.attention_cls) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = build_marin_rms_norm(config, config.hidden_size)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.chunk_size = getattr(config, "chunk_size", 4096)
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int) or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer.")
        self.v_gap = getattr(config, "v_gap", 95)

        # Perf knob: skip the internal per-layer-chunk checkpoint for these
        # layer indices (memory-for-speed). Numerically identical — the same
        # ops run, their activations are just stored instead of replayed in
        # backward; CacheAppend writes stay idempotent either way. Unset env
        # (production default) = checkpoint everything, exactly as before.
        _no_ckpt = os.environ.get("FLAME_RMARIN_NO_CKPT_LAYERS", "")
        self._no_ckpt_layers = {int(x) for x in _no_ckpt.split(",") if x.strip()}
        # Opt-in (default OFF): one shared raw deep-KV storage for all layers
        # instead of 32 per-layer scaled copies (~-9GB at bs8); layers apply
        # their rescale at read time inside _compose_full_kv.
        self._shared_deep_cache = os.environ.get("FLAME_RMARIN_SHARED_DEEP_CACHE", "") == "1"
        _fb = os.environ.get("FLAME_RMARIN_FUSED_BLEND", "") == "1"
        _fr = os.environ.get("FLAME_RMARIN_FUSED_ROPE", "") == "1"
        _fa = os.environ.get("FLAME_RMARIN_FUSED_ADDNORM", "") == "1"
        _fs = os.environ.get("FLAME_RMARIN_FUSED_SWIGLU", "") == "1"
        if _fa and not getattr(config, "fuse_norm", False):
            raise ValueError("FLAME_RMARIN_FUSED_ADDNORM requires fuse_norm=true (fla RMSNorm)")
        for _layer in self.layers:
            if _fb:
                _layer.self_attn._fused_blend = True
            if _fr:
                _layer.self_attn._fused_rope = True
            if _fa:
                _layer._fused_addnorm = True
            if _fs:
                _layer._fused_swiglu = True
        if self._shared_deep_cache:
            _src = self._v_layer_idx(0)
            if any(self._v_layer_idx(i) != _src for i in range(len(self.layers))):
                raise ValueError("FLAME_RMARIN_SHARED_DEEP_CACHE requires a uniform deep-source layer")
            self._shared_deep_src = _src
            for _layer in self.layers:
                _layer.self_attn._shared_deep_cache = True

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _v_layer_idx(self, layer_idx: int) -> int:
        if self.v_gap is None:
            return layer_idx
        return max(min(layer_idx + int(self.v_gap), len(self.layers) - 1), 0)

    def extra_repr(self) -> str:
        lines = [f"vmap: {i} <- {self._v_layer_idx(i)}" for i in range(len(self.layers))]
        return "\n".join(lines)

    def _init_layer_cache(
        self,
        layer: AKV254MarinDecoderLayer,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return layer.self_attn.cache_init(batch_size, seq_len, device=device, dtype=dtype)

    def _append_deep_cache(
        self,
        layer: AKV254MarinDecoderLayer,
        cache: tuple[torch.Tensor, torch.Tensor],
        new_kv: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_k, new_v = new_kv
        if new_k is None or new_v is None:
            return cache
        if not getattr(self, "_shared_deep_cache", False):
            # shared mode stores RAW source KV once; layers rescale at read
            new_k, new_v = layer.self_attn.scale_deep_kv(new_k, new_v)
        return layer.self_attn.append_cache(cache, new_k, new_v)

    def _next_prev_deep_kv(
        self,
        curr_local_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [curr_local_kv[self._v_layer_idx(i)] for i in range(len(self.layers))]

    def _clear_forward_kv(self):
        for layer in self.layers:
            layer.self_attn._fwd_k = None
            layer.self_attn._fwd_v = None

    def _call_layer(
        self,
        layer: AKV254MarinDecoderLayer,
        layer_idx: int,
        is_first_chunk: bool,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        akv_past_kv: AKV254PastKV,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # First chunk goes through Module.__call__ so FSDP2 pre/post-forward
        # hooks run (unshard); later chunks call `forward` directly to skip
        # them (requires fsdp_reshard_after_forward="never"). During training
        # every call is wrapped in non-reentrant checkpoint: the backward
        # recompute replays the chunk so skip-layer KV stays correct.
        layer_fn = layer if is_first_chunk else layer.forward
        if (
            self.training
            and torch.is_grad_enabled()
            and layer_idx not in self._no_ckpt_layers
        ):
            return checkpoint(
                layer_fn,
                hidden_states,
                position_embeddings,
                akv_past_kv,
                use_reentrant=False,
                **kwargs,
            )
        return layer_fn(hidden_states, position_embeddings, akv_past_kv, **kwargs)

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if past_key_values is not None:
            raise NotImplementedError("custom_recurrent_marin_akv254 does not support external past_key_values.")
        if output_attentions:
            raise NotImplementedError("custom_recurrent_marin_akv254 does not return attentions.")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        _validate_no_padding_attention_mask(attention_mask)

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        elif position_ids.dim() == 1:
            # varlen mode: flame emits fla-style 1-D position_ids that restart
            # at 0 for every document. Everything below slices [B, S].
            position_ids = position_ids.unsqueeze(0)

        cache_dtype = inputs_embeds.dtype
        chunk_size = self.chunk_size
        if self._shared_deep_cache:
            _shared = self._init_layer_cache(
                self.layers[self._shared_deep_src], batch_size, seq_len + chunk_size, inputs_embeds.device, cache_dtype
            )
            deep_caches = [_shared for _ in self.layers]
        else:
            deep_caches = [
                self._init_layer_cache(layer, batch_size, seq_len + chunk_size, inputs_embeds.device, cache_dtype)
                for layer in self.layers
            ]
        prev_local_kv: list[tuple[torch.Tensor | None, torch.Tensor | None]] = [(None, None) for _ in self.layers]
        prev_deep_kv: list[tuple[torch.Tensor | None, torch.Tensor | None]] = [(None, None) for _ in self.layers]

        all_hidden_states = () if output_hidden_states else None
        chunk_outputs = []

        # Packed-sequence (varlen) support. flame hands us cu_seqlens for a
        # batch-1 packed sequence; documents are laid out contiguously, so for
        # any chunk the keys of the documents it touches are a contiguous
        # window ending at chunk_end. That makes per-chunk cu_seqlens a pure
        # index computation and lets us drop the prefix of already-finished
        # documents from the deep-KV read entirely.
        doc_bounds = kwargs.get("cu_seqlens")
        if doc_bounds is not None:
            if batch_size != 1:
                raise ValueError("cu_seqlens (varlen) requires batch_size == 1")
            bounds = doc_bounds.detach().to("cpu").flatten().tolist()
            bounds = [int(b) for b in bounds]
            if bounds[0] != 0 or bounds[-1] != seq_len:
                raise ValueError(
                    f"cu_seqlens must span [0, seq_len]; got [{bounds[0]}, {bounds[-1]}] for seq_len={seq_len}"
                )
            # Non-monotone boundaries would silently produce negative segment
            # lengths below and hand garbage cu_seqlens to the kernel, so reject
            # them here rather than train on nonsense.
            if any(b <= a for a, b in zip(bounds, bounds[1:])):
                raise ValueError(f"cu_seqlens must be strictly increasing; got {bounds[:8]}...")

        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            is_first_chunk = chunk_start == 0
            hidden_states = inputs_embeds[:, chunk_start:chunk_end, :]
            chunk_position_ids = position_ids[:, chunk_start:chunk_end]
            position_embeddings = self.rotary_emb(hidden_states, chunk_position_ids)

            chunk_kwargs = {}
            if doc_bounds is not None:
                # documents overlapping [chunk_start, chunk_end)
                segs = [
                    (bounds[j], bounds[j + 1])
                    for j in range(len(bounds) - 1)
                    if bounds[j] < chunk_end and bounds[j + 1] > chunk_start
                ]
                key_start = segs[0][0]
                cu_q, cu_k = [0], [0]
                for ds, de in segs:
                    cu_q.append(min(de, chunk_end) - chunk_start - max(ds - chunk_start, 0) + cu_q[-1])
                    cu_k.append(min(de, chunk_end) - max(ds, key_start) + cu_k[-1])
                max_q = max(cu_q[i + 1] - cu_q[i] for i in range(len(cu_q) - 1))
                max_k = max(cu_k[i + 1] - cu_k[i] for i in range(len(cu_k) - 1))
                dev = inputs_embeds.device
                chunk_kwargs["akv_varlen"] = (
                    key_start,
                    torch.tensor(cu_q, dtype=torch.int32, device=dev),
                    torch.tensor(cu_k, dtype=torch.int32, device=dev),
                    max_q,
                    max_k,
                )

            curr_local_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
            for layer_idx, decoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                past_kv = AKV254PastKV(
                    deep_kv_cache=deep_caches[layer_idx],
                    prev_local_kv_cache=prev_local_kv[layer_idx],
                    prev_deep_kv_cache=prev_deep_kv[layer_idx],
                )
                hidden_states, curr_kv = self._call_layer(
                    decoder_layer,
                    layer_idx,
                    is_first_chunk,
                    hidden_states,
                    position_embeddings,
                    past_kv,
                    **chunk_kwargs,
                )
                curr_local_kv.append(curr_kv)

            chunk_outputs.append(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self._shared_deep_cache:
                _shared = self._append_deep_cache(
                    self.layers[self._shared_deep_src],
                    deep_caches[0],
                    prev_deep_kv[0],
                )
                deep_caches = [_shared for _ in self.layers]
            else:
                for layer_idx, layer in enumerate(self.layers):
                    deep_caches[layer_idx] = self._append_deep_cache(
                        layer,
                        deep_caches[layer_idx],
                        prev_deep_kv[layer_idx],
                    )

            if chunk_end < seq_len:
                prev_deep_kv = self._next_prev_deep_kv(curr_local_kv)
                prev_local_kv = curr_local_kv
            self._clear_forward_kv()

        hidden_states = torch.cat(chunk_outputs, dim=1)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if return_dict is False:
            return tuple(v for v in [hidden_states, None, all_hidden_states, None] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class CustomRecurrentMarinAKV254ReuseKVForCausalLM(AKV254MarinPreTrainedModel, GenerationMixin):
    config_class = CustomRecurrentMarinAKV254ReuseKVConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomRecurrentMarinAKV254ReuseKVModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> dict[str, object]:
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs["past_key_values"] = None
        model_inputs["use_cache"] = False
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        for key, value in kwargs.items():
            if key not in {"cache_position", "position_ids"}:
                model_inputs[key] = value
        return model_inputs

    def _chunked_linear_cross_entropy(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor,
        num_items_in_batch: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        shift_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Liger FLCE for TRAINING only (identical (total, ce, z) contract, no
        # materialized logits -> lower peak, faster at chunk 2048); eval keeps
        # chunked CE — the fused Triton loss/norm kernels hang in the 448-rank
        # benchmark-eval path (see qk_norm.py / liger_flce.py). Mirrors the
        # custom_marin arm.
        loss_fn = (
            liger_linear_cross_entropy
            if self.training and getattr(self.config, "use_liger_flce", False)
            else chunked_linear_cross_entropy
        )
        total, ce_detached, z_detached = loss_fn(
            hidden_states=hidden_states,
            labels=labels,
            lm_head=self.lm_head,
            config=self.config,
            num_items_in_batch=num_items_in_batch,
            ignore_index=ignore_index,
            shift_labels=shift_labels,
            **kwargs,
        )
        self._last_ce_loss = ce_detached
        self._last_z_loss = z_detached
        return total

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # flame passes cu_seqlens when training.varlen=true. It used to be
        # swallowed here by **kwargs and never reached the model, so document
        # boundaries were silently ignored (verified: logits were bit-identical
        # with and without it). Thread it through explicitly.
        model_kwargs = {}
        if kwargs.get("cu_seqlens") is not None:
            model_kwargs["cu_seqlens"] = kwargs["cu_seqlens"]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **model_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        loss = None
        use_chunked_loss = labels is not None and getattr(self.config, "fuse_linear_cross_entropy", True)
        if use_chunked_loss:
            loss = self._chunked_linear_cross_entropy(hidden_states=hidden_states, labels=labels, **kwargs)
            if isinstance(logits_to_keep, int) and logits_to_keep == 0:
                logits_to_keep = 1

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        if labels is not None and loss is None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if return_dict is False:
            output = (logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return (loss,) + output if loss is not None else output

        out = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        if use_chunked_loss:
            out.ce_loss = self._last_ce_loss
            out.z_loss = self._last_z_loss
        return out


__all__ = [
    "CustomRecurrentMarinAKV254ReuseKVForCausalLM",
    "CustomRecurrentMarinAKV254ReuseKVModel",
    "cache_append",
]
