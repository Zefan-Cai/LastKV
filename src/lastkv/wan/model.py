# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Adapted for LastKV inference from world_models commit b65c563ffac350811444cab1710c6903a6295215.
# Licensed under Apache-2.0; see LICENSES/Apache-2.0.txt and THIRD_PARTY_NOTICES.md.

"""
Aggregate KV v2541 Causal WAN Model.

Standalone implementation. AKV2541 keeps AKV254 previous-block interpolation
and uses each layer's own current K/V for the whole first block by default.
AKV2542 sets ``force_first_local_kv_tokens`` to keep a smaller token prefix,
such as one latent frame, local instead.
"""

import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import attention
from .layers import causal_rope_apply, CausalHead
from .layers import (
    WanRMSNorm,
    WanLayerNorm,
    WanT2VCrossAttention,
    rope_params,
    sinusoidal_embedding_1d,
)


VALID_AKV2541_MODES = {"perlayerkvcache", "sharekv", "reusekv", "perlayer_k_share_v", "perlayer_k_reuse_v"}
AKV2541_MODE_ALIASES = {
    "sharekvcache": "sharekv",
    "reusekvcache": "reusekv",
    "perlayerk_sharev": "perlayer_k_share_v",
    "perlayerk_reusev": "perlayer_k_reuse_v",
}


def normalize_akv2541_mode(akv_mode):
    mode = str(akv_mode).lower()
    return AKV2541_MODE_ALIASES.get(mode, mode)


class CacheAppend(torch.autograd.Function):
    """Append K/V into pre-allocated storage with backward connectivity."""

    @staticmethod
    def forward(ctx, storage, active_cache, x, start):
        bs = active_cache.size(0)
        end = start + x.size(1)
        if end > storage.size(1):
            raise RuntimeError(
                "Cache append would exceed storage capacity: "
                f"start={start}, new_tokens={x.size(1)}, end={end}, "
                f"capacity={storage.size(1)}, active_len={active_cache.size(1)}."
            )
        storage.data[:bs, start:end] = x.to(storage.dtype)
        ctx.bs = bs
        ctx.start = start
        ctx.end = end
        return storage[:bs, :end]

    @staticmethod
    def backward(ctx, grad_output):
        bs = ctx.bs
        start = ctx.start
        end = ctx.end
        return None, grad_output[:bs, :start], grad_output[:bs, start:end], None


cache_append = CacheAppend.apply


def append_to_cache(kv_cache, k_new, v_new, start=None):
    if kv_cache.get("shared_deep", False):
        from .cache import append_shared
        return append_shared(kv_cache, k_new, v_new, start)
    if "k_storage" in kv_cache:
        if start is None:
            start = kv_cache["k"].size(1)
        expected_start = kv_cache["cache_len"].item()
        if start != expected_start:
            raise RuntimeError(
                f"AKV2541 cache append expected start={expected_start}, got {start}."
            )
        new_k = cache_append(kv_cache["k_storage"], kv_cache["k"], k_new, start)
        new_v = cache_append(kv_cache["v_storage"], kv_cache["v"], v_new, start)
        kv_cache["k"] = new_k
        kv_cache["v"] = new_v
        kv_cache["cache_len"].fill_(new_k.size(1))
        return

    if start is None:
        start = kv_cache["cache_len"].item()
    expected_start = kv_cache["cache_len"].item()
    if start != expected_start:
        raise RuntimeError(
            f"AKV2541 cache append expected start={expected_start}, got {start}."
        )
    end = start + k_new.shape[1]
    kv_cache["k"][:, start:end] = k_new
    kv_cache["v"][:, start:end] = v_new
    kv_cache["cache_len"].fill_(end)


def _assert_cache_position(kv_cache, current_start, owner):
    cache_len = kv_cache["cache_len"].item()
    if current_start != cache_len:
        raise RuntimeError(
            f"{owner} expected current_start ({current_start}) to match "
            f"cache_len ({cache_len})."
        )


def _num_cache_heads(num_heads):
    return num_heads


def _local_head_values(values):
    return values


def _local_frames_per_block(num_frame_per_block):
    return num_frame_per_block


def _sp_prepare_qkv(q, k, v, grid_sizes, num_frame_per_block):
    return q, k, v, grid_sizes, None


def _sp_prepare_kv(k, v, grid_sizes, freqs, start_frame, num_frame_per_block):
    return causal_rope_apply(k, grid_sizes, freqs, start_frame=start_frame).type_as(v), v


def _sp_prepare_v(v, grid_sizes, num_frame_per_block):
    return v


def _sp_finish_attention_output(x, block_size):
    return x


def _sp_scatter_model_input(x, grid_sizes, num_frame_per_block):
    return x


def _sp_gather_model_output(x, grid_sizes, num_frame_per_block):
    return x


def _select_first_local_kv_segment(local_kv, source_kv, segment_start, keep_tokens):
    local_k, local_v = local_kv
    source_k, source_v = source_kv
    if local_k is None or local_v is None or source_k is None or source_v is None:
        return source_kv

    chunk_len = source_k.size(1)
    if keep_tokens is None:
        keep_tokens = chunk_len
    keep_n = min(max(int(keep_tokens) - int(segment_start), 0), chunk_len, local_k.size(1), local_v.size(1))
    if keep_n <= 0:
        return source_kv
    if keep_n >= chunk_len:
        return local_k[:, :chunk_len].to(source_k.dtype), local_v[:, :chunk_len].to(source_v.dtype)
    return (
        torch.cat((local_k[:, :keep_n].to(source_k.dtype), source_k[:, keep_n:]), dim=1),
        torch.cat((local_v[:, :keep_n].to(source_v.dtype), source_v[:, keep_n:]), dim=1),
    )


def _blend_first_local_kv_segment(local_kv, source_kv, segment_start, keep_tokens, alpha):
    local_k, local_v = local_kv
    source_k, source_v = source_kv
    chunk_len = min(source_k.size(1), source_v.size(1), local_k.size(1), local_v.size(1))
    if keep_tokens is None:
        keep_tokens = chunk_len
    local_k = local_k[:, :chunk_len].to(source_k.dtype)
    local_v = local_v[:, :chunk_len].to(source_v.dtype)
    source_k = source_k[:, :chunk_len]
    source_v = source_v[:, :chunk_len]
    alpha = alpha.to(device=source_k.device, dtype=source_k.dtype)
    blended_k = alpha * source_k + (1 - alpha) * local_k
    blended_v = alpha * source_v + (1 - alpha) * local_v

    keep_n = min(max(int(keep_tokens) - int(segment_start), 0), chunk_len)
    if keep_n <= 0:
        return blended_k, blended_v
    if keep_n >= chunk_len:
        return local_k, local_v
    return (
        torch.cat((local_k[:, :keep_n], blended_k[:, keep_n:]), dim=1),
        torch.cat((local_v[:, :keep_n], blended_v[:, keep_n:]), dim=1),
    )


class AKV2541CausalSelfAttention(nn.Module):
    """AKV attention reads deep history plus prev-block current K/V."""

    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    @staticmethod
    def _compose_history_kv(kv_cache, history_len, prev_block_len):
        history_len = min(history_len, kv_cache["cache_len"].item())
        if history_len <= 0:
            return None, None
        if kv_cache.get("shared_deep", False):
            from .cache import history_kv
            deep_k, deep_v = history_kv(kv_cache, history_len)
        else:
            deep_k = kv_cache["k"][:, :history_len]
            deep_v = kv_cache["v"][:, :history_len]
        curr_k = kv_cache.get("curr_k")
        curr_v = kv_cache.get("curr_v")
        if curr_k is None or curr_v is None:
            raise RuntimeError(
                "AKV2541 requires curr_k/curr_v in kv_cache once history is non-empty. "
                "Use the model or causal inference pipeline AKV cache allocator."
            )
        curr_start = kv_cache.get("curr_start", 0)
        curr_start = int(curr_start.item()) if torch.is_tensor(curr_start) else int(curr_start)
        prev_len = min(prev_block_len, history_len)
        if prev_len <= 0:
            return deep_k, deep_v
        prev_start = history_len - prev_len
        pieces_k = []
        pieces_v = []
        if prev_start > 0:
            pieces_k.append(deep_k[:, :prev_start])
            pieces_v.append(deep_v[:, :prev_start])
        deep_prev_k = deep_k[:, prev_start:history_len]
        deep_prev_v = deep_v[:, prev_start:history_len]
        curr_end = curr_start + curr_k.size(1)
        if curr_start > prev_start or curr_end < history_len or curr_v.size(1) != curr_k.size(1):
            raise RuntimeError(
                f"AKV2541 current-history cache does not cover previous block: "
                f"curr_start={curr_start}, curr_len={curr_k.size(1)}, "
                f"curr_v_len={curr_v.size(1)}, prev=[{prev_start}, {history_len})."
            )
        rel_start = prev_start - curr_start
        rel_end = history_len - curr_start
        curr_prev_k = curr_k[:, rel_start:rel_end]
        curr_prev_v = curr_v[:, rel_start:rel_end]
        alpha = kv_cache.get("prev_block_alpha")
        if alpha is None:
            alpha = deep_prev_k.new_full((1, 1, deep_prev_k.size(2), 1), 0.5)
        blended_prev_k, blended_prev_v = _blend_first_local_kv_segment(
            (curr_prev_k, curr_prev_v),
            (deep_prev_k, deep_prev_v),
            prev_start,
            kv_cache.get("force_first_local_kv_tokens"),
            alpha,
        )
        pieces_k.append(blended_prev_k)
        pieces_v.append(blended_prev_v)
        if len(pieces_k) == 1:
            return pieces_k[0], pieces_v[0]
        return torch.cat(pieces_k, dim=1), torch.cat(pieces_v, dim=1)

    def forward(
        self, x, grid_sizes, freqs, kv_cache=None,
        current_start=0, is_cache_building=False,
        is_teacher_forcing=False, num_frame_per_block=1,
        return_attn_weights=False, return_current_kv=False,
    ):
        current_kv = None
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        if kv_cache is None:
            raise RuntimeError("AKV2541CausalSelfAttention requires kv_cache.")
        if is_cache_building:
            attn_weights = None
            x, current_kv = self._forward_cache_building(
                q, k, v, grid_sizes, freqs, kv_cache, current_start,
                num_frame_per_block=num_frame_per_block,
            )
        else:
            x, attn_weights, current_kv = self._forward_teacher_forcing(
                q, k, v, grid_sizes, freqs, kv_cache, current_start,
                is_teacher_forcing=is_teacher_forcing,
                num_frame_per_block=num_frame_per_block,
                return_attn_weights=return_attn_weights,
            )
        x = x.flatten(2)
        x = self.o(x)
        if return_attn_weights and return_current_kv:
            return x, attn_weights, current_kv
        if return_attn_weights:
            return x, attn_weights
        if return_current_kv:
            return x, current_kv
        return x

    def _forward_cache_building(
        self, q, k, v, grid_sizes, freqs, kv_cache, current_start,
        num_frame_per_block=1,
    ):
        _assert_cache_position(kv_cache, current_start, "AKV2541 cache building")
        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        frame_idx = current_start // frame_seqlen
        block_len = frame_seqlen * num_frame_per_block
        q, k, v, attn_grid_sizes, sp_block_size = _sp_prepare_qkv(
            q, k, v, grid_sizes, num_frame_per_block,
        )
        roped_q = causal_rope_apply(q, attn_grid_sizes, freqs, start_frame=frame_idx).type_as(v)
        roped_k = causal_rope_apply(k, attn_grid_sizes, freqs, start_frame=frame_idx).type_as(v)
        history_k, history_v = self._compose_history_kv(kv_cache, current_start, block_len)
        if history_k is not None:
            full_k = torch.cat([history_k, roped_k], dim=1)
            full_v = torch.cat([history_v, v], dim=1)
        else:
            full_k = roped_k
            full_v = v
        x = _sp_finish_attention_output(attention(roped_q, full_k, full_v), sp_block_size)
        return x, (roped_k, v)

    def _forward_teacher_forcing(
        self, q, k, v, grid_sizes, freqs, kv_cache, current_start,
        is_teacher_forcing=False, num_frame_per_block=1, return_attn_weights=False,
    ):
        frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        local_num_new_tokens = q.shape[1]
        attn_weights = None
        if is_teacher_forcing:
            q, k, v, attn_grid_sizes, sp_block_size = _sp_prepare_qkv(
                q, k, v, grid_sizes, num_frame_per_block,
            )
            roped_q = causal_rope_apply(q, attn_grid_sizes, freqs, start_frame=0).type_as(v)
            roped_k = causal_rope_apply(k, attn_grid_sizes, freqs, start_frame=0).type_as(v)
            block_size = frame_seqlen * num_frame_per_block
            local_block_size = frame_seqlen * _local_frames_per_block(num_frame_per_block)
            if local_num_new_tokens % local_block_size != 0:
                raise ValueError(
                    f"AKV2541 teacher forcing requires local token count ({local_num_new_tokens}) "
                    f"to be divisible by local block size ({local_block_size})."
                )
            num_blocks = local_num_new_tokens // local_block_size
            outputs = []
            for block_idx in range(num_blocks):
                block_start = block_idx * block_size
                block_end = block_start + block_size
                clean_context_end = block_idx * block_size
                history_k, history_v = self._compose_history_kv(kv_cache, clean_context_end, block_size)
                k_noisy = roped_k[:, block_start:block_end]
                v_noisy = v[:, block_start:block_end]
                if history_k is not None:
                    k_block = torch.cat([history_k, k_noisy], dim=1)
                    v_block = torch.cat([history_v, v_noisy], dim=1)
                else:
                    k_block = k_noisy
                    v_block = v_noisy
                out_block = attention(roped_q[:, block_start:block_end], k_block, v_block)
                outputs.append(out_block)
            return _sp_finish_attention_output(torch.cat(outputs, dim=1), sp_block_size), attn_weights, (roped_k, v)

        frame_idx = current_start // frame_seqlen
        block_len = frame_seqlen * num_frame_per_block
        _assert_cache_position(kv_cache, current_start, "AKV2541 inference")
        q, k, v, attn_grid_sizes, sp_block_size = _sp_prepare_qkv(
            q, k, v, grid_sizes, num_frame_per_block,
        )
        roped_q = causal_rope_apply(q, attn_grid_sizes, freqs, start_frame=frame_idx).type_as(v)
        roped_k = causal_rope_apply(k, attn_grid_sizes, freqs, start_frame=frame_idx).type_as(v)
        history_k, history_v = self._compose_history_kv(kv_cache, current_start, block_len)
        if history_k is not None:
            full_k = torch.cat([history_k, roped_k], dim=1)
            full_v = torch.cat([history_v, v], dim=1)
        else:
            full_k = roped_k
            full_v = v
        x = _sp_finish_attention_output(attention(roped_q, full_k, full_v), sp_block_size)
        if return_attn_weights:
            raise NotImplementedError("Attention visualization is not part of this inference package.")
        return x, attn_weights, (roped_k, v)

    def project_kv(self, x, grid_sizes, freqs, start_frame, num_frame_per_block):
        b, s = x.shape[:2]
        n, d = self.num_heads, self.head_dim
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return _sp_prepare_kv(k, v, grid_sizes, freqs, start_frame, num_frame_per_block)


class AKV2541CausalAttentionBlock(nn.Module):
    """Transformer block for AKV2541."""

    def __init__(self, dim, ffn_dim, num_heads, qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = AKV2541CausalSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(
        self, x, e, grid_sizes, freqs, context, context_lens,
        kv_cache=None, crossattn_cache=None, current_start=0,
        is_cache_building=False, is_teacher_forcing=False,
        num_frame_per_block=1, return_attn_input=False, return_attn_weights=False,
        project_kv_only=False, project_kv_start_frame=0,
    ):
        if project_kv_only:
            return self.self_attn.project_kv(
                x, grid_sizes, freqs, project_kv_start_frame, num_frame_per_block,
            )

        num_frames = e.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        with torch.autocast(device_type=x.device.type, enabled=False):
            e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        x_norm = (self.norm1(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2)
        attn_weights = None
        current_kv = None
        if return_attn_weights:
            if return_attn_input:
                y, attn_weights, current_kv = self.self_attn(
                    x_norm, grid_sizes, freqs, kv_cache, current_start,
                    is_cache_building=is_cache_building,
                    is_teacher_forcing=is_teacher_forcing,
                    num_frame_per_block=num_frame_per_block,
                    return_attn_weights=True,
                    return_current_kv=True,
                )
            else:
                y, attn_weights = self.self_attn(
                    x_norm, grid_sizes, freqs, kv_cache, current_start,
                    is_cache_building=is_cache_building,
                    is_teacher_forcing=is_teacher_forcing,
                    num_frame_per_block=num_frame_per_block,
                    return_attn_weights=True,
                )
        else:
            if return_attn_input:
                y, current_kv = self.self_attn(
                    x_norm, grid_sizes, freqs, kv_cache, current_start,
                    is_cache_building=is_cache_building,
                    is_teacher_forcing=is_teacher_forcing,
                    num_frame_per_block=num_frame_per_block,
                    return_current_kv=True,
                )
            else:
                y = self.self_attn(
                    x_norm, grid_sizes, freqs, kv_cache, current_start,
                    is_cache_building=is_cache_building,
                    is_teacher_forcing=is_teacher_forcing,
                    num_frame_per_block=num_frame_per_block,
                )
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e[2]).flatten(1, 2)
        x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache)
        x_norm2 = (self.norm2(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
        y = self.ffn(x_norm2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
        if return_attn_input and return_attn_weights:
            return x, current_kv, attn_weights
        if return_attn_input:
            return x, current_kv
        if return_attn_weights:
            return x, attn_weights
        return x


class AKV2541ReuseKVCausalWanModel(ModelMixin, ConfigMixin):
    """Standalone WAN causal model for AKV2541."""

    ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim']
    _no_split_modules = ['AKV2541CausalAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        model_type='t2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        qk_norm=True,
        cross_attn_norm=True,
        akv_mode='reusekv',
        num_frame_per_block=1,
        v_gap=95,
        force_first_local_kv_tokens=None,
        eps=1e-6,
    ):
        super().__init__()
        assert model_type == 't2v'
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        requested_akv_mode = AKV2541_MODE_ALIASES.get(str(akv_mode).lower(), str(akv_mode).lower())
        if requested_akv_mode != 'reusekv':
            raise ValueError("This standalone model only implements akv_mode='reusekv'.")
        self.akv_mode = 'reusekv'
        self.eps = eps
        self.num_frame_per_block = num_frame_per_block
        self.v_gap = v_gap
        if force_first_local_kv_tokens is None:
            self.force_first_local_kv_tokens = None
        else:
            self.force_first_local_kv_tokens = int(force_first_local_kv_tokens)
            if self.force_first_local_kv_tokens < 1:
                raise ValueError("force_first_local_kv_tokens must be a positive integer.")
        self.v_layer_idxs = []
        for layer_idx in range(num_layers):
            if v_gap is None:
                self.v_layer_idxs.append(layer_idx)
            else:
                self.v_layer_idxs.append(max(min(layer_idx + int(v_gap), num_layers - 1), 0))

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.pre_v_rms_norm = WanLayerNorm(dim, eps)
        if self.akv_mode == "sharekv":
            self.shared_kv_norm = nn.Identity()
            self.shared_kv = nn.Linear(dim, 2 * dim)
        if self.akv_mode == "perlayer_k_share_v":
            self.shared_v_proj = nn.Linear(dim, dim)
        self.prev_block_alphas = nn.Parameter(torch.zeros(num_layers, num_heads))
        self.blocks = nn.ModuleList([AKV2541CausalAttentionBlock(dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps) for _ in range(num_layers)])
        self.head = CausalHead(dim, out_dim, patch_size, eps)
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)
        self.init_weights()
        self.gradient_checkpointing = False

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for module in self.text_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        for module in self.time_embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.zeros_(self.head.head.weight)
        nn.init.zeros_(self.prev_block_alphas)


    def _prev_block_alpha(self, layer_idx, dtype, device):
        alpha = _local_head_values(self.prev_block_alphas[layer_idx].sigmoid())
        return alpha.to(device=device, dtype=dtype).view(1, 1, alpha.numel(), 1)

    def _set_prev_block_alpha(self, cache, layer_idx, dtype, device):
        cache["prev_block_alpha"] = self._prev_block_alpha(layer_idx, dtype, device)
        cache["force_first_local_kv_tokens"] = self.force_first_local_kv_tokens

    def _allocate_per_layer_caches(self, batch_size, total_tokens, dtype, device):
        head_dim = self.dim // self.num_heads
        cache_heads = _num_cache_heads(self.num_heads)
        caches = []
        for _ in range(self.num_layers):
            k_storage = torch.empty(batch_size, total_tokens, cache_heads, head_dim, dtype=dtype, device=device)
            v_storage = torch.empty(batch_size, total_tokens, cache_heads, head_dim, dtype=dtype, device=device)
            curr_k_storage = torch.empty(batch_size, total_tokens, cache_heads, head_dim, dtype=dtype, device=device)
            curr_v_storage = torch.empty(batch_size, total_tokens, cache_heads, head_dim, dtype=dtype, device=device)
            caches.append({
                "k_storage": k_storage,
                "v_storage": v_storage,
                "k": k_storage[:batch_size, :0],
                "v": v_storage[:batch_size, :0],
                "cache_len": torch.tensor([0], dtype=torch.long, device=device),
                "curr_k_storage": curr_k_storage,
                "curr_v_storage": curr_v_storage,
                "curr_k": curr_k_storage[:batch_size, :0],
                "curr_v": curr_v_storage[:batch_size, :0],
                "prev_block_alpha": None,
                "force_first_local_kv_tokens": self.force_first_local_kv_tokens,
            })
        return caches

    def _append_current_history(self, cache, k_new, v_new, start=None):
        if start is None:
            start = cache.get("curr_start", 0) + cache["curr_k"].size(1)
        compact_len = cache.get("curr_cache_max_len")
        if compact_len is not None:
            if k_new.size(1) > compact_len or v_new.size(1) > compact_len:
                raise RuntimeError(
                    f"AKV2541 current-history block length exceeds compact cache: "
                    f"k={k_new.size(1)}, v={v_new.size(1)}, max={compact_len}."
                )
            cache["curr_k_storage"][:, :k_new.size(1)].copy_(k_new)
            cache["curr_v_storage"][:, :v_new.size(1)].copy_(v_new)
            cache["curr_k"] = cache["curr_k_storage"][:k_new.shape[0], :k_new.size(1)]
            cache["curr_v"] = cache["curr_v_storage"][:v_new.shape[0], :v_new.size(1)]
            cache["curr_start"] = start
            return
        curr_start = cache.get("curr_start", 0)
        curr_start = int(curr_start.item()) if torch.is_tensor(curr_start) else int(curr_start)
        expected_start = curr_start + cache["curr_k"].size(1)
        if start != expected_start:
            raise RuntimeError(
                f"AKV2541 current-history cache append expected start={expected_start}, got {start}."
            )
        cache["curr_k"] = cache_append(cache["curr_k_storage"], cache["curr_k"], k_new, start - curr_start)
        cache["curr_v"] = cache_append(cache["curr_v_storage"], cache["curr_v"], v_new, start - curr_start)

    @staticmethod
    def _ensure_current_history_cache(cache):
        if "curr_k" in cache and "curr_v" in cache:
            return
        current_len = cache["cache_len"].item()
        compact_len = cache.get("curr_cache_max_len")
        if "k_storage" in cache and "v_storage" in cache:
            if compact_len is None:
                curr_k_storage = torch.empty_like(cache["k_storage"])
                curr_v_storage = torch.empty_like(cache["v_storage"])
                curr_start = 0
                curr_len = current_len
            else:
                curr_len = min(current_len, compact_len)
                curr_start = current_len - curr_len
                curr_k_storage = torch.empty(
                    cache["k_storage"].shape[0],
                    compact_len,
                    cache["k_storage"].shape[2],
                    cache["k_storage"].shape[3],
                    dtype=cache["k_storage"].dtype,
                    device=cache["k_storage"].device,
                )
                curr_v_storage = torch.empty_like(curr_k_storage)
            if curr_len > 0:
                curr_k_storage[:, :curr_len].copy_(cache["k"][:, curr_start:current_len])
                curr_v_storage[:, :curr_len].copy_(cache["v"][:, curr_start:current_len])
            cache["curr_k_storage"] = curr_k_storage
            cache["curr_v_storage"] = curr_v_storage
            cache["curr_k"] = curr_k_storage[:cache["k"].shape[0], :curr_len]
            cache["curr_v"] = curr_v_storage[:cache["v"].shape[0], :curr_len]
            cache["curr_start"] = curr_start
            return
        curr_k = torch.empty_like(cache["k"])
        curr_v = torch.empty_like(cache["v"])
        if current_len > 0:
            curr_k[:, :current_len].copy_(cache["k"][:, :current_len])
            curr_v[:, :current_len].copy_(cache["v"][:, :current_len])
        cache["curr_k_storage"] = curr_k
        cache["curr_v_storage"] = curr_v
        cache["curr_k"] = curr_k[:cache["k"].shape[0], :current_len]
        cache["curr_v"] = curr_v[:cache["v"].shape[0], :current_len]
        cache["curr_start"] = 0

    def _ensure_current_history_caches(self, caches):
        for cache in caches:
            self._ensure_current_history_cache(cache)

    def _project_current_kv(self, layer_idx, hidden_states, grid_sizes, start_frame):
        return self.blocks[layer_idx](
            hidden_states, None, grid_sizes, self.freqs, None, None,
            num_frame_per_block=self.num_frame_per_block,
            project_kv_only=True,
            project_kv_start_frame=start_frame,
        )

    def _project_kv(self, layer_idx, hidden_states, grid_sizes, start_frame):
        if self.akv_mode == "sharekv":
            b, s = hidden_states.shape[:2]
            n, d = self.num_heads, self.dim // self.num_heads
            shared_kv = self.shared_kv(self.shared_kv_norm(hidden_states))
            k, v = shared_kv.unflatten(-1, (2, n, d)).unbind(2)
            return _sp_prepare_kv(k, v, grid_sizes, self.freqs, start_frame, self.num_frame_per_block)
        return self._project_current_kv(layer_idx, hidden_states, grid_sizes, start_frame)

    def _append_prev_current_history(self, caches, current_layer_kvs, current_start):
        for layer_idx, (curr_k, curr_v) in enumerate(current_layer_kvs):
            self._append_current_history(caches[layer_idx], curr_k, curr_v, start=current_start)

    def _update_caches_akv(self, caches, current_layer_kvs, layer_outputs, grid_sizes, start_frame, current_start):
        num_active_layers = len(current_layer_kvs)
        if self.akv_mode != "reusekv":
            raise RuntimeError(f"Unexpected AKV mode for reusekv model: {self.akv_mode}.")
        for layer_idx in range(num_active_layers):
            source_idx = self.v_layer_idxs[layer_idx]
            source_kv = current_layer_kvs[source_idx]
            k_new, v_new = _select_first_local_kv_segment(
                current_layer_kvs[layer_idx],
                source_kv,
                current_start,
                self.force_first_local_kv_tokens,
            )
            append_to_cache(caches[layer_idx], k_new, v_new, start=current_start)
        self._append_prev_current_history(caches, current_layer_kvs, current_start)



    def forward(self, x, t, context, clean_x=None, clean_t=None, kv_cache=None, crossattn_cache=None, current_start=0, update_cache=False, return_attn_weights=False):
        if kv_cache is not None:
            return self._forward_inference(
                x, t, context, kv_cache, crossattn_cache, current_start, update_cache,
                return_attn_weights=return_attn_weights,
            )
        raise ValueError("Inference requires a cache allocated with allocate_caches().")



    def _forward_inference(self, x, t, context, kv_cache, crossattn_cache, current_start, update_cache=False, return_attn_weights=False):
        self._ensure_current_history_caches(kv_cache)
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        x = torch.cat(x)
        x = _sp_scatter_model_input(x, grid_sizes, self.num_frame_per_block)
        with torch.autocast(device_type=x.device.type, enabled=False):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
            e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, t.shape)
        context = self.text_embedding(torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))
        current_layer_kvs = []
        layer_outputs = []
        attn_weights_dict = {} if return_attn_weights else None
        kwargs = dict(e=e0, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=None, current_start=current_start, is_cache_building=False, return_attn_input=True, return_attn_weights=return_attn_weights, num_frame_per_block=self.num_frame_per_block)
        for layer_idx, block in enumerate(self.blocks):
            layer_cache = kv_cache[layer_idx]
            self._set_prev_block_alpha(layer_cache, layer_idx, x.dtype, x.device)
            kwargs["kv_cache"] = layer_cache
            kwargs["crossattn_cache"] = crossattn_cache[layer_idx] if crossattn_cache else None
            if return_attn_weights:
                x, current_kv, attn_w = block(x, **kwargs)
                if attn_w is not None:
                    attn_weights_dict[layer_idx] = attn_w
            else:
                x, current_kv = block(x, **kwargs)
            current_layer_kvs.append(current_kv)
            layer_outputs.append(x)
        if update_cache:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            start_frame = current_start // frame_seqlen
            self._update_caches_akv(kv_cache, current_layer_kvs, layer_outputs, grid_sizes, start_frame=start_frame, current_start=current_start)
        x = _sp_gather_model_output(x, grid_sizes, self.num_frame_per_block)
        x = self.head(x, e.unflatten(0, t.shape).unsqueeze(2))
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if return_attn_weights:
            return x, attn_weights_dict
        return x

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        with torch.autocast(device_type=x.device.type, enabled=False):
            for u, v in zip(x, grid_sizes.tolist()):
                u = u[:math.prod(v)].view(*v, *self.patch_size, c)
                u = torch.einsum('fhwpqrc->cfphqwr', u)
                u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
                out.append(u)
        return out


    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        from .loading import load_model
        return load_model(model_id, **kwargs)
