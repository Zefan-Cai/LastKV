# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Adapted for LastKV. Licensed under Apache-2.0.
"""Wan attention using PyTorch's memory-efficient SDPA dispatcher."""

import torch
import torch.nn.functional as F


def attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
              softmax_scale=None, q_scale=None, causal=False,
              window_size=(-1, -1), deterministic=False,
              dtype=torch.bfloat16, fa_version=None, unflatten=True):
    original_dtype = q.dtype
    compute_dtype = dtype if q.is_cuda and q.dtype == torch.float32 else q.dtype
    q, k, v = (x.transpose(1, 2).to(compute_dtype) for x in (q, k, v))
    if q_scale is not None:
        q = q * q_scale
    mask = None
    if k_lens is not None:
        lengths = torch.as_tensor(k_lens, device=k.device)
        mask = torch.arange(k.shape[-2], device=k.device)[None, None, None, :] < lengths[:, None, None, None]
    if window_size != (-1, -1):
        raise ValueError("Sliding-window attention is not used by the LastKV Wan checkpoints.")
    result = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=dropout_p,
        is_causal=causal, scale=softmax_scale).transpose(1, 2).contiguous()
    if q_lens is not None:
        lengths = torch.as_tensor(q_lens, device=q.device)
        valid = torch.arange(result.shape[1], device=q.device)[None, :] < lengths[:, None]
        result = result * valid[:, :, None, None]
    result = result.to(original_dtype)
    return result if unflatten else result.flatten(0, 1)


flash_attention = attention
