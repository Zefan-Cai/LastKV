"""Fused per-layer QKV epilogue for the merged-attention fast path (L5, 2026-10-25).

One Triton kernel replaces five per-layer kernels of the fast path:

    vLLM rotary_embedding (q, k) -> vLLM rms_norm (q) -> vLLM rms_norm (k)
    -> torch.mul(v, fp32 1/s, out=bf16) -> reshape_and_cache_flash (ring k, v)
    (+ on the source layer: reshape_and_cache_flash (raw slab k, v))

The arithmetic replicates those kernels operation by operation (see
``_rope_rms``), so the fused path is bit-identical to the unfused one;
``tools/fused_qkv_test.py`` checks this on the target GPU.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_rms(x_ptr, pos, cos_ptr, cache_stride, w_ptr, eps, HALF: tl.constexpr):
    """Neox rotary on a 128-dim head, then RMSNorm; bit-identical to vLLM's kernels.

    rotary_embedding_kernel<bf16>: float math, x*cos - y*sin contracted to one
    fused multiply-add, a single bf16 rounding.  rms_norm_kernel<bf16, 8>: 16
    threads each accumulate 8 squares sequentially (fused multiply-add), the CUB
    warp reduction pairs adjacent partials, rsqrt.approx, and ``x * inv * w``
    in float with one rounding.  (Checked: 0 mismatches over 8.4M elements.)
    """

    D: tl.constexpr = 2 * HALF
    offs = tl.arange(0, HALF)
    xf = tl.load(x_ptr + offs).to(tl.float32)
    yf = tl.load(x_ptr + HALF + offs).to(tl.float32)
    cf = tl.load(cos_ptr + pos * cache_stride + offs).to(tl.float32)
    sf = tl.load(cos_ptr + pos * cache_stride + HALF + offs).to(tl.float32)
    r1 = tl.fma(xf, cf, -(yf * sf)).to(tl.bfloat16)
    r2 = tl.fma(yf, cf, xf * sf).to(tl.bfloat16)
    full = tl.reshape(tl.permute(tl.join(r1, r2), (1, 0)), (D,)).to(tl.float32)
    grp = tl.reshape(full, (D // 8, 8))
    lane = tl.arange(0, 8)
    acc = tl.zeros((D // 8,), dtype=tl.float32)
    for jj in tl.static_range(8):
        v = tl.sum(tl.where(lane[None, :] == jj, grp, 0.0), axis=1)
        acc = tl.fma(v, v, acc)
    t8 = tl.sum(tl.reshape(acc, (8, 2)), axis=1)
    t4 = tl.sum(tl.reshape(t8, (4, 2)), axis=1)
    t2 = tl.sum(tl.reshape(t4, (2, 2)), axis=1)
    var = tl.sum(t2, axis=0)
    inv = tl.math.rsqrt(var / D + eps)
    w = tl.load(w_ptr + tl.arange(0, D)).to(tl.float32)
    return (full * inv * w).to(tl.bfloat16)


@triton.jit
def _fused_qkv_kernel(
    q_ptr, k_ptr, v_ptr, q_row, k_row, v_row,
    pos_ptr, cos_ptr, cache_stride,
    qw_ptr, kw_ptr, eps,
    inv_ptr,
    q_out_ptr,
    ring_idx_ptr, ring_k_ptr, ring_v_ptr,
    src_idx_ptr, src_k_ptr, src_v_ptr,
    NQ: tl.constexpr, NK: tl.constexpr, HALF: tl.constexpr,
    TOP: tl.constexpr,
):
    D: tl.constexpr = 2 * HALF
    t = tl.program_id(0)
    h = tl.program_id(1)
    pos = tl.load(pos_ptr + t)
    d = tl.arange(0, D)
    if h < NQ:
        out = _rope_rms(q_ptr + t * q_row + h * D, pos, cos_ptr, cache_stride, qw_ptr, eps, HALF)
        tl.store(q_out_ptr + (t * NQ + h) * D + d, out)
    else:
        j = h - NQ
        kn = _rope_rms(k_ptr + t * k_row + j * D, pos, cos_ptr, cache_stride, kw_ptr, eps, HALF)
        slot = tl.load(ring_idx_ptr + t)
        tl.store(ring_k_ptr + (slot * NK + j) * D + d, kn)
        v = tl.load(v_ptr + t * v_row + j * D + d)
        inv = tl.load(inv_ptr + j)
        vt = (v.to(tl.float32) * inv).to(tl.bfloat16)
        tl.store(ring_v_ptr + (slot * NK + j) * D + d, vt)
        if TOP:
            sslot = tl.load(src_idx_ptr + t)
            tl.store(src_k_ptr + (sslot * NK + j) * D + d, kn)
            tl.store(src_v_ptr + (sslot * NK + j) * D + d, v)


def fused_qkv(
    q: Any, k: Any, v: Any, positions: Any, cos_sin_cache: Any, q_weight: Any, k_weight: Any, eps: float,
    inv_scale: Any, ring_idx: Any, ring_k: Any, ring_v: Any,
    src_idx: Any | None = None, src_k: Any | None = None, src_v: Any | None = None,
    q_out: Any | None = None,
) -> Any:
    """q [T, NQ*D], k/v [T, NK*D] (row-strided views allowed); returns q_out [T, NQ, D].

    ring_k/ring_v/src_k/src_v: [slots, NK, D] flat token views of the page pools.
    """

    T = int(q.shape[0])
    D = int(cos_sin_cache.shape[-1])
    if D != 128:
        raise ValueError("fused_qkv supports head_dim 128 (full-dim neox rotary)")
    NQ = int(q.shape[-1]) // D
    NK = int(k.shape[-1]) // D
    if q_out is None:
        q_out = torch.empty((T, NQ, D), dtype=q.dtype, device=q.device)
    top = src_idx is not None
    if T == 0:
        return q_out
    _fused_qkv_kernel[(T, NQ + NK)](
        q, k, v, q.stride(0), k.stride(0), v.stride(0),
        positions, cos_sin_cache, cos_sin_cache.stride(0),
        q_weight, k_weight, float(eps),
        inv_scale,
        q_out,
        ring_idx, ring_k, ring_v,
        src_idx if top else ring_idx, src_k if top else ring_k, src_v if top else ring_v,
        NQ=NQ, NK=NK, HALF=D // 2, TOP=top,
        num_warps=1,
    )
    return q_out


@triton.jit
def _head_scale_kernel(x_ptr, s_ptr, n, HEADS: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    sc = tl.load(s_ptr + (offs // D) % HEADS, mask=mask)
    # torch bf16 mul: float product, one rounding
    tl.store(x_ptr + offs, (x.to(tl.float32) * sc.to(tl.float32)).to(tl.bfloat16), mask=mask)


def head_scale_(x: Any, scale: Any) -> Any:
    """In place ``x[t, h, :] *= scale[h]`` for a contiguous [T, H, D] bf16 tensor.

    Bit-identical to ``x.mul_(scale.view(1, H, 1))`` but vectorized (PyTorch's
    broadcast path ran at ~0.3 TB/s: 34 ms per 38k-token prefill).
    """

    if not x.is_contiguous():
        return x.mul_(scale.view(1, -1, 1))
    n = x.numel()
    if n == 0:
        return x
    heads = int(x.shape[1])
    block = 2048
    _head_scale_kernel[(triton.cdiv(n, block),)](x, scale, n, HEADS=heads, D=int(x.shape[2]), BLOCK=block, num_warps=4)
    return x


@triton.jit
def _blend_kernel(k_ptr, v_ptr, layer_stride, rk_ptr, rv_ptr, a_ptr, n, HEADS: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    layer = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + layer * HEADS + (offs // D) % HEADS, mask=mask).to(tl.float32)
    ia = (1.0 - a).to(tl.bfloat16).to(tl.float32)
    base = layer * layer_stride
    for which in tl.static_range(2):
        ring_ptr = k_ptr if which == 0 else v_ptr
        raw_ptr = rk_ptr if which == 0 else rv_ptr
        ring = tl.load(ring_ptr + base + offs, mask=mask).to(tl.float32)
        raw = tl.load(raw_ptr + offs, mask=mask).to(tl.float32)
        # torch: alpha*src -> bf16, (1-alpha) -> bf16, (1-alpha)*ring -> bf16, sum -> bf16
        t1 = (a * raw).to(tl.bfloat16).to(tl.float32)
        t3 = (ia * ring).to(tl.bfloat16).to(tl.float32)
        tl.store(ring_ptr + base + offs, (t1 + t3).to(tl.bfloat16), mask=mask)


def blend_chunk_(kview: Any, vview: Any, raw_k: Any, raw_v: Any, alpha: Any) -> bool:
    """In place ``ring = alpha*raw + (1-alpha)*ring`` for all layers of one chunk.

    kview/vview [L, C, H, D] (per-layer contiguous, common layer stride), raw [C, H, D]
    contiguous, alpha [L, 1, H, 1].  Bit-identical to the eager four-op expression
    (each op rounds to bf16).  Returns False if the layout is unsupported.
    """

    L, C, H, D = (int(x) for x in kview.shape)
    inner = (H * D, D, 1)
    if (
        tuple(kview.stride()[1:]) != inner or tuple(vview.stride()[1:]) != inner
        or kview.stride(0) != vview.stride(0)
        or not raw_k.is_contiguous() or not raw_v.is_contiguous()
        or tuple(raw_k.shape) != (C, H, D) or tuple(raw_v.shape) != (C, H, D)
        or not alpha.is_contiguous() or alpha.numel() != L * H
    ):
        return False
    n = C * H * D
    block = 4096
    _blend_kernel[(L, triton.cdiv(n, block))](
        kview, vview, kview.stride(0), raw_k, raw_v, alpha, n, HEADS=H, D=D, BLOCK=block, num_warps=8,
        enable_fp_fusion=False,  # the eager ops round each product to bf16 before the add
    )
    return True


__all__ = ["fused_qkv", "head_scale_", "blend_chunk_"]
