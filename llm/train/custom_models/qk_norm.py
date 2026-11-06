"""Eager RMSNorm helper for mode-routed qk-norm.

Training uses fla's fused Triton RMSNorm (fast); eval routes through this
eager implementation with the SAME weight tensor, because the fla kernel
hangs deterministically in the 448-rank benchmark-eval path (variable padded
shapes; incident 2026-07-09, segments died at the first post-activation
benchmark step). Semantics match transformers.LlamaRMSNorm exactly.
"""

import torch


def eager_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(dtype)
