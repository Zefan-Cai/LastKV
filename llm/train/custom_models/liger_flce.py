"""Liger fused-linear-cross-entropy route for the dense loss path.

Drop-in alternative to chunked_linear_cross_entropy with identical loss
semantics: total = ce + z_loss, z_loss = z_coeff * mean(lse^2) over the same
valid-token denominator, ce/z returned detached for logging. Never
materializes logits: gradients are produced inside the forward chunk loop
with an FP32 dW accumulator, so peak memory drops by the saved [N, V] FP32
logits (measured -6.8 GiB at V=128256, N=8192, z=1e-4 on H100, at equal
fwd+bwd time vs chunk-512).

Shape sensitivity: Liger splits the flattened batch into ~32 internal chunks
(next_pow2(N / ceil(V/H)) tokens each = 256 at marin 8k), so per-forward token
counts BELOW the production 8192 shrink its GEMMs and lose the time tie (~25%
slower e2e at N=4096). Do not enable for runs whose bsz*seq per forward is
< 8192.

Chunk override: with patches/liger-flce-chunk-size-env.patch applied to the
Liger install, config.liger_flce_chunk_size forces a larger chunk via the
LIGER_FORCE_CHUNK_SIZE env var. H100 sweep at the production shape (z=1e-4):
native-256 95.9 ms / +3.06 GiB; 512 60.0 / +3.12; 1024 46.2 / +3.24;
2048 42.3 / +3.49 — i.e. 2048 is 2.28x faster than the production chunk-512
CE at ~1/3 of its peak memory, with dW error unchanged (1.66e-3).

Requires Liger-Kernel at or after c9fc646 ("perf: optimize grad_weight
accumulation with addmm", 2026-07-01): earlier versions (including stable
v0.8.0) cast the dW GEMM operands to FP32, which drops the BF16 tensor-core
path and roughly doubles the loss-path time. The import guard below checks
for the addmm out_dtype fast path instead of a version string.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .chunked_ce import chunked_linear_cross_entropy

_CRITERION_CACHE: dict = {}
_CHUNK_ENV = "LIGER_FORCE_CHUNK_SIZE"


def _liger_criterion(z_coeff: float, ignore_index: int, chunk_size: int = 0):
    key = (z_coeff, ignore_index, chunk_size)
    if key not in _CRITERION_CACHE:
        try:
            from liger_kernel.ops import fused_linear_cross_entropy as _ops
            from liger_kernel.transformers.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyLoss,
            )
        except ImportError as exc:
            raise ImportError(
                "use_liger_flce=True requires liger_kernel; install from git at "
                "or after c9fc646 (v0.8.0 and earlier lack the addmm out_dtype "
                "dW fast path and run ~2x slower)."
            ) from exc
        if not hasattr(_ops, "_ADDMM_SUPPORTS_OUT_DTYPE"):
            raise ImportError(
                "installed liger_kernel predates c9fc646: its FLCE casts the dW "
                "GEMM operands to FP32 (no BF16 tensor cores). Install from git "
                "at or after c9fc646."
            )
        if chunk_size:
            with open(_ops.__file__) as f:
                if _CHUNK_ENV not in f.read():
                    raise ImportError(
                        "liger_flce_chunk_size is set but the installed "
                        "liger_kernel lacks the chunk-size override — apply "
                        "patches/liger-flce-chunk-size-env.patch to the install."
                    )
            import os

            os.environ[_CHUNK_ENV] = str(chunk_size)
        _CRITERION_CACHE[key] = LigerFusedLinearCrossEntropyLoss(
            ignore_index=ignore_index,
            lse_square_scale=z_coeff,
            reduction="mean",
            return_z_loss=z_coeff > 0.0,
            accum_dtype=torch.float32,
        )
    return _CRITERION_CACHE[key]


def liger_linear_cross_entropy(
    hidden_states: torch.Tensor,
    labels: torch.LongTensor,
    lm_head,
    config,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Same contract as chunked_linear_cross_entropy: (total, ce_det, z_det)."""
    if num_items_in_batch is not None:
        # mean-over-valid-tokens is baked into the Liger kernel; an external
        # denominator needs the chunked path (flame never passes one today)
        return chunked_linear_cross_entropy(
            hidden_states, labels, lm_head, config,
            num_items_in_batch=num_items_in_batch,
            ignore_index=ignore_index, shift_labels=shift_labels, **kwargs,
        )
    kwargs.pop("cu_seqlens", None)
    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
    if shift_labels.shape[-1] != hidden_states.shape[1]:
        raise ValueError("shift_labels must have the same sequence length as hidden_states.")

    z_coeff = float(getattr(config, "z_loss", 0.0) or 0.0)
    chunk_size = int(getattr(config, "liger_flce_chunk_size", 0) or 0)
    criterion = _liger_criterion(z_coeff, ignore_index, chunk_size)
    out = criterion(
        lm_head.weight,
        hidden_states.reshape(-1, hidden_states.shape[-1]),
        shift_labels.to(hidden_states.device).reshape(-1),
        lm_head.bias,
    )
    if z_coeff > 0.0:
        total, z_detached = out.loss, out.z_loss.detach()
    else:
        total, z_detached = out, hidden_states.new_zeros((), dtype=torch.float32)
    return total, total.detach() - z_detached, z_detached
