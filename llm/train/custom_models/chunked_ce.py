"""Shared chunked linear cross-entropy with optional z-loss.

custom_marin and custom_recurrent_marin both use this implementation so the
two arms of the architecture comparison share identical loss numerics.
z-loss (lse^2 penalty, Levanter/marin convention) activates when the model
config sets z_loss > 0; the penalty is differentiable and averaged over the
same valid-token denominator as the CE term.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def chunked_linear_cross_entropy(
    hidden_states: torch.Tensor,
    labels: torch.LongTensor,
    lm_head,
    config,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Token-chunked lm_head + CE. Returns (total_loss, ce_detached, z_detached)."""
    kwargs.pop("cu_seqlens", None)
    if shift_labels is None:
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
    if shift_labels.shape[-1] != hidden_states.shape[1]:
        raise ValueError("shift_labels must have the same sequence length as hidden_states.")

    chunk_size = getattr(config, "linear_cross_entropy_chunk_size", 512)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("linear_cross_entropy_chunk_size must be a positive integer.")
    z_coeff = float(getattr(config, "z_loss", 0.0) or 0.0)

    total_ce = hidden_states.new_zeros((), dtype=torch.float32)
    total_z = hidden_states.new_zeros((), dtype=torch.float32)
    total_items = hidden_states.new_zeros((), dtype=torch.float32)
    vocab_size = config.vocab_size
    for chunk_start in range(0, hidden_states.size(1), chunk_size):
        chunk_end = min(chunk_start + chunk_size, hidden_states.size(1))
        target = shift_labels[:, chunk_start:chunk_end].to(hidden_states.device).reshape(-1)
        logits = lm_head(hidden_states[:, chunk_start:chunk_end, :]).float().reshape(-1, vocab_size)
        total_ce = total_ce + F.cross_entropy(
            logits, target, ignore_index=ignore_index, reduction="sum"
        )
        valid = target != ignore_index
        if z_coeff > 0.0:
            lse = torch.logsumexp(logits, dim=-1)
            total_z = total_z + (lse * lse * valid.to(lse.dtype)).sum()
        total_items = total_items + valid.sum().to(total_items.dtype)

    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(total_ce.device)
        denom = num_items_in_batch
    else:
        denom = total_items.clamp_min(1)
    ce_loss = total_ce / denom
    z_loss = z_coeff * total_z / denom
    return ce_loss + z_loss, ce_loss.detach(), z_loss.detach()
