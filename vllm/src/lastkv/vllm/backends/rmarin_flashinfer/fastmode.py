"""Speed-mode switches for the RMarin vLLM plugin (2026-10-24).

``RMARIN_FAST=1`` turns on the standard-numerics fast path as a bundle; every
sub-switch can still be set explicitly.  Defaults (all off) reproduce the
audited engine bit-for-bit.
"""

from __future__ import annotations

import os


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip() == "1"


FAST = _flag("RMARIN_FAST", False)
# GEMM primitive: batch_invariant (Triton persistent matmul, audited) | cublas
LINEAR = os.environ.get("RMARIN_LINEAR", "cublas" if FAST else "batch_invariant")
# Fused CUDA RMSNorm / fused add+RMSNorm instead of eager float32 ops.
FUSED_NORM = _flag("RMARIN_FUSED_NORM", FAST)
# Layer-local K/V rings stored as FlashInfer pages read zero-copy.
PAGED_RING = _flag("RMARIN_PAGED_RING", FAST)
# Per-layer contract re-validation and the per-step positions sync.
STRICT = _flag("RMARIN_STRICT", not FAST)
# Minimal attention metadata builder (skips the stock FlashInfer per-step plans).
LEAN_METADATA = _flag("RMARIN_LEAN_METADATA", FAST)
# Permit vLLM CUDA graphs (stage B).
ALLOW_CUDAGRAPH = _flag("RMARIN_CUDAGRAPH", False)
# Decode fast path: host prologue in the metadata builder + static per-layer ops.
FDP = _flag("RMARIN_FDP", FAST)
# FlashInfer prefill-wrapper backend for the chunk passes (fa2 audited; fa3 on SM90 is ~2x faster).
PREFILL_BACKEND = os.environ.get("RMARIN_PREFILL_BACKEND", "fa3" if FAST else "fa2")
# Subclass vLLM's AsyncScheduler so scheduling overlaps GPU execution (needs --async-scheduling).
ASYNC_SCHED = _flag("RMARIN_ASYNC_SCHED", FAST)
# Shared source V: keep only the raw top-layer K/V (4 KiB/token) and apply the
# per-layer per-head value scale to the source attention output instead of
# materializing 32 scaled V planes (64 KiB/token).
SHARED_V = _flag("RMARIN_SHARED_V", FAST)
# Session/sequence limit (audited engine: 32).
MAX_SEQS = int(os.environ.get("RMARIN_MAX_SEQS", "256" if FAST else "32"))
# One attention call per layer over [older history | 2C ring] instead of
# source + local + merge.  Source pages and every layer's ring live in one
# paged K pool / V pool; rings hold V_l / s_l and the raw top-layer V slab is
# shared by all layers unscaled, so a single softmax over both parts times s_l
# equals source*s_l merged with local.
MERGED_ATTN = _flag("RMARIN_MERGED_ATTN", FAST and SHARED_V and PAGED_RING)
# Source (older-history) pages mirror vLLM's paged KV blocks instead of one
# static max_model_len range per slot: the placeholder KV spec is made tiny,
# ``num_gpu_blocks_override`` is set to our source page count, and vLLM's block
# tables / slot mapping address our K/V pools directly.  Admission, preemption
# and (with --enable-prefix-caching) retention of finished conversations for
# session resume are then vLLM's.
PAGED_SOURCE = _flag("RMARIN_PAGED_SOURCE", FAST and MERGED_ATTN)
# Explicit source budget in tokens (0 = derive from free GPU memory after rings).
SOURCE_TOKENS = int(os.environ.get("RMARIN_SOURCE_TOKENS", "0"))
# Placeholder spec geometry (1 head x 8 dims x bf16 x K,V = 32 B/token).
PLACEHOLDER_HEAD_SIZE = 8
# One Triton kernel per layer for rotary + q/k RMSNorm + V*(1/s) + ring/slab
# K,V writes (bit-identical to the five kernels it replaces; fused_qkv.py).
FUSED_QKV = _flag("RMARIN_FUSED_QKV", FAST and MERGED_ATTN and FUSED_NORM)
# Shared buffers for lazily spilled resume snapshots (0 = auto: max(4, slots/4)).
# A snapshot only needs a buffer when its request writes past the next chunk
# boundary after the snapshot (~response/C of turns), so this replaces the
# former 128 MiB-per-slot eager copy.
SNAPSHOT_BUFFERS = int(os.environ.get("RMARIN_SNAPSHOT_BUFFERS", "0"))


def snapshot_buffers(num_slots: int) -> int:
    return SNAPSHOT_BUFFERS if SNAPSHOT_BUFFERS > 0 else max(4, -(-int(num_slots) // 4))


# vLLM's fused reshape_and_cache_flash for the ring/slab K,V writes (one kernel
# instead of two index_copy_).
RAC = _flag("RMARIN_RAC", FAST)
# Apply the per-layer value scale as two small kernels around the merged
# attention (ring V = round(V * fp32(1/s)); output *= s).  RMARIN_FOLD_SCALES=1
# folds them into W_v / W_o instead: one kernel less per layer, but the bf16
# re-rounding of the scaled weights perturbs the model noticeably (mean
# |dlogprob| 0.06 vs 0.02 for the kernel form), so it stays off.
FOLD_SCALES = _flag("RMARIN_FOLD_SCALES", False)


def summary() -> dict[str, object]:
    return {
        "fast": FAST,
        "linear": LINEAR,
        "fused_norm": FUSED_NORM,
        "paged_ring": PAGED_RING,
        "strict": STRICT,
        "lean_metadata": LEAN_METADATA,
        "cudagraph": ALLOW_CUDAGRAPH,
        "fdp": FDP,
        "prefill_backend": PREFILL_BACKEND,
        "async_sched": ASYNC_SCHED,
        "shared_v": SHARED_V,
        "max_seqs": MAX_SEQS,
        "merged_attn": MERGED_ATTN,
        "fold_scales": FOLD_SCALES,
        "paged_source": PAGED_SOURCE,
        "source_tokens": SOURCE_TOKENS,
        "rac": RAC,
        "snapshot_buffers": SNAPSHOT_BUFFERS,
        "fused_qkv": FUSED_QKV,
    }
