"""Dependency-free validation for the recurrent pilot's KV cache layout."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .vllm_compat import KV_LAYOUT_BLOCKS_KV, KV_LAYOUT_KV_BLOCKS


@dataclass(frozen=True)
class FlashAttentionCacheAxes:
    kv: int
    blocks: int
    block_size: int
    heads: int
    head_dim: int


def flash_attn_cache_axes(layout: str) -> FlashAttentionCacheAxes:
    """Return semantic axes for a supported vLLM FlashAttention cache."""

    if layout == KV_LAYOUT_BLOCKS_KV:
        return FlashAttentionCacheAxes(
            kv=1,
            blocks=0,
            block_size=2,
            heads=3,
            head_dim=4,
        )
    if layout == KV_LAYOUT_KV_BLOCKS:
        return FlashAttentionCacheAxes(
            kv=0,
            blocks=1,
            block_size=2,
            heads=3,
            head_dim=4,
        )
    raise RuntimeError(f"unsupported recurrent KV cache layout {layout!r}")


def has_bound_recurrent_cache(
    *,
    target_metadata_present: bool,
    source_metadata_present: bool,
    target_numel: int,
    source_numel: int,
    explicit_dummy_forward: bool = False,
) -> bool:
    """Distinguish an explicit profile/dummy pass from a real request."""

    if target_metadata_present != source_metadata_present:
        raise RuntimeError("ordinary and raw-source cache metadata binding differs")
    if not target_metadata_present:
        if target_numel or source_numel:
            if explicit_dummy_forward:
                return False
            raise RuntimeError("populated recurrent cache has no metadata")
        return False
    return True


def validate_independent_recurrent_cache_bindings(
    *,
    target_is_source: bool,
    target_data_ptr: int,
    source_data_ptr: int,
) -> None:
    """Require ordinary and auxiliary source KV to use distinct storage views."""

    if target_is_source or target_data_ptr == source_data_ptr:
        raise RuntimeError(
            "ordinary and raw-source KV caches are not independently bound"
        )


def validate_flash_attn_cache_shape(
    shape: Sequence[int],
    *,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    layout: str = KV_LAYOUT_BLOCKS_KV,
) -> tuple[int, int, int, int, int]:
    """Require the selected vLLM FlashAttention NHD paged-cache layout.

    vLLM 0.11 uses ``[2, blocks, block, kv_heads, head_dim]`` while vLLM 0.25
    uses ``[blocks, 2, block, kv_heads, head_dim]``.  FlashInfer HND layouts
    also have rank five, so recurrent page mutation is permitted only for the
    exact versioned FlashAttention layout.
    """

    values = tuple(int(item) for item in shape)
    axes = flash_attn_cache_axes(layout)
    if len(values) != 5:
        raise RuntimeError("recurrent pilot requires a non-empty rank-five KV cache")
    if values[axes.blocks] < 1:
        raise RuntimeError("recurrent pilot requires a non-empty rank-five KV cache")
    expected = [0] * 5
    expected[axes.kv] = 2
    expected[axes.blocks] = values[axes.blocks]
    expected[axes.block_size] = block_size
    expected[axes.heads] = num_kv_heads
    expected[axes.head_dim] = head_dim
    if values != tuple(expected):
        raise RuntimeError(
            "recurrent pilot requires FlashAttention NHD cache shape "
            f"{tuple(expected)}, observed={values}"
        )
    return values


__all__ = [
    "FlashAttentionCacheAxes",
    "flash_attn_cache_axes",
    "has_bound_recurrent_cache",
    "validate_independent_recurrent_cache_bindings",
    "validate_flash_attn_cache_shape",
]
