"""In-place recurrent boundary transition for vLLM's paged KV cache.

The compatibility backend keeps an ordinary cache for every transformer
layer plus one raw-source cache for every layer referenced by ``v_gap``.  At a
chunk boundary it mutates the ordinary pages into exactly the representation
consumed by the next chunk:

* the formerly-recent chunk is promoted to scaled raw source KV;
* the newly-completed chunk becomes the learned source/local blend.

This module is GPU-runtime code.  It intentionally imports torch/vLLM and is
not imported by the dependency-free static probe.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .cache_layout import (
    flash_attn_cache_axes,
    has_bound_recurrent_cache,
    validate_independent_recurrent_cache_bindings,
    validate_flash_attn_cache_shape,
)
from .vllm_compat import (
    VLLM_0_11_0,
    installed_runtime_profile,
    resolve_forward_context_binding,
    vllm_0_11_dummy_forward_active,
)


_RUNTIME_PROFILE = installed_runtime_profile()

if _RUNTIME_PROFILE.attention_context_strategy == "attention_context":
    from vllm.model_executor.layers.attention.attention import (
        get_attention_context,
    )
elif _RUNTIME_PROFILE.attention_context_strategy == "forward_context":
    from vllm.forward_context import get_forward_context
else:  # pragma: no cover - guarded by the dependency-free profile tests.
    raise RuntimeError("unsupported recurrent attention context strategy")


def _logical_slots(
    block_table_row: torch.Tensor,
    *,
    start: int,
    end: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if start < 0 or end < start:
        raise RuntimeError(f"invalid recurrent cache range [{start}, {end})")
    positions = torch.arange(start, end, device=block_table_row.device)
    block_indices = torch.div(positions, block_size, rounding_mode="floor")
    if (
        block_indices.numel()
        and int(block_indices[-1].item()) >= block_table_row.numel()
    ):
        raise RuntimeError("recurrent cache range exceeds the vLLM block table")
    block_ids = block_table_row.index_select(0, block_indices).long()
    if block_ids.numel() and bool(torch.any(block_ids < 0).item()):
        raise RuntimeError("recurrent cache range contains an unallocated KV block")
    offsets = torch.remainder(positions, block_size).long()
    return block_ids, offsets


def _read_range(
    cache: torch.Tensor,
    block_table_row: torch.Tensor,
    *,
    start: int,
    end: int,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    axes = flash_attn_cache_axes(layout)
    block_ids, offsets = _logical_slots(
        block_table_row,
        start=start,
        end=end,
        block_size=cache.shape[axes.block_size],
    )
    if axes.kv == 0:
        return cache[0, block_ids, offsets], cache[1, block_ids, offsets]
    return cache[block_ids, 0, offsets], cache[block_ids, 1, offsets]


def _write_range(
    cache: torch.Tensor,
    block_table_row: torch.Tensor,
    *,
    start: int,
    end: int,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: str,
) -> None:
    axes = flash_attn_cache_axes(layout)
    block_ids, offsets = _logical_slots(
        block_table_row,
        start=start,
        end=end,
        block_size=cache.shape[axes.block_size],
    )
    expected = (
        end - start,
        cache.shape[axes.heads],
        cache.shape[axes.head_dim],
    )
    if tuple(key.shape) != expected or tuple(value.shape) != expected:
        raise RuntimeError(
            f"recurrent KV write shape differs: expected={expected} "
            f"key={tuple(key.shape)} value={tuple(value.shape)}"
        )
    if axes.kv == 0:
        cache[0, block_ids, offsets] = key.to(dtype=cache.dtype)
        cache[1, block_ids, offsets] = value.to(dtype=cache.dtype)
    else:
        cache[block_ids, 0, offsets] = key.to(dtype=cache.dtype)
        cache[block_ids, 1, offsets] = value.to(dtype=cache.dtype)


def _scale_source(
    attention: Any,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not attention.deep_kv_rescale:
        return key, value
    if attention.deep_rescale_mode == "vnormqk":
        return key, attention.deep_v_norm(value)
    scale = attention.deep_v_scale.to(device=value.device, dtype=value.dtype)
    scale = scale.view(1, -1, 1)
    if attention.deep_rescale_mode == "v":
        return key, value * scale
    if attention.deep_rescale_mode == "vnorm":
        variance = value.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = (
            value.float() * torch.rsqrt(variance + attention.deep_norm_epsilon)
        ).to(value.dtype)
        return key, normalized * scale
    raise RuntimeError(
        f"unsupported recurrent deep rescale mode {attention.deep_rescale_mode!r}"
    )


def _metadata_rows(metadata: Any) -> tuple[list[int], list[int]]:
    if metadata is None:
        return [], []
    seq_lens = [int(item) for item in metadata.seq_lens.tolist()]
    starts = [int(item) for item in metadata.query_start_loc.tolist()]
    if len(starts) != len(seq_lens) + 1:
        raise RuntimeError("vLLM query_start_loc does not match seq_lens")
    query_lens = [starts[index + 1] - starts[index] for index in range(len(seq_lens))]
    return seq_lens, query_lens


@torch.no_grad()
def rewrite_completed_boundaries(
    layers: Sequence[Any],
    *,
    chunk_size: int,
    source_layer_by_target: Sequence[int],
) -> int:
    """Rewrite all request rows that completed exactly one chunk this step.

    The custom scheduler guarantees that a scheduled span cannot cross a
    boundary.  This function rechecks that invariant from attention metadata
    and returns the number of ``(request, target-layer)`` rewrites performed.
    """

    if len(layers) != len(source_layer_by_target):
        raise RuntimeError("recurrent source-layer map differs from model layers")
    if chunk_size < 1:
        raise RuntimeError("chunk_size must be positive")

    rewrites = 0
    context_cache: dict[str, tuple[Any, torch.Tensor]] = {}

    def context(attention: Any) -> tuple[Any, torch.Tensor]:
        layer_name = attention.layer_name
        result = context_cache.get(layer_name)
        if result is None:
            if _RUNTIME_PROFILE.attention_context_strategy == "attention_context":
                metadata, _, cache, _ = get_attention_context(layer_name)
                result = metadata, cache
            else:
                forward_context = get_forward_context()
                result = resolve_forward_context_binding(
                    forward_context=forward_context,
                    attention=attention,
                    layer_name=layer_name,
                )
            context_cache[layer_name] = result
        return result

    for target_index, layer in enumerate(layers):
        target_attention = layer.self_attn
        source_index = int(source_layer_by_target[target_index])
        if source_index < 0 or source_index >= len(layers):
            raise RuntimeError("recurrent source-layer index is out of range")
        source_attention = layers[source_index].self_attn
        if source_attention.source_cache is None:
            raise RuntimeError(
                f"source layer {source_index} has no auxiliary raw-source cache"
            )

        target_metadata, target_cache = context(target_attention.attn)
        source_metadata, source_cache = context(source_attention.source_cache)
        explicit_dummy_forward = (
            _RUNTIME_PROFILE.vllm_version == VLLM_0_11_0
            and target_metadata is None
            and source_metadata is None
            and vllm_0_11_dummy_forward_active()
        )
        if not has_bound_recurrent_cache(
            target_metadata_present=target_metadata is not None,
            source_metadata_present=source_metadata is not None,
            target_numel=target_cache.numel(),
            source_numel=source_cache.numel(),
            explicit_dummy_forward=explicit_dummy_forward,
        ):
            # Model initialization can run with empty placeholder caches.
            # vLLM 0.11 also performs one explicitly scoped post-allocation
            # memory-profile dummy forward with populated caches and no
            # metadata. Neither pass has request state to rewrite; populated
            # metadata-free caches outside the explicit scope remain an error.
            continue
        validate_independent_recurrent_cache_bindings(
            target_is_source=target_cache is source_cache,
            target_data_ptr=target_cache.data_ptr(),
            source_data_ptr=source_cache.data_ptr(),
        )
        target_seq_lens, target_query_lens = _metadata_rows(target_metadata)
        source_seq_lens, source_query_lens = _metadata_rows(source_metadata)
        if target_seq_lens != source_seq_lens or target_query_lens != source_query_lens:
            raise RuntimeError("ordinary and raw-source cache metadata differ")
        if target_cache.dtype != source_cache.dtype:
            raise RuntimeError("ordinary and raw-source KV cache dtypes differ")
        expected_shape = {
            "block_size": target_attention.cache_block_size,
            "num_kv_heads": target_attention.num_kv_heads,
            "head_dim": target_attention.head_dim,
        }
        validate_flash_attn_cache_shape(
            tuple(target_cache.shape),
            layout=_RUNTIME_PROFILE.flash_attn_cache_layout,
            **expected_shape,
        )
        validate_flash_attn_cache_shape(
            tuple(source_cache.shape),
            layout=_RUNTIME_PROFILE.flash_attn_cache_layout,
            **expected_shape,
        )
        if target_cache.shape != source_cache.shape:
            raise RuntimeError("ordinary and raw-source KV cache shapes differ")

        for row, (sequence_length, query_length) in enumerate(
            zip(target_seq_lens, target_query_lens, strict=True)
        ):
            if query_length < 1:
                continue
            context_length = sequence_length - query_length
            next_boundary = ((context_length // chunk_size) + 1) * chunk_size
            if sequence_length > next_boundary:
                raise RuntimeError(
                    "scheduled recurrent span crossed a chunk boundary: "
                    f"context={context_length} sequence={sequence_length} "
                    f"chunk={chunk_size}"
                )
            if sequence_length != next_boundary:
                continue

            recent_start = sequence_length - chunk_size
            target_blocks = target_metadata.block_table[row]
            source_blocks = source_metadata.block_table[row]

            local_k, local_v = _read_range(
                target_cache,
                target_blocks,
                start=recent_start,
                end=sequence_length,
                layout=_RUNTIME_PROFILE.flash_attn_cache_layout,
            )
            source_k, source_v = _read_range(
                source_cache,
                source_blocks,
                start=recent_start,
                end=sequence_length,
                layout=_RUNTIME_PROFILE.flash_attn_cache_layout,
            )
            source_k, source_v = _scale_source(target_attention, source_k, source_v)
            alpha = target_attention.prev_chunk_alphas.sigmoid().to(
                device=local_k.device, dtype=local_k.dtype
            )
            alpha = alpha.view(1, -1, 1)
            blended_k = alpha * source_k.to(local_k.dtype) + (1 - alpha) * local_k
            blended_v = alpha * source_v.to(local_v.dtype) + (1 - alpha) * local_v
            _write_range(
                target_cache,
                target_blocks,
                start=recent_start,
                end=sequence_length,
                key=blended_k,
                value=blended_v,
                layout=_RUNTIME_PROFILE.flash_attn_cache_layout,
            )

            older_start = recent_start - chunk_size
            if older_start >= 0:
                older_k, older_v = _read_range(
                    source_cache,
                    source_blocks,
                    start=older_start,
                    end=recent_start,
                    layout=_RUNTIME_PROFILE.flash_attn_cache_layout,
                )
                older_k, older_v = _scale_source(target_attention, older_k, older_v)
                _write_range(
                    target_cache,
                    target_blocks,
                    start=older_start,
                    end=recent_start,
                    key=older_k,
                    value=older_v,
                    layout=_RUNTIME_PROFILE.flash_attn_cache_layout,
                )
            rewrites += 1

    return rewrites


__all__ = ["rewrite_completed_boundaries"]
