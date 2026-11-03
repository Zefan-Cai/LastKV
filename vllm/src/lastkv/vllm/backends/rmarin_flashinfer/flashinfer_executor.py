"""Small framework-neutral executor for two-segment FlashInfer attention."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

from .runtime_contract import (
    RMarinBackendContractError,
    RMarinLayerExecution,
)


@dataclass(frozen=True)
class AttentionState:
    """FlashInfer attention output plus its per-query/head log-sum-exp."""

    output: Any
    lse: Any


def _shape(value: Any, *, name: str) -> tuple[int, ...]:
    raw = getattr(value, "shape", None)
    if raw is None:
        raise RMarinBackendContractError(f"{name} has no tensor shape")
    try:
        result = tuple(int(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise RMarinBackendContractError(
            f"{name} has an invalid tensor shape"
        ) from exc
    return result


def _attribute(value: Any, name: str) -> Any:
    return getattr(value, name, None)


def _validate_state(
    state: AttentionState,
    *,
    query_tokens: int,
    label: str,
) -> None:
    output_shape = _shape(state.output, name=f"{label} output")
    lse_shape = _shape(state.lse, name=f"{label} LSE")
    if len(output_shape) != 3:
        raise RMarinBackendContractError(
            f"{label} output must have [tokens, heads, head_dim] shape"
        )
    if len(lse_shape) != 2:
        raise RMarinBackendContractError(
            f"{label} LSE must have [tokens, heads] shape"
        )
    if output_shape[0] != query_tokens or lse_shape[0] != query_tokens:
        raise RMarinBackendContractError(
            f"{label} state query-token count differs from its access plan"
        )
    if output_shape[:2] != lse_shape:
        raise RMarinBackendContractError(
            f"{label} output and LSE token/head dimensions differ"
        )
    output_device = _attribute(state.output, "device")
    lse_device = _attribute(state.lse, "device")
    if (
        output_device is not None
        and lse_device is not None
        and output_device != lse_device
    ):
        raise RMarinBackendContractError(
            f"{label} output and LSE devices differ"
        )
    lse_dtype = _attribute(state.lse, "dtype")
    if lse_dtype is not None and str(lse_dtype).lower() not in {
        "float32",
        "torch.float32",
    }:
        raise RMarinBackendContractError(
            f"{label} LSE must use float32"
        )


def _cache_pointers(value: Any) -> frozenset[int]:
    """Return available data pointers without importing torch."""

    if isinstance(value, (tuple, list)):
        pointers: set[int] = set()
        for item in value:
            pointers.update(_cache_pointers(item))
        return frozenset(pointers)
    pointer = getattr(value, "data_ptr", None)
    if not callable(pointer):
        return frozenset()
    result = pointer()
    if isinstance(result, bool) or not isinstance(result, int) or result <= 0:
        raise RMarinBackendContractError("KV cache exposes an invalid data_ptr")
    return frozenset({result})


def validate_independent_caches(execution: RMarinLayerExecution) -> None:
    """Reject source/local aliasing, especially at the final transformer layer."""

    if execution.source_tokens == 0:
        return
    canonical = execution.canonical_source_kv
    if canonical is None:
        raise RMarinBackendContractError("canonical raw-source KV cache is missing")
    if canonical is execution.local_kv:
        raise RMarinBackendContractError(
            "canonical raw-source and effective-local KV caches alias"
        )
    canonical_pointers = _cache_pointers(canonical)
    local_pointers = _cache_pointers(execution.local_kv)
    if (
        canonical_pointers
        and local_pointers
        and canonical_pointers & local_pointers
    ):
        raise RMarinBackendContractError(
            "canonical raw-source and effective-local KV caches share storage"
        )
    kernel_pointers = _cache_pointers(execution.source_kv)
    if kernel_pointers and local_pointers and kernel_pointers & local_pointers:
        raise RMarinBackendContractError(
            "kernel-ready source and effective-local KV caches share storage"
        )
    if execution.source_kernel_representation == "raw_shared":
        if (
            kernel_pointers
            and canonical_pointers
            and kernel_pointers != canonical_pointers
        ):
            raise RMarinBackendContractError(
                "raw_shared kernel source does not reference the canonical cache"
            )
    elif (
        kernel_pointers
        and canonical_pointers
        and kernel_pointers & canonical_pointers
    ):
        raise RMarinBackendContractError(
            "materialized target-layer source scratch aliases canonical raw source"
        )


def _invoke_return_lse(
    wrapper: Any,
    query: Any,
    kv_cache: Any,
    kwargs: dict[str, Any],
    *,
    label: str,
) -> AttentionState:
    method = getattr(wrapper, "run_return_lse", None)
    if not callable(method):
        method = getattr(wrapper, "forward_return_lse", None)
    if not callable(method):
        raise RMarinBackendContractError(
            f"{label} FlashInfer wrapper does not expose a return-LSE API"
        )
    result = method(query, kv_cache, **kwargs)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RMarinBackendContractError(
            f"{label} FlashInfer return-LSE API returned an invalid state"
        )
    return AttentionState(output=result[0], lse=result[1])


def _load_merge_state() -> Callable[[Any, Any, Any, Any], tuple[Any, Any]]:
    try:
        cascade = import_module("flashinfer.cascade")
    except ImportError as exc:
        raise RMarinBackendContractError(
            "FlashInfer cascade.merge_state is required by RMarin"
        ) from exc
    merger = getattr(cascade, "merge_state", None)
    if not callable(merger):
        raise RMarinBackendContractError(
            "installed FlashInfer has no callable cascade.merge_state"
        )
    return merger


def merge_attention_states(
    source: AttentionState,
    local: AttentionState,
    *,
    query_tokens: int,
    merger: Callable[[Any, Any, Any, Any], tuple[Any, Any]] | None = None,
) -> AttentionState:
    """Merge exact softmax states from disjoint old-source and local ranges."""

    _validate_state(source, query_tokens=query_tokens, label="old-source")
    _validate_state(local, query_tokens=query_tokens, label="local")
    if _shape(source.output, name="old-source output") != _shape(
        local.output, name="local output"
    ):
        raise RMarinBackendContractError(
            "old-source and local attention output shapes differ"
        )
    for attribute in ("dtype", "device"):
        source_value = _attribute(source.output, attribute)
        local_value = _attribute(local.output, attribute)
        if (
            source_value is not None
            and local_value is not None
            and source_value != local_value
        ):
            raise RMarinBackendContractError(
                f"old-source and local attention output {attribute}s differ"
            )
        source_lse_value = _attribute(source.lse, attribute)
        local_lse_value = _attribute(local.lse, attribute)
        if (
            source_lse_value is not None
            and local_lse_value is not None
            and source_lse_value != local_lse_value
        ):
            raise RMarinBackendContractError(
                f"old-source and local attention LSE {attribute}s differ"
            )
    merge = merger or _load_merge_state()
    result = merge(source.output, source.lse, local.output, local.lse)
    if not isinstance(result, tuple) or len(result) != 2:
        raise RMarinBackendContractError(
            "FlashInfer cascade.merge_state returned an invalid state"
        )
    merged = AttentionState(output=result[0], lse=result[1])
    _validate_state(merged, query_tokens=query_tokens, label="merged")
    return merged


def execute_rmarin_flashinfer(
    query: Any,
    execution: RMarinLayerExecution,
    *,
    merger: Callable[[Any, Any, Any, Any], tuple[Any, Any]] | None = None,
) -> AttentionState:
    """Run the local state and optional old-source state, then merge exactly."""

    query_shape = _shape(query, name="query")
    if len(query_shape) != 3 or query_shape[0] != execution.query_tokens:
        raise RMarinBackendContractError(
            "query must have [planned_tokens, heads, head_dim] shape"
        )
    validate_independent_caches(execution)

    local_kwargs = dict(execution.local_kwargs)
    local = _invoke_return_lse(
        execution.local_wrapper,
        query,
        execution.local_kv,
        local_kwargs,
        label="local",
    )
    _validate_state(local, query_tokens=execution.query_tokens, label="local")
    if execution.source_tokens == 0:
        return local

    source_kwargs = dict(execution.source_kwargs)
    source = _invoke_return_lse(
        execution.source_wrapper,
        query,
        execution.source_kv,
        source_kwargs,
        label="old-source",
    )
    return merge_attention_states(
        source,
        local,
        query_tokens=execution.query_tokens,
        merger=merger,
    )


__all__ = [
    "AttentionState",
    "execute_rmarin_flashinfer",
    "merge_attention_states",
    "validate_independent_caches",
]
