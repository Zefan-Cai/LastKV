"""Concrete callback bridge from framework model/cache code to the backend.

The serving frameworks own allocation and model projection code.  This class
centralizes the part that must not be reimplemented independently by the
vLLM and SGLang integrations: boundary validation, request ownership, access
planning, and construction of :class:`RMarinLayerExecution`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .access_plan import (
    RMarinLayerSpec,
    RMarinRequestState,
    plan_rmarin_access,
)
from .runtime_contract import (
    RMarinBackendContractError,
    RMarinLayerExecution,
    RMarinRuntimeConstraints,
)


@dataclass(frozen=True)
class RMarinRequestSpan:
    """One request's boundary-safe query span in a framework forward."""

    request_id: str
    num_computed_tokens: int
    num_query_tokens: int
    chunk_size: int = 512
    source_initialized_end: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise RMarinBackendContractError(
                "request_id must be a non-empty string"
            )
        for name, minimum in (
            ("num_computed_tokens", 0),
            ("num_query_tokens", 1),
            ("chunk_size", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise RMarinBackendContractError(
                    f"{name} must be an integer >= {minimum}"
                )
        next_boundary = (
            (self.num_computed_tokens // self.chunk_size) + 1
        ) * self.chunk_size
        if self.num_computed_tokens + self.num_query_tokens > next_boundary:
            raise RMarinBackendContractError(
                "scheduled query span crosses an RMarin chunk boundary"
            )
        current_start = (
            self.num_computed_tokens // self.chunk_size
        ) * self.chunk_size
        source_end = (
            current_start
            if self.source_initialized_end is None
            else self.source_initialized_end
        )
        if (
            isinstance(source_end, bool)
            or not isinstance(source_end, int)
            or source_end < current_start
            or source_end > self.num_computed_tokens + self.num_query_tokens
        ):
            raise RMarinBackendContractError(
                "source_initialized_end must cover completed chunks without "
                "including future tokens"
            )

    @property
    def final_query_position(self) -> int:
        return self.num_computed_tokens + self.num_query_tokens - 1

    @property
    def resolved_source_initialized_end(self) -> int:
        if self.source_initialized_end is not None:
            return self.source_initialized_end
        return (self.num_computed_tokens // self.chunk_size) * self.chunk_size


@dataclass(frozen=True)
class RMarinLayerResources:
    """Framework-owned wrappers/caches after writes and page-table planning."""

    local_wrapper: Any
    local_kv: Any
    source_wrapper: Any | None = None
    source_kv: Any | None = None
    canonical_source_kv: Any | None = None
    source_kwargs: Mapping[str, Any] = field(default_factory=dict)
    local_kwargs: Mapping[str, Any] = field(default_factory=dict)
    source_wrapper_noncausal: bool = True
    local_wrapper_causal: bool = True
    source_kernel_representation: str = "raw_shared"
    source_transform_verified: bool = True
    canonical_source_is_raw: bool = True
    recent_blend_materialized: bool = True
    boundary_safe: bool = True
    final_layer_representations_distinct: bool = True


@dataclass
class CallbackRMarinFrameworkBindings:
    """A strict bridge usable by both thin serving-framework adapters."""

    num_layers: int
    source_layer_index: int
    runtime_constraints: RMarinRuntimeConstraints
    request_span_provider: Callable[[Any], RMarinRequestSpan]
    layer_resources_provider: Callable[..., RMarinLayerResources]
    layer_index_provider: Callable[[Any], int]
    runtime_validator: Callable[[Any], None] = lambda _: None
    forward_preparer: Callable[[Any, Any], None] = lambda _backend, _batch: None
    qk_norm_bound: bool = True
    alpha_bound: bool = True
    source_scaling_bound: bool = True
    top_layer_dual_cache_bound: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_layers, bool)
            or not isinstance(self.num_layers, int)
            or self.num_layers < 1
        ):
            raise RMarinBackendContractError("num_layers must be positive")
        if self.source_layer_index != self.num_layers - 1:
            raise RMarinBackendContractError(
                "RMarin shared source must be the final transformer layer"
            )
        if not isinstance(self.runtime_constraints, RMarinRuntimeConstraints):
            raise RMarinBackendContractError(
                "runtime_constraints must be RMarinRuntimeConstraints"
            )
        for name in (
            "request_span_provider",
            "layer_resources_provider",
            "layer_index_provider",
            "runtime_validator",
            "forward_preparer",
        ):
            if not callable(getattr(self, name)):
                raise RMarinBackendContractError(f"{name} must be callable")
        markers = (
            self.qk_norm_bound,
            self.alpha_bound,
            self.source_scaling_bound,
            self.top_layer_dual_cache_bound,
        )
        if any(marker is not True for marker in markers):
            raise RMarinBackendContractError(
                "all recurrent model-adapter binding markers must be true"
            )

    @property
    def chunk_size(self) -> int:
        return self.runtime_constraints.chunk_size

    def validate_runtime(self, model_runner: Any) -> None:
        # Construction has already validated the shared static envelope.
        self.runtime_validator(model_runner)

    def prepare_forward(self, backend: Any, forward_batch: Any) -> None:
        span = self.request_span_provider(forward_batch)
        if not isinstance(span, RMarinRequestSpan):
            raise RMarinBackendContractError(
                "request_span_provider must return RMarinRequestSpan"
            )
        if span.chunk_size != self.chunk_size:
            raise RMarinBackendContractError(
                "request chunk size differs from runtime constraints"
            )
        self.forward_preparer(backend, forward_batch)

    def prepare_layer(
        self,
        *,
        backend: Any,
        layer: Any,
        forward_batch: Any,
        query: Any,
        key: Any,
        value: Any,
        save_kv_cache: bool,
    ) -> RMarinLayerExecution:
        span = self.request_span_provider(forward_batch)
        if not isinstance(span, RMarinRequestSpan):
            raise RMarinBackendContractError(
                "request_span_provider must return RMarinRequestSpan"
            )
        if span.chunk_size != self.chunk_size:
            raise RMarinBackendContractError(
                "request chunk size differs from runtime constraints"
            )
        layer_index = self.layer_index_provider(layer)
        layer_spec = RMarinLayerSpec(
            layer_index=layer_index,
            num_layers=self.num_layers,
            source_layer_index=self.source_layer_index,
        )
        state = RMarinRequestState.for_query(
            request_id=span.request_id,
            query_position=span.final_query_position,
            chunk_size=span.chunk_size,
            source_initialized_end=span.resolved_source_initialized_end,
        )
        access_plan = plan_rmarin_access(state, layer_spec)
        resources = self.layer_resources_provider(
            backend=backend,
            layer=layer,
            forward_batch=forward_batch,
            query=query,
            key=key,
            value=value,
            save_kv_cache=save_kv_cache,
            request_span=span,
            access_plan=access_plan,
        )
        if not isinstance(resources, RMarinLayerResources):
            raise RMarinBackendContractError(
                "layer_resources_provider must return RMarinLayerResources"
            )
        return RMarinLayerExecution(
            request_ids=(span.request_id,),
            layer_index=layer_index,
            source_layer_index=self.source_layer_index,
            query_tokens=span.num_query_tokens,
            source_tokens=access_plan.older.length,
            local_tokens=access_plan.local_pass.token_range.length,
            local_wrapper=resources.local_wrapper,
            local_kv=resources.local_kv,
            access_plan=access_plan,
            source_wrapper=resources.source_wrapper,
            source_kv=resources.source_kv,
            canonical_source_kv=resources.canonical_source_kv,
            source_kwargs=resources.source_kwargs,
            local_kwargs=resources.local_kwargs,
            source_wrapper_noncausal=resources.source_wrapper_noncausal,
            local_wrapper_causal=resources.local_wrapper_causal,
            source_kernel_representation=resources.source_kernel_representation,
            source_transform_verified=resources.source_transform_verified,
            canonical_source_is_raw=resources.canonical_source_is_raw,
            recent_blend_materialized=resources.recent_blend_materialized,
            boundary_safe=resources.boundary_safe,
            final_layer_representations_distinct=(
                resources.final_layer_representations_distinct
            ),
        )


__all__ = [
    "CallbackRMarinFrameworkBindings",
    "RMarinLayerResources",
    "RMarinRequestSpan",
]
