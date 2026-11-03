"""Dependency-free RMarin KV access planning.

For a query in chunk ``n`` the exact logical history has three regions:

* chunks before ``n - 1`` use the scaled raw KV emitted by the top layer;
* completed chunk ``n - 1`` uses the learned source/local blend;
* the incomplete chunk ``n`` uses same-layer causal KV.

The first region is one FlashInfer call over the shared source cache.  The
last two regions are contiguous in a bounded per-layer cache and form a
second call.  When both calls exist, their ``(output, LSE)`` states must be
combined with ``flashinfer.cascade.merge_state``.  Concatenating attention
outputs or averaging them is not equivalent.

All token ranges are half-open.  ``query_position`` is zero based and the
current token is already present in the layer-local cache when attention is
invoked, so the local causal range ends at ``query_position + 1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


def _require_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _require_id(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, order=True)
class TokenRange:
    """A contiguous half-open logical token range."""

    start: int
    end: int

    def __post_init__(self) -> None:
        start = _require_int("range start", self.start, minimum=0)
        end = _require_int("range end", self.end, minimum=0)
        if end < start:
            raise ValueError("range end must be >= range start")

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def empty(self) -> bool:
        return self.start == self.end

    def contains(self, other: TokenRange) -> bool:
        if not isinstance(other, TokenRange):
            raise ValueError("contained range must be a TokenRange")
        return self.start <= other.start and other.end <= self.end


class CacheRole(str, Enum):
    """Physical cache family used by an attention pass."""

    SHARED_SOURCE = "shared_source"
    LAYER_LOCAL = "layer_local"


class SegmentKind(str, Enum):
    """Semantic representation stored in a logical range."""

    OLDER_SCALED_SOURCE = "older_scaled_source"
    RECENT_EFFECTIVE = "recent_effective"
    CURRENT_LOCAL = "current_local"


@dataclass(frozen=True)
class LogicalKVSegment:
    """One semantic region within an attention call."""

    request_id: str
    token_range: TokenRange
    kind: SegmentKind
    cache_role: CacheRole
    apply_source_scale: bool

    def __post_init__(self) -> None:
        _require_id("request_id", self.request_id)
        if not isinstance(self.token_range, TokenRange) or self.token_range.empty:
            raise ValueError("a logical KV segment must have a non-empty TokenRange")
        if not isinstance(self.kind, SegmentKind):
            raise ValueError("kind must be a SegmentKind")
        if not isinstance(self.cache_role, CacheRole):
            raise ValueError("cache_role must be a CacheRole")
        expected_role = (
            CacheRole.SHARED_SOURCE
            if self.kind is SegmentKind.OLDER_SCALED_SOURCE
            else CacheRole.LAYER_LOCAL
        )
        if self.cache_role is not expected_role:
            raise ValueError(f"{self.kind.value} must use {expected_role.value}")
        expected_scale = self.kind is SegmentKind.OLDER_SCALED_SOURCE
        if self.apply_source_scale is not expected_scale:
            raise ValueError(
                f"{self.kind.value} apply_source_scale must be {expected_scale}"
            )


@dataclass(frozen=True)
class AttentionPass:
    """One FlashInfer attention call returning both output and LSE."""

    name: str
    cache_role: CacheRole
    segments: tuple[LogicalKVSegment, ...]
    return_lse: bool = True

    def __post_init__(self) -> None:
        if self.name not in {"source", "local"}:
            raise ValueError("attention pass name must be source or local")
        if not isinstance(self.cache_role, CacheRole):
            raise ValueError("cache_role must be a CacheRole")
        try:
            segments = tuple(self.segments)
        except TypeError as exc:
            raise ValueError("segments must be a non-empty sequence") from exc
        if not segments or any(
            not isinstance(segment, LogicalKVSegment) for segment in segments
        ):
            raise ValueError("segments must contain LogicalKVSegment values")
        if any(segment.cache_role is not self.cache_role for segment in segments):
            raise ValueError("attention pass cannot mix cache roles")
        request_ids = {segment.request_id for segment in segments}
        if len(request_ids) != 1:
            raise ValueError("attention pass cannot mix request state")
        for left, right in zip(segments, segments[1:]):
            if left.token_range.end != right.token_range.start:
                raise ValueError("attention pass segments must be gap-free")
        if self.name == "source":
            if self.cache_role is not CacheRole.SHARED_SOURCE or len(segments) != 1:
                raise ValueError("source pass must contain one shared-source segment")
            if segments[0].kind is not SegmentKind.OLDER_SCALED_SOURCE:
                raise ValueError("source pass must read older scaled source")
        else:
            if self.cache_role is not CacheRole.LAYER_LOCAL:
                raise ValueError("local pass must use the layer-local cache")
            kinds = tuple(segment.kind for segment in segments)
            if kinds not in {
                (SegmentKind.CURRENT_LOCAL,),
                (SegmentKind.RECENT_EFFECTIVE, SegmentKind.CURRENT_LOCAL),
            }:
                raise ValueError(
                    "local pass must contain current local, optionally after recent "
                    "effective"
                )
        if self.return_lse is not True:
            raise ValueError("RMarin attention passes must return LSE")
        object.__setattr__(self, "segments", segments)

    @property
    def request_id(self) -> str:
        return self.segments[0].request_id

    @property
    def token_range(self) -> TokenRange:
        return TokenRange(
            self.segments[0].token_range.start,
            self.segments[-1].token_range.end,
        )


@dataclass(frozen=True)
class RMarinLayerSpec:
    """Layer identity and the canonical shared source layer."""

    layer_index: int
    num_layers: int
    source_layer_index: int

    def __post_init__(self) -> None:
        layer = _require_int("layer_index", self.layer_index, minimum=0)
        count = _require_int("num_layers", self.num_layers, minimum=1)
        source = _require_int(
            "source_layer_index", self.source_layer_index, minimum=0
        )
        if layer >= count:
            raise ValueError("layer_index must be smaller than num_layers")
        if source >= count:
            raise ValueError("source_layer_index must be smaller than num_layers")
        if source != count - 1:
            raise ValueError(
                "RMarin shared source must be the final layer; source-layer gaps "
                "are not supported"
            )

    @property
    def is_top_layer(self) -> bool:
        return self.layer_index == self.source_layer_index


@dataclass(frozen=True)
class RMarinRequestState:
    """Valid logical cache prefixes for one request at one query token.

    The three request identifiers intentionally duplicate ownership metadata
    carried by real serving runtimes.  A stale block table from another
    request therefore fails here instead of silently cross-contaminating KV.
    """

    request_id: str
    source_request_id: str
    local_request_id: str
    query_position: int
    chunk_size: int
    source_valid: TokenRange
    local_valid: TokenRange

    def __post_init__(self) -> None:
        request_id = _require_id("request_id", self.request_id)
        source_request_id = _require_id(
            "source_request_id", self.source_request_id
        )
        local_request_id = _require_id("local_request_id", self.local_request_id)
        if {request_id, source_request_id, local_request_id} != {request_id}:
            raise ValueError("source and local caches must belong to the same request")
        position = _require_int("query_position", self.query_position, minimum=0)
        size = _require_int("chunk_size", self.chunk_size, minimum=1)
        if not isinstance(self.source_valid, TokenRange):
            raise ValueError("source_valid must be a TokenRange")
        if not isinstance(self.local_valid, TokenRange):
            raise ValueError("local_valid must be a TokenRange")

        sequence_length = position + 1
        current_start = (position // size) * size
        recent_start = max(0, current_start - size)
        if self.source_valid.start != 0:
            raise ValueError("shared source cache must be a gap-free prefix from zero")
        if self.source_valid.end < current_start:
            raise ValueError("shared source cache is missing a completed chunk")
        if self.source_valid.end > sequence_length:
            raise ValueError("shared source cache contains future tokens")
        expected_local = TokenRange(recent_start, sequence_length)
        if self.local_valid != expected_local:
            raise ValueError(
                "layer-local cache must exactly cover recent plus current tokens "
                f"({expected_local.start}:{expected_local.end})"
            )
        if self.local_valid.length > 2 * size:
            raise ValueError("layer-local cache may retain at most 2 * chunk_size")

    @classmethod
    def for_query(
        cls,
        *,
        request_id: str,
        query_position: int,
        chunk_size: int,
        source_initialized_end: int | None = None,
    ) -> RMarinRequestState:
        """Construct the canonical state for a query.

        By default only completed chunks are assumed present in the shared
        source cache.  A top-layer adapter can pass ``query_position + 1``
        after separately writing the current raw top-layer KV.
        """

        position = _require_int("query_position", query_position, minimum=0)
        size = _require_int("chunk_size", chunk_size, minimum=1)
        current_start = (position // size) * size
        recent_start = max(0, current_start - size)
        source_end = (
            current_start
            if source_initialized_end is None
            else _require_int(
                "source_initialized_end", source_initialized_end, minimum=0
            )
        )
        return cls(
            request_id=request_id,
            source_request_id=request_id,
            local_request_id=request_id,
            query_position=position,
            chunk_size=size,
            source_valid=TokenRange(0, source_end),
            local_valid=TokenRange(recent_start, position + 1),
        )

    @property
    def sequence_length(self) -> int:
        return self.query_position + 1

    @property
    def current_chunk_start(self) -> int:
        return (self.query_position // self.chunk_size) * self.chunk_size

    @property
    def recent_chunk_start(self) -> int:
        return max(0, self.current_chunk_start - self.chunk_size)


@dataclass(frozen=True)
class TopLayerStorageContract:
    """The top layer must keep raw source and effective local in two stores."""

    raw_source_role: CacheRole = CacheRole.SHARED_SOURCE
    effective_local_role: CacheRole = CacheRole.LAYER_LOCAL
    requires_distinct_allocations: bool = True

    def __post_init__(self) -> None:
        if self.raw_source_role is not CacheRole.SHARED_SOURCE:
            raise ValueError("raw top-layer KV must use the shared source cache")
        if self.effective_local_role is not CacheRole.LAYER_LOCAL:
            raise ValueError("effective top-layer KV must use the local cache")
        if self.raw_source_role is self.effective_local_role:
            raise ValueError("raw source and effective local roles must differ")
        if self.requires_distinct_allocations is not True:
            raise ValueError("top-layer raw and effective KV must not alias")


@dataclass(frozen=True)
class RMarinAccessPlan:
    """Validated two-pass attention plan for one request and layer."""

    request_state: RMarinRequestState
    layer_spec: RMarinLayerSpec
    older: TokenRange
    recent: TokenRange
    current: TokenRange
    source_pass: AttentionPass | None
    local_pass: AttentionPass
    top_layer_storage: TopLayerStorageContract | None

    def __post_init__(self) -> None:
        if not isinstance(self.request_state, RMarinRequestState):
            raise ValueError("request_state must be RMarinRequestState")
        if not isinstance(self.layer_spec, RMarinLayerSpec):
            raise ValueError("layer_spec must be RMarinLayerSpec")
        if any(
            not isinstance(value, TokenRange)
            for value in (self.older, self.recent, self.current)
        ):
            raise ValueError("older, recent, and current must be TokenRange values")
        state = self.request_state
        expected_older = TokenRange(0, state.recent_chunk_start)
        expected_recent = TokenRange(
            state.recent_chunk_start, state.current_chunk_start
        )
        expected_current = TokenRange(
            state.current_chunk_start, state.sequence_length
        )
        if (self.older, self.recent, self.current) != (
            expected_older,
            expected_recent,
            expected_current,
        ):
            raise ValueError("access-plan ranges do not match request state")
        if self.current.empty:
            raise ValueError("current causal segment must contain the query token")
        if self.source_pass is None:
            if not self.older.empty:
                raise ValueError("non-empty older history requires a source pass")
        else:
            if self.older.empty:
                raise ValueError("empty older history may not create a source pass")
            if self.source_pass.request_id != state.request_id:
                raise ValueError("source pass request differs from request state")
            if self.source_pass.token_range != self.older:
                raise ValueError("source pass range differs from older history")
            if not state.source_valid.contains(self.source_pass.token_range):
                raise ValueError("source pass reads uninitialized or future source KV")
        if not isinstance(self.local_pass, AttentionPass):
            raise ValueError("local_pass must be an AttentionPass")
        if self.local_pass.request_id != state.request_id:
            raise ValueError("local pass request differs from request state")
        expected_local = TokenRange(self.recent.start, self.current.end)
        if self.local_pass.token_range != expected_local:
            raise ValueError("local pass must cover recent plus current history")
        if not state.local_valid.contains(self.local_pass.token_range):
            raise ValueError("local pass reads uninitialized layer-local KV")
        if self.layer_spec.is_top_layer:
            if not isinstance(self.top_layer_storage, TopLayerStorageContract):
                raise ValueError("top layer requires a dual-storage contract")
        elif self.top_layer_storage is not None:
            raise ValueError("only the top layer may declare top-layer storage")

    @property
    def requires_lse_merge(self) -> bool:
        return self.source_pass is not None

    @property
    def attention_passes(self) -> tuple[AttentionPass, ...]:
        if self.source_pass is None:
            return (self.local_pass,)
        return (self.source_pass, self.local_pass)

    @property
    def merge_operator(self) -> str | None:
        if not self.requires_lse_merge:
            return None
        return "flashinfer.cascade.merge_state"


def plan_rmarin_access(
    state: RMarinRequestState,
    layer_spec: RMarinLayerSpec,
) -> RMarinAccessPlan:
    """Create the exact source/local plan for one query and layer."""

    if not isinstance(state, RMarinRequestState):
        raise ValueError("state must be an RMarinRequestState")
    if not isinstance(layer_spec, RMarinLayerSpec):
        raise ValueError("layer_spec must be an RMarinLayerSpec")

    older = TokenRange(0, state.recent_chunk_start)
    recent = TokenRange(state.recent_chunk_start, state.current_chunk_start)
    current = TokenRange(state.current_chunk_start, state.sequence_length)

    source_pass: AttentionPass | None = None
    if not older.empty:
        source_pass = AttentionPass(
            name="source",
            cache_role=CacheRole.SHARED_SOURCE,
            segments=(
                LogicalKVSegment(
                    request_id=state.request_id,
                    token_range=older,
                    kind=SegmentKind.OLDER_SCALED_SOURCE,
                    cache_role=CacheRole.SHARED_SOURCE,
                    apply_source_scale=True,
                ),
            ),
        )

    local_segments: list[LogicalKVSegment] = []
    if not recent.empty:
        local_segments.append(
            LogicalKVSegment(
                request_id=state.request_id,
                token_range=recent,
                kind=SegmentKind.RECENT_EFFECTIVE,
                cache_role=CacheRole.LAYER_LOCAL,
                apply_source_scale=False,
            )
        )
    local_segments.append(
        LogicalKVSegment(
            request_id=state.request_id,
            token_range=current,
            kind=SegmentKind.CURRENT_LOCAL,
            cache_role=CacheRole.LAYER_LOCAL,
            apply_source_scale=False,
        )
    )
    local_pass = AttentionPass(
        name="local",
        cache_role=CacheRole.LAYER_LOCAL,
        segments=tuple(local_segments),
    )
    return RMarinAccessPlan(
        request_state=state,
        layer_spec=layer_spec,
        older=older,
        recent=recent,
        current=current,
        source_pass=source_pass,
        local_pass=local_pass,
        top_layer_storage=(
            TopLayerStorageContract() if layer_spec.is_top_layer else None
        ),
    )


def plan_batch_access(
    states: Iterable[RMarinRequestState],
    layer_spec: RMarinLayerSpec,
) -> tuple[RMarinAccessPlan, ...]:
    """Plan each request independently; never fuse their logical state."""

    if not isinstance(layer_spec, RMarinLayerSpec):
        raise ValueError("layer_spec must be an RMarinLayerSpec")
    try:
        frozen = tuple(states)
    except TypeError as exc:
        raise ValueError("states must be an iterable") from exc
    seen: set[str] = set()
    plans: list[RMarinAccessPlan] = []
    for state in frozen:
        if not isinstance(state, RMarinRequestState):
            raise ValueError("states must contain RMarinRequestState values")
        if state.request_id in seen:
            raise ValueError(f"duplicate or mixed request state: {state.request_id}")
        seen.add(state.request_id)
        plans.append(plan_rmarin_access(state, layer_spec))
    return tuple(plans)


@dataclass(frozen=True)
class CacheSlotAccounting:
    """KV-token-slot accounting, excluding heads, dimensions, and K/V factor."""

    max_sequence_tokens: int
    chunk_size: int
    num_layers: int
    shared_source_slots: int
    local_slots_per_layer: int
    local_slots_total: int
    rmarin_slots_total: int
    standard_slots_total: int

    def __post_init__(self) -> None:
        maximum = _require_int(
            "max_sequence_tokens", self.max_sequence_tokens, minimum=1
        )
        size = _require_int("chunk_size", self.chunk_size, minimum=1)
        layers = _require_int("num_layers", self.num_layers, minimum=1)
        expected_source = maximum
        expected_local_per_layer = min(maximum, 2 * size)
        expected_local_total = layers * expected_local_per_layer
        expected_total = expected_source + expected_local_total
        expected_standard = layers * maximum
        if self.shared_source_slots != expected_source:
            raise ValueError("shared_source_slots does not match full history")
        if self.local_slots_per_layer != expected_local_per_layer:
            raise ValueError("local_slots_per_layer must be bounded by 2 * chunk_size")
        if self.local_slots_total != expected_local_total:
            raise ValueError("local_slots_total is inconsistent")
        if self.rmarin_slots_total != expected_total:
            raise ValueError("rmarin_slots_total is inconsistent")
        if self.standard_slots_total != expected_standard:
            raise ValueError("standard_slots_total is inconsistent")

    @property
    def slot_reduction_factor(self) -> float:
        return self.standard_slots_total / self.rmarin_slots_total

    def estimated_bytes(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        bytes_per_element: int,
    ) -> int:
        """Return K+V bytes for the RMarin cache slot budget."""

        heads = _require_int("num_kv_heads", num_kv_heads, minimum=1)
        width = _require_int("head_dim", head_dim, minimum=1)
        element = _require_int("bytes_per_element", bytes_per_element, minimum=1)
        return self.rmarin_slots_total * 2 * heads * width * element


def account_cache_slots(
    *,
    max_sequence_tokens: int,
    chunk_size: int,
    num_layers: int,
) -> CacheSlotAccounting:
    """Account for one full shared source plus at most ``2C`` per layer."""

    maximum = _require_int(
        "max_sequence_tokens", max_sequence_tokens, minimum=1
    )
    size = _require_int("chunk_size", chunk_size, minimum=1)
    layers = _require_int("num_layers", num_layers, minimum=1)
    local_per_layer = min(maximum, 2 * size)
    local_total = layers * local_per_layer
    return CacheSlotAccounting(
        max_sequence_tokens=maximum,
        chunk_size=size,
        num_layers=layers,
        shared_source_slots=maximum,
        local_slots_per_layer=local_per_layer,
        local_slots_total=local_total,
        rmarin_slots_total=maximum + local_total,
        standard_slots_total=layers * maximum,
    )
