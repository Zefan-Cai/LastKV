"""Small executable oracle for the RMarin access plan.

This module operates on immutable Python tuples.  It exists to prove routing
and source/local transformations without importing a tensor or serving
runtime.  Production adapters should use :mod:`access_plan` for metadata and
compare their kernels against this reference at chunk boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .access_plan import (
    RMarinAccessPlan,
    SegmentKind,
    TokenRange,
)


def _require_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _freeze_heads(
    value: Sequence[Sequence[float]],
    *,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a non-empty sequence of heads")
    heads: list[tuple[float, ...]] = []
    width: int | None = None
    for head_index, head in enumerate(value):
        if isinstance(head, (str, bytes)) or not isinstance(head, Sequence):
            raise ValueError(f"{name}[{head_index}] must be a vector")
        vector = tuple(
            _require_finite(f"{name}[{head_index}][{index}]", item)
            for index, item in enumerate(head)
        )
        if not vector:
            raise ValueError(f"{name}[{head_index}] must not be empty")
        if width is None:
            width = len(vector)
        elif len(vector) != width:
            raise ValueError(f"{name} head dimensions must match")
        heads.append(vector)
    if not heads:
        raise ValueError(f"{name} must contain at least one head")
    return tuple(heads)


@dataclass(frozen=True)
class ReferenceKVToken:
    """One reference token shaped ``[num_kv_heads, head_dim]``."""

    key: tuple[tuple[float, ...], ...]
    value: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        key = _freeze_heads(self.key, name="key")
        value = _freeze_heads(self.value, name="value")
        if len(key) != len(value):
            raise ValueError("key and value must have the same number of heads")
        if len(key[0]) != len(value[0]):
            raise ValueError("key and value must have the same head dimension")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "value", value)

    @property
    def num_kv_heads(self) -> int:
        return len(self.key)

    @property
    def head_dim(self) -> int:
        return len(self.key[0])


@dataclass(frozen=True)
class SourceScale:
    """Layer-specific transformation of raw top-layer source KV."""

    mode: str = "none"
    value_scales: tuple[float, ...] = ()
    value_norm_weights: tuple[float, ...] = ()
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.mode not in {"none", "v", "vnorm", "vnormqk"}:
            raise ValueError("source scale mode must be none, v, vnorm, or vnormqk")
        try:
            scales = tuple(
                _require_finite(f"value_scales[{index}]", value)
                for index, value in enumerate(self.value_scales)
            )
            weights = tuple(
                _require_finite(f"value_norm_weights[{index}]", value)
                for index, value in enumerate(self.value_norm_weights)
            )
        except TypeError as exc:
            raise ValueError("source scale values must be finite sequences") from exc
        epsilon = _require_finite("epsilon", self.epsilon)
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.mode == "none" and (scales or weights):
            raise ValueError("mode=none may not contain scale parameters")
        if self.mode in {"v", "vnorm"} and (not scales or weights):
            raise ValueError(f"mode={self.mode} requires only value_scales")
        if self.mode == "vnormqk" and (scales or not weights):
            raise ValueError(
                "mode=vnormqk requires only channel-wise value_norm_weights"
            )
        object.__setattr__(self, "value_scales", scales)
        object.__setattr__(self, "value_norm_weights", weights)
        object.__setattr__(self, "epsilon", epsilon)

    def validate_shape(self, token: ReferenceKVToken) -> None:
        if self.mode in {"v", "vnorm"} and (
            len(self.value_scales) != token.num_kv_heads
        ):
            raise ValueError("value_scales length must equal num_kv_heads")
        if self.mode == "vnormqk" and (
            len(self.value_norm_weights) != token.head_dim
        ):
            raise ValueError("value_norm_weights length must equal head_dim")


def sigmoid_alpha(logit: float) -> float:
    """Stable sigmoid used by the per-head recurrent blend."""

    value = _require_finite("alpha logit", logit)
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def scale_source_token(
    raw_source: ReferenceKVToken,
    scale: SourceScale,
) -> ReferenceKVToken:
    """Apply one layer's source transformation without modifying raw KV."""

    if not isinstance(raw_source, ReferenceKVToken):
        raise ValueError("raw_source must be a ReferenceKVToken")
    if not isinstance(scale, SourceScale):
        raise ValueError("scale must be a SourceScale")
    scale.validate_shape(raw_source)
    if scale.mode == "none":
        # Return a distinct immutable representation.  The reference thereby
        # catches code that aliases the final layer's raw and effective stores.
        return ReferenceKVToken(key=raw_source.key, value=raw_source.value)

    values: list[tuple[float, ...]] = []
    for head_index, vector in enumerate(raw_source.value):
        if scale.mode in {"vnorm", "vnormqk"}:
            variance = sum(item * item for item in vector) / len(vector)
            denominator = math.sqrt(variance + scale.epsilon)
        else:
            denominator = 1.0
        if scale.mode == "vnormqk":
            values.append(
                tuple(
                    item / denominator * weight
                    for item, weight in zip(
                        vector, scale.value_norm_weights, strict=True
                    )
                )
            )
        else:
            gain = scale.value_scales[head_index]
            values.append(tuple(item / denominator * gain for item in vector))
    return ReferenceKVToken(key=raw_source.key, value=tuple(values))


def _blend_heads(
    source: tuple[tuple[float, ...], ...],
    local: tuple[tuple[float, ...], ...],
    alphas: tuple[float, ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            alpha * source_item + (1.0 - alpha) * local_item
            for source_item, local_item in zip(
                source_head, local_head, strict=True
            )
        )
        for source_head, local_head, alpha in zip(
            source, local, alphas, strict=True
        )
    )


def blend_recent_token(
    *,
    raw_source: ReferenceKVToken,
    local: ReferenceKVToken,
    alpha_logits: Sequence[float],
    source_scale: SourceScale,
) -> ReferenceKVToken:
    """Return ``alpha * scaled_source + (1 - alpha) * same_layer_local``."""

    if not isinstance(raw_source, ReferenceKVToken):
        raise ValueError("raw_source must be a ReferenceKVToken")
    if not isinstance(local, ReferenceKVToken):
        raise ValueError("local must be a ReferenceKVToken")
    if (
        raw_source.num_kv_heads,
        raw_source.head_dim,
    ) != (local.num_kv_heads, local.head_dim):
        raise ValueError("raw source and local KV shapes must match")
    if isinstance(alpha_logits, (str, bytes)) or not isinstance(
        alpha_logits, Sequence
    ):
        raise ValueError("alpha_logits must be a per-head sequence")
    alphas = tuple(sigmoid_alpha(value) for value in alpha_logits)
    if len(alphas) != local.num_kv_heads:
        raise ValueError("alpha_logits length must equal num_kv_heads")
    scaled = scale_source_token(raw_source, source_scale)
    return ReferenceKVToken(
        key=_blend_heads(scaled.key, local.key, alphas),
        value=_blend_heads(scaled.value, local.value, alphas),
    )


@dataclass(frozen=True)
class IndexedReferenceCache:
    """An immutable, request-owned logical cache inventory."""

    request_id: str
    entries: tuple[tuple[int, ReferenceKVToken], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        try:
            entries = tuple(self.entries)
        except TypeError as exc:
            raise ValueError("entries must be an indexed KV sequence") from exc
        indices: list[int] = []
        for index, entry in enumerate(entries):
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or isinstance(entry[0], bool)
                or not isinstance(entry[0], int)
                or entry[0] < 0
                or not isinstance(entry[1], ReferenceKVToken)
            ):
                raise ValueError(f"entries[{index}] must be (token, ReferenceKVToken)")
            indices.append(entry[0])
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise ValueError("cache token indices must be unique and sorted")
        object.__setattr__(self, "entries", entries)

    @classmethod
    def from_mapping(
        cls,
        *,
        request_id: str,
        entries: Mapping[int, ReferenceKVToken],
    ) -> IndexedReferenceCache:
        if not isinstance(entries, Mapping):
            raise ValueError("entries must be a mapping")
        return cls(request_id=request_id, entries=tuple(sorted(entries.items())))

    def read(self, token_range: TokenRange) -> tuple[ReferenceKVToken, ...]:
        """Read an exact contiguous range and reject missing token gaps."""

        if not isinstance(token_range, TokenRange):
            raise ValueError("token_range must be a TokenRange")
        by_index = dict(self.entries)
        missing = [
            token
            for token in range(token_range.start, token_range.end)
            if token not in by_index
        ]
        if missing:
            raise ValueError(
                "reference cache has an uninitialized token gap: "
                + ",".join(str(token) for token in missing)
            )
        return tuple(
            by_index[token] for token in range(token_range.start, token_range.end)
        )


@dataclass(frozen=True)
class TopLayerKVWrite:
    """Separate destinations for canonical raw and effective top-layer KV."""

    token_index: int
    raw_source: ReferenceKVToken
    effective_local: ReferenceKVToken

    def __post_init__(self) -> None:
        if (
            isinstance(self.token_index, bool)
            or not isinstance(self.token_index, int)
            or self.token_index < 0
        ):
            raise ValueError("token_index must be a non-negative integer")
        if not isinstance(self.raw_source, ReferenceKVToken):
            raise ValueError("raw_source must be a ReferenceKVToken")
        if not isinstance(self.effective_local, ReferenceKVToken):
            raise ValueError("effective_local must be a ReferenceKVToken")
        if self.raw_source is self.effective_local:
            raise ValueError("raw source and effective local KV must not alias")


@dataclass(frozen=True)
class ResolvedKVToken:
    """One effective token plus its audited semantic route."""

    token_index: int
    kind: SegmentKind
    effective: ReferenceKVToken


@dataclass(frozen=True)
class ResolvedHistory:
    """Effective causal history produced by the reference plan."""

    request_id: str
    tokens: tuple[ResolvedKVToken, ...]
    top_layer_writes: tuple[TopLayerKVWrite, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        indices = [token.token_index for token in self.tokens]
        if indices != list(range(len(indices))):
            raise ValueError("resolved history must be a gap-free prefix")


def resolve_reference_history(
    plan: RMarinAccessPlan,
    *,
    raw_source: IndexedReferenceCache,
    same_layer_local: IndexedReferenceCache,
    alpha_logits: Sequence[float],
    source_scale: SourceScale,
) -> ResolvedHistory:
    """Resolve a plan against sentinel/reference cache contents.

    The function intentionally reads raw source only for older/recent tokens
    and same-layer local only for recent/current tokens.  For the top layer,
    raw KV for the local region must also exist so the returned dual writes
    prove the two representations remain separately addressable.
    """

    if not isinstance(plan, RMarinAccessPlan):
        raise ValueError("plan must be an RMarinAccessPlan")
    if not isinstance(raw_source, IndexedReferenceCache):
        raise ValueError("raw_source must be an IndexedReferenceCache")
    if not isinstance(same_layer_local, IndexedReferenceCache):
        raise ValueError("same_layer_local must be an IndexedReferenceCache")
    request_ids = {
        plan.request_state.request_id,
        raw_source.request_id,
        same_layer_local.request_id,
    }
    if len(request_ids) != 1:
        raise ValueError("reference caches contain mixed request state")
    if not isinstance(source_scale, SourceScale):
        raise ValueError("source_scale must be a SourceScale")

    resolved: list[ResolvedKVToken] = []
    if not plan.older.empty:
        for offset, raw in enumerate(raw_source.read(plan.older)):
            resolved.append(
                ResolvedKVToken(
                    token_index=plan.older.start + offset,
                    kind=SegmentKind.OLDER_SCALED_SOURCE,
                    effective=scale_source_token(raw, source_scale),
                )
            )

    if not plan.recent.empty:
        raw_recent = raw_source.read(plan.recent)
        local_recent = same_layer_local.read(plan.recent)
        for offset, (raw, local) in enumerate(
            zip(raw_recent, local_recent, strict=True)
        ):
            resolved.append(
                ResolvedKVToken(
                    token_index=plan.recent.start + offset,
                    kind=SegmentKind.RECENT_EFFECTIVE,
                    effective=blend_recent_token(
                        raw_source=raw,
                        local=local,
                        alpha_logits=alpha_logits,
                        source_scale=source_scale,
                    ),
                )
            )

    local_current = same_layer_local.read(plan.current)
    for offset, local in enumerate(local_current):
        resolved.append(
            ResolvedKVToken(
                token_index=plan.current.start + offset,
                kind=SegmentKind.CURRENT_LOCAL,
                effective=local,
            )
        )

    top_layer_writes: list[TopLayerKVWrite] = []
    if plan.layer_spec.is_top_layer:
        local_range = plan.local_pass.token_range
        if not plan.request_state.source_valid.contains(local_range):
            raise ValueError(
                "top layer must preserve raw source for every effective local token"
            )
        raw_local = raw_source.read(local_range)
        effective_by_index = {
            token.token_index: token.effective for token in resolved
        }
        for offset, raw in enumerate(raw_local):
            token_index = local_range.start + offset
            top_layer_writes.append(
                TopLayerKVWrite(
                    token_index=token_index,
                    raw_source=raw,
                    effective_local=effective_by_index[token_index],
                )
            )

    return ResolvedHistory(
        request_id=plan.request_state.request_id,
        tokens=tuple(resolved),
        top_layer_writes=tuple(top_layer_writes),
    )
