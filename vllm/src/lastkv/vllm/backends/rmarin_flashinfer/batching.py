"""Multi-sequence batching for the RMarin FlashInfer vLLM backend.

This module adds batched decode/prefill support on top of the audited P0
single-sequence machinery:

* ``RMarinChunkAlignedScheduler`` — a vLLM v1 scheduler subclass that caps every
  scheduled span at the request's next recurrent chunk boundary, so no request
  span ever crosses a boundary regardless of batch composition.
* ``install_runner_capture`` — a pinned, audited patch of
  ``GPUModelRunner._prepare_inputs`` that records the post-reorder request
  order (request ids, computed tokens, scheduled tokens, finished ids) for the
  exact forward the model is about to run.
* ``RMarinBatchSpanTracker`` — derives one boundary-safe ``RMarinRequestSpan``
  per request from the capture and cross-validates it against the model's
  ``positions`` tensor, failing closed on any mismatch.
* ``RMarinSessionPool`` — one ``RMarinCompositeKVSession`` per scheduler slot,
  with request-id keyed acquire/evict and per-request boundary commits.
* ``RMarinBatchMaterializer`` / ``execute_rmarin_flashinfer_batch`` — one
  batched FlashInfer prefill-wrapper plan+run per pass per layer (ragged
  qo/kv indptr), with ``flashinfer.cascade.merge_state`` applied to the rows of
  requests that have older source history.

Semantics per request are identical to the single-sequence P0 path: the same
session write/gather/transform code runs per request; only the final
FlashInfer invocation is batched, which is mathematically equivalent because
requests never share KV.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
import os
import threading
from typing import Any

from .access_plan import (
    RMarinAccessPlan,
    RMarinLayerSpec,
    RMarinRequestState,
    TokenRange,
    plan_rmarin_access,
)
from .bindings import RMarinRequestSpan
from .composite_session import (
    RMarinCompositeKVSession,
    RMarinLayerParameters,
    TorchRMarinTensorOps,
)
from .runtime_contract import (
    RMarinBackendContractError,
    RMarinRuntimeConstraints,
    require_supported_chunk_size,
)
from .resume import REGISTRY as RESUME_REGISTRY, resume_enabled
from . import fastmode

RMARIN_MAX_BATCH_SEQS = 32
SCHEDULER_CLASS_PATH = (
    "lastkv.vllm.backends.rmarin_flashinfer.batching.RMarinChunkAlignedScheduler"
)
FUSED_ONREAD_ENV = "RMARIN_FUSED_ONREAD"


def fused_onread_enabled() -> bool:
    """Feature gate for the fused source-on-read path (default off).

    When off, this tree behaves exactly like the W4 rmarin_batch control:
    per-layer source gather + transform + scratch-page pack + per-layer
    wrapper plans.  When on:

    * the shared raw top-layer K store and per-layer transformed source V
      planes live in persistent paged slabs, appended once per chunk
      boundary (the per-token transform is elementwise, so appending at the
      boundary is value-identical to P0's per-step transform);
    * the history ("source") FlashInfer call reads those slabs zero-copy as
      a ``(k_cache, v_cache)`` tuple with one ragged plan per scheduler
      step shared by every layer;
    * the local causal call computes the same tensors and the same plan
      parameters, but the plan is built once per step and reused across
      layers, and ring gathers reuse one cached index tensor per step.
    """

    return os.environ.get(FUSED_ONREAD_ENV, "0") == "1"


def chunk_aligned_new_tokens(
    num_computed_tokens: int,
    num_new_tokens: int,
    chunk_size: int,
) -> int:
    """Cap a scheduled span at the request's next recurrent chunk boundary."""

    if num_computed_tokens < 0 or chunk_size < 1:
        raise RMarinBackendContractError(
            "chunk_aligned_new_tokens received invalid scheduling state"
        )
    if num_new_tokens < 1:
        # vLLM invokes the split hook before its own zero-token skip check;
        # pass such values through untouched.
        return num_new_tokens
    next_boundary = ((num_computed_tokens // chunk_size) + 1) * chunk_size
    allowed = next_boundary - num_computed_tokens
    return min(num_new_tokens, allowed)


def sliver_deferred_new_tokens(
    num_computed_tokens: int,
    num_new_tokens: int,
    chunk_size: int,
    num_tokens: int,
) -> int:
    """Chunk-aligned span, deferring any sliver that would split a chunk.

    2026-10-18 engine revision (G5d): with ``max_num_batched_tokens = chunk +
    max_num_seqs`` a second resident request could receive the few tokens
    left after another request's full chunk, so one recurrent chunk was
    computed as several sub-pieces.  The boundary stayed aligned but the
    numeric path differed from single-request serving and the difference
    propagated through the recurrent state (max |dlogprob| 0.14 at 60k
    tokens).  A span is now scheduled only when it reaches the request's next
    chunk boundary or is the request's final piece (which includes every
    decode step); otherwise it waits for a step with room for the whole chunk.
    """

    span = chunk_aligned_new_tokens(num_computed_tokens, num_new_tokens, chunk_size)
    if span < 1:
        return span
    next_boundary = ((num_computed_tokens // chunk_size) + 1) * chunk_size
    reaches_boundary = num_computed_tokens + span >= next_boundary
    final_piece = num_computed_tokens + span >= num_tokens
    return span if reaches_boundary or final_piece else 0


def _build_chunk_aligned_scheduler_type() -> type[Any]:
    scheduler_module = import_module("vllm.v1.core.sched.scheduler")
    base_scheduler = scheduler_module.Scheduler
    if fastmode.ASYNC_SCHED:
        try:
            base_scheduler = import_module("vllm.v1.core.sched.async_scheduler").AsyncScheduler
        except Exception:  # pragma: no cover - older vLLM
            base_scheduler = scheduler_module.Scheduler

    class RMarinChunkAlignedScheduler(base_scheduler):
        """vLLM v1 scheduler that never schedules across an RMarin boundary."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            hf_config = self.vllm_config.model_config.hf_config
            self.rmarin_chunk_size = require_supported_chunk_size(
                getattr(hf_config, "chunk_size", None)
            )
            if self.has_mamba_layers:
                raise RMarinBackendContractError(
                    "RMarin chunk-aligned scheduling cannot coexist with mamba "
                    "block-aligned scheduling"
                )
            # Reuse vLLM's existing per-request split hook (applied on both the
            # running and waiting scheduling paths) to enforce alignment.
            self.need_mamba_block_aligned_split = True
            if fastmode.PAGED_SOURCE:
                self._install_paged_source_resume()
            import weakref as _weakref

            RESUME_REGISTRY.scheduler_ref = _weakref.ref(self)

        def _install_paged_source_resume(self) -> None:
            """Paged source: session resume rides on vLLM's prefix cache.

            A new request may only start at (a) 0 or (b) the chunk boundary
            where a finished conversation with the same prefix left a ring
            snapshot, and only if vLLM still holds the cached blocks for that
            prefix (they then become the request's older-history pages).
            """

            mgr = self.kv_cache_manager
            if not getattr(mgr, "enable_caching", False):
                raise RMarinBackendContractError(
                    "RMARIN_PAGED_SOURCE requires --enable-prefix-caching (session resume reuses cached blocks)"
                )
            # the connector used to do this; without it the registry would never
            # arm snapshots (snapshot_target needs the chunk size)
            RESUME_REGISTRY.configure(int(self.rmarin_chunk_size))
            original = mgr.get_computed_blocks
            block_size = int(self.cache_config.block_size)
            kv_cache_blocks_type = import_module("vllm.v1.core.kv_cache_manager").KVCacheBlocks
            from .resume import resume_enabled

            def clamped(request: Any) -> tuple[Any, int]:
                blocks, cached = original(request)
                boundary = 0
                ids = getattr(request, "prompt_token_ids", None)
                if resume_enabled() and ids is not None:
                    boundary = RESUME_REGISTRY.register(
                        request.request_id, ids, int(request.num_prompt_tokens)
                    )
                if boundary and cached >= boundary:
                    keep = boundary // block_size
                    stale = self._evict_stale_tail(blocks, keep)
                    RESUME_REGISTRY.log(
                        "match", request_id=request.request_id, boundary=boundary,
                        cached=cached, num_prompt_tokens=int(request.num_prompt_tokens), evicted=stale,
                    )
                    return kv_cache_blocks_type(tuple(list(group[:keep]) for group in blocks.blocks)), boundary
                if boundary:
                    # a ring snapshot exists but vLLM holds no (or too few) cached
                    # pages for the prefix (evicted, or the request bypasses the
                    # prefix cache, e.g. prompt_logprobs): cold start
                    RESUME_REGISTRY.cancel(request.request_id)
                    RESUME_REGISTRY.log("cold", request_id=request.request_id, boundary=boundary, cached=cached)
                return mgr.empty_kv_cache_blocks, 0

            mgr.get_computed_blocks = clamped

        def _evict_stale_tail(self, blocks: Any, keep: int) -> int:
            """Drop cached blocks past the resume boundary from the prefix cache.

            They were written by an earlier turn's partial-chunk prefill or
            decode steps (other GEMM shapes than a full-chunk prefill), and
            vLLM's lookup returns the first-inserted block of a hash.  This
            turn recomputes those positions with full-chunk spans; evicting the
            old versions (and their duplicates) makes the fresh blocks the ones
            the next turn reuses, so a resumed turn equals a cold prefill
            bit-for-bit instead of at bf16 level.
            """

            pool = self.kv_cache_manager.block_pool
            table = pool.cached_block_hash_to_block
            ids: set[int] = set()
            for group in blocks.blocks:
                for blk in group[keep:]:
                    key = blk.block_hash
                    if key is None:
                        continue
                    entry = table._cache.get(key)
                    if entry is None:
                        ids.add(blk.block_id)
                    elif isinstance(entry, dict):
                        ids.update(entry.keys())
                    else:
                        ids.add(entry.block_id)
            if ids:
                pool.evict_blocks(ids)
            return len(ids)

        def _free_request(self, request: Any, *args: Any, **kwargs: Any) -> Any:
            if fastmode.PAGED_SOURCE:
                RESUME_REGISTRY.forget(request.request_id)
            return super()._free_request(request, *args, **kwargs)

        def add_request(self, request: Any) -> None:
            super().add_request(request)
            # Reserve the conversation's snapshot slot on arrival, not when the
            # scheduler first examines the request: under load a waiting turn
            # can sit in the queue for many steps, and an unreserved snapshot
            # slot would be handed to a cold request (OA3: 87 conversations,
            # 13 of 957 resumable turns resumed).
            from .resume import resume_enabled as _resume_enabled

            if not _resume_enabled() or RESUME_REGISTRY.chunk_size is None:
                return
            ids = getattr(request, "prompt_token_ids", None)
            if ids is None:
                return
            RESUME_REGISTRY.register(request.request_id, ids, int(request.num_prompt_tokens))

        def _mamba_block_aligned_split(
            self,
            request: Any,
            num_new_tokens: int,
            num_new_local_computed_tokens: int = 0,
            num_external_computed_tokens: int = 0,
            num_uncached_common_prefix_tokens: int = 0,
        ) -> int:
            if num_uncached_common_prefix_tokens:
                raise RMarinBackendContractError(
                    "RMarin requires prefix caching disabled"
                )
            computed = (
                request.num_computed_tokens
                + num_new_local_computed_tokens
                + num_external_computed_tokens
            )
            total = getattr(request, "num_tokens_with_spec", None)
            if not isinstance(total, int):
                total = request.num_tokens
            return sliver_deferred_new_tokens(
                computed,
                num_new_tokens,
                self.rmarin_chunk_size,
                total,
            )

    RMarinChunkAlignedScheduler.__name__ = "RMarinChunkAlignedScheduler"
    RMarinChunkAlignedScheduler.__qualname__ = "RMarinChunkAlignedScheduler"
    RMarinChunkAlignedScheduler.__module__ = __name__
    return RMarinChunkAlignedScheduler


_SCHEDULER_TYPE: type[Any] | None = None


def __getattr__(name: str) -> Any:
    if name == "RMarinChunkAlignedScheduler":
        global _SCHEDULER_TYPE
        if _SCHEDULER_TYPE is None:
            _SCHEDULER_TYPE = _build_chunk_aligned_scheduler_type()
        return _SCHEDULER_TYPE
    raise AttributeError(name)


@dataclass(frozen=True)
class RMarinBatchCapture:
    """Post-reorder request-order snapshot for exactly one model forward."""

    generation: int
    request_ids: tuple[str, ...]
    num_computed: tuple[int, ...]
    num_scheduled: tuple[int, ...]
    finished_request_ids: frozenset[str]

    def __post_init__(self) -> None:
        counts = {
            len(self.request_ids),
            len(self.num_computed),
            len(self.num_scheduled),
        }
        if len(counts) != 1 or not self.request_ids:
            raise RMarinBackendContractError(
                "runner capture request arrays are inconsistent or empty"
            )
        if len(set(self.request_ids)) != len(self.request_ids):
            raise RMarinBackendContractError("runner capture has duplicate requests")
        for value in self.num_computed:
            if value < 0:
                raise RMarinBackendContractError("negative computed-token count")
        for value in self.num_scheduled:
            if value < 1:
                raise RMarinBackendContractError("non-positive scheduled-token count")

    @property
    def total_tokens(self) -> int:
        return sum(self.num_scheduled)


class _CaptureCell:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.generation = 0
        self.value: RMarinBatchCapture | None = None
        self.installed = False

    def publish(
        self,
        *,
        request_ids: tuple[str, ...],
        num_computed: tuple[int, ...],
        num_scheduled: tuple[int, ...],
        finished_request_ids: frozenset[str],
    ) -> None:
        with self.lock:
            self.generation += 1
            self.value = RMarinBatchCapture(
                generation=self.generation,
                request_ids=request_ids,
                num_computed=num_computed,
                num_scheduled=num_scheduled,
                finished_request_ids=finished_request_ids,
            )

    def latest(self) -> RMarinBatchCapture:
        with self.lock:
            if self.value is None:
                raise RMarinBackendContractError(
                    "RMarin batch forward ran before the runner capture was "
                    "populated; install_runner_capture must patch "
                    "GPUModelRunner._prepare_inputs"
                )
            return self.value


_CAPTURE = _CaptureCell()


def install_runner_capture() -> None:
    """Wrap the V2 ``GPUModelRunner.prepare_inputs`` to publish batch order.

    vLLM 0.25 uses the V2 model runner (``vllm.v1.worker.gpu.model_runner``)
    whose ``prepare_inputs`` returns the per-forward ``InputBatch`` with
    request ids sorted decode-first (the exact order used to build the
    ``positions``/``input_ids`` buffers the model receives, including the
    synthetic kernel-warmup batches).  The wrapper captures request ids,
    pre-step computed-token counts, and per-request scheduled counts from the
    returned batch, plus finished/preempted ids from the scheduler output.
    Idempotent.
    """

    with _CAPTURE.lock:
        if _CAPTURE.installed:
            return
        runner_module = import_module("vllm.v1.worker.gpu.model_runner")
        runner_cls = runner_module.GPUModelRunner
        original = runner_cls.prepare_inputs

        def prepare_inputs_with_capture(
            self: Any,
            scheduler_output: Any,
            batch_desc: Any,
        ) -> Any:
            input_batch = original(self, scheduler_output, batch_desc)
            num_reqs = int(input_batch.num_reqs)
            request_ids = tuple(str(item) for item in input_batch.req_ids)
            if len(request_ids) != num_reqs:
                raise RMarinBackendContractError(
                    "InputBatch req_ids length differs from num_reqs"
                )
            scheduled = tuple(
                int(item) for item in input_batch.num_scheduled_tokens[:num_reqs]
            )
            computed = tuple(
                int(item) for item in input_batch.num_computed_tokens_np[:num_reqs]
            )
            if sum(scheduled) != int(input_batch.num_tokens):
                raise RMarinBackendContractError(
                    "InputBatch scheduled-token accounting is inconsistent"
                )
            finished = frozenset(
                str(item) for item in (scheduler_output.finished_req_ids or ())
            ) | frozenset(
                str(item)
                for item in (getattr(scheduler_output, "preempted_req_ids", None) or ())
            )
            _CAPTURE.publish(
                request_ids=request_ids,
                num_computed=computed,
                num_scheduled=scheduled,
                finished_request_ids=finished,
            )
            return input_batch

        runner_cls.prepare_inputs = prepare_inputs_with_capture
        if hasattr(runner_cls, "_dummy_run"):
            from . import fastpath as _fastpath

            original_dummy = runner_cls._dummy_run

            def dummy_run_with_flag(self: Any, *args: Any, **kwargs: Any) -> Any:
                _fastpath.STATE.dummy_run = True
                try:
                    return original_dummy(self, *args, **kwargs)
                finally:
                    _fastpath.STATE.dummy_run = False

            runner_cls._dummy_run = dummy_run_with_flag
        _CAPTURE.installed = True


@dataclass(frozen=True)
class RMarinBatchSpans:
    """Validated per-request spans for one forward, in batch order."""

    generation: int
    spans: tuple[RMarinRequestSpan, ...]
    qo_indptr: tuple[int, ...]
    finished_request_ids: frozenset[str]

    @property
    def total_tokens(self) -> int:
        return self.qo_indptr[-1]


def derive_batch_spans(
    capture: RMarinBatchCapture,
    positions: Sequence[int],
    chunk_size: int,
) -> RMarinBatchSpans:
    """Build boundary-safe spans and validate them against model positions."""

    expected_positions: list[int] = []
    spans: list[RMarinRequestSpan] = []
    qo_indptr = [0]
    for request_id, computed, scheduled in zip(
        capture.request_ids,
        capture.num_computed,
        capture.num_scheduled,
    ):
        expected_positions.extend(range(computed, computed + scheduled))
        spans.append(
            RMarinRequestSpan(
                request_id=request_id,
                num_computed_tokens=computed,
                num_query_tokens=scheduled,
                chunk_size=chunk_size,
            )
        )
        qo_indptr.append(qo_indptr[-1] + scheduled)
    observed = None if positions is None else list(positions)
    if observed is not None and observed != expected_positions:
        raise RMarinBackendContractError(
            "model positions differ from the captured request spans: "
            f"expected {len(expected_positions)} tokens over "
            f"{len(spans)} requests, observed head "
            f"{observed[:8]} vs {expected_positions[:8]}"
        )
    return RMarinBatchSpans(
        generation=capture.generation,
        spans=tuple(spans),
        qo_indptr=tuple(qo_indptr),
        finished_request_ids=capture.finished_request_ids,
    )


class RMarinBatchSpanTracker:
    """Stage per-request spans for one forward from the runner capture."""

    def __init__(self, *, chunk_size: int, capture_cell: _CaptureCell | None = None):
        self.chunk_size = require_supported_chunk_size(chunk_size)
        self._capture_cell = capture_cell or _CAPTURE
        self._staged: RMarinBatchSpans | None = None
        self._activated_generation = 0

    def stage_positions(self, positions: Any) -> RMarinBatchSpans:
        ndim = getattr(positions, "ndim", None)
        if ndim != 1:
            raise RMarinBackendContractError("RMarin positions must have shape [tokens]")
        capture = self._capture_cell.latest()
        if not fastmode.STRICT:
            # SPEED: trust the runner capture; no GPU->CPU sync per step.
            if capture.total_tokens > int(positions.shape[0]):
                raise RMarinBackendContractError("runner capture token count exceeds model positions")
            staged = derive_batch_spans(capture, None, self.chunk_size)
            self._staged = staged
            return staged
        values = positions.detach().to(device="cpu").tolist()
        if not isinstance(values, list) or not values:
            raise RMarinBackendContractError("RMarin forward requires at least one token")
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise RMarinBackendContractError("RMarin positions must contain integers")
        if capture.total_tokens > len(values):
            raise RMarinBackendContractError(
                "runner capture token count exceeds model positions "
                f"({capture.total_tokens} > {len(values)}); the capture is stale"
            )
        # The V2 runner may hand the model a padded positions buffer; only the
        # first num_actual_tokens rows are real.  Validate the real prefix
        # exactly; provide() separately requires the attention metadata's
        # num_actual_tokens to equal the staged total.
        staged = derive_batch_spans(
            capture,
            values[: capture.total_tokens],
            self.chunk_size,
        )
        self._staged = staged
        return staged

    def stage_from_capture(self) -> RMarinBatchSpans:
        """Derive spans from the runner capture only (fast path, no sync)."""

        staged = derive_batch_spans(self._capture_cell.latest(), None, self.chunk_size)
        self._staged = staged
        return staged

    @property
    def staged(self) -> RMarinBatchSpans:
        if self._staged is None:
            raise RMarinBackendContractError(
                "RMarin attention ran before model positions were staged"
            )
        return self._staged

    def provide(self, attention_metadata: Any, pool: "RMarinSessionPool") -> RMarinBatchSpans:
        staged = self.staged
        token_count = getattr(attention_metadata, "num_actual_tokens", None)
        if token_count != staged.total_tokens:
            raise RMarinBackendContractError(
                "vLLM attention metadata token count differs from staged spans"
            )
        if getattr(attention_metadata, "causal", None) is not True:
            raise RMarinBackendContractError(
                "RMarin requires causal vLLM attention metadata"
            )
        if getattr(attention_metadata, "use_cascade", None) is not False:
            raise RMarinBackendContractError(
                "RMarin owns the source/local LSE merge and rejects vLLM cascade"
            )
        decode_tokens = getattr(attention_metadata, "num_decode_tokens", None)
        prefill_tokens = getattr(attention_metadata, "num_prefill_tokens", None)
        if (
            not isinstance(decode_tokens, int)
            or not isinstance(prefill_tokens, int)
            or decode_tokens < 0
            or prefill_tokens < 0
            or decode_tokens + prefill_tokens != staged.total_tokens
        ):
            raise RMarinBackendContractError(
                "vLLM decode/prefill token counts differ from staged spans"
            )
        slot_mapping = getattr(attention_metadata, "slot_mapping", None)
        slot_shape = getattr(slot_mapping, "shape", None)
        try:
            slot_tokens = int(slot_shape[0])
        except (IndexError, TypeError, ValueError) as exc:
            raise RMarinBackendContractError(
                "vLLM attention metadata slot_mapping must have shape [tokens]"
            ) from exc
        if len(slot_shape) != 1 or slot_tokens != staged.total_tokens:
            raise RMarinBackendContractError(
                "vLLM slot_mapping length differs from staged spans"
            )
        if self._activated_generation != staged.generation:
            pool.prepare_forward(staged)
            self._activated_generation = staged.generation
        return staged


@dataclass(frozen=True)
class RMarinLayerTensors:
    """Per-request gathered tensors for one layer, before batched attention."""

    request_id: str
    access_plan: RMarinAccessPlan
    query_tokens: int
    local_key: Any
    local_value: Any
    source_key: Any | None
    source_value: Any | None
    source_fused: bool = False
    paged_local: bool = False

    def __post_init__(self) -> None:
        older = self.access_plan.older.length
        has_source = self.source_key is not None and self.source_value is not None
        if self.source_fused:
            if has_source:
                raise RMarinBackendContractError(
                    "fused source rows must not carry materialized source tensors"
                )
            if older < 1:
                raise RMarinBackendContractError(
                    "fused source marker requires non-empty older history"
                )
            return
        if (older > 0) != has_source:
            raise RMarinBackendContractError(
                "per-request source tensors differ from the access plan"
            )


class RMarinPooledSession(RMarinCompositeKVSession):
    """Composite session with a tensors-only layer path for batched attention."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fused_v_views: tuple[Any, ...] | None = None
        self._fused_appended_through = 0
        self._fused_ring_cache: dict[tuple[int, int, int], Any] = {}
        self.fused_slot: int | None = None
        self.paged_ring = False
        self._step_row = -1  # row of this session in the current step (paged source)
        # L2 session resume: ring snapshot at the prompt's last chunk boundary.
        self.snapshot_target: int | None = None
        self._snapshot_prompt_ids: tuple[int, ...] | None = None
        self._snap_key: list[Any] = [None] * self.num_layers
        self._snap_value: list[Any] = [None] * self.num_layers
        self._snap_all: tuple[Any, Any] | None = None
        self._pool_ref: Any = None
        self._snap_boundary: int | None = None
        self._snap_publish_pending = False
        # Lazy snapshots (pool snapshot buffers present): "ring" = the committed
        # chunk [b-C, b) is still intact in the ring itself (no copy); it is
        # spilled to a shared buffer ("buffer") only if this request goes on to
        # write position b+C, which reuses that ring half.
        self._snap_state: str | None = None
        self._snap_buf: int | None = None
        self._snapshot_info: Any = None

    # ------------------------------------------------------------------
    # Fused source-on-read support.  All fused paths are value-identical to
    # the control paths: the per-layer source transform is per-token
    # elementwise, so appending it once when a chunk enters the "older"
    # region produces exactly the tensors P0 rebuilt every step, and the
    # fast ring gather uses the same physical indices the control path
    # builds in Python.
    # ------------------------------------------------------------------

    def install_fused_source(
        self,
        *,
        source_store: Any,
        layer_value_views: Sequence[Any],
        slot: int,
    ) -> None:
        if self._persistent_reserved or self._source is not None:
            raise RMarinBackendContractError(
                "fused source planes must be installed before reservation"
            )
        views = tuple(layer_value_views)
        if len(views) != self.num_layers:
            raise RMarinBackendContractError(
                "fused source planes require one value view per layer"
            )
        self._source = source_store
        self._fused_v_views = views
        self._fused_appended_through = 0
        self.fused_slot = int(slot)

    @property
    def fused_source_ready_through(self) -> int:
        if self._fused_v_views is None:
            raise RMarinBackendContractError(
                "session has no fused source planes installed"
            )
        return self._fused_appended_through

    def reset_request(self, request_id: str) -> None:
        super().reset_request(request_id)
        self._fused_appended_through = 0
        self._fused_ring_cache.clear()
        self._discard_snapshot()

    def _drop_snapshot(self) -> None:
        """Invalidate the published/pending snapshot and free its buffer."""

        self._snap_boundary = None
        self._snap_publish_pending = False
        self._snap_state = None
        if self._snap_buf is not None and self._pool_ref is not None:
            self._pool_ref.release_snap_buf(self._snap_buf)
        self._snap_buf = None
        if self.fused_slot is not None:
            RESUME_REGISTRY.clear_snapshot(self.fused_slot)

    def _discard_snapshot(self) -> None:
        self._drop_snapshot()
        self.snapshot_target = None
        self._snapshot_prompt_ids = None
        self._snapshot_info = None

    def _poison(self, exc: BaseException) -> None:
        super()._poison(exc)
        self._discard_snapshot()

    def track_prompt(self, info: Any) -> None:
        """Arm the snapshot for a fresh request (call after prepare_forward)."""

        if info is None or self.fused_slot is None or not resume_enabled():
            self.snapshot_target = None
            self._snapshot_prompt_ids = None
            return
        target = RESUME_REGISTRY.snapshot_target(info.num_prompt_tokens)
        self.snapshot_target = target if target > 0 else None
        self._snapshot_prompt_ids = info.prompt_ids
        self._snapshot_info = info

    def _take_snapshot(self, end: int) -> None:
        size = self.chunk_size
        first = (end - size) % (2 * size)
        torch = self.tensor_ops.torch
        pool = self._pool_ref
        if self._stacked() is not None and pool is not None and getattr(pool, "snap_k", None) is not None:
            # lazy: the blended chunk [end-C, end) stays in the ring until position end+C
            self._drop_snapshot()
            self._snap_boundary = end
            self._snap_state = "ring"
            self._snap_publish_pending = True
            return
        if self._stacked() is not None:
            kview, vview = self._ring_chunk_all(TokenRange(end - size, end))
            if self._snap_all is None:
                self._snap_all = (torch.empty_like(kview), torch.empty_like(vview))
            self._snap_all[0].copy_(kview)
            self._snap_all[1].copy_(vview)
            self._snap_boundary = end
            self._snap_state = "copy"
            self._snap_publish_pending = True
            return
        for layer_index in range(self.num_layers):
            store = self._locals[layer_index]
            if store is None or store.capacity != 2 * size:
                raise RMarinBackendContractError("snapshot requires initialized 2C rings")
            key = store.key[first:first + size]
            value = store.value[first:first + size]
            if self._snap_key[layer_index] is None:
                self._snap_key[layer_index] = torch.empty_like(key)
                self._snap_value[layer_index] = torch.empty_like(value)
            self._snap_key[layer_index].copy_(key)
            self._snap_value[layer_index].copy_(value)
        self._snap_boundary = end
        self._snap_state = "copy"
        self._snap_publish_pending = True

    def _spill_snapshot(self) -> None:
        """Position b+C is about to overwrite the ring half holding [b-C, b)."""

        boundary = self._snap_boundary
        pool = self._pool_ref
        superseded = self.snapshot_target is not None and boundary is not None and self.snapshot_target > boundary
        buf = None if superseded else pool.acquire_snap_buf(self)
        if buf is None:
            # either this request will publish a later snapshot itself, or no
            # buffer could be freed: the old resume point is lost
            self._drop_snapshot()
            if not superseded:
                pool.snap_stats["dropped"] += 1
            return
        kview, vview = self._ring_chunk_all(TokenRange(boundary - self.chunk_size, boundary))
        pool.snap_k[buf].copy_(kview)
        pool.snap_v[buf].copy_(vview)
        self._snap_buf = buf
        self._snap_state = "buffer"
        pool.snap_stats["spilled"] += 1

    def _publish_snapshot(self) -> None:
        if not self._snap_publish_pending or self._snap_boundary is None:
            return
        if self._snapshot_prompt_ids is None or self.fused_slot is None:
            self._snap_publish_pending = False
            return
        if self._fused_appended_through != self._snap_boundary - self.chunk_size:
            raise RMarinBackendContractError("snapshot published before source planes caught up")
        boundary = self._snap_boundary
        prefix_hash = RESUME_REGISTRY.prefix_hash(self._snapshot_info, boundary) if self._snapshot_info is not None else None
        if prefix_hash is None:
            self._drop_snapshot()
            return
        RESUME_REGISTRY.set_snapshot(self.fused_slot, boundary, prefix_hash)
        self._snap_publish_pending = False
        RESUME_REGISTRY.log("snapshot", request_id=self.request_id, slot=self.fused_slot, boundary=boundary)

    def fdp_advance(self, span: RMarinRequestSpan) -> None:
        """Bookkeeping for one single-token step executed by the fast path."""

        position = span.num_computed_tokens
        n = span.num_query_tokens
        end = position + n
        capacity = 2 * self.chunk_size
        first = position % capacity
        for layer_index in range(self.num_layers):
            self._layer_next[layer_index] = end
            tags = self._local_tags[layer_index]
            if n == 1:
                tags[first] = position
            else:
                count = min(n, capacity - first)
                tags[first : first + count] = range(position, position + count)
                if n - count:
                    tags[0 : n - count] = range(position + count, position + n)
        self._source_next = end
        self._active_span = span
        self._active_layers = set(range(self.num_layers))

    def can_resume(self, boundary: int) -> bool:
        return (
            self._snap_boundary == boundary
            and not self._snap_publish_pending
            and self._poison_reason is None
            and self._fused_v_views is not None
            and self._fused_appended_through >= boundary - self.chunk_size
            and all(store is not None for store in self._locals)
            and (
                self._snap_state == "buffer"
                or (self._snap_state == "ring" and self._ring_half_intact(boundary))
                or (self._snap_state == "copy" and (self._snap_all is not None or self._snap_key[0] is not None))
            )
        )

    def _ring_half_intact(self, boundary: int) -> bool:
        size = self.chunk_size
        first = (boundary - size) % (2 * size)
        expect = list(range(boundary - size, boundary))
        return all(self._local_tags[layer][first:first + size] == expect for layer in (0, self.num_layers - 1))

    def adopt_resume(self, request_id: str, boundary: int, info: Any) -> None:
        """Re-own this slot for ``request_id`` continuing from ``boundary``.

        The rings are restored from the snapshot taken when ``boundary`` was
        committed; the raw source slab and transformed V planes for
        ``[0, boundary)`` are untouched since then (the slot had no other
        owner), so the state equals a cold prefill's state at ``boundary``.
        """

        if not self.can_resume(boundary):
            raise RMarinBackendContractError(
                f"session slot cannot resume at {boundary}: snapshot={self._snap_boundary}"
            )
        size = self.chunk_size
        capacity = 2 * size
        first = (boundary - size) % capacity
        other = (first + size) % capacity
        if self._snap_state in ("ring", "buffer"):
            if self._snap_state == "buffer":
                pool = self._pool_ref
                kview, vview = self._ring_chunk_all(TokenRange(boundary - size, boundary))
                kview.copy_(pool.snap_k[self._snap_buf])
                vview.copy_(pool.snap_v[self._snap_buf])
                pool.release_snap_buf(self._snap_buf)
                self._snap_buf = None
                self._snap_state = "ring"  # the ring holds it again
            for layer_index in range(self.num_layers):
                tags = self._local_tags[layer_index]
                tags[first:first + size] = range(boundary - size, boundary)
                tags[other:other + size] = [None] * size
        elif self._stacked() is not None and self._snap_all is not None:
            kview, vview = self._ring_chunk_all(TokenRange(boundary - size, boundary))
            kview.copy_(self._snap_all[0])
            vview.copy_(self._snap_all[1])
            for layer_index in range(self.num_layers):
                tags = self._local_tags[layer_index]
                tags[first:first + size] = range(boundary - size, boundary)
                tags[other:other + size] = [None] * size
        else:
            for layer_index in range(self.num_layers):
                store = self._locals[layer_index]
                store.key[first:first + size].copy_(self._snap_key[layer_index])
                store.value[first:first + size].copy_(self._snap_value[layer_index])
                tags = self._local_tags[layer_index]
                tags[first:first + size] = range(boundary - size, boundary)
                tags[other:other + size] = [None] * size
        self.request_id = request_id
        self._layer_next = [boundary] * self.num_layers
        self._source_next = boundary
        self._committed_boundary = boundary
        self._fused_appended_through = boundary - size
        self._active_span = None
        self._active_layers = set()
        self._fused_ring_cache.clear()
        self._poison_reason = None
        # keep the snapshot (still valid for the next turn); re-arm the target
        self.track_prompt(info)
        if self.snapshot_target is not None and self.snapshot_target < boundary:
            self.snapshot_target = None
        RESUME_REGISTRY.stats["adopted"] += 1

    def _write_source(self, *, logical_start: int, key: Any, value: Any) -> None:
        pool = getattr(self, "_pool_ref", None)
        if pool is not None and getattr(pool, "paged_source", False):
            raise RMarinBackendContractError("paged source: raw K/V are written by the fast path only")
        super()._write_source(logical_start=logical_start, key=key, value=value)

    def prepare_forward(self, span: RMarinRequestSpan) -> None:
        super().prepare_forward(span)
        if self._fused_v_views is not None:
            self._fused_ring_cache.clear()
            try:
                self._fused_catch_up()
                self._publish_snapshot()
            except Exception as exc:
                self._poison(exc)
                raise

    def _fused_catch_up(self) -> None:
        """Append transformed source V for chunks that entered ``older``.

        After ``_commit_previous_boundary(end)`` the upcoming queries in
        chunk ``end / C`` read older history ``[0, end - C)``.  The raw
        top-layer K/V for that whole range is already persisted, so extend
        every layer's transformed V plane to exactly ``end - C``.
        """

        target = max(0, self._committed_boundary - self.chunk_size)
        done = self._fused_appended_through
        if done == target:
            return
        pool_ref = getattr(self, "_pool_ref", None)
        if pool_ref is not None and getattr(pool_ref, "shared_v", False):
            if done > target or (target - done) % self.chunk_size:
                raise RMarinBackendContractError(f"fused source planes lost append contiguity ({done} -> {target})")
            if self._source is None or target > self._source_next:
                raise RMarinBackendContractError("fused source append precedes the raw top-layer write")
            self._fused_appended_through = target  # raw V is the plane
            return
        stacked = self._stacked()
        if stacked is not None and (target - done) % self.page_size == 0 and done % self.page_size == 0:
            if done > target or (target - done) % self.chunk_size:
                raise RMarinBackendContractError(f"fused source planes lost append contiguity ({done} -> {target})")
            if self._source is None or target > self._source_next:
                raise RMarinBackendContractError("fused source append precedes the raw top-layer write")
            pool = self._pool_ref
            page = self.page_size
            base = self.fused_slot * pool._fused_pages_per_slot
            heads = int(pool._fused_value_slab.shape[3])
            width = int(pool._fused_value_slab.shape[4])
            dest = pool._fused_value_slab[:, base + done // page : base + target // page].reshape(
                self.num_layers, target - done, heads, width
            )
            dest.copy_(self._source.value[done:target].unsqueeze(0) * stacked["scale"])
            self._fused_appended_through = target
            return
        if done > target or (target - done) % self.chunk_size:
            raise RMarinBackendContractError(
                "fused source planes lost append contiguity "
                f"({done} -> {target})"
            )
        if self._source is None or target > self._source_next:
            raise RMarinBackendContractError(
                "fused source append precedes the raw top-layer write"
            )
        raw_key = self._source.key[done:target]
        raw_value = self._source.value[done:target]
        for layer_index, parameters in enumerate(self.layer_parameters):
            _key, transformed = self.tensor_ops.transform_source(
                raw_key,
                raw_value,
                parameters,
            )
            self.tensor_ops.write(
                self._fused_v_views[layer_index],
                done,
                transformed,
            )
        self._fused_appended_through = target

    def _ring_chunk_view(self, layer_index: int, token_range: TokenRange) -> tuple[Any, Any]:
        """Contiguous ring view of one chunk-aligned range (no copy).

        Value-identical to ``_gather_ring``: the physical indices of a
        chunk-aligned range inside a 2C ring never wrap, so index_select over
        ``position % capacity`` is exactly a slice.  Every tag is still
        validated (C-speed list compare instead of a Python loop).
        """

        store = self._locals[layer_index]
        if store is None:
            raise RMarinBackendContractError("layer-local cache is uninitialized")
        capacity = store.capacity
        first = token_range.start % capacity
        if (
            token_range.start % self.chunk_size
            or capacity % self.chunk_size
            or first + token_range.length > capacity
        ):
            raise RMarinBackendContractError("ring chunk view requires a chunk-aligned range")
        tags = self._local_tags[layer_index]
        if tags[first:first + token_range.length] != list(
            range(token_range.start, token_range.end)
        ):
            raise RMarinBackendContractError(
                "layer-local ring has stale or missing logical positions"
            )
        return (
            store.key[first:first + token_range.length],
            store.value[first:first + token_range.length],
        )

    # ------------------------------------------------------------------
    # Batched (all-layer) boundary commit and catch-up for stacked rings.
    # Elementwise math is identical to the per-layer control path
    # (bf16 `alpha*source + (1-alpha)*local`, `value*scale`), so results are
    # bitwise equal; only the number of kernel launches changes (~250 -> ~8).
    # ------------------------------------------------------------------
    def _stacked(self) -> dict[str, Any] | None:
        pool = getattr(self, "_pool_ref", None)
        if pool is None or getattr(pool, "_ring_k_all", None) is None:
            return None
        if pool._stacked_params is None:
            torch = self.tensor_ops.torch
            like = pool._ring_k_all
            heads = int(like.shape[3])
            alphas = []
            scales = []
            modes = set()
            for parameters in self.layer_parameters:
                modes.add(parameters.source_mode)
                alphas.append(
                    self.tensor_ops._parameter(parameters.alpha_logits, like=like, length=heads, name="alpha_logits")
                )
                if parameters.source_mode == "v":
                    scales.append(
                        self.tensor_ops._parameter(parameters.value_scales, like=like, length=heads, name="value_scales")
                    )
            if modes != {"v"}:
                pool._stacked_params = {"supported": False}
            else:
                pool._stacked_params = {
                    "supported": True,
                    "alpha": torch.stack(alphas).sigmoid().view(self.num_layers, 1, heads, 1),
                    "scale": torch.stack(scales).view(self.num_layers, 1, heads, 1),
                }
        return pool._stacked_params if pool._stacked_params.get("supported") else None

    def _ring_chunk_all(self, token_range: TokenRange) -> tuple[Any, Any]:
        """[L, C, H, D] views of one chunk-aligned range in every layer ring."""

        pool = self._pool_ref
        size = self.chunk_size
        page = self.page_size
        first = (token_range.start % (2 * size)) // page
        base = self.fused_slot * pool._ring_pages_per_slot + first
        npages = size // page
        heads = int(pool._ring_k_all.shape[3])
        width = int(pool._ring_k_all.shape[4])
        k = pool._ring_k_all[:, base : base + npages].reshape(self.num_layers, size, heads, width)
        v = pool._ring_v_all[:, base : base + npages].reshape(self.num_layers, size, heads, width)
        return k, v

    def _commit_previous_boundary(self, end: int) -> None:
        stacked = self._stacked() if self._fused_v_views is not None else None
        if stacked is not None:
            if end == self._committed_boundary:
                return
            if end != self._committed_boundary + self.chunk_size:
                raise RMarinBackendContractError("boundary commits must be contiguous and exactly one chunk")
            if end % self.chunk_size or end > self._source_next:
                raise RMarinBackendContractError("cannot commit an incomplete recurrent chunk")
            if any(position < end for position in self._layer_next):
                raise RMarinBackendContractError("cannot commit before every layer has written the completed chunk")
            recent = TokenRange(end - self.chunk_size, end)
            if self._source is None or recent.end > self._source_next:
                raise RMarinBackendContractError("canonical raw source range is uninitialized")
            for layer_index in range(self.num_layers):
                self._ring_chunk_view(layer_index, recent)  # tag validation only
            pool = self._pool_ref
            if getattr(pool, "paged_source", False):
                # the chunk's raw top-layer K/V live in this request's vLLM blocks
                raw_key, raw_value = pool.gather_source_chunk(self._step_row, recent)
            else:
                raw_key = self._source.key[recent.start : recent.end]
                raw_value = self._source.value[recent.start : recent.end]
            kview, vview = self._ring_chunk_all(recent)
            alpha = stacked["alpha"]
            scale = stacked["scale"]
            source_key = raw_key.unsqueeze(0)
            if getattr(self._pool_ref, "ring_v_scaled", False):
                # rings hold V_l / s_l and the slab holds raw V: blend unscaled
                source_value = raw_value.unsqueeze(0)
            else:
                source_value = raw_value.unsqueeze(0) * scale
            fused = False
            if fastmode.FUSED_QKV and getattr(self._pool_ref, "ring_v_scaled", False):
                from .fused_qkv import blend_chunk_

                fused = blend_chunk_(kview, vview, raw_key.contiguous(), raw_value.contiguous(), alpha)
            if not fused:
                new_key = alpha * source_key + (1 - alpha) * kview
                new_value = alpha * source_value + (1 - alpha) * vview
                kview.copy_(new_key)
                vview.copy_(new_value)
            shadow = getattr(getattr(self._pool_ref, "_fastpath_ref", None), "shadow", None)
            if shadow is not None:
                first_tok = self.fused_slot * 2 * self.chunk_size + recent.start % (2 * self.chunk_size)
                sh = shadow[:, first_tok : first_tok + self.chunk_size]
                sh.copy_(alpha * (raw_value.unsqueeze(0) * scale) + (1 - alpha) * sh)
            self._committed_boundary = end
            if (
                self._snap_state == "ring"
                and self._snap_boundary is not None
                and end == self._snap_boundary + self.chunk_size
            ):
                self._spill_snapshot()
            if self.snapshot_target == end and self._snapshot_prompt_ids is not None:
                self._take_snapshot(end)
            return
        # SPEED-P2: fused-path commit with slices instead of Python index
        # tuples + H2D index tensors + index_select copies.  Same checks,
        # same blend arithmetic, same write order as the control path.
        if self._fused_v_views is None:
            super()._commit_previous_boundary(end)
            return
        if end == self._committed_boundary:
            return
        if end != self._committed_boundary + self.chunk_size:
            raise RMarinBackendContractError(
                "boundary commits must be contiguous and exactly one chunk"
            )
        if end % self.chunk_size or end > self._source_next:
            raise RMarinBackendContractError(
                "cannot commit an incomplete recurrent chunk"
            )
        if any(position < end for position in self._layer_next):
            raise RMarinBackendContractError(
                "cannot commit before every layer has written the completed chunk"
            )
        recent = TokenRange(end - self.chunk_size, end)
        if self._source is None or recent.end > self._source_next:
            raise RMarinBackendContractError(
                "canonical raw source range is uninitialized"
            )
        raw_key = self._source.key[recent.start:recent.end]
        raw_value = self._source.value[recent.start:recent.end]
        replacements: list[tuple[Any, Any]] = []
        for layer_index, parameters in enumerate(self.layer_parameters):
            local_key, local_value = self._ring_chunk_view(layer_index, recent)
            replacements.append(
                self.tensor_ops.blend_recent(
                    local_key,
                    local_value,
                    raw_key,
                    raw_value,
                    parameters,
                )
            )
        for layer_index, (key, value) in enumerate(replacements):
            self._write_ring(
                layer_index=layer_index,
                logical_start=recent.start,
                key=key,
                value=value,
            )
        self._committed_boundary = end
        if self.snapshot_target == end and self._snapshot_prompt_ids is not None:
            self._take_snapshot(end)

    def _write_ring(
        self,
        *,
        layer_index: int,
        logical_start: int,
        key: Any,
        value: Any,
    ) -> None:
        if self._fused_v_views is None:
            super()._write_ring(
                layer_index=layer_index,
                logical_start=logical_start,
                key=key,
                value=value,
            )
            return
        # Value-identical to the control path: gathering the contiguous
        # index ranges 0..first_count and first_count..tokens is exactly
        # slicing.  Only the redundant index-tensor construction and copy
        # are removed.
        store = self._locals[layer_index]
        if store is None:
            store = self._allocate_store(2 * self.chunk_size, key, value)
            self._locals[layer_index] = store
        tokens = int(key.shape[0])
        first_physical = logical_start % store.capacity
        first_count = min(tokens, store.capacity - first_physical)
        if first_count:
            self.tensor_ops.write(store.key, first_physical, key[:first_count])
            self.tensor_ops.write(store.value, first_physical, value[:first_count])
        if tokens - first_count:
            self.tensor_ops.write(store.key, 0, key[first_count:])
            self.tensor_ops.write(store.value, 0, value[first_count:])
        tags = self._local_tags[layer_index]
        # SPEED-P3: identical tag contents, C-speed slice assignment.
        tags[first_physical:first_physical + first_count] = range(
            logical_start, logical_start + first_count
        )
        if tokens - first_count:
            tags[0:tokens - first_count] = range(
                logical_start + first_count, logical_start + tokens
            )

    def _fused_gather_ring(
        self,
        layer_index: int,
        token_range: TokenRange,
    ) -> tuple[Any, Any]:
        """Ring gather with one cached index tensor per forward.

        The physical indices are identical to the control path's
        ``position % capacity`` enumeration; the O(range) Python tag loop
        is replaced by the arithmetic validity window (writes are enforced
        contiguous elsewhere) plus boundary tag spot-checks.
        """

        store = self._locals[layer_index]
        if store is None:
            raise RMarinBackendContractError("layer-local cache is uninitialized")
        valid_end = self._layer_next[layer_index]
        if (
            token_range.end != valid_end
            or token_range.length > store.capacity
            or token_range.start < max(0, valid_end - store.capacity)
        ):
            raise RMarinBackendContractError(
                "fused ring gather is outside the valid local window"
            )
        tags = self._local_tags[layer_index]
        for position in (token_range.start, token_range.end - 1):
            if tags[position % store.capacity] != position:
                raise RMarinBackendContractError(
                    "layer-local ring has stale or missing logical positions: "
                    f"{position}"
                )
        first = token_range.start % store.capacity
        if first + token_range.length <= store.capacity:
            # SPEED: non-wrapping window -> slice (identical to index_select)
            return (
                store.key[first:first + token_range.length],
                store.value[first:first + token_range.length],
            )
        cache_key = (token_range.start, token_range.end, store.capacity)
        index = self._fused_ring_cache.get(cache_key)
        if index is None:
            torch = self.tensor_ops.torch
            index = (
                torch.arange(
                    token_range.start,
                    token_range.end,
                    dtype=torch.long,
                    device=store.key.device,
                )
                % store.capacity
            )
            self._fused_ring_cache[cache_key] = index
        return (
            store.key.index_select(0, index),
            store.value.index_select(0, index),
        )

    def prepare_layer_tensors(
        self,
        *,
        layer_index: int,
        query: Any,
        key: Any,
        value: Any,
        request_span: RMarinRequestSpan,
        access_plan: RMarinAccessPlan,
        fused_source: bool = False,
    ) -> RMarinLayerTensors:
        """Write one layer's K/V and return gathered per-request tensors.

        This mirrors :meth:`RMarinCompositeKVSession.prepare_layer` exactly,
        except that FlashInfer materialization/planning is deferred to the
        batched materializer.  With ``fused_source`` the source gather and
        transform are skipped entirely: the batched executor reads the
        persistent paged K slab and this layer's transformed V plane.
        """

        self._require_healthy()
        index = layer_index
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise RMarinBackendContractError("layer_index must be a non-negative int")
        if index >= self.num_layers:
            raise RMarinBackendContractError("layer_index is out of range")
        if self._active_span != request_span:
            raise RMarinBackendContractError(
                "prepare_layer span differs from prepare_forward"
            )
        if index != len(self._active_layers) or index in self._active_layers:
            raise RMarinBackendContractError(
                "RMarin requires each layer exactly once in increasing order"
            )
        strict = fastmode.STRICT
        if strict:
            if not isinstance(access_plan, RMarinAccessPlan):
                raise RMarinBackendContractError("access_plan is invalid")
            if (
                access_plan.layer_spec.layer_index != index
                or access_plan.layer_spec.source_layer_index != self.source_layer_index
                or access_plan.request_state.request_id != self.request_id
                or access_plan.request_state.query_position
                != request_span.final_query_position
            ):
                raise RMarinBackendContractError(
                    "access plan does not match the active request/layer/span"
                )
        start = request_span.num_computed_tokens
        if self._layer_next[index] != start:
            raise RMarinBackendContractError(
                "layer write is not contiguous with its local ring"
            )
        try:
            if strict:
                key_descriptor = self._validate_pair(key, value, request_span.num_query_tokens)
                query_descriptor = self.tensor_ops.describe(query)
                if (
                    query_descriptor.shape[0] != request_span.num_query_tokens
                    or query_descriptor.shape[2] != key_descriptor.shape[2]
                    or query_descriptor.dtype != key_descriptor.dtype
                    or query_descriptor.device != key_descriptor.device
                ):
                    raise RMarinBackendContractError(
                        "query and K/V token/head-dimension/dtype/device are incompatible"
                    )
            self._write_ring(
                layer_index=index,
                logical_start=start,
                key=key,
                value=value,
            )
            end = start + request_span.num_query_tokens
            self._layer_next[index] = end
            if index == self.source_layer_index:
                if self._source_next != start:
                    raise RMarinBackendContractError(
                        "raw source write is not contiguous"
                    )
                self._write_source(logical_start=start, key=key, value=value)
                self._source_next = end

            if fused_source and self._fused_v_views is None:
                raise RMarinBackendContractError(
                    "fused source requested without installed source planes"
                )
            if self.paged_ring:
                # SPEED: the local pass reads the ring pages zero-copy.
                local_key = local_value = None
            elif self._fused_v_views is not None:
                local_key, local_value = self._fused_gather_ring(
                    index,
                    access_plan.local_pass.token_range,
                )
            else:
                local_key, local_value = self._gather_ring(
                    index,
                    access_plan.local_pass.token_range,
                )
            source_key = source_value = None
            if not access_plan.older.empty:
                if fused_source:
                    if self._fused_appended_through != access_plan.older.end:
                        raise RMarinBackendContractError(
                            "fused source planes do not cover the older history "
                            f"({self._fused_appended_through} != "
                            f"{access_plan.older.end})"
                        )
                else:
                    raw_key, raw_value = self._gather_source(access_plan.older)
                    source_key, source_value = self.tensor_ops.transform_source(
                        raw_key,
                        raw_value,
                        self.layer_parameters[index],
                    )
            self._active_layers.add(index)
        except Exception as exc:
            self._poison(exc)
            raise
        return RMarinLayerTensors(
            request_id=self.request_id,
            access_plan=access_plan,
            query_tokens=request_span.num_query_tokens,
            local_key=local_key,
            local_value=local_value,
            source_key=source_key,
            source_value=source_value,
            source_fused=bool(fused_source and not access_plan.older.empty),
            paged_local=self.paged_ring,
        )


@dataclass(frozen=True)
class RMarinBatchLayerExecution:
    """One layer's batched source/local FlashInfer invocation."""

    request_ids: tuple[str, ...]
    layer_index: int
    source_layer_index: int
    total_query_tokens: int
    qo_indptr: tuple[int, ...]
    source_request_rows: Any | None  # int64 row-index tensor into the batch
    local_wrapper: Any
    local_kv: Any
    source_wrapper: Any | None
    source_kv: Any | None
    source_tokens: tuple[int, ...] = ()
    source_out_scale: Any | None = None

    def __post_init__(self) -> None:
        if not self.request_ids or len(set(self.request_ids)) != len(self.request_ids):
            raise RMarinBackendContractError(
                "batched execution requires unique, non-empty request ids"
            )
        if len(self.qo_indptr) != len(self.request_ids) + 1:
            raise RMarinBackendContractError("qo_indptr length is inconsistent")
        if self.qo_indptr[0] != 0 or self.qo_indptr[-1] != self.total_query_tokens:
            raise RMarinBackendContractError("qo_indptr does not cover the batch")
        if self.local_wrapper is None or self.local_kv is None:
            raise RMarinBackendContractError(
                "every batched RMarin attention call requires a local pass"
            )
        has_source = self.source_wrapper is not None and self.source_kv is not None
        if (self.source_request_rows is not None) != has_source:
            raise RMarinBackendContractError(
                "batched source pass bindings are incomplete"
            )


class RMarinBatchMaterializer:
    """Batched scratch NHD page materializer for FlashInfer 0.6.x."""

    def __init__(
        self,
        *,
        workspace_bytes: int = 128 * 1024 * 1024,
        backend: str = "fa2",
    ) -> None:
        if workspace_bytes < 1:
            raise RMarinBackendContractError("workspace_bytes must be positive")
        self.workspace_bytes = workspace_bytes
        self.backend = backend
        try:
            self.torch = import_module("torch")
            self.flashinfer = import_module("flashinfer")
        except ImportError as exc:
            raise RMarinBackendContractError(
                "RMarinBatchMaterializer requires torch and flashinfer"
            ) from exc
        self._workspaces: dict[tuple[str, str], Any] = {}
        self._wrappers: dict[tuple[str, str], Any] = {}

    @property
    def reserved_workspace_bytes(self) -> int:
        return 2 * self.workspace_bytes

    def reserve(self, like: Any) -> None:
        if isinstance(like, self.torch.Tensor):
            device = like.device
        else:
            try:
                device = self.torch.device(like)
            except (TypeError, ValueError) as exc:
                raise RMarinBackendContractError(
                    "workspace reservation requires a torch tensor or device"
                ) from exc
        self._workspace("source", device)
        self._workspace("local", device)

    def materialize(self, **_kwargs: Any) -> Any:
        raise RMarinBackendContractError(
            "RMarinBatchMaterializer only supports materialize_batch"
        )

    def _workspace(self, role: str, device: Any) -> Any:
        key = (role, str(device))
        workspace = self._workspaces.get(key)
        if workspace is None:
            workspace = self.torch.empty(
                self.workspace_bytes,
                dtype=self.torch.uint8,
                device=device,
            )
            self._workspaces[key] = workspace
        return workspace

    def _pinned(self, role: str, name: str, length: int, dtype: Any) -> Any:
        cache = getattr(self, "_pinned_cache", None)
        if cache is None:
            cache = self._pinned_cache = {}
            self._pinned_rot = {}
        # rotate 4 buffers per (role, name): non_blocking copies of the previous
        # plan may still be pending when the next plan is staged (async scheduling)
        rot = self._pinned_rot.get((role, name), 0)
        self._pinned_rot[(role, name)] = (rot + 1) % 4
        key = (role, name, rot)
        buf = cache.get(key)
        if buf is None or int(buf.shape[0]) < length:
            buf = self.torch.zeros(max(length, 64) * 2, dtype=dtype, device="cpu").pin_memory()
            cache[key] = buf
        return buf[:length]

    def _wrapper(self, role: str, device: Any) -> Any:
        key = (role, str(device))
        wrapper = self._wrappers.get(key)
        if wrapper is not None:
            return wrapper
        workspace = self._workspace(role, device)
        factory = getattr(
            self.flashinfer,
            "BatchPrefillWithPagedKVCacheWrapper",
            None,
        )
        if not callable(factory):
            raise RMarinBackendContractError(
                "FlashInfer batch prefill paged wrapper is unavailable"
            )
        wrapper = factory(workspace, "NHD", backend=self.backend)
        self._wrappers[key] = wrapper
        return wrapper

    def _pack_pages(self, key: Any, value: Any, page_size: int) -> Any:
        torch = self.torch
        if key.shape != value.shape or key.ndim != 3:
            raise RMarinBackendContractError(
                "materialized K/V must have matching [tokens, heads, dim] shape"
            )
        tokens, heads, width = (int(item) for item in key.shape)
        pages = (tokens + page_size - 1) // page_size
        cache = torch.zeros(
            (pages, 2, page_size, heads, width),
            dtype=key.dtype,
            device=key.device,
        )
        complete_pages, tail_tokens = divmod(tokens, page_size)
        complete_tokens = complete_pages * page_size
        if complete_pages:
            cache[:complete_pages, 0].copy_(
                key[:complete_tokens].reshape(complete_pages, page_size, heads, width)
            )
            cache[:complete_pages, 1].copy_(
                value[:complete_tokens].reshape(complete_pages, page_size, heads, width)
            )
        if tail_tokens:
            cache[complete_pages, 0, :tail_tokens].copy_(key[complete_tokens:])
            cache[complete_pages, 1, :tail_tokens].copy_(value[complete_tokens:])
        return cache

    def pack_batch(
        self,
        *,
        keys: Sequence[Any],
        values: Sequence[Any],
        page_size: int,
    ) -> tuple[Any, tuple[int, ...], tuple[int, ...]]:
        """Pack per-request K/V into one paged cache without planning.

        Returns ``(cache, per_request_pages, per_request_tokens)``.  The
        packing is byte-identical to :meth:`materialize_batch`'s packing.
        """

        torch = self.torch
        if not keys or len(keys) != len(values):
            raise RMarinBackendContractError(
                "batched packing requires aligned per-request tensors"
            )
        caches = []
        page_counts = []
        token_counts = []
        for key, value in zip(keys, values):
            caches.append(self._pack_pages(key, value, page_size))
            page_counts.append(int(caches[-1].shape[0]))
            token_counts.append(int(key.shape[0]))
        cache = caches[0] if len(caches) == 1 else torch.cat(caches, dim=0)
        return cache, tuple(page_counts), tuple(token_counts)

    def plan_ragged(
        self,
        *,
        role: str,
        query_heads: int,
        head_dim: int,
        query_dtype: Any,
        kv_dtype: Any,
        device: Any,
        qo_lens: Sequence[int],
        page_counts: Sequence[int],
        last_page_lens: Sequence[int],
        page_indices: Any,
        kv_heads: int,
        page_size: int,
        causal: bool,
        sm_scale: float | None,
    ) -> Any:
        """Plan one ragged paged pass on the cached role wrapper.

        The plan arguments are exactly those :meth:`materialize_batch`
        derives; callers supply explicit ``page_indices`` so the plan can
        target either freshly packed scratch pages or the persistent fused
        slabs.
        """

        torch = self.torch
        if role not in {"source", "local"}:
            raise RMarinBackendContractError("role must be source or local")
        if causal is not (role == "local"):
            raise RMarinBackendContractError(
                "source must be non-causal and local must be causal"
            )
        if (
            not qo_lens
            or len(qo_lens) != len(page_counts)
            or len(qo_lens) != len(last_page_lens)
        ):
            raise RMarinBackendContractError("ragged plan inputs are inconsistent")
        if query_heads % kv_heads:
            raise RMarinBackendContractError(
                "FlashInfer query/KV heads are incompatible"
            )
        qo_indptr = [0]
        kv_indptr = [0]
        for qo_len, pages, last_len in zip(qo_lens, page_counts, last_page_lens):
            if qo_len < 1 or pages < 1 or not 1 <= last_len <= page_size:
                raise RMarinBackendContractError("ragged plan entry is invalid")
            qo_indptr.append(qo_indptr[-1] + int(qo_len))
            kv_indptr.append(kv_indptr[-1] + int(pages))
        if int(page_indices.shape[0]) != kv_indptr[-1]:
            raise RMarinBackendContractError("ragged plan page accounting mismatch")
        int32 = torch.int32
        plan_kwargs: dict[str, Any] = {
            "q_data_type": query_dtype,
            "kv_data_type": kv_dtype,
        }
        if sm_scale is not None:
            plan_kwargs["sm_scale"] = sm_scale
        wrapper = self._wrapper(role, device)
        if fastmode.FAST:
            # SPEED: stage on pinned host memory; FlashInfer copies non_blocking,
            # so planning never waits for queued GPU work.
            qo_t = self._pinned(role, "qo", len(qo_indptr), int32)
            kv_t = self._pinned(role, "kv", len(kv_indptr), int32)
            last_t = self._pinned(role, "last", len(last_page_lens), int32)
            qo_t.copy_(torch.tensor(qo_indptr, dtype=int32))
            kv_t.copy_(torch.tensor(kv_indptr, dtype=int32))
            last_t.copy_(torch.tensor(last_page_lens, dtype=int32))
            if page_indices.device.type == "cpu":
                pi = self._pinned(role, "pages", int(page_indices.shape[0]), int32)
                pi.copy_(page_indices.to(int32))
                page_indices = pi
            else:
                page_indices = page_indices.to(dtype=int32)
            wrapper.plan(qo_t, kv_t, page_indices, last_t, query_heads, kv_heads, head_dim, page_size, causal=causal, **plan_kwargs)
            return wrapper
        wrapper.plan(
            torch.tensor(qo_indptr, dtype=int32, device=device),
            torch.tensor(kv_indptr, dtype=int32, device=device),
            page_indices.to(dtype=int32, device=device),
            torch.tensor(last_page_lens, dtype=int32, device=device),
            query_heads,
            kv_heads,
            head_dim,
            page_size,
            causal=causal,
            **plan_kwargs,
        )
        return wrapper

    def materialize_batch(
        self,
        *,
        role: str,
        query_heads: int,
        head_dim: int,
        query_dtype: Any,
        qo_lens: Sequence[int],
        keys: Sequence[Any],
        values: Sequence[Any],
        page_size: int,
        causal: bool,
        sm_scale: float | None,
    ) -> tuple[Any, Any]:
        """Pack per-request K/V into one paged cache and plan one ragged pass."""

        torch = self.torch
        if role not in {"source", "local"}:
            raise RMarinBackendContractError("role must be source or local")
        if causal is not (role == "local"):
            raise RMarinBackendContractError(
                "source must be non-causal and local must be causal"
            )
        if not keys or len(keys) != len(values) or len(keys) != len(qo_lens):
            raise RMarinBackendContractError(
                "batched materialization requires aligned per-request tensors"
            )
        device = keys[0].device
        kv_dtype = keys[0].dtype
        kv_heads = int(keys[0].shape[1])
        caches = []
        kv_indptr = [0]
        last_page_lens = []
        qo_indptr = [0]
        for qo_len, key, value in zip(qo_lens, keys, values):
            if qo_len < 1:
                raise RMarinBackendContractError("qo length must be positive")
            tokens = int(key.shape[0])
            if tokens < 1:
                raise RMarinBackendContractError("kv length must be positive")
            if key.device != device or key.dtype != kv_dtype:
                raise RMarinBackendContractError(
                    "batched K/V tensors must share device and dtype"
                )
            if causal and qo_len > tokens:
                raise RMarinBackendContractError(
                    "causal query tokens exceed the materialized K/V range"
                )
            caches.append(self._pack_pages(key, value, page_size))
            pages = int(caches[-1].shape[0])
            kv_indptr.append(kv_indptr[-1] + pages)
            last_page_lens.append(tokens - (pages - 1) * page_size)
            qo_indptr.append(qo_indptr[-1] + int(qo_len))
        cache = caches[0] if len(caches) == 1 else torch.cat(caches, dim=0)
        total_pages = int(cache.shape[0])
        if total_pages != kv_indptr[-1]:
            raise RMarinBackendContractError("page accounting mismatch")
        int32 = torch.int32
        qo_indptr_t = torch.tensor(qo_indptr, dtype=int32, device=device)
        kv_indptr_t = torch.tensor(kv_indptr, dtype=int32, device=device)
        page_indices_t = torch.arange(total_pages, dtype=int32, device=device)
        last_page_t = torch.tensor(last_page_lens, dtype=int32, device=device)
        if query_heads % kv_heads:
            raise RMarinBackendContractError(
                "FlashInfer query/KV heads are incompatible"
            )
        plan_kwargs: dict[str, Any] = {
            "q_data_type": query_dtype,
            "kv_data_type": kv_dtype,
        }
        if sm_scale is not None:
            plan_kwargs["sm_scale"] = sm_scale
        wrapper = self._wrapper(role, device)
        wrapper.plan(
            qo_indptr_t,
            kv_indptr_t,
            page_indices_t,
            last_page_t,
            query_heads,
            kv_heads,
            head_dim,
            page_size,
            causal=causal,
            **plan_kwargs,
        )
        return wrapper, cache


def execute_rmarin_flashinfer_batch(
    query: Any,
    execution: RMarinBatchLayerExecution,
    *,
    merger: Callable[[Any, Any, Any, Any], tuple[Any, Any]],
) -> Any:
    """Run the batched local pass, optional batched source pass, and merge."""

    query_shape = getattr(query, "shape", None)
    if query_shape is None or len(query_shape) != 3:
        raise RMarinBackendContractError(
            "query must have [tokens, heads, head_dim] shape"
        )
    if int(query_shape[0]) != execution.total_query_tokens:
        raise RMarinBackendContractError(
            "query token count differs from the batched execution"
        )
    local_out, local_lse = execution.local_wrapper.run_return_lse(
        query,
        execution.local_kv,
    )
    if (
        tuple(int(item) for item in local_out.shape) != tuple(
            int(item) for item in query_shape
        )
        or len(local_lse.shape) != 2
        or int(local_lse.shape[0]) != execution.total_query_tokens
    ):
        raise RMarinBackendContractError("batched local pass returned invalid shapes")
    if str(local_lse.dtype).lower() not in {"float32", "torch.float32"}:
        raise RMarinBackendContractError("batched local LSE must be float32")
    if execution.source_wrapper is None:
        return local_out
    rows = execution.source_request_rows
    query_subset = query.index_select(0, rows)
    source_out, source_lse = execution.source_wrapper.run_return_lse(
        query_subset,
        execution.source_kv,
    )
    if execution.source_out_scale is not None:
        source_out = source_out * execution.source_out_scale
    if (
        tuple(int(item) for item in source_out.shape)
        != tuple(int(item) for item in query_subset.shape)
        or len(source_lse.shape) != 2
        or int(source_lse.shape[0]) != int(query_subset.shape[0])
    ):
        raise RMarinBackendContractError("batched source pass returned invalid shapes")
    if str(source_lse.dtype).lower() not in {"float32", "torch.float32"}:
        raise RMarinBackendContractError("batched source LSE must be float32")
    merged = merger(
        source_out,
        source_lse,
        local_out.index_select(0, rows),
        local_lse.index_select(0, rows),
    )
    if not isinstance(merged, tuple) or len(merged) != 2:
        raise RMarinBackendContractError(
            "FlashInfer cascade.merge_state returned an invalid state"
        )
    merged_out = merged[0]
    if tuple(int(item) for item in merged_out.shape) != tuple(
        int(item) for item in query_subset.shape
    ):
        raise RMarinBackendContractError("merged state shape is invalid")
    local_out.index_copy_(0, rows, merged_out.to(dtype=local_out.dtype))
    return local_out


class RMarinSessionPool:
    """Request-id keyed pool of per-slot composite sessions."""

    def __init__(
        self,
        *,
        num_slots: int,
        max_sequence_tokens: int,
        num_layers: int,
        source_layer_index: int,
        layer_parameters: Sequence[RMarinLayerParameters],
        tensor_ops: Any,
        materializer: RMarinBatchMaterializer,
        chunk_size: int,
        page_size: int,
        sm_scale: float | None = None,
        source_tokens: int = 0,
    ) -> None:
        if (
            isinstance(num_slots, bool)
            or not isinstance(num_slots, int)
            or not 1 <= num_slots <= max(RMARIN_MAX_BATCH_SEQS, fastmode.MAX_SEQS)
        ):
            raise RMarinBackendContractError(
                f"num_slots must be in [1, {max(RMARIN_MAX_BATCH_SEQS, fastmode.MAX_SEQS)}]"
            )
        self.num_slots = num_slots
        self.chunk_size = require_supported_chunk_size(chunk_size)
        self.page_size = page_size
        self.num_layers = num_layers
        self.source_layer_index = source_layer_index
        self.sm_scale = sm_scale
        self.materializer = materializer
        self.max_sequence_tokens = max_sequence_tokens
        self.fused_onread = fused_onread_enabled()
        self._fused_source_slab: Any | None = None
        self._fused_raw_value_slab: Any | None = None
        self._fused_value_slab: Any | None = None
        self._fused_pages_per_slot = 0
        self._fused_padded_tokens = 0
        self.fused_slab_bytes = 0
        self._fused_source_plan_generation = 0
        self._fused_source_receipt: tuple[Any, ...] | None = None
        self._fused_local_plan_generation = 0
        self._fused_local_receipt: tuple[Any, ...] | None = None
        self._ring_k: list[Any] | None = None  # per layer [num_slots*PAGES, P, H, D]
        self._ring_v: list[Any] | None = None
        self._ring_pages_per_slot = 0
        self._access_plan_cache: dict[tuple[str, int, int], RMarinAccessPlan] = {}
        self.sessions = tuple(
            RMarinPooledSession(
                request_id=f"rmarin-pool-slot-{slot}",
                max_sequence_tokens=max_sequence_tokens,
                num_layers=num_layers,
                source_layer_index=source_layer_index,
                layer_parameters=layer_parameters,
                tensor_ops=tensor_ops,
                materializer=materializer,
                chunk_size=chunk_size,
                page_size=page_size,
                sm_scale=sm_scale,
            )
            for slot in range(num_slots)
        )
        self._assignment: dict[str, int] = {}
        self._last_seen_generation: dict[str, int] = {}
        self._active: tuple[tuple[RMarinRequestSpan, RMarinPooledSession], ...] = ()
        self._active_generation = 0
        self._ring_k_all = None
        self._ring_v_all = None
        self._stacked_params = None
        # MERGED_ATTN: one paged K pool / V pool = [source pages | L x ring pages];
        # rings hold V_l / s_l (scales folded into the weights).
        self.merged = False
        self.ring_v_scaled = False
        self._pool_k: Any | None = None
        self._pool_v: Any | None = None
        self.merged_src_pages = 0
        self.merged_ring_stride = 0
        # PAGED_SOURCE: source pages = vLLM blocks (ids from block tables / slot mapping)
        self.paged_source = bool(fastmode.PAGED_SOURCE)
        self.source_tokens = int(source_tokens)
        self.source_pages = 0
        self.step_block_table: Any | None = None
        # shared lazy-snapshot buffers [N, L, C, H, D] (allocated with the rings)
        self.snap_k: Any | None = None
        self.snap_v: Any | None = None
        self._snap_free: list[int] = []
        self.snap_stats = {"spilled": 0, "dropped": 0, "stolen": 0}
        if self.paged_source:
            RESUME_REGISTRY.configure(int(self.chunk_size))
        for session in self.sessions:
            session._pool_ref = self

    def reserve(self, key_template: Any, value_template: Any) -> None:
        if self.fused_onread and self._fused_source_slab is None:
            self._reserve_fused_slabs(key_template)
        for session in self.sessions:
            session.reserve(key_template, value_template)

    def _reserve_fused_slabs(self, key_template: Any) -> None:
        """Allocate the shared paged K slab, raw V slab, and per-layer V planes.

        Each pool slot owns one contiguous page range, so a slot's flat
        token view of its range is exactly the token-major store the P0
        session code writes; the FlashInfer paged wrappers read the same
        memory zero-copy through global page indices.
        """

        torch = self.materializer.torch
        if not isinstance(key_template, torch.Tensor) or key_template.ndim != 3:
            raise RMarinBackendContractError(
                "fused slab reservation requires a [1, kv_heads, head_dim] template"
            )
        heads = int(key_template.shape[1])
        width = int(key_template.shape[2])
        page = self.page_size
        pages_per_slot = -(-self.max_sequence_tokens // page)
        padded_tokens = pages_per_slot * page
        total_pages = pages_per_slot * self.num_slots
        device = key_template.device
        dtype = key_template.dtype
        scratch = 1 if fastmode.PAGED_RING else 0  # one scratch page for padded rows
        self.merged = bool(fastmode.MERGED_ATTN)
        if self.paged_source:
            if not self.merged:
                raise RMarinBackendContractError("paged source requires merged attention")
            if self.source_tokens < self.max_sequence_tokens or self.source_tokens % page:
                raise RMarinBackendContractError(
                    f"paged source budget {self.source_tokens} must be a page multiple >= max_model_len {self.max_sequence_tokens}"
                )
            # block 0 is vLLM's null block (never handed to a request): our scratch
            total_pages = self.source_tokens // page
            scratch = 0
            self.source_pages = total_pages
        if self.merged:
            if not (fastmode.PAGED_RING and fastmode.SHARED_V):
                raise RMarinBackendContractError("merged attention requires paged rings and shared V")
            src_pages = total_pages + scratch
            ring_stride = (2 * self.chunk_size // page) * self.num_slots + 1
            self._pool_k = torch.zeros(
                (src_pages + self.num_layers * ring_stride, page, heads, width), dtype=dtype, device=device
            )
            self._pool_v = torch.zeros_like(self._pool_k)
            self.merged_src_pages = src_pages
            self.merged_ring_stride = ring_stride
            self.ring_v_scaled = True
            self._fused_source_slab = self._pool_k[:src_pages]
            self._fused_raw_value_slab = self._pool_v[:src_pages]
        else:
            self._fused_source_slab = torch.zeros(
                (total_pages + scratch, page, heads, width), dtype=dtype, device=device
            )
            self._fused_raw_value_slab = torch.zeros_like(self._fused_source_slab)
        self.shared_v = bool(fastmode.SHARED_V)
        if self.shared_v:
            # SPEED/MEM: no per-layer V planes; the source pass reads the raw V
            # slab for every layer and scales its output per layer/head.
            self._fused_value_slab = None
        else:
            self._fused_value_slab = torch.zeros(
                (self.num_layers, total_pages + scratch, page, heads, width),
                dtype=dtype,
                device=device,
            )
        self._fused_pages_per_slot = pages_per_slot
        self._fused_padded_tokens = padded_tokens
        self.fused_slab_bytes = sum(
            slab.numel() * slab.element_size()
            for slab in (
                (self._pool_k, self._pool_v) if self.merged
                else (self._fused_source_slab, self._fused_raw_value_slab, self._fused_value_slab)
            )
            if slab is not None
        )
        from .composite_session import RMarinKVStore

        if fastmode.PAGED_RING:
            ring_tokens = 2 * self.chunk_size
            ring_pages = ring_tokens // page
            if ring_pages * page != ring_tokens:
                raise RMarinBackendContractError("ring must be page aligned")
            self._ring_pages_per_slot = ring_pages
            # One stacked tensor per K/V so boundary commits can blend all
            # layers in a single batched op; per-layer entries are views.
            # +1 scratch page: padded cudagraph rows write/read here.
            if self.merged:
                if self.merged_ring_stride != ring_pages * self.num_slots + 1:
                    raise RMarinBackendContractError("merged pool ring stride mismatch")
                shape = (self.num_layers, self.merged_ring_stride, page, heads, width)
                self._ring_k_all = self._pool_k[self.merged_src_pages :].view(shape)
                self._ring_v_all = self._pool_v[self.merged_src_pages :].view(shape)
            else:
                self._ring_k_all = torch.zeros(
                    (self.num_layers, ring_pages * self.num_slots + 1, page, heads, width), dtype=dtype, device=device
                )
                self._ring_v_all = torch.zeros_like(self._ring_k_all)
            self._ring_k = [self._ring_k_all[layer] for layer in range(self.num_layers)]
            self._ring_v = [self._ring_v_all[layer] for layer in range(self.num_layers)]
            if resume_enabled():
                nbuf = fastmode.snapshot_buffers(self.num_slots)
                self.snap_k = torch.zeros((nbuf, self.num_layers, self.chunk_size, heads, width), dtype=dtype, device=device)
                self.snap_v = torch.zeros_like(self.snap_k)
                self._snap_free = list(range(nbuf))
                self.fused_slab_bytes += 2 * self.snap_k.numel() * self.snap_k.element_size()
            self._stacked_params: dict[str, Any] | None = None
            for slot, session in enumerate(self.sessions):
                base = slot * ring_pages
                for layer in range(self.num_layers):
                    session._locals[layer] = RMarinKVStore(
                        key=self._ring_k[layer][base : base + ring_pages].view(ring_tokens, heads, width),
                        value=self._ring_v[layer][base : base + ring_pages].view(ring_tokens, heads, width),
                        capacity=ring_tokens,
                    )
                session.paged_ring = True

        for slot, session in enumerate(self.sessions):
            if self.paged_source:
                # bookkeeping store only (capacity / _source_next); pages are vLLM's
                k_flat = self._fused_source_slab[0:1].view(page, heads, width)
                v_flat = self._fused_raw_value_slab[0:1].view(page, heads, width)
                session.install_fused_source(
                    source_store=RMarinKVStore(key=k_flat, value=v_flat, capacity=self.max_sequence_tokens),
                    layer_value_views=tuple(v_flat for _layer in range(self.num_layers)),
                    slot=slot,
                )
                continue
            base = slot * pages_per_slot
            k_flat = self._fused_source_slab[base : base + pages_per_slot].view(
                padded_tokens, heads, width
            )
            v_flat = self._fused_raw_value_slab[base : base + pages_per_slot].view(
                padded_tokens, heads, width
            )
            if self.shared_v:
                layer_views = tuple(v_flat for _layer in range(self.num_layers))
            else:
                layer_views = tuple(
                    self._fused_value_slab[
                        layer, base : base + pages_per_slot
                    ].view(padded_tokens, heads, width)
                    for layer in range(self.num_layers)
                )
            session.install_fused_source(
                source_store=RMarinKVStore(
                    key=k_flat,
                    value=v_flat,
                    capacity=self.max_sequence_tokens,
                ),
                layer_value_views=layer_views,
                slot=slot,
            )

    def memory_budget(self, **kwargs: Any) -> Any:
        return self.sessions[0].memory_budget(**kwargs)

    def shared_v_scale(self, query_heads: int) -> Any:
        """[L, 1, H_q, 1] per-layer value scale expanded to query heads (GQA)."""

        cache = getattr(self, "_shared_v_scale_cache", None)
        if cache is not None and cache[0] == query_heads:
            return cache[1]
        stacked = self.sessions[0]._stacked()
        if stacked is None:
            raise RMarinBackendContractError("shared V requires source_mode=v stacked parameters")
        scale = stacked["scale"]  # [L, 1, H_kv, 1]
        kv_heads = int(scale.shape[2])
        if query_heads % kv_heads:
            raise RMarinBackendContractError("query heads must be a multiple of kv heads")
        scale_q = scale.repeat_interleave(query_heads // kv_heads, dim=2).contiguous()
        self._shared_v_scale_cache = (query_heads, scale_q)
        return scale_q

    def acquire_snap_buf(self, requester: Any) -> int | None:
        if self._snap_free:
            return self._snap_free.pop()
        # steal the oldest idle snapshot buffer (slot neither running nor reserved)
        victims = [
            session
            for session in self.sessions
            if session is not requester
            and session._snap_buf is not None
            and RESUME_REGISTRY.owner_of(session.fused_slot) is None
            and not RESUME_REGISTRY.is_reserved(session.fused_slot)
        ]
        if not victims:
            return None
        victim = min(victims, key=lambda item: RESUME_REGISTRY.snapshot_epoch(item.fused_slot))
        victim._drop_snapshot()
        self.snap_stats["stolen"] += 1
        return self._snap_free.pop() if self._snap_free else None

    def release_snap_buf(self, buf: int) -> None:
        if buf not in self._snap_free:
            self._snap_free.append(buf)

    def gather_source_chunk(self, row: int, token_range: TokenRange) -> tuple[Any, Any]:
        """Raw top-layer K/V of a chunk-aligned range from the request's vLLM blocks."""

        if self.step_block_table is None or row < 0 or row >= int(self.step_block_table.shape[0]):
            raise RMarinBackendContractError("paged source: step block table unavailable for the commit")
        page = self.page_size
        if token_range.start % page or token_range.end % page:
            raise RMarinBackendContractError("source chunk gather requires page-aligned bounds")
        pages = self.step_block_table[row, token_range.start // page : token_range.end // page]
        heads = int(self._pool_k.shape[2])
        width = int(self._pool_k.shape[3])
        key = self._pool_k.index_select(0, pages.long()).view(token_range.length, heads, width)
        value = self._pool_v.index_select(0, pages.long()).view(token_range.length, heads, width)
        return key, value

    def invalidate_parameter_caches(self) -> None:
        """Drop cached views of the recurrent parameters (they are re-read from
        the loaded Parameters on next use).  Needed because the pool is built
        before load_weights and the tensor-ops cache keys on the Parameter
        object, not its values."""

        self._stacked_params = None
        self._shared_v_scale_cache = None
        self._top_v_scale_cache = None
        for session in self.sessions:
            cache = getattr(session.tensor_ops, "_parameter_cache", None)
            if cache is not None:
                cache.clear()

    def top_v_scale(self) -> Any:
        """[1, H_kv, 1] value scale of the source (top) layer in the KV dtype.

        With folded weights the top layer's V projection yields V / s_top; the
        shared raw slab must hold V, so writes multiply by this.
        """

        cached = getattr(self, "_top_v_scale_cache", None)
        if cached is None:
            stacked = self.sessions[0]._stacked()
            if stacked is None:
                raise RMarinBackendContractError("merged attention requires source_mode=v stacked parameters")
            cached = stacked["scale"][self.source_layer_index].contiguous()
            self._top_v_scale_cache = cached
        return cached

    def _assignment_census(self, batch_ids: frozenset[str]) -> dict[str, Any]:
        ref = getattr(RESUME_REGISTRY, "scheduler_ref", None)
        sched = ref() if ref is not None else None
        out: dict[str, Any] = {"in_batch": sum(1 for rid in self._assignment if rid in batch_ids)}
        if sched is not None:
            running = {req.request_id for req in sched.running}
            waiting = {req.request_id for req in sched.waiting}
            known = set(sched.requests.keys())
            out.update(
                running_assigned=sum(1 for rid in self._assignment if rid in running),
                waiting_assigned=sum(1 for rid in self._assignment if rid in waiting),
                gone_assigned=sum(1 for rid in self._assignment if rid not in known),
                sched_running=len(running), sched_waiting=len(waiting),
            )
        gens = sorted(self._active_generation - self._last_seen_generation.get(rid, -1) for rid in self._assignment)
        if gens:
            out["age_p50"] = gens[len(gens) // 2]
            out["age_max"] = gens[-1]
        return out

    def _reclaim_gone(self, batch_ids: frozenset[str]) -> int:
        """Free slots of ids the scheduler no longer knows (finished requests
        whose finish never reached a forward capture, and vLLM's startup
        ``_warmup_N_`` dummy requests, which filled max_num_seqs slots and were
        never reported finished)."""

        ref = getattr(RESUME_REGISTRY, "scheduler_ref", None)
        sched = ref() if ref is not None else None
        if sched is None:
            return 0
        known = sched.requests
        gone = [rid for rid in self._assignment if rid not in batch_ids and rid not in known]
        for rid in gone:
            slot = self._assignment.pop(rid)
            self._last_seen_generation.pop(rid, None)
            RESUME_REGISTRY.release(slot, rid)
        return len(gone)

    def _acquire_slot(self, request_id: str, batch_ids: frozenset[str]) -> int:
        slot = self._assignment.get(request_id)
        if slot is not None:
            return slot
        debug = os.environ.get("RMARIN_RESUME_DEBUG") == "1"
        reserved = RESUME_REGISTRY.reserved_for_others(request_id)

        def free_slots() -> list[int]:
            used = set(self._assignment.values())
            return [c for c in range(self.num_slots) if c not in used and c not in reserved]

        free = free_slots()
        if not free and self._reclaim_gone(batch_ids):
            free = free_slots()
        if free:
            # Prefer slots holding no resumable snapshot, then the oldest one.
            free.sort(key=lambda item: RESUME_REGISTRY.snapshot_epoch(item))
            if debug and RESUME_REGISTRY.snapshot_epoch(free[0]) >= 0:
                RESUME_REGISTRY.log("steal", request_id=request_id, slot=free[0], epoch=RESUME_REGISTRY.snapshot_epoch(free[0]),
                                    free=len(free), reserved=len(reserved), assigned=len(self._assignment),
                                    **self._assignment_census(batch_ids))
            self._assignment[request_id] = free[0]
            return free[0]
        # Some completed/aborted requests are never reported in a later model
        # forward, so their slots can remain assigned.  Do not use insertion
        # order here: a long-lived sequence is inserted early and may still be
        # active after many short requests have completed.  Reclaim the
        # request least recently observed in an actual forward, excluding the
        # complete current batch so no co-scheduled state can be displaced.
        # This comes before taking a slot reserved for a waiting resumable
        # turn: that would silently turn the turn into a cold prefill.
        ref = getattr(RESUME_REGISTRY, "scheduler_ref", None)
        sched = ref() if ref is not None else None

        def evict_lru(candidates: list[str]) -> int | None:
            if not candidates:
                return None
            stale_id = min(candidates, key=lambda item: self._last_seen_generation.get(item, -1))
            stale_slot = self._assignment.pop(stale_id)
            self._last_seen_generation.pop(stale_id, None)
            RESUME_REGISTRY.release(stale_slot, stale_id)
            self._assignment[request_id] = stale_slot
            return stale_slot

        if sched is not None:
            # with the scheduler visible, only non-running holders are safe to evict
            running = {req.request_id for req in sched.running}
            slot = evict_lru([rid for rid in self._assignment if rid not in batch_ids and rid not in running])
            if slot is not None:
                return slot
        used = set(self._assignment.values())
        for candidate in range(self.num_slots):
            if candidate not in used:
                if debug:
                    RESUME_REGISTRY.log("steal_reserved", request_id=request_id, slot=candidate, used=len(used),
                                        reserved=len(reserved), assigned=len(self._assignment),
                                        **self._assignment_census(batch_ids))
                self._assignment[request_id] = candidate
                return candidate
        slot = evict_lru([rid for rid in self._assignment if rid not in batch_ids])
        if slot is not None:
            return slot
        raise RMarinBackendContractError(
            "RMarin session pool is exhausted; batch exceeds pool slots"
        )

    def prepare_forward(self, staged: RMarinBatchSpans) -> None:
        batch_ids = frozenset(span.request_id for span in staged.spans)
        finished_all = set(staged.finished_request_ids) | RESUME_REGISTRY.drain_finished(batch_ids)
        for finished in finished_all:
            if finished in batch_ids:
                continue
            slot = self._assignment.pop(finished, None)
            self._last_seen_generation.pop(finished, None)
            if slot is not None:
                RESUME_REGISTRY.release(slot, finished)
        if len(batch_ids) != len(staged.spans):
            raise RMarinBackendContractError("duplicate request ids in one batch")
        active: list[tuple[RMarinRequestSpan, RMarinPooledSession]] = []
        for span in staged.spans:
            known = span.request_id in self._assignment
            if not known and span.num_computed_tokens != 0:
                resume = RESUME_REGISTRY.resume_for(span.request_id)
                if resume is None or resume[1] != span.num_computed_tokens:
                    raise RMarinBackendContractError(
                        "unknown request arrived mid-sequence; recurrent state is "
                        f"unavailable for {span.request_id!r} at "
                        f"{span.num_computed_tokens}"
                    )
                slot, boundary = resume
                stale = [rid for rid, item in self._assignment.items() if item == slot]
                for rid in stale:
                    # A request that finished in a step without a forward is never
                    # reported through the capture; the scheduler-side registry
                    # already released it at request_finished.
                    if rid in batch_ids or RESUME_REGISTRY.owner_of(slot) == rid:
                        raise RMarinBackendContractError(
                            f"resume slot {slot} is owned by active request {rid!r}"
                        )
                    self._assignment.pop(rid, None)
                    self._last_seen_generation.pop(rid, None)
                session = self.sessions[slot]
                session.adopt_resume(
                    span.request_id, boundary, RESUME_REGISTRY.info(span.request_id)
                )
                self._assignment[span.request_id] = slot
                RESUME_REGISTRY.claim(slot, span.request_id)
                RESUME_REGISTRY.log(
                    "adopted", request_id=span.request_id, slot=slot, boundary=boundary
                )
                self._last_seen_generation[span.request_id] = staged.generation
                session._step_row = len(active)
                session.prepare_forward(span)
                active.append((span, session))
                continue
            slot = self._acquire_slot(span.request_id, batch_ids)
            self._last_seen_generation[span.request_id] = staged.generation
            session = self.sessions[slot]
            session._step_row = len(active)
            session.prepare_forward(span)
            if not known:
                RESUME_REGISTRY.claim(slot, span.request_id)
                info = RESUME_REGISTRY.info(span.request_id)
                session.track_prompt(info)
                if RESUME_REGISTRY.stats["registered"] <= 3 or os.environ.get("RMARIN_RESUME_DEBUG") == "1":
                    RESUME_REGISTRY.log("fresh", request_id=span.request_id, slot=slot, have_info=info is not None,
                                        target=session.snapshot_target, fused_slot=session.fused_slot)
            active.append((span, session))
        self._active = tuple(active)
        self._active_generation = staged.generation

    def prepare_layer_batch(
        self,
        *,
        staged: RMarinBatchSpans,
        layer_index: int,
        query: Any,
        key: Any,
        value: Any,
    ) -> RMarinBatchLayerExecution:
        if staged.generation != self._active_generation or not self._active:
            raise RMarinBackendContractError(
                "prepare_layer_batch ran without a matching prepare_forward"
            )
        if self.merged:
            # rings hold V_l / s_l in this mode; only the fast path reads them.
            raise RMarinBackendContractError(
                "merged attention requires the fast path for every step (eager layer batch requested)"
            )
        if len(self._active) != len(staged.spans):
            raise RMarinBackendContractError("active session count changed mid-forward")
        layer_spec = RMarinLayerSpec(
            layer_index=layer_index,
            num_layers=self.num_layers,
            source_layer_index=self.source_layer_index,
        )
        fused = self.fused_onread
        tensors: list[RMarinLayerTensors] = []
        strict = fastmode.STRICT
        for index, (span, session) in enumerate(self._active):
            start = staged.qo_indptr[index]
            end = staged.qo_indptr[index + 1]
            if not strict:
                # SPEED: the plan only depends on the request position; reuse
                # layer 0's plan for every layer of this step.
                plan_key = (span.request_id, span.final_query_position, span.num_computed_tokens)
                access_plan = self._access_plan_cache.get(plan_key)
                if access_plan is None or access_plan.layer_spec.layer_index != 0:
                    if layer_index == 0:
                        state = RMarinRequestState.for_query(
                            request_id=span.request_id,
                            query_position=span.final_query_position,
                            chunk_size=span.chunk_size,
                            source_initialized_end=span.resolved_source_initialized_end,
                        )
                        access_plan = plan_rmarin_access(state, layer_spec)
                        if len(self._access_plan_cache) > 4096:
                            self._access_plan_cache.clear()
                        self._access_plan_cache[plan_key] = access_plan
                    else:
                        state = RMarinRequestState.for_query(
                            request_id=span.request_id,
                            query_position=span.final_query_position,
                            chunk_size=span.chunk_size,
                            source_initialized_end=span.resolved_source_initialized_end,
                        )
                        access_plan = plan_rmarin_access(state, layer_spec)
            else:
                state = RMarinRequestState.for_query(
                    request_id=span.request_id,
                    query_position=span.final_query_position,
                    chunk_size=span.chunk_size,
                    source_initialized_end=span.resolved_source_initialized_end,
                )
                access_plan = plan_rmarin_access(state, layer_spec)
            tensors.append(
                session.prepare_layer_tensors(
                    layer_index=layer_index,
                    query=query[start:end],
                    key=key[start:end],
                    value=value[start:end],
                    request_span=span,
                    access_plan=access_plan,
                    fused_source=fused,
                )
            )

        query_heads = int(query.shape[1])
        head_dim = int(query.shape[2])
        qo_lens = [entry.query_tokens for entry in tensors]
        if self._ring_k is not None:
            local_wrapper, local_cache = self._paged_local_pass(
                staged=staged,
                tensors=tensors,
                qo_lens=qo_lens,
                query=query,
                layer_index=layer_index,
            )
        elif fused:
            local_wrapper, local_cache = self._fused_local_pass(
                staged=staged,
                tensors=tensors,
                qo_lens=qo_lens,
                query=query,
            )
        else:
            local_wrapper, local_cache = self.materializer.materialize_batch(
                role="local",
                query_heads=query_heads,
                head_dim=head_dim,
                query_dtype=query.dtype,
                qo_lens=qo_lens,
                keys=[entry.local_key for entry in tensors],
                values=[entry.local_value for entry in tensors],
                page_size=self.page_size,
                causal=True,
                sm_scale=self.sm_scale,
            )
        source_entries = [
            (index, entry)
            for index, entry in enumerate(tensors)
            if (entry.source_fused if fused else entry.source_key is not None)
        ]
        source_wrapper = source_cache = source_rows = None
        source_tokens = tuple(entry.access_plan.older.length for entry in tensors)
        if source_entries:
            torch = self.materializer.torch
            row_chunks = []
            for index, _entry in source_entries:
                row_chunks.append(
                    torch.arange(
                        staged.qo_indptr[index],
                        staged.qo_indptr[index + 1],
                        dtype=torch.int64,
                        device=query.device,
                    )
                )
            source_rows = (
                row_chunks[0] if len(row_chunks) == 1 else torch.cat(row_chunks)
            )
            if fused:
                source_wrapper, source_cache = self._fused_source_pass(
                    staged=staged,
                    source_entries=source_entries,
                    qo_lens=qo_lens,
                    layer_index=layer_index,
                    query=query,
                )
            else:
                source_wrapper, source_cache = self.materializer.materialize_batch(
                    role="source",
                    query_heads=query_heads,
                    head_dim=head_dim,
                    query_dtype=query.dtype,
                    qo_lens=[qo_lens[index] for index, _entry in source_entries],
                    keys=[entry.source_key for _index, entry in source_entries],
                    values=[entry.source_value for _index, entry in source_entries],
                    page_size=self.page_size,
                    causal=False,
                    sm_scale=self.sm_scale,
                )
        return RMarinBatchLayerExecution(
            request_ids=tuple(span.request_id for span, _session in self._active),
            layer_index=layer_index,
            source_layer_index=self.source_layer_index,
            total_query_tokens=staged.total_tokens,
            qo_indptr=staged.qo_indptr,
            source_request_rows=source_rows,
            local_wrapper=local_wrapper,
            local_kv=local_cache,
            source_wrapper=source_wrapper,
            source_kv=source_cache,
            source_tokens=source_tokens,
            source_out_scale=(
                self.shared_v_scale(int(query.shape[1]))[layer_index]
                if (self.shared_v and source_wrapper is not None)
                else None
            ),
        )

    def _paged_local_pass(
        self,
        *,
        staged: RMarinBatchSpans,
        tensors: Sequence[RMarinLayerTensors],
        qo_lens: Sequence[int],
        query: Any,
        layer_index: int,
    ) -> tuple[Any, Any]:
        """Local pass reading each slot's paged ring zero-copy.

        The window [ws, p] is chunk-aligned at ``ws`` and never longer than
        the 2C ring, so its pages are ``ring_pages`` consecutive ring pages
        (modulo wrap) of the request's slot.  One plan per step is shared by
        all layers; each layer passes its own (K, V) ring tensors.
        """

        torch = self.materializer.torch
        page = self.page_size
        ring_pages = self._ring_pages_per_slot
        if self._fused_local_plan_generation != staged.generation:
            page_chunks = []
            page_counts = []
            last_page_lens = []
            for (span, session), entry in zip(self._active, tensors):
                window = entry.access_plan.local_pass.token_range
                if window.start % page or window.length < 1 or window.length > ring_pages * page:
                    raise RMarinBackendContractError("local window is not page aligned or exceeds the ring")
                npages = -(-window.length // page)
                first = (window.start // page) % ring_pages
                base = session.fused_slot * ring_pages
                idx = torch.arange(first, first + npages, dtype=torch.int32) % ring_pages + base
                page_chunks.append(idx)
                page_counts.append(npages)
                last_page_lens.append(window.length - (npages - 1) * page)
            page_indices = page_chunks[0] if len(page_chunks) == 1 else torch.cat(page_chunks)
            if not fastmode.FAST:
                page_indices = page_indices.to(device=query.device)
            for qo_len, pages, last_len in zip(qo_lens, page_counts, last_page_lens):
                if qo_len < 1 or qo_len > (pages - 1) * page + last_len:
                    raise RMarinBackendContractError("causal query tokens exceed the local window")
            self.materializer.plan_ragged(
                role="local",
                query_heads=int(query.shape[1]),
                head_dim=int(query.shape[2]),
                query_dtype=query.dtype,
                kv_dtype=self._ring_k[0].dtype,
                device=query.device,
                qo_lens=qo_lens,
                page_counts=page_counts,
                last_page_lens=last_page_lens,
                page_indices=page_indices,
                kv_heads=int(self._ring_k[0].shape[2]),
                page_size=page,
                causal=True,
                sm_scale=self.sm_scale,
            )
            self._fused_local_plan_generation = staged.generation
            self._fused_local_receipt = (tuple(qo_lens), tuple(page_counts), tuple(last_page_lens))
        elif fastmode.STRICT:
            # receipts are layer-invariant by construction; only re-check in strict mode
            pass
        return self.materializer._wrapper("local", query.device), (self._ring_k[layer_index], self._ring_v[layer_index])

    def _fused_local_pass(
        self,
        *,
        staged: RMarinBatchSpans,
        tensors: Sequence[RMarinLayerTensors],
        qo_lens: Sequence[int],
        query: Any,
    ) -> tuple[Any, Any]:
        """Pack this layer's local K/V; plan once per forward, run per layer.

        The packed values are identical to the control path.  The ragged
        plan depends only on per-request query/KV token counts, which are
        layer-invariant, so one plan per scheduler step is exact; every
        later layer verifies its packed shape against the plan receipt.
        """

        torch = self.materializer.torch
        cache, page_counts, token_counts = self.materializer.pack_batch(
            keys=[entry.local_key for entry in tensors],
            values=[entry.local_value for entry in tensors],
            page_size=self.page_size,
        )
        for qo_len, tokens in zip(qo_lens, token_counts):
            if qo_len < 1 or qo_len > tokens:
                raise RMarinBackendContractError(
                    "causal query tokens exceed the materialized K/V range"
                )
        receipt = (tuple(qo_lens), page_counts, token_counts)
        if self._fused_local_plan_generation != staged.generation:
            last_page_lens = [
                tokens - (pages - 1) * self.page_size
                for pages, tokens in zip(page_counts, token_counts)
            ]
            self.materializer.plan_ragged(
                role="local",
                query_heads=int(query.shape[1]),
                head_dim=int(query.shape[2]),
                query_dtype=query.dtype,
                kv_dtype=cache.dtype,
                device=cache.device,
                qo_lens=qo_lens,
                page_counts=page_counts,
                last_page_lens=last_page_lens,
                page_indices=torch.arange(
                    int(cache.shape[0]),
                    dtype=torch.int32,
                    device=cache.device,
                ),
                kv_heads=int(cache.shape[3]),
                page_size=self.page_size,
                causal=True,
                sm_scale=self.sm_scale,
            )
            self._fused_local_plan_generation = staged.generation
            self._fused_local_receipt = receipt
        elif self._fused_local_receipt != receipt:
            raise RMarinBackendContractError(
                "fused local plan receipt differs across layers in one forward"
            )
        return self.materializer._wrapper("local", cache.device), cache

    def _fused_source_pass(
        self,
        *,
        staged: RMarinBatchSpans,
        source_entries: Sequence[tuple[int, RMarinLayerTensors]],
        qo_lens: Sequence[int],
        layer_index: int,
        query: Any,
    ) -> tuple[Any, Any]:
        """Read the older history zero-copy from the persistent slabs.

        One ragged non-causal plan per scheduler step targets the global
        slab pages of every request with older history; each layer runs the
        planned wrapper against ``(K slab, transformed V plane[layer])``.
        No gather, no transform, no packing, no per-layer plan.
        """

        if (
            self._fused_source_slab is None
            or (self._fused_value_slab is None and not self.shared_v)
            or self._fused_pages_per_slot < 1
        ):
            raise RMarinBackendContractError("fused source slabs are not reserved")
        torch = self.materializer.torch
        page = self.page_size
        receipt_entries = []
        for index, entry in source_entries:
            session = self._active[index][1]
            older = entry.access_plan.older
            if older.start != 0 or older.end % page:
                raise RMarinBackendContractError(
                    "older history must be a chunk-aligned prefix from zero"
                )
            if session.fused_source_ready_through != older.end:
                raise RMarinBackendContractError(
                    "fused source planes do not cover a request's older history"
                )
            slot = session.fused_slot
            if slot is None or not 0 <= slot < self.num_slots:
                raise RMarinBackendContractError("fused session slot is invalid")
            receipt_entries.append((index, slot, older.end // page))
        receipt = tuple(receipt_entries)
        if self._fused_source_plan_generation != staged.generation:
            index_chunks = []
            page_counts = []
            for _index, slot, pages in receipt:
                base = slot * self._fused_pages_per_slot
                index_chunks.append(
                    torch.arange(
                        base,
                        base + pages,
                        dtype=torch.int32,
                        device=query.device,
                    )
                )
                page_counts.append(pages)
            page_indices = (
                index_chunks[0]
                if len(index_chunks) == 1
                else torch.cat(index_chunks)
            )
            self.materializer.plan_ragged(
                role="source",
                query_heads=int(query.shape[1]),
                head_dim=int(query.shape[2]),
                query_dtype=query.dtype,
                kv_dtype=self._fused_source_slab.dtype,
                device=query.device,
                qo_lens=[qo_lens[index] for index, _slot, _pages in receipt],
                page_counts=page_counts,
                last_page_lens=[page] * len(page_counts),
                page_indices=page_indices,
                kv_heads=int(self._fused_source_slab.shape[2]),
                page_size=page,
                causal=False,
                sm_scale=self.sm_scale,
            )
            self._fused_source_plan_generation = staged.generation
            self._fused_source_receipt = receipt
        elif self._fused_source_receipt != receipt:
            raise RMarinBackendContractError(
                "fused source plan receipt differs across layers in one forward"
            )
        if not 0 <= layer_index < self.num_layers:
            raise RMarinBackendContractError("fused source layer index out of range")
        if self.shared_v:
            source_cache = (self._fused_source_slab, self._fused_raw_value_slab)
        else:
            source_cache = (
                self._fused_source_slab,
                self._fused_value_slab[layer_index],
            )
        return self.materializer._wrapper("source", query.device), source_cache


@dataclass
class RMarinBatchBindings:
    """Batch-aware model/cache bridge satisfying RMarinFrameworkBindings."""

    num_layers: int
    source_layer_index: int
    runtime_constraints: RMarinRuntimeConstraints
    tracker: RMarinBatchSpanTracker
    pool: RMarinSessionPool
    layer_index_provider: Callable[[Any], int]
    runtime_validator: Callable[[Any], None] = lambda _: None
    qk_norm_bound: bool = True
    alpha_bound: bool = True
    source_scaling_bound: bool = True
    top_layer_dual_cache_bound: bool = True

    def __post_init__(self) -> None:
        if self.source_layer_index != self.num_layers - 1:
            raise RMarinBackendContractError(
                "RMarin shared source must be the final transformer layer"
            )
        if not isinstance(self.runtime_constraints, RMarinRuntimeConstraints):
            raise RMarinBackendContractError(
                "runtime_constraints must be RMarinRuntimeConstraints"
            )
        if self.pool.chunk_size != self.runtime_constraints.chunk_size:
            raise RMarinBackendContractError(
                "session pool chunk size differs from runtime constraints"
            )

    @property
    def chunk_size(self) -> int:
        return self.runtime_constraints.chunk_size

    def validate_runtime(self, model_runner: Any) -> None:
        self.runtime_validator(model_runner)

    def prepare_forward(self, backend: Any, forward_batch: Any) -> None:
        self.tracker.provide(forward_batch, self.pool)

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
    ) -> RMarinBatchLayerExecution:
        if save_kv_cache is not True:
            raise RMarinBackendContractError(
                "RMarin attention cannot run with KV cache writes disabled"
            )
        staged = self.tracker.provide(forward_batch, self.pool)
        layer_index = self.layer_index_provider(layer)
        kv_heads = None
        key_shape = getattr(key, "shape", None)
        if key_shape is None or len(key_shape) != 3:
            raise RMarinBackendContractError(
                "RMarin batch bindings require [tokens, kv_heads, head_dim] K/V"
            )
        del kv_heads
        return self.pool.prepare_layer_batch(
            staged=staged,
            layer_index=layer_index,
            query=query,
            key=key,
            value=value,
        )


__all__ = [
    "RMARIN_MAX_BATCH_SEQS",
    "SCHEDULER_CLASS_PATH",
    "RMarinBatchBindings",
    "RMarinBatchCapture",
    "RMarinBatchLayerExecution",
    "RMarinBatchMaterializer",
    "RMarinBatchSpanTracker",
    "RMarinBatchSpans",
    "RMarinLayerTensors",
    "RMarinPooledSession",
    "RMarinSessionPool",
    "chunk_aligned_new_tokens",
    "derive_batch_spans",
    "execute_rmarin_flashinfer_batch",
    "install_runner_capture",
]
