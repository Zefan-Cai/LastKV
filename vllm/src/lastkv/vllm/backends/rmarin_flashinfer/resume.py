"""Cross-request RMarin session resume (L2 prototype, 2026-10-23).

Scheduler side: ``RMarinResumeConnector`` (a vLLM v1 KV connector) reports ``B``
externally-computed tokens for a new request whose prompt token prefix ``[0, B)``
equals (by hash) the prefix snapshotted by an earlier request on a currently free
session slot.  Worker side: ``RMarinSessionPool`` adopts that slot, restores the
layer-local rings from the snapshot and continues from boundary ``B``.  The raw
top-layer source slab and the per-layer transformed V planes for ``[0, B)`` are
still resident in the slot, so every later chunk is computed from exactly the
tensors a cold prefill would have produced at that boundary.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

RESUME_ENV = "RMARIN_SESSION_RESUME"


def resume_enabled() -> bool:
    return os.environ.get(RESUME_ENV, "1") == "1"


def chunk_prefix_hashes(ids: Any, chunk: int) -> tuple[int, ...]:
    """Chained hash of every chunk-aligned prompt prefix: h[k] identifies ids[: (k+1)*chunk].

    Computed once per request (O(prompt)); matching a snapshot is then O(1)
    instead of re-hashing a 35k-token prefix per snapshot per scheduling attempt.
    """

    out: list[int] = []
    h = 0
    for start in range(0, len(ids) - chunk + 1, chunk):
        h = hash((h, tuple(ids[start : start + chunk])))
        out.append(h)
    return tuple(out)


@dataclass
class _RequestInfo:
    prompt_ids: tuple[int, ...]
    num_prompt_tokens: int
    resume: tuple[int, int] | None = None  # (slot, boundary)
    chain: tuple[int, ...] = ()


@dataclass
class _SlotSnapshot:
    boundary: int
    prefix_hash: int
    epoch: int


class ResumeRegistry:
    """In-process registry shared by the scheduler-side connector and the pool."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.chunk_size: int | None = None
        self.requests: dict[str, _RequestInfo] = {}
        self.snapshots: dict[int, _SlotSnapshot] = {}
        self.owners: dict[int, str] = {}
        self.reserved: dict[int, str] = {}
        self.epoch = 0
        # request ids freed by the scheduler, drained by the session pool at its
        # next forward (the per-forward capture misses ids from steps without a
        # model forward, which left finished requests holding their slots)
        self.finished_pending: set[str] = set()
        self.stats = {"registered": 0, "hits": 0, "resumed_tokens": 0, "snapshots": 0, "adopted": 0}

    def configure(self, chunk_size: int) -> None:
        with self.lock:
            if self.chunk_size not in (None, chunk_size):
                raise RuntimeError("resume registry chunk size mismatch")
            self.chunk_size = int(chunk_size)

    def snapshot_target(self, num_prompt_tokens: int) -> int:
        size = self.chunk_size or 0
        if size <= 0 or num_prompt_tokens <= size:
            return 0
        return ((num_prompt_tokens - 1) // size) * size

    def register(self, request_id: str, prompt_ids: Any, num_prompt_tokens: int) -> int:
        """Reserve the best resumable snapshot for ``request_id`` (idempotent).

        Called when the request arrives (scheduler ``add_request``) so a waiting
        conversation turn holds its snapshot slot against cold requests, and
        again at scheduling time, which then returns the same reservation.
        """

        with self.lock:
            info = self.requests.get(request_id)
            if info is not None and info.resume is not None:
                slot, boundary = info.resume
                snap = self.snapshots.get(slot)
                if (
                    snap is not None
                    and snap.boundary == boundary
                    and self.reserved.get(slot) == request_id
                    and slot not in self.owners
                ):
                    return boundary
                if self.reserved.get(slot) == request_id:
                    self.reserved.pop(slot, None)
                info.resume = None
            if info is None:
                ids = tuple(int(item) for item in prompt_ids)
                info = _RequestInfo(
                    prompt_ids=ids,
                    num_prompt_tokens=int(num_prompt_tokens),
                    chain=chunk_prefix_hashes(ids, int(self.chunk_size or 0)) if self.chunk_size else (),
                )
                self.requests[request_id] = info
                self.stats["registered"] += 1
            chain = info.chain
            best: tuple[int, int] | None = None
            for slot, snap in self.snapshots.items():
                if slot in self.owners:
                    continue
                holder = self.reserved.get(slot)
                if holder is not None and holder != request_id:
                    continue
                boundary = snap.boundary
                if boundary <= 0 or boundary > info.num_prompt_tokens - 1:
                    continue
                k = boundary // int(self.chunk_size or 1) - 1
                if k < 0 or k >= len(chain) or chain[k] != snap.prefix_hash:
                    continue
                if best is None or boundary > best[1]:
                    best = (slot, boundary)
            if best is None:
                if os.environ.get("RMARIN_RESUME_DEBUG") == "1" and self.snapshots:
                    why = []
                    for slot, snap in self.snapshots.items():
                        k = snap.boundary // int(self.chunk_size or 1) - 1
                        if 0 <= k < len(chain) and chain[k] == snap.prefix_hash:
                            why.append({"slot": slot, "boundary": snap.boundary, "owner": self.owners.get(slot),
                                        "reserved": self.reserved.get(slot), "n": info.num_prompt_tokens})
                    self.log("nomatch", request_id=request_id, num_prompt_tokens=info.num_prompt_tokens,
                             hash_matches=why[:4], snapshots=len(self.snapshots))
                return 0
            self.reserved[best[0]] = request_id
            info.resume = best
            self.stats["hits"] += 1
            self.stats["resumed_tokens"] += best[1]
            return best[1]

    def prefix_hash(self, request_id_or_info: Any, boundary: int) -> int | None:
        info = request_id_or_info if isinstance(request_id_or_info, _RequestInfo) else self.requests.get(request_id_or_info)
        if info is None or not self.chunk_size:
            return None
        k = boundary // int(self.chunk_size) - 1
        return info.chain[k] if 0 <= k < len(info.chain) else None

    def is_reserved(self, slot: int) -> bool:
        with self.lock:
            return slot in self.reserved

    def info(self, request_id: str) -> _RequestInfo | None:
        with self.lock:
            return self.requests.get(request_id)

    def owner_of(self, slot: int) -> str | None:
        with self.lock:
            return self.owners.get(slot)

    def resume_for(self, request_id: str) -> tuple[int, int] | None:
        with self.lock:
            info = self.requests.get(request_id)
            return None if info is None else info.resume

    def reserved_for_others(self, request_id: str) -> set[int]:
        with self.lock:
            return {slot for slot, holder in self.reserved.items() if holder != request_id}

    def claim(self, slot: int, request_id: str) -> None:
        with self.lock:
            self.owners[slot] = request_id
            if self.reserved.get(slot) == request_id:
                self.reserved.pop(slot, None)

    def release(self, slot: int, request_id: str) -> None:
        with self.lock:
            if self.owners.get(slot) == request_id:
                self.owners.pop(slot, None)
            if self.reserved.get(slot) == request_id:
                self.reserved.pop(slot, None)
            self.requests.pop(request_id, None)

    def cancel(self, request_id: str) -> None:
        """Drop a reservation made by ``register`` (the prefix pages are gone)."""
        with self.lock:
            info = self.requests.get(request_id)
            if info is not None and info.resume is not None:
                slot = info.resume[0]
                if self.reserved.get(slot) == request_id:
                    self.reserved.pop(slot, None)
                info.resume = None
                self.stats["hits"] -= 1

    def forget(self, request_id: str) -> None:
        """Scheduler-side finish: free ownership immediately (the pool's own
        release only happens at the next forward, which is too late for a
        follow-up request scheduled in between)."""
        with self.lock:
            info = self.requests.get(request_id)
            if info is not None and info.resume is not None:
                slot = info.resume[0]
                if self.reserved.get(slot) == request_id:
                    self.reserved.pop(slot, None)
            for slot, owner in list(self.owners.items()):
                if owner == request_id:
                    self.owners.pop(slot, None)
            self.finished_pending.add(request_id)

    def drain_finished(self, keep: frozenset[str]) -> set[str]:
        with self.lock:
            out = self.finished_pending - keep
            self.finished_pending &= keep
            return out

    def set_snapshot(self, slot: int, boundary: int, prefix_hash: int) -> None:
        with self.lock:
            self.epoch += 1
            self.snapshots[slot] = _SlotSnapshot(boundary, prefix_hash, self.epoch)
            self.stats["snapshots"] += 1

    def clear_snapshot(self, slot: int) -> None:
        with self.lock:
            self.snapshots.pop(slot, None)

    def snapshot_epoch(self, slot: int) -> int:
        with self.lock:
            snap = self.snapshots.get(slot)
            return -1 if snap is None else snap.epoch

    def log(self, event: str, **fields: Any) -> None:
        print(json.dumps({"rmarin_resume": event, "ts": time.time(), **fields, "stats": dict(self.stats)}), flush=True)


REGISTRY = ResumeRegistry()


def _connector_base() -> Any:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata

    return KVConnectorBase_V1, KVConnectorMetadata


try:
    _Base, _Meta = _connector_base()
except Exception:  # vllm not importable (unit tests); connector unusable then
    _Base = _Meta = object


class RMarinResumeMeta(_Meta):
    def __init__(self, resumes: dict[str, tuple[int, int]] | None = None) -> None:
        self.resumes = dict(resumes or {})


class RMarinResumeConnector(_Base):
    """vLLM v1 connector: scheduler-side prefix lookup, worker-side no-op."""

    def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any = None) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        REGISTRY.configure(int(vllm_config.model_config.hf_config.chunk_size))
        self._pending: dict[str, tuple[int, int]] = {}
        REGISTRY.log("connector_init", role=str(role), enabled=resume_enabled())

    # -------- scheduler side --------
    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> tuple[int | None, bool]:
        from . import fastmode as _fm

        if not resume_enabled() or num_computed_tokens or _fm.PAGED_SOURCE:
            # paged source: the chunk-aligned scheduler resolves resume through
            # vLLM's own prefix cache (see RMarinChunkAlignedScheduler).
            return 0, False
        ids = getattr(request, "prompt_token_ids", None)
        if ids is None:
            return 0, False
        matched = REGISTRY.register(request.request_id, ids, int(request.num_prompt_tokens))
        if matched:
            self._pending[request.request_id] = REGISTRY.resume_for(request.request_id)  # type: ignore[assignment]
            REGISTRY.log("match", request_id=request.request_id, boundary=matched, num_prompt_tokens=int(request.num_prompt_tokens))
        return matched, False

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None:
        return None

    def build_connector_meta(self, scheduler_output: Any) -> Any:
        meta = RMarinResumeMeta(self._pending)
        self._pending = {}
        return meta

    def request_finished(self, request: Any, block_ids: Any) -> tuple[bool, dict[str, Any] | None]:
        REGISTRY.forget(request.request_id)
        return False, None

    # -------- worker side (session pool performs the restore lazily) --------
    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return None

    def save_kv_layer(self, layer_name: str, kv_layer: Any, attn_metadata: Any, **kwargs: Any) -> None:
        return None

    def wait_for_save(self) -> None:
        return None


__all__ = ["REGISTRY", "RMarinResumeConnector", "RMarinResumeMeta", "ResumeRegistry", "resume_enabled"]
