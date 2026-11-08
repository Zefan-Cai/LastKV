"""Standalone FlashInfer engine for LastKV and a shape-matched Transformer control.

Single GPU, eager, static request batch per ``generate()`` call, no CUDA graphs,
no prefix cache (the R-KV/FlashInfer execution model). Both model families run
on the same GEMM / norm / RoPE / sampling kernels so their numbers differ only
by the attention path and the KV layout.

Transformer (``family == "transformer"``)
    One paged pool ``[L, pages, page, Hkv, D]`` for K and for V with a static
    per-request region; full-prompt ragged prefill (one call per layer) or
    optional chunked prefill; paged decode with one ``plan()`` per step shared
    by all layers.

LastKV (``family == "lastkv"``)
    * ``source`` pool: ONE shared copy of the final layer's K/V for every
      completed chunk (4 KiB/token for the 8B model), static per-request region.
    * ``ring`` pool ``[L, pages, page, Hkv, D]``: per layer, per request,
      two chunk halves — half 0 holds the alpha-blended previous chunk,
      half 1 the current (partial) chunk. 256 MiB per sequence for the 8B model.
    * Per layer attention = local pass over [half 0 | half 1] (causal,
      bottom-right aligned) + source pass over the older chunks (non-causal,
      output scaled by ``deep_v_scale[l, h]``) merged with
      ``flashinfer.cascade.merge_state``. Two ``plan()`` calls per step
      (local, source), both shared by the 32 layers because every layer's
      ring has identical page ids and the source pool is shared.
    * Chunk boundary (``_commit``): copy the final layer's raw K/V of the
      finished chunk into the source region; for every layer write
      ``alpha * deep + (1 - alpha) * own`` (V with ``deep * s_l``) into half 0
      with the same bf16 op order as ``lastkv/llm/modeling.py``.
    * Prefill is chunk-synchronous: chunk n of any layer needs chunk n-1 of the
      final layer, so a 126K prompt is 124 sequential 1024-token chunk steps.
"""

from __future__ import annotations

import math
import time
from itertools import accumulate
from typing import Callable, Sequence

import torch

from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from flashinfer.cascade import merge_state
from flashinfer.sampling import top_p_sampling_from_probs

from .config import GenOutput, ModelConfig
from .loader import load_weights
from .models import CausalLM

_WORKSPACE_BYTES = 256 * 1024 * 1024

# on_logits(rows, start_positions, token_counts, logits) — logits [sum(counts), vocab]
LogitsHook = Callable[[list[int], list[int], list[int], torch.Tensor], None]


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


class Engine:
    def __init__(
        self,
        model_path: str,
        max_batch_size: int = 8,
        max_seq_len: int = 32768,
        page_size: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        tf_prefill_chunk: int | None = None,
        prefill_backend: str = "auto",
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.page = page_size
        self.config = cfg = ModelConfig.from_model_dir(model_path)
        self.family = cfg.family
        self.tf_prefill_chunk = tf_prefill_chunk

        self.model = CausalLM(cfg, max_position=max_seq_len + 8, device=self.device, dtype=dtype)
        load_weights(self.model, model_path)
        self.model.eval()

        L, Hkv, D = cfg.num_layers, cfg.num_kv_heads, cfg.head_dim
        B, P = max_batch_size, page_size
        self.seq_pages = _cdiv(max_seq_len, P)

        if self.family == "lastkv":
            self.chunk = C = cfg.chunk_size
            if C % P:
                raise ValueError("chunk_size must be a multiple of page_size")
            self.chunk_pages = C // P
            self.ring_pages = 2 * self.chunk_pages  # half 0 = previous (blended), half 1 = current
            shape_src = (B * self.seq_pages, P, Hkv, D)
            shape_ring = (L, B * self.ring_pages, P, Hkv, D)
            self.src_k = torch.zeros(shape_src, dtype=dtype, device=self.device)
            self.src_v = torch.zeros(shape_src, dtype=dtype, device=self.device)
            self.ring_k = torch.zeros(shape_ring, dtype=dtype, device=self.device)
            self.ring_v = torch.zeros(shape_ring, dtype=dtype, device=self.device)
            # Flat token-slot views: slot = page * P + offset.
            self.src_k_flat = self.src_k.view(-1, Hkv, D)
            self.src_v_flat = self.src_v.view(-1, Hkv, D)
            self.ring_k_flat = self.ring_k.view(L, -1, Hkv, D)
            self.ring_v_flat = self.ring_v.view(L, -1, Hkv, D)
            # Recurrence parameters, rounded exactly like the HF reference
            # (bf16 parameters -> bf16 sigmoid / bf16 scale).
            alphas = self.model.stacked_alphas().to(dtype)
            self.alpha = alphas.sigmoid().view(L, 1, Hkv, 1)  # [L,1,Hkv,1]
            self.one_minus_alpha = (1 - self.alpha)
            scale = self.model.stacked_deep_v_scale()
            if scale is None:
                scale = torch.ones(L, Hkv, device=self.device, dtype=dtype)
            self.v_scale = scale.to(dtype).view(L, 1, Hkv, 1)  # applied to deep V
            # per-query-head scale for the source attention output
            self.v_scale_q = (
                scale.to(dtype).repeat_interleave(cfg.group_size, dim=1).view(L, 1, cfg.num_q_heads, 1)
            )
            self._ring_arange = torch.arange(B * self.ring_pages, dtype=torch.int32, device=self.device)
            self._src_arange = torch.arange(B * self.seq_pages, dtype=torch.int32, device=self.device)
        else:
            shape = (L, B * self.seq_pages, P, Hkv, D)
            self.kv_k = torch.zeros(shape, dtype=dtype, device=self.device)
            self.kv_v = torch.zeros(shape, dtype=dtype, device=self.device)
            self.kv_k_flat = self.kv_k.view(L, -1, Hkv, D)
            self.kv_v_flat = self.kv_v.view(L, -1, Hkv, D)
            self._kv_arange = torch.arange(B * self.seq_pages, dtype=torch.int32, device=self.device)

        def ws() -> torch.Tensor:
            return torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=self.device)

        use_tc = cfg.group_size >= 4
        self._ragged_prefill = BatchPrefillWithRaggedKVCacheWrapper(ws(), "NHD", backend=prefill_backend)
        self._paged_prefill_local = BatchPrefillWithPagedKVCacheWrapper(ws(), "NHD", backend=prefill_backend)
        self._paged_prefill_src = BatchPrefillWithPagedKVCacheWrapper(ws(), "NHD", backend=prefill_backend)
        self._decode_local = BatchDecodeWithPagedKVCacheWrapper(ws(), "NHD", use_tensor_cores=use_tc)
        self._decode_src = BatchDecodeWithPagedKVCacheWrapper(ws(), "NHD", use_tensor_cores=use_tc)

        self._kw = dict(q_data_type=dtype, kv_data_type=dtype)
        self.stats: dict[str, float] = {}
        # FlashInfer plan() stages its scheduler metadata through ONE pinned
        # host buffer per wrapper and copies it to the device asynchronously
        # (stream-ordered after the previously launched kernels). A second
        # plan() issued while that copy is still pending overwrites the pinned
        # buffer and corrupts the first plan. generate() is protected by the
        # per-step sampling sync; score()/prefill chunk loops are not, so every
        # step records an event after its plan()s and waits on it before the
        # next step plans. The host still runs one step ahead of the GPU.
        self._plan_event: torch.cuda.Event | None = None
        self._reset_state(0)

    def _before_plan(self) -> None:
        if self._plan_event is not None:
            self._plan_event.synchronize()

    def _after_plan(self) -> None:
        if self.device.type == "cuda":
            ev = torch.cuda.Event()
            ev.record()
            self._plan_event = ev

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------
    def _reset_state(self, batch: int) -> None:
        self.seen = [0] * batch  # tokens processed per request (= next position)
        self.n_done = [0] * batch  # completed chunks (lastkv)
        self.cur = [0] * batch  # tokens in the current chunk (lastkv)

    def kv_bytes_in_use(self) -> int:
        """Physical KV bytes holding live state (not pool capacity)."""
        cfg = self.config
        tok = cfg.kv_bytes_per_token_per_layer
        if self.family == "lastkv":
            total = 0
            for r in range(len(self.seen)):
                if self.seen[r] == 0:
                    continue
                total += self.n_done[r] * self.chunk * tok  # source (final layer only)
                total += 2 * self.chunk * cfg.num_layers * tok  # ring
            return total
        return sum(self.seen) * cfg.num_layers * tok

    def pool_bytes(self) -> int:
        if self.family == "lastkv":
            return sum(t.numel() * t.element_size() for t in (self.src_k, self.src_v, self.ring_k, self.ring_v))
        return sum(t.numel() * t.element_size() for t in (self.kv_k, self.kv_v))

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_token_ids: Sequence[int] = (),
        seed: int = 0,
        record_step_times: bool = False,
    ) -> list[GenOutput]:
        batch = len(prompts)
        lens = [len(p) for p in prompts]
        if not 0 < batch <= self.max_batch_size:
            raise ValueError(f"batch {batch} not in [1, {self.max_batch_size}]")
        for i, n in enumerate(lens):
            if n < 1 or n + max_new_tokens > self.max_seq_len:
                raise ValueError(f"prompt {i}: len {n} + {max_new_tokens} exceeds max_seq_len {self.max_seq_len}")
        self._reset_state(batch)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        self._sync()
        t0 = time.perf_counter()
        logits = self._prefill(prompts, lens)
        self._sync()
        prefill_seconds = time.perf_counter() - t0
        kv_after_prefill = self.kv_bytes_in_use()

        token_ids: list[list[int]] = [[] for _ in range(batch)]
        finish = [""] * batch
        last = [0] * batch
        stop = set(stop_token_ids)

        def record(rows: list[int], sampled: list[int]) -> None:
            for r, tok in zip(rows, sampled):
                token_ids[r].append(tok)
                last[r] = tok
                if tok in stop:
                    finish[r] = "stop"
                elif len(token_ids[r]) >= max_new_tokens:
                    finish[r] = "length"

        sampled = self._sample(logits, temperature, top_p, generator).tolist()
        record(list(range(batch)), sampled)

        events: list[tuple] = []
        self._sync()
        t0 = time.perf_counter()
        steps = 0
        while True:
            active = [r for r in range(batch) if not finish[r]]
            if not active:
                break
            if record_step_times:
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
            logits = self._decode_step(active, [last[r] for r in active])
            sampled = self._sample(logits, temperature, top_p, generator).tolist()
            if record_step_times:
                e1.record()
                events.append((e0, e1, len(active)))
            record(active, sampled)
            steps += 1
        self._sync()
        decode_seconds = time.perf_counter() - t0

        self.stats = {
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "decode_steps": float(steps),
            "peak_memory_bytes": float(torch.cuda.max_memory_allocated(self.device)) if self.device.type == "cuda" else 0.0,
            "kv_bytes_after_prefill": float(kv_after_prefill),
            "kv_bytes_end": float(self.kv_bytes_in_use()),
            "pool_bytes": float(self.pool_bytes()),
        }
        if record_step_times:
            self.stats["step_ms"] = [(a.elapsed_time(b), n) for a, b, n in events]
        return [
            GenOutput(token_ids=token_ids[r], num_prefill_tokens=lens[r], num_output_tokens=len(token_ids[r]), finish_reason=finish[r])
            for r in range(batch)
        ]

    @torch.inference_mode()
    def score(self, prompt: Sequence[int], prefill_tokens: int | None = None) -> dict:
        """Teacher-forced next-token log-probs over ``prompt`` (batch 1).

        ``prompt[:prefill_tokens]`` goes through the prefill path, the rest is
        fed one token per decode step. Returns ``logprob[i]`` = log p(prompt[i+1] |
        prompt[:i+1]) and ``argmax[i]`` for i in [0, len-1), plus the path per position.
        """
        n = len(prompt)
        split = n if prefill_tokens is None else max(1, min(prefill_tokens, n))
        if n > self.max_seq_len:
            raise ValueError("prompt longer than max_seq_len")
        self._reset_state(1)
        logprob = torch.empty(n - 1, dtype=torch.float32, device=self.device)
        argmax = torch.empty(n - 1, dtype=torch.long, device=self.device)
        target = torch.tensor(prompt[1:], dtype=torch.long, device=self.device)

        def hook(rows, starts, counts, logits):
            start = starts[0]
            count = counts[0]
            end = min(start + count, n - 1)
            if end <= start:
                return
            lp = torch.log_softmax(logits[: end - start].float(), dim=-1)
            logprob[start:end] = lp.gather(1, target[start:end, None]).squeeze(1)
            argmax[start:end] = lp.argmax(dim=-1)

        logits = self._prefill([prompt[:split]], [split], on_logits=hook)
        pos = split
        while pos < n:
            lp = torch.log_softmax(logits[0].float(), dim=-1)
            logprob[pos - 1] = lp[prompt[pos]]
            argmax[pos - 1] = lp.argmax()
            if pos == n - 1:
                break
            logits = self._decode_step([0], [prompt[pos]])
            pos += 1
        self._sync()
        return {
            "logprob": logprob.cpu(),
            "argmax": argmax.cpu(),
            "prefill_tokens": split,
            "path": ["prefill"] * (split - 1) + ["decode"] * (n - split),
        }

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------
    def _prefill(self, prompts, lens, on_logits: LogitsHook | None = None) -> torch.Tensor:
        if self.family == "lastkv":
            return self._prefill_lastkv(prompts, lens, on_logits)
        if self.tf_prefill_chunk:
            return self._prefill_tf_chunked(prompts, lens, on_logits)
        return self._prefill_tf_full(prompts, lens, on_logits)

    def _decode_step(self, active: list[int], last_tokens: list[int]) -> torch.Tensor:
        if self.family == "lastkv":
            return self._decode_lastkv(active, last_tokens)
        return self._decode_tf(active, last_tokens)

    def _stage(self, ids: list[int], positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        staged = torch.tensor([ids, positions], dtype=torch.long).to(self.device, non_blocking=True)
        return staged[0], staged[1].to(torch.int32)

    def _cat_pages(self, arange: torch.Tensor, ranges: list[tuple[int, int]]) -> torch.Tensor:
        parts = [arange[a:b] for a, b in ranges if b > a]
        if not parts:
            return arange[:0]
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    # ------------------------------------------------------------------
    # Transformer control
    # ------------------------------------------------------------------
    def _tf_write_slots(self, rows: list[int], starts: list[int], counts: list[int]) -> torch.Tensor:
        slots = []
        for r, s, c in zip(rows, starts, counts):
            base = r * self.seq_pages * self.page
            slots.append(torch.arange(base + s, base + s + c, dtype=torch.long))
        return torch.cat(slots).to(self.device, non_blocking=True)

    def _tf_attention_factory(self, run):
        def attention(layer, q, k, v):
            self.kv_k_flat[layer, self._write_slots] = k
            self.kv_v_flat[layer, self._write_slots] = v
            return run(layer, q)

        return attention

    def _prefill_tf_full(self, prompts, lens, on_logits):
        """One ragged causal prefill per request (bounded activation memory)."""
        model = self.model
        cfg = self.config
        out = torch.empty(len(prompts), cfg.vocab_size, dtype=self.dtype, device=self.device)
        for r, prompt in enumerate(prompts):
            n = lens[r]
            ids, positions = self._stage(list(prompt), list(range(n)))
            self._write_slots = self._tf_write_slots([r], [0], [n])
            indptr = torch.tensor([0, n], dtype=torch.int32)
            self._before_plan()
            self._ragged_prefill.plan(indptr, indptr, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, causal=True, **self._kw)
            self._after_plan()

            def run(layer, q, _k=None, _v=None):
                return self._ragged_prefill.run(q, self._cur_k, self._cur_v)

            def attention(layer, q, k, v):
                self.kv_k_flat[layer, self._write_slots] = k
                self.kv_v_flat[layer, self._write_slots] = v
                self._cur_k, self._cur_v = k, v
                return run(layer, q)

            hidden = model(ids, positions, attention)
            self.seen[r] = n
            if on_logits is not None:
                self._emit_logits(hidden, [r], [0], [n], on_logits)
            out[r] = model.compute_logits(hidden[-1:])[0]
        return out

    def _emit_logits(self, hidden, rows, starts, counts, on_logits, chunk=4096):
        # chunked lm_head to bound the [tokens, vocab] temporary
        if len(rows) == 1:
            n = counts[0]
            for s in range(0, n, chunk):
                e = min(n, s + chunk)
                on_logits(rows, [starts[0] + s], [e - s], self.model.compute_logits(hidden[s:e]))
        else:
            on_logits(rows, starts, counts, self.model.compute_logits(hidden))

    def _prefill_tf_chunked(self, prompts, lens, on_logits):
        """Chunked prefill of C tokens per step over the request's own pages."""
        C = self.tf_prefill_chunk
        cfg = self.config
        model = self.model
        batch = len(prompts)
        out = torch.empty(batch, cfg.vocab_size, dtype=self.dtype, device=self.device)
        n = 0
        while True:
            active = [r for r in range(batch) if lens[r] > n * C]
            if not active:
                break
            counts = [min(C, lens[r] - n * C) for r in active]
            ids = [t for r, c in zip(active, counts) for t in prompts[r][n * C : n * C + c]]
            positions = [n * C + i for c in counts for i in range(c)]
            ids_t, pos_t = self._stage(ids, positions)
            self._write_slots = self._tf_write_slots(active, [n * C] * len(active), counts)
            qo_indptr = torch.tensor([0, *accumulate(counts)], dtype=torch.int32)
            pages = [_cdiv(n * C + c, self.page) for c in counts]
            kv_indptr = torch.tensor([0, *accumulate(pages)], dtype=torch.int32)
            kv_indices = self._cat_pages(self._kv_arange, [(r * self.seq_pages, r * self.seq_pages + p) for r, p in zip(active, pages)])
            last = torch.tensor([(n * C + c - 1) % self.page + 1 for c in counts], dtype=torch.int32)
            self._before_plan()
            self._paged_prefill_local.plan(
                qo_indptr, kv_indptr, kv_indices, last, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, self.page,
                causal=True, **self._kw,
            )
            self._after_plan()
            attention = self._tf_attention_factory(
                lambda layer, q: self._paged_prefill_local.run(q, (self.kv_k[layer], self.kv_v[layer]))
            )
            hidden = model(ids_t, pos_t, attention)
            if on_logits is not None:
                self._emit_logits(hidden, active, [n * C] * len(active), counts, on_logits)
            ends = list(accumulate(counts))
            for i, r in enumerate(active):
                self.seen[r] += counts[i]
                if self.seen[r] == lens[r]:
                    out[r] = model.compute_logits(hidden[ends[i] - 1 : ends[i]])[0]
            n += 1
        return out

    def _decode_tf(self, active, last_tokens):
        cfg = self.config
        positions = [self.seen[r] for r in active]
        ids_t, pos_t = self._stage(last_tokens, positions)
        self._write_slots = self._tf_write_slots(active, positions, [1] * len(active))
        pages = [_cdiv(p + 1, self.page) for p in positions]
        kv_indptr = torch.tensor([0, *accumulate(pages)], dtype=torch.int32)
        kv_indices = self._cat_pages(self._kv_arange, [(r * self.seq_pages, r * self.seq_pages + p) for r, p in zip(active, pages)])
        last = torch.tensor([p % self.page + 1 for p in positions], dtype=torch.int32)
        self._before_plan()
        self._decode_local.plan(kv_indptr, kv_indices, last, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, self.page, **self._kw)
        self._after_plan()
        attention = self._tf_attention_factory(lambda layer, q: self._decode_local.run(q, (self.kv_k[layer], self.kv_v[layer])))
        hidden = self.model(ids_t, pos_t, attention)
        for r in active:
            self.seen[r] += 1
        return self.model.compute_logits(hidden)

    # ------------------------------------------------------------------
    # LastKV
    # ------------------------------------------------------------------
    def _ring_slots(self, rows: list[int], offsets: list[int], counts: list[int]) -> torch.Tensor:
        """Flat ring slots of the CURRENT half (half 1) for (row, offset-in-chunk, count)."""
        slots = []
        for r, o, c in zip(rows, offsets, counts):
            base = r * self.ring_pages * self.page + self.chunk
            slots.append(torch.arange(base + o, base + o + c, dtype=torch.long))
        return torch.cat(slots).to(self.device, non_blocking=True)

    def _plan_local(self, prefill: bool, rows, n_done, cur_counts, qo_counts):
        """Plan the local pass over [half 0 (if a chunk is done) | half 1 (cur tokens)]."""
        cfg = self.config
        ranges, pages = [], []
        for r, nd, cur in zip(rows, n_done, cur_counts):
            base = r * self.ring_pages
            start = 0 if nd >= 1 else self.chunk_pages
            end = self.chunk_pages + _cdiv(cur, self.page)
            ranges.append((base + start, base + end))
            pages.append(end - start)
        kv_indptr = torch.tensor([0, *accumulate(pages)], dtype=torch.int32)
        kv_indices = self._cat_pages(self._ring_arange, ranges)
        last = torch.tensor([(cur - 1) % self.page + 1 for cur in cur_counts], dtype=torch.int32)
        if prefill:
            qo_indptr = torch.tensor([0, *accumulate(qo_counts)], dtype=torch.int32)
            self._paged_prefill_local.plan(
                qo_indptr, kv_indptr, kv_indices, last, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, self.page,
                causal=True, **self._kw,
            )
        else:
            self._decode_local.plan(kv_indptr, kv_indices, last, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, self.page, **self._kw)

    def _plan_source(self, prefill: bool, rows, n_done, qo_counts) -> bool:
        """Plan the source pass over the older completed chunks (all but the last).

        Rows without older history get one dummy page and are masked out of the
        merge via an LSE of -inf. Returns False (no plan) if no row has history.
        """
        cfg = self.config
        old_pages = [max(0, nd - 1) * self.chunk_pages for nd in n_done]
        if not any(old_pages):
            self._nosrc_mask = None
            return False
        ranges, pages, last = [], [], []
        nosrc = []
        for r, op in zip(rows, old_pages):
            base = r * self.seq_pages
            if op == 0:
                ranges.append((base, base + 1))
                pages.append(1)
                last.append(1)
                nosrc.append(True)
            else:
                ranges.append((base, base + op))
                pages.append(op)
                last.append(self.page)
                nosrc.append(False)
        kv_indptr = torch.tensor([0, *accumulate(pages)], dtype=torch.int32)
        kv_indices = self._cat_pages(self._src_arange, ranges)
        last_t = torch.tensor(last, dtype=torch.int32)
        if any(nosrc):
            mask = torch.tensor(nosrc, dtype=torch.bool)
            if prefill:
                mask = torch.repeat_interleave(mask, torch.tensor(qo_counts))
            self._nosrc_mask = mask.to(self.device, non_blocking=True)
        else:
            self._nosrc_mask = None
        if prefill:
            qo_indptr = torch.tensor([0, *accumulate(qo_counts)], dtype=torch.int32)
            self._paged_prefill_src.plan(
                qo_indptr, kv_indptr, kv_indices, last_t, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, self.page,
                causal=False, **self._kw,
            )
        else:
            self._decode_src.plan(kv_indptr, kv_indices, last_t, cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, self.page, **self._kw)
        return True

    def _lastkv_attention_factory(self, prefill: bool, has_src: bool):
        local_w = self._paged_prefill_local if prefill else self._decode_local
        src_w = self._paged_prefill_src if prefill else self._decode_src

        def attention(layer, q, k, v):
            self.ring_k_flat[layer, self._write_slots] = k
            self.ring_v_flat[layer, self._write_slots] = v
            o_l, lse_l = local_w.run(q, (self.ring_k[layer], self.ring_v[layer]), return_lse=True)
            if not has_src:
                return o_l
            o_s, lse_s = src_w.run(q, (self.src_k, self.src_v), return_lse=True)
            o_s.mul_(self.v_scale_q[layer])  # shared final-layer V scaled per layer/head
            if self._nosrc_mask is not None:
                lse_s.masked_fill_(self._nosrc_mask[:, None], float("-inf"))
            o, _ = merge_state(o_s, lse_s, o_l, lse_l)
            return o

        return attention

    def _commit(self, rows: list[int], chunk_idx: list[int]) -> None:
        """Chunk boundary for ``rows`` whose chunk ``chunk_idx[i]`` just completed.

        deep <- append(current[final layer]); previous <- blend(deep_chunk, current).
        """
        if not rows:
            return
        C, L = self.chunk, self.config.num_layers
        half1 = torch.cat([
            torch.arange(r * self.ring_pages * self.page + C, r * self.ring_pages * self.page + 2 * C, dtype=torch.long) for r in rows
        ]).to(self.device, non_blocking=True)
        half0 = half1 - C
        src = torch.cat([
            torch.arange(r * self.seq_pages * self.page + n * C, r * self.seq_pages * self.page + (n + 1) * C, dtype=torch.long)
            for r, n in zip(rows, chunk_idx)
        ]).to(self.device, non_blocking=True)
        deep_k = self.ring_k_flat[L - 1, half1]  # raw final-layer K of the finished chunk
        deep_v = self.ring_v_flat[L - 1, half1]
        self.src_k_flat[src] = deep_k
        self.src_v_flat[src] = deep_v
        # Same bf16 op order as modeling.py:
        #   k: alpha * deep_k + (1 - alpha) * previous_k
        #   v: alpha * (deep_v * s) + (1 - alpha) * previous_v
        for layer in range(L):
            cur_k = self.ring_k_flat[layer, half1]
            cur_v = self.ring_v_flat[layer, half1]
            a = self.alpha[layer]
            b = self.one_minus_alpha[layer]
            self.ring_k_flat[layer, half0] = a * deep_k + b * cur_k
            self.ring_v_flat[layer, half0] = a * (deep_v * self.v_scale[layer]) + b * cur_v

    def _prefill_lastkv(self, prompts, lens, on_logits):
        cfg = self.config
        C = self.chunk
        model = self.model
        batch = len(prompts)
        out = torch.empty(batch, cfg.vocab_size, dtype=self.dtype, device=self.device)
        n = 0
        while True:
            active = [r for r in range(batch) if lens[r] > n * C]
            if not active:
                break
            counts = [min(C, lens[r] - n * C) for r in active]
            ids = [t for r, c in zip(active, counts) for t in prompts[r][n * C : n * C + c]]
            positions = [n * C + i for c in counts for i in range(c)]
            ids_t, pos_t = self._stage(ids, positions)
            self._write_slots = self._ring_slots(active, [0] * len(active), counts)
            n_done = [self.n_done[r] for r in active]  # == n for every active row
            self._before_plan()
            self._plan_local(True, active, n_done, counts, counts)
            has_src = self._plan_source(True, active, n_done, counts)
            self._after_plan()
            hidden = model(ids_t, pos_t, self._lastkv_attention_factory(True, has_src))
            if on_logits is not None:
                self._emit_logits(hidden, active, [n * C] * len(active), counts, on_logits)
            ends = list(accumulate(counts))
            commit_rows = []
            for i, r in enumerate(active):
                self.seen[r] += counts[i]
                if self.seen[r] == lens[r]:
                    out[r] = model.compute_logits(hidden[ends[i] - 1 : ends[i]])[0]
                if counts[i] == C:
                    commit_rows.append(r)
                else:
                    self.cur[r] = counts[i]
            self._commit(commit_rows, [n] * len(commit_rows))
            for r in commit_rows:
                self.n_done[r] += 1
                self.cur[r] = 0
            n += 1
        return out

    def _decode_lastkv(self, active, last_tokens):
        C = self.chunk
        positions = [self.seen[r] for r in active]
        ids_t, pos_t = self._stage(last_tokens, positions)
        cur_after = [self.cur[r] + 1 for r in active]
        self._write_slots = self._ring_slots(active, [self.cur[r] for r in active], [1] * len(active))
        n_done = [self.n_done[r] for r in active]
        self._before_plan()
        self._plan_local(False, active, n_done, cur_after, None)
        has_src = self._plan_source(False, active, n_done, None)
        self._after_plan()
        hidden = self.model(ids_t, pos_t, self._lastkv_attention_factory(False, has_src))
        logits = self.model.compute_logits(hidden)
        commit_rows, commit_idx = [], []
        for r, c in zip(active, cur_after):
            self.seen[r] += 1
            if c == C:
                commit_rows.append(r)
                commit_idx.append(self.n_done[r])
            else:
                self.cur[r] = c
        self._commit(commit_rows, commit_idx)
        for r in commit_rows:
            self.n_done[r] += 1
            self.cur[r] = 0
        return logits

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float, generator) -> torch.Tensor:
        if temperature == 0.0:
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits.float() / temperature, dim=-1)
        return top_p_sampling_from_probs(probs, top_p, deterministic=True, generator=generator)
