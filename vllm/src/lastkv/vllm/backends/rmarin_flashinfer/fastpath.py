"""Decode fast path (stage B, 2026-10-24).

For uniform single-token decode steps the whole per-step host logic (slot
ownership, boundary commit, catch-up, snapshot, index buffers, FlashInfer
plans) runs *before* the model forward, inside the attention metadata
builder.  The forward then executes a fixed sequence of GPU ops per layer that
only reads device buffers, so vLLM's FULL_DECODE_ONLY CUDA graphs can capture
and replay it.  Prefill chunks and mixed batches keep the eager path.
"""

from __future__ import annotations

import math
from typing import Any

from .access_plan import TokenRange
from .runtime_contract import RMarinBackendContractError


class _FastPathState:
    def __init__(self) -> None:
        self.dummy_run = False
        self.model: dict[str, Any] = {}
        self.enabled = False
        self.stats = {"steps": 0, "dummy": 0, "ineligible": 0, "prologue_ms": 0.0, "plan_ms": 0.0, "pool_ms": 0.0}


STATE = _FastPathState()
DUMMY_GENERATION = -1


class RMarinDecodeFastPath:
    def __init__(self, *, pool: Any, tracker: Any, max_batch: int, sm_scale: float | None, max_tokens: int = 0) -> None:
        self.pool = pool
        self.tracker = tracker
        self.torch = pool.materializer.torch
        self.flashinfer = pool.materializer.flashinfer
        self.num_layers = pool.num_layers
        self.chunk = pool.chunk_size
        self.page = pool.page_size
        self.ring_pages = pool._ring_pages_per_slot
        self.ring_tokens = self.ring_pages * self.page
        self.num_slots = pool.num_slots
        self.src_pages_per_slot = pool._fused_pages_per_slot
        self.src_padded_tokens = pool._fused_padded_tokens
        self.max_batch = max(1, int(max_batch))
        self.sm_scale = sm_scale
        torch = self.torch
        ring_k = pool._ring_k
        device = ring_k[0].device
        self.device = device
        heads = int(ring_k[0].shape[2])
        width = int(ring_k[0].shape[3])
        self.kv_heads = heads
        self.head_dim = width
        self.kv_dtype = ring_k[0].dtype
        self.ring_k = ring_k
        self.ring_v = pool._ring_v
        self.ring_k_flat = [t.view(-1, heads, width) for t in ring_k]
        self.ring_v_flat = [t.view(-1, heads, width) for t in pool._ring_v]
        self.src_k = pool._fused_source_slab
        self.src_v_raw = pool._fused_raw_value_slab
        self.src_k_flat = self.src_k.view(-1, heads, width)
        self.src_v_raw_flat = self.src_v_raw.view(-1, heads, width)
        self.v_planes = pool._fused_value_slab  # [L, pages, P, H, D] or None (shared V)
        self.shared_v = pool._fused_value_slab is None
        self.scale_q: Any | None = None
        self.scratch_ring_token = self.num_slots * self.ring_tokens
        self.scratch_ring_page = self.num_slots * self.ring_pages
        self.paged = bool(getattr(pool, "paged_source", False))
        if self.paged:
            # source pages are vLLM blocks; block 0 (vLLM's null block) is the scratch
            self.scratch_src_page = 0
            self.scratch_src_token = 0
        else:
            self.scratch_src_page = self.num_slots * self.src_pages_per_slot
            self.scratch_src_token = self.scratch_src_page * self.page
        if int(self.ring_k_flat[0].shape[0]) <= self.scratch_ring_token:
            raise RMarinBackendContractError("ring scratch page missing")
        if int(self.src_k_flat.shape[0]) <= self.scratch_src_token:
            raise RMarinBackendContractError("source scratch page missing")
        mb = self.max_batch
        # token-level index buffers: decode uses one row per request; prefill
        # chunk steps (FPP) use one entry per scheduled token.
        self.max_tokens = max(mb, int(max_tokens))
        mt = self.max_tokens
        self.ring_idx = torch.zeros(mt, dtype=torch.int64, device=device)
        self.src_idx = torch.zeros(mt, dtype=torch.int64, device=device)
        self.nosrc_mask = torch.zeros(mt, dtype=torch.bool, device=device)
        # Host staging is rotated per step: with async scheduling the host
        # prepares step N+1 while step N's non_blocking copies may still be
        # pending, so a single pinned buffer would be overwritten too early.
        self.NSETS = 4
        self._set = 0
        self._hs_ring_idx = [torch.zeros(mt, dtype=torch.int64, device="cpu").pin_memory() for _ in range(self.NSETS)]
        self._hs_src_idx = [torch.zeros(mt, dtype=torch.int64, device="cpu").pin_memory() for _ in range(self.NSETS)]
        self._hs_nosrc = [torch.zeros(mt, dtype=torch.bool, device="cpu").pin_memory() for _ in range(self.NSETS)]
        self.tokens = 0
        self.max_q = 1
        self._workspace = {
            "local": torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
            "source": torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        }
        self._wrappers: dict[tuple[str, int], Any] = {}
        self._bufs: dict[tuple[str, int], tuple[Any, Any, Any]] = {}
        # pinned host staging so FlashInfer's non_blocking copies never block
        max_src_pages = self.max_batch * self.src_pages_per_slot
        max_loc_pages = self.max_batch * self.ring_pages
        self._h_indptr = {r: torch.zeros(self.max_batch + 1, dtype=torch.int32, device="cpu").pin_memory() for r in ("local", "source")}
        self._h_indices = {"local": torch.zeros(max(max_loc_pages, 1), dtype=torch.int32, device="cpu").pin_memory(),
                           "source": torch.zeros(max(max_src_pages, 1), dtype=torch.int32, device="cpu").pin_memory()}
        self._h_last = {r: torch.ones(self.max_batch, dtype=torch.int32, device="cpu").pin_memory() for r in ("local", "source")}
        self.active_generation: int | None = None
        self.last_real_generation: int | None = None
        self.batch = 0
        self.neg_inf = float("-inf")
        import os as _os

        self.debug = _os.environ.get("RMARIN_FDP_DEBUG") == "1"
        # Attention kernel for the fast path: fa3 (vLLM flash-attn varlen, no
        # per-step plan -> never blocks on the GPU) or flashinfer (wrapper plan
        # blocks in cudagraph mode).
        self.attn = _os.environ.get("RMARIN_FDP_ATTN", "fa3")
        # Fast prefill path (chunk steps / mixed batches) through the same op sequence.
        self.prefill_enabled = _os.environ.get("RMARIN_FPP", "1") == "1"
        self._scale = float(sm_scale) if sm_scale is not None else float(width) ** -0.5
        if self.attn == "fa3":
            from vllm.vllm_flash_attn import flash_attn_varlen_func
            from vllm._custom_ops import merge_attn_states

            self._fa = flash_attn_varlen_func
            self._merge = merge_attn_states
            self.cu_q = torch.arange(0, mb + 1, dtype=torch.int32, device=device)
            self.cu_q_var = torch.zeros(mb + 1, dtype=torch.int32, device=device)
            self._hs_cu_q = [torch.zeros(mb + 1, dtype=torch.int32, device="cpu").pin_memory() for _ in range(self.NSETS)]
            self.seq_local = torch.ones(mb, dtype=torch.int32, device=device)
            self.seq_src = torch.ones(mb, dtype=torch.int32, device=device)
            self.bt_local = torch.zeros((mb, self.ring_pages), dtype=torch.int32, device=device)
            self.bt_src = torch.zeros((mb, self.src_pages_per_slot), dtype=torch.int32, device=device)
            self._hs_seq_local = [torch.ones(mb, dtype=torch.int32, device="cpu").pin_memory() for _ in range(self.NSETS)]
            self._hs_seq_src = [torch.ones(mb, dtype=torch.int32, device="cpu").pin_memory() for _ in range(self.NSETS)]
            self._hs_bt_local = [torch.zeros((mb, self.ring_pages), dtype=torch.int32, device="cpu").pin_memory() for _ in range(self.NSETS)]
            self._hs_bt_src = [torch.zeros((mb, self.src_pages_per_slot), dtype=torch.int32, device="cpu").pin_memory() for _ in range(self.NSETS)]
            self.nosrc_lse_mask = torch.zeros((1, mt), dtype=torch.bool, device=device)
        # MERGED_ATTN: one FA3 call per layer over [older history | ring window]
        # read from the shared K/V page pools through a per-layer block table.
        self.merged = bool(getattr(pool, "merged", False))
        if self.merged:
            if self.attn != "fa3" or not self.prefill_enabled:
                raise RMarinBackendContractError("merged attention requires RMARIN_FDP_ATTN=fa3 and RMARIN_FPP=1")
            self.pool_k = pool._pool_k
            self.pool_v = pool._pool_v
            self.W = self.src_pages_per_slot + self.ring_pages
            self.max_k = self.src_padded_tokens + self.ring_tokens
            # [L, 1, 1] absolute pool page of ring page 0 for each layer
            self.ring_base = (
                pool.merged_src_pages + torch.arange(self.num_layers, dtype=torch.int64) * pool.merged_ring_stride
            ).to(torch.int32).view(self.num_layers, 1, 1).to(device)
            self.col_ids = torch.arange(self.W, dtype=torch.int32, device=device)
            self.bt_dev = torch.zeros((mb, self.W), dtype=torch.int32, device=device)
            self.bt_all = torch.zeros((self.num_layers, mb, self.W), dtype=torch.int32, device=device)
            self.nsrc = torch.zeros(mb, dtype=torch.int32, device=device)
            self.seq_k = torch.ones(mb, dtype=torch.int32, device=device)
            self._hs_bt = [torch.zeros((mb, self.W), dtype=torch.int32, device="cpu").pin_memory() for _ in range(self.NSETS)]
            self._hs_nsrc = [torch.zeros(mb, dtype=torch.int32, device="cpu").pin_memory() for _ in range(self.NSETS)]
            self._hs_seqk = [torch.ones(mb, dtype=torch.int32, device="cpu").pin_memory() for _ in range(self.NSETS)]
            # NOTE: the recurrent parameters (alpha logits, value scales) are
            # still at their init values when the fast path is installed (model
            # __init__ precedes load_weights), and the tensor-ops parameter
            # cache keys on the Parameter object -> resolve every scale lazily
            # at the first forward (vLLM's profile run, after loading).
            self.top_scale = None  # [1, H_kv, 1]
            self.inv_scale = None  # [L, 1, H_kv, 1] fp32
            from . import fastmode as _fm

            # PAGED_SOURCE: older-history pages come from vLLM's block table (device),
            # token write slots from vLLM's slot_mapping; ring entries are scattered
            # behind the request's source pages.
            if self.paged:
                nring = mb * self.ring_pages
                self.ring_flat_dev = torch.zeros(nring, dtype=torch.int64, device=device)
                self.ring_vals_dev = torch.zeros(nring, dtype=torch.int32, device=device)
                self._hs_ring_flat = [torch.zeros(nring, dtype=torch.int64, device="cpu").pin_memory() for _ in range(self.NSETS)]
                self._hs_ring_vals = [torch.zeros(nring, dtype=torch.int32, device="cpu").pin_memory() for _ in range(self.NSETS)]
                self.step_block_table: Any | None = None
                self.step_slot_mapping: Any | None = None
            self.rac = None
            if _fm.RAC:
                try:
                    from vllm._custom_ops import reshape_and_cache_flash as _rac

                    self.rac = _rac
                    self._k_scale = torch.ones(1, dtype=torch.float32, device=device)
                    self._v_scale = torch.ones(1, dtype=torch.float32, device=device)
                except Exception:  # pragma: no cover - older vLLM
                    self.rac = None

            self.fold = bool(_fm.FOLD_SCALES)
            # RMARIN_MERGED_CHECK=1: recompute every layer two-pass from the same pool and report the deviation (eager only)
            self.check = _os.environ.get("RMARIN_MERGED_CHECK") in ("1", "2")
            # 2: additionally REPLACE the merged output by the E1-style reference (diagnostic)
            self.check_replace = _os.environ.get("RMARIN_MERGED_CHECK") == "2"
            self._check_steps = 0
            self.out_scale = None  # [L, 1, H_q, 1] bf16 (s is bf16-exact), lazily sized by query heads
            self.fused_qkv = False
            self._fq_modules: list[Any] | None = None
            self._fq: list[tuple[Any, Any, float, Any]] | None = None
            self.positions: Any = None
            self.shadow = None
            if self.check:
                # E1-units raw ring of slot 0 (all layers) to recompute the two-pass reference
                self.shadow = torch.zeros((self.num_layers, self.num_slots * self.ring_tokens + self.page, heads, width), dtype=self.kv_dtype, device=device)
                self._check_slot0 = False
                self._check_worst = 0.0
            if not self.fold:
                self._vtmp = torch.empty((mt, heads, width), dtype=self.kv_dtype, device=device)
            print({"rmarin_merged_attn": "fastpath", "pool_pages": int(self.pool_k.shape[0]), "bt_all_mib": round(self.bt_all.numel() * 4 / 2**20, 1)}, flush=True)

    def register_fused_qkv(self, modules: list[Any]) -> None:
        """Per-layer (q_norm, k_norm, rotary) modules for the fused epilogue."""

        if not self.merged or self.fold or self.check:
            print({"rmarin_fused_qkv": "disabled", "merged": self.merged, "fold": self.fold, "check": self.check}, flush=True)
            return
        if len(modules) != self.num_layers:
            raise RMarinBackendContractError("fused qkv needs one module triple per layer")
        self._fq_modules = list(modules)
        self.fused_qkv = True
        self._q_out = self.torch.empty((self.max_tokens, 1), dtype=self.kv_dtype, device=self.device)  # resized at bind
        print({"rmarin_fused_qkv": "registered", "layers": len(modules)}, flush=True)

    def _bind_fused_qkv(self, query: Any) -> None:
        torch = self.torch
        bound = []
        for q_norm, k_norm, rotary in self._fq_modules:
            if not getattr(rotary, "is_neox_style", False) or int(getattr(rotary, "rotary_dim", self.head_dim)) != self.head_dim:
                raise RMarinBackendContractError("fused qkv supports full-dim neox rotary only")
            cache = rotary._match_cos_sin_cache_dtype(query)  # the exact cache vLLM's kernel reads
            if cache.dtype != self.kv_dtype or q_norm.weight.dtype != self.kv_dtype or k_norm.weight.dtype != self.kv_dtype:
                raise RMarinBackendContractError("fused qkv expects bf16 rotary cache and norm weights")
            if float(q_norm.eps) != float(k_norm.eps):
                raise RMarinBackendContractError("fused qkv expects one eps for q and k norms")
            bound.append((q_norm.weight, k_norm.weight, float(q_norm.eps), cache.contiguous()))
        self._fq = bound
        heads = int(query.shape[1])
        self._q_out = torch.empty((self.max_tokens, heads, self.head_dim), dtype=self.kv_dtype, device=self.device)
        from .fused_qkv import fused_qkv as _fused_qkv, head_scale_ as _head_scale

        self._fused_qkv_fn = _fused_qkv
        self._head_scale = _head_scale
        self._out_scale_flat = [self.out_scale[layer].reshape(-1).contiguous() for layer in range(self.num_layers)]
        print({"rmarin_fused_qkv": "bound", "eps": bound[0][2], "cache_rows": int(bound[0][3].shape[0])}, flush=True)

    def _resolve_scales(self, query_heads: int) -> None:
        """Bind the loaded per-layer value scales (first forward after load_weights)."""

        pool = self.pool
        pool.invalidate_parameter_caches()
        stacked = pool.sessions[0]._stacked()
        if stacked is None:
            raise RMarinBackendContractError("merged attention requires source_mode=v stacked parameters")
        scale = stacked["scale"]  # [L, 1, H_kv, 1] (kv dtype; the checkpoint values are bf16-exact)
        if not bool(self.torch.isfinite(scale).all()) or bool((scale == 0).any()):
            raise RMarinBackendContractError("value scales must be finite and non-zero for merged attention")
        self.top_scale = pool.top_v_scale()
        self.inv_scale = (1.0 / scale.float()).contiguous()
        self.out_scale = pool.shared_v_scale(query_heads)
        print({"rmarin_merged_attn": "scales bound", "abs_scale_min": float(scale.float().abs().min()), "abs_scale_max": float(scale.float().abs().max()),
               "alpha_min": float(stacked["alpha"].float().min()), "alpha_max": float(stacked["alpha"].float().max())}, flush=True)

    # ------------------------------------------------------------------
    def _wrapper(self, role: str, batch: int) -> Any:
        key = (role, batch)
        wrapper = self._wrappers.get(key)
        if wrapper is not None:
            return wrapper
        torch = self.torch
        max_pages = batch * (self.ring_pages if role == "local" else self.src_pages_per_slot)
        indptr = torch.zeros(batch + 1, dtype=torch.int32, device=self.device)
        indices = torch.zeros(max(max_pages, 1), dtype=torch.int32, device=self.device)
        last = torch.ones(batch, dtype=torch.int32, device=self.device)
        wrapper = self.flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace[role],
            "NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=indptr,
            paged_kv_indices_buffer=indices,
            paged_kv_last_page_len_buffer=last,
        )
        self._wrappers[key] = wrapper
        self._bufs[key] = (indptr, indices, last)
        return wrapper

    def _plan(self, role: str, batch: int, page_lists: list[list[int]], last_lens: list[int], query_heads: int, query_dtype: Any) -> Any:
        torch = self.torch
        wrapper = self._wrapper(role, batch)
        indptr = [0]
        flat: list[int] = []
        for pages in page_lists:
            flat.extend(pages)
            indptr.append(len(flat))
        h_indptr = self._h_indptr[role]
        h_indices = self._h_indices[role]
        h_last = self._h_last[role]
        h_indptr[: batch + 1].copy_(torch.tensor(indptr, dtype=torch.int32))
        h_indices[: len(flat)].copy_(torch.tensor(flat, dtype=torch.int32))
        h_last[:batch].copy_(torch.tensor(last_lens, dtype=torch.int32))
        wrapper.plan(
            h_indptr[: batch + 1],
            h_indices[: len(flat)],
            h_last[:batch],
            query_heads,
            self.kv_heads,
            self.head_dim,
            self.page,
            q_data_type=query_dtype,
            kv_data_type=self.kv_dtype,
            sm_scale=self.sm_scale,
        )
        return wrapper

    def _rotate(self) -> None:
        self._set = (self._set + 1) % self.NSETS
        i = self._set
        self._h_ring_idx = self._hs_ring_idx[i]
        self._h_src_idx = self._hs_src_idx[i]
        self._h_nosrc = self._hs_nosrc[i]
        self._h_ring_idx_np = self._h_ring_idx.numpy()
        self._h_src_idx_np = self._h_src_idx.numpy()
        self._h_nosrc_np = self._h_nosrc.numpy()
        if self.attn == "fa3":
            self._h_cu_q = self._hs_cu_q[i]
            self._h_seq_local = self._hs_seq_local[i]
            self._h_seq_src = self._hs_seq_src[i]
            self._h_bt_local = self._hs_bt_local[i]
            self._h_bt_src = self._hs_bt_src[i]
            self._h_bt_local_np = self._h_bt_local.numpy()
            self._h_bt_src_np = self._h_bt_src.numpy()
        if self.merged:
            self._h_bt = self._hs_bt[i]
            self._h_bt_np = self._h_bt.numpy()
            self._h_nsrc = self._hs_nsrc[i]
            self._h_nsrc_np = self._h_nsrc.numpy()
            self._h_seqk = self._hs_seqk[i]
            self._h_seqk_np = self._h_seqk.numpy()
            if self.paged:
                self._h_ring_flat = self._hs_ring_flat[i]
                self._h_ring_flat_np = self._h_ring_flat.numpy()
                self._h_ring_vals = self._hs_ring_vals[i]
                self._h_ring_vals_np = self._h_ring_vals.numpy()

    def _fa_fill(self, batch: int, local_pages: list[list[int]], local_last: list[int], source_pages: list[list[int]], source_last: list[int]) -> None:
        import numpy as _np

        page = self.page
        hl = self._h_bt_local_np
        hs = self._h_bt_src_np
        seq_l = self._h_seq_local
        seq_s = self._h_seq_src
        for row in range(batch):
            lp = local_pages[row]
            hl[row, : len(lp)] = lp
            seq_l[row] = (len(lp) - 1) * page + local_last[row]
            sp = source_pages[row]
            hs[row, : len(sp)] = sp
            seq_s[row] = (len(sp) - 1) * page + source_last[row]
        self.bt_local[:batch].copy_(self._h_bt_local[:batch], non_blocking=True)
        self.bt_src[:batch].copy_(self._h_bt_src[:batch], non_blocking=True)
        self.seq_local[:batch].copy_(seq_l[:batch], non_blocking=True)
        self.seq_src[:batch].copy_(seq_s[:batch], non_blocking=True)

    def _merged_fill(self, batch: int, local_pages: list[list[int]], local_last: list[int], source_pages: list[list[int]], nosrc: list[bool]) -> None:
        """Per-layer block tables [older pages | ring pages] and key lengths."""

        import numpy as _np

        page = self.page
        hb = self._h_bt_np
        hn = self._h_nsrc_np
        hs = self._h_seqk_np
        torch = self.torch
        if self.paged:
            # source pages: vLLM block table rows (device); ring pages: scatter behind them
            hf = self._h_ring_flat_np
            hv = self._h_ring_vals_np
            n = 0
            W = self.W
            for row in range(batch):
                lp = local_pages[row]
                ns = 0 if nosrc[row] else len(source_pages[row])
                m = len(lp)
                hf[n : n + m] = _np.arange(row * W + ns, row * W + ns + m, dtype=_np.int64)
                hv[n : n + m] = lp
                n += m
                hn[row] = ns
                hs[row] = (ns + m - 1) * page + local_last[row]
            bt = self.step_block_table
            if bt is not None:
                rows = min(batch, int(bt.shape[0]))
                cols = min(self.W, int(bt.shape[1]))
                self.bt_dev[:rows, :cols].copy_(bt[:rows, :cols])
            self.ring_flat_dev[:n].copy_(self._h_ring_flat[:n], non_blocking=True)
            self.ring_vals_dev[:n].copy_(self._h_ring_vals[:n], non_blocking=True)
            self.bt_dev.view(-1).index_copy_(0, self.ring_flat_dev[:n], self.ring_vals_dev[:n])
        else:
            for row in range(batch):
                lp = local_pages[row]
                if nosrc[row]:
                    ns = 0
                else:
                    sp = source_pages[row]
                    ns = len(sp)
                    # older history pages are one contiguous range of the slot
                    hb[row, :ns] = _np.arange(sp[0], sp[0] + ns, dtype=_np.int32)
                hb[row, ns : ns + len(lp)] = lp
                hn[row] = ns
                hs[row] = (ns + len(lp) - 1) * page + local_last[row]
            self.bt_dev[:batch].copy_(self._h_bt[:batch], non_blocking=True)
        self.nsrc[:batch].copy_(self._h_nsrc[:batch], non_blocking=True)
        self.seq_k[:batch].copy_(self._h_seqk[:batch], non_blocking=True)
        base = self.bt_dev[:batch].unsqueeze(0)
        mask = (self.col_ids.unsqueeze(0) < self.nsrc[:batch].unsqueeze(1)).unsqueeze(0)
        torch.where(mask, base, base + self.ring_base, out=self.bt_all[:, :batch])

    # ------------------------------------------------------------------
    def _row_plan(self, slot: int, position: int, n: int = 1) -> tuple[int, int, list[int], int, list[int], int, bool]:
        chunk = self.chunk
        page = self.page
        last = position + n - 1  # last query position of the span (same chunk)
        current_start = (last // chunk) * chunk
        recent_start = max(0, current_start - chunk)
        # ring write index of the first token (consecutive, no wrap inside a chunk) and raw-source index
        ring_tok = slot * self.ring_tokens + (position % self.ring_tokens)
        src_tok = slot * self.src_padded_tokens + position
        # local window [recent_start, last]
        window_len = last + 1 - recent_start
        npages = -(-window_len // page)
        first = (recent_start // page) % self.ring_pages
        base = slot * self.ring_pages
        local_pages = [base + ((first + j) % self.ring_pages) for j in range(npages)]
        local_last = window_len - (npages - 1) * page
        # older history [0, recent_start) from the slot's source slab / V planes
        if recent_start > 0:
            spages = recent_start // page
            sbase = slot * self.src_pages_per_slot
            source_pages = list(range(sbase, sbase + spages))
            source_last = page
            nosrc = False
        else:
            source_pages = [self.scratch_src_page]
            source_last = 1
            nosrc = True
        return ring_tok, src_tok, local_pages, local_last, source_pages, source_last, nosrc

    def prepare_real(self, staged: Any, num_reqs_padded: int, query_heads: int, query_dtype: Any,
                     block_table: Any | None = None, slot_mapping: Any | None = None) -> bool:
        """Host prologue for one uniform decode step; returns eligibility."""

        spans = staged.spans
        if self.debug:
            print({"rmarin_fdp": "prepare_real", "generation": staged.generation, "spans": len(spans), "num_reqs_padded": num_reqs_padded, "last": self.last_real_generation}, flush=True)
        if staged.generation == self.last_real_generation:
            # Duplicate build() for the same runner step (e.g. several attention
            # groups): the prologue already ran; keep the prepared buffers.
            return self.active_generation is not None
        if not spans:
            return False
        uniform = all(span.num_query_tokens == 1 for span in spans)
        if not uniform and (self.attn != "fa3" or not self.prefill_enabled):
            return False
        pool = self.pool
        batch = max(int(num_reqs_padded), len(spans)) if uniform else len(spans)
        total_tokens = batch if uniform else sum(span.num_query_tokens for span in spans)
        if batch > self.max_batch or total_tokens > self.max_tokens:
            return False
        self.last_real_generation = staged.generation
        self._rotate()
        import time as _time

        t0 = _time.perf_counter()
        if self.merged and self.paged:
            if block_table is None or slot_mapping is None:
                raise RMarinBackendContractError("paged source needs vLLM's block table and slot mapping")
            self.step_block_table = block_table
            self.step_slot_mapping = slot_mapping
            pool.step_block_table = block_table
        # slot ownership / resume / boundary commit / catch-up / snapshot
        pool.prepare_forward(staged)
        self.tracker._activated_generation = staged.generation
        t1 = _time.perf_counter()
        ring_np = self._h_ring_idx_np
        src_np = self._h_src_idx_np
        nosrc_np = self._h_nosrc_np
        local_pages: list[list[int]] = []
        local_last: list[int] = []
        source_pages: list[list[int]] = []
        source_last: list[int] = []
        nosrc_rows: list[bool] = []
        cu = [0]
        tok = 0
        max_q = 1
        for row, (span, session) in enumerate(pool._active):
            position = span.num_computed_tokens
            n = span.num_query_tokens
            slot = session.fused_slot
            r_tok, s_tok, lp, ll, sp, sl, ns = self._row_plan(slot, position, n)
            last = position + n - 1
            recent_start = max(0, (last // self.chunk) * self.chunk - self.chunk)
            if session._fused_appended_through != recent_start:
                raise RMarinBackendContractError(
                    f"fast path: source planes at {session._fused_appended_through} != {recent_start}"
                )
            ring_np[tok : tok + n] = range(r_tok, r_tok + n)
            src_np[tok : tok + n] = range(s_tok, s_tok + n)
            nosrc_np[tok : tok + n] = ns
            tok += n
            cu.append(tok)
            if n > max_q:
                max_q = n
            local_pages.append(lp)
            local_last.append(ll)
            source_pages.append(sp)
            source_last.append(sl)
            nosrc_rows.append(ns)
            session.fdp_advance(span)
        for row in range(len(spans), batch):
            ring_np[tok] = self.scratch_ring_token
            src_np[tok] = self.scratch_src_token
            nosrc_np[tok] = True
            tok += 1
            cu.append(tok)
            local_pages.append([self.scratch_ring_page])
            local_last.append(1)
            source_pages.append([self.scratch_src_page])
            source_last.append(1)
            nosrc_rows.append(True)
        self.ring_idx[:tok].copy_(self._h_ring_idx[:tok], non_blocking=True)
        if self.merged and self.paged:
            # raw top-layer K/V go to vLLM's slots; padded tokens (-1) to the null block
            sm = self.step_slot_mapping
            n_sm = min(tok, int(sm.shape[0]))
            if n_sm:
                src = self.src_idx[:n_sm]
                self.torch.clamp(sm[:n_sm], min=0, out=src)
            if n_sm < tok:
                self.src_idx[n_sm:tok].fill_(0)
        else:
            self.src_idx[:tok].copy_(self._h_src_idx[:tok], non_blocking=True)
        if not self.merged:
            self.nosrc_mask[:tok].copy_(self._h_nosrc[:tok], non_blocking=True)
            if self.attn == "fa3":
                self.nosrc_lse_mask[0, :tok].copy_(self._h_nosrc[:tok], non_blocking=True)
        self.tokens = tok
        self.max_q = max_q
        self.uniform = uniform
        if not uniform:
            self._h_cu_q[: batch + 1].copy_(self.torch.tensor(cu, dtype=self.torch.int32))
            self.cu_q_var[: batch + 1].copy_(self._h_cu_q[: batch + 1], non_blocking=True)
        t2 = _time.perf_counter()
        if self.merged and self.check:
            self._check_slot0 = len(pool._active) == 1 and batch == 1
        if self.merged:
            self._merged_fill(batch, local_pages, local_last, source_pages, nosrc_rows)
        elif self.attn == "fa3":
            self._fa_fill(batch, local_pages, local_last, source_pages, source_last)
        else:
            self._local_wrapper = self._plan("local", batch, local_pages, local_last, query_heads, query_dtype)
            self._source_wrapper = self._plan("source", batch, source_pages, source_last, query_heads, query_dtype)
        t3 = _time.perf_counter()
        self.batch = batch
        self.active_generation = staged.generation
        st = STATE.stats
        st["steps"] += 1
        st["pool_ms"] += (t1 - t0) * 1e3
        st["prologue_ms"] += (t3 - t0) * 1e3
        st["plan_ms"] += (t3 - t2) * 1e3
        if st["steps"] % 500 == 0:
            n = st["steps"]
            print({"rmarin_fastpath_stats": {"steps": n, "prologue_ms_avg": round(st["prologue_ms"] / n, 3), "plan_ms_avg": round(st["plan_ms"] / n, 3), "pool_ms_avg": round(st["pool_ms"] / n, 3), "ineligible": st["ineligible"],
                   "snapshots": dict(getattr(self.pool, "snap_stats", {}) or {}), "snap_free": len(getattr(self.pool, "_snap_free", []) or [])}}, flush=True)
        return True

    def prepare_dummy(self, num_reqs: int, query_heads: int, query_dtype: Any) -> None:
        batch = max(1, int(num_reqs))
        if batch > self.max_batch:
            raise RMarinBackendContractError(f"fast path dummy batch {batch} exceeds max {self.max_batch}")
        self._rotate()
        if self.merged and getattr(self, "paged", False):
            self.step_block_table = None
            self.step_slot_mapping = None
        ring_np = self._h_ring_idx_np
        src_np = self._h_src_idx_np
        nosrc_np = self._h_nosrc_np
        ring_np[:batch] = self.scratch_ring_token
        src_np[:batch] = self.scratch_src_token
        nosrc_np[:batch] = True
        self.ring_idx[:batch].copy_(self._h_ring_idx[:batch], non_blocking=True)
        self.src_idx[:batch].copy_(self._h_src_idx[:batch], non_blocking=True)
        if not self.merged:
            self.nosrc_mask[:batch].copy_(self._h_nosrc[:batch], non_blocking=True)
            if self.attn == "fa3":
                self.nosrc_lse_mask[0, :batch].copy_(self._h_nosrc[:batch], non_blocking=True)
        self.tokens = batch
        self.max_q = 1
        self.uniform = True
        if self.merged:
            self._merged_fill(batch, [[self.scratch_ring_page]] * batch, [1] * batch, [[self.scratch_src_page]] * batch, [True] * batch)
        elif self.attn == "fa3":
            self._fa_fill(batch, [[self.scratch_ring_page]] * batch, [1] * batch, [[self.scratch_src_page]] * batch, [1] * batch)
        else:
            self._local_wrapper = self._plan("local", batch, [[self.scratch_ring_page]] * batch, [1] * batch, query_heads, query_dtype)
            self._source_wrapper = self._plan("source", batch, [[self.scratch_src_page]] * batch, [1] * batch, query_heads, query_dtype)
        self.batch = batch
        self.active_generation = DUMMY_GENERATION
        STATE.stats["dummy"] += 1

    # ------------------------------------------------------------------
    def run_layer(self, layer_index: int, query: Any, key: Any, value: Any, output: Any) -> Any:
        """Static per-layer op sequence (captured by the CUDA graph)."""

        batch = self.batch
        tokens = self.tokens
        q = query[:tokens]
        k = key[:tokens]
        v = value[:tokens]
        ring_idx = self.ring_idx[:tokens]
        if self.merged and self.fused_qkv:
            if self.out_scale is None:
                self._resolve_scales(int(q.shape[1]))
            if self._fq is None:
                self._bind_fused_qkv(q)
            qw, kw, eps, cache = self._fq[layer_index]
            top = layer_index == self.num_layers - 1
            q = self._fused_qkv_fn(
                q.flatten(1),
                k.flatten(1), v.flatten(1), self.positions[:tokens], cache, qw, kw, eps,
                self.inv_scale[layer_index].reshape(-1), ring_idx,
                self.ring_k_flat[layer_index], self.ring_v_flat[layer_index],
                self.src_idx[:tokens] if top else None, self.src_k_flat if top else None, self.src_v_raw_flat if top else None,
                q_out=self._q_out[:tokens],
            )
            cu = self.cu_q[: batch + 1] if self.uniform else self.cu_q_var[: batch + 1]
            self._fa(
                q, self.pool_k, self.pool_v,
                max_seqlen_q=self.max_q, cu_seqlens_q=cu, max_seqlen_k=self.max_k,
                seqused_k=self.seq_k[:batch], causal=True, block_table=self.bt_all[layer_index, :batch],
                softmax_scale=self._scale, fa_version=3, out=output[:tokens],
            )
            self._head_scale(output[:tokens], self._out_scale_flat[layer_index])
            return output
        if self.merged:
            if self.out_scale is None:
                self._resolve_scales(int(q.shape[1]))
            if self.fold:
                # folded weights: v already is V_l / s_l; the shared raw slab needs V = v * s_top
                vt = v
                raw_v = v * self.top_scale
            else:
                vt = self._vtmp[:tokens]
                self.torch.mul(v, self.inv_scale[layer_index], out=vt)  # fp32 math, one bf16 rounding
                raw_v = v
            if self.rac is not None:
                # one fused K+V page write (vLLM reshape_and_cache_flash; slot = flat token index)
                self.rac(k, vt, self.ring_k[layer_index], self.ring_v[layer_index], ring_idx, "auto", self._k_scale, self._v_scale)
                if layer_index == self.num_layers - 1:
                    self.rac(k, raw_v, self.src_k, self.src_v_raw, self.src_idx[:tokens], "auto", self._k_scale, self._v_scale)
            else:
                self.ring_k_flat[layer_index].index_copy_(0, ring_idx, k)
                self.ring_v_flat[layer_index].index_copy_(0, ring_idx, vt)
                if layer_index == self.num_layers - 1:
                    src_idx = self.src_idx[:tokens]
                    self.src_k_flat.index_copy_(0, src_idx, k)
                    self.src_v_raw_flat.index_copy_(0, src_idx, raw_v)
            cu = self.cu_q[: batch + 1] if self.uniform else self.cu_q_var[: batch + 1]
            self._fa(
                q, self.pool_k, self.pool_v,
                max_seqlen_q=self.max_q, cu_seqlens_q=cu, max_seqlen_k=self.max_k,
                seqused_k=self.seq_k[:batch], causal=True, block_table=self.bt_all[layer_index, :batch],
                softmax_scale=self._scale, fa_version=3, out=output[:tokens],
            )
            if self.check:
                self._merged_check(layer_index, q, output[:tokens], batch, tokens)
                if batch == 1 and not self.fold:
                    self._merged_check_true(layer_index, q, k, v, output[:tokens], tokens)
                if self._check_slot0 and not self.fold:
                    self.shadow[layer_index].index_copy_(0, ring_idx, v)
                    self._merged_check_e1(layer_index, q, output[:tokens], tokens)
            if not self.fold:
                output[:tokens].mul_(self.out_scale[layer_index])  # s_l is bf16-exact
            return output
        self.ring_k_flat[layer_index].index_copy_(0, ring_idx, k)
        self.ring_v_flat[layer_index].index_copy_(0, ring_idx, v)
        if layer_index == self.num_layers - 1:
            src_idx = self.src_idx[:tokens]
            self.src_k_flat.index_copy_(0, src_idx, k)
            self.src_v_raw_flat.index_copy_(0, src_idx, v)
        if self.attn == "fa3":
            fa = self._fa
            cu = self.cu_q[: batch + 1] if self.uniform else self.cu_q_var[: batch + 1]
            local_out, local_lse = fa(
                q, self.ring_k[layer_index], self.ring_v[layer_index],
                max_seqlen_q=self.max_q, cu_seqlens_q=cu, max_seqlen_k=self.ring_tokens,
                seqused_k=self.seq_local[:batch], causal=True, block_table=self.bt_local[:batch],
                softmax_scale=self._scale, return_softmax_lse=True, fa_version=3,
            )
            source_out, source_lse = fa(
                q, self.src_k, self.src_v_raw if self.shared_v else self.v_planes[layer_index],
                max_seqlen_q=self.max_q, cu_seqlens_q=cu, max_seqlen_k=self.src_padded_tokens,
                seqused_k=self.seq_src[:batch], causal=False, block_table=self.bt_src[:batch],
                softmax_scale=self._scale, return_softmax_lse=True, fa_version=3,
            )
            if self.shared_v:
                if self.scale_q is None:
                    self.scale_q = self.pool.shared_v_scale(int(q.shape[1]))
                source_out.mul_(self.scale_q[layer_index])
            # lse: [H, tokens]; tokens without older history -> -inf so merge keeps local only
            source_lse.masked_fill_(self.nosrc_lse_mask[:, :tokens], self.neg_inf)
            self._merge(output[:tokens], source_out, source_lse, local_out, local_lse)
            return output
        local_out, local_lse = self._local_wrapper.run(
            q, (self.ring_k[layer_index], self.ring_v[layer_index]), return_lse=True
        )
        source_out, source_lse = self._source_wrapper.run(
            q, (self.src_k, self.src_v_raw if self.shared_v else self.v_planes[layer_index]), return_lse=True
        )
        if self.shared_v:
            if self.scale_q is None:
                self.scale_q = self.pool.shared_v_scale(int(q.shape[1]))
            source_out = source_out * self.scale_q[layer_index]
        source_lse.masked_fill_(self.nosrc_mask[:batch].unsqueeze(1), self.neg_inf)
        merged, _lse = self.flashinfer.cascade.merge_state(source_out, source_lse, local_out, local_lse)
        output[:batch].copy_(merged)
        return output


def _merged_check_impl(self: "RMarinDecodeFastPath", layer_index: int, q: Any, out: Any, batch: int, tokens: int) -> None:
    torch = self.torch
    bt = self.bt_all[layer_index, :batch]
    nsrc = self.nsrc[:batch]
    seqk = self.seq_k[:batch]
    cu = self.cu_q[: batch + 1] if self.uniform else self.cu_q_var[: batch + 1]
    idx = (nsrc.unsqueeze(1).to(torch.int64) + torch.arange(self.ring_pages, device=bt.device).unsqueeze(0)).clamp_(max=self.W - 1)
    bt_loc = torch.gather(bt, 1, idx).contiguous()
    seq_loc = seqk - nsrc * self.page
    seq_src = torch.clamp(nsrc * self.page, min=1)
    lo, llse = self._fa(q, self.pool_k, self.pool_v, max_seqlen_q=self.max_q, cu_seqlens_q=cu, max_seqlen_k=self.ring_tokens,
                        seqused_k=seq_loc, causal=True, block_table=bt_loc, softmax_scale=self._scale, return_softmax_lse=True, fa_version=3)
    so, slse = self._fa(q, self.pool_k, self.pool_v, max_seqlen_q=self.max_q, cu_seqlens_q=cu, max_seqlen_k=self.src_padded_tokens,
                        seqused_k=seq_src, causal=False, block_table=bt, softmax_scale=self._scale, return_softmax_lse=True, fa_version=3)
    rows_nosrc = (nsrc == 0)
    if bool(rows_nosrc.any()):
        tok_mask = torch.zeros(tokens, dtype=torch.bool, device=bt.device)
        cu_l = cu.tolist()
        for r in range(batch):
            if bool(rows_nosrc[r]):
                tok_mask[cu_l[r] : cu_l[r + 1]] = True
        slse.masked_fill_(tok_mask.unsqueeze(0), float("-inf"))
    ref = torch.empty_like(out)
    self._merge(ref, so, slse, lo, llse)
    diff = (ref.float() - out.float()).norm() / ref.float().norm().clamp(min=1e-6)
    self._check_steps += 1
    if self._check_steps <= 32 * 6 or diff > 0.02:
        print({"rmarin_merged_check": {"layer": layer_index, "batch": batch, "tokens": tokens, "nsrc": nsrc.tolist()[:4], "seqk": seqk.tolist()[:4], "rel": round(float(diff), 5),
                                      "out_absmax": round(float(out.float().abs().max()), 2), "ref_absmax": round(float(ref.float().abs().max()), 2)}}, flush=True)


RMarinDecodeFastPath._merged_check = _merged_check_impl


def _merged_check_true_impl(self: "RMarinDecodeFastPath", layer_index: int, q: Any, k: Any, v: Any, out: Any, tokens: int) -> None:
    """Chunk-0 prefill of one request (no older history, ring == these tokens):
    compare merged(V/s)*s and the plain bf16 attention over raw V against fp32."""

    torch = self.torch
    if int(self.seq_k[0].item()) != tokens or int(self.nsrc[0].item()) != 0:
        return
    cu = torch.tensor([0, tokens], dtype=torch.int32, device=q.device)
    plain = self._fa(q, k, v, max_seqlen_q=tokens, cu_seqlens_q=cu, max_seqlen_k=tokens, cu_seqlens_k=cu, causal=True, softmax_scale=self._scale, fa_version=3)
    sq = self.out_scale[layer_index].float()  # [1, Hq, 1]
    merged_true = out.float() * sq
    group = q.shape[1] // k.shape[1]
    kk = k.float().repeat_interleave(group, dim=1)
    vv = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), kk) * self._scale
    mask = torch.ones(tokens, tokens, dtype=torch.bool, device=q.device).triu(1)
    scores.masked_fill_(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    exact = torch.einsum("hqk,khd->qhd", probs, vv)
    def rel(a):
        return ((a - exact).norm(dim=(0, 2)) / exact.norm(dim=(0, 2)).clamp(min=1e-6)).view(k.shape[1], -1).mean(1)
    rp, rm = rel(plain.float()), rel(merged_true)
    print({"rmarin_merged_true": {"layer": layer_index, "tokens": tokens, "plain_rel": [round(x, 5) for x in rp.tolist()], "merged_rel": [round(x, 5) for x in rm.tolist()],
                                 "v_absmax": round(float(v.float().abs().max()), 1), "exact_absmax": round(float(exact.abs().max()), 2)}}, flush=True)


RMarinDecodeFastPath._merged_check_true = _merged_check_true_impl


def _merged_check_e1_impl(self: "RMarinDecodeFastPath", layer_index: int, q: Any, out: Any, tokens: int) -> None:
    """Slot-0 request: E1-style two-pass (raw shadow ring, raw slab * s, lse merge) vs merged * s."""

    torch = self.torch
    page = self.page
    ns = int(self.nsrc[0].item())
    seqk = int(self.seq_k[0].item())
    wl = seqk - ns * page
    nl = -(-wl // page)
    bt_row = self.bt_all[layer_index, 0]
    loc_pages_pool = bt_row[ns : ns + nl].to(torch.int64)
    loc_pages_shadow = loc_pages_pool - int(self.ring_base[layer_index].item())  # layer-relative ring page (slot aware)
    kloc = self.pool_k.index_select(0, loc_pages_pool)  # [nl, P, H, D]
    vloc = self.shadow[layer_index].view(-1, page, self.kv_heads, self.head_dim).index_select(0, loc_pages_shadow)
    cu = torch.tensor([0, tokens], dtype=torch.int32, device=q.device)
    bt_local = torch.arange(nl, dtype=torch.int32, device=q.device).view(1, nl)
    lo, llse = self._fa(q, kloc, vloc, max_seqlen_q=tokens, cu_seqlens_q=cu, max_seqlen_k=self.ring_tokens, seqused_k=torch.tensor([wl], dtype=torch.int32, device=q.device),
                        causal=True, block_table=bt_local, softmax_scale=self._scale, return_softmax_lse=True, fa_version=3)
    sq = self.out_scale[layer_index]
    if ns:
        so, slse = self._fa(q, self.pool_k, self.pool_v, max_seqlen_q=tokens, cu_seqlens_q=cu, max_seqlen_k=self.src_padded_tokens,
                            seqused_k=torch.tensor([ns * page], dtype=torch.int32, device=q.device), causal=False, block_table=bt_row[:ns].view(1, ns),
                            softmax_scale=self._scale, return_softmax_lse=True, fa_version=3)
        so = so * sq
        ref = torch.empty_like(out)
        self._merge(ref, so, slse, lo, llse)
    else:
        ref = lo
    merged_true = out.float() * sq.float()
    diff = (ref.float() - merged_true).norm(dim=(0, 2)) / ref.float().norm(dim=(0, 2)).clamp(min=1e-6)
    if self.check_replace:
        # hand the E1-style result to the model: out * s must equal ref -> store ref / s
        out.copy_((ref.float() / sq.float()).to(out.dtype))
    per_kv = diff.view(-1, q.shape[1] // self.kv_heads).mean(1)
    worst = float(per_kv.max())
    gain = (merged_true.norm(dim=(0, 2)) / ref.float().norm(dim=(0, 2)).clamp(min=1e-6)).view(-1, q.shape[1] // self.kv_heads).mean(1)
    if layer_index == 0:
        self._check_worst = 0.0
    self._check_worst = max(self._check_worst, worst)
    if worst > 0.005 or layer_index == self.num_layers - 1:
        print({"rmarin_merged_e1": {"layer": layer_index, "tokens": tokens, "ns_pages": ns, "window": wl, "step_worst": round(self._check_worst, 4),
                                   "rel_per_kv": [round(x, 4) for x in per_kv.tolist()], "gain_per_kv": [round(x, 4) for x in gain.tolist()],
                                   "ref_absmax": round(float(ref.float().abs().max()), 3)}}, flush=True)


RMarinDecodeFastPath._merged_check_e1 = _merged_check_e1_impl


def install(pool: Any, tracker: Any, max_batch: int, sm_scale: float | None, max_tokens: int = 0) -> RMarinDecodeFastPath:
    fast = RMarinDecodeFastPath(pool=pool, tracker=tracker, max_batch=max_batch, sm_scale=sm_scale, max_tokens=max_tokens)
    STATE.model = {"pool": pool, "tracker": tracker, "fastpath": fast}
    pool._fastpath_ref = fast
    STATE.enabled = True
    return fast


def current() -> RMarinDecodeFastPath | None:
    return STATE.model.get("fastpath") if STATE.enabled else None


__all__ = ["RMarinDecodeFastPath", "STATE", "DUMMY_GENERATION", "install", "current"]
