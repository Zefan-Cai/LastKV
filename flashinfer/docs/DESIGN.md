# LastKV on FlashInfer — Design

A standalone, single-GPU, eager, static-batch engine for the released LastKV
language models and a shape-matched Transformer control, modeled on
[R-KV/FlashInfer](https://github.com/Zefan-Cai/R-KV/tree/main/FlashInfer).
It exists to measure things the vLLM backend (`../vllm/`) cannot measure
cleanly: same-kernel A/B against a Transformer, step time versus resident
sequences at fixed context, explicit capacity, and the prefill/decode
disaggregation hand-off. It is **not** a serving system (no continuous
batching, prefix cache, CUDA graphs or tensor parallelism).

## 1. What the model needs (`../llm/src/lastkv/llm/modeling.py`)

* `deep` — the **final layer's** K/V of every completed chunk, shared by all
  layers. Older chunks are read directly; their V is scaled per layer and head
  by `deep_v_scale[l, h]`.
* `previous[l]` — layer `l`'s own K/V of the previous chunk, blended with the
  final-layer K/V of that chunk: `alpha * deep + (1 - alpha) * previous`
  (`alpha = sigmoid(prev_chunk_alphas[l, h])`, V uses `deep * s`).
* `current[l]` — layer `l`'s K/V of the current partial chunk.
* Per layer, queries attend to `[deep[:old] | blend(previous chunk) | current]`
  with a bottom-right causal mask. At a chunk boundary `deep <- append(current[L-1])`,
  `previous <- current`, `current <- empty`.
* Chunk `n` of **any** layer depends on chunk `n-1` of the **final** layer, so
  prefill is a sequence of chunk steps; layers inside a step are ordinary.
* RoPE first, then per-head q/k RMSNorm (`use_qk_norm`).

## 2. Pools (`engine.py`)

All pools use FlashInfer's NHD paged layout `(k, v)` each `[pages, page, Hkv, D]`
with `page_size = 16` and static per-request page regions; a token's flat slot
is `page * 16 + offset`.

| pool | shape | per request | bytes (8B, 126,976 tokens) |
| --- | --- | --- | --- |
| Transformer `kv_k/kv_v` | `[L, B*seq_pages, 16, Hkv, D]` | `ceil(max_seq_len/16)` pages per layer | 15.5 GiB |
| LastKV `src_k/src_v` | `[B*seq_pages, 16, Hkv, D]` | `ceil(max_seq_len/16)` pages, one layer | 0.48 GiB (4 KiB/token) |
| LastKV `ring_k/ring_v` | `[L, B*2C/16, 16, Hkv, D]` | 128 pages per layer: half 0 = blended previous chunk, half 1 = current chunk | 256 MiB |

Because the ring is layer-major with identical page ids in every layer, the
same `plan()` serves all 32 layers (`run(q, (ring_k[l], ring_v[l]))`); the
source pool is a single tensor shared by all layers. Hence **two plans per
step**: local and source.

## 3. Attention per layer

```
o_local, lse_local = local_wrapper.run(q, (ring_k[l], ring_v[l]), return_lse=True)   # causal over [half0 | half1]
o_src,   lse_src   = src_wrapper.run(q, (src_k, src_v), return_lse=True)             # non-causal over deep[:old]
o_src *= deep_v_scale[l] (expanded to query heads)                                    # V scale commutes with the softmax average
lse_src[rows without older history] = -inf
o = merge_state(o_src, lse_src, o_local, lse_local)                                   # flashinfer.cascade, base-2 LSE
```

Prefill chunk steps use `BatchPrefillWithPagedKVCacheWrapper` (`causal=True`
is bottom-right aligned when `kv_len > qo_len`, so a 1024-token chunk over the
2048-token ring sees the previous chunk fully and itself causally); decode
steps use `BatchDecodeWithPagedKVCacheWrapper` (tensor-core path for GQA 4).
Scaling the source **output** instead of V is exact in real arithmetic (the
softmax weights do not depend on V) and avoids 32 scaled copies of the shared
V; the bf16 rounding point differs from the HF reference.

## 4. Chunk boundary (`_commit`)

For every row whose chunk `n` just completed:

1. copy the final layer's raw K/V of half 1 into the source region at
   `[n*C, (n+1)*C)`;
2. for every layer, write into half 0 (same bf16 op order as `modeling.py`):
   `k = alpha * deep_k + (1 - alpha) * cur_k`,
   `v = alpha * (deep_v * s_l) + (1 - alpha) * cur_v`.

Half 1 is then free for chunk `n+1`; no data moves between halves.

## 5. Prefill and decode loops

* **LastKV prefill** is chunk-synchronous over the static batch: step `n`
  processes tokens `[nC, (n+1)C)` of every prompt longer than `nC` as one
  ragged batch (`qo_indptr`), with local kv = `[half 0 if n >= 1] + half 1`
  and source kv = chunks `0..n-2` when `n >= 2`. Rows whose chunk filled up
  are committed after the step.
* **LastKV decode** writes the new token at half 1 offset `cur`, attends over
  `half0 + cur + 1` ring tokens and `(n_done - 1) * C` source tokens, and
  commits when `cur + 1 == C`. Rows with and without older history can share a
  step (dummy page + `-inf` LSE).
* **Transformer prefill** runs one ragged full-prompt call per request (bounded
  activation memory) or, with `tf_prefill_chunk=C`, chunk steps over its own
  pages — the Table 17 style chunked baseline.
* **Transformer decode**: one plan per step, `run(q, (kv_k[l], kv_v[l]))`.

### plan() hazard

FlashInfer `plan()` stages scheduler metadata through **one pinned host
buffer per wrapper** and copies it to the device asynchronously, stream-ordered
after the kernels already launched. A second `plan()` issued while that copy
is pending overwrites the buffer and silently corrupts the first plan. In
`generate()` the per-step sampling `.tolist()` syncs the host; `score()` and
the prefill chunk loop do not, and we observed sporadic parity degradation up
to NaN before guarding every step: record a CUDA event after a step's plans and
wait on it before the next step plans. The host still runs one step ahead.

## 6. Numerics gate (`tools/parity_hf.py`)

Teacher-forced next-token log-probabilities versus the HF reference
(`lastkv.llm`, or `transformers.LlamaForCausalLM` for the control), bf16, one
H100, 5120 tokens of a real prompt, four prefill/decode splits:

| engine | HF reference | mean abs dlogprob | top-1 agree | greedy 32 |
| --- | --- | --- | --- | --- |
| LastKV (prefill path) | lastkv.llm | 0.016 | 99.3% | 32/32 |
| LastKV (decode path) | lastkv.llm | 0.018–0.019 | 98.9–99.1% | |
| Llama-3.1-8B control (prefill) | LlamaForCausalLM | 0.021 | 98.7% | 10/32 |
| Llama-3.1-8B control (decode) | LlamaForCausalLM | 0.020 | 98.9% | |

The control sits at the same level as LastKV, so the residual is the shared
kernel stack (fused cuBLAS GEMM shapes, FlashInfer norm/RoPE/attention) rather
than the recurrence; the vLLM backend's batch-invariant reference path reaches
0.004–0.007 on the same prompts. HF's own prefill-vs-decode disagreement is of
the same order, so gates are on the mean, never the max.

Greedy outputs are not batch-invariant: the GEMM kernel changes with the batch
size, so a near-tie can flip. `tools/batch_check.py` compares single-request
and joint greedy continuations; every two-row combination of 1,000 / 2,024 /
2,500 / 3,048-token prompts (rows with and without older history, decode-time
commits of different chunks in one step) matched 48/48, while a batch of four
identical 2,024-token rows diverged at token 18 and 36 in two rows — the
Llama control shows the same effect at batch 2. A bookkeeping bug diverges
within the first few tokens.

Benchmarks and capacity numbers are in [RESULTS_H100.md](RESULTS_H100.md).

## 7. Known limits

* Eager: ~13–20 ms of host work per decode step dominates at batch 1 (R-KV's
  7B eager engine runs 68 tok/s); single-stream decode numbers are only
  meaningful as same-engine ratios or at larger batches.
* Static batch, no admission, no prefix cache / session resume.
* No tensor parallelism; both 8B models fit one 80 GB GPU with ~60 GiB left
  for KV.
* `page_size` must divide `chunk_size`.
