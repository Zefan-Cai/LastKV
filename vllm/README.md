# LastKV vLLM serving

A vLLM 0.25 backend for the released language models
([`ZefanCai/LastKV-LLM-8B-Base`](https://huggingface.co/ZefanCai/LastKV-LLM-8B-Base),
architecture `CustomRecurrentMarinAKV254ReuseKVForCausalLM`). It serves both
revisions through vLLM's OpenAI-compatible server with chunk-aligned scheduling,
CUDA-graph decoding, one attention call per layer over vLLM's paged KV blocks,
prefix-cache reuse across requests, and fused per-layer kernels.

## Install

Use an environment separate from the other domains: vLLM 0.25.0 needs newer
Transformers and huggingface-hub releases than the `llm` extra pins.

```bash
pip install vllm==0.25.0
export PYTHONPATH=$PWD/vllm/src
```

The backend runs one model per GPU (no tensor or pipeline parallelism) and has
been validated on H100 (SM90) GPUs.

## Serve

```bash
python vllm/examples/serve.py --revision 256k-step-001209 \
  --max-model-len 131072 --port 8000
```

`--model` also accepts a local checkpoint directory. Arguments the launcher
does not define are passed to `vllm serve`. The launcher fixes the settings the
backend depends on (16-token blocks, chunked prefill with a one-chunk prefill
budget, and the LastKV scheduler and attention backend) and enables the fast
path: decode CUDA graphs, prefix caching and asynchronous scheduling.

`--eager` selects the reference path instead, without fused kernels, CUDA
graphs or prefix caching. It reserves scratch memory per sequence outside
`--gpu-memory-utilization`, so it defaults to 4 sequences instead of 32.

The released models are base models, so use the completions endpoint:

```bash
curl http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "ZefanCai/LastKV-LLM-8B-Base",
  "prompt": "The key to understanding the universe is",
  "max_tokens": 64, "temperature": 0}'
```

## Memory

Each sequence slot reserves a two-chunk ring of per-layer K/V (256 MiB for the
8B model), so `--max-num-seqs` sets a fixed reservation. Older history keeps
only the final layer's K/V, 4 KiB per token, in vLLM's paged blocks; the rest
of GPU memory after weights and rings becomes that page pool. A Transformer of
the same shape stores 128 KiB per token.

## Fast-path switches

`RMARIN_FAST=1`, which the launcher sets unless `--eager` is given, enables the
switches below; each can also be set explicitly before launching.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RMARIN_FAST` | `1` (`0` with `--eager`) | Enable the fast bundle below |
| `RMARIN_CUDAGRAPH` | `1` (`0` with `--eager`) | Allow vLLM `FULL_DECODE_ONLY` CUDA graphs |
| `RMARIN_LINEAR` | `cublas` if fast, else `batch_invariant` | GEMM primitive |
| `RMARIN_FUSED_NORM` | fast | Fused CUDA RMSNorm and Q/K norm |
| `RMARIN_PAGED_RING` | fast | Paged layer-local K/V rings read without copies |
| `RMARIN_PREFILL_BACKEND` | `fa3` if fast, else `fa2` | FlashInfer prefill backend for chunk passes |
| `RMARIN_FDP` / `RMARIN_FPP` | fast / `1` | Static decode / prefill op sequence (graph-capturable) |
| `RMARIN_ASYNC_SCHED` | fast | Asynchronous scheduler (needs `--async-scheduling`) |
| `RMARIN_SHARED_V` | fast | Keep only the raw final-layer K/V and scale source outputs per layer |
| `RMARIN_MERGED_ATTN` | fast | One FlashAttention call per layer over older pages and the ring |
| `RMARIN_PAGED_SOURCE` | fast | Store older history in vLLM's paged blocks (needs prefix caching) |
| `RMARIN_RAC` | fast | vLLM `reshape_and_cache_flash` for ring K/V writes |
| `RMARIN_FUSED_QKV` | fast | Triton kernels for RoPE, Q/K norm, value scaling and cache writes |
| `RMARIN_SESSION_RESUME` | `1` | Reuse cached prefixes of earlier requests |
| `RMARIN_MAX_SEQS` | `256` if fast, else `32` | Upper bound on `--max-num-seqs` |

## Numerics

The reference path (`--eager`) uses deterministic batch-invariant GEMMs and
FP32 RMSNorm reductions. The fast path uses standard BF16 kernels (cuBLAS
GEMMs, fused norms, FlashAttention 3), so its logprobs differ slightly from the
reference path, as for any BF16 serving stack. The fused kernels are
bit-identical to the operations they replace; the differences come from the
GEMM, norm and attention kernel choices.

On one H100, teacher-forced logprobs over 1.5K- and 5K-token prompts were
compared with the Transformers implementation in `llm/` for both revisions.
The reference path and the fast path with the default 32 sequences both differ
by a mean |Δlogprob| of 0.004–0.007, with 99.3–99.7% top-1 agreement, and
reproduce its 32-token greedy continuations of those prompts. With
`--max-num-seqs 64`, positions after the first chunk differ more: the mean over
the 5K-token prompt rises to 0.011, with 99.2% top-1 agreement. The cause has
not been identified.

## Source provenance

The backend is adapted from `Zefan-Cai/lastkv_inference` at
`c733cec2b699812cf989926436de781caf4a5505`. The package is renamed to
`lastkv.vllm` and contains only the modules the vLLM server imports; the
evaluation harness, SGLang backend and benchmark tools are excluded.
