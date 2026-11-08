# LastKV on FlashInfer

A standalone FlashInfer engine for the released LastKV language models
([`ZefanCai/LastKV-LLM-8B-Base`](https://huggingface.co/ZefanCai/LastKV-LLM-8B-Base))
and a shape-matched Transformer control (Llama-3.1-8B), modeled on
[R-KV/FlashInfer](https://github.com/Zefan-Cai/R-KV/tree/main/FlashInfer):
single GPU, eager, one static request batch per `generate()` call, no CUDA
graphs, no prefix cache, no tensor parallelism. Both model families share the
GEMM / norm / RoPE / sampling kernels, so a LastKV-vs-Transformer number from
this engine differs only by the attention path and the KV layout.

Use `../vllm/` for serving. Use this directory for controlled measurements:
same-kernel prefill and decode comparisons, step time versus resident
sequences at a fixed context, explicit KV capacity, and the prefill/decode
disaggregation hand-off (the LastKV state of a request is two tensors).

## Layout

```
flashinfer/
├── src/lastkv/flashinfer/
│   ├── config.py      # ModelConfig from config.json (no AutoConfig), GenOutput
│   ├── models.py      # Llama / LastKV decoder on FlashInfer kernels
│   ├── loader.py      # streaming safetensors loader (fused qkv / gate_up)
│   └── engine.py      # Engine: pools, chunk-synchronous prefill, two-pass decode, score()
├── tools/
│   ├── parity_hf.py           # teacher-forced log-prob parity vs the HF reference
│   ├── smoke.py               # load, generate, print timings
│   ├── batch_check.py         # mixed-length batch vs single-request consistency
│   ├── kernel_check.py        # FlashInfer paged decode respects last_page_len
│   └── pd_transfer_bench.py   # GPU-to-GPU hand-off bytes and time, LastKV vs Transformer
├── benchmark/                 # prefill / decode / plan() benchmarks (one JSON line last)
├── docs/DESIGN.md
└── requirements-flashinfer.txt
```

## Install

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r flashinfer/requirements-flashinfer.txt    # torch 2.10.0 cu128 + flashinfer 0.6.12
export PYTHONPATH=$PWD/flashinfer/src:$PWD/llm/src          # llm/src only for the HF parity reference
```

Validated on H100 80GB (SM90) with driver CUDA 12.8; FlashInfer JIT-compiles
its kernels on first use (`CUDA_HOME` must point at a CUDA 12.x toolkit).

## Run

```python
from lastkv.flashinfer import Engine

engine = Engine("/path/to/LastKV-LLM-8B-Base", max_batch_size=8, max_seq_len=32768)
outs = engine.generate([token_ids], max_new_tokens=64, temperature=0.0)
print(outs[0].token_ids, engine.stats)   # prefill_seconds, decode_seconds, peak_memory_bytes, kv_bytes_*
```

The same `Engine` class serves a Llama checkpoint (`family == "transformer"`).
Prompts are token ids; give every request in a batch a distinct first token if
you want distinct outputs. `engine.score(ids, prefill_tokens=k)` returns
teacher-forced next-token log-probabilities with the first `k` tokens through
the prefill path and the rest through single-token decode steps.

```bash
python flashinfer/tools/smoke.py --model /path/to/LastKV-LLM-8B-Base --prompt-json prompt.json --tokens 8000 --new 64
python flashinfer/tools/parity_hf.py --model /path/to/LastKV-LLM-8B-Base --prompt-json prompt.json --tokens 5120 --prefill-tokens 4000 --greedy 32
CUDA_VISIBLE_DEVICES=0,1 python flashinfer/tools/pd_transfer_bench.py --lens 24576,32768,65536,126976
```

`prompt.json` is `{"prompt": "<text>"}`, tokenised with the checkpoint's
`tokenizer.json`.

## How LastKV runs here

* **Source pool**: one shared copy of the final layer's K/V for every
  completed chunk, 4 KiB per token for the 8B model.
* **Ring pool**: per layer and request, two 1,024-token halves — the
  alpha-blended previous chunk and the current chunk — 256 MiB per sequence.
* **Per layer**: a causal pass over the ring plus a non-causal pass over the
  older source chunks, merged with `flashinfer.cascade.merge_state`; the source
  output is scaled by the layer's `deep_v_scale`. Two `plan()` calls per step
  serve all 32 layers.
* **Prefill** is chunk-synchronous (chunk *n* of any layer needs chunk *n-1*
  of the final layer), so a 126K prompt is 124 sequential 1,024-token steps.

See [docs/DESIGN.md](docs/DESIGN.md) for pools, the chunk boundary, the
`plan()` hazard, and the numerics gate.

## Numerics

Teacher-forced over 5,120 tokens of a real prompt on one H100 (bf16), with the
sequence split between the prefill and decode paths at four points:

| engine | reference | mean abs dlogprob | top-1 agreement | greedy 32 |
| --- | --- | --- | --- | --- |
| LastKV, prefill path | `lastkv.llm` | 0.016 | 99.3% | 32/32 |
| LastKV, decode path | `lastkv.llm` | 0.018–0.019 | 98.9–99.1% | |
| Llama-3.1-8B control, prefill | `LlamaForCausalLM` | 0.021 | 98.7% | 10/32 |
| Llama-3.1-8B control, decode | `LlamaForCausalLM` | 0.020 | 98.9% | |

The control sits at the same level as LastKV, so the residual comes from the
shared kernel stack (fused cuBLAS GEMM shapes, FlashInfer norm / RoPE /
attention), not from the recurrence.

## Limits

Eager execution makes single-stream decode host-bound (about 12 ms per step
for LastKV and 8 ms for the control at 8K context on H100); compare decode
only as same-engine ratios or at larger batches. No continuous batching,
admission, prefix cache or session resume. Both 8B models fit one 80 GB GPU;
a 126,976-token Transformer sequence needs 15.5 GiB of KV, a LastKV sequence
0.73 GiB.
