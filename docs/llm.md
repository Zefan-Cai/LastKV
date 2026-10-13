# LastKV language-model inference

The released 8B model uses Marin/Llama weights with LastKV chunk recurrence.
Both revisions load **419 tensors / 8,030,269,952 parameters** with their original
names. Install the repository's LLM dependencies, including **Transformers
4.57.6** and its compatible `huggingface-hub<1` dependency.

```bash
python examples/llm_generate.py --device cuda --dtype bfloat16 \
  --prompt "The key to understanding the universe is" --max-new-tokens 64
```

For the long-context model, add `--revision 256k-step-001209`. `--model` also
accepts a local directory containing the config, tokenizer and all four HF
safetensors shards. CPU inference is supported with `--device cpu --dtype
float32`; the full FP32 model alone needs approximately 32 GB of memory.

| Revision | Training stage | Saved step | Configured context |
| --- | --- | ---: | ---: |
| `main` / `8k-step-059904` | Continued pretraining and cooldown | 59,904 | 8,192 |
| `256k-step-001209` | Subsequent long-context training | 1,209 | 262,144 |

The published tensors are BF16. `--dtype float32` expands those same values; it
does not recover the training checkpoint's original FP32 precision. The model
is a base text-completion model; the example does not apply a chat template.

```python
import torch
from transformers import AutoTokenizer
from lastkv.llm import load_model

repo = "ZefanCai/LastKV-LLM-8B-Base"
revision = "main"
tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision, fix_mistral_regex=False)
model = load_model(repo, revision=revision, device="cuda", dtype=torch.bfloat16)
inputs = tokenizer("Once upon a time", return_tensors="pt").to("cuda")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Loading is strict: missing, unexpected or mismatched checkpoint tensors raise
an error. Importing `lastkv.llm` also registers the custom config/model with
Transformers. Stock Llama attention and its standard `DynamicCache` do not
implement this checkpoint's recurrence.
The example preserves the released Marin tokenizer by disabling Transformers'
unrelated Mistral regex rewrite.

## Cache and attention

Chunks contain 1,024 tokens. Older completed chunks share the final layer's
KV; the immediately preceding chunk blends each layer's own KV with that
shared KV. Learned per-head scales apply to incoming deep values. The current
chunk retains its own per-layer KV until its boundary, including when a prompt
or a decoding call ends midway through a chunk.

`past_key_values` is a `LastKVCache`; pass **only new tokens** when calling
`forward` directly with it. Cache state belongs to that sequence and is not
stored on the model. Use unpadded prompts; generate padded or different-length
prompts separately. Packed-document training inputs, training losses and
attention/hidden-state diagnostics are outside this inference interface.

PyTorch SDPA is the default attention backend and chooses available CUDA
kernels. `--attention-backend eager` provides a materialized attention reference
and can use substantially more memory. RoPE, post-RoPE Q/K normalization and
the previous-chunk blend follow the archived training implementation.

## Source provenance

The inference implementation is adapted from `Zefan-Cai/llm` at
`3d914d3217c2f3ea1513852e621c146099ece394` (8K) and
`580a6c2a46b68a0312d5e7df8ae1d30f8734c812` (long context). The latter adds
packed-document training masks; ordinary unpadded inference has the same
recurrence in both revisions. Training-only FLA/Liger kernels and flame
infrastructure are excluded. Transformers supplies the Llama MLP, RMSNorm and
RoPE components. The inference cache adds incremental decoding to the original
full-prefix generation path.

Tests compare full prefill and differently segmented decoding against an
independent eager transcription of the original chunk recurrence, including
chunk-boundary crossings, nontrivial blend/scale parameters and strict sharded
loading. Configured context lengths describe training configuration; they do
not guarantee quality or memory availability at every sequence length.

Both published revisions also passed strict loading and greedy text generation
with their complete weights on CPU FP32 (PyTorch 2.8.0, Transformers 4.57.6).
This short execution check does not exercise the full 8K or 256K context window.
