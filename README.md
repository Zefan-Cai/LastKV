# LastKV

Inference code for **Last Layer KV Cache is All You Need**, covering language,
causal text-to-video generation, and novel-view synthesis.

[Project page](https://zefan-cai.github.io/LastKV.github.io/) ·
[Paper](https://zefan-cai.github.io/LastKV.github.io/assets/paper/lastkv.pdf) ·
[Model collection](https://huggingface.co/collections/ZefanCai/lastkv)

## Install

Use Python 3.10 or newer and install PyTorch for your CUDA environment first.
Then install the model families you need:

```bash
git clone https://github.com/Zefan-Cai/LastKV.git
cd LastKV
pip install -e '.[llm,wan,nvs]'
```

The implementations use PyTorch scaled-dot-product attention. A separate
FlashAttention installation is not required. Full Wan video generation uses CUDA;
LLM and NVS also support CPU inference.

## Language models

```bash
python examples/llm_generate.py --model ZefanCai/LastKV-LLM-8B-Base \
  --prompt "Explain why sharing a KV cache can save memory."
```

`main` contains the 8K pretraining checkpoint at step 59,904. Select
`--revision 256k-step-001209` for the subsequent 256K long-context checkpoint.
See [language inference](docs/llm.md) for device, dtype, and cache options.

## Video generation

```bash
python examples/wan_generate.py --model ZefanCai/LastKV-Wan-1.3B \
  --prompt "A red paper boat floating on calm water in warm sunlight." \
  --output outputs/wan-1.3b.mp4
```

Use `ZefanCai/LastKV-Wan-14B` for the larger model. Both default to EMA weights;
raw weights are available as `step-020500-raw` and `step-010000-raw`, respectively.
The loader supports the published single-file and sharded checkpoints and obtains
the separate Wan2.1 text encoder, tokenizer, and VAE assets automatically.
See [video inference](docs/wan.md) for memory use and shorter smoke examples.

## Novel-view synthesis

Create a small procedural camera/image example, then render its target views:

```bash
python examples/nvs_demo/make_demo.py --domain dl3dv --output outputs/nvs-demo
python examples/nvs_render.py --domain dl3dv \
  --manifest outputs/nvs-demo/scene.json --output outputs/nvs-render
```

The DL3DV checkpoint renders at 512 × 512; use `--domain objaverse` in both
commands for the 1024 × 1024 checkpoint. Input manifests contain context RGB
images, pixel-unit intrinsics, and OpenCV camera-to-world matrices; target RGB
images are not needed. See [NVS inference](docs/nvs.md) for the input contract.

## Validation

```bash
pip install -e '.[llm,wan,nvs,dev]'
pytest -q
```

Tests cover strict checkpoint layouts, incremental/chunked LLM decoding,
Wan cache equivalence, and NVS camera/rendering semantics. See the model-specific
docs for validation with the released checkpoints.

## License

LastKV code is released under the [MIT License](LICENSE). Wan-derived components
retain their Apache-2.0 license and original attribution. See
[third-party notices](THIRD_PARTY_NOTICES.md). Model weights and base assets retain
the licenses in their respective Hugging Face repositories.
