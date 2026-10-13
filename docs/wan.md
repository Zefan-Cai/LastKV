# LastKV Wan text-to-video inference

The inference package loads the native LastKV Wan generator tensors strictly,
including the 14B safetensors shard index. `main` selects the published EMA
checkpoint; raw checkpoint branches remain available.

Install the Wan dependencies using the repository's `wan` extra, then run:

```bash
python examples/wan_generate.py \
  --model ZefanCai/LastKV-Wan-1.3B \
  --prompt "A small sailboat moving across a calm lake at sunrise." \
  --output outputs/sailboat.mp4
```

Use `--model ZefanCai/LastKV-Wan-14B` for 14B. To load raw weights, use
`--revision step-020500-raw` for 1.3B or `--revision step-010000-raw` for 14B.
`--model` also accepts a local checkpoint directory. `--base-model` accepts
a local official Wan directory containing `Wan2.1_VAE.pth`,
`models_t5_umt5-xxl-enc-bf16.pth`, and `google/umt5-xxl/`. Otherwise these
assets are downloaded from the corresponding public `Wan-AI/Wan2.1-T2V-*`
repository. Generator weights are downloaded from the selected LastKV revision.

For a short execution check:

```bash
python examples/wan_generate.py \
  --prompt "A red ball rolling on a table." \
  --frames 17 --height 160 --width 160 --steps 2 \
  --output outputs/wan-smoke.mp4
```

The short command checks the real generation path; its reduced resolution and
two sampling steps are not recommended for assessing model quality. Defaults
are 81 frames, 480 × 832, 50 UniPC steps, guidance 5, timestep shift 5, seed 42,
and 16 output frames per second. Height and width must be multiples of 16.
The VAE maps a latent sequence of length `L` to `4 * (L - 1) + 1` video frames.
Generation rounds up to complete three-latent-frame blocks and crops the
decoded video to the requested frame count.

CUDA is required for full generation. The implementation moves UMT5, the
generator, and the VAE onto the selected GPU in successive stages, keeping
inactive models on CPU. Allow sufficient host RAM for the models and download
cache. FP32 files are loaded without modifying them; inference explicitly uses
BF16. The generator weights alone occupy approximately 2.84 GB / 28.58 GB in
BF16 for 1.3B / 14B, with additional GPU memory needed for activations and caches.

```python
from lastkv.wan import LastKVWanPipeline
from lastkv.wan.pipeline import save_video

pipe = LastKVWanPipeline.from_pretrained("ZefanCai/LastKV-Wan-1.3B")
video = pipe.generate("Clouds moving above a mountain lake.")
save_video(video, "outputs/lake.mp4", fps=16)
```

The architecture is `AKV2541ReuseKVCausalWanModel` with the AKV2542 settings:
`akv_mode="reusekv"`, `v_gap=95`, `num_frame_per_block=3`, and
`force_first_local_kv_tokens=1560`. At 480 × 832, 1,560 patch tokens correspond
to the first latent frame. These settings are read from the checkpoint config.
The retention count stays 1,560 when requesting another resolution.

Each generated block is denoised against the frozen clean prefix, then a
zero-timestep pass commits that block for subsequent generation. Positive and
negative prompts have independent histories. The cache shares deep final-layer
KV across layers while retaining each layer's first 1,560 tokens and immediately
previous block separately. Tests compare this representation with the original
per-layer cache, including a multi-block CFG rollout.

Model, UMT5, VAE, and UniPC code are adapted from the recorded evaluator source
at `world_models` commit `b65c563ffac350811444cab1710c6903a6295215`, with the
Alibaba Wan / Diffusers notices preserved. This package contains inference
code and uses public base assets; it does not require the original training
infrastructure. Stock Wan loaders do not select the LastKV architecture.
No generation-quality or benchmark-score equivalence is claimed by the smoke
tests.

## Release validation

On 2026-10-13, the 1.3B EMA checkpoint at revision
`87fe2841b84c4b22d4e7fa2a84ea6fc3e8c1fac7` completed the real prompt-to-MP4
path on an A100 80 GB with PyTorch 2.8.0 and Diffusers 0.36.0: 480 × 832,
17 frames, two UniPC steps, and 16 fps. The output had the expected dimensions
and frame count, finite pixels, and nonconstant spatial and temporal content.
Generation took 20.03 seconds after model loading; peak PyTorch allocation was
11.04 GiB. This is a functionality smoke test, not a quality benchmark.

Both published parameter layouts and strict single-file / indexed loading
are covered by CPU tests. A real 14B GPU run was not completed in this release
check because its download endpoint returned HTTP 429.
