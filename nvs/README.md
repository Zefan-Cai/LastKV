# Novel-view synthesis

The inference model loads both public releases with their original parameter names:

| Checkpoint | Resolution | Layers / width | Parameters |
| --- | --- | --- | --- |
| `ZefanCai/LastKV-NVS-Objaverse` | 1024 × 1024 | 24 / 768 | 184,800,768 |
| `ZefanCai/LastKV-NVS-DL3DV` | 512 × 512 | 24 / 768 | 184,800,768 |

Each original `model.pt` contains 196 model tensors plus optimizer/training state. Loading uses `weights_only=True`, memory mapping, and strict key/shape validation. These are larger models than the paper's 8-layer NVS comparison; paper PSNR results are not measurements of these released files.

## Run

Install the repository dependencies, then supply a scene manifest:

```bash
python nvs/examples/render.py --domain dl3dv --manifest /path/to/scene.json --output outputs/scene
```

The checkpoint downloads from Hugging Face. Use `--checkpoint /path/to/model.pt` for a local copy. CUDA BF16 is the default; attention uses PyTorch scaled dot-product attention. CPU `--device cpu --dtype float32` is supported for debugging but full-resolution rendering is expensive.

A small procedural input can check installation and execution:

```bash
python nvs/examples/make_demo.py --domain dl3dv
python nvs/examples/render.py --domain dl3dv --manifest outputs/nvs_demo/scene.json
```

The generated plane pattern is a synthetic smoke-test input, not a reconstruction-quality benchmark. Use calibrated photographs from one scene to evaluate visual quality.

## Scene manifest

```json
{
  "context": [
    {
      "file_path": "context.png",
      "fx": 512.0, "fy": 512.0, "cx": 256.0, "cy": 256.0,
      "c2w": [[1,0,0,0], [0,1,0,0], [0,0,1,-2], [0,0,0,1]]
    }
  ],
  "targets": [
    {
      "fx": 512.0, "fy": 512.0, "cx": 256.0, "cy": 256.0,
      "c2w": [[1,0,0,0.05], [0,1,0,0], [0,0,1,-2], [0,0,0,1]]
    }
  ]
}
```

Paths are relative to the manifest. Supply exactly one of `c2w` or `w2c` for every frame; `w2c` is inverted. Cameras use OpenCV axes (right, down, forward), and all views must share a world frame and scene scale. This loader does not recenter or rescale the scene. Context order is preserved and can affect results.

Context intrinsics refer to the original image in pixels. The loader resizes to cover the requested resolution with LANCZOS, center crops, and scales/shifts intrinsics by the same transform. RGBA images are composited on white. Target intrinsics must already refer to the output resolution; targets require no image files.

## Python API

```python
import torch
from lastkv.nvs import NVSConfig, NVSModel, load_manifest

config = NVSConfig.for_domain("dl3dv")
model = NVSModel.from_pretrained("ZefanCai/LastKV-NVS-DL3DV", device="cuda", dtype=torch.bfloat16)
batch = load_manifest("scene.json", config.image_size)
images = model.render(**batch)  # [1, targets, 3, H, W], float32 in [0, 1]
```

For tensor inputs, `context_images` is `[B,C,3,H,W]` RGB in `[0,1]`; `context_intrinsics` and `target_intrinsics` are `[B,C,4]` / `[B,T,4]` ordered `[fx,fy,cx,cy]`; camera-to-world matrices are `[B,C,4,4]` / `[B,T,4,4]`.

With the release preset (`to_kv_mode="shared_last"`), input views update one shared pair of historical key/value tensors using the last layer. Each target view reads that history independently. No target RGB, predicted-image feedback, training loss, LPIPS, dataset service, or author cloud credentials are used. The persistent cache is one pair rather than one pair per layer; attention still uses transient current-view projections.

The reference's alternative `NVSConfig(to_kv_mode="per_layer", ...)` is also implemented. It retains one final-layer feature history and projects that history with each layer's own KV matrix on read. This is a different attention computation; select it only for a checkpoint whose training configuration specifies `per_layer`.

## Implementation provenance and validation limits

The inference equations were checked against the authors' `model/lvsm_ttt_nanoauto_mbv2.py` and `model/blocks/transformer_diffkvcache.py` at source snapshot `01e0ba5e94cb3c9e9bf17edc447500bc118a1129`. The input convention also matches `data/min_dataset.py` in the research checkout. Both published checkpoints strictly match this implementation's parameter names and shapes, including patch size 8 and MLP expansion 3, and their file hashes match the original release records.

The recovered 24-layer ablation1 configurations, `configs/2026.08.15-vgap95-big/vgap95big_abl1_dl3dv_512.yaml` and `vgap95big_abl1_obj_1024.yaml`, confirm 64-dimensional attention heads, `v_gap=95`, `transformer_version="v0"`, `vi_rms_norm=True`, and `pre_vi_source="xi_memory_in"`. Their `use_separate_last_to_kv=False` resolves to `shared_last` in the referenced implementation. The release presets follow this configuration family. Earlier big-model ablation5 configurations use `per_layer` and must not be substituted merely because their tensor shapes match.

The checkpoint files do not embed an exact training commit or effective launch configuration. The source snapshot above identifies the implementation that was inspected and compared; it is not a claim that the serialized checkpoints uniquely establish that training commit or rule out launch-time overrides. No paper quality result is claimed for this inference port.

Both original checkpoints were strictly loaded into this model and the research model. At 16 × 16 input resolution with the complete 24-layer weights, BF16 outputs matched exactly for both checkpoints and both projection modes when both implementations used PyTorch SDPA. The reference's shared-storage optimization was enabled for `shared_last`. Changing the reference's final target RGB did not change its prediction. This verifies the inference port against the selected reference configurations; the reduced-resolution check is not a visual-quality evaluation or a FlashAttention-versus-SDPA comparison. Local tests additionally cover camera rays, intrinsics adjustment, target-order independence, context-image sensitivity, and cache isolation.

Native-resolution CPU FP32 smoke renders also completed at 512 × 512 with DL3DV and 1024 × 1024 with Objaverse, for both projection modes. All produced finite RGB PNGs in `[0,1]` from the procedural plane input. These checks establish execution at the released resolutions; the synthetic plane is outside the Objaverse object domain and does not establish reconstruction quality.
