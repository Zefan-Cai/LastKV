"""Render novel views from local posed RGB images and target camera matrices."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from lastkv.nvs import NVSConfig, NVSModel, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("objaverse", "dl3dv"), required=True)
    parser.add_argument("--checkpoint", help="Local model.pt or a release repository ID; defaults to domain release")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", default=Path("outputs/nvs"), type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()
    domain = {"objaverse": "Objaverse", "dl3dv": "DL3DV"}[args.domain]
    config = NVSConfig.for_domain(args.domain)
    model = NVSModel.from_pretrained(
        args.checkpoint or f"ZefanCai/LastKV-NVS-{domain}", config=config,
        device=args.device, dtype=getattr(torch, args.dtype),
    )
    images = model.render(**load_manifest(args.manifest, config.image_size))[0]
    args.output.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        pixels = (image.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8)
        destination = args.output / f"view_{index:04d}.png"
        Image.fromarray(pixels).save(destination)
        print(destination)


if __name__ == "__main__":
    main()
