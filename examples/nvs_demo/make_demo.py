"""Create a procedural plane scene for an NVS installation smoke test."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("objaverse", "dl3dv"), default="dl3dv")
    parser.add_argument("--output", type=Path, default=Path("outputs/nvs_demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    size = 256
    y, x = np.mgrid[:size, :size]
    pixels = np.stack((x, y, 48 + 128 * ((x // 32 + y // 32) % 2)), axis=-1).astype(np.uint8)
    Image.fromarray(pixels).save(args.output / "context.png")
    output_size = 1024 if args.domain == "objaverse" else 512
    camera = np.eye(4)
    camera[2, 3] = -2.0
    context = dict(file_path="context.png", fx=256., fy=256., cx=128., cy=128., c2w=camera.tolist())
    targets = []
    for offset in (0.0, 0.05):
        target_camera = camera.copy()
        target_camera[0, 3] = offset
        targets.append(dict(fx=float(output_size), fy=float(output_size), cx=output_size / 2, cy=output_size / 2, c2w=target_camera.tolist()))
    path = args.output / "scene.json"
    path.write_text(json.dumps({"context": [context], "targets": targets}, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
