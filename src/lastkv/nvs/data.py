"""Local image/JSON inputs; camera preprocessing matches the research loader."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def resize_and_crop(image, target_size, intrinsics):
    """Resize to cover, center crop, and adjust pixel-unit [fx,fy,cx,cy]."""
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    height, width = target_size
    old_width, old_height = image.size
    scale = max(width / old_width, height / old_height)
    new_width, new_height = round(old_width * scale), round(old_height * scale)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left, top = (new_width - width) // 2, (new_height - height) // 2
    image = image.crop((left, top, left + width, top + height))
    fx, fy, cx, cy = intrinsics
    scale_x, scale_y = new_width / old_width, new_height / old_height
    return image, [fx * scale_x, fy * scale_y, cx * scale_x - left, cy * scale_y - top]


def _camera(frame):
    intrinsics = [frame[key] for key in ("fx", "fy", "cx", "cy")]
    if ("c2w" in frame) == ("w2c" in frame):
        raise ValueError("Each camera needs exactly one of c2w or w2c")
    c2w = torch.tensor(frame.get("c2w", frame.get("w2c")), dtype=torch.float32)
    if "w2c" in frame:
        c2w = torch.linalg.inv(c2w)
    return intrinsics, c2w


def load_manifest(path, image_size):
    """Read {context:[image cameras], targets:[cameras]} for a single scene.

    Context intrinsics refer to original image pixels. Target intrinsics already
    refer to the requested output resolution. Paths are relative to the manifest.
    """
    path = Path(path)
    manifest = json.loads(path.read_text())
    images, context_intrinsics, context_c2w = [], [], []
    for frame in manifest["context"]:
        intrinsics, c2w = _camera(frame)
        with Image.open(path.parent / frame["file_path"]) as image:
            image, intrinsics = resize_and_crop(image, image_size, intrinsics)
            if image.mode == "RGBA":
                rgb = Image.new("RGB", image.size, (255, 255, 255))
                rgb.paste(image, mask=image.getchannel("A"))
                image = rgb
            else:
                image = image.convert("RGB")
            images.append(torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255)
        context_intrinsics.append(intrinsics)
        context_c2w.append(c2w)
    target_intrinsics, target_c2w = [], []
    for frame in manifest["targets"]:
        intrinsics, c2w = _camera(frame)
        target_intrinsics.append(intrinsics)
        target_c2w.append(c2w)
    if not images or not target_intrinsics:
        raise ValueError("Manifest needs at least one context image and one target camera")
    return {
        "context_images": torch.stack(images)[None],
        "context_intrinsics": torch.tensor(context_intrinsics, dtype=torch.float32)[None],
        "context_c2w": torch.stack(context_c2w)[None],
        "target_intrinsics": torch.tensor(target_intrinsics, dtype=torch.float32)[None],
        "target_c2w": torch.stack(target_c2w)[None],
    }
