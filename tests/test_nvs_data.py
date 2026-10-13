import json

from PIL import Image
import pytest
import torch

from lastkv.nvs import load_manifest, resize_and_crop


def test_resize_crop_preserves_camera_projection():
    image = Image.new("RGB", (200, 100))
    resized, intrinsics = resize_and_crop(image, 100, [80., 90., 100., 50.])
    assert resized.size == (100, 100)
    assert intrinsics == [80., 90., 50., 50.]
    resized, intrinsics = resize_and_crop(image, 200, [80., 90., 100., 50.])
    assert resized.size == (200, 200)
    assert intrinsics == [160., 180., 100., 100.]


def test_resize_crop_intrinsics_use_rounded_image_dimensions():
    image = Image.new("RGB", (641, 480))
    resized, intrinsics = resize_and_crop(image, 512, [300., 300., 320.5, 240.])
    assert resized.size == (512, 512)
    assert intrinsics == pytest.approx([300. * 684 / 641, 320., 256., 256.])


def test_manifest_uses_context_rgb_and_target_camera_only(tmp_path):
    Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(tmp_path / "context.png")
    frame = dict(fx=8., fy=8., cx=4., cy=4., w2c=torch.eye(4).tolist())
    target = dict(frame, file_path="target_image_is_not_needed.png")
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({"context": [dict(frame, file_path="context.png")], "targets": [target]}))
    batch = load_manifest(path, 8)
    assert batch["context_images"].shape == (1, 1, 3, 8, 8)
    assert batch["context_images"].eq(1).all()  # RGBA is composited on white.
    torch.testing.assert_close(batch["target_c2w"], torch.eye(4)[None, None])
    assert set(batch) == {"context_images", "context_intrinsics", "context_c2w", "target_intrinsics", "target_c2w"}
