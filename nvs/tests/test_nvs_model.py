import pytest
import torch

from lastkv.nvs import NVSConfig, NVSModel, compute_rays


def tiny_model(to_kv_mode="shared_last"):
    torch.manual_seed(42)
    return NVSModel(NVSConfig(image_size=8, patch_size=2, dim=16, layers=2, head_dim=4, to_kv_mode=to_kv_mode)).eval()


def inputs():
    images = torch.rand(1, 2, 3, 8, 8)
    intrinsics = torch.tensor([[[8., 8., 4., 4.], [8., 8., 4., 4.]]])
    context = torch.eye(4).repeat(1, 2, 1, 1)
    context[:, 1, 0, 3] = 0.2
    target = context.clone()
    target[:, 0, 0, 3] = -0.1
    target[:, 1, 0, 3] = 0.3
    return images, intrinsics, context, intrinsics.clone(), target


def test_released_checkpoint_architecture():
    with torch.device("meta"):
        model = NVSModel()
    assert len(model.state_dict()) == 196
    assert sum(p.numel() for p in model.parameters()) == 184800768
    assert model.state_dict()["image_tokenizer.1.weight"].shape == (768, 768)
    assert model.state_dict()["transformer_blocks.blocks.23.self_attention.to_kv.weight"].shape == (1536, 768)


@pytest.mark.parametrize("to_kv_mode", ["shared_last", "per_layer"])
def test_targets_are_independent_and_camera_conditioned(to_kv_mode):
    model = tiny_model(to_kv_mode)
    args = inputs()
    together = model.render(*args)
    alone = model.render(*args[:3], args[3][:, 1:], args[4][:, 1:])
    reversed_views = model.render(*args[:3], args[3].flip(1), args[4].flip(1))
    torch.testing.assert_close(together[:, 1:], alone)
    torch.testing.assert_close(together.flip(1), reversed_views)
    assert together.shape == (1, 2, 3, 8, 8)
    assert torch.isfinite(together).all()
    assert together.min() >= 0 and together.max() <= 1
    assert not torch.allclose(together[:, 0], together[:, 1])


@pytest.mark.parametrize("to_kv_mode", ["shared_last", "per_layer"])
def test_context_images_affect_rendering_and_no_cache_leaks_between_calls(to_kv_mode):
    model = tiny_model(to_kv_mode)
    args = inputs()
    first = model.render(*args)
    changed = model.render(1 - args[0], *args[1:])
    repeated = model.render(*args)
    assert not torch.allclose(first, changed)
    torch.testing.assert_close(first, repeated)


def test_rays_use_pixel_centers_camera_rotation_and_translation():
    intrinsics = torch.tensor([[[2., 2., 0.5, 0.5]]])
    c2w = torch.tensor([[[[0., 0., 1., 2.], [0., 1., 0., 3.], [-1., 0., 0., 4.], [0., 0., 0., 1.]]]])
    origins, rays = compute_rays(intrinsics, c2w, 1, 2)
    torch.testing.assert_close(origins[0, 0, :, 0, 0], torch.tensor([2., 3., 4.]))
    torch.testing.assert_close(rays[0, 0, :, 0, 0], torch.tensor([1., 0., 0.]))
    expected = torch.tensor([1., 0., -0.5])
    torch.testing.assert_close(rays[0, 0, :, 0, 1], expected / expected.norm())
    torch.testing.assert_close(rays.norm(dim=2), torch.ones(1, 1, 1, 2))


def test_one_cache_pair_is_shared_across_layers():
    model = tiny_model()
    tokens = torch.randn(1, 16, 16)
    with torch.inference_mode():
        _, cache = model.transformer_blocks(tokens, None, update=True)
        assert len(cache) == 2
        assert cache[0].shape == (1, 4, 16, 4)
        _, read_only = model.transformer_blocks(tokens, cache, update=False)
        assert read_only is cache
        _, extended = model.transformer_blocks(tokens, cache, update=True)
        assert extended[0].shape == (1, 4, 32, 4)


def test_per_layer_projection_stores_one_feature_history():
    model = tiny_model("per_layer")
    tokens = torch.randn(1, 16, 16)
    with torch.inference_mode():
        _, cache = model.transformer_blocks(tokens, None, update=True)
        assert cache.shape == (1, 16, 16)
        _, read_only = model.transformer_blocks(tokens, cache, update=False)
        assert read_only is cache
        _, extended = model.transformer_blocks(tokens, cache, update=True)
        assert extended.shape == (1, 32, 16)
