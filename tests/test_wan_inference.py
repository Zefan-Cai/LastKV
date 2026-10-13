import json

import pytest
import torch
from safetensors.torch import save_file

from lastkv.wan.cache import allocate_caches, history_kv
from lastkv.wan.loading import load_model
from lastkv.wan.model import AKV2541ReuseKVCausalWanModel
from lastkv.wan.pipeline import LastKVWanPipeline, latent_frame_count


def tiny_config():
    return dict(model_type="t2v", patch_size=[1, 2, 2], dim=16, ffn_dim=32,
                num_heads=2, num_layers=3, text_dim=8, text_len=5, freq_dim=16,
                in_dim=2, out_dim=2, qk_norm=True, cross_attn_norm=True,
                akv_mode="reusekv", v_gap=95, num_frame_per_block=3,
                force_first_local_kv_tokens=3, eps=1e-6)


@pytest.mark.parametrize("dim,ffn,heads,layers,keys,parameters", [
    (1536, 8960, 12, 30, 826, 1418997160),
    (5120, 13824, 40, 40, 1096, 14288493184),
])
def test_published_native_parameter_layout(dim, ffn, heads, layers, keys, parameters):
    with torch.device("meta"):
        model = AKV2541ReuseKVCausalWanModel(
            dim=dim, ffn_dim=ffn, num_heads=heads, num_layers=layers,
            num_frame_per_block=3, force_first_local_kv_tokens=1560)
    assert len(model.state_dict()) == keys
    assert sum(t.numel() for t in model.state_dict().values()) == parameters
    assert model.prev_block_alphas.shape == (layers, heads)


@pytest.mark.parametrize("shards", [1, 2])
def test_strict_native_weight_loading(tmp_path, shards):
    torch.manual_seed(10)
    config = tiny_config()
    model = AKV2541ReuseKVCausalWanModel(**config)
    (tmp_path / "config.json").write_text(json.dumps(dict(config, _class_name=type(model).__name__)))
    state = model.state_dict()
    mapping = {}
    for index in range(shards):
        filename = ("diffusion_pytorch_model.safetensors" if shards == 1 else
                    f"diffusion_pytorch_model-{index + 1:05d}-of-{shards:05d}.safetensors")
        tensors = {name: state[name] for name in list(state)[index::shards]}
        save_file(tensors, str(tmp_path / filename))
        mapping.update({name: filename for name in tensors})
    if shards > 1:
        (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    loaded = load_model(tmp_path)
    for name, tensor in loaded.state_dict().items():
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
    assert loaded.force_first_local_kv_tokens == 3
    assert not loaded.freqs.is_meta
    broken = {name: state[name] for name, file in mapping.items() if file == filename}
    broken.pop(next(iter(broken)))
    save_file(broken, str(tmp_path / filename))
    with pytest.raises(ValueError, match="Tensor keys"):
        load_model(tmp_path)


def test_shared_history_matches_native_rollout_and_saves_storage():
    torch.manual_seed(123)
    model = AKV2541ReuseKVCausalWanModel(**tiny_config()).eval()
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.1)
        model.prev_block_alphas.normal_()
    pipeline = LastKVWanPipeline(model, device="cpu", dtype=torch.float32)
    noise = torch.randn(1, 9, 2, 4, 4)
    positive, negative = [torch.randn(4, 8)], [torch.randn(3, 8)]
    native = pipeline.generate_latents(noise, positive, negative, steps=2, progress=False, shared_cache=False)
    shared = pipeline.generate_latents(noise, positive, negative, steps=2, progress=False, shared_cache=True)
    torch.testing.assert_close(shared, native, atol=0, rtol=0)
    assert not torch.equal(shared, noise)
    caches = allocate_caches(model, 1, 24, 12, torch.float32, "cpu")
    assert len({cache["k_storage"].data_ptr() for cache in caches}) == 1
    assert len({cache["prefix_k"].data_ptr() for cache in caches}) == model.num_layers


def test_retained_prefix_and_shared_deep_history_are_layer_correct():
    from lastkv.wan.cache import append_shared

    model = AKV2541ReuseKVCausalWanModel(**tiny_config())
    caches = allocate_caches(model, 1, 24, 12, torch.float32, "cpu")
    with torch.no_grad():
        for layer, cache in enumerate(caches):
            k = torch.full((1, 12, 2, 8), 99.)
            k[:, :3] = layer + 1
            append_shared(cache, k, -k, 0)
        for layer, cache in enumerate(caches):
            k, v = history_kv(cache, 12)
            assert torch.all(k[:, :3] == layer + 1)
            assert torch.all(k[:, 3:] == 99)
            torch.testing.assert_close(v, -k)


def test_requested_frames_round_up_to_complete_latent_blocks():
    assert latent_frame_count(81) == 21
    assert latent_frame_count(17) == 6
    assert latent_frame_count(9) == 3
    assert latent_frame_count(1) == 3
    with pytest.raises(ValueError):
        latent_frame_count(0)
