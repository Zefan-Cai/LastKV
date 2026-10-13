"""Strict native Wan safetensors loading, including the 14B shard index."""

import inspect
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from .layers import rope_params
from .model import AKV2541ReuseKVCausalWanModel


def checkpoint_directory(model_id, revision="main"):
    path = Path(model_id).expanduser()
    if path.is_dir():
        return path
    return Path(snapshot_download(str(model_id), revision=revision, allow_patterns=[
        "config.json", "diffusion_pytorch_model*.safetensors", "diffusion_pytorch_model.safetensors.index.json"]))


def load_model(model_id, revision="main", device="cpu", dtype=torch.float32):
    path = checkpoint_directory(model_id, revision)
    config = json.loads((path / "config.json").read_text())
    if config.get("_class_name") != "AKV2541ReuseKVCausalWanModel":
        raise ValueError("Expected a LastKV AKV2541ReuseKVCausalWanModel checkpoint.")
    parameters = inspect.signature(AKV2541ReuseKVCausalWanModel.__init__).parameters
    kwargs = {key: value for key, value in config.items() if key in parameters}
    for key in ("akv_mode", "v_gap", "num_frame_per_block", "force_first_local_kv_tokens"):
        if key not in kwargs:
            raise ValueError(f"Missing required LastKV architecture setting: {key}")
    with torch.device("meta"):
        model = AKV2541ReuseKVCausalWanModel(**kwargs)
    expected = model.state_dict()
    index_path = path / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        mapping = json.loads(index_path.read_text())["weight_map"]
        if set(mapping) != set(expected):
            raise ValueError("Safetensors index does not match the exact native model keys.")
    else:
        mapping = {key: "diffusion_pytorch_model.safetensors" for key in expected}
    state = {}
    for filename in sorted(set(mapping.values())):
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError("Weight index contains a non-local shard filename.")
        tensors = load_file(str(path / filename), device="cpu")
        if set(tensors) != {key for key, value in mapping.items() if value == filename}:
            raise ValueError(f"Tensor keys differ from the shard index: {filename}")
        for key, tensor in tensors.items():
            if tensor.shape != expected[key].shape:
                raise ValueError(f"Checkpoint tensor shape mismatch: {key}")
            if tensor.dtype != torch.float32:
                raise ValueError(f"Expected the published FP32 tensor: {key}")
        state.update(tensors)
    model.load_state_dict(state, strict=True, assign=True)
    head_dim = model.dim // model.num_heads
    model.freqs = torch.cat([rope_params(1024, head_dim - 4 * (head_dim // 6)),
                             rope_params(1024, 2 * (head_dim // 6)),
                             rope_params(1024, 2 * (head_dim // 6))], dim=1)
    return model.eval().requires_grad_(False).to(device=device, dtype=dtype)
