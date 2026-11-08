"""Safetensors weight loading into pre-allocated model parameters.

Streams shards one tensor at a time straight into the model's parameters
(dtype/device conversion happens in ``copy_``), so the checkpoint is never
materialized twice. Follows the R-KV/FlashInfer loader; the only addition is
that every model parameter must be covered (LastKV's per-layer
``prev_chunk_alphas`` / ``deep_v_scale`` / ``q_norm`` / ``k_norm`` included).
"""

from __future__ import annotations

import json
import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn


def shard_files(model_path: str) -> list[str]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        return sorted({os.path.join(model_path, shard) for shard in weight_map.values()})
    single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single):
        return [single]
    files = sorted(glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint under {model_path!r}")
    return files


@torch.no_grad()
def load_weights(model: nn.Module, model_path: str) -> None:
    """Stream all checkpoint tensors into ``model``'s named parameters.

    Keys listed in ``model.packed_map`` (checkpoint name -> (fused parameter
    name, row offset)) are copied into the row slice of the fused parameter;
    a fused parameter counts as loaded only once every source tensor landed.
    Checkpoint keys with no matching parameter are skipped. Every model
    parameter must be covered or a ``ValueError`` is raised.
    """
    params = dict(model.named_parameters())
    packed: dict[str, tuple[str, int]] = dict(getattr(model, "packed_map", {}) or {})
    outstanding: dict[str, set[str]] = {}
    for source, (target, _) in packed.items():
        outstanding.setdefault(target, set()).add(source)
    loaded: set[str] = set()
    for file in shard_files(model_path):
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                param = params.get(name)
                if param is not None:
                    tensor = f.get_tensor(name)
                    if tensor.shape != param.shape:
                        raise ValueError(
                            f"shape mismatch for {name!r}: checkpoint "
                            f"{tuple(tensor.shape)} vs model {tuple(param.shape)}"
                        )
                    param.copy_(tensor)
                    loaded.add(name)
                elif name in packed:
                    target, offset = packed[name]
                    param = params[target]
                    tensor = f.get_tensor(name)
                    rows = tensor.shape[0]
                    if tensor.shape[1:] != param.shape[1:] or offset + rows > param.shape[0]:
                        raise ValueError(
                            f"packed shape mismatch for {name!r}: checkpoint "
                            f"{tuple(tensor.shape)} into rows [{offset}, {offset + rows}) "
                            f"of {target!r} {tuple(param.shape)}"
                        )
                    param[offset : offset + rows].copy_(tensor)
                    outstanding[target].discard(name)
                    if not outstanding[target]:
                        loaded.add(target)
    missing = sorted(set(params) - loaded)
    if missing:
        raise ValueError(f"checkpoint {model_path!r} is missing weights: {missing}")
