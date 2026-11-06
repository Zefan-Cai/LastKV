# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoConfig, AutoModelForCausalLM

import fla  # noqa
from torchtitan.tools.logging import init_logger, logger


def _load_safetensors_state_dict(model: str) -> dict[str, torch.Tensor]:
    snapshot_path = Path(
        snapshot_download(
            model,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "model-*.safetensors",
                "model.safetensors.index.json",
            ],
        ),
    )
    index_path = snapshot_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        shard_names = ["model.safetensors"]

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard_path = snapshot_path / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing safetensors shard: {shard_path}")
        logger.info(f"Loading safetensors shard {shard_path}")
        state_dict.update(load_safetensors_file(shard_path, device="cpu"))
    return state_dict


def _pack_trinity_afmoe_state_dict(
    state_dict: dict[str, torch.Tensor],
    config,
) -> dict[str, torch.Tensor]:
    """Pack HF AFMoE per-expert tensors into flame's grouped Trinity tensors."""

    num_layers = int(config.num_hidden_layers)
    num_dense_layers = int(getattr(config, "num_dense_layers", 0) or 0)
    num_experts = int(config.num_experts)
    num_key_value_heads = int(config.num_key_value_heads)

    for layer_idx in range(num_dense_layers, num_layers):
        prefix = f"model.layers.{layer_idx}.mlp"
        gate_up_key = f"{prefix}.experts.gate_up_proj"
        down_key = f"{prefix}.experts.down_proj"
        if gate_up_key in state_dict and down_key in state_dict:
            continue

        gate_up_tensors = []
        down_tensors = []
        for expert_idx in range(num_experts):
            expert_prefix = f"{prefix}.experts.{expert_idx}"
            gate = state_dict.pop(f"{expert_prefix}.gate_proj.weight")
            up = state_dict.pop(f"{expert_prefix}.up_proj.weight")
            down = state_dict.pop(f"{expert_prefix}.down_proj.weight")
            gate_up_tensors.append(torch.cat([gate, up], dim=0))
            down_tensors.append(down)

        state_dict[gate_up_key] = torch.stack(gate_up_tensors, dim=0).contiguous()
        state_dict[down_key] = torch.stack(down_tensors, dim=0).contiguous()
        state_dict.setdefault(
            f"{prefix}.expert_bias",
            torch.zeros(num_experts, dtype=torch.float32),
        )

    for layer_idx in range(num_layers):
        state_dict.setdefault(
            f"model.layers.{layer_idx}.self_attn.prev_chunk_alphas",
            torch.zeros(num_key_value_heads, dtype=torch.float32),
        )
    return state_dict


def _maybe_cast_state_dict(
    state_dict: dict[str, torch.Tensor],
    torch_dtype: str | torch.dtype | None,
) -> dict[str, torch.Tensor]:
    if torch_dtype in (None, "auto"):
        return state_dict
    return {
        key: value.to(dtype=torch_dtype) if torch.is_floating_point(value) else value
        for key, value in state_dict.items()
    }


@torch.inference_mode()
def convert_hf_weights(
    model: str,
    checkpoint: str,
    trust_remote_code: bool = False,
    torch_dtype: str | torch.dtype | None = "auto",
):
    logger.info(f"Loading model from {model}")
    config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    if getattr(config, "model_type", None) == "afmoe":
        state_dict = _load_safetensors_state_dict(model)
        state_dict = _pack_trinity_afmoe_state_dict(state_dict, config)
        state_dict = _maybe_cast_state_dict(state_dict, torch_dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        state_dict = model.state_dict()

    logger.info(f"Writing to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    DCP.save(state_dict, storage_writer=storage_writer)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(description="Convert huggingface-style model weights to DCP format.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom Hugging Face model code while loading the source model.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="dtype passed to from_pretrained, e.g. auto, bfloat16, float16, or float32.",
    )
    args = parser.parse_args()

    dtype = args.torch_dtype
    if dtype in {"none", "None", ""}:
        dtype = None
    elif dtype != "auto":
        dtype = getattr(torch, dtype)
    convert_hf_weights(args.model, args.checkpoint, args.trust_remote_code, dtype)
