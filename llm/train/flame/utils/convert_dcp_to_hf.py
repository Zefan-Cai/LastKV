# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
import io
import os
import tempfile
from datetime import timedelta

import fla  # noqa
import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torchtitan.tools.logging import init_logger, logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import custom_models


def _unsafe_torch_load(*args, **kwargs):
    # These checkpoints come from a trusted internal training pipeline.
    # PyTorch 2.6 changed torch.load to default weights_only=True, which
    # rejects some objects stored in these DCP shards and temp checkpoint files.
    kwargs["weights_only"] = False
    return torch_load_original(*args, **kwargs)


def _save_pretrained(path: str, model: torch.nn.Module):
    try:
        model.save_pretrained(path)
    except RuntimeError as exc:
        if "shared tensors" not in str(exc):
            raise
        logger.warning(
            "Falling back to safe_serialization=False because this trusted "
            "internal checkpoint contains shared tensors."
        )
        model.save_pretrained(path, safe_serialization=False)


def _maybe_disable_tied_embeddings(config, state_dict: dict[str, torch.Tensor]):
    if not getattr(config, "tie_word_embeddings", False):
        return

    input_key = "model.embed_tokens.weight"
    output_key = "lm_head.weight"
    input_weight = state_dict.get(input_key)
    output_weight = state_dict.get(output_key)
    if input_weight is None or output_weight is None:
        return
    if input_weight.shape != output_weight.shape:
        return
    if torch.equal(input_weight, output_weight):
        return

    logger.warning(
        "Checkpoint has distinct input and output embedding weights; "
        "setting tie_word_embeddings=False for the converted model."
    )
    config.tie_word_embeddings = False


@torch.inference_mode()
def save_pretrained(
    path: str,
    step: int,
    config: str,
    tokenizer: str
):
    logger.info(f"Loading the config from {config}")
    config = AutoConfig.from_pretrained(config, trust_remote_code=True)

    logger.info(f"Loading the tokenizer from {tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    logger.info(f"Saving the tokenizer to {path}")
    tokenizer.save_pretrained(path)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = os.path.join(path, f'checkpoint/step-{step}')
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        logger.info(f"Saving the distributed checkpoint to {checkpoint_path}")
        try:
            torch.load = _unsafe_torch_load
            dcp_to_torch_save(checkpoint, checkpoint_path)

            state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
            _maybe_disable_tied_embeddings(config, state_dict)
            logger.info(f"Saving the config to {path}")
            config.save_pretrained(path)

            logger.info(f"Initializing the model from config\n{config}")
            model = AutoModelForCausalLM.from_config(config)
            logger.info(model)
            logger.info("Loading state dict from the checkpoint")

            # Keep the safe globals for legacy checkpoint payloads even though
            # we also disable weights_only for trusted internal checkpoints.
            torch.serialization.add_safe_globals([timedelta, io.BytesIO])
            model.load_state_dict(state_dict)
        finally:
            torch.load = torch_load_original

        logger.info(f"Saving the model to {path}")
        _save_pretrained(path, model)


torch_load_original = torch.load


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser("Convert DCP format model weights to huggingface-style.")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    args = parser.parse_args()
    save_pretrained(args.path, args.step, args.config, args.tokenizer)
