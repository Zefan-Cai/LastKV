"""Import the small-scale models and build a tiny instance of each.

Needs torch, transformers and fla.  Skips when fla is unavailable, which is the
case on machines that only have the inference extras installed.
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fla", reason="flash-linear-attention is a training dependency")

KV_CONSTRUCTIONS = ["reusekv", "sharekv", "perlayerkv"]


@pytest.mark.parametrize("arm", KV_CONSTRUCTIONS)
def test_tiny_forward(arm):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    import models  # noqa: F401  (registers the model types)

    with open(os.path.join(ROOT, "configs", "124M", f"lastkv_124M_{arm}.json")) as f:
        cfg_dict = json.load(f)
    # shrink to something that fits on CPU while keeping the recurrence shape
    cfg_dict.update(
        hidden_size=64, num_heads=4, num_hidden_layers=4,
        vocab_size=256, chunk_size=8, max_position_embeddings=32,
        fuse_cross_entropy=False, fuse_norm=False,
    )
    config = AutoConfig.for_model(**cfg_dict)
    model = AutoModelForCausalLM.from_config(config).eval()

    input_ids = torch.randint(0, cfg_dict["vocab_size"], (1, 24))
    with torch.no_grad():
        out = model(input_ids=input_ids)
    assert out.logits.shape == (1, 24, cfg_dict["vocab_size"])
    assert torch.isfinite(out.logits).all()
