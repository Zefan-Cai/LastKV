"""Model configuration read straight from ``config.json`` (no ``AutoConfig``).

The released LastKV checkpoints use ``model_type``
``custom_recurrent_marin_akv254_reusekv`` without an ``auto_map`` entry, so
``transformers.AutoConfig`` cannot resolve them; the fields this engine needs
are plain JSON anyway. The same reader serves the Llama control checkpoint.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

LASTKV_ARCH = "CustomRecurrentMarinAKV254ReuseKVForCausalLM"
LLAMA_ARCH = "LlamaForCausalLM"


@dataclass(frozen=True)
class ModelConfig:
    family: str  # "lastkv" | "transformer"
    num_layers: int
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: dict | None
    tie_word_embeddings: bool
    attention_bias: bool
    mlp_bias: bool
    max_position_embeddings: int
    # LastKV-only fields (defaults match the released checkpoints).
    chunk_size: int = 1024
    use_qk_norm: bool = True
    use_nope: bool = False
    deep_kv_rescale: bool = True
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def group_size(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @property
    def kv_bytes_per_token_per_layer(self) -> int:
        """K+V bytes for one token in one layer at bf16."""
        return 2 * self.num_kv_heads * self.head_dim * 2

    @classmethod
    def from_model_dir(cls, model_path: str) -> "ModelConfig":
        with open(os.path.join(model_path, "config.json")) as f:
            cfg = json.load(f)
        arch = (cfg.get("architectures") or [""])[0]
        if arch == LASTKV_ARCH:
            family = "lastkv"
            if cfg.get("akv254_kv_mode", "reusekv") != "reusekv":
                raise ValueError("only final-layer KV reuse (reusekv) is supported")
            if cfg.get("deep_kv_rescale", True) and cfg.get("deep_rescale_mode", "v") != "v":
                raise ValueError("only deep_rescale_mode='v' is supported")
        elif arch == LLAMA_ARCH:
            family = "transformer"
        else:
            raise ValueError(f"unsupported architecture {arch!r}")
        head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
        if cfg["num_attention_heads"] % cfg["num_key_value_heads"]:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        rope_scaling = cfg.get("rope_scaling")
        if rope_scaling is not None:
            rope_scaling = dict(rope_scaling)
            rope_scaling.setdefault("rope_type", rope_scaling.get("type", "default"))
        return cls(
            family=family,
            num_layers=int(cfg["num_hidden_layers"]),
            hidden_size=int(cfg["hidden_size"]),
            num_q_heads=int(cfg["num_attention_heads"]),
            num_kv_heads=int(cfg["num_key_value_heads"]),
            head_dim=int(head_dim),
            intermediate_size=int(cfg["intermediate_size"]),
            vocab_size=int(cfg["vocab_size"]),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-5)),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),
            rope_scaling=rope_scaling,
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            attention_bias=bool(cfg.get("attention_bias", False)),
            mlp_bias=bool(cfg.get("mlp_bias", False)),
            max_position_embeddings=int(cfg.get("max_position_embeddings", 0)),
            chunk_size=int(cfg.get("chunk_size", 1024)),
            use_qk_norm=bool(cfg.get("use_qk_norm", True)) if family == "lastkv" else False,
            use_nope=bool(cfg.get("use_nope", False)),
            deep_kv_rescale=bool(cfg.get("deep_kv_rescale", True)),
            raw=cfg,
        )


@dataclass
class GenOutput:
    """Per-request result of ``Engine.generate``."""

    token_ids: list[int]
    num_prefill_tokens: int
    num_output_tokens: int
    finish_reason: str  # "stop" | "length"
