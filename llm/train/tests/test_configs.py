"""Config sanity checks.

Asserts the shipped configs match the backbone and recurrence settings stated in
the paper's long-context pretraining appendix, that every ``model_type`` has an
implementation in this release, and that no internal absolute path survived
packaging.
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIGS = os.path.join(ROOT, "configs")
MODELS = os.path.join(ROOT, "custom_models")

# 12 layers / width 768 / 12 heads at 124M, 24 / 1536 / 24 at 760M;
# RMSNorm eps 1e-6, rotary base 1e6, untied 32,000-token embeddings.
BACKBONE = {
    "124M": {"num_hidden_layers": 12, "hidden_size": 768, "num_heads": 12},
    "760M": {"num_hidden_layers": 24, "hidden_size": 1536, "num_heads": 24},
}
SCALES = ("124M", "760M", "8B")


def _load(scale, name):
    with open(os.path.join(CONFIGS, scale, name)) as f:
        return json.load(f)


def _jsons(scale):
    d = os.path.join(CONFIGS, scale)
    return [f for f in sorted(os.listdir(d)) if f.endswith(".json")]


def test_small_backbones_match_paper():
    for scale, expected in BACKBONE.items():
        names = _jsons(scale)
        assert names, f"no configs for {scale}"
        for name in names:
            cfg = _load(scale, name)
            for key, value in expected.items():
                assert cfg[key] == value, f"{scale}/{name}: {key}={cfg[key]} != {value}"
            assert cfg["vocab_size"] == 32000, name
            assert cfg["rope_theta"] == 1000000, name
            assert cfg["tie_word_embeddings"] is False, name
            assert float(cfg["norm_eps"]) == 1e-6, name
            assert cfg["initializer_range"] == 0.02, name


def test_lastkv_recurrence_settings():
    checked = 0
    for scale in BACKBONE:
        for name in _jsons(scale):
            if not name.startswith("lastkv_"):
                continue
            cfg = _load(scale, name)
            assert cfg["chunk_size"] == 4096, name
            assert cfg["max_position_embeddings"] == 32768, name
            assert cfg["v_gap"] == 95, name          # last layer is the KV source
            assert cfg["window_num"] == 1, name      # blend only the previous chunk
            checked += 1
    assert checked == 6, f"expected 3 KV constructions at each small scale, saw {checked}"


def test_every_model_type_has_an_implementation():
    shipped = {
        d for d in os.listdir(MODELS)
        if os.path.isdir(os.path.join(MODELS, d)) and d != "__pycache__"
    }
    fla_builtin = {"transformer"}
    for scale in SCALES:
        for name in _jsons(scale):
            mt = _load(scale, name)["model_type"]
            assert mt in shipped | fla_builtin, f"{scale}/{name}: no implementation for {mt}"


def test_8b_matches_the_published_checkpoint():
    cfg = _load("8B", "lastkv_8B_reusekv.json")
    assert cfg["model_type"] == "custom_recurrent_marin_akv254_reusekv"
    assert cfg["architectures"] == ["CustomRecurrentMarinAKV254ReuseKVForCausalLM"]


def test_small_scale_decays_embeddings_and_norms():
    """The 124M / 760M recipe applies weight decay to everything but the blend gates."""
    for scale in BACKBONE:
        toml = [f for f in os.listdir(os.path.join(CONFIGS, scale)) if f.endswith(".toml")]
        assert len(toml) == 1, f"{scale}: expected one training TOML, saw {toml}"
        text = open(os.path.join(CONFIGS, scale, toml[0])).read()
        assert "decay_embed_norm = true" in text, f"{scale}/{toml[0]}"


def test_dropped_trainer_options_are_absent():
    """ema, skip-guard, neike and benchmark-loss are not part of this release."""
    dropped = ("ema_beta", "enable_neike", "enable_benchmark_loss", "benchmark_loss_")
    for scale in SCALES:
        d = os.path.join(CONFIGS, scale)
        for name in sorted(os.listdir(d)):
            if not name.endswith(".toml"):
                continue
            text = open(os.path.join(d, name)).read()
            for key in dropped:
                assert key not in text, f"{scale}/{name} still sets {key}"


def test_no_internal_absolute_paths():
    pattern = re.compile(r'"(?:/mnt/|/sensei|/home/|s3://)[^"]*"')
    for scale in SCALES:
        d = os.path.join(CONFIGS, scale)
        for name in sorted(os.listdir(d)):
            if not name.endswith((".toml", ".json")):
                continue
            hits = pattern.findall(open(os.path.join(d, name)).read())
            assert not hits, f"{scale}/{name} has internal paths: {hits}"
