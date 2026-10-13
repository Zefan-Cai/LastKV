"""Compare inference against the pinned training model's chunk recurrence."""

import math

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from lastkv.llm import (
    CustomRecurrentMarinAKV254ReuseKVConfig as Config,
    CustomRecurrentMarinAKV254ReuseKVForCausalLM as Model,
    LastKVCache,
    load_model,
)


def tiny_model(backend="sdpa"):
    torch.manual_seed(19)
    config = Config(
        hidden_size=32, intermediate_size=48, num_attention_heads=4,
        num_key_value_heads=2, num_hidden_layers=3, head_dim=8,
        vocab_size=71, chunk_size=4, max_position_embeddings=64,
        bos_token_id=1, eos_token_id=None, pad_token_id=0,
        deep_v_scale_init=[[0.3, 0.7], [0.8, 1.2], [1.1, 0.9]],
        rope_theta=500000.0, attn_implementation=backend,
    )
    model = Model(config).eval()
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.prev_chunk_alphas.copy_(torch.tensor([-1.2 + i * 0.2, 0.8 - i * 0.3]))
            # Nonuniform norm gains make applying norm before vs after RoPE
            # observably different, as it is in the trained checkpoints.
            layer.self_attn.q_norm.weight.copy_(torch.linspace(0.7, 1.3, config.head_dim))
            layer.self_attn.k_norm.weight.copy_(torch.linspace(1.2, 0.8, config.head_dim))
    return model


@torch.no_grad()
def reference_logits(model, tokens):
    """Materialize per-layer history independently of the inference cache.

    This is the unpadded, unfused eager path of llm@580a6c2a:
    older chunks use final-layer KV, the preceding chunk blends final/local
    KV, and the current chunk uses its own causally masked KV. Norm follows
    RoPE; deep V is scaled before its blend. No LastKVCache methods are used.
    """
    decoder, config = model.model, model.config
    batch, length = tokens.shape
    histories = []
    outputs = []
    for start in range(0, length, config.chunk_size):
        hidden = decoder.embed_tokens(tokens[:, start:start + config.chunk_size])
        q_len = hidden.shape[1]
        positions = torch.arange(start, start + q_len, device=tokens.device)[None]
        cos, sin = decoder.rotary_emb(hidden, positions)
        local = []
        for index, layer in enumerate(decoder.layers):
            attention = layer.self_attn
            normalized = layer.input_layernorm(hidden)
            q = attention.q_proj(normalized).view(batch, q_len, config.num_attention_heads, config.head_dim).transpose(1, 2)
            k = attention.k_proj(normalized).view(batch, q_len, config.num_key_value_heads, config.head_dim).transpose(1, 2)
            v = attention.v_proj(normalized).view(batch, q_len, config.num_key_value_heads, config.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            q, k = attention.q_norm(q), attention.k_norm(k)
            local.append((k, v))
            all_keys, all_values = [], []
            scale = attention.deep_v_scale.view(1, -1, 1, 1)
            alpha = attention.prev_chunk_alphas.sigmoid().view(1, -1, 1, 1)
            for past_index, past in enumerate(histories):
                source_k, source_v = past[-1]
                if past_index == len(histories) - 1:
                    own_k, own_v = past[index]
                    source_k = alpha * source_k + (1 - alpha) * own_k
                    source_v = alpha * (source_v * scale) + (1 - alpha) * own_v
                else:
                    source_v = source_v * scale
                all_keys.append(source_k)
                all_values.append(source_v)
            all_keys.append(k)
            all_values.append(v)
            keys = torch.cat(all_keys, dim=2).repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=1)
            values = torch.cat(all_values, dim=2).repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=1)
            scores = q @ keys.transpose(-1, -2) / math.sqrt(config.head_dim)
            mask = torch.arange(start + q_len, device=tokens.device)[None] > positions.T
            scores.masked_fill_(mask, torch.finfo(scores.dtype).min)
            probabilities = scores.float().softmax(-1).to(q.dtype)
            attended = (probabilities @ values).transpose(1, 2).reshape(batch, q_len, -1)
            hidden = hidden + attention.o_proj(attended)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        histories.append(local)
        outputs.append(hidden)
    return model.lm_head(decoder.norm(torch.cat(outputs, dim=1)))


@pytest.mark.parametrize("backend", ["sdpa", "eager"])
def test_full_prefill_matches_original_chunk_recurrence(backend):
    model = tiny_model(backend)
    tokens = torch.randint(3, 71, (2, 15))
    with torch.inference_mode():
        actual = model(tokens, use_cache=False).logits
        expected = reference_logits(model, tokens)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("segments", [[1] * 15, [3, 2, 7, 3], [4, 4, 4, 3], [9, 6]])
def test_incremental_decoding_matches_reference_across_boundaries(segments):
    model = tiny_model()
    tokens = torch.randint(3, 71, (1, sum(segments)))
    cache, start, logits = None, 0, []
    with torch.inference_mode():
        expected = reference_logits(model, tokens)
        for size in segments:
            result = model(tokens[:, start:start + size], past_key_values=cache, use_cache=True)
            logits.append(result.logits)
            cache = result.past_key_values
            start += size
            assert cache.get_seq_length() == start
            assert cache.deep is None or cache.deep[0].shape[2] == start // 4 * 4
            assert all(kv is None or kv[0].shape[2] == start % 4 for kv in cache.current)
            assert all(kv is None or kv[0].shape[2] == 4 for kv in cache.previous)
    torch.testing.assert_close(torch.cat(logits, dim=1), expected, rtol=2e-5, atol=2e-6)


def test_generate_matches_full_prefix_replay():
    model = tiny_model()
    prompt = torch.tensor([[1, 5, 17]])
    replay = prompt.clone()
    with torch.inference_mode():
        generated = model.generate(prompt, max_new_tokens=10, do_sample=False, use_cache=True)
        for _ in range(10):
            logits = reference_logits(model, replay)
            replay = torch.cat((replay, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    assert torch.equal(generated, replay)


def test_published_8b_parameter_layout_without_allocating_weights():
    config = Config(hidden_size=4096, intermediate_size=14336, num_attention_heads=32,
                    num_key_value_heads=8, num_hidden_layers=32, head_dim=128,
                    vocab_size=128256, rms_norm_eps=1e-5, tie_word_embeddings=False)
    with torch.device("meta"):
        model = Model(config)
    state = model.state_dict()
    assert len(state) == 419
    assert sum(t.numel() for t in state.values()) == 8_030_269_952
    assert state["model.layers.31.self_attn.prev_chunk_alphas"].shape == (8,)
    assert state["model.layers.0.self_attn.deep_v_scale"].shape == (8,)
    assert state["model.layers.0.self_attn.q_norm.weight"].shape == (128,)
    assert state["lm_head.weight"].shape == (128256, 4096)


def test_strict_loading_and_local_shard_index(tmp_path):
    model = tiny_model()
    model.save_pretrained(tmp_path, max_shard_size="10KB")
    assert (tmp_path / "model.safetensors.index.json").is_file()
    loaded = load_model(tmp_path, dtype=torch.float32)
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
    # A missing learned history scale must never be replaced by an initialized
    # default, as the upstream training loader permits when grafting a base.
    import json
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    key = "model.layers.0.self_attn.deep_v_scale"
    filename = index["weight_map"].pop(key)
    shard = load_file(tmp_path / filename)
    del shard[key]
    save_file(shard, tmp_path / filename)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="strict LastKV model match"):
        load_model(tmp_path)


def test_cache_reorder_and_padding_rejection():
    model = tiny_model()
    tokens = torch.randint(3, 71, (2, 7))
    order = torch.tensor([1, 0])
    with torch.inference_mode():
        cache = model(tokens[:, :5], use_cache=True).past_key_values
        cache.reorder_cache(order)
        actual = model(tokens[order, 5:], past_key_values=cache, use_cache=True).logits
        expected = reference_logits(model, tokens[order])[:, 5:]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    with pytest.raises(ValueError, match="unpadded"):
        model(tokens[:1], attention_mask=torch.tensor([[0, 1, 1, 1, 1, 1, 1]]))
    with pytest.raises(ValueError, match="Cache architecture"):
        model(tokens[:1], past_key_values=LastKVCache(3, 5))
    with pytest.raises(ValueError, match="Packed-document"):
        model(tokens[:1], cu_seqlens=torch.tensor([0, 3, 7]))
