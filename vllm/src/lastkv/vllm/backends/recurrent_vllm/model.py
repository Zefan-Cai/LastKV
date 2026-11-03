"""Out-of-tree vLLM model for ``CustomRecurrentMarinAKV254ReuseKV``.

This is a compatibility implementation for vLLM 0.25.0.  It deliberately
uses ordinary per-layer paged attention and performs an exact cache transition
at recurrent chunk boundaries.  It therefore removes HF's full-prefix decode
recomputation, but does not yet realize the architecture's theoretical shared
deep-KV memory reduction.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.sequence import IntermediateTensors

from .cache_rewrite import rewrite_completed_boundaries
from .vllm_compat import installed_runtime_profile, resolve_attention_symbols


_RUNTIME_PROFILE = installed_runtime_profile()
Attention, AttentionType = resolve_attention_symbols(_RUNTIME_PROFILE)

if _RUNTIME_PROFILE.raw_source_write_strategy == "unified_kv_cache_update":
    from vllm.model_executor.layers.attention.attention import (
        unified_kv_cache_update,
    )
elif _RUNTIME_PROFILE.raw_source_write_strategy != "attention_forward_discard":
    raise RuntimeError("unsupported recurrent raw-source cache write strategy")


class TrainingEagerRMSNorm(nn.Module):
    """Match the recurrent training model's evaluation-time RMSNorm.

    The frozen training implementation deliberately routes every model and
    Q/K norm through an eager float32 variance calculation during evaluation.
    vLLM's stock Llama layer instead uses a fused add+RMSNorm operation.  The
    two are mathematically equivalent, but their bf16 rounding differs enough
    to perturb close greedy decisions after several recurrent chunks.  Keep
    vLLM's ``(hidden, residual)`` interface while preserving the training
    implementation's explicit residual addition and cast order.
    """

    def __init__(self, hidden_size: int, *, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            residual = residual + hidden_states
            hidden_states = residual
        dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        # This order is intentional: training casts the normalized activation
        # back to bf16 before multiplying by the (bf16) learned weight.
        output = self.weight.to(dtype=dtype) * normalized.to(dtype)
        if residual is None:
            return output
        return output, residual


def _source_layer_map(config: Any) -> tuple[int, ...]:
    layer_count = int(config.num_hidden_layers)
    gap = getattr(config, "v_gap", None)
    if isinstance(gap, bool) or not isinstance(gap, int):
        raise ValueError("recurrent vLLM model requires integer v_gap")
    return tuple(
        max(min(index + gap, layer_count - 1), 0) for index in range(layer_count)
    )


def _layer_scale(config: Any, layer_index: int, kv_heads: int) -> list[float]:
    raw = getattr(config, "deep_v_scale_init", 1.0)
    entry = raw[layer_index] if isinstance(raw, (list, tuple)) else raw
    if isinstance(entry, (list, tuple)):
        if len(entry) != kv_heads:
            raise ValueError(
                f"layer {layer_index} deep_v_scale_init has {len(entry)} "
                f"values, expected {kv_heads}"
            )
        return [float(value) for value in entry]
    return [float(entry)] * kv_heads


def _layer_vnormqk_fill(config: Any, layer_index: int, kv_heads: int) -> float:
    """Match training's scalar initialization for ``DeepVNorm.weight``."""

    values = _layer_scale(config, layer_index, kv_heads)
    return sum(values) / len(values)


class RecurrentMarinAttention(LlamaAttention):
    """Llama projection/attention plus qk-norm and raw-source cache capture."""

    def __init__(
        self,
        config: Any,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        quant_config: Any | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: Any | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super_kwargs = dict(
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=prefix,
            attn_type=attn_type,
        )
        if _RUNTIME_PROFILE.vllm_version == "0.11.0":
            super_kwargs["rope_theta"] = rope_theta
            super_kwargs["rope_scaling"] = rope_scaling
        super().__init__(**super_kwargs)
        if cache_config is None:
            raise ValueError("recurrent P0 requires an explicit KV cache config")
        self.cache_block_size = int(cache_config.block_size)
        self.layer_index = extract_layer_index(prefix)
        if self.num_kv_heads != self.total_num_kv_heads:
            raise ValueError("recurrent P0 model requires tensor_parallel_size=1")
        if self.attn.attn_backend.get_name() != "FLASH_ATTN":
            raise ValueError("recurrent P0 requires the FLASH_ATTN backend")

        self.prev_chunk_alphas = nn.Parameter(torch.zeros(self.num_kv_heads))
        if bool(getattr(config, "use_qk_norm", False)):
            self.q_norm = TrainingEagerRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = TrainingEagerRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.deep_kv_rescale = bool(getattr(config, "deep_kv_rescale", False))
        self.deep_rescale_mode = getattr(config, "deep_rescale_mode", "v")
        self.deep_scale_frozen = bool(getattr(config, "deep_scale_frozen", False))
        if self.deep_rescale_mode not in {"v", "vnorm", "vnormqk"}:
            raise ValueError(
                "recurrent P0 supports deep scale mode v, vnorm, or vnormqk"
            )
        scale = torch.tensor(
            _layer_scale(config, self.layer_index, self.num_kv_heads),
            dtype=torch.float32,
        )
        if self.deep_kv_rescale and self.deep_rescale_mode == "vnormqk":
            self.deep_v_norm = TrainingEagerRMSNorm(
                self.head_dim, eps=config.rms_norm_eps
            )
            with torch.no_grad():
                self.deep_v_norm.weight.fill_(
                    _layer_vnormqk_fill(config, self.layer_index, self.num_kv_heads)
                )
        elif self.deep_kv_rescale and not self.deep_scale_frozen:
            # Training stores one learned gain per KV head for v/vnorm.
            self.deep_v_scale = nn.Parameter(scale)
        else:
            # Frozen variants do not carry a checkpoint key for this value.
            self.register_buffer("deep_v_scale", scale, persistent=False)
        self.deep_norm_epsilon = 1e-6

        source_layers = set(_source_layer_map(config))
        self.source_cache: Attention | None = None
        if self.layer_index in source_layers:
            self.source_cache = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                attn_type=attn_type,
                prefix=f"{prefix}.recurrent_raw_source",
            )
            if self.source_cache.attn_backend.get_name() != "FLASH_ATTN":
                raise ValueError(
                    "recurrent raw-source cache requires the FLASH_ATTN backend"
                )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query, key = self.rotary_emb(positions, query, key)
        query = self.q_norm(query.view(-1, self.num_heads, self.head_dim)).reshape(
            -1, self.q_size
        )
        key = self.k_norm(key.view(-1, self.num_kv_heads, self.head_dim)).reshape(
            -1, self.kv_size
        )

        if self.source_cache is not None:
            # Keep the unblended, unscaled source-layer KV in a separate paged
            # cache.  The boundary transition reads it after all layers finish.
            if _RUNTIME_PROFILE.raw_source_write_strategy == "unified_kv_cache_update":
                # vLLM 0.25 exposes a side-effect-only cache update.
                unified_kv_cache_update(
                    key.view(-1, self.num_kv_heads, self.head_dim),
                    value.view(-1, self.num_kv_heads, self.head_dim),
                    self.source_cache.layer_name,
                )
            else:
                # vLLM 0.11 has no public side-effect-only update.  Calling
                # the auxiliary Attention writes the raw K/V to its own pages;
                # its output is intentionally discarded.  This bounded port is
                # therefore correct-by-construction but incurs one diagnostic
                # attention invocation on each referenced source layer.
                self.source_cache(query, key, value)

        attention_output = self.attn(query, key, value)
        output, _ = self.o_proj(attention_output)
        return output


class RecurrentMarinDecoderLayer(LlamaDecoderLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        if _RUNTIME_PROFILE.decoder_layer_strategy == "injected_attention_type":
            super().__init__(
                vllm_config=vllm_config,
                prefix=prefix,
                attn_layer_type=RecurrentMarinAttention,
            )
        elif _RUNTIME_PROFILE.decoder_layer_strategy == "manual_llama_decoder":
            # vLLM 0.11's LlamaDecoderLayer cannot inject an attention class.
            # Reconstruct the small stock layer shell instead of constructing
            # and replacing a stock Attention: replacement would leave a stale
            # entry in compilation_config.static_forward_context.
            nn.Module.__init__(self)
            cache_config = vllm_config.cache_config
            quant_config = vllm_config.quant_config
            self.hidden_size = config.hidden_size
            rope_theta = getattr(config, "rope_theta", 10000)
            rope_scaling = getattr(config, "rope_scaling", None)
            if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None
            ):
                rope_scaling = dict(rope_scaling)
                rope_scaling["original_max_position_embeddings"] = (
                    config.original_max_position_embeddings
                )
            max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
            attention_bias = getattr(config, "attention_bias", False) or getattr(
                config, "bias", False
            )
            bias_o_proj = attention_bias
            if hasattr(config, "qkv_bias"):
                attention_bias = config.qkv_bias
            attn_type = (
                AttentionType.DECODER
                if getattr(config, "is_causal", True)
                else AttentionType.ENCODER_ONLY
            )
            self.self_attn = RecurrentMarinAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(
                    config,
                    "num_key_value_heads",
                    config.num_attention_heads,
                ),
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                bias=attention_bias,
                bias_o_proj=bias_o_proj,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
                attn_type=attn_type,
            )
            self.mlp = LlamaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                bias=getattr(config, "mlp_bias", False),
                prefix=f"{prefix}.mlp",
            )
        else:  # pragma: no cover - guarded by compatibility profile tests.
            raise RuntimeError("unsupported recurrent Llama decoder strategy")
        self.input_layernorm = TrainingEagerRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = TrainingEagerRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class RecurrentMarinModel(LlamaModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        self.chunk_size = int(getattr(config, "chunk_size"))
        self.source_layer_by_target = _source_layer_map(config)
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("recurrent P0 requires tensor_parallel_size=1")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("recurrent P0 requires pipeline_parallel_size=1")
        if bool(vllm_config.cache_config.enable_prefix_caching):
            raise ValueError(
                "recurrent cache mutation is incompatible with prefix caching"
            )
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=RecurrentMarinDecoderLayer,
        )
        # P0 is TP=PP=1, so this is always the real final norm rather than a
        # pipeline placeholder.
        self.norm = TrainingEagerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **extra_layer_kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        output = super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **extra_layer_kwargs,
        )
        rewrite_completed_boundaries(
            self.layers,
            chunk_size=self.chunk_size,
            source_layer_by_target=self.source_layer_by_target,
        )
        return output


class RecurrentMarinAKV254ReuseKVForCausalLM(LlamaForCausalLM):
    """Class registered against the training export's architecture string."""

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = RecurrentMarinDecoderLayer,
    ) -> RecurrentMarinModel:
        del layer_type
        return RecurrentMarinModel(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(weights)


__all__ = ["RecurrentMarinAKV254ReuseKVForCausalLM"]
