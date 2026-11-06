"""Complete Marin-8B (Llama-3.1 architecture) under repo control.

Marin is stock Llama, so the modeling subclasses transformers' Llama classes
1:1 — every weight name matches the marin-8b-base-phoenix checkpoint. Fork the
parent class bodies in here if the architecture ever needs local surgery.
"""

from torch import nn
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    ALL_ATTENTION_FUNCTIONS,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..chunked_ce import chunked_linear_cross_entropy
from ..liger_flce import liger_linear_cross_entropy
from ..qk_norm import eager_rmsnorm

from .config_custom_marin import CustomMarinConfig


def _llama_rms_norm_reset_parameters(self):
    nn.init.ones_(self.weight)


if not hasattr(LlamaRMSNorm, "reset_parameters"):
    LlamaRMSNorm.reset_parameters = _llama_rms_norm_reset_parameters


_MODE_ROUTED_FUSED_RMS_NORM = None


def _mode_routed_fused_rms_norm_cls():
    """fla fused Triton RMSNorm for TRAINING; eval routes to eager_rmsnorm with
    the same weight — the fla kernel hangs deterministically in the 448-rank
    benchmark-eval path (see qk_norm.py). Weight name stays 'weight', so
    checkpoints are byte-compatible with LlamaRMSNorm."""
    global _MODE_ROUTED_FUSED_RMS_NORM
    if _MODE_ROUTED_FUSED_RMS_NORM is None:
        from fla.modules import RMSNorm as FlaRMSNorm

        class ModeRoutedFusedRMSNorm(FlaRMSNorm):
            def forward(self, x):
                if self.training:
                    return super().forward(x)
                return eager_rmsnorm(x, self.weight, self.eps)

        _MODE_ROUTED_FUSED_RMS_NORM = ModeRoutedFusedRMSNorm
    return _MODE_ROUTED_FUSED_RMS_NORM


def build_marin_rms_norm(config, hidden_size: int) -> nn.Module:
    if getattr(config, "fuse_norm", False):
        return _mode_routed_fused_rms_norm_cls()(hidden_size, eps=config.rms_norm_eps)
    return LlamaRMSNorm(hidden_size, eps=config.rms_norm_eps)


class CustomMarinAttention(LlamaAttention):
    """LlamaAttention plus optional per-head qk-norm.

    Placement mirrors the recurrent arm exactly: RMSNorm on query/key AFTER
    rotary embedding. With use_qk_norm=false this is numerically identical
    to stock LlamaAttention.
    """

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self._fla_qk = False
        self._qk_eps = config.rms_norm_eps
        if getattr(config, "use_qk_norm", False):
            norm_cls = LlamaRMSNorm
            if getattr(config, "use_fla_qk_norm", False):
                # fla fused RMSNorm for TRAINING only (+5.8% at 448); eval
                # routes to eager_rmsnorm with the same weight — the fla
                # kernel hangs in the 448-rank benchmark-eval path
                from fla.modules import RMSNorm as norm_cls  # noqa: N813
                self._fla_qk = True
            self.q_norm = norm_cls(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = norm_cls(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # base marin checkpoints predate qk-norm: default the fresh norm
        # weights to ones so partial loads are deterministic (same pattern
        # as the recurrent arm's prev_chunk_alphas)
        if not isinstance(self.q_norm, nn.Identity):
            state_dict.setdefault(prefix + "q_norm.weight", torch.ones_like(self.q_norm.weight))
            state_dict.setdefault(prefix + "k_norm.weight", torch.ones_like(self.k_norm.weight))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if self._fla_qk and not self.training:
            query_states = eager_rmsnorm(query_states, self.q_norm.weight, self._qk_eps)
            key_states = eager_rmsnorm(key_states, self.k_norm.weight, self._qk_eps)
        else:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CustomMarinDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = CustomMarinAttention(config, layer_idx)
        self.input_layernorm = build_marin_rms_norm(config, config.hidden_size)
        self.post_attention_layernorm = build_marin_rms_norm(config, config.hidden_size)


class CustomMarinModel(LlamaModel):
    config_class = CustomMarinConfig

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            CustomMarinDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = build_marin_rms_norm(config, config.hidden_size)
        self.post_init()


class CustomMarinForCausalLM(LlamaForCausalLM):
    config_class = CustomMarinConfig

    def __init__(self, config):
        super().__init__(config)
        # the inherited __init__ hardcodes LlamaModel; swap in our model so
        # CustomMarinAttention (qk-norm) is actually instantiated
        self.model = CustomMarinModel(config)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ):
        # Same chunked linear CE (+ optional z-loss) as the recurrent arm so
        # both models share identical loss numerics; falls back to the stock
        # HF full-logit path when the config flag is off.
        use_chunked_loss = labels is not None and getattr(
            self.config, "fuse_linear_cross_entropy", False
        )
        if not use_chunked_loss:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )
        hidden_states = outputs.last_hidden_state
        # Liger FLCE for TRAINING only: same loss semantics, no materialized
        # logits (-6.8 GiB peak at 8k/128k-vocab). Eval keeps chunked CE — this
        # branch already lost a production run to a fused Triton kernel hanging
        # in the 448-rank benchmark-eval path (see qk_norm.py), so new Triton
        # loss kernels stay out of eval until proven there.
        loss_fn = (
            liger_linear_cross_entropy
            if self.training and getattr(self.config, "use_liger_flce", False)
            else chunked_linear_cross_entropy
        )
        loss, ce_detached, z_detached = loss_fn(
            hidden_states=hidden_states,
            labels=labels,
            lm_head=self.lm_head,
            config=self.config,
            **kwargs,
        )
        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            logits_to_keep = 1
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        out = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        out.ce_loss = ce_detached
        out.z_loss = z_detached
        return out


__all__ = ["CustomMarinForCausalLM", "CustomMarinModel"]
