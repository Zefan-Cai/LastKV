"""Non-gated AKV254 with last-layer reads that never feed back into stored KV.

The shared-weight writer is an ordinary causal Transformer with per-layer KV
history. The reader uses its writer's last-layer history and the original
AKV254 previous-chunk blend, but its states are never stored for future chunks.
This follows LVSM lastnorecurrent's update/apply separation without detaching
memory gradients. Concatenation keeps both streams' checkpoint inputs immutable.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.deprecation import deprecate_kwarg

try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    try:
        from flash_attn import flash_attn_func
    except ImportError:

        def flash_attn_func(q, k, v, causal=False, **kwargs):
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if k.shape[1] != q.shape[1]:
                groups = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(groups, dim=1)
                v = v.repeat_interleave(groups, dim=1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            if causal:
                q_len = q.size(-2)
                k_len = k.size(-2)
                offset = max(k_len - q_len, 0)
                q_pos = torch.arange(q_len, device=q.device).view(q_len, 1) + offset
                k_pos = torch.arange(k_len, device=q.device).view(1, k_len)
                scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, v).transpose(1, 2).contiguous()

from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as TransformerMLP
from fla.modules.l2warp import l2_warp

from .configuration_recurrent_transformer import RecurrentTransformerConfig
from ..recurrent_transformer_aggregate_kv_v254_reusekv.rotary import RotaryEmbedding

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

def _validate_no_padding_attention_mask(attention_mask: torch.Tensor | None) -> None:
    if attention_mask is None:
        return
    if not bool(torch.all(attention_mask != 0).item()):
        raise ValueError("aggregate_kv_v254 does not support padded attention_mask.")


class RecurrentAttention(nn.Module):
    def __init__(self, config: RecurrentTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=config.rope_theta)
        self.prev_chunk_alpha = nn.Parameter(torch.zeros(self.num_kv_heads))

    def reset_parameters(self):
        nn.init.zeros_(self.prev_chunk_alpha)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict.setdefault(prefix + "prev_chunk_alpha", torch.zeros_like(self.prev_chunk_alpha))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def cache_init(self, batch_size, max_len, device, dtype):
        self.rotary._update_cos_sin_cache(max_len * 2, device=device, dtype=dtype)
        shape = (batch_size, 0, self.num_kv_heads, self.head_dim)
        return torch.empty(shape, device=device, dtype=dtype), torch.empty(shape, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        prev_chunk_curr_layer_kv_cache: Tuple[torch.Tensor | None, torch.Tensor | None],
        prev_chunk_last_layer_kv_cache: Tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, q_len, _ = hidden_states.size()
        cache_dtype = kv_cache[0].dtype if kv_cache[0].numel() else hidden_states.dtype

        q = self.q_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).to(cache_dtype)
        k = self.k_proj(hidden_states).view(batch_size, q_len, self.num_kv_heads, self.head_dim).to(cache_dtype)
        v = self.v_proj(hidden_states).view(batch_size, q_len, self.num_kv_heads, self.head_dim).to(cache_dtype)

        k_cache, v_cache = kv_cache
        prev_currlayer_k, prev_currlayer_v = prev_chunk_curr_layer_kv_cache
        prev_lastlayer_k, prev_lastlayer_v = prev_chunk_last_layer_kv_cache
        has_prev = prev_currlayer_k is not None
        if (
            has_prev != (prev_currlayer_v is not None)
            or has_prev != (prev_lastlayer_k is not None)
            or has_prev != (prev_lastlayer_v is not None)
        ):
            raise ValueError("prev chunk KV caches must be all None or all set")
        recent_ctx_len = prev_currlayer_k.size(1) if has_prev else 0
        q, k = self.rotary(q, k, seqlen_offset=k_cache.size(1) + recent_ctx_len)

        full_k = k_cache
        full_v = v_cache
        if has_prev:
            blend_alpha = self.prev_chunk_alpha.sigmoid().to(dtype=cache_dtype).view(
                1, 1, self.num_kv_heads, 1
            )
            blended_prev_k = blend_alpha * prev_lastlayer_k.to(cache_dtype) + (1 - blend_alpha) * prev_currlayer_k.to(cache_dtype)
            blended_prev_v = blend_alpha * prev_lastlayer_v.to(cache_dtype) + (1 - blend_alpha) * prev_currlayer_v.to(cache_dtype)
            full_k = torch.cat((full_k.to(blended_prev_k.dtype), blended_prev_k), dim=1)
            full_v = torch.cat((full_v.to(blended_prev_v.dtype), blended_prev_v), dim=1)
        full_k = torch.cat((full_k.to(k.dtype), k), dim=1)
        full_v = torch.cat((full_v.to(v.dtype), v), dim=1)

        attn_output = flash_attn_func(q, full_k, full_v, causal=True)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        attn_output = attn_output.reshape(batch_size, q_len, self.hidden_size).to(hidden_states.dtype)
        return self.o_proj(attn_output), (k, v)


class RecurrentTransformerBlock(GradientCheckpointingLayer):
    def __init__(self, config: RecurrentTransformerConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = RecurrentAttention(config, layer_idx)
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def _forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        prev_chunk_curr_layer_kv_cache: Tuple[torch.Tensor | None, torch.Tensor | None],
        prev_chunk_last_layer_kv_cache: Tuple[torch.Tensor | None, torch.Tensor | None],
        **kwargs: Unpack[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states_after_attn_norm = hidden_states
        hidden_states, currlayer_kv = self.attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            prev_chunk_curr_layer_kv_cache=prev_chunk_curr_layer_kv_cache,
            prev_chunk_last_layer_kv_cache=prev_chunk_last_layer_kv_cache,
        )

        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)

        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        return hidden_states, hidden_states_after_attn_norm, currlayer_kv

    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)



class RecurrentTransformerPreTrainedModel(PreTrainedModel):
    config_class = RecurrentTransformerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["RecurrentTransformerBlock"]
    _supports_cache_class = False

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

        if hasattr(module, "prev_chunk_alpha"):
            nn.init.zeros_(module.prev_chunk_alpha)

        if rescale_prenorm_residual:
            p = None
            if hasattr(module, "o_proj"):
                p = module.o_proj.weight
            elif hasattr(module, "down_proj"):
                p = module.down_proj.weight
            if p is not None:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class RecurrentTransformerModel(RecurrentTransformerPreTrainedModel):
    def __init__(self, config: RecurrentTransformerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.chunk_size = config.chunk_size
        self.v_gap = config.v_gap

        self.v_layer_idxs = [config.num_hidden_layers - 1] * config.num_hidden_layers

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [RecurrentTransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def extra_repr(self) -> str:
        return "last_no_recurrent: write=per-layer-kv, read=last-layer-kv + AKV254 recent blend"

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Unpack[Any],
    ) -> tuple | BaseModelOutputWithPast:
        if output_attentions:
            warnings.warn(
                "`RecurrentTransformerModel` does not support output attention weights, "
                "so `output_attentions` is set to `False`.",
            )
            output_attentions = False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        _validate_no_padding_attention_mask(attention_mask)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        batch_size, seq_len, _ = inputs_embeds.shape
        write_caches = [
            layer.attn.cache_init(
                batch_size,
                seq_len,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            for layer in self.layers
        ]
        prev_write_kv = [(None, None) for _ in self.layers]
        all_hidden_states = () if output_hidden_states else None
        chunk_outputs = []

        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            chunk_inputs = inputs_embeds[:, chunk_start:chunk_end]
            hidden_states = chunk_inputs

            # The reader sees only earlier writer chunks. Keep the immediately
            # previous chunk separate for AKV254's learned same/last-layer mix.
            previous_length = 0 if prev_write_kv[-1][0] is None else prev_write_kv[-1][0].size(1)
            old_length = chunk_start - previous_length
            read_history = tuple(kv[:, :old_length] for kv in write_caches[-1])
            read_chunk_kv = []
            for layer_idx, layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                hidden_states, _, chunk_kv = checkpoint(
                    layer,
                    hidden_states,
                    read_history,
                    prev_write_kv[layer_idx],
                    prev_write_kv[-1],
                    use_reentrant=False,
                )
                read_chunk_kv.append(chunk_kv)

            chunk_outputs.append(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # No later prediction can use a write of the final chunk.
            if chunk_end == seq_len:
                continue
            if chunk_start == 0:
                # Empty-history read and write passes are exactly identical.
                write_chunk_kv = read_chunk_kv
            else:
                # Restart from embeddings, never from reader hidden states.
                # This ordinary causal writer reads only its own layer's KV;
                # last-layer reads and the alpha blend cannot feed its memory.
                write_states = chunk_inputs
                write_chunk_kv = []
                for layer_idx, layer in enumerate(self.layers):
                    write_states, _, chunk_kv = checkpoint(
                        layer,
                        write_states,
                        write_caches[layer_idx],
                        (None, None),
                        (None, None),
                        use_reentrant=False,
                    )
                    write_chunk_kv.append(chunk_kv)

            # Functional caches keep writer history immutable for checkpoint
            # recomputation and retain gradients through every stored chunk.
            write_caches = [
                tuple(torch.cat((past.to(current.dtype), current), dim=1) for past, current in zip(history, chunk_kv))
                for history, chunk_kv in zip(write_caches, write_chunk_kv)
            ]
            prev_write_kv = write_chunk_kv

        hidden_states = torch.cat(chunk_outputs, dim=1)
        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, None] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class RecurrentTransformerForCausalLM(RecurrentTransformerPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = RecurrentTransformerModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[Any],
    ) -> dict[str, Any]:
        model_inputs: dict[str, Any]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs["past_key_values"] = None
        model_inputs["use_cache"] = False
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        for key, value in kwargs.items():
            if key not in {"cache_position", "position_ids"}:
                model_inputs[key] = value
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        **kwargs: Unpack[Any],
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        fuse_lce_training = self.config.fuse_linear_cross_entropy and labels is not None
        logits = None if fuse_lce_training else self.lm_head(
            hidden_states[:, -logits_to_keep:] if logits_to_keep else hidden_states
        )

        loss = None
        if labels is not None:
            if getattr(self, "criterion", None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                if logits_to_keep:
                    labels = labels[:, -logits_to_keep:]
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "RecurrentAttention",
    "RecurrentTransformerBlock",
    "RecurrentTransformerForCausalLM",
    "RecurrentTransformerModel",
]
