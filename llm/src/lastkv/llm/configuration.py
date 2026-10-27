"""Configuration for the released LastKV Marin checkpoints.

Adapted from Zefan-Cai/llm, commits 3d914d32 and 580a6c2a.
The original training framework is derived from fla-org/flame (MIT).
"""

from transformers import LlamaConfig


class CustomRecurrentMarinAKV254ReuseKVConfig(LlamaConfig):
    model_type = "custom_recurrent_marin_akv254_reusekv"

    def __init__(
        self,
        chunk_size=1024,
        v_gap=95,
        use_qk_norm=True,
        use_nope=False,
        deep_kv_rescale=True,
        deep_rescale_mode="v",
        deep_v_scale_init=1.0,
        deep_scale_frozen=False,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["CustomRecurrentMarinAKV254ReuseKVForCausalLM"])
        kwargs.pop("akv254_kv_mode", None)
        # Archived configs request a training FlashAttention implementation.
        # PyTorch SDPA selects its available CUDA kernels without that package.
        backend = kwargs.pop("attn_implementation", None)
        super().__init__(**kwargs)
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if v_gap is None or v_gap < self.num_hidden_layers - 1:
            raise ValueError("This inference implementation requires final-layer KV reuse")
        if deep_rescale_mode != "v" or deep_scale_frozen:
            raise ValueError("The released checkpoints use learnable deep_rescale_mode='v'")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.chunk_size = chunk_size
        self.v_gap = v_gap
        self.use_qk_norm = use_qk_norm
        self.use_nope = use_nope
        self.deep_kv_rescale = deep_kv_rescale
        self.deep_rescale_mode = deep_rescale_mode
        self.deep_v_scale_init = deep_v_scale_init
        self.deep_scale_frozen = deep_scale_frozen
        self.akv254_kv_mode = "reusekv"
        self._attn_implementation = backend if backend in ("eager", "sdpa") else "sdpa"
