import warnings

from transformers.configuration_utils import PretrainedConfig


class RecurrentTransformerConfig(PretrainedConfig):

    model_type = "recurrent_transformer_aggregate_kv_v254_reusekv_detach_lastkv"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        use_short_conv: bool = False,
        conv_size: int = 4,
        window_size: int | None = None,
        rope_theta: float | None = 10000.0,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        chunk_size: int = 1024,
        v_gap: int = 95,
        window_num: int = 1,
        **kwargs,
    ):
        if "force_first_local_kv_tokens" in kwargs or "force_first_local_kv" in kwargs:
            raise ValueError("aggregate_kv_v254 does not support local anchor KV; use aggregate_kv_v2541.")
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.v_gap = v_gap
        self.window_num = window_num

        if isinstance(window_num, bool) or not isinstance(window_num, int) or window_num < 1:
            raise ValueError("`window_num` must be a positive integer.")
        if window_num != 1:
            raise ValueError("`window_num` must be 1 for aggregate_kv_v254_reusekv.")

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"`hidden_size` ({hidden_size}) must be divisible by `num_heads` ({num_heads})."
            )
        if num_kv_heads is not None and num_heads % num_kv_heads != 0:
            raise ValueError(
                f"`num_heads` ({num_heads}) must be divisible by `num_kv_heads` ({num_kv_heads})."
            )
        if rope_theta is None:
            raise ValueError(
                "`rope_theta` must be a positive float (e.g. 10000.0), got None."
            )

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.",
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improves memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
