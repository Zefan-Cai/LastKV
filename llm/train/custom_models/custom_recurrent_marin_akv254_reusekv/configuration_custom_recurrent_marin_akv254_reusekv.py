from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils import is_flash_attn_2_available, is_flash_attn_3_available


class CustomRecurrentMarinAKV254ReuseKVConfig(LlamaConfig):
    """Marin-8B (Llama-3.1) with AKV254 reuse-KV chunked-recurrent attention.

    Self-contained: inlines the fields the shared recurrent configs pull from
    their `custom_recurrent_llama` parent, keeping this branch free of that
    dependency chain.
    """

    model_type = "custom_recurrent_marin_akv254_reusekv"

    def __init__(
        self,
        chunk_size: int = 4096,
        v_gap: int | None = 95,
        use_nope: bool = False,
        use_qk_norm: bool = False,
        fuse_linear_cross_entropy: bool = True,
        linear_cross_entropy_chunk_size: int = 512,
        fuse_norm: bool = False,
        use_liger_flce: bool = False,
        liger_flce_chunk_size: int = 0,
        deep_kv_rescale: bool = False,
        deep_rescale_mode: str = "kv",
        deep_v_scale_init=1.0,
        deep_scale_frozen: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["CustomRecurrentMarinAKV254ReuseKVForCausalLM"])
        kwargs.pop("akv254_kv_mode", None)
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError("`chunk_size` must be a positive integer.")
        if isinstance(linear_cross_entropy_chunk_size, bool) or not isinstance(
            linear_cross_entropy_chunk_size, int
        ) or linear_cross_entropy_chunk_size < 1:
            raise ValueError("`linear_cross_entropy_chunk_size` must be a positive integer.")
        # main-norm fusion + Liger FLCE loss route, mode-routed like qk-norm
        # (fla/Liger Triton for TRAINING; eager / chunked CE for eval, which
        # hangs the fused kernels at 448-rank). Mirrors the custom_marin arm.
        if not isinstance(fuse_norm, bool):
            raise ValueError("`fuse_norm` must be a boolean.")
        if not isinstance(use_liger_flce, bool):
            raise ValueError("`use_liger_flce` must be a boolean.")
        if isinstance(liger_flce_chunk_size, bool) or not isinstance(
            liger_flce_chunk_size, int
        ) or liger_flce_chunk_size < 0:
            raise ValueError("`liger_flce_chunk_size` must be a non-negative integer.")
        self.chunk_size = chunk_size
        self.v_gap = v_gap
        self.use_nope = use_nope
        self.use_qk_norm = use_qk_norm
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.linear_cross_entropy_chunk_size = linear_cross_entropy_chunk_size
        self.fuse_norm = fuse_norm
        self.use_liger_flce = use_liger_flce
        self.liger_flce_chunk_size = liger_flce_chunk_size
        # deep-KV entry rescale: phoenix layer-31 V RMS is 2.5-32x (median
        # 5.5x) larger than the receiving layers' native V (K is matched by
        # qk-norm). When enabled, each layer scales the INCOMING deep KV
        # (pure-deep cache + the deep half of the t-1 blend) by a learnable
        # per-kv-head factor; deep_v_scale_init = per-layer list (or scalar)
        # of RMS(V_i)/RMS(V_31) so deep V arrives in-distribution at step 0.
        if not isinstance(deep_kv_rescale, bool):
            raise ValueError("`deep_kv_rescale` must be a boolean.")
        # mode "kv": learnable k_scale (init 1) + v_scale (init list) on deep KV
        # mode "v":  learnable v_scale only (K untouched — qk-norm matches it)
        # mode "vnorm": per-token per-head RMSNorm on deep V x learnable gain
        #   (deep_v_scale_init then holds the ABSOLUTE per-layer gain init,
        #    i.e. the receiving layer's native RMS(V_i))
        # "vnormqk": the qk-norm treatment for deep V — RMSNorm(head_dim) with
        # a learnable [head_dim] weight shared across kv-heads; weight INIT =
        # the per-layer measured RMS scalar (fill), so deep V starts
        # in-distribution AND has per-channel (diagonal-adapter) freedom.
        if deep_rescale_mode not in ("kv", "v", "vnorm", "vnormqk"):
            raise ValueError("`deep_rescale_mode` must be one of kv|v|vnorm|vnormqk.")
        if deep_scale_frozen and deep_rescale_mode == "vnormqk":
            raise ValueError("deep_scale_frozen does not apply to vnormqk (use mode v).")
        # frozen: the rescale/gain is a CONSTANT (no learnable parameter at
        # all) — removes the mute-switch shortcut after the gate-collapse
        # finding (all learnable variants drove deep_v_scale to ~0 by step
        # 256, discarding the deep-KV channel instead of learning to use it).
        if not isinstance(deep_scale_frozen, bool):
            raise ValueError("`deep_scale_frozen` must be a boolean.")
        self.deep_kv_rescale = deep_kv_rescale
        self.deep_rescale_mode = deep_rescale_mode
        self.deep_v_scale_init = deep_v_scale_init
        self.deep_scale_frozen = deep_scale_frozen
        self.akv254_kv_mode = "reusekv"

        # PretrainedConfig.from_dict passes attn_implementation=None which would
        # overwrite the JSON value; capture it before super() and re-apply after.
        requested_attn_implementation = kwargs.get("attn_implementation")
        super().__init__(**kwargs)

        if requested_attn_implementation:
            self._attn_implementation = requested_attn_implementation
        elif is_flash_attn_3_available():
            self._attn_implementation = "flash_attention_3"
        elif is_flash_attn_2_available():
            self._attn_implementation = "flash_attention_2"


__all__ = ["CustomRecurrentMarinAKV254ReuseKVConfig"]
