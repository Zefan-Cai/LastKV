from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils import is_flash_attn_2_available, is_flash_attn_3_available


class CustomMarinConfig(LlamaConfig):
    model_type = "custom_marin"

    def __init__(self, **kwargs):
        kwargs.setdefault("architectures", ["CustomMarinForCausalLM"])
        # main-norm fusion (input/post-attn/final RMSNorm -> fla Triton kernel,
        # mode-routed like qk-norm). Default off: fla multiplies weight in fp32
        # before the single downcast while LlamaRMSNorm rounds x_hat to bf16
        # first, so enabling it on a live run is a (tiny) numeric graft.
        fuse_norm = kwargs.pop("fuse_norm", False)
        if not isinstance(fuse_norm, bool):
            raise ValueError("`fuse_norm` must be a boolean.")
        # Liger FLCE loss route (training only; eval keeps chunked CE). Same
        # loss semantics as chunked CE at equal speed but ~-6.8 GiB peak at
        # marin 8k shapes; needs liger_kernel >= c9fc646. Default off: swapping
        # the loss kernel mid-run is a numeric graft like fuse_norm.
        use_liger_flce = kwargs.pop("use_liger_flce", False)
        if not isinstance(use_liger_flce, bool):
            raise ValueError("`use_liger_flce` must be a boolean.")
        # 0 = Liger's native heuristic (256-token chunks at marin 8k, time-tie
        # with chunk-512 CE). 2048 measured 42.3 ms vs 96.6 ms production CE at
        # +3.5 GiB; needs the liger-flce-chunk-size-env patch on the install.
        liger_flce_chunk_size = kwargs.pop("liger_flce_chunk_size", 0)
        if isinstance(liger_flce_chunk_size, bool) or not isinstance(
            liger_flce_chunk_size, int
        ) or liger_flce_chunk_size < 0:
            raise ValueError("`liger_flce_chunk_size` must be a non-negative integer.")
        # PretrainedConfig.from_dict passes attn_implementation=None which would
        # overwrite the JSON value; capture it before super() and re-apply after.
        requested_attn_implementation = kwargs.get("attn_implementation")
        super().__init__(**kwargs)
        self.fuse_norm = fuse_norm
        self.use_liger_flce = use_liger_flce
        self.liger_flce_chunk_size = liger_flce_chunk_size

        if requested_attn_implementation:
            self._attn_implementation = requested_attn_implementation
        elif is_flash_attn_3_available():
            self._attn_implementation = "flash_attention_3"
        elif is_flash_attn_2_available():
            self._attn_implementation = "flash_attention_2"


__all__ = ["CustomMarinConfig"]
