"""LastKV video generation with the Wan architecture (optional dependencies loaded lazily)."""

__all__ = ["AKV2541ReuseKVCausalWanModel", "LastKVWanPipeline"]


def __getattr__(name):
    if name == "AKV2541ReuseKVCausalWanModel":
        from .model import AKV2541ReuseKVCausalWanModel
        return AKV2541ReuseKVCausalWanModel
    if name == "LastKVWanPipeline":
        from .pipeline import LastKVWanPipeline
        return LastKVWanPipeline
    raise AttributeError(name)
