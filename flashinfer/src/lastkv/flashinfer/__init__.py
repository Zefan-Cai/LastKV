"""Standalone FlashInfer engine for LastKV and a shape-matched Transformer control."""

from .config import GenOutput, ModelConfig

__all__ = ["Engine", "GenOutput", "ModelConfig"]


def __getattr__(name):
    if name == "Engine":  # lazy: importing the engine needs flashinfer + CUDA
        from .engine import Engine

        return Engine
    raise AttributeError(name)
