"""Importable Transformers config used by vLLM's model-config machinery.

The class must live at module scope.  vLLM may pickle ``VllmConfig`` when its
EngineCore uses multiprocessing; a function-local config class cannot be
resolved by Python's standard pickle protocol.
"""

from __future__ import annotations

from transformers import LlamaConfig


class RecurrentMarinConfig(LlamaConfig):
    model_type = "custom_recurrent_marin_akv254_reusekv"


__all__ = ["RecurrentMarinConfig"]
