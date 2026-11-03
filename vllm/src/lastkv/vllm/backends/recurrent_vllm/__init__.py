"""Native vLLM pilot for Marin's AKV254 chunk-recurrent architecture.

The package is intentionally isolated from the ordinary Marin vLLM backend.
Importing it does not import torch or vLLM; heavyweight runtime modules are
loaded only by the pilot command on a GPU node.
"""

from .contract import (
    RECURRENT_ARCHITECTURE,
    RECURRENT_MODEL_TYPE,
    RecurrentModelContract,
    RecurrentProbeReport,
    probe_model_directory,
)

__all__ = [
    "RECURRENT_ARCHITECTURE",
    "RECURRENT_MODEL_TYPE",
    "RecurrentModelContract",
    "RecurrentProbeReport",
    "probe_model_directory",
]
