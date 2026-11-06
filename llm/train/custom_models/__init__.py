"""LastKV language-model implementations for flame.

``flame/train.py`` does ``import custom_models``, which resolves to this
package when ``llm/train`` is on ``PYTHONPATH``; no copying is needed.
Importing a model subpackage registers its config and model classes with
Transformers' Auto classes, so a ``model_type`` in a config JSON resolves to the
right implementation.

The Transformer baselines at 124M and 760M are fla's built-in ``transformer``
model type and need no code from here.

Small-scale models (124M / 760M):

=========================================================  ===============================
``model_type``                                             Role
=========================================================  ===============================
``recurrent_transformer_aggregate_kv_v254_reusekv``        LastKV, last-layer K/V verbatim
``recurrent_transformer_aggregate_kv_v254_sharekv``        LastKV, one shared K/V projection
``recurrent_transformer_aggregate_kv_v254_perlayerkv``     LastKV, per-layer K/V projections
``..._reusekv_lastnorecurrent``                            ablation: no recurrence
``..._reusekv_detach_lastkv``                              ablation: detached shared KV
``..._reusekv_source_layer``                               ablation: source-layer depth
=========================================================  ===============================

8B models:

=========================================  =====================
``custom_marin``                           Transformer baseline
``custom_recurrent_marin_akv254_reusekv``  LastKV 8B
=========================================  =====================
"""

from . import recurrent_transformer_aggregate_kv_v254_reusekv  # noqa: F401
from . import recurrent_transformer_aggregate_kv_v254_sharekv  # noqa: F401
from . import recurrent_transformer_aggregate_kv_v254_perlayerkv  # noqa: F401
from . import recurrent_transformer_aggregate_kv_v254_reusekv_lastnorecurrent  # noqa: F401
from . import recurrent_transformer_aggregate_kv_v254_reusekv_detach_lastkv  # noqa: F401
from . import recurrent_transformer_aggregate_kv_v254_reusekv_source_layer  # noqa: F401

# The 8B families need liger-kernel for the fused cross entropy; keep them
# optional so the small-scale models import without it.
MARIN_IMPORT_ERROR = None
try:
    from . import custom_marin  # noqa: F401
    from . import custom_recurrent_marin_akv254_reusekv  # noqa: F401
except ImportError as exc:  # pragma: no cover - depends on optional deps
    MARIN_IMPORT_ERROR = exc

__all__ = [
    "recurrent_transformer_aggregate_kv_v254_reusekv",
    "recurrent_transformer_aggregate_kv_v254_sharekv",
    "recurrent_transformer_aggregate_kv_v254_perlayerkv",
    "recurrent_transformer_aggregate_kv_v254_reusekv_lastnorecurrent",
    "recurrent_transformer_aggregate_kv_v254_reusekv_detach_lastkv",
    "recurrent_transformer_aggregate_kv_v254_reusekv_source_layer",
]
