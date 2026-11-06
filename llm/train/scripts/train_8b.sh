#!/usr/bin/env bash
# 8B continued pretraining at 8K context (paper Table 6).
#
# Same trainer as the smaller scales.  The 8B recipe differs only in the
# config: constant peak learning rate after re-warmup, and weight decay masked
# off embeddings and RMSNorm gains as well as the LastKV blend gates.
#
# Environment, beyond scripts/common.sh:
#   LASTKV_INIT_CKPT   distributed-checkpoint directory of the base model
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/common.sh"

: "${LASTKV_INIT_CKPT:?set LASTKV_INIT_CKPT to the base checkpoint directory}"

ARM=${ARM:-lastkv}                # lastkv | transformer
if [ "$ARM" = "transformer" ]; then
  MODEL_CONFIG="${HERE}/../configs/8B/transformer_8B.json"
else
  MODEL_CONFIG="${HERE}/../configs/8B/lastkv_8B_reusekv.json"
fi

run_flame "${HERE}/../configs/8B/cpt_8B_8k.toml" \
  --model.config "${MODEL_CONFIG}" \
  --job.dump_folder "${DUMP_FOLDER:-exp/lastkv_8B_${ARM}}" \
  --checkpoint.initial_load_path "${LASTKV_INIT_CKPT}" \
  "$@"
