#!/usr/bin/env bash
# 124M, 32K context, 10B tokens (paper Table 1 / A1 / A2, 124M rows).
# Default arm is perlayerkv, the 124M LastKV row of Table 1.
# Global batch must be 32 sequences: 8 GPUs x per-GPU batch 4.
# Uses the vendored trainer in flame/; no external flame checkout is needed.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/common.sh"

ARM=${ARM:-perlayerkv}            # perlayerkv | sharekv | reusekv | transformer
if [ "$ARM" = "transformer" ]; then
  MODEL_CONFIG="${HERE}/../configs/124M/transformer_124M.json"
else
  MODEL_CONFIG="${HERE}/../configs/124M/lastkv_124M_${ARM}.json"
fi

run_flame "${HERE}/../configs/124M/train_124M_32k.toml" \
  --model.config "${MODEL_CONFIG}" \
  --job.dump_folder "${DUMP_FOLDER:-exp/lastkv_124M_${ARM}}" \
  --training.batch_size "${PER_GPU_BATCH:-4}" \
  --activation_checkpoint.mode full \
  "$@"
