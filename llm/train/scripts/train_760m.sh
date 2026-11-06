#!/usr/bin/env bash
# 760M, 32K context, 40B tokens (paper Table 1 / A1 / A2, 760M rows).
# Default arm is sharekv, the 760M LastKV row of Table 1.
# Global batch must be 32 sequences: 32 GPUs x per-GPU batch 1.
# Uses the vendored trainer in flame/; no external flame checkout is needed.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${HERE}/common.sh"

ARM=${ARM:-sharekv}               # sharekv | perlayerkv | reusekv | transformer
if [ "$ARM" = "transformer" ]; then
  MODEL_CONFIG="${HERE}/../configs/760M/transformer_760M.json"
else
  MODEL_CONFIG="${HERE}/../configs/760M/lastkv_760M_${ARM}.json"
fi

run_flame "${HERE}/../configs/760M/train_760M_32k.toml" \
  --model.config "${MODEL_CONFIG}" \
  --job.dump_folder "${DUMP_FOLDER:-exp/lastkv_760M_${ARM}}" \
  --training.batch_size "${PER_GPU_BATCH:-1}" \
  --activation_checkpoint.mode full \
  "$@"
