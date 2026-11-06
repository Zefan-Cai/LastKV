#!/usr/bin/env bash
# Shared torchrun launcher.  One trainer serves all three scales.
#
# Environment:
#   LASTKV_DATA_FILES  glob of tokenized shards (required)
#   NNODE, NGPU        node / per-node GPU count (default 1, 8)
#   NODE_RANK, MASTER_ADDR, MASTER_PORT   multi-node rendezvous
#   WANDB_API_KEY, WANDB_BASE_URL         optional Weights & Biases logging
set -euo pipefail

: "${LASTKV_DATA_FILES:?set LASTKV_DATA_FILES to the tokenized shard glob}"

TRAIN_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NNODE=${NNODE:-1}
NGPU=${NGPU:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
LOG_RANK=${LOG_RANK:-0}

run_flame() {
  local config_file=$1; shift
  cd "$TRAIN_ROOT"
  PYTHONPATH="${TRAIN_ROOT}:${PYTHONPATH:-}" torchrun \
    --nnodes="${NNODE}" \
    --nproc_per_node="${NGPU}" \
    --node_rank="${NODE_RANK}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    --local-ranks-filter "${LOG_RANK}" \
    --role rank --tee 3 \
    -m flame.train \
    --job.config_file "${config_file}" \
    --training.data_files "${LASTKV_DATA_FILES}" \
    "$@"
}
