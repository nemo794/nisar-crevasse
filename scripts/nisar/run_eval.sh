#!/usr/bin/env bash
# Score a checkpoint on a shard and render the figures, without slurm.
#
# Usage: scripts/run_eval.sh [CHECKPOINT] [SHARD] [SUBSET]
#
# SUBSET 'all' for a shard the model never saw; 'val' on the training shard, which is the
# control that must reproduce the checkpoint's recorded val_iou. Keep --batch-size equal to
# the training run's, because the per-batch IoU is the only one comparable to it.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:-models/unet_025_019_soft70/unet_best.pth}"
SHARD="${2:-data/train_025_091_soft70.npz}"
SUBSET="${3:-all}"
BATCH="${BATCH:-8}"

OUT="$(dirname "$CKPT")"
TAG="$(basename "${SHARD%.npz}")_$SUBSET"

python src/eval_shard.py \
  --checkpoint "$CKPT" \
  --shard "$SHARD" \
  --subset "$SUBSET" \
  --batch-size "$BATCH" \
  --sweep \
  --out-npz "$OUT/eval_$TAG.npz"

python src/plot_eval.py --eval-npz "$OUT/eval_$TAG.npz" --tb-logdir "$OUT/logs"
