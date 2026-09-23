#!/usr/bin/env bash
# Score a checkpoint on a shard and render the figures, without slurm.
#
# Usage: scripts/run_eval.sh [CHECKPOINT] [SHARD] [SUBSET]
#
# SUBSET 'val' reproduces the checkpoint's recorded val_iou -- the control. Keep
# --batch-size equal to the training run's, because the per-batch IoU is the only one
# comparable to it.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${1:-models/bio_unet_4ch/unet_best.pth}"
SHARD="${2:-data/train_biomass_v1.npz}"
SUBSET="${3:-val}"
BATCH="${BATCH:-8}"

OUT="$(dirname "$CKPT")"
TAG="$(basename "${SHARD%.npz}")_$SUBSET"

python src/bio_eval_shard.py \
  --checkpoint "$CKPT" \
  --shard "$SHARD" \
  --subset "$SUBSET" \
  --batch-size "$BATCH" \
  --sweep \
  --out-npz "$OUT/eval_$TAG.npz"

python src/bio_plot_eval.py --eval-npz "$OUT/eval_$TAG.npz" --tb-logdir "$OUT/logs"
