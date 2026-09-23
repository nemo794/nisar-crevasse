#!/usr/bin/env bash
# First real training run: 025_019 soft labels (t=0.7), spatial band split.
# 025_091 is deliberately NOT trained on -- it is the second-acquisition check.
#
# Usage: scripts/run_train.sh [SHARD] [OUTPUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."

SHARD="${1:-data/train_025_019_soft70.npz}"
OUT="${2:-models/unet_025_019_soft70}"

python src/train_unet_supervised.py \
  --shard "$SHARD" \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --base-channels 64 \
  --consistency-weight 0.1 \
  --split-mode spatial \
  --num-workers 4 \
  --output-dir "$OUT"

# Reported val IoU thresholds BOTH sides at 0.2 (the level every soft-label coverage
# figure in this project is quoted at). It is not comparable to a binary-label IoU.
