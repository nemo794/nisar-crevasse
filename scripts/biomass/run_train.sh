#!/usr/bin/env bash
# First real training run: all 4 polarizations, 82 km block / 41 km buffer split.
#
# Usage: scripts/run_train.sh [SHARD] [OUTPUT_DIR] [CHANNELS]
set -euo pipefail
cd "$(dirname "$0")/.."

SHARD="${1:-data/train_biomass_v1.npz}"
OUT="${2:-models/bio_unet_4ch}"
CHANNELS="${3:-HH,HV,VH,VV}"

python src/bio_train_unet_supervised.py \
  --shard "$SHARD" \
  --channels "$CHANNELS" \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --base-channels 64 \
  --consistency-weight 0.1 \
  --block-tiles 32 \
  --buffer-tiles 16 \
  --num-workers 4 \
  --output-dir "$OUT"

# The target is a real 0/1 AlphaEarth mask, not a continuous manufactured one -- val IoU
# thresholds it at a plain 0.5, not NISAR's soft-label 0.2 coverage cut.
