#!/usr/bin/env bash
# Acceptance test on a training shard. RUN THIS FIRST after rsync -- it is the only
# thing standing between a truncated transfer and a training run that appears to work.
set -euo pipefail
cd "$(dirname "$0")/.."
SHARD="${1:-data/train_025_019_soft70.npz}"
python src/check_labels.py "$SHARD"
