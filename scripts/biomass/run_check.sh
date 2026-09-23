#!/usr/bin/env bash
# Acceptance test on a training shard. RUN THIS FIRST after building or rsyncing it.
set -euo pipefail
cd "$(dirname "$0")/.."
SHARD="${1:-data/train_biomass_v1.npz}"
python src/bio_check_labels.py "$SHARD"
