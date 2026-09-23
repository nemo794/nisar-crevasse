#!/usr/bin/env bash
# Retrain the 38-feature forest and reprint every number this repo quotes.
#
#   scripts/run_rf_training.sh                      # -> data/bio_gate_retrained.joblib
#   scripts/run_rf_training.sh data/mine.joblib
#
# Writes to a NEW path by default. Never overwrite data/bio_gate_t512.joblib -- train beside
# it and compare, because that bundle is what the pinned controls in docs/CONTRIBUTING.md
# are pinned to.
#
# Reproducibility targets (NOT performance claims -- read docs/RESULTS.md for why):
#   buffered per-fold AUC   0.621 / 0.762 / 0.918   over n = 521 / 947 / 279
#   pooled 0.889 with (x,y)-only control at 0.601
#   ZERO tiles flagged on folds 0 and 2 at every cut from 0.5 to 0.8
# ~4 min on 10 cores.
set -euo pipefail

OUT="${1:-data/bio_gate_retrained.joblib}"
ENV_NAME="${CONDA_ENV:-biomass-gate}"

cd "$(dirname "$0")/.."
conda run --no-capture-output -n "$ENV_NAME" python src/bio_train_gate.py --out "$OUT"
