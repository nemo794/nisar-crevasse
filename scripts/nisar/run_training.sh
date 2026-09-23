#!/usr/bin/env bash
# Reproduce the shipped gate bundle from the shipped label CSVs.
#
#   scripts/run_training.sh                    # writes data/gate_retrained.joblib
#   GATE_ROOT=/mnt/nisar scripts/run_training.sh
#
# Needs the granules: 025_019 and 025_091 under $GATE_ROOT/data/nisar/. See docs/DATA.md.
#
# Expected, and worth checking against docs/RESULTS.md:
#   3692 tiles, random-fold OOF 0.951, spatial-block OOF 0.932, threshold 0.333
#
# Writes to a NEW path rather than overwriting the shipped bundle, so a retrain that
# lands somewhere different is a comparison rather than a loss.
set -euo pipefail

ENV_NAME="${CONDA_ENV:-nisar-gate}"
cd "$(dirname "$0")/.."
ROOT="${GATE_ROOT:-$PWD}"
OUT="${1:-$ROOT/data/gate_retrained.joblib}"

# tile_labels_ongrid_5m.csv holds 025_091's hand labels; the speedmatched CSVs are
# AlphaEarth-derived and supply the bulk. Rows are deduped by (granule, row, col).
#
# 025_048 is deliberately ABSENT: it fails a matched-ground A/B (0.627 vs 0.904 on
# identical tiles and labels) for reasons still unknown. Its CSV ships so you can
# reproduce that finding with src/check_granule.py, not so you can train on it.
#
# --neg-buffer/--match-neg-speed choices are baked into the CSVs already; see
# docs/LIMITATIONS.md for why the negatives had to be speed-matched.
conda run --no-capture-output -n "$ENV_NAME" python src/train_gate_classifier.py \
  --labels-csv \
    "$ROOT/data/tile_labels_ongrid_5m.csv" \
    "$ROOT/data/tile_labels_aoi_025_019_speedmatched.csv" \
    "$ROOT/data/tile_labels_aoi_025_091_speedmatched.csv" \
  --out "$OUT"

echo
echo "wrote $OUT"
echo "Compare its oof_auc_spatial against 0.932 before using it for anything."
