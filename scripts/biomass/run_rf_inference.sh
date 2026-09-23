#!/usr/bin/env bash
# Score every tile with the 38-feature random forest and write a CSV.
#
#   scripts/run_rf_inference.sh 0.65 /tmp/tiles.csv     # flag at P >= 0.65
#   scripts/run_rf_inference.sh none  /tmp/tiles.csv     # probabilities only, no flag column
#
# THE THRESHOLD IS A REQUIRED ARGUMENT. There is no default and there will not be one: on
# the buffered held-out folds this forest flags ZERO tiles on two of three folds at any cut
# between 0.5 and 0.8, and 460 of 947 on the third. See docs/RESULTS.md 'Operating point'.
# Pass `none` if you have no basis for choosing one -- that is the honest answer more often
# than not, and downstream code can rank on P.
#
# Needs only data/bio_gate_t512.joblib and data/bio_tile_features_t512.npz, both shipped.
# No granules, no chip cache, no torch.
set -euo pipefail

THRESH="${1:?usage: run_rf_inference.sh THRESH|none OUT.csv}"
OUT="${2:?usage: run_rf_inference.sh THRESH|none OUT.csv}"
ENV_NAME="${CONDA_ENV:-biomass-gate}"

cd "$(dirname "$0")/.."

if [ "$THRESH" = "none" ]; then
  CUT=(--no-thresh)
else
  CUT=(--thresh-rf "$THRESH")
fi

conda run --no-capture-output -n "$ENV_NAME" python src/bio_score_tiles.py \
  --method rf "${CUT[@]}" --out "$OUT"
