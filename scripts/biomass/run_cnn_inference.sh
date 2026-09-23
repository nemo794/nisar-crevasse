#!/usr/bin/env bash
# Score every tile with the 4-pol CNN ensemble and write a CSV.
#
#   BIO_CHIPCACHE=/path/bio_chipcache_t512_c128_all.npz \
#     scripts/run_cnn_inference.sh 0.65 /tmp/tiles.csv
#   BIO_CHIPCACHE=... scripts/run_cnn_inference.sh none /tmp/tiles.csv
#
# THE CHIP CACHE IS NOT SHIPPED (626 MB). This script therefore does NOT run from a clean
# checkout, and that is a property of the repo rather than a bug -- rebuild the cache with
# src/bio_tile_cache.py --all from the granules, per docs/DATA.md. The CNN's recorded fold
# AUCs stay checkable without it, from the shipped data/bio_cnn_oof_c128.npz.
#
# The threshold is required for the same reason as in the RF script; see docs/RESULTS.md.
set -euo pipefail

THRESH="${1:?usage: run_cnn_inference.sh THRESH|none OUT.csv}"
OUT="${2:?usage: run_cnn_inference.sh THRESH|none OUT.csv}"
ENV_NAME="${CONDA_ENV:-biomass-gate-cnn}"
WEIGHTS="${BIO_WEIGHTS:-data/bio_cnn_gate_sarvel.pt}"
CACHE="${BIO_CHIPCACHE:?set BIO_CHIPCACHE to the 4-channel chip cache; it is not shipped}"

cd "$(dirname "$0")/.."

if [ "$THRESH" = "none" ]; then
  CUT=(--no-thresh)
else
  CUT=(--thresh-cnn "$THRESH")
fi

# --weights loads the shipped 3-seed ensemble WITH its normalisation constants. Without
# --weights the inference script would retrain from scratch, which is a different model.
conda run --no-capture-output -n "$ENV_NAME" python src/bio_score_tiles.py \
  --method cnn "${CUT[@]}" --weights "$WEIGHTS" --cache "$CACHE" --out "$OUT"
