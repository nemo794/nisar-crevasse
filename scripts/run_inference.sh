#!/usr/bin/env bash
# Score one granule with the shipped gate and render a map. Inference only -- no labels,
# no context rasters, no training data needed. Just the granule and data/*.joblib.
#
#   scripts/run_inference.sh 025_019
#   GATE_ROOT=/mnt/nisar scripts/run_inference.sh 025_019
#
# The granule id is the 3_3 digit pair in the filename (GSLC_<id>_...). The .tif must be
# under $GATE_ROOT/data/nisar/ (defaults to this repo).
set -euo pipefail

GRANULE="${1:?usage: run_inference.sh GRANULE_ID  (e.g. 025_019)}"
THRESH="${GATE_THRESH:-0.333}"
ENV_NAME="${CONDA_ENV:-nisar-gate}"

cd "$(dirname "$0")/.."
ROOT="${GATE_ROOT:-$PWD}"
mkdir -p "$ROOT/data/maps"

# --save-scores writes every tile's probability, so a different threshold later is a
# re-render (seconds) rather than a re-score (minutes). Probabilities do not depend on
# the threshold; it only decides which tiles get drawn.
#
# Deliberately NOT passed:
#   --normalize-per-granule  the gate is already brightness-invariant; this only adds drift
#   --context-prior          discredited -- it was memorising geography, not learning ice
#   --grounded-only          see docs/LIMITATIONS.md; keep rates below are upper bounds
conda run --no-capture-output -n "$ENV_NAME" python src/map_crevasse_tiles.py \
  --granule "$GRANULE" \
  --gate-thresh "$THRESH" \
  --save-scores "$ROOT/data/maps/gate_sar_${GRANULE}.npz" \
  --out "$ROOT/data/maps/crevasse_map_${GRANULE}.png"

echo
echo "scores  $ROOT/data/maps/gate_sar_${GRANULE}.npz"
echo "map     $ROOT/data/maps/crevasse_map_${GRANULE}.png"
echo
echo "To try another threshold without re-scoring:"
echo "  conda run -n $ENV_NAME python src/render_score_map.py \\"
echo "    $ROOT/data/maps/gate_sar_${GRANULE}.npz --gate-thresh 0.65"
