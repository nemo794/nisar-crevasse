#!/usr/bin/env bash
# Score a granule this repo has never seen: accept it, build features, score it.
#
#   scripts/run_new_granule.sh data/biomass/BIO_S1_..._M04_... rf   0.65 /tmp/new.csv
#   scripts/run_new_granule.sh data/biomass/BIO_S1_..._M04_... both 0.65 /tmp/new.csv
#   scripts/run_new_granule.sh data/biomass/BIO_S1_..._M04_... rf   none /tmp/new.csv
#
# NO LABELS ARE NEEDED. The AlphaEarth mosaic is 7.1 GB and not shipped; nothing in the
# 38-feature vector or the CNN chips reads it, so the feature table comes out with aoi_pos
# NaN and aoi_inbox False. Such a table is scoreable and NOT trainable, and that is the
# whole point of this script. The grounded prescan reads data/context_grid_2560m.npz, which
# is shipped and continent-wide, so this works off Thwaites.
#
# THE THRESHOLD IS ONE NUMBER HERE AND IS APPLIED TO EACH MODEL SEPARATELY. With `both` it
# is passed as --thresh-rf AND --thresh-cnn, which is a convenience, not a claim that the
# two scales are comparable -- they are not. Call src/bio_score_tiles.py directly to cut
# them at different values, which is usually what you want. Pass `none` for probabilities
# only. docs/RESULTS.md 'Operating point'.
#
# `rf` runs in biomass-gate (torch-free, invariant 1). `cnn` and `both` run in
# biomass-gate-cnn AND need the 4-channel chip cache, which this script builds for the
# granule -- that read is the slow step.
#
# AND: what comes out is a ranking on ground nothing here was measured on. Every number in
# docs/RESULTS.md is Thwaites. Read docs/LIMITATIONS.md 6 before quoting anything.
set -euo pipefail

GRANULE="${1:?usage: run_new_granule.sh GRANULE_DIR rf|cnn|both THRESH|none OUT.csv}"
METHOD="${2:?usage: run_new_granule.sh GRANULE_DIR rf|cnn|both THRESH|none OUT.csv}"
THRESH="${3:?usage: run_new_granule.sh GRANULE_DIR rf|cnn|both THRESH|none OUT.csv}"
OUT="${4:?usage: run_new_granule.sh GRANULE_DIR rf|cnn|both THRESH|none OUT.csv}"

case "$METHOD" in
  rf)        ENV_NAME="${CONDA_ENV:-biomass-gate}" ;;
  cnn|both)  ENV_NAME="${CONDA_ENV:-biomass-gate-cnn}" ;;
  *) echo "method must be rf, cnn or both (got '$METHOD')" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."
GRANULE="$(cd "$GRANULE" && pwd)"
NAME="$(basename "$GRANULE")"
BASE="$(dirname "$GRANULE")"
WORK="${BIO_WORKDIR:-data/new/$NAME}"
FEATURES="$WORK/features_t512.npz"
CACHE="$WORK/chipcache_t512_c128_all.npz"
WEIGHTS="${BIO_WEIGHTS:-data/bio_cnn_gate_sarvel.pt}"
mkdir -p "$WORK"

CUT=()
if [ "$THRESH" = "none" ]; then
  CUT=(--no-thresh)
else
  case "$METHOD" in
    rf)   CUT=(--thresh-rf "$THRESH") ;;
    cnn)  CUT=(--thresh-cnn "$THRESH") ;;
    both) CUT=(--thresh-rf "$THRESH" --thresh-cnn "$THRESH") ;;
  esac
fi

# [1] Acceptance test FIRST. It can convict a granule; it cannot clear one. Failing here
# saves the feature build, and on the NISAR sibling a granule that passed every cheap
# screen still scored 0.627 where a known-good one scored 0.904.
echo "== [1/4] acceptance test =="
conda run --no-capture-output -n "$ENV_NAME" python src/bio_check_granule.py "$GRANULE"

echo "== [2/4] tile features (no labels) =="
conda run --no-capture-output -n "$ENV_NAME" python src/bio_tile_features.py \
  --base "$BASE" --only "$NAME" --aoi /nonexistent/labels-not-required --out "$FEATURES"

if [ "$METHOD" != "rf" ]; then
  echo "== [3/4] 4-channel chip cache =="
  conda run --no-capture-output -n "$ENV_NAME" python src/bio_tile_cache.py \
    --all --base "$BASE" --features "$FEATURES" --out "$CACHE"
  EXTRA=(--weights "$WEIGHTS" --cache "$CACHE")
else
  echo "== [3/4] chip cache: skipped, the RF does not read chips =="
  EXTRA=()
fi

echo "== [4/4] score =="
conda run --no-capture-output -n "$ENV_NAME" python src/bio_score_tiles.py \
  --method "$METHOD" "${CUT[@]}" --features "$FEATURES" "${EXTRA[@]}" --out "$OUT"
