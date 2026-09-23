#!/usr/bin/env bash
# Decide whether a new granule is safe to use, BEFORE spending effort labelling it.
#
#   scripts/run_acceptance.sh 025_048              # compare against 025_019
#   scripts/run_acceptance.sh 025_048 025_091      # compare against 025_091 instead
#
# Read docs/LIMITATIONS.md first. Short version: the cheap screens can convict a granule
# but they cannot clear one. Only the matched-ground A/B has ever caught a bad granule
# that the screens passed, and it needs labels for both granules plus overlapping ground.
set -euo pipefail

NEW="${1:?usage: run_acceptance.sh NEW_GRANULE [KNOWN_GOOD_GRANULE]}"
KNOWN="${2:-025_019}"
ENV_NAME="${CONDA_ENV:-nisar-gate}"

cd "$(dirname "$0")/.."
run() { conda run --no-capture-output -n "$ENV_NAME" python src/check_granule.py "$@"; }

echo "###############################################################"
echo "# 1/4  oriented-artifact screen (no labels needed, ~2 min)"
echo "###############################################################"
run orientation "$NEW"

echo
echo "###############################################################"
echo "# 2/4  co-registration vs $KNOWN (no labels needed)"
echo "###############################################################"
run register "$KNOWN" "$NEW" || echo "(skipped: no overlap with $KNOWN)"

echo
echo "###############################################################"
echo "# 3/4  speckle statistics on matched ground (needs labels)"
echo "###############################################################"
run speckle "$KNOWN" "$NEW" || echo "(skipped: needs labels for both granules)"

echo
echo "###############################################################"
echo "# 4/4  MATCHED-GROUND A/B -- the decisive test (needs labels)"
echo "###############################################################"
run ab "$NEW" "$KNOWN" || echo "(skipped: needs labels for both + overlapping ground)"

echo
echo "Steps 1-3 are screens. Step 4 is the verdict. If step 4 could not run, you do not"
echo "yet know whether this granule works -- and steps 1-3 passing does not tell you."
