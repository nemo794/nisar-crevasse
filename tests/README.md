# Tests

```bash
pip install -e .          # once per environment -- see the top-level README
python -m pytest tests/
```

## Scope: pure logic, synthetic tiles, never real granule data

Every other doc in this repo (`docs/nisar/`, `docs/biomass/`) says some version of "no
unit-test suite -- pinned real-data controls play that role instead," and that's still
true for anything that makes a claim about crevasse-detection *accuracy*. This directory
does not compete with that: it covers logic that's either wrong or right regardless of
what a real tile looks like, and doesn't need one to check:

- `test_pipeline_factory.py` -- `Pipeline("nisar"|"biomass", **kwargs)`'s construction,
  error handling, and the module-isolation guarantee that lets both sensors be used in
  one process (both sensors' `pipeline_predict.py` share a filename by design; a naive
  bare import of both would have the second shadow the first in `sys.modules`).
- `test_pipeline_scatter_gather.py` -- the scatter/gather bookkeeping (NaN-filling a
  gate-rejected tile's slot, never dropping or reordering a tile, `unet_mask` matching
  `unet_prob >= thresh`) using synthetic degenerate (all-zero / all-NaN) tiles -- the
  same "unscoreable tile" case each sensor's own pinned P2 control already checks
  against a real granule, exercised here without needing one.
- `test_unet_shape_validation.py` -- `UNetSegmenter`/`BiomassUNetSegmenter`'s
  `(N, 1, 512, 512)` / `(N, 4, 512, 512)` input contract and its guard rails (wrong
  channel count, wrong tile size, negative/dB-not-linear values).

All three load the real shipped default models from `models/` (small, already in the
repo) -- what's synthetic is the *input tiles*, never the model weights. Whether a real
tile gets a correct probability is exactly what the pinned real-data controls in
`docs/nisar/PIPELINE_CONTRIBUTING.md` / `docs/biomass/PIPELINE_CONTRIBUTING.md` (and
each sensor's own gate/U-Net docs) check -- nothing here makes or should make that claim.

## Adding a test

Ask first: does this need a real tile to mean anything? If yes, it belongs in a pinned
control in `docs/{sensor}/`, not here. If no -- it's shape/error/bookkeeping logic that's
either right or wrong on a synthetic zero array -- it belongs here.
