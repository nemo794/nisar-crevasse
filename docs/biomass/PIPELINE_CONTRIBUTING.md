# Adding to this repo

This repo has one module, `src/pipeline_predict.py`, and it exists to do exactly one
thing: filter a stack of tiles through `BiomassGate`, run `BiomassUNetSegmenter` only on
the survivors, and scatter the result back into a full-length stack. It has no model of
its own, so most of the discipline in `biomass-crevasse-gate/docs/CONTRIBUTING.md`
(feature engineering, bundle protocol, granule acceptance) and
`biomass-crevasse-unet/docs/PIPELINE.md` (shard building, training, the aux-branch
investigation) doesn't apply here — read those for that. What applies here is narrower:
**every measurement ships with its control**, same as every other repo in this project.

## The one invariant

**The gate filter is not optional.** `BiomassCrevassePipeline.run` always scores the gate
first and only calls `BiomassUNetSegmenter` on tiles `gate_flag` accepted. There is no
argument to bypass this — `bio_build_shard_frangi.py`'s soft labels were generated
exclusively on `BiomassGate` (RF, `P>=0.65`) survivors, so the U-Net has never been
evaluated off-gate.

## What this repo does NOT preserve

Unlike `biomass-crevasse-gate`'s Invariant 1 (the RF path stays torch-free even though the
CNN path needs torch), **this repo makes no attempt to keep any import path torch-free.**
`pipeline_predict.py` imports both `BiomassGate` and `BiomassUNetSegmenter` at module
load, and the latter needs torch unconditionally. `environment.yml` is the union of both
upstream envs for this reason — same reasoning `nisar-crevasse-pipeline`'s own
`docs/CONTRIBUTING.md` gives for the identical decision there.

## Why the default gate method is RF, not CNN

`bio_build_shard_frangi.py` (the actual Frangi soft-label generator) gates with
`method="rf"` at `gate_thresh=0.65` — this repo's default reproduces exactly that,
rather than picking a method independently. `method="cnn"` needs `x_m`/`y_m` (EPSG:3031
tile top-left, for the CNN's ice-velocity lookup); `run_granule.tile_granule` always
computes and returns them, so switching `--gate-method cnn` on the CLI scripts costs
nothing extra to supply.

## Controls

Verified 2026-09-22 against a real local BIOMASS granule
(`BIO_S2_SCS__1S_20251217T135546_..._DJT7YH`), `conda run -n biomass` (has torch,
rasterio, scikit-learn — the portable `environment.yml` here is the union of both
upstream envs' dependencies for a fresh checkout).

| # | check | result |
|---|---|---|
| P1 | 13-tile batch (12 real tiles from the granule via `tile_granule`, seed 1, plus 1 all-zero nodata tile appended) run through `BiomassCrevassePipeline().run()` | `gate_prob`/`gate_flag` **bit-identical** (maxdiff 0.0) to calling `BiomassGate().predict_proba(..., method="rf")` directly on the same batch; **4 of 13** flagged; `unet_prob` at the flagged indices **bit-identical** (maxdiff 0.0) to calling `BiomassUNetSegmenter().predict_proba()` directly on `db(inten[idx])` |
| P2 | the all-zero tile in the P1 batch | `gate_prob` is `NaN`, `gate_flag` is `False`, and its slot in `unet_prob` stays all-`NaN` — it never reaches the U-Net (`BiomassGate`'s own `min_valid=0.99` guard does the excluding; nothing in this repo re-checks validity) |
| P3 | `gate_thresh=`/`unet_thresh=` overrides on the P1 batch | flagged count **7** at `gate_thresh=0.01`, **4** at `0.65` (the default), **3** at `0.99` — monotonic as expected; `unet_thresh=0.5` populates `unet_mask` with a true-count that **exactly matches** `(unet_prob >= 0.5).sum()` (measured: 31537); leaving both `None` leaves `unet_mask is None` |
| P4 | `python src/crevasse/biomass/run_granule.py --granule <granule dir> --check` — the "start from a real granule" integration test, see below | 200 candidate positions from the coarse valid-fraction prescan (seed 0) -> **200/200** clear the fine-grained valid-data cut -> **200/200** scored by the gate -> **84** flagged -> the U-Net runs on those 84 (mean P(crevasse) over their pixels 0.1684) |

**P1 is the control that matters** — like N1/B1/U1 in the upstream repos, it's an
equality claim (not a tolerance) because `run()` does nothing to `gate_prob`, `gate_flag`,
or the flagged tiles' `unet_prob` beyond calling the two upstream APIs and copying their
output into the right slots. Any deviation from the direct calls is a bug in the
scatter/gather, not drift in a model.

### P4: starting from a real granule, not a hand-picked tile list

P1-P3 build their batch from `tile_granule` directly for convenience, but a production
caller starts from a granule directory, not an in-memory batch. `src/run_granule.py` is
that path — it prescans the granule's 4 polarization GeoTIFFs at a cheap decimated
resolution (`DEC=32`, matching `bio_tile_features.coarse_masks`, minus its
grounded/AlphaEarth/NESZ machinery), re-checks candidates at full resolution, runs
`BiomassCrevassePipeline`, and prints a summary:

```bash
cd biomass-crevasse-pipeline
python src/crevasse/biomass/run_granule.py --granule ../biomass/data/biomass/BIO_S2_SCS__1S_20251217T135546_..._DJT7YH --check
# -> CONTROL P4: 200 candidates -> 200 cleared the valid-data cut, 200 scored, 84 flagged

# a normal run, no assertions, any granule, any sample size:
python src/crevasse/biomass/run_granule.py --granule <dir> --max-tiles 500 --out-npz data/scored.npz
```

`--check` pins `--max-tiles 200 --seed 0` on that granule and asserts the exact
tile/score/flag counts above (the sampling is seeded, so it's deterministic).
`--max-tiles` subsamples the *valid-tile list* rather than taking a prefix of it — the
coarse-prescan order is not a representative sample of tile content, so a prefix would
silently bias any summary statistic derived from the run (same rationale as the NISAR
sibling's `run_granule.py`). Omit `--max-tiles` to run the whole granule; that's slow on
CPU (the U-Net runs on every gate-flagged tile) so it isn't what `--check` uses.

### Reproducing the controls

There's no packaged test tile set here (deliberately — this repo doesn't ship data, only
code, same as `nisar-crevasse-pipeline`). The recipe used to produce the P1-P3 numbers
above:

```python
# pip install -e . first (see the top-level README) -- no sys.path setup needed
import numpy as np
from crevasse.biomass.run_granule import tile_granule
from crevasse.biomass.pipeline_predict import BiomassCrevassePipeline
from crevasse.biomass.bio_predict import BiomassGate
from crevasse.biomass.bio_unet_predict import BiomassUNetSegmenter

granule = "../biomass/data/biomass/BIO_S2_SCS__1S_20251217T135546_..._DJT7YH"
positions, x_m, y_m, inten = tile_granule(granule, max_tiles=12, seed=1)

inten = np.concatenate([inten, np.zeros((1, 4, 512, 512), dtype=np.float32)], axis=0)
x_m = np.append(x_m, x_m[0]); y_m = np.append(y_m, y_m[0])   # the unscoreable tile

pipe = BiomassCrevassePipeline()
out = pipe.run(inten, x_m=x_m, y_m=y_m)
```

## Where things go

| you are adding | put it in |
|---|---|
| a caller convenience (batching, position tracking) that doesn't change gate or U-Net behavior | `src/pipeline_predict.py` |
| a fix to gate scoring itself | `biomass-crevasse-gate`, not here |
| a fix to U-Net scoring itself | `biomass-crevasse-unet`, not here |
| granule tiling / valid-data prescan logic | `src/run_granule.py`'s `tile_granule` — it already wraps the coarse-decimated prescan + fine-grained windowed reads; don't re-derive tiling elsewhere |
| a figure | `src/plot_granule.py` — reuses `tile_granule` and `BiomassCrevassePipeline` rather than re-scoring by hand; it is a viewer, not a control, so nothing in it is pinned |
| a raster/GeoTIFF export | `src/export_geotiff.py` — streams windowed writes rather than building a full array in memory; new export formats go here, not in `run_granule.py` |
| a `scripts/*.sh` wrapper | not built yet — `python src/crevasse/biomass/run_granule.py ...` is the CLI for now; add a thin wrapper if a one-liner is actually needed |

## Before you call a change done

Re-run P1-P3 against the current tree. If any upstream repo's controls haven't been
re-verified recently, do that first — this repo's controls assume they're already honest.
