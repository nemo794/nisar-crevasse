# Adding to this repo

This repo has one module, `src/pipeline_predict.py`, and it exists to do exactly one
thing: filter a stack of tiles through `NisarGate`, run `UNetSegmenter` only on the
survivors, and scatter the result back into a full-length stack. It has no model of its
own, so most of the discipline in `nisar-crevasse-gate/docs/CONTRIBUTING.md` (feature
engineering, bundle protocol, granule acceptance) doesn't apply here — read that file for
those. What applies here is narrower: **every measurement ships with its control**, same
as both upstream repos.

## The one invariant

**The gate filter is not optional.** `CrevassePipeline.run` always scores the gate first
and only calls `UNetSegmenter` on tiles `gate_flag` accepted. There is no argument to
bypass this — see the module docstring for why (the Frangi pseudo-labels the U-Net
trained on fire on pure speckle, so the U-Net has never been evaluated off-gate).

## What this repo does NOT preserve

Unlike `biomass-crevasse-gate`'s Invariant 1 (the RF path stays torch-free even though
the CNN path needs torch), **this repo makes no attempt to keep any import path
torch-free.** `pipeline_predict.py` imports both `NisarGate` and `UNetSegmenter` at
module load, and the latter needs torch unconditionally. `environment.yml` is the union
of both upstream envs for this reason. Do not go looking for a lazy-torch pattern here —
it was never a goal, because there is no code path in this repo that runs without both
models.

## Controls

Verified 2026-09-16, `conda env` with both upstream envs' dependencies (torch, rasterio,
scikit-image, scikit-learn — `biomass` on this machine has all of them; `environment.yml`
here is the portable version for a fresh checkout). Re-verified 2026-09-22 after
`UNetSegmenter`'s input convention changed from `(N, 512, 512)` to `(N, 1, 512, 512)`
(to match `BiomassUNetSegmenter`'s `(N, 4, 512, 512)`) — all P1-P3 numbers below are
unchanged, as expected: the shape change only affects how `pipeline_predict.py` calls
`UNetSegmenter` internally (`a[idx][:, None]`), not `CrevassePipeline.run`'s own public
`(N, 512, 512)` contract, which still matches `NisarGate`'s.

| # | check | expected |
|---|---|---|
| U1 (lives in `nisar-crevasse-unet`, quoted here for context) | `UNetSegmenter.predict_proba` on 24 real `025_019` shard tiles vs `eval_shard.py`'s own inference path (same checkpoint, same `ShardCrevasseDataset`) | **maxdiff 0.0e+00** — exact, see `nisar-crevasse-unet/README.md` |
| P1 | 13-tile batch (6 `crevasse`-labelled, 6 `none`-labelled, 1 all-zero) cut from real `025_019` amplitude via `read_amp`, run through `CrevassePipeline().run()` | `gate_prob`/`gate_flag` **identical** to calling `NisarGate()` directly on the same batch; gate flagged **6 of 13** (exactly the 6 `crevasse`-labelled tiles); `unet_prob` at the 6 flagged indices **bit-identical** (maxdiff 0.0) to calling `UNetSegmenter().predict_proba()` directly on `amp[idx][:, None]`; `unet_prob` all-`NaN` at all 7 unflagged indices |
| P2 | the all-zero tile from the P1 batch | `gate_prob` is `NaN`, `gate_flag` is `False`, and its slot in `unet_prob` stays all-`NaN` — it never reaches the U-Net (the RF's own nodata guard does the excluding; nothing in this repo re-checks validity) |
| P3 | `gate_thresh=`/`unet_thresh=` overrides on the P1 batch | flagged count **11** at `gate_thresh=0.01`, **6** at the default (bundle threshold), **0** at `gate_thresh=0.99` — monotonic as expected; `unet_thresh=0.5` populates `unet_mask` with a true-count that **exactly matches** `(unet_prob >= 0.5).sum()` (measured: 83625); leaving both `None` leaves `unet_mask is None` |
| P4 | `python src/crevasse/nisar/run_granule.py --granule <025_019 GeoTIFF> --check` — the "start from a real granule" integration test, see below | 300 candidate positions sampled from the swath (seed 0) -> **169** clear `read_amp`'s 50%-valid-data cut -> **169/169** scored by the gate -> **43** flagged -> the U-Net runs on those 43. Reproduced on 025_091 with `--max-tiles 80 --seed 1` (not pinned, just a second-granule smoke check): 50/80 valid, 18 flagged |

**P1 is the control that matters** — like N1 and B1 in the two upstream repos, it's an
equality claim (not a tolerance) because `run()` does nothing to `gate_prob`, `gate_flag`
or the flagged tiles' `unet_prob` beyond calling the two upstream APIs and copying their
output into the right slots. Any deviation from the direct calls is a bug in the
scatter/gather, not drift in a model.

### P4: starting from a real granule, not a hand-picked tile list

P1-P3 build their batch from a labelled CSV, which is convenient for controls but not
how a production caller works: they have a granule, not a tile list. `src/run_granule.py`
is that path — it tiles a real GeoTIFF itself (`find_data_swath` for the swath bounds,
`train_gate_classifier.tile_scale`/`read_amp` for the native-grid tiling and reads, same
as `map_crevasse_tiles.py` uses), runs `CrevassePipeline`, and prints a summary:

```bash
cd nisar-crevasse-pipeline
python src/crevasse/nisar/run_granule.py --granule ../nisar-crevasse-gate/data/nisar/NISAR_..._025_019_..._frequencyA_HH_amplitude.tif --check
# -> CONTROL P4: 300 candidates -> 169 cleared read_amp, 169 scored, 43 flagged

# a normal run, no assertions, any granule, any sample size:
python src/crevasse/nisar/run_granule.py --granule <path> --max-tiles 500 --out-npz data/scored.npz
```

`--check` pins `--max-tiles 300 --seed 0` on 025_019 and asserts the exact tile/flag
counts above (the sampling is seeded, so it's deterministic). `--max-tiles` subsamples
the *valid-tile list* rather than taking a prefix of it — the swath corner is not a
representative sample of tile content, so a prefix would silently bias any summary
statistic derived from the run. Omit `--max-tiles` to run the whole swath; that's slow on
CPU (the U-Net runs on every gate-flagged tile) so it isn't what `--check` uses.

### Reproducing the controls

There's no packaged test tile set here (deliberately — this repo doesn't ship data, only
code). The recipe used to produce the numbers above:

```python
# pip install -e . first (see the top-level README) -- no sys.path setup needed
import numpy as np, pandas as pd, rasterio
import crevasse.nisar.train_gate_classifier as T
from crevasse.nisar.gate_predict import NisarGate
from crevasse.nisar.unet_predict import UNetSegmenter
from crevasse.nisar.pipeline_predict import CrevassePipeline

df = pd.read_csv("<your tile_labels_aoi_025_019_speedmatched.csv>")  # see docs/nisar/DATA.md
pos = df[df.label == "crevasse"].sample(6, random_state=1)
neg = df[df.label == "none"].sample(6, random_state=1)

granule = "<your NISAR_L2_PR_GSLC_025_019_..._frequencyA_HH_amplitude.tif>"
tiles = []
with rasterio.open(granule) as src:
    for _, r in pd.concat([pos, neg]).iterrows():
        t = T.read_amp(src, int(r.row), int(r.col))
        tiles.append(np.nan_to_num(t, nan=0.0))
tiles.insert(6, np.zeros((512, 512), dtype=np.float32))   # the unscoreable tile
amp = np.stack(tiles).astype(np.float32)

pipe = CrevassePipeline()
out = pipe.run(amp)

# to reproduce the "vs direct call" half of P1, note UNetSegmenter itself now takes an
# explicit channel axis: seg.predict_proba(amp[idx][:, None]), not amp[idx]
```

## Where things go

| you are adding | put it in |
|---|---|
| a caller convenience (batching, position tracking) that doesn't change gate or U-Net behavior | `src/pipeline_predict.py` |
| a fix to gate scoring itself | `nisar-crevasse-gate`, not here |
| a fix to U-Net scoring itself | `nisar-crevasse-unet`, not here |
| granule tiling / swath-bounds logic | `src/run_granule.py`'s `tile_granule` — it already wraps `find_data_swath` + `train_gate_classifier.tile_scale`/`read_amp`; don't re-derive tiling elsewhere |
| a figure | `src/plot_granule.py` — reuses `tile_granule` and `CrevassePipeline` rather than re-scoring by hand; it is a viewer, not a control, so nothing in it is pinned |
| a raster/GeoTIFF export | `src/export_geotiff.py` — streams windowed writes rather than building a full array in memory; new export formats go here, not in `run_granule.py` |
| a `scripts/*.sh` wrapper | not built yet — `python src/crevasse/nisar/run_granule.py ...` is the CLI for now; add a thin wrapper if a one-liner is actually needed |

## Before you call a change done

Re-run P1-P3 against the current tree. If either upstream repo's controls (N1-N3, B*, U1)
haven't been re-verified recently, do that first — this repo's controls assume both of
those are already honest.
