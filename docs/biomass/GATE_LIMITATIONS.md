# Limitations

Read this before quoting anything from [RESULTS.md](RESULTS.md). Each item below is a reason a
number is narrower than it looks, and none of them is hypothetical — every one was measured.

## 1. There are three folds, and one of them is the evaluation

Pair counts are **2 068 / 216 892 / 3 204**, so any pair-weighted statistic over this data is
**97% fold 1**. Fold 1 has 947 tiles and trains on 446. Folds 0 and 2 have **4 and 12
positives** respectively.

Consequences you cannot design around with the data that exists:

- The pooled AUC and the within-fold AUC **disagree on the sign** of the RF's margin over map
  coordinates (+0.287 against -0.069). Both are correctly computed. The disagreement is the
  result.
- A per-fold difference of a few hundredths is inside the noise of 4 positives.
- Finer blocking would give more folds but smaller ones, and the buffer already consumes
  40.96 km on each side. There is not enough labelled ground for both.

Quote the per-fold table with `n` and positive counts beside it. `fold_n`, `fold_pos`,
`fold_base_rate` and `fold_pairs` are all in the shipped bundle for exactly this reason.

## 2. No operating point is defensible

Covered in full in [RESULTS.md](RESULTS.md) "Operating point". The short form: at P >= 0.65 the
CNN's lift over base rate is **0.00x / 1.67x / 2.45x** across the three folds and the RF flags
**zero tiles** on two of them. The probability scale is fitted to 26-58% positive training
ground and floods on 0.8% positive ground. The ranking transfers; the calibration does not.

This is why `--thresh-rf` / `--thresh-cnn` are required rather than defaulted, why the bundle has
no `thresh` key, and why lift is printed instead of precision.

It is also why the two thresholds are **separate flags** and why `--method both` writes `p_rf`
and `p_cnn` without combining them. Cutting both models at one number would imply the scales are
comparable; at 0.65 one of them flags nothing on two folds and the other flags tiles on all
three. `scripts/run_new_granule.sh` does take a single number and pass it to both, which is a
convenience for the common case — call `src/bio_score_tiles.py` directly to cut them apart, which
is usually what you want.

## 3. The CNN comparison changed two variables at once

The CNN differs from the RF in **both** input representation (raw 4-pol chips vs 38 hand-merged
statistics) **and** model class (conv net vs forest). So "the CNN is better" is true of this
pair and does not isolate *why*.

The open experiment — same CNN, same folds, but a 2-channel merged (`co`, `cross`) input — would
separate the raw-polarimetry effect from the spatial-structure effect. Given that HV/VH
correlate at 0.997-0.999 and HH/VV at 0.990-0.994, the prior is that the gain is spatial and
the raw 4-channel input contributes little. **That has not been tested.** It is the most
important open question about this repo and it is deliberately out of scope for the packaging.

## 4. Ice type cannot be tested at all, by construction

`bedmap_class` is **1 (grounded) for all 4244 labelled tiles**, and `grounded_frac` is exactly
1.0 for **98.8%** of them. Not a coincidence: a `grounded_frac >= 0.95` prescan three stages
upstream admits only grounded tiles. Ice type has zero variance in this dataset, so it cannot
contribute to a model and cannot be evaluated as a confound. Testing it requires relaxing the
prescan, which is a separate decision with its own label implications.

## 5. Ice speed is safe here — and that reverses a warning from NISAR

Velocity alone scores **0.924 pooled but 0.518 buffered**. The buffer removes the shortcut, so
the `sar+vel` arm is not inflated by it.

Record the direction, because it matters more than the number: on NISAR, velocity *was* a real
confound (AUC 0.866 alone) and the context prior was dropped because of it. Here it is not.
**A control must be re-run per sensor, not cited.** Any inherited caveat from the NISAR repo is
a hypothesis about BIOMASS, not a finding.

## 6. All labelled ground is Thwaites, and the repeat passes make it worse

Six granules over three footprints (M01 2487 tiles, M02 1413, M03 344), all overlapping the
same Thwaites region. BIOMASS S1 and S2 are **exact repeat passes** — identical `dr/dc` at M02
and M04 — so 4767 tile slots collapse onto **1806 distinct ground tiles at 2.64 passes each**.

Two things follow:

- **Leave-one-footprint-out is not a cross-ground test.** `(x,y)` alone scores **0.994** there
  and *beats* the model at 0.953. It is in the bundle under `lofo_*_discredited` names.
- **The unbuffered block split is very nearly pure geography**: model 0.954 against `(x,y)`
  0.937, a margin of **+0.016**.

Nothing here has been evaluated on a different ice regime, a different region, or a different
season. The standing decision is to **hold the headline AUC until non-Thwaites ground
arrives** rather than promote the buffered number to a general claim.

## 7. `ratio_dB` drifts per scene

The single strongest feature degrades monotonically as the evaluation widens: **0.890** on the
120-tile pilot, **0.866** within a single granule, **0.764** pooled over six granules
unbuffered, **0.639** buffered. That is per-scene radiometric calibration drift, and it puts a
ceiling on how much of the model's skill can be trusted to survive a new acquisition.

A per-granule normalisation would probably help and has **not** been implemented — it is an
open item, not a fix that exists.

## 8. The labels are incomplete by construction

AlphaEarth: recall ~**0.42** at precision ~**0.94**. So:

- **Every recall number from this repo is a lower bound.** Positives are missing, not wrong.
- Zeros are only usable *inside* the export box. Outside it, byte 0 means "never evaluated";
  those tiles are dropped rather than counted as negative.
- 523 tile slots are dropped as ambiguous (AlphaEarth positive fraction strictly between 0.05
  and 0.5) rather than forced to a side.
- The Earth Engine export recipe that produced these labels has not been recovered, so the
  label set cannot currently be regenerated or extended from scratch.

## 9. The CNN path does not run from a clean checkout

The 4-channel chip caches are 558 MB and 626 MB and are not shipped. Without them you cannot
train the CNN or score new tiles with it. What *is* checkable without them: the checkpoint's
metadata round-trip, and the CNN's per-fold AUCs and lift table, recomputed in pure numpy from
the shipped `data/bio_cnn_oof_c128.npz`. See [CONTRIBUTING.md](CONTRIBUTING.md) controls 4-5,
which say plainly which controls are runnable and which are not.

## 10. A labels-absent feature table can only be scored

`bio_tile_features.py` runs without the AlphaEarth mosaic, because nothing in the 38 features
reads it. What it writes then has `aoi_pos` NaN and `aoi_inbox` False, which means the table
**cannot be trained on and cannot be evaluated** — only scored. `bio_train_gate.py` refuses it and
`bio_tile_cache.py` requires `--all`, so this fails loudly rather than producing an all-negative
fit. Two further consequences worth naming:

- Its `grounded_frac` comes from the 2560 m `context_grid_2560m.npz` rather than a
  native-resolution `bedmap3_mask.tif` read, so a tile straddling the grounding line can be
  admitted or rejected differently than in the shipped table. Measured on the shipped 4767:
  mean |diff| 3.9e-4, max 0.078, and 15 shipped tiles would fall on the other side of the 0.95
  cut.
- The NESZ screen still applies, so a granule whose grounded cross-pol median sits at or below
  −27 dB is skipped and produces no rows at all.

## 11. Two models, two shipped forests, one silent trap

The buffered numbers this repo quotes were measured with `bio_gate_controls.rf` — **300** trees.
The forest that *ships* is 500 trees. `bio_train_gate.py` therefore imports the first as
`rf_eval` for the buffered evaluation, so the quoted figures reproduce exactly, and fits the
second for the bundle. Swapping one for the other moves the numbers by more than the margin
being reported, which is how a pinned control quietly stops pinning anything.

Relatedly: a pickled forest is version-coupled. The bundle records `sklearn_version` and
`numpy_version`, and `bio_score_tiles.py` prints a note naming **both** versions on a mismatch
rather than letting sklearn emit its generic warning.

## 12. The array API's CNN chips are a numpy block mean, not GDAL's

`bio_predict.py` has no raster to read, so it builds the CNN's 128 px chips by block-averaging
the 512 px tile in numpy, where the shipped chip cache used
`rasterio.read(..., resampling=Resampling.average)`. Measured on fully-valid blocks the two are
**bit-identical**, and end to end (CONTRIBUTING.md control B2, 48 tiles) `p_cnn` matches the
shipped `p_full` to **1.29e-4** max, 5.8e-8 median, corr 1.000000.

Exactness is claimed only for fully-valid blocks. A 4x4 block straddling nodata is the one
disagreement — GDAL emits a value there, the block mean emits NaN — so a tile near a swath edge
can deviate more than the number above, which was measured on tiles that all cleared the 0.99
valid cut. Note also that `np.nanmean` is **not** the fix: it is 1.17 dB out on a tile with
interior nodata. The RF path has no such gap and control B1 is bit-identical.

## What this gate is not

It is a **gate**. It says which 2.56 km tiles are worth expensive per-pixel processing. It does
not delineate, count, or measure crevasses, and a flagged tile is a tile to look at, not a
detection.
