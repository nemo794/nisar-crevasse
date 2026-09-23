# The whole pipeline, end to end

Antarctic crevasse detection on NISAR L2 GSLC amplitude imagery. This document is the
walkthrough that no single repo can give you, because the pipeline spans three of them.
It is written to be read cold — by a person or by a language model — and it names the
controls that have to pass at each stage, not just the commands.

**If you read only one thing, read [What you are tempted to skip](#what-you-are-tempted-to-skip).**
Every item on that list is there because skipping it has already cost this project real
work.

## The four repos

| repo | stage | what it is | ships? |
|---|---|---|---|
| `biomass/` | research | everything, including the parts not settled enough to package. `code/` holds ~114 scripts; the `_`-prefixed ones are one-off experiments, the rest are pipeline | no, it's the workbench |
| `nisar-crevasse-gate/` | 4 | the packaged **tile gate**: a random forest that says whether a 2.56 km tile contains a crevasse field. CPU only, torch-free | yes, ~19 MB |
| `nisar-crevasse-unet/` | 5 | the packaged **segmentation** stage: a U-Net that regresses a per-pixel crevasse target. Built to rsync to a GPU cluster | yes, data as shards |
| `nisar-crevasse-pipeline/` | 4+5 | pure plumbing: array-in/array-out coordination over the other two — gate the stack, run the U-Net on survivors only, scatter the masks back to the original stack. No model of its own | yes, code only |

The gate and U-Net repos are *copies* of code that also lives in `biomass/code/`. Two files
have deliberate divergences — `sar_dataset.py` and `train_unet_supervised.py` — so **do not
blindly sync them**. `soft_labels.py` is kept verbatim-identical in both places on purpose:
it is the target definition, and two versions of that would be two experiments.

`nisar-crevasse-pipeline/` has no copy of anything; it imports `gate_predict.NisarGate`
and `unet_predict.UNetSegmenter` from the other two repos' `src/` directly, so it can only
drift by having a stale sibling checkout, not by a forked copy going out of sync.

## The shape of the problem, in one paragraph

There is **no hand-annotated per-pixel ground truth**. Nobody has traced crevasses on these
scenes. So the pipeline manufactures its own supervision: an independent satellite product
(AlphaEarth) says *which tiles* are crevassed, a random forest learns that from SAR texture
so it can be applied to unlabelled tiles, and an unsupervised ridge filter (Frangi) says
*which pixels* within a surviving tile look like a crevasse. The U-Net learns that pixel
target. Consequently **a good validation IoU means the U-Net reproduced the ridge detector**,
not that it found crevasses. Every claim in this project has to be read against that.

```
  NISAR product (HDF5/GeoTIFF, 5-11 GB each)
        |
   [0]  intake: amplitude GeoTIFF, 5 m square pixels, frequencyA HH
        |
   [1]  ACCEPTANCE TESTS  <-- before any labelling effort
        |
   [2]  context: Bedmap3 ice type + ITS_LIVE surface speed, on the tile grid
        |
   [3]  AlphaEarth -> tile-level crevasse/none labels  (context speed-matches the negatives)
        |
   [4]  train the RF gate  ->  gate_5m_freqA_2gran.joblib
        |
   [4b] run the gate over a granule -> per-tile P(crevasse)
        |
   [5]  Frangi soft pseudo-labels on gate survivors -> a ~2 GB .pkl per granule
        |
   [6]  pack: pkl + granule -> a ~1 GB .npz shard (no rasterio needed downstream)
        |
   [6b] context sidecar: per-tile grounded_frac + vel_mean -> ~25 KB .npz
        |
   [7]  train the U-Net on the shard
        |
   [8]  eval, with its control; optional full-scene inference
```

---

## [0] Intake: what counts as usable data

A NISAR granule arrives as an L2 GSLC product. What the pipeline consumes is a **single-band
amplitude GeoTIFF**, geocoded to EPSG:3031 at **5.0 m square** spacing. Drop it in
`data/nisar/`.

The filename is load-bearing and must be kept intact — `GRANULE_RE` in
`gate_common.py` parses the granule ID out of `GSLC_(\d{3}_\d{3})_`:

```
NISAR_L2_PR_GSLC_025_019_A_140_4005_SHSH_A_20260709T131853_..._frequencyA_HH_amplitude.tif
                 ^^^^^^^                                        ^^^^^^^^^^^^^
                 granule ID                                     band + polarization
```

Four hard preconditions, each of which is a guard in code because each has already bitten:

- **`frequencyA`, not `LSAR_HH`.** Three older `LSAR_HH` products were quarantined as
  defective (`data/nisar/lsar_defective/`). A matched-footprint comparison gave R = 0.893
  against 0.439, which convicts the older product rather than our resampling.
- **Square pixels.** `tile_scale()` raises on non-square rasters. A 2.5 × 5.0 m product
  would silently produce 2.56 × 5.12 km tiles and hand every scale-dependent texture
  feature a built-in preferred direction.
- **5.0 m spacing.** The shipped bundle records `train_px_m_positive: [5.0]` and
  `map_crevasse_tiles.py` refuses other spacings without `--allow-unseen-spacing`. A 2.5 m
  granule is a different domain: `read_amp` block-averages to a fixed *ground* size, so a
  2.5 m product is read at 4 looks and a 5 m product at 1 look, and that looks difference —
  not the SAR band — is what caused the cross-granule domain shift.
- **No duplicate granule IDs** in the directory. `granule_rasters()` refuses to proceed.

### Tiling, and the grid mismatch you will hit

Tiles are **512 × 512 samples at 5.0 m = 2.56 × 2.56 km**. Zeros are nodata → NaN, averaging
is NaN-aware, and a tile with under 50% valid data is dropped.

**There are two incompatible tile origins in this project.** Scored/labelled tiles anchor to
the *swath corner* (on 025_019, `row % 512 == 152` and `col % 512 == 4`; on 025_091, 348 and
28), while the AlphaEarth and context grids anchor to *raster origin 0*. So an exact
`(row, col)` join between a label set and a context grid **matches zero rows**. Anything
crossing those grids must do an area-weighted overlap join — that is exactly what
`src/build_tile_context.py` (stage 6b) does. This is a known unfixed wart, not a subtlety
you can design around by being careful.

---

## [1] Acceptance tests — run these before labelling anything

```bash
cd nisar-crevasse-gate
scripts/run_acceptance.sh <new_granule>            # compares against 025_019
scripts/run_acceptance.sh <new_granule> 025_091    # or against 025_091
```

Four tests, in `src/check_granule.py`. **The first two need no labels; the last two do**,
which is the awkward part — the decisive test needs the labelling effort you are trying to
justify. Run 1 and 2 immediately; run 3 and 4 as soon as any labels exist.

1. **`orientation`** (~2 min, no labels) — orientation-concentration screen for a scene-wide
   azimuth-streak artifact. This convicted granule **003_064**: R = 0.948 concentrated at
   73–75° across the *whole scene*, against R = 0.17 on a real crevasse field. Its
   `crevasse` labels turned out to *be* the artifact — all 42 re-reviewed as none. Cheap and
   it has caught a real disaster; always run it.
2. **`register`** (no labels) — co-registration against a known-good granule.
3. **`speckle`** (needs labels) — speckle statistics on matched ground.
4. **`ab`** (needs labels for both + overlapping ground) — the **matched-ground A/B**, and
   the only test that is a verdict rather than a screen. Same tiles, same labels, same code,
   swap only the granule.

**Why this is not boilerplate:** granule **025_048** looked fine to the eye, passed every
cheap screen, was labelled, was trained on — and scores **0.627 where 025_019 scores 0.904
on identical tiles with identical labels**. The cause is still unknown; geolocation, speckle
correlation, advection, labels and coverage have all been closed off. It is excluded from
training and its label CSV ships only so you can reproduce the failure. The A/B would have
caught it *before* the labelling effort.

**The screens can convict a granule. They cannot clear one.** If test 4 could not run, you
do not yet know whether the granule works.

---

## [2] Context: ice type and ice speed

Two external products, neither of them SAR:

- **Bedmap3** surface classification → `grounded_frac`, `bedmap_class`
- **ITS_LIVE** 120 m Antarctic surface velocity mosaic → `vel_mean`, `vel_max` (m/yr)

```bash
cd biomass
python code/build_context_grid.py                          # one continent-wide 2560 m grid
python code/build_context_masks.py --granule 025_019       # per-granule tile-grid masks
# -> data/_context_masks_025_019_t512.npz
```

`build_context_grid.py` produces a single **7.2 MB continent-wide grid at 2560 m**
(`data/context_grid_2560m.npz`) that replaced per-granule warping. Building it is also what
exposed a 1.28-vs-2.56 km cell-size bug that is real on 2.5 m granules.

Bedmap3 class encoding, needed to read anything downstream — **nodata (−9999) is filled to
0, which means open ocean / sea ice**:

| class | meaning |
|---|---|
| 0 | ocean / sea ice |
| 1 | **grounded ice** |
| 2 | transient shelf |
| 3 | floating shelf |
| 4 | rock |

`grounded_frac` is the fraction of a tile that is class 1. It is a *fraction*, not a class,
so consumers threshold it (≥ 0.5, ≥ 0.95) rather than testing equality — a tile straddling a
grounding line is genuinely partial.

### Context is used for three things, and is never a model feature

This is the single most important design constraint in the project, and it was learned the
hard way.

Ice type and speed are strong predictors of crevassing. Adding them as features **raises
AUC**. They are still excluded, because the gain was tested against the control that settles
it: **raw `(row, col)` tile indices as features**. Geography alone scored **+0.031**, more
than the context prior's **+0.027**. The prior was memorising *where* crevasses are in this
scene — worthless on a new scene and actively misleading in a transfer test.

So context is used only for:

1. **Speed-matching the gate's negatives** (stage 3).
2. **Restricting to grounded ice** — at label time and at scoring time.
3. **Splitting and stratifying the U-Net's data** (stages 6b, 7, 8).

**Run the `(row, col)` control on any new label set or feature set.** It is cheap and it is
the only thing that reliably catches this class of confound.

---

## [3] AlphaEarth → tile-level labels

AlphaEarth is a Google satellite-embedding product. A classification mosaic derived from it
is the pipeline's *independent* opinion about where crevasse fields are — independent of
SAR amplitude, of Frangi, and of the U-Net, which is what makes it admissible as an
arbitrator later.

### PREREQUISITE: the export itself is not in this codebase

**Read this before planning work on a new region.** Every stage from here down is scripted
and reproducible. This one is not. The AlphaEarth classification was produced by hand in
the Earth Engine console, and **the EE script was never saved into the repo.** What exists
on disk is only its output:

```
biomass/data/aois/
  old/thwaites_aoi_crevasse_labels-*.tif   4 tiles + thwaites_mosaic.vrt      (1st export)
  new/thwaites_aoi_crevasse_labels-*.tif   4 tiles + thwaites_mosaic_new.vrt  (2nd export)
```

So for a new region, **this is the blocking step, and it needs whoever ran EE, not this
repo.** Without positives there is no gate; without a gate no tiles are selected for
Frangi; without Frangi there are no pixel targets. Nothing downstream can be bootstrapped
around it.

**What is unrecoverable** and must be asked for by name: the AlphaEarth/embedding asset ID
and year, the classifier (type, training points, bands), and the probability threshold that
produced the binary output. None of it is inferable from a uint8 raster of 0s and 1s.

**What IS recoverable — the contract the export must satisfy.** These were measured off the
existing artifacts, and `build_aoi_tile_labels.py` enforces the first four; an export that
violates them exits rather than silently misindexing.

| property | required value | why |
|---|---|---|
| CRS | `EPSG:3031` | must match the granule; `aoi_offset()` refuses a mismatch |
| pixel size | **5.0 m**, square | refused if it differs from the granule by >1e-6 |
| grid origin | on the **granule's** pixel grid | `aoi_offset()` requires an *integer* pixel shift, so a categorical mask is never resampled |
| dtype / encoding | `uint8`, **1 = crevasse, 0 = everything else**, no nodata set | the reader does `== 1`; any other value reads as negative |
| bands | 1 | — |
| extent | ≥ the granule footprint, ideally aligned to it | outside-ROI reads fill with 0, which is indistinguishable from a true negative |

The existing export is **pixel-identical to 025_019** — transform
`(5.0, 0, -1716480 / 0, -5.0, -23760)`, shape `(73728, 72576)`, extent
−1716.5…−1353.6 km × −392.4…−23.8 km, about 363 × 369 km over Thwaites. That is not a
coincidence and it is the thing to copy: in EE, export with `crsTransform` and `dimensions`
taken from the granule rather than with `scale`, and the grids line up exactly. 025_091 sits
on the same 5 m grid at a different origin (`-1720080 / 29520`), which is why it can be
indexed by an integer shift with no interpolation.

**Two properties of the output to carry forward, both measured, both load-bearing.**
Whoever regenerates the export should expect the same and should not be surprised into
"fixing" them:

- **Recall is ~0.42 at precision ~0.94.** It is a conservative, high-precision detector that
  misses most of what is there. That is why `--neg-buffer` has to be wide: near a positive,
  a zero is actively wrong.
- **A zero far from the AOI means "never evaluated", not "no crevasse."** EE writes byte 0
  outside the ROI, and the ROI is smaller than the export box. It is only safe to treat that
  remainder as negative here because it happens to be slow interior ice. In a new region,
  **check that assumption before relying on it** — export the ROI as a second mask band if
  you can, which would remove the ambiguity permanently.

Two known gaps in the current export, unchanged by the second one: **sea ice is not
included**, and the second export's additions are 100% on grounded ice.

```bash
cd biomass
python code/build_aoi_tile_labels.py \
  --granule 025_019 \
  --aoi data/aois/thwaites_mosaic.vrt \
  --neg-buffer 8 \
  --context-npz data/_context_masks_025_019_t512.npz \
  --min-grounded 0.5 \
  --match-neg-speed 1 \
  --n-per-class 0                   # 0 = every qualifying tile
```

The rule:

```
crevasse : aoi_pos_frac >  --pos-thresh (0.25)
none     : aoi_pos_frac == 0  AND no positive tile within --neg-buffer tiles
dropped  : 0 < aoi_pos_frac <= pos-thresh        # ambiguous field edge
```

Output is a CSV of `granule, row, col, label`, where row/col are the tile's top-left corner
in **native raster pixels**. Sampling is *prefix-stable*: candidates are shuffled once with
`--seed` and accepted greedily, so re-running with a larger `--n-per-class` **extends** the
previous set instead of reshuffling it.

Three properties of these labels that you must carry forward:

- **AlphaEarth zeros are NOT negatives.** The ROI is smaller than the export box and Earth
  Engine writes outside-ROI pixels as byte 0, indistinguishable from a true negative. Far
  from the AOI that remainder happens to be slow interior ice, which is the only reason this
  is survivable. Near the AOI the zeros are *actively wrong* — AlphaEarth's recall is ~0.42.
  Hence the wide `--neg-buffer 8`.
- **`--neg-buffer` created a velocity confound, and `--match-neg-speed` is the fix.** Buffered
  negatives are systematically slower-moving ice, and **velocity alone then scored AUC
  0.866**. Speed matching kills most of that. The shipped CSVs are already speed-matched.
- **AlphaEarth labels fields, not tiles.** Precision ~0.94, recall ~0.42. It is a good
  positive-class source and a poor negative-class source.

There are **two AlphaEarth exports**; the second adds +11.9% coverage, all of it on grounded
ice. **Sea ice is still not covered by either**, which matters because 42.6% of RF-only gate
detections are non-grounded.

---

## [4] Train the RF gate

```bash
cd nisar-crevasse-gate
scripts/run_training.sh                      # -> data/gate_retrained.joblib
```

Reproduces the shipped bundle from the shipped CSVs. **Expected: 3692 tiles, random-fold OOF
0.951, spatial-block OOF 0.932, threshold 0.333 — verified bit-identical.** It writes to a
*new* path rather than overwriting `data/gate_5m_freqA_2gran.joblib`, so a retrain that lands
somewhere different is a comparison rather than a loss.

**Model:** `RandomForestClassifier`, balanced class weights, seed 42, **270 features**:

- **14 hand features** — coefficient of variation; GLCM contrast / correlation / homogeneity
  / energy / dissimilarity, each as a mean over 4 angles × 2 distances **and** as an
  anisotropy (std across angles); structure-tensor coherence mean, p90, and fraction > 0.5.
  The anisotropy terms are where directionality shows up.
- **256 FFT features** — four 384-px Hann-windowed corner windows, each transformed and
  binned to a 16 × 16 magnitude grid, then **max-pooled across the four windows**. Pooling
  sub-windows rather than transforming the whole tile is what makes this work: a crevasse
  field filling a quarter of the tile still registers. Use **raw magnitude bins** — an
  earlier version summarised the spectrum into orientation statistics and concluded "FFT
  features are dead"; that conclusion was an artifact of the summarisation.

The tile is **despeckled first**: log10 → NL-means with `h`/`sigma` set from a robust MAD
noise estimate off a median-filter residual → normalise to [0,1]. Speckle *is* texture as far
as a GLCM is concerned, so despeckling is what makes the features measure the scene. (A
heavier PPB despeckler exists in `code/ppb.py` and is better in isolation but ~40× the cost;
it is reserved for the Frangi stage and actively *hurts* the hand GLCM features.)

Deliberately not a neural net: 3692 tiles is small, an ImageNet ResNet18 was evaluated
head-to-head and lost to the 270-feature forest while needing a GPU, and feature importances
are readable — which mattered repeatedly for catching confounds.

### Reading the AUC honestly

Three numbers are printed and they mean different things:

| number | what it is |
|---|---|
| random-fold OOF | **optimistic.** Ice texture is spatially smooth, so a randomly held-out tile usually has a training neighbour a few hundred metres away |
| spatial-block OOF | whole `--block-tiles` × `--block-tiles` blocks held out |
| leave-one-granule-out | per-granule — but the two training granules are the same frame five days apart, so this is closer to a repeat-pass consistency check than a transfer test |

**Treat 0.932 as a reproducibility target, not a performance claim.** At the default 8-tile
(~20 km) blocking, a model given *only the raw tile indices* scores 0.934 — it is tied by
geography. Use `--block-tiles 16` (~41 km) for a number that means something: **0.942 against
a 0.891 position floor, +0.051, stable out to 64 tiles.**

### Two thresholds, and they are not the same number

- **0.333** — the gate's shipped operating point, set as the highest threshold reaching
  **recall ≥ 0.95** out-of-fold. Calibrated for recall because a false negative is a crevasse
  field never looked at, while a false positive is only wasted compute. (Honest caveat: 0.95
  was a plausible round number and was never priced against the actual downstream cost.)
- **0.65** — the threshold used to select which tiles get *Frangi labels* in stage 5
  (`build_crevasse_labels.py --gate-thresh 0.65`, 72.8% recall / 90.5% negative discard OOF).
  A stricter, precision-leaning cut, because a bad tile here poisons the U-Net's supervision.

The threshold is a property of the *bundle*, not the scores. Changing it is a re-render, not
a re-score.

**Corollary that gets forgotten: `gate_prob` may be used as a SELECTOR but never as
EVIDENCE.** Every tile in a label pickle is a gate positive by construction (observed minimum
0.651), so "the gate agrees" is circular there.

---

## [4b] Run the gate over a granule

```bash
cd nisar-crevasse-gate
scripts/run_inference.sh 025_019
# -> data/maps/gate_sar_025_019.npz   (every tile's probability)
# -> data/maps/crevasse_map_025_019.png

# different threshold, seconds not minutes:
python src/crevasse/nisar/render_score_map.py data/maps/gate_sar_025_019.npz --gate-thresh 0.65
```

Needs only the granule and the `.joblib` — no labels, no context rasters, no training data.
On the granules tested it **discards 63–75% of tiles while keeping 95% of crevassed ones**,
which is the whole point: the expensive per-pixel stage then runs on a quarter of the swath.

Three flags deliberately *not* passed: `--normalize-per-granule` (the gate is already
brightness-invariant — amplitude × k gives 40/40 identical probabilities — so this only adds
drift), `--context-prior` (discredited, see stage 2), `--grounded-only` (see the repo's
LIMITATIONS.md).

---

## [5] Frangi soft pseudo-labels

This is where per-pixel supervision is manufactured, and it is the least settled stage in the
pipeline. **Read `src/soft_labels.py` before changing anything** — its docstring carries the
evidence for every constant.

```bash
cd biomass
python code/build_crevasse_labels.py \
  --raster data/nisar/NISAR_..._025_019_..._frequencyA_HH_amplitude.tif \
  --tile-size 512 \
  --speckle-filter ppb_fast \
  --ridge-downscale-factor 4 \
  --ridge-percentile 70 \
  --orient-gate 0.12 \
  --gate-thresh 0.65 \
  --soft-rect-t 0.7 \
  --output-dir data/crevasse_labels
# -> crevasse_labels_025_019_ppb_fast_ridge_tile512_f4_pct70_og12_g65_soft70.pkl  (~2.5 GB)
```

The filename encodes the whole rule (`f4`, `pct70`, `og12`, `g65`, `soft70`) so two label
definitions cannot collide on one path. **Nothing is ever overwritten or renamed in this
project — new rules get new tags.**

The generator (in the research repo, not this one) still writes `.pkl`. This repo's own
pipeline no longer reads pickle anywhere -- convert once with `python
../biomass/scripts/convert_labels_pkl_to_npz.py path/to/that_file.pkl` (writes a `.npz`
alongside it, non-destructively; see `label_io.py`) before `build_shard.py`. That
converter moved out of this repo entirely on 2026-09-22 -- it's the one place that still
deliberately imports pickle (reading a file the generator produced, not a pipeline
dependency), and living outside every repo keeps a pickle-import scan of this one clean
without needing a per-file suppression.

### The target is continuous, not a mask

```
target = clip((frangi_response - 0.08)/(0.48 - 0.08), 0, 1) * rect(orient_agree, 0.7)
         \------------- amplitude: the localizer ---------/   \--- the discriminator ---/
```

- **The previous binary label was a per-tile quota** — it thresholded the Frangi response at
  the 70th percentile *of its own nonzero values*, keeping ~30% by construction. Measured
  coverage was 23.4 / 24.4 / 23.9% on three unrelated tiles. All discrimination lived in the
  tile-level gates and none in the mask.
- **Amplitude alone is anti-correlated with the label.** `preprocess_sar`'s per-tile p2/p98
  contrast stretch amplifies residual speckle hardest on the *emptiest* tiles, so a
  featureless tile carries more Frangi mass than a real crevasse field — tile-mean AUC
  **0.384**, below chance. The `rect(orient_agree, t)` factor is what reverses this, and it
  must be **per-pixel**; a tile-level gate does not fix it.
- `t = 0.7` matches the old binary label's density on the calibration positive (23.74% vs
  23.72%) while the calibration negative falls from 11.61% to **0.53%**.
- Coverage is quoted at a **0.2** threshold everywhere in this project. Keep it there, or
  nothing is comparable.

### The scale band — the thing most likely to surprise you

`ridge_sigmas = [2,4,6,8,10]` are on the **downscaled** grid, so
`--ridge-downscale-factor` slides the whole ridge-*width* band at 5 m native pixels:

| factor | ridge widths it can see |
|---|---|
| f4 (production) | 40–200 m |
| f2 | 20–100 m |
| f1 | 10–50 m |

Production labels are f4, so they are blind **by construction** to individual 10–30 m
crevasses. That is arithmetic, not a hypothesis, and it explains a real measured failure:
65% of the U-Net's false positives far from any label are "Frangi-blind" — the ridge
detector looked there and found nothing.

A multiscale rule is wired in but **the currently-tested version was refuted**:

```bash
--ridge-factors 4,2,1 --no-orient-rect        # target = mean_s soft(resp_s), no rect
```

The U-Net *learned the speckle* this produced: mean connected-component size fell from
157 px (f4) to 16 px, area in sub-200-px components rose to 82.5%, and a 100 m crevasse at
5 m pixels is ≥ 60 px however thin — so those are dots, not crevasses. A shape filter and a
high-gate-P tile cut were both pre-registered as fixes and **both refuted** on AlphaEarth-
arbitrated tiles. The evidence supports `{4,2}` with a **per-scale** `t_s` instead; `RECT_T =
0.7` is an artifact of f4's bilinear upsampling and does not transfer. That work is open.

**Cost and a macOS trap:** all three scales run ~4 s/tile, so ~2100 tiles is ~2.4 h
single-core. Parallelise with a `__main__` guard — bare `Pool` calls at module top level
**hang forever on macOS**, and the symptom is silence, so check CPU time with `ps aux` rather
than log output before concluding a job is stuck.

---

## [6] Pack a shard

```bash
cd nisar-crevasse-unet
python src/crevasse/nisar/build_shard.py \
  --results-npz .../crevasse_labels_025_019_..._soft70.npz \
  --raster .../GSLC_025_019_..._frequencyA_HH_amplitude.tif \
  --out data/train_025_019_soft70.npz
```

**Why this stage exists.** The label files are 1.8–2.4 GB and the granules they index are
4.7–10.5 GB — ~19 GB for two granules, of which the labelled tiles are about 0.3 GB of real
pixels. A shard stores each labelled tile beside its target and drops the superseded
`binary_mask` and unused statistics: **~1.05 MB/tile, and no rasterio at training time.**

That last part is load-bearing, not cosmetic. Training and eval from a shard **do not import
rasterio at all** — `sar_dataset.py` defers that import into the two raster-backed datasets —
so a broken GDAL stack cannot stop a training run. On the cluster this is concrete:
conda-forge's `libgdal.so.38` needs `GCC_12.0.0` symbols the system `libgcc_s.so.1` does not
export, and no gcc ≥ 12 module exists there.

Two deliberate choices:

- **Amplitude is stored raw**, with preprocessing left in the loader, so the shard path and
  the GeoTIFF path are **interchangeable**.
- **`float16`** resolves ~5e-4 in [0,1], three orders finer than the 0.2 coverage threshold.
  `check_labels.py --compare-labels` verifies shard-against-label-file agreement (measured
  max abs diff 2.44e-04). Where a float16 control is needed on a *derived* quantity, use the
  **threshold-crossing rate**, not the max-abs diff.

Per-tile fields packed alongside: `positions`, `coverage`, `orient_conc`, `gate_prob`,
`tile_size`. Those last two are what make the eval stratification free.

### Then run the acceptance test. Every time.

```bash
scripts/run_check.sh data/train_025_019_soft70.npz
# 14 checks, 17 with --compare-labels; each prints its measurement beside its bound; nonzero exit on failure
```

It exists because **three of the four things that have gone wrong at this stage were silent
rather than crashes**:

- A tile filter kept 0 of 510 tiles and training proceeded on an empty set.
- A train/val split put spatially adjacent tiles on both sides.
- `A.GaussNoise(var_limit=...)` — albumentations 2.x renamed that kwarg to `std_range` and
  **silently drops unknown kwargs**, so the augmentation ran at its default
  `std_range=(0.2, 0.44)`: gaussian noise at up to half the dynamic range of a [0,1] tile,
  burying the very speckle texture the model has to read. The code *read* as if it were
  asking for std 0.03.

None of those raise. All three are now checked.

---

## [6b] The context sidecar

```bash
cd nisar-crevasse-unet
python src/crevasse/nisar/build_tile_context.py \
  --shard data/train_025_019_soft70.npz \
  --context-masks ../biomass/data/_context_masks_025_019_t512.npz \
  --out data/context_025_019_t512.npz          # ~25 KB
```

One per granule. It joins Bedmap3 `grounded_frac` and ITS_LIVE `vel_mean` onto the shard's
tiles and is required by the U-Net's split.

Three reasons it is a sidecar rather than a shard field: Bedmap3 and ITS_LIVE need rasterio
and live in `biomass/`, which the training machine does not have; ~25 KB rsyncs beside a 1 GB
shard for free; and it leaves existing shards **byte-identical**, so every number already
quoted against them stays reproducible.

**The join is approximate by construction** — see the grid mismatch in stage 0. It is an
area-weighted mean over the context cells each tile straddles, which is why a
grounding-line tile gets a fraction and consumers threshold rather than test equality.

---

## [7] Train the U-Net

```bash
rsync -avP nisar-crevasse-unet/ user@cluster:~/nisar-crevasse-unet/
# the two ~25 KB context sidecars must go too

cd ~/nisar-crevasse-unet/jobs
sbatch setup_env.slurm        # once per machine; REPLACE=1 to rebuild (renames aside, deletes nothing)
sbatch check_shards.slurm     # RUN THIS FIRST
sbatch train_unet.slurm       # check -> train -> eval (with its control) -> figures, one job
```

`jobs/common.sh` **aborts if `torch.cuda.is_available()` is False** rather than letting
training fall back to CPU — otherwise a 12-hour allocation runs correctly and 50× too
slowly, which looks like success.

Defaults: 50 epochs, batch 8, AdamW lr 1e-4, `base_channels` 64, `ReduceLROnPlateau` on val
loss, BCE + soft Dice + 0.1 × flip-consistency, from scratch (MAE pretraining exists in
`train_mae_pretrain.py` and is optional; it needs a GeoTIFF).

### The split: grounded ice only, 3×3-tile blocks

```
--min-grounded 0.5 --split-mode blocks --block-tiles 3 --split-buffer-tiles 1 \
--context-npz data/context_025_019_t512.npz
```

The earlier runs used a contiguous spatial band, and **Bedmap3 says 68.9% of that validation
band was floating shelf or sea ice.** Grounded ice sits at low rows on 025_019, and a band
takes the last 20% of the sorted axis — so validation landed on the shelf *by construction*,
on any seed. Shelf rifts and grounded-ice crevasses are different physical processes, and
the contamination flattered the headline: non-grounded pooled IoU 0.5855 (f4) / 0.5389
(mean{4,2,1}) against grounded-only **0.5500 / 0.4244**.

A grounded-only *band* does not fix it — it puts validation on slow interior ice (f4 coverage
2.02% vs train 8.05%), and since **IoU here tracks `orient_conc` at r = +0.84**, that swaps
one confound for another. Holding out whole blocks samples the same difficulty mix from
disjoint geography. Block size was swept; **8×8 is the worst option** (folds ran to
4.88%/11.06% coverage), because only ~8 blocks land in validation so *which* ones dominates.
3×3 gives 466 train / 160 val at 6.80%/5.50% coverage and 0.557/0.532 `orient_conc`.

Blocks are chosen by **systematic sampling down an ITS_LIVE surface-speed ordering**, not by
a seed — `--split-seed` only selects which of k=5 folds you get. Speed is admissible because
it is label-independent (r = +0.45 with both rules' coverage, not derived from the imagery).
**Do not fish for a seed** — that is fitting the split to the metric.

**The known limit, to report rather than paper over: minimum train/val separation is 5.1 km,
and Thwaites crevasse fields are tens of km across.** Some validation tiles almost certainly
sit in the same field as a training tile. This is a difficulty-matched holdout, **not** clean
spatial generalization.

### Three checkpoints, on purpose

`unet_best.pth` (max val IoU), `unet_best_loss.pth` (min val loss), `unet_last.pth`. All
three carry `val_iou` and `val_loss`. Selecting on val IoU alone is not defensible: over the
last 10 epochs of one run val IoU was **0.504 ± 0.039** while val loss was still improving
past the epoch the IoU picked. **Score all three before quoting one.**

Two consequences of the continuous target, both already handled: `BCEWithLogitsLoss` takes
targets in [0,1] natively and the Dice term is already a soft Dice, so **no loss change is
needed**; and **val IoU thresholds the target as well as the prediction**, both at 0.2 —
comparing a binarized prediction against a continuous mask put a hard count and a soft mass
in the same union term and the number was not readable.

---

## [8] Evaluate

```bash
# the CONTROL: same shard, val subset, same batch size
python src/crevasse/nisar/eval_shard.py --checkpoint models/<run>/unet_best.pth \
  --shard data/train_025_019_soft70.npz --subset val --batch-size 8 \
  --sweep --out-npz models/<run>/eval_019_val.npz

# the held-back acquisition -- note its OWN context sidecar
python src/crevasse/nisar/eval_shard.py --checkpoint models/<run>/unet_best.pth \
  --shard data/train_025_091_soft70.npz --subset all \
  --context-npz data/context_025_091_t512.npz \
  --batch-size 8 --sweep --out-npz models/<run>/eval_091_all.npz

python src/crevasse/nisar/plot_eval.py --eval-npz models/<run>/eval_019_val.npz --tb-logdir models/<run>/logs
```

### The control that must pass

`--subset val` on the **training** shard at the **same `--batch-size`**: the per-batch IoU
must reproduce the checkpoint's own recorded `val_iou`. The script prints **`MATCH` /
`MISMATCH`** and has matched to `|d| 0.0000`. On `MISMATCH`, the split or the preprocessing
has drifted and **no other number it printed is readable**. It reconstructs the split by
calling the same `split_indices` and `pool_mask` that training called, so the control cannot
pass for the wrong reason.

`eval_shard.py` also **refuses to run** if the checkpoint trained with `--min-grounded` and
the sidecar handed to it does not match the shard's tiles — falling back to "no filter" would
score shelf tiles the model never trained on and call it generalization.

### Three IoUs, not interchangeable

| | what it is | use it for |
|---|---|---|
| **pooled** | one intersection and one union over every pixel | the headline; independent of batch size |
| **per-tile** | mean of per-tile IoU | the pessimistic read; weights a sparse tile like a full one |
| **per-batch** | mean of per-batch IoU at `--batch-size` | **only** this reproduces what the training script printed — it exists to be the control |

Target coverage is printed beside predicted coverage, because an IoU is not readable without
knowing how much there was to find. Stratification by `orient_conc`, `gate_prob`, target
coverage and **`grounded_frac`** is free. `grounded_frac` uses **fixed** cuts (<0.05 /
0.05–0.95 / ≥0.95) rather than terciles, so the bands mean the same thing on every shard.

`--sweep` moves the **prediction** threshold with the target threshold held fixed; moving
both makes the curve unreadable. The best value on a sweep is fitted to the subset you swept
it on, so **quote 0.5** unless the threshold was chosen elsewhere.

### Optional: full-scene inference

`infer_unet.py` / `infer_unet_batch_gpu.py` predict over a whole swath and therefore need the
granule and rasterio. `infer_from_labels.py` runs only on labelled tiles and is the right
tool for comparing prediction against pseudo-label.

### A production path: [4b] -> [4+5 in memory] -> restore the stack

For a caller that already holds a stack of tiles (rather than a GeoTIFF on disk) and
wants stage 4 and stage 5 wired together, `nisar-crevasse-unet/src/unet_predict.py`
(`UNetSegmenter`, array in / array out, mirrors `gate_predict.NisarGate`) and
`nisar-crevasse-pipeline/src/pipeline_predict.py` (`CrevassePipeline`, filters through
the gate then runs the U-Net only on survivors, scatters the result back to the full
stack) replace this whole section with:

```python
from pipeline_predict import CrevassePipeline
out = CrevassePipeline().run(amp)     # amp: (N, 1, 512, 512) linear amplitude, 5 m
# out.gate_prob, out.gate_flag, out.unet_prob (NaN where the gate rejected), out.unet_mask
```

Both new modules carry their own pinned controls (U1 in `nisar-crevasse-unet`, P1-P3 in
`nisar-crevasse-pipeline`) the same way `gate_predict.py`'s N1-N3 do. Neither retrains,
refits, or changes any number quoted in this document — they are additive, the same
design choice made for `gate_predict.py`.

---

## What you are tempted to skip

Ordered by how much it has already cost.

1. **`run_acceptance.sh` on a new granule, before labelling it.** 025_048 was labelled and
   trained on and is unusable; 003_064's positive labels were an instrument artifact.
2. **`run_check.sh` on a shard, every time.** Three of four failures at that stage were
   silent.
3. **The `--subset val` eval control.** Without `MATCH`, every other number in the run is
   unreadable.
4. **The `(row, col)` confound control on any new label set or feature set.** It is the only
   cheap test that catches memorised geography.
5. **The paired background/control for any "recovery" or "coverage" claim.** A label that
   merely covers more area lifts every recovery rate. One measured example: a naive
   far-from-label false-positive count fell 182408 → 21628, which looks like a fix — but the
   available ground shrank from 143.7 M to 3.69 M pixels, so as a *share* the new model fired
   **4.6× more**, and scored against the fixed original label it produced a **32×** expansion.
6. **Stratifying by `grounded_frac`.** Not doing it is how a val IoU that was 69% shelf got
   reported as a crevasse result.
7. **Reading `soft_labels.py`'s docstring before touching the target.** Every constant in it
   is a measurement.

## Standing rules in this project

- **Nothing is deleted or renamed.** Superseded artifacts move to a directory; new rules get
  new filename tags. Every number ever quoted must stay reproducible.
- **The f4 label path must stay bit-identical** when new rules are added (verified:
  max |d| = 0.000e+00), for the same reason.
- **`gate_prob` is a selector, never evidence** — label pickles are gate-selected, so every
  tile in one is a gate positive by construction.
- **Stratifying label quality by Frangi coverage is circular.** The admissible independent
  yardsticks are AlphaEarth, Bedmap3 and ITS_LIVE.
- **Never tune on the test statistic.** If far-from-label recovery is the test, it cannot
  also be the objective.
- **025_091 is not a generalization test.** Same track 140, frame 4005, five days after
  025_019, 82% bbox overlap, 73% shared ground. It measures acquisition-to-acquisition
  stability, however good the number is.

## What is genuinely unresolved

- **Why 025_048 fails the matched-ground A/B** (0.904 vs 0.627 on identical tiles and
  labels). Geolocation, speckle correlation, advection, labels and coverage are all closed
  off. Look direction is untested and needs the original HDF5.
- **The label rule.** `mean{4,2,1}` is refuted; `{4,2}` with a per-scale `t_s` is the
  candidate, to be screened on AlphaEarth-arbitrated tiles for coverage *and* component shape
  **before** spending another training run.
- **57.3% of the U-Net's far-from-label false positives are found by no Frangi scale at
  all**, so multiscale accounts for under half of what the model fires on.
- **Why high-confidence gate positives are enriched off grounded ice.**
- **The tile-grid origin mismatch** (stage 0) is worked around, not fixed.
- **One place only.** Every granule here is track 140 / frame 4005. Granules from other
  regions have been requested and are the only route to an honest leave-one-granule-out.
- **The AlphaEarth export is not reproducible from this codebase** (stage [3] prerequisite).
  Recovering the EE script is the single highest-value thing to do before new-region data
  lands, because it blocks every stage after it.

## Adapting to a new region: what transfers and what does not

Ranked by how much work each tier needs. The tooling is in better shape than the *evidence*.

| | new acquisition, same frame | new granule, elsewhere, 5 m frequencyA | new region needing its own labels |
|---|---|---|---|
| intake guards, tiling | works | works | works |
| acceptance screens 1–3 | works | works | works |
| **acceptance step 4 (the A/B)** | works | works | **cannot run** — needs labels on overlapping ground |
| ice type (Bedmap3) | works | works — continent-wide, ±3334 km | works |
| ice speed (ITS_LIVE) | works | check for nodata over the footprint first | check first |
| **AlphaEarth labels** | not needed | not needed if inside the Thwaites box | **BLOCKED** — see stage [3] |
| RF gate | works | **unvalidated**: LOGO AUC 0.59/0.35 vs pooled 0.911 | needs retraining, hence labels |
| Frangi scale band | works | **recalibrate** — f4 covers 40–200 m ridge widths at 5 m px |  recalibrate |
| shard, U-Net, eval | works | works | works |

Three traps specific to new ground:

- **Step 4 is the only test that has ever caught a bad granule the screens passed** (025_048:
  0.627 against 0.904 on identical tiles). On new ground it cannot run, so you are relying on
  screens that can convict a granule but cannot clear one. Plan for that rather than
  discovering it.
- **Frangi's scale band is physical, not a hyperparameter.** If the new region's crevasses are
  a different width, f4 is blind to them *by arithmetic* — the same way it was blind to 10–30 m
  features here.
- **2.5 m granules are a different family.** Resampling to 5 m is the existing workaround, but
  `read_amp`'s looks asymmetry (trains at 1 look, deploys at 4) is the documented cause of the
  cross-granule shift, and equalising it was built and made things *worse*.

## Where the reasoning is written down

- `docs/nisar/` — `METHOD.md`, `RESULTS.md` (every number with its control),
  `LIMITATIONS.md` (**read before quoting any number**), `DATA.md`, `CONTRIBUTING.md`.
- `src/crevasse/nisar/soft_labels.py` — the target definition and the evidence for every
  constant.
- `docs/nisar/PIPELINE_CONTRIBUTING.md` — the gate->U-Net coordination controls (P1-P4)
  and why this repo deliberately does not preserve BIOMASS's torch-free invariant.

**The long-term goal is porting this framework to ESA BIOMASS.** NISAR is the pathfinder, the
framework is the deliverable — which is why the acceptance tests and the controls baked into
every script matter more than any particular AUC on a Thwaites frame. Rank work by whether it
survives a sensor change.
