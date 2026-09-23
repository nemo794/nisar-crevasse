# Results

All numbers from `data/gate_5m_freqA_2gran.joblib`, trained 2026-09-08. Reproduce with
`scripts/run_training.sh`.

Read [LIMITATIONS.md](LIMITATIONS.md) alongside this. Every number below is honest about
what it measured; what it measured is narrower than it looks.

## The bundle

| | |
|---|---|
| training tiles | 3692 |
| training granules | 025_019, 025_091 (both track 140, frame 4005) |
| pixel spacing of positives | 5.0 m only |
| features | 270 (14 hand + 256 FFT) on NL-means despeckled log amplitude |
| threshold | 0.333, calibrated to recall ≥ 0.95 out-of-fold |
| context features | none (`ctx_keys: []`) — see METHOD.md |

## Separation

**Read the `(row, col)` column before the AUC column.** It is a model given nothing but the
tile's raw indices — no pixels, no physics. Where it matches the SAR model, the protocol is
not measuring skill.

| measure | SAR 270 | `(row, col)` only | margin | what it means |
|---|---|---|---|---|
| random-fold OOF | 0.951 | 0.981 | −0.030 | optimistic both ways — neighbouring tiles leak across folds |
| spatial-block, 8×8 tiles (~20 km) | **0.932** | **0.934** | **+0.000** | the shipped bundle's number. **Blocks too small — position ties it.** |
| spatial-block, 16×16 tiles (~41 km) | **0.942** | 0.891 | **+0.051** | **quote this one** |
| LOGO 025_019 | 0.921 | 0.876 | +0.045 | |
| LOGO 025_091 | 0.943 | 0.865 | +0.078 | |
| **matched-ground A/B** | **0.904 vs 0.627** | n/a — position held fixed | | the load-bearing measurement; see below |

**The `(row, col)` column is the point of this table.** It is a model given nothing but the
tile's raw indices — no pixels, no physics. Where it matches the SAR model, the protocol is
not measuring skill.

At the shipped 8×8 blocking it matches exactly, and the reason is that the block is smaller
than the thing being predicted: the crevasse field is far larger than 20 km, so a held-out
block's label is still predictable from where it is. This is a property of the labels, not of
the forest, which never sees `row`/`col`.

Widening the block fixes it. Measured 2026-09-08:

| block | ground | blocks | SAR | `(row, col)` | margin |
|---|---|---|---|---|---|
| 8 | 20.5 km | 124 | 0.935 | 0.934 | +0.000 |
| **16** | **41.0 km** | **42** | **0.942** | **0.891** | **+0.051** |
| 32 | 81.9 km | 17 | 0.922 | 0.878 | +0.044 |
| 64 | 163.8 km | 7 | 0.908 | 0.862 | +0.046 |

The margin opens at 41 km and is then **stable at +0.044 to +0.051 across a four-fold change
in block size**, which is the reassuring part — it is not a number that keeps sliding as the
control gets stricter. Use `--block-tiles 16`. (At 64 there are only 7 groups for 5 folds, so
read that row as a consistency check, not a measurement.)

The evidence that the gate reads pixels at all is the **matched-ground A/B**: position,
labels, folds and code all held fixed, only the source pixels vary, and AUC moves 0.904 →
0.627. Position cannot explain that, because position did not change.

So: **0.932 is a reproducibility target, not a performance claim** — it is what the shipped
bundle recorded and what `scripts/run_training.sh` must reproduce. The performance claim is
0.942 at 41 km blocking, against a position floor of 0.891.

The two LOGO numbers look like a transfer result and are not — see LIMITATIONS.md.

### Controls that were run

- **`(row, col)` control — convicts the 8×8 blocking, clears the gate at 16×16.** Measured
  2026-09-08 on the shipped label set (1846 native tiles, 889 positive); see the sweep above.
  Cross-granule leakage is *not* the mechanism: pooling blocks across granules so shared
  ground is held out together *raises* position to 0.972, and it is block width that matters.
  Reproduce with the recipe in [CONTRIBUTING.md](CONTRIBUTING.md). This control killed the
  context prior once already; here it convicts a CV setting rather than a feature.
- **Speed-matched negatives.** Velocity alone scored 0.866 on an earlier label set built with
  a spatial buffer. The shipped CSVs are speed-matched, which removes that.
- **Square-pixel and duplicate-granule-ID guards** were in place for this training run and
  fired on nothing.

## Keep rates at deployment

Fraction of scored tiles retained, per granule, at three thresholds. These are what the
gate actually buys downstream. ("Tiles scored" is smaller than the candidate count — 9326
candidates on 025_019 — because tiles under 50% valid data are dropped.)

| granule | tiles scored | P ≥ 0.333 | P ≥ 0.65 | P ≥ 0.80 |
|---|---|---|---|---|
| 025_019 | 9236 | 2327 (25.2%) | 1476 (16.0%) | 1084 (11.7%) |
| 025_091 | 4470 | 1659 (37.1%) | 1148 (25.7%) | 859 (19.2%) |
| 025_048 | 9505 | 3092 (32.5%) | 1851 (19.5%) | 1356 (14.3%) |

So at the calibrated threshold the gate discards 63-75% of tiles.

Two caveats that matter more than the numbers:

- **025_048's rate is not interpretable.** That granule fails a matched-ground A/B (see
  below). Its keep rate is reported for completeness, not use.
- **These are upper bounds.** They were measured without `--grounded-only`, so they include
  tiles on floating shelf and sea ice.

## A finding that argues against raising the threshold

Raising the threshold makes maps look cleaner while making the surviving detections *less*
likely to be grounded-ice crevasses.

Share of retained positives that fall on grounded ice:

| granule | base rate | P ≥ 0.333 | P ≥ 0.65 | P ≥ 0.80 |
|---|---|---|---|---|
| 025_019 | 93.1% | 74.0% | 65.8% | 61.0% |
| 025_091 | 95.5% | 88.0% | 83.9% | 82.1% |
| 025_048 | 82.4% | 52.8% | 37.9% | 31.9% |

Every column moves the wrong way. On 025_048 at P ≥ 0.80 the retained set is enriched ~2.6×
*off* grounded ice relative to base rate. Whatever the highest-confidence detections are
keying on, it is more common on floating ice — plausibly shelf and sea-ice rifts, which are
real linear features and arguably a labelling-definition question rather than an error.

This is unresolved. It is the reason `--grounded-only` is documented but not made the
default: making it default would hide the effect rather than explain it.

## The granule that does not work

`025_048` (track 140, frame 4005, 2026-07-11) was labelled and added to training to get the
project's first leave-one-granule-out number on partly-unseen ground. It does not work.

The controlled measurement: 359 tiles of ground shared with `025_019`, **identical labels**,
same `read_amp`, same `featurize`, same folds. Only the source granule varies.

| read from | spatial-block OOF |
|---|---|
| 025_019 (control) | **0.904** |
| 025_048 | **0.627** |

Reproduce with `python src/crevasse/nisar/check_granule.py ab 025_048 025_019`. The known-good control pair
returns a +0.004 delta, so the harness is sound.

Ruled out: label misplacement, partial coverage, pixel geometry, a scene-wide oriented
artifact, geolocation error (phase cross-correlation median shift 0 px), speckle correlation
(a real 1.6× azimuth difference exists, and a causal test that reproduced it on the *good*
granule left AUC at 0.907 vs 0.904 — so it is not the cause), and temporal change (the three
granules span five days, ~30 m of motion against a 2560 m tile).

Still inconclusive: look direction. Crevasses are anisotropic scatterers, so a different
radar look direction is a plausible mechanism, but the GeoTIFFs carry no orbit or heading
metadata and the swath outlines are too close to square (aspect 1.04, 1.01) for a principal-
axis fit to be meaningful. Settling it needs the original HDF5 products.

**The cause is unknown.** The `025_048` label CSV ships so this can be reproduced, not so it
can be trained on.

## The product that was quarantined

Three older `LSAR_HH` granules were removed from the active set. On a matched footprint —
100% overlap, so location and resampling are controlled out — cross-tile orientation
concentration is **0.893 on `LSAR_HH` vs 0.439 on `frequencyA`**: a scene-wide oriented
streak artifact. Its `crevasse` labels turned out to be the artifact (42 of 42 relabelled
`none` on inspection). Those granules are also 2.5 × 5.0 m, non-square, which is what
prompted the square-pixel guard.

288 hand-labelled negatives from that product were dropped from training. They had been
buying about +0.020 AUC, which is now understood as the model learning to reject a defective
product's signature — not skill that ports anywhere.
