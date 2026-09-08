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

| measure | AUC | what it means |
|---|---|---|
| random-fold OOF | 0.951 | optimistic — neighbouring tiles leak across folds |
| **spatial-block OOF** | **0.932** | 8×8 tile (~20 km) blocks held out, 124 blocks. **Quote this one.** |
| LOGO 025_019 | 0.921 | train on 025_091, test on 025_019 |
| LOGO 025_091 | 0.943 | train on 025_019, test on 025_091 |

The two LOGO numbers look like a transfer result and are not — see LIMITATIONS.md. The gap
between random-fold and spatial-block (0.951 → 0.932) is the honest cost of spatial
autocorrelation, and it is small, which is the good news in this table.

### Controls that were run

- **`(row, col)` control.** Raw tile indices as the only features scored near chance on this
  label set. Geography alone does not separate these classes, so the AUC above is not
  memorised location. This control has caught a real confound before and is not decoration.
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

Reproduce with `python src/check_granule.py ab 025_048 025_019`. The known-good control pair
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
