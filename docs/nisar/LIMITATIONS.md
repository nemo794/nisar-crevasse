# Limitations

Read this before quoting any number from this repo. Nothing here is softened. Each item is
something that was measured, not something feared.

## 1. The headline numbers describe one place

Spatial-block OOF 0.932 is honest for the ground it was measured on — but read item 8 for
what the blocking does and does not control. That ground is a single NISAR frame.

`025_019` and `025_091` are **both ascending track 140, frame 4005**, five days apart, with
82% bounding-box overlap. So the two leave-one-granule-out numbers (0.921, 0.943) look like
a transfer result and are not one — they are closer to a **repeat-pass consistency check**:
mostly the same ice, mostly the same look geometry, five days of motion (~30 m against a
2560 m tile).

**There is no honest cross-ground LOGO number in this project.** The one granule that would
have supplied it fails for unrelated reasons (item 2). Do not present the LOGO figures as
evidence the gate generalises to a new region, a new track, or a new sensor.

What would fix this: a labelled granule on different ground — ideally a different ice
stream — from a different track.

## 2. `025_048` fails, and nobody knows why

On 359 tiles of ground shared with `025_019`, with **identical labels, identical code,
identical folds**, spatial-block OAUC is **0.904 reading from 025_019 and 0.627 reading
from 025_048**. Only the source granule varies. The known-good control pair returns +0.004,
so the harness is sound.

Ruled out: label misplacement, partial coverage, pixel geometry, a scene-wide oriented
artifact, geolocation error (median shift 0 px), speckle correlation (a real 1.6× azimuth
difference exists, but reproducing it causally on the *good* granule left AUC at 0.907 vs
0.904), and temporal change.

Still open: look direction. Plausible — crevasses are anisotropic scatterers — but the
GeoTIFFs carry no orbit or heading metadata, and the swath outlines are too close to square
(aspect 1.04, 1.01) for a principal-axis fit to mean anything. Settling it needs the
original HDF5 products.

**Consequences.** A granule can look fine, pass every cheap screen, be labelled, be trained
on, and still be unusable. `025_048`'s labels ship so this is reproducible; it is excluded
from training.

## 3. The cheap screens can convict a granule but cannot clear one

`scripts/run_acceptance.sh` runs four checks. Only the fourth is a verdict.

The orientation screen catches scene-wide streak artifacts — that is real, it convicted the
`LSAR_HH` product. But a **per-region** orientation breakdown was tried as a way to catch
`025_048` and it does not discriminate:

| granule | scene-wide R | worst regional R |
|---|---|---|
| 025_048 (fails the A/B) | 0.490 | 0.984 |
| 025_019 (known good) | 0.531 | 0.971 |

Known-good ice produces regional highs indistinguishable from the failing granule's. The
subcommand still prints the regional table, labelled as context rather than a verdict. Do
not build a gate on it.

Note also that **"orientation R" has meant two different statistics in this project** —
cross-tile resultant vs within-tile gradient resultant (~0.46/0.86 vs ~0.013/0.026 for the
same two granules). Any output quoting R must say which.

The decisive test, the matched-ground A/B, needs labels for both granules *and* overlapping
ground. For a genuinely new region it cannot run at all — which is exactly the case where
you most want it.

## 4. Scoped to square 5.0 m `frequencyA` products

The bundle records `train_px_m_positive: [5.0]` and `map_crevasse_tiles.py` refuses other
spacings. This is not conservatism:

- Crevasse labels only ever existed at 5.0 m. At 2.5 m the gate had been taught what to
  reject and never what a crevasse looks like, so a 2.5 m keep rate measured the rejection,
  not the detection.
- An attempt to equalise looks across spacings was **built and failed** — the arms then
  carried different amounts of information rather than the same information at different
  smoothness. `LOOKS_POLICY = "native"` is the reverted state, not the untried one.
- Non-square products are refused outright by `tile_scale()`.

`--allow-unseen-spacing` exists for experiments. A number produced with it is not
comparable to anything in RESULTS.md.

## 5. The `LSAR_HH` product is quarantined as defective

On a matched footprint (100% overlap, so location and resampling are controlled out),
cross-tile orientation concentration is **0.893 on `LSAR_HH` vs 0.439 on `frequencyA`** —
a scene-wide oriented streak. Its `crevasse` labels turned out to be the artifact (42 of 42
relabelled `none`). Those products are also 2.5 × 5.0 m.

288 hand-labelled negatives from it were dropped. They had been buying about +0.020 AUC,
now understood as the model learning to reject one defective product's signature.

## 6. The threshold was never priced against downstream cost

0.333 is the highest threshold reaching recall ≥ 0.95 out-of-fold. **0.95 was chosen as a
plausible round number.** The actual trade — one missed crevasse field against one wasted
per-pixel segmentation — was never costed. If the downstream stage is cheaper or dearer
than assumed, this number should move.

Cheap to revisit: probabilities are stored, so a new threshold is a re-render
(`src/render_score_map.py`), not a re-score.

## 7. Raising the threshold makes results worse in a way maps do not show

Share of retained positives on grounded ice **falls** as the threshold rises — on 025_048
from a 82.4% base rate to 31.9% at P ≥ 0.80, an enrichment of ~2.6× *off* grounded ice.
Cleaner-looking maps, less grounded-ice crevassing. Plausibly shelf and sea-ice rifts,
which are real linear features; arguably a labelling-definition question. **Unresolved.**

`--grounded-only` is documented but deliberately not the default: making it default would
hide the effect rather than explain it. Every keep rate in RESULTS.md was measured without
it and is therefore an **upper bound**.

## 8. The shipped bundle's CV blocking is too narrow to be a performance claim

The bundle records spatial-block OOF 0.932 at `--block-tiles 8` (~20 km blocks). A model
given **only the raw tile indices** — no pixels — scores **0.934** on those same folds. The
crevasse field is much larger than 20 km, so a held-out block's label is still predictable
from where it is.

This is not a claim that the gate memorises position; the forest never sees `row`/`col`, and
the matched-ground A/B (item 2) varies the pixels with position held fixed and moves AUC by
0.277. It means **0.932 does not by itself demonstrate skill**, and should be quoted as the
reproducibility target for `scripts/run_training.sh` rather than as performance.

Widening the block fixes it, and the fix is stable: margin over position is +0.051 at 16×16
tiles (41 km), +0.044 at 32×32, +0.046 at 64×64. Quote **0.942 at `--block-tiles 16`**
against a position floor of 0.891. Full sweep in RESULTS.md.

The shipped bundle was not retrained at 16, because 0.932/8 is the value already recorded in
it, in `run_training.sh` and in every verification control — changing the default would have
made the reproduction checks disagree with the artifact. That is a real loose end: the
default in `train_gate_classifier.py:525` is still 8.

## 9. Context is a confound, not a feature

Ice type and surface velocity raise AUC and are still not used, because the `(row, col)`
control settles it: raw tile indices scored +0.031 against the context prior's +0.027. The
prior was memorising where crevasses are in this scene.

**Run the `(row, col)` control on any new label set**, and sweep the block width while you do
— item 8 is what happens when you don't. It has caught a real confound in this project, it
caught a bad CV setting too, and it is cheap. Recipe in CONTRIBUTING.md.

## 10. Known-invalid numbers not in this repo

An earlier recall-by-log-dynamic-range analysis (recall 99.6% → 75.2% across contrast
thirds) was invalidated: ~82% of the apparent misses were AlphaEarth mislabels, and
correcting them took recall to 94.4%. The contrast threshold it suggested was **not**
shipped and the figures are not here. Mentioned so the finding is not rediscovered as new.

## 11. What is not shipped at all

Everything under `data/` -- see the top-level `README.md`'s "Building the training data
from raw input swaths" and `docs/nisar/UNET_PIPELINE.md` for the U-Net stage.
