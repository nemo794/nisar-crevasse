# NISAR crevasse gate

A binary classifier that decides whether a 2.56 km tile of NISAR L2 GSLC HH amplitude
contains crevasses. It exists to cut the search space before an expensive per-pixel
segmentation stage runs: on the granules tested it discards 63-75% of tiles while keeping
95% of crevassed ones.

Random forest, 270 texture features, no deep learning, no GPU. Trained on 3692 tiles from
two Thwaites-sector granules.

**Read [docs/LIMITATIONS.md](docs/LIMITATIONS.md) before quoting any number from this
repo.** The headline AUCs are honest for what they measure, and what they measure is one
place. That document says so precisely.

## Install

```bash
conda env create -f environment.yml
conda activate nisar-gate
```

Deliberately torch-free — scikit-learn only, CPU only.

## Run inference on a granule

Put a NISAR GSLC amplitude GeoTIFF in `data/nisar/` (see [docs/DATA.md](docs/DATA.md)), or
point `GATE_ROOT` at a tree that has one, then:

```bash
scripts/run_inference.sh 025_019
```

Writes a per-tile score file and a map PNG to `data/maps/`. Nothing else is needed — not
labels, not the context rasters, not the training data. The shipped bundle
`data/gate_5m_freqA_2gran.joblib` is all the model there is.

To try a different threshold, re-render rather than re-score — the probabilities do not
depend on the threshold, so this takes seconds instead of minutes:

```bash
python src/render_score_map.py data/maps/gate_sar_025_019.npz --gate-thresh 0.65
```

## Retrain

```bash
scripts/run_training.sh
```

Reproduces the shipped bundle from the shipped label CSVs. Requires the granules.
Expected: 3692 tiles, spatial-block OOF 0.932, threshold 0.333.

## Before you use a new granule

```bash
scripts/run_acceptance.sh <new_granule>
```

This is not optional boilerplate. One granule in this project's history
(`025_048`) looked fine to the eye, passed every cheap screen, was labelled, was trained
on — and scores 0.627 where a known-good granule scores 0.904 on *identical tiles with
identical labels*. The cause is still unknown. The matched-ground A/B in
`src/check_granule.py` is what caught it, and it would have caught it before the labelling
effort rather than after.

## What is here

| path | what |
|---|---|
| `src/gate_common.py` | shared constants, granule lookup, the 14 hand features |
| `src/train_gate_classifier.py` | feature extraction, training, CV, threshold calibration |
| `src/map_crevasse_tiles.py` | inference over a whole granule + map rendering |
| `src/render_score_map.py` | re-render a saved score file at a new threshold |
| `src/eval_holdout.py` | score a held-out label set |
| `src/check_granule.py` | granule acceptance tests — run these on anything new |
| `src/build_aoi_tile_labels.py` | build tile labels from an AlphaEarth-derived mosaic |
| `src/build_context_masks.py`, `src/build_context_grid.py` | ice-type and velocity context, used for **label construction and masking only** — never as model features (see docs/METHOD.md for why) |
| `src/plot_*.py` | diagnostic figures |
| `docs/lessons/` | what was tried and refuted along the way. More useful than the code if you are porting this. |

## Scope

This is stage 4 of a 5-stage pipeline (AlphaEarth AOI discovery → tiling → **this gate** →
Frangi ridge pseudo-labels → U-Net segmentation). Only the gate is shipped here; the
downstream stages are not settled enough to hand over.

The longer-term goal is porting this framework to ESA BIOMASS. NISAR is the pathfinder —
which is why `src/check_granule.py` and the controls baked into every script matter more
here than the specific AUC on a Thwaites frame.
