# Data

## What ships in this merged repo, and what doesn't

**Only the default inference model ships**: `models/nisar/gate/gate_5m_freqA_2gran.joblib`
(12 MB). Nothing under `data/` ships — see the top-level `README.md`'s "Building the
training data from raw input swaths" for how to regenerate the files this document
describes. What follows is the format/provenance reference for those files, not a
manifest of what's in this repo:

| file | size | needed for |
|---|---|---|
| `tile_labels_ongrid_5m.csv` | 15 KB | retraining (hand labels, 025_091) |
| `tile_labels_aoi_025_019_speedmatched.csv` | 100 KB | retraining |
| `tile_labels_aoi_025_091_speedmatched.csv` | 76 KB | retraining |
| `tile_labels_aoi_025_048_speedmatched.csv` | 64 KB | **reproducing the 025_048 failure only** — not training |
| `context_grid_2560m.npz` | 7 MB | rebuilding labels; continent-wide ice type + speed on the 2560 m tile grid |
| `_context_masks_025_*_t512.npz` | 145 KB each | `--grounded-only` scoring of those three granules |

Total ~19 MB were they to ship, which is why this list is small enough that "not shipped"
is a distribution-policy choice here, not a hard technical necessity the way it is for
the granules and shards below.

## What does not ship: the granules

The three NISAR products are **26 GB together** and are not in this repo.

| granule | date | size | filename |
|---|---|---|---|
| 025_019 | 2026-07-09 | 10.5 GB | `NISAR_L2_PR_GSLC_025_019_A_140_4005_SHSH_A_20260709T131853_20260709T131929_P05023_N_F_J_001_frequencyA_HH_amplitude.tif` |
| 025_091 | 2026-07-14 | 5.1 GB | `NISAR_L2_PR_GSLC_025_091_A_140_4005_SHSH_A_20260714T131053_20260714T131110_P05023_F_P_J_001_frequencyA_HH_amplitude.tif` |
| 025_048 | 2026-07-11 | 10.7 GB | `NISAR_L2_PR_GSLC_025_048_A_140_4005_SHSH_A_20260711T133531_20260711T133608_P05023_N_F_J_001_frequencyA_HH_amplitude.tif` |

All three are ascending track 140, frame 4005, `frequencyA` HH amplitude, geocoded to
EPSG:3031 at 5.0 m square spacing. That they are the same frame is not incidental — see
[LIMITATIONS.md](LIMITATIONS.md).

The filename is parsed, so keep it. `GRANULE_RE` in `src/gate_common.py` pulls the
granule ID out of `GSLC_(\d{3}_\d{3})_`, and `granule_rasters()` refuses to proceed if two
files in the directory carry the same ID.

### Where to put them

Raw granules live separately from the rest of the gate's data now: `find_granule`/
`granule_rasters` in `gate_common.py` look under `<repo>/data/nisar/granules/` by default
(a sibling of `GATE_ROOT`, not inside it). Either drop them there, or leave them wherever
they are and point `GRANULES` at that tree (see `gate_common.py`).

`GATE_ROOT` (default `<repo>/data/nisar/gate`, see `gate_common.py`'s module docstring)
covers everything else this document describes — tile labels, context grids/masks — and
is independent of where the granules live. It does not need
`gate_5m_freqA_2gran.joblib`; the shipped model always loads from
`models/nisar/gate/gate_5m_freqA_2gran.joblib`, regardless of `GATE_ROOT`.

### Product requirements

- **`frequencyA`, not `LSAR_HH`.** Three older `LSAR_HH` granules were quarantined as
  defective; see LIMITATIONS.md.
- **Square pixels.** `tile_scale()` raises on non-square products. A 2.5 × 5.0 m raster
  would silently give 2.56 × 5.12 km tiles and hand every scale-dependent feature a
  preferred direction.
- **5.0 m spacing for anything you intend to interpret.** The bundle records
  `train_px_m_positive: [5.0]` and `map_crevasse_tiles.py` refuses other spacings unless
  you pass `--allow-unseen-spacing`. The override exists for experiments, not for results.

## Context rasters — needed only to rebuild labels

Not needed for inference, and never used as model features (METHOD.md explains why).

- **Bedmap3** ice-type / grounding mask — [BAS, doi:10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2](https://data.bas.ac.uk)
- **ITS_LIVE** Antarctic surface velocity mosaic — [NSIDC / JPL](https://its-live.jpl.nasa.gov)

`src/build_context_grid.py` reduces them to the shipped `context_grid_2560m.npz`;
`src/build_context_masks.py` cuts per-granule tile masks from them. One continent-wide grid
replaced per-granule warping, which is also what exposed a cell-size bug on 2.5 m granules.

## The label CSVs

Columns: `granule, row, col, label` (`crevasse` / `none`), where `row`/`col` are the
top-left corner of the tile in **native raster pixels**. Rows are deduplicated by
`(granule, row, col)` at load.

The `_speedmatched` CSVs are AlphaEarth-derived and supply the bulk of the labels. They are
speed-matched on purpose: an earlier spatial-buffer sampling made negatives systematically
slower ice, and velocity alone then scored AUC 0.866. `tile_labels_ongrid_5m.csv` holds
025_091's hand labels.

**Run the `(row, col)` control on any label set you build.** It is the only cheap thing
that reliably catches this class of confound.
