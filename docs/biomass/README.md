# BIOMASS docs index

Carried over unchanged (content-wise) from the three pre-merge repos this sensor's code
came from — only relocated and, where two files would have collided on name, renamed
with a `GATE_`/nothing prefix distinguishing the gate repo's docs from the U-Net/pipeline
ones.

| file | from | about |
|---|---|---|
| [`GATE_CONTRIBUTING.md`](GATE_CONTRIBUTING.md) | `biomass-crevasse-gate` (was `docs/CONTRIBUTING.md`) | the gate's invariants (RF stays torch-free), controls, where things go |
| [`GATE_DATA.md`](GATE_DATA.md) | `biomass-crevasse-gate` | label provenance, granule/tile conventions |
| [`GATE_METHOD.md`](GATE_METHOD.md) | `biomass-crevasse-gate` | RF + CNN gate feature engineering and training |
| [`GATE_LIMITATIONS.md`](GATE_LIMITATIONS.md) | `biomass-crevasse-gate` | **read before quoting any gate number** |
| [`GATE_RESULTS.md`](GATE_RESULTS.md) | `biomass-crevasse-gate` | measured gate performance, RF vs CNN |
| [`UNET_PIPELINE.md`](UNET_PIPELINE.md) | `biomass-crevasse-unet` (was `docs/PIPELINE.md`) | shard building (incl. Frangi soft labels), training, the aux-noise-branch investigation, the current default checkpoint's rationale |
| [`model_comparison_2026-09-21/`](model_comparison_2026-09-21/) | `biomass-crevasse-unet` | figures from the noise-weight sweep comparison |
| [`PIPELINE_CONTRIBUTING.md`](PIPELINE_CONTRIBUTING.md) | `biomass-crevasse-pipeline` (was `docs/CONTRIBUTING.md`) | the gate+U-Net coordination layer's controls (P1-P4) |

**`data/` is not shipped in this repo** — one sensor's U-Net shards alone were 24 GB, too
heavy to distribute. See the top-level [`README.md`](../../README.md)'s "Building the
training data from raw input swaths" for the exact pipeline (`bio_tile_features.py` →
`bio_train_gate.py` → `bio_build_shard_frangi.py` → `bio_train_unet_supervised.py`) and
the external input it still needs (an AlphaEarth mosaic -- the Frangi/edge-detector
pseudo-label generator itself is part of this repo now, at
`src/crevasse/biomass/{multiscale_labels,edge_crevasse_v2,edge_crevasse_ppb,ppb_filter,
speckle_filters,crevasse_filters}.py`). `GATE_DATA.md` below still documents the
granule/label provenance and format in full, just not as a "what ships" list anymore.

The CNN gate's separate torch-required environment is `../../environment-cnn.yml` at the
repo root, not merged into the top-level `environment.yml` — see that file's own header.
