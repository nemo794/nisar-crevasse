# NISAR docs index

Carried over unchanged (content-wise) from the three pre-merge repos this sensor's code
came from — only relocated and, where two files would have collided on name, renamed.

| file | from | about |
|---|---|---|
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | `nisar-crevasse-gate` | the gate's controls discipline, invariants, where things go |
| [`DATA.md`](DATA.md) | `nisar-crevasse-gate` | label provenance, granule/tile conventions |
| [`METHOD.md`](METHOD.md) | `nisar-crevasse-gate` | the RF gate's feature engineering and training method |
| [`LIMITATIONS.md`](LIMITATIONS.md) | `nisar-crevasse-gate` | **read before quoting any gate number** |
| [`RESULTS.md`](RESULTS.md) | `nisar-crevasse-gate` | measured gate performance |
| [`UNET_PIPELINE.md`](UNET_PIPELINE.md) | `nisar-crevasse-unet` (was `docs/PIPELINE.md`) | shard building, training, the U-Net's own controls |
| [`PIPELINE_CONTRIBUTING.md`](PIPELINE_CONTRIBUTING.md) | `nisar-crevasse-pipeline` (was `docs/CONTRIBUTING.md`) | the gate+U-Net coordination layer's controls (P1-P4) |

**`data/` is not shipped in this repo** — the granules and training shards ran into the
GBs and were too heavy to distribute. See the top-level [`README.md`](../../README.md)'s
"Building the training data from raw input swaths" for the exact pipeline
(`build_aoi_tile_labels.py` → `train_gate_classifier.py` → `build_shard.py` →
`train_unet_supervised.py`) and the external inputs it needs (an AlphaEarth export, and a
pseudo-label generator that lives outside this merge). `DATA.md` below still documents
the granule/label provenance and format in full, just not as a "what ships" list anymore.
