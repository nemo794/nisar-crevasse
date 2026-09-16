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

## Python API: N tiles in, N labels out

For a deployed pipeline that already holds tiles in memory and should not have to write
rasters to a temp directory to call its own model:

```python
from gate_predict import NisarGate

g = NisarGate()
p = g.predict_proba(amp)      # amp (N, 512, 512) -> (N,) float32 probabilities
keep = g.predict(amp)         # (N,) bool at the bundle's own threshold, 0.33301
```

The input contract, all of it load-bearing:

| | |
|---|---|
| units | **linear amplitude**, what the GSLC GeoTIFFs hold — not dB, not power. A negative value raises rather than scoring |
| spacing | **5.0 m**. `NisarGate(px_m=...)` is the *native* spacing of the granule the tiles were cut from, and a spacing the bundle never saw a crevasse label at is refused (`allow_unseen_spacing=True` overrides) |
| shape | `(N, 512, 512)` or `(512, 512)`. 512 is fixed: the FFT block reads four 384 px corner windows |
| nodata | **0 or NaN**. A tile under 50% valid is not scored |

**N in, N out, same order.** An unscoreable tile is `NaN` from `predict_proba` and `False`
from `predict` — never dropped, never reordered, so the result indexes against your own
tile list directly.

`predict()` defaults to the bundle's `threshold` and also exposes `threshold_mode`, which
is `recall>=0.95` — a recall target chosen on out-of-fold probabilities, not a tuned
optimum. Pass `thresh=` to override.

The API is additive: `src/map_crevasse_tiles.py` keeps its own path, so the pinned controls
still test the CLI rather than testing this code twice.

## Retrain

```bash
scripts/run_training.sh
```

Reproduces the shipped bundle from the shipped label CSVs. Requires the granules.
Expected: 3692 tiles, spatial-block OOF 0.932, threshold 0.333 — verified bit-identical.

Treat 0.932 as a **reproducibility target, not a performance claim**: at the default 8-tile
blocking a model given only the raw tile indices scores 0.934. Add `--block-tiles 16` for a
number that means something (0.942, against a 0.891 position floor). LIMITATIONS.md §8.

If you plan to modify anything, start with [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

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
| `src/gate_predict.py` | array-in Python API — `NisarGate`, N tiles in, N labels out |
| `src/render_score_map.py` | re-render a saved score file at a new threshold |
| `src/eval_holdout.py` | score a held-out label set |
| `src/check_granule.py` | granule acceptance tests — run these on anything new |
| `src/build_aoi_tile_labels.py` | build tile labels from an AlphaEarth-derived mosaic |
| `src/build_context_masks.py`, `src/build_context_grid.py` | ice-type and velocity context, used for **label construction and masking only** — never as model features (see docs/METHOD.md for why) |
| `src/plot_*.py` | diagnostic figures |
| `docs/METHOD.md` | tiling, features, classifier, CV, threshold |
| `docs/RESULTS.md` | the numbers, each with its control |
| `docs/LIMITATIONS.md` | what these numbers do not show. Read before quoting any of them. |
| `docs/DATA.md` | what to download and where to put it |
| `docs/CONTRIBUTING.md` | **how to add to this repo** — layout, recipes, the controls to re-run |
| `docs/lessons/` | what was tried and refuted along the way. More useful than the code if you are porting this. |

## Scope

This is stage 4 of a 5-stage pipeline (AlphaEarth AOI discovery → tiling → **this gate** →
Frangi ridge pseudo-labels → U-Net segmentation). Only the gate is shipped here; the
downstream stages are not settled enough to hand over.

The longer-term goal is porting this framework to ESA BIOMASS. NISAR is the pathfinder —
which is why `src/check_granule.py` and the controls baked into every script matter more
here than the specific AUC on a Thwaites frame.

## Export Classification

This software may be subject to U.S. export control laws. By accepting
this software, the user agrees to comply with all applicable U.S. export
laws and regulations. User has the responsibility to obtain export licenses,
or other export authority as may be required before exporting such
information to foreign countries or providing access to foreign persons.

## License

This software was developed at the Jet Propulsion Laboratory, California Institute of Technology.

This software is licensed under your choice of BSD-3-Clause or Apache-2.0
licenses. The exact terms of each license can be found in the accompanying
[LICENSE-BSD-3-Clause.txt] and [LICENSE-Apache-2.0.txt] files, respectively.

[LICENSE-BSD-3-Clause.txt]: LICENSE-BSD-3-Clause.txt
[LICENSE-Apache-2.0.txt]: LICENSE-Apache-2.0.txt

SPDX-License-Identifier: BSD-3-Clause OR Apache-2.0

## Disclaimer

This software is provided "as is" without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising from,
out of or in connection with the software or the use or other dealings in
the software.
