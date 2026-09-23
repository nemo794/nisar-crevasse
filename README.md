# crevasse

Crevasse detection for two SAR sensors, NISAR and BIOMASS, merged into one repo on
2026-09-22 from six previously separate repos (`nisar-crevasse-{gate,unet,pipeline}`,
`biomass-crevasse-{gate,unet,pipeline}`). Each sensor's own gate (tile-level classifier)
and U-Net (per-pixel segmentation) code, training data, and docs are preserved, merged as
flat, self-contained directories rather than force-unified — the two sensors' physics,
preprocessing, and feature engineering genuinely differ; only what was actually
duplicated code got shared.

## Quickstart

```bash
pip install -e .     # once per environment -- registers the crevasse package
```

```python
from crevasse import Pipeline

pipe = Pipeline("nisar")
out = pipe.run(amp)                       # amp: (N, 512, 512) linear amplitude

pipe = Pipeline("biomass")
out = pipe.run(inten, x_m=x_m, y_m=y_m)   # inten: (N, 4, 512, 512) linear intensity

out.gate_prob     # (N,)             float32  per-tile P(crevasse) from the gate
out.gate_flag     # (N,)             bool     gate_prob >= threshold
out.unet_prob     # (N, 512, 512)    float32  per-pixel P(crevasse); NaN where gate_flag is False
out.unet_mask     # (N, 512, 512) | None  bool  only set if unet_thresh= was passed
```

`Pipeline(sensor, **kwargs)` is the one entry point for either sensor — see
`src/crevasse/common/pipeline.py`'s docstring for why (both sensors' coordinator files
share filenames by design; nesting them under distinct dotted package names,
`crevasse.nisar.*` vs `crevasse.biomass.*`, is what makes that safe). Everything else
about each sensor — the actual gate/U-Net array APIs, granule tiling, plotting, GeoTIFF
export — lives untouched in `src/crevasse/nisar/`/`src/crevasse/biomass/`, same shape as
it did in the six original repos, just importable as a real installed package instead of
via `sys.path` tricks.

## Running predictions on a real granule

The Quickstart's `Pipeline.run()` takes an in-memory tile stack; `export_geotiff.py`
(one per sensor) is what runs it end-to-end from a real granule on disk and writes two
GeoTIFFs, registered on the source granule's own grid (same CRS/transform/shape) so they
open lined up in QGIS or any GIS tool with no separate georeferencing step:

- **`gate_prob.tif`** — the gate's output: each tile's scalar `P(crevasse)`, broadcast
  flat over its whole footprint.
- **`unet_prob.tif`** — the crevasse segmentation: the U-Net's real per-pixel
  `P(crevasse)`, written only for tiles the gate flagged; `NaN` everywhere else.

Both are continuous probabilities, not binary masks (see each sensor's
`export_geotiff.py` docstring for why).

By default the output is the size of the *whole source raster*, even when `--max-tiles`
only actually scans a small cluster of it — GDAL fills every block this script never
touches with `0`, not `NaN` (a documented limitation, not a crash), which can look like a
real low score rather than "never scanned" when you view the whole extent. Pass
**`--crop-to-scanned`** to avoid that entirely: the output shrinks to the bounding box of
the positions this run actually processed (still on the source grid, so it lines up the
same way, just over a smaller extent), and every grid cell inside that box that was never
even a candidate also gets an explicit `NaN` write — so every pixel in the file is either
a real score or an honest `NaN`, nothing silently defaults to `0`.

Independent, non-overlapping tile inference has a separate, real artifact: each tile is
scored with no context beyond its own edge, which shows up as a faint but real grid of
dimmer U-Net probability running along the exact 512px tile boundaries (confirmed
2026-09-23 on a real BIOMASS export — continuous ridge texture on either side of a
tile boundary with a visibly dimmer seam right at the join). Pass **`--edge-margin N`**
to fix it: tiles are rescored on an overlapping, `(512 - 2*N)`-pixel-stride grid instead,
and only each tile's center region is kept (the outer `N`-pixel ring is discarded and
covered by the *next* tile's center instead, except at the true boundary of the scanned
area, where nothing follows it and it's kept). Costs roughly `(512/(512-2*N))^2` times
more U-Net forward passes — `--edge-margin 64` is about 1.8x. Not combinable with
`--crop-to-scanned` yet. Off by default.

`pip install -e .` also registers `crevasse-export-geotiff` as a console script —
the sensor is the first positional argument, everything after it is that sensor's own
flags, unchanged:

**NISAR** (raw input: one `frequencyA` HH amplitude GeoTIFF):

```bash
crevasse-export-geotiff nisar \
    --granule /path/to/NISAR_..._frequencyA_HH_amplitude.tif \
    --out-dir data/export_nisar --max-tiles 600 --crop-to-scanned  # quick check, cropped

crevasse-export-geotiff nisar \
    --granule /path/to/NISAR_..._frequencyA_HH_amplitude.tif \
    --out-dir data/export_nisar --edge-margin 64      # no tile-boundary seam, ~1.8x slower

crevasse-export-geotiff nisar \
    --granule /path/to/NISAR_..._frequencyA_HH_amplitude.tif \
    --out-dir data/export_nisar                       # the full swath -- slow, see below
```

**BIOMASS** (raw input: a granule *directory* holding
`*_{HH,HV,VH,VV}_intensity.tif`):

```bash
crevasse-export-geotiff biomass \
    --granule /path/to/BIO_S2_SCS__..._DJT7YH \
    --out-dir data/export_biomass --max-tiles 20 --crop-to-scanned  # quick check, cropped

crevasse-export-geotiff biomass \
    --granule /path/to/BIO_S2_SCS__..._DJT7YH \
    --out-dir data/export_biomass --edge-margin 64    # no tile-boundary seam, ~1.8x slower

crevasse-export-geotiff biomass \
    --granule /path/to/BIO_S2_SCS__..._DJT7YH \
    --out-dir data/export_biomass                     # the full granule -- slow, see below
```

Equivalent to calling `python src/crevasse/nisar/export_geotiff.py ...` /
`python src/crevasse/biomass/export_geotiff.py ...` directly — the console script is a
thin dispatcher (see `src/crevasse/cli.py`), not a different implementation.

Both stream tile-by-tile with windowed writes (never holding more than `--batch-size`
tiles in memory), so they run on a laptop regardless of granule size — the cost is
wall-clock time, not memory. **A full granule is slow** (the U-Net forward pass on CPU
dominates) — always try `--max-tiles` first, and prefer a GPU machine for a real run.
`--gate-thresh` overrides the gate's flagging cutoff; BIOMASS also takes `--gate-method
{rf,cnn}` (default `rf`, matching its own training pipeline). See
`docs/nisar/PIPELINE_CONTRIBUTING.md` / `docs/biomass/PIPELINE_CONTRIBUTING.md` for the
pinned real-data numbers these scripts reproduce.

The same `<sensor> --granule ...` dispatch pattern is registered for the other two
coordinator CLIs:

```bash
crevasse-run-granule  nisar   --granule <path> --check   # tile + score + print a summary
crevasse-run-granule  biomass --granule <dir>  --check
crevasse-plot-granule nisar   --granule <path> --check   # render a figure to eyeball a run
crevasse-plot-granule biomass --granule <dir>  --check
```

## Layout

```
pyproject.toml    src-layout, single top-level package: pip install -e . registers it,
                  including the crevasse-{run-granule,export-geotiff,plot-granule}
                  console scripts ([project.scripts])
src/
  crevasse/
    __init__.py   re-exports Pipeline (from crevasse.common.pipeline import Pipeline)
    cli.py        the console scripts' dispatcher: sensor is the first positional arg,
                  the rest is that sensor's own run_granule.py/export_geotiff.py/
                  plot_granule.py argv, unchanged
    common/       code shared or superset-compatible across both sensors:
                  ckpt_io.py, convert_pth_to_safetensors.py, unet_model.py,
                  tile_utils_v2.py, pipeline.py
    nisar/        flat merge of nisar-crevasse-{gate,unet,pipeline}/src -- gate_predict.py
                  (NisarGate), unet_predict.py (UNetSegmenter), pipeline_predict.py
                  (CrevassePipeline), run_granule.py, plot_granule.py, export_geotiff.py,
                  plus every training/feature/eval script behind them
    biomass/      flat merge of biomass-crevasse-{gate,unet,pipeline}/src -- bio_predict.py
                  (BiomassGate), bio_unet_predict.py (BiomassUNetSegmenter),
                  pipeline_predict.py (BiomassCrevassePipeline), run_granule.py,
                  plot_granule.py, export_geotiff.py, plus training/feature/eval scripts,
                  plus the Frangi pseudo-label generator (multiscale_labels.py,
                  edge_crevasse_v2.py, edge_crevasse_ppb.py, ppb_filter.py,
                  speckle_filters.py, crevasse_filters.py) -- brought in from the external
                  research repo it used to live in, so `bio_build_shard_frangi.py` has no
                  dependency left outside this package
models/
  nisar/        the DEFAULT shipped weights only -- not every past experiment checkpoint
    gate/       gate_5m_freqA_2gran.joblib
    unet/       unet_025_019_f421_meansoft_g3/unet_best.{safetensors,json}
  biomass/
    gate/       bio_gate_t512.joblib, bio_cnn_gate_sarvel.{safetensors,json}
    unet/       bio_unet_4ch_frangi_aux_nw0p02/unet_best.{safetensors,json}
docs/
  nisar/        every doc from nisar-crevasse-gate (CONTRIBUTING/DATA/METHOD/LIMITATIONS/
                RESULTS), nisar-crevasse-unet (UNET_PIPELINE.md), and
                nisar-crevasse-pipeline (PIPELINE_CONTRIBUTING.md) -- unchanged content,
                just relocated. This is also where "how to retrain from scratch" lives.
  biomass/      same set, biomass side (GATE_*.md, UNET_PIPELINE.md, model_comparison_*,
                PIPELINE_CONTRIBUTING.md)
scripts/
  nisar/        run_acceptance.sh, run_inference.sh, run_training.sh, run_check.sh,
                run_eval.sh, run_train.sh
  biomass/      run_rf_training.sh, run_rf_inference.sh, run_cnn_inference.sh,
                run_new_granule.sh, run_check.sh, run_eval.sh, run_train.sh
```

## `models/` ships weights; `data/` does not exist in this repo

`models/{nisar,biomass}/` holds only the **default** checkpoint each sensor's
`gate_predict.py`/`unet_predict.py`/`bio_predict.py`/`bio_unet_predict.py` loads with no
arguments (a few hundred MB total) — not the full history of past experiment runs.

Everything upstream of that — raw granules, tile labels, training shards, feature caches
— is **not shipped**: those directories ran from single-digit MB up to tens of GB (one
sensor's U-Net shards alone were 24 GB), too heavy to distribute in a repo. Every script
that reads a `data/...`-shaped default path (`gate_common.py`'s `ROOT`,
`bio_predict.py`'s `CTX_PATH`, etc.) still works exactly as documented — you supply that
directory yourself, in the same shape the original per-sensor docs describe, by running
the pipeline below on your own raw input swaths.

### Building the training data from raw input swaths

Both sensors follow the same shape, gate first then U-Net. NISAR's U-Net pseudo-labels
still depend on code outside this repo; BIOMASS's equivalent (the Frangi/edge-detector
generator) has been brought in (`src/crevasse/biomass/{multiscale_labels,
edge_crevasse_v2, edge_crevasse_ppb, ppb_filter, speckle_filters,
crevasse_filters}.py`) — flagged explicitly below, not glossed over, since it's a real
asymmetry between the two sensors' reproducibility:

**NISAR** (full detail: [`docs/nisar/DATA.md`](docs/nisar/DATA.md),
[`docs/nisar/METHOD.md`](docs/nisar/METHOD.md),
[`docs/nisar/UNET_PIPELINE.md`](docs/nisar/UNET_PIPELINE.md)):

1. **Raw input**: a NISAR L2 GSLC `frequencyA` HH amplitude GeoTIFF (~5-10 GB each,
   downloaded separately — see `docs/nisar/DATA.md` for exact products used). Square
   pixels, 5.0 m spacing, filename kept as-is (the granule ID is parsed from it).
2. **Gate tile labels**: `build_aoi_tile_labels.py` derives `crevasse`/`none` labels from
   an **AlphaEarth classification raster export for that granule** — an external input
   this repo does not generate, pixel-aligned to the granule.
3. **Train the gate**: `train_gate_classifier.py --labels-csv <csv>` → a `.joblib` bundle
   (same format as `models/nisar/gate/gate_5m_freqA_2gran.joblib`).
4. **U-Net pseudo-labels**: generated by an **external pseudo-label generator (a separate
   research repo, not part of this merge)**, which still writes a pickled `.pkl`. Convert
   it with `../biomass/scripts/convert_labels_pkl_to_npz.py` (kept out of this repo
   specifically because it's the one place still allowed to touch pickle — see that
   script's own docstring) into the `.npz` `build_shard.py` reads.
5. **Build the training shard**: `build_shard.py --results-npz <label npz> --raster
   <granule>` → a packed `.npz` shard (same format as what used to live under
   `data/nisar/unet/`).
6. **Train the U-Net**: `train_unet_supervised.py --shard <shard.npz>`.

**BIOMASS** (full detail: [`docs/biomass/GATE_DATA.md`](docs/biomass/GATE_DATA.md),
[`docs/biomass/GATE_METHOD.md`](docs/biomass/GATE_METHOD.md),
[`docs/biomass/UNET_PIPELINE.md`](docs/biomass/UNET_PIPELINE.md)):

1. **Raw input**: a BIOMASS granule *directory* (`*_{HH,HV,VH,VV}_intensity.tif`, one per
   polarization).
2. **Gate tile features + labels**: `bio_tile_features.py`/`bio_tile_cache.py` build the
   per-tile feature table from the raw granules, arbitrated against an **AlphaEarth
   classification mosaic** (same external-input caveat as NISAR).
3. **Train the gate**: `bio_train_gate.py` (RF) or `bio_cnn_gate.py` (CNN) → a `.joblib`
   or safetensors+json bundle.
4. **U-Net labels**: `bio_build_shard.py` (plain) or `bio_build_shard_frangi.py` (Frangi
   soft labels, the default checkpoint's training data) — the Frangi/edge detector this
   calls (`multiscale_labels.py` and friends) lives in this repo now, not an external one.
5. **Train the U-Net**: `bio_train_unet_supervised.py --shard <shard.npz>`.

Both sensors still need an AlphaEarth export as an external input for gate labeling
(neither generates it). NISAR's U-Net pseudo-labels additionally depend on a generator
that lives outside this merge (`convert_labels_pkl_to_npz.py`, at
`../biomass/scripts/`, converts its `.pkl` output); BIOMASS's equivalent no longer has
that gap. That's a real, documented asymmetry between the two sensors' "retrain from
scratch here" story, not an oversight: see each sensor's own `DATA.md`/`GATE_DATA.md`
for exactly what's still external.

## Environment

`environment.yml` is the union of all six original repos' environments (torch, rasterio,
scikit-learn, scikit-image, safetensors, albumentations) — this repo imports both
sensors' gate and U-Net code in the same process (`Pipeline`), so there is no torch-free
import path here even though `NisarGate`/`BiomassGate`'s RF paths are individually
torch-free.

## Tests

```bash
pip install -e .          # if not already done
python -m pytest tests/
```

Pure-logic unit tests on synthetic tiles (`Pipeline()` construction/error-handling,
scatter/gather bookkeeping, U-Net input-shape validation) — not a replacement for the
pinned real-data controls each sensor's own docs describe, which are what actually
validates crevasse-detection accuracy. See [`tests/README.md`](tests/README.md) for the
scope boundary.

## Per-sensor documentation

- [`docs/nisar/`](docs/nisar/) — NISAR gate + U-Net + pipeline docs, science and controls.
- [`docs/biomass/`](docs/biomass/) — BIOMASS gate + U-Net + pipeline docs, science and controls.

Read the relevant sensor's docs before trusting a number from this repo — the controls
discipline (pinned real-data checks, P1-P4-style equality claims) is per-sensor and
unchanged by this merge.

## Export Classification

Copyright 2026, by the California Institute of Technology. ALL RIGHTS
RESERVED. United States Government Sponsorship acknowledged. Any commercial
use must be negotiated with the Office of Technology Transfer at the
California Institute of Technology.

This software may be subject to U.S. export control laws. By accepting
this software, the user agrees to comply with all applicable U.S. export
laws and regulations. User has the responsibility to obtain export licenses,
or other export authority as may be required before exporting such
information to foreign countries or providing access to foreign persons.

If you have questions regarding this, please contact the JPL Software
Release Authority at x4-2458.

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
