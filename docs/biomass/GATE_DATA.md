# Data

Format/provenance reference for the gate's data. **Nothing under `data/` ships in this
merged repo** — only `models/biomass/gate/{bio_gate_t512.joblib,
bio_cnn_gate_sarvel.{safetensors,json}}` do (the default inference models, ~15 MB
together). Everything below, including the files this document used to describe as
"shipped" in the pre-merge repo, must be rebuilt from raw input swaths — see the
top-level `README.md`'s "Building the training data from raw input swaths".

## Format reference (not shipped — rebuild these)

| file | size | what it is |
|---|---:|---|
| `bio_context_grid_2560m.npz` (`context_grid_2560m.npz`) | 7.2 MB | continent-wide ITS_LIVE speed + bedmap3 class + grounded fraction on a 2560 m grid |
| `bio_tile_features_t512.npz` | 2.7 MB | 4767 tiles x 38 features, plus labels, coordinates, granule, footprint, date |
| `bio_cnn_oof_c128.npz` | 261 KB | buffered held-out scores for all five arms over 4244 labelled tiles |
| `bio_cnn_prob_all.npz` | 39 KB | `p_full` and `p_oof` over all 4767 tiles |

The features npz is the load-bearing one: it makes the RF fully reproducible with no
granules present once you've built it — `run_rf_training.sh` reads nothing else.

## Bigger inputs, also not shipped

| what | size | rebuild |
|---|---:|---|
| `bio_chipcache_t512_c128.npz` | 558 MB | `python src/crevasse/biomass/bio_tile_cache.py` (labelled tiles only) |
| `bio_chipcache_t512_c128_all.npz` | 626 MB | `python src/crevasse/biomass/bio_tile_cache.py --all` (all 4767) |
| the granules themselves | 53 GB | 10 directories of 4 single-pol intensity GeoTIFFs — the raw input swaths |
| AlphaEarth exports + `bedmap3_mask.tif` | 7.1 GB | **labels only** — see below |
| `bio_*.png` | — | figures are outputs, not inputs |

**`data/aois/` is needed for labels only, and nothing else.** No feature in the 38-vector and no
channel in the CNN chips reads it; its whole contribution is `aoi_pos` and `aoi_inbox`. So
`bio_tile_features.py` guards it behind `os.path.exists` and, when it is absent, writes a table
with `aoi_pos` NaN and `aoi_inbox` False — **scoreable, and refused by the trainer.** The
grounded prescan that used to open `bedmap3_mask.tif` now reads the shipped
`data/context_grid_2560m.npz` for the same reason. That substitution is exact in shape (one
cell per tile) and approximate in value: `grounded_frac` moves by up to 0.047 on a tile
straddling the grounding line, which admits or rejects a handful of boundary tiles differently.
[CONTRIBUTING.md](CONTRIBUTING.md) control 9 pins the size of that difference and asserts the 38
feature columns themselves are bit-identical.

**Consequence, stated plainly:** without the chip caches, no CNN training and no CNN scoring
runs from a clean checkout. The CNN's *recorded* numbers stay checkable, because
`bio_cnn_oof_c128.npz` holds its held-out scores and the fold split is recomputable in pure
numpy. See [CONTRIBUTING.md](CONTRIBUTING.md) controls 4-5.

## The granules

Six of the ten on disk carry labels and are what everything is fitted on:

```
BIO_S1_..._20251126T135534_..._M01_..._T038_F245_02_DV6GT7
BIO_S2_..._20251217T135546_..._M01_..._T038_F245_01_DJT7YH
BIO_S3_..._20260107T135559_..._M01_..._T038_F245_01_DKW0CE
BIO_S1_..._20260215T140018_..._M02_..._T038_F245_01_DMWB5O
BIO_S2_..._20260308T140031_..._M02_..._T038_F245_01_DNZESH
BIO_S2_..._20260519T140512_..._M03_..._T038_F245_02_DRP7P7
```

Three footprints (M01 / M02 / M03), three stacks (S1 / S2 / S3), six dates spanning
2025-11-26 to 2026-05-19. Tile counts by footprint: M01 2487, M02 1413, M03 344.

Each granule directory holds four GeoTIFFs matching `*_{HH,HV,VH,VV}_intensity.tif`. The
feature builder requires all four and admits a directory only if it finds at least four `.tif`
files.

### The grid property everything else rests on

Every granule is **EPSG:3031 at 5.0 m with bounds on exact 2560 m (= 512 px) multiples**. So
all six tile grids coincide with each other, with the AlphaEarth label grid, and with the
mosaic grid. Placement is therefore **integer index arithmetic — nothing is reprojected or
interpolated anywhere in this repo.**

That is asserted in code, and the assert is load-bearing:

```python
assert s.crs.to_epsg() == 3031 and s.transform.a == 5.0
assert abs(v / TILE_M - round(v / TILE_M)) < 1e-6      # bounds on the 2560 m grid
assert abs(dr - round(dr)) < 1e-4 and abs(dc - round(dc)) < 1e-4   # AlphaEarth offset
```

If a new granule trips one of these, **do not relax the assert** — resample the granule onto
the grid instead. The NISAR sibling repo has a documented case where a tile-grid origin
mismatch made exact `(row, col)` label joins match zero rows, and it took a while to find.

### Repeat passes

4767 tile slots collapse onto **1806 distinct ground tiles at 2.64 passes each**, because S1
and S2 are exact repeat passes with identical `dr/dc` at M02 and M04. This is why folds are cut
in map coordinates and never per granule: a per-granule split puts the same ground in train and
test through a different acquisition. See [LIMITATIONS.md](LIMITATIONS.md) §6.

## Labels

AlphaEarth crevasse mosaic (`data/aois/new/thwaites_mosaic_new.vrt`), recall ~0.42 at
precision ~0.94. A tile becomes:

- **positive** if its AlphaEarth positive fraction >= 0.5
- **negative** if <= 0.05
- **dropped** if strictly between (ambiguous — 523 slots), or outside the export box (byte 0
  there means "never evaluated", not "no crevasse")

4767 slots → **4244 labelled, 1948 positive** (base rate 0.459).

The Earth Engine export recipe that produced the mosaic has not been recovered, so the label
set cannot currently be regenerated from scratch. That is an open gap, not a design choice.

## Context grid

`data/context_grid_2560m.npz` is one continent-wide grid at 2560 m — **exactly one cell per
512 px tile at 5 m**, so reading it is an index lookup, not a resample. Fields:

- `vel_mean` — ITS_LIVE annual mosaic ice speed, m/yr
- `bedmap_class` — bedmap3 mask class (1 = grounded)
- `grounded_frac` — fraction of the cell that is grounded ice

The `sar+vel` CNN arm needs `vel_mean` **at inference time**, so this file is required for CNN
scoring even though it is not SAR. It is carried as a **deliberate confound** and any run using
it must be reported beside the velocity-only control on the same folds — which is why
`oof_C_vel` exists (0.518 buffered; see [LIMITATIONS.md](LIMITATIONS.md) §5).

Note also that this single grid replaces per-granule warping, and building it exposed a real
1.28-vs-2.56 km cell-size bug on 2.5 m granules elsewhere in the project.

## Chips

`bio_tile_cache.py` writes `(n, 4, chip, chip)` float16 **raw dB**, NaN where the SAR is
nodata. Two properties that are not incidental:

- **No per-tile and no per-channel stretch.** A per-tile p2/p98 stretch is the documented cause
  of amplitude coming out *anti*-correlated with the label on NISAR, because it deletes the
  absolute level the label depends on. A per-channel stretch would additionally destroy the
  cross-to-co offset, which is the one thing this sensor actually carries. Normalisation is a
  single shared mean/sd applied at training time from the fold's own training tiles only.
- **128 px is not lossy here.** 512 px at 5 m down to 128 gives 20 m cells, and BIOMASS's 5 m
  grid is oversampled with a ~3:1 PSF whose long axis puts effective resolution near 25 m. So
  20 m discards oversampling, not signal. Finer buys speckle; coarser than 40 m starts erasing
  the across-axis structure the model reads.

## Adding a granule

Run the acceptance test **before** labelling, not after:

```bash
python src/crevasse/biomass/bio_check_granule.py <granule_dir>
```

The reason is specific: on the NISAR sibling repo a granule passed every cheap screen, was
labelled, was trained on, and then scored 0.627 where a known-good granule scored 0.904 on
identical tiles. The label cycle was wasted. Screens can convict a granule; they cannot clear
one.
