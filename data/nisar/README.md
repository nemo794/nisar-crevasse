# Put NISAR granules here

`*.tif` in this directory is gitignored — the three products used in this project are
~26 GB together. Full filenames, dates and download notes are in
[../../docs/DATA.md](../../docs/DATA.md).

Requirements, all enforced in code:

- **`frequencyA` HH amplitude GeoTIFF**, geocoded (EPSG:3031 for these).
- **Square pixels.** `tile_scale()` refuses non-square rasters.
- **5.0 m spacing** for anything you intend to interpret; other spacings need
  `--allow-unseen-spacing` and produce keep rates that are not comparable to
  `docs/RESULTS.md`.
- **Keep the original filename.** The granule ID is parsed out of it
  (`GSLC_<ddd>_<ddd>_`), and two files with the same ID in this directory is a hard error.

If the granules live elsewhere, don't symlink them one by one — point the whole tree:

```bash
export GATE_ROOT=/mnt/big/nisar-gate     # must contain data/nisar/ and data/*.joblib
```

Before labelling or trusting a granule you have not used before:

```bash
scripts/run_acceptance.sh <granule_id>
```

Read `docs/LIMITATIONS.md` §2-3 first. One granule in this project passed every cheap screen
and still scores 0.627 where a known-good granule scores 0.904 on identical tiles.
