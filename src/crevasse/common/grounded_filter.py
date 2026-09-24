"""Grounded-ice (Bedmap3) pre-filter for the export/run_granule candidate stage, shared
by both sensors.

Crevasse fields of interest form in fast-flowing GROUNDED ice; floating shelf, sea ice
and rock are different phenomena (see `docs/nisar/METHOD.md`'s "Why context is not a
feature" -- ice type is deliberately kept OUT of the gate's own features, since it was
found to be a confound there, but filtering the CANDIDATE list before the gate ever runs
is a different, purely pre-scoring decision).

This warps the Bedmap3 grounded-ice mask onto the input granule's OWN CRS/transform/
shape via a `WarpedVRT`, then reads each candidate tile's grounded fraction straight
from that tile's own real pixel window. That is deliberately different from
`build_context_masks.py`'s approach (a separate, quantized tile-grid file keyed by
`row // step, col // step` from an assumed pixel-(0,0) origin): a granule's valid-data
grid is NOT always anchored at pixel (0, 0) (see `geotiff_crop.py`'s own note on exactly
this), so indexing a precomputed grid by floor-division from zero can silently misalign.
Reading straight from each candidate's actual `rasterio.windows.Window` sidesteps that
by construction, and needs only the Bedmap3 mask itself -- no ITS_LIVE velocity raster,
no separate continent-wide grid-building step.

Bedmap3 is available from NERC BAS; this repo does not ship it or a precomputed context
grid (removed along with the rest of `data/` for being too heavy to distribute -- see
the top-level README's "Building the training data from raw input swaths"). Point
`--bedmap-mask` at your own copy.
"""
from contextlib import contextmanager

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

GROUNDED = 1  # Bedmap3 class 1. Nodata (-9999) is filled to 0 (ocean), same convention
              # as build_context_masks.py -- unmapped is treated as not-grounded.


@contextmanager
def grounded_vrt(bedmap_path, crs, transform, width, height):
    """A `WarpedVRT` of the Bedmap3 mask resampled onto (crs, transform, width, height)
    -- typically an open granule raster's own grid. Reads are lazy and windowed, so
    opening this does not warp the whole continent, only whatever windows are read
    through it afterward."""
    with rasterio.open(bedmap_path) as src, \
         WarpedVRT(src, crs=crs, transform=transform, width=width, height=height,
                   resampling=Resampling.nearest) as vrt:
        yield vrt


def grounded_fraction(vrt, window):
    """Fraction of `window` (a `rasterio.windows.Window` in the VRT's own pixel grid,
    i.e. the granule's grid) classified as Bedmap3 grounded ice."""
    cls = vrt.read(1, window=window, masked=True).filled(0)
    return float((cls == GROUNDED).mean())


def filter_grounded(vrt, positions, tile_wh, width, height, min_grounded):
    """Keep only `positions` (`[(row, col), ...]`, native pixels) whose own tile window
    has Bedmap3 grounded fraction >= `min_grounded`. `tile_wh(row, col)` -> `(w, h)`,
    the tile's real footprint at that position (lets the caller's own edge-clipping
    convention -- `min(step, width - col)` etc. -- decide the window size)."""
    kept = []
    for row, col in positions:
        w, h = tile_wh(row, col)
        win = Window(col, row, w, h)
        if grounded_fraction(vrt, win) >= min_grounded:
            kept.append((row, col))
    return kept
