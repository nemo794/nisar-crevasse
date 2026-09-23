"""Shared helpers for `export_geotiff.py`'s `--crop-to-scanned` and `--edge-margin`
options, used by both sensors.

`--crop-to-scanned`: a granule's tile grid is almost always much bigger than the
positions any one run actually visits (especially under `--max-tiles`), so writing
against the full source raster's extent wastes space and -- worse -- leaves the
untouched remainder silently reading back as `0` (GDAL's default fill for a block a
driver never wrote), which looks like a real low score rather than "never scanned".
Cropping to the bounding box of the positions actually processed this run, and
explicitly writing NaN for every step-aligned grid cell inside that box that ISN'T one
of those positions, fixes both at once: the output is only as big as the area that
matters, and every pixel in it is either a real score or an honest NaN.

`--edge-margin`: independent, non-overlapping tile inference has a real, visible
artifact -- confirmed 2026-09-23 on a real BIOMASS export, a faint but real grid of
dimmer U-Net probability running along the exact tile boundaries, because each tile is
scored with no context beyond its own edge (a convolution near a tile's border sees
less real signal than one near its center). Rescoring on an overlapping, finer-stride
grid and keeping only each tile's CENTER region removes it: every output pixel then
always comes from a tile placement where it was NOT near that tile's own edge, except
at the true boundary of the scanned area itself, where there's no neighboring tile to
cover the trimmed strip and the full edge is kept instead.
"""
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


def crop_window(positions, step, src_width, src_height):
    """positions: [(row, col), ...] in native pixels, all on the `step`-aligned grid
    (`row = grid_row * step`, same for col). Returns the tight `rasterio.windows.Window`
    covering all of them, clipped to the source raster's own bounds (an edge tile's
    footprint can run past `src_width`/`src_height`)."""
    rows = [r for r, c in positions]
    cols = [c for r, c in positions]
    row0, col0 = min(rows), min(cols)
    row1 = min(max(rows) + step, src_height)
    col1 = min(max(cols) + step, src_width)
    return Window(col0, row0, col1 - col0, row1 - row0)


def crop_transform(window, src_transform):
    """The transform for `window`'s sub-raster, keeping the source's pixel size/CRS."""
    return window_transform(window, src_transform)


def grid_gap_positions(positions, step, window):
    """Every `step`-aligned (row, col) -- same coordinate space as `positions` -- inside
    `window` that is NOT already in `positions`. These are grid cells the coarse prescan
    or `--max-tiles` cutoff skipped entirely: within the crop, they'd otherwise silently
    read back as GDAL's default-filled `0` instead of an honest NaN.

    `window` is `crop_window(positions, ...)`'s output, so `window.row_off`/`col_off`
    already equal some real candidate's row/col -- a legitimate point on the tile grid.
    The grid is NOT necessarily anchored at pixel (0, 0) (a swath's valid-data bounds can
    start at an arbitrary offset), so walking forward from `window.row_off`/`col_off`
    themselves -- not from the nearest multiple of `step` from absolute zero, which can
    land strictly outside the true grid -- is what keeps every generated cell on it.
    """
    have = set(positions)
    out = []
    r = window.row_off
    while r < window.row_off + window.height:
        c = window.col_off
        while c < window.col_off + window.width:
            if (r, c) not in have:
                out.append((r, c))
            c += step
        r += step
    return out


def overlap_positions(row_min, row_max, col_min, col_max, stride):
    """(row, col) anchor positions on a `stride`-spaced grid covering
    `[row_min, row_max) x [col_min, col_max)` -- the finer, overlapping placement grid
    `--edge-margin` reads tiles from (`stride = tile_size - 2 * margin`), replacing the
    coarser non-overlapping `tile_size`-spaced grid used when `--edge-margin` is 0."""
    return [(r, c) for r in range(row_min, row_max, stride)
            for c in range(col_min, col_max, stride)]


def center_crop(row, col, tile_h, tile_w, margin, row_min, col_min):
    """What to keep of a `(tile_h, tile_w)` tile's output when writing it back at
    `--edge-margin margin`: `(top, left, bottom, right)` slice bounds into the tile
    array, and `(out_row, out_col)` -- the position in the OUTPUT raster the kept slice
    starts at. Trims `margin` off every edge that has a neighboring tile to cover the
    trimmed strip instead; an edge that's the true boundary of the scanned area (the
    first tile in its row/col, via `row_min`/`col_min`, or a tile whose read was already
    clipped short of a full `tile_size` by the raster's own edge) is left untrimmed, so
    the full scanned area still ends up covered with no permanent gap."""
    top = 0 if row == row_min else margin
    left = 0 if col == col_min else margin
    # a short read (tile_h/tile_w < a full tile) already means "this is the raster's
    # true edge in that direction" -- nothing follows it, so don't trim the far side.
    bottom = tile_h if tile_h < 2 * margin + 1 else tile_h - margin
    right = tile_w if tile_w < 2 * margin + 1 else tile_w - margin
    bottom, right = max(bottom, top + 1), max(right, left + 1)
    return (top, left, bottom, right), (row + top, col + left)
