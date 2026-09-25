"""Memory-bounded COG writing: stream windowed writes to a temporary tiled GeoTIFF,
then convert to a Cloud-Optimized GeoTIFF with a single `gdal_translate` pass.

Why not write the COG directly. GDAL's COG driver is *create-copy oriented*: the final
COG byte layout (internal tiles followed by overviews, in a fixed order) can only be
produced in one pass at close time. When you open an output with `driver="COG"` and then
issue per-tile *windowed* `write()` calls -- exactly what `export_geotiff.py` does, one
tile at a time down the scan order -- GDAL cannot lay those blocks down incrementally, so
it retains every written block in an intermediate dataset until the file is closed. On a
full-swath granule that intermediate grows with the scanned area and OOM-kills the worker
(observed 2026-09-25: SIGKILL / exit 137 at 10176/13338 candidates on a 16 GB DPS worker,
memory climbing monotonically with candidates processed). An earlier note in the export
docstrings claimed the COG driver "supports the same incremental windowed write() calls";
that is true only in the sense that the calls *succeed* -- they are buffered, not flushed,
so memory is not actually bounded.

The fix -- write to a temporary *plain tiled GeoTIFF* first. A tiled GTiff supports
genuine random-access windowed writes that flush to disk block-by-block, so peak memory
stays flat regardless of granule size (only `--batch-size` tiles are ever held at once,
as the export always intended). A single `gdal_translate` pass then produces the COG with
identical creation options, plus `-stats` for the embedded per-band statistics that the
export used to compute via `dst.stats(approx=False)`. This mirrors `rift/cogutil.py`,
which writes every NISAR/BIOMASS amplitude COG the same way for the same reason.
"""
import subprocess
from pathlib import Path

import numpy as np
import rasterio

# gdal_translate COG creation options -- identical to what the export used to pass to the
# COG driver directly (BLOCKSIZE=512, DEFLATE/LEVEL=1/PREDICTOR=3, AVERAGE overviews,
# all-CPU threads, BigTIFF). PREDICTOR=3 is the floating-point predictor (both outputs are
# float32); OVERVIEW_RESAMPLING=AVERAGE suits the continuous probability rasters.
_COG_CREATION_OPTIONS = [
    "-of", "COG",
    "-co", "BLOCKSIZE=512",
    "-co", "COMPRESS=DEFLATE",
    "-co", "LEVEL=1",
    "-co", "PREDICTOR=3",
    "-co", "OVERVIEW_RESAMPLING=AVERAGE",
    "-co", "NUM_THREADS=ALL_CPUS",
    "-co", "BIGTIFF=YES",
]


def open_temp_tiff(final_path, profile, step):
    """Open a temporary tiled GeoTIFF for memory-bounded windowed writes.

    `final_path` is the FINAL COG path the caller wants; the temp file is created
    alongside it (``<stem>.temp.tif``) and returned so the caller can hand it to
    :func:`finalize_cog` once all windows are written. Returns ``(dataset, temp_path)``.

    The profile is forced to a single float32 band with NaN nodata and 512px internal
    tiling, matching what the export writes and what the final COG will use -- so the
    ``gdal_translate`` pass is a pure container-format conversion with no reblocking of
    the pixel data.
    """
    prof = profile.copy()
    blocksize = min(step, 512)
    prof.update(driver="GTiff", count=1, dtype="float32", nodata=np.nan,
                tiled=True, blockxsize=blocksize, blockysize=blocksize,
                compress="DEFLATE", predictor=3, bigtiff="YES")
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_suffix(".temp.tif")
    return rasterio.open(temp_path, "w", **prof), temp_path


def finalize_cog(temp_path, final_path, cleanup=True):
    """Convert the temporary tiled GeoTIFF at `temp_path` to a COG at `final_path` via
    ``gdal_translate`` (with ``-stats`` for embedded per-band statistics), then remove the
    temp file. Raises ``RuntimeError`` if ``gdal_translate`` fails."""
    temp_path = Path(temp_path)
    final_path = Path(final_path)
    cmd = ["gdal_translate", "-stats", str(temp_path), str(final_path),
           *_COG_CREATION_OPTIONS]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdal_translate failed: {result.stderr}")
    if cleanup and temp_path.exists():
        temp_path.unlink()
    return final_path
