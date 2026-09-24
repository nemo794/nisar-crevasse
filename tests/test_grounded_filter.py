"""Pure-logic tests for crevasse.common.grounded_filter, using a small synthetic
Bedmap3-style mask (rasterio MemoryFile) instead of the real, unshipped Bedmap3 raster --
see the module docstring for why this repo can't ship a real one."""
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.windows import Window

from crevasse.common.grounded_filter import grounded_vrt, grounded_fraction, filter_grounded

CRS = "EPSG:3031"
TRANSFORM = Affine(10, 0, 0, 0, -10, 0)  # 10 m/px, origin at (0, 0)


def _bedmap_bytes(cls_array):
    """A tiny in-memory Bedmap3-style mask GeoTIFF: cls_array's own shape/values, on
    TRANSFORM/CRS exactly -- so warping it onto the same grid is an identity op and the
    per-window fraction can be predicted exactly by hand."""
    h, w = cls_array.shape
    mem = MemoryFile()
    with mem.open(driver="GTiff", width=w, height=h, count=1, dtype="uint8",
                  crs=CRS, transform=TRANSFORM, nodata=0) as dst:
        dst.write(cls_array.astype("uint8"), 1)
    return mem


def test_grounded_fraction_all_grounded():
    cls = np.ones((40, 40), dtype=np.uint8)  # every pixel class 1 (grounded)
    with _bedmap_bytes(cls) as mem, mem.open() as src:
        with grounded_vrt(src.name, CRS, TRANSFORM, 40, 40) as vrt:
            assert grounded_fraction(vrt, Window(0, 0, 40, 40)) == 1.0


def test_grounded_fraction_half_grounded():
    cls = np.zeros((40, 40), dtype=np.uint8)
    cls[:, 20:] = 1  # right half grounded, left half ocean (class 0)
    with _bedmap_bytes(cls) as mem, mem.open() as src:
        with grounded_vrt(src.name, CRS, TRANSFORM, 40, 40) as vrt:
            assert grounded_fraction(vrt, Window(0, 0, 40, 40)) == 0.5
            assert grounded_fraction(vrt, Window(20, 0, 20, 40)) == 1.0
            assert grounded_fraction(vrt, Window(0, 0, 20, 40)) == 0.0


def test_filter_grounded_keeps_only_tiles_at_or_above_threshold():
    """Three 10x10 tiles side by side: fully grounded, fully ocean, and a 70/30 mix --
    a 0.70 cut must keep the first and third but not the second."""
    cls = np.zeros((10, 30), dtype=np.uint8)
    cls[:, 0:10] = 1                    # tile 0: 100% grounded
    cls[:, 10:20] = 0                   # tile 1: 0% grounded
    cls[:7, 20:30] = 1                  # tile 2: 70% grounded
    with _bedmap_bytes(cls) as mem, mem.open() as src:
        with grounded_vrt(src.name, CRS, TRANSFORM, 30, 10) as vrt:
            positions = [(0, 0), (0, 10), (0, 20)]
            kept = filter_grounded(vrt, positions, lambda row, col: (10, 10),
                                    30, 10, min_grounded=0.70)
    assert kept == [(0, 0), (0, 20)]


def test_filter_grounded_empty_when_nothing_clears_threshold():
    cls = np.zeros((10, 10), dtype=np.uint8)
    with _bedmap_bytes(cls) as mem, mem.open() as src:
        with grounded_vrt(src.name, CRS, TRANSFORM, 10, 10) as vrt:
            kept = filter_grounded(vrt, [(0, 0)], lambda row, col: (10, 10),
                                    10, 10, min_grounded=0.01)
    assert kept == []
