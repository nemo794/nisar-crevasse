"""One continent-wide context grid at the gate's tile resolution, built once.

build_context_masks.py warps Bedmap3 and ITS_LIVE onto each granule's own tile grid,
which re-reads a 7.5 GB velocity source per granule and makes cell size depend on the
granule's native pixel spacing (on a 2.5 m granule a cell is 1.28 km while a resampled
tile spans 2.56 km, so the lookup returns the tile's leading quadrant). Both rasters are
static and already cover all of Antarctica in EPSG:3031, so neither is necessary.

This builds a single EPSG:3031 grid at TILE_M ground spacing covering the Bedmap3 extent.
Any tile anywhere looks itself up by projected coordinate, cell size is uniform by
construction, and the result is small enough to ship to every worker.

    conda run -n nisar-gate python src/build_context_grid.py

Writes data/context_grid_2560m.npz: transform, grounded_frac, bedmap_class, vel_mean.
"""
import argparse

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT

import crevasse.nisar.gate_common as E
from crevasse.nisar.build_context_masks import BEDMAP, BEDMAP_CLASSES, GROUNDED, VEL

TILE_M = 2560.0        # 512 px * 5 m, the gate's tile footprint
SUB = 4                # sub-cell sampling, matching build_context_masks.py


def warp(path, crs, transform, nrow, ncol, resampling):
    with rasterio.open(path) as src:
        with WarpedVRT(src, crs=crs, transform=transform, width=ncol, height=nrow,
                       resampling=resampling) as vrt:
            return vrt.read(1, masked=True)


def lookup(grid, xs, ys):
    """Row/col into the grid for projected coordinate arrays. Out-of-extent -> -1."""
    t = grid["transform"]
    inv = ~Affine(*t) if not isinstance(t, Affine) else ~t
    cols, rows = inv * (np.asarray(xs), np.asarray(ys))
    rows, cols = np.floor(rows).astype(int), np.floor(cols).astype(int)
    h, w = grid["grounded_frac"].shape
    bad = (rows < 0) | (rows >= h) | (cols < 0) | (cols >= w)
    rows[bad], cols[bad] = -1, -1
    return rows, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-m", type=float, default=TILE_M)
    ap.add_argument("--out", default=str(E.ROOT / "data" / "context_grid_2560m.npz"))
    args = ap.parse_args()

    with rasterio.open(BEDMAP) as b:
        crs, bnd = b.crs, b.bounds
    ncol = int(np.ceil((bnd.right - bnd.left) / args.tile_m))
    nrow = int(np.ceil((bnd.top - bnd.bottom) / args.tile_m))
    transform = Affine(args.tile_m, 0, bnd.left, 0, -args.tile_m, bnd.top)
    print(f"grid {nrow}x{ncol} @ {args.tile_m:g} m, {crs}, "
          f"{nrow * ncol / 1e6:.2f}M cells")

    fine = Affine(args.tile_m / SUB, 0, bnd.left, 0, -args.tile_m / SUB, bnd.top)
    print(f"warping Bedmap3 at {args.tile_m / SUB:g} m ...")
    cls = warp(BEDMAP, crs, fine, nrow * SUB, ncol * SUB, Resampling.nearest).filled(0)
    frac = np.stack([(cls == k).reshape(nrow, SUB, ncol, SUB).mean(axis=(1, 3))
                     for k in BEDMAP_CLASSES])
    grounded_frac = frac[GROUNDED].astype(np.float32)
    bedmap_class = frac.argmax(0).astype(np.int8)
    del cls, frac

    print(f"warping ITS_LIVE at {args.tile_m / SUB:g} m (reads a 7.5 GB source) ...")
    v = warp(VEL, crs, fine, nrow * SUB, ncol * SUB,
             Resampling.bilinear).astype(np.float32).filled(np.nan)
    v[v <= 0] = np.nan
    with np.errstate(all="ignore"):
        vel_mean = np.nanmean(v.reshape(nrow, SUB, ncol, SUB), axis=(1, 3)).astype(np.float32)
    del v

    np.savez_compressed(args.out, transform=np.array(transform).reshape(3, 3)[:2].ravel(),
                        grounded_frac=grounded_frac, bedmap_class=bedmap_class,
                        vel_mean=vel_mean, tile_m=args.tile_m, crs=str(crs))
    nb = grounded_frac.nbytes + bedmap_class.nbytes + vel_mean.nbytes
    print(f"\nwrote {args.out}  ({nb / 1e6:.1f} MB in memory)")
    print(f"  grounded (majority) : {int((bedmap_class == GROUNDED).sum()):,} cells")
    print(f"  velocity mapped     : {int(np.isfinite(vel_mean).sum()):,} cells")
    for k, name in BEDMAP_CLASSES.items():
        print(f"  majority {name:16s}: {int((bedmap_class == k).sum()):,}")


if __name__ == "__main__":
    main()
