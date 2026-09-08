"""Warp Bedmap3 grounded-ice mask and ITS_LIVE surface speed onto the granule's
512-px tile grid, so every tile carries the physical context needed to decide
whether a crevasse label there is even plausible.

Crevasses of interest form in fast-flowing *grounded* ice; sea ice and slow
interior ice are different phenomena. Restricting to grounded & fast is what
separates "AlphaEarth missed it" from "there is nothing there to miss".

    python src/build_context_masks.py --granule 025_019

Writes data/_context_masks_<granule>_t<tile>.npz with, on the tile grid:
    grounded_frac : fraction of the tile that is Bedmap3 class 1 (grounded ice)
    bedmap_class  : majority Bedmap3 class, 0 ocean / 1 grounded / 2 transient
                    shelf / 3 floating shelf / 4 rock
    vel_mean      : mean ITS_LIVE surface speed, m/yr (NaN where unmapped)
    vel_max       : max ITS_LIVE surface speed, m/yr
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT

import gate_common as E
from gate_common import find_granule

BEDMAP = E.ROOT / "data" / "aois" / "bedmap3_mask.tif"
VEL = E.ROOT / "data" / "aois" / "ITS_LIVE_velocity_120m_RGI04A_0000_V02.1_v.tif"
GROUNDED = 1
# Bedmap3 nodata (-9999) is filled to 0, which is open ocean / sea ice.
BEDMAP_CLASSES = {0: "ocean", 1: "grounded", 2: "transient_shelf",
                  3: "floating_shelf", 4: "rock"}


def tile_grid_spec(granule_path, tile):
    with rasterio.open(granule_path) as g:
        t, crs = g.transform, g.crs
        nrow, ncol = g.height // tile, g.width // tile
    res = t.a * tile
    return crs, Affine(res, 0, t.c, 0, -res, t.f), nrow, ncol


def warp_to_grid(path, crs, transform, nrow, ncol, resampling, band_expr=None):
    """Read `path` onto the tile grid. band_expr maps the source array to float."""
    with rasterio.open(path) as src:
        with WarpedVRT(src, crs=crs, transform=transform, width=ncol, height=nrow,
                       resampling=resampling) as vrt:
            a = vrt.read(1, masked=True)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granule", default="025_019")
    ap.add_argument("--tile-size", type=int, default=E.TS)
    ap.add_argument("--bedmap", default=str(BEDMAP))
    ap.add_argument("--velocity", default=str(VEL))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gpath = find_granule(args.granule)
    crs, transform, nrow, ncol = tile_grid_spec(gpath, args.tile_size)
    res_km = transform.a / 1000
    print(f"granule {args.granule}: tile grid {nrow}x{ncol} @ {res_km:g} km/tile")
    out = Path(args.out) if args.out else (
        E.ROOT / "data" / f"_context_masks_{args.granule}_t{args.tile_size}.npz")

    # Bedmap3 is 500 m and categorical: warp at native res with mode, then average
    # the binary at tile res would need two passes; instead oversample by 4 and mean.
    sub = 4
    fine = Affine(transform.a / sub, 0, transform.c, 0, transform.e / sub, transform.f)
    print(f"warping Bedmap3 (class {GROUNDED} = grounded ice)...")
    bm = warp_to_grid(args.bedmap, crs, fine, nrow * sub, ncol * sub, Resampling.nearest)
    cls = bm.filled(0)
    frac = np.stack([(cls == k).reshape(nrow, sub, ncol, sub).mean(axis=(1, 3))
                     for k in BEDMAP_CLASSES])
    grounded_frac = frac[GROUNDED]
    bedmap_class = frac.argmax(0).astype(np.int16)

    print("warping ITS_LIVE speed (this reads a 7.5 GB source)...")
    vm = warp_to_grid(args.velocity, crs, fine, nrow * sub, ncol * sub, Resampling.bilinear)
    v = vm.astype(np.float32).filled(np.nan)
    v[v <= 0] = np.nan
    vb = v.reshape(nrow, sub, ncol, sub)
    with np.errstate(all="ignore"):
        vel_mean = np.nanmean(vb, axis=(1, 3))
        vel_max = np.nanmax(vb, axis=(1, 3))

    np.savez_compressed(out, grounded_frac=grounded_frac, bedmap_class=bedmap_class,
                        vel_mean=vel_mean, vel_max=vel_max)
    print(f"\nwrote {out}")
    print(f"  grounded_frac >0.5 : {int((grounded_frac > 0.5).sum())} of {nrow * ncol} tiles")
    for k, name in BEDMAP_CLASSES.items():
        print(f"  majority {name:16s}: {int((bedmap_class == k).sum()):5d} tiles")
    fin = np.isfinite(vel_mean)
    print(f"  velocity mapped    : {int(fin.sum())} tiles")
    if fin.any():
        print(f"  speed percentiles  : " + "  ".join(
            f"p{q}={np.nanpercentile(vel_mean, q):.0f}" for q in (10, 50, 90, 99)))
    for t in (50, 100, 250, 500):
        print(f"  grounded & >{t:4d} m/yr : "
              f"{int(((grounded_frac > 0.5) & (vel_mean > t)).sum()):5d} tiles")


if __name__ == "__main__":
    main()
