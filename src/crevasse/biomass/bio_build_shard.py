"""Build a packed .npz training shard: 4-pol dB tile stacks paired with the AlphaEarth
per-pixel crevasse mask, for every tile fully inside the AlphaEarth export box.

    conda run -n biomass python src/bio_build_shard.py \
        --features ../biomass-crevasse-gate/data/bio_tile_features_t512.npz \
        --aoi ../biomass/data/aois/new/thwaites_mosaic_new.vrt \
        --base ../biomass/data/biomass \
        --out data/train_biomass_v1.npz

WHY THIS IS SIMPLER THAN THE NISAR SHARD
-----------------------------------------
NISAR had to manufacture a pixel-level target (Frangi ridge detection) because there is
no ground truth for it. BIOMASS does not need to: AlphaEarth is already a pixel-level
byte 0/1 mask, exactly grid-aligned to every BIOMASS SAR tile at an integer pixel offset
(same `dr`/`dc` calculation `bio_tile_features.py` computes inline). So the target here
is a real label read straight off the mosaic, not a soft, gated response requiring a
discriminator term the way `soft_labels.py`'s `rect(orient_agree, t)` is for NISAR.

WHY ONLY aoi_inbox TILES, AND WHY THAT MAKES A PER-PIXEL MASK UNNECESSARY
--------------------------------------------------------------------------
AlphaEarth's export box is smaller than its raster extent -- byte 0 OUTSIDE the box means
"never evaluated," not "no crevasse" (the same trap the RF/CNN gate's `aoi_inbox` guards
against). Rather than carry a per-pixel in/out-of-box mask through training and mask the
loss with it, this shard is built ONLY from tiles whose full 512x512 footprint lies
inside the export box (`aoi_inbox` in `bio_tile_features_t512.npz`) -- so every kept
tile's mask is trustworthy across its whole area (up to AlphaEarth's own recall ~0.42 /
precision ~0.94, which is a property of the label itself, not something a mask can fix).
This also means every admissible tile is used regardless of gate score -- unlike NISAR,
there is no risk of poisoning a manufactured label with speckle, so gate-selection buys
nothing here and is deliberately not applied.

INPUT CHANNELS
---------------
Reuses `bio_tile_cache.granule_chips` from biomass-crevasse-gate at `chip=512`, i.e. no
block-average resampling -- the full 512x512 tile at native 5 m, 4 channels
(HH, HV, VH, VV) in dB, NaN where the SAR is nodata. Raw dB, no per-tile or per-channel
stretch, for the same reason `bio_tile_cache.py` itself avoids one: destroying the
absolute level would destroy the cross/co offset, the one thing the radar actually
carries on this sensor. Normalization (one shared mean/sd, fit on the training split
only) happens later, in `bio_sar_dataset.py` / at training time -- see that module.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from crevasse.biomass.bio_tile_cache import granule_chips, POLS, T  # noqa: E402


def aoi_offset(hh_path, aoi_path):
    """Integer (dr, dc) pixel offset from the granule's own grid into the AlphaEarth
    mosaic's grid -- the same calculation `bio_tile_features.py` inlines; factored out
    here so it isn't a second, possibly-drifting copy."""
    with rasterio.open(hh_path) as s:
        tf = s.transform
    with rasterio.open(aoi_path) as a:
        dr = int(round((tf.f - a.transform.f) / a.transform.e))
        dc = int(round((tf.c - a.transform.c) / a.transform.a))
    return dr, dc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True,
                     help="bio_tile_features_t512.npz from biomass-crevasse-gate")
    ap.add_argument("--aoi", required=True, help="AlphaEarth mosaic .vrt/.tif")
    ap.add_argument("--base", required=True, help="directory holding granule directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tiles", type=int, default=None,
                     help="cap on tiles, taken as a seeded random sample across granules "
                          "-- for a quick correctness check before building the real "
                          "shard. Omit for every aoi_inbox tile.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    keep = d["aoi_inbox"] & np.isfinite(d["X"]).all(axis=1)
    n_total = len(keep)
    print(f"{int(keep.sum())}/{n_total} tiles are fully inside the AlphaEarth export box")
    if not keep.any():
        raise SystemExit("no aoi_inbox tiles in --features -- nothing to build a shard from")

    idx = np.flatnonzero(keep)
    if args.max_tiles is not None and len(idx) > args.max_tiles:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(idx, size=args.max_tiles, replace=False))
        print(f"  capped to {len(idx)} tiles (seed={args.seed})")

    gran = d["granule"][idx]
    row, col = d["row"][idx].astype(np.int64), d["col"][idx].astype(np.int64)
    xm, ym = d["x_m"][idx], d["y_m"][idx]
    apos = d["aoi_pos"][idx].astype(np.float32)

    images = np.full((len(gran), len(POLS), T, T), np.nan, np.float16)
    targets = np.zeros((len(gran), T, T), np.uint8)

    for g in sorted(set(gran.tolist())):
        sel = np.where(gran == g)[0]
        gdir = os.path.join(args.base, g)
        hh_path = glob.glob(os.path.join(gdir, "*_HH_intensity.tif"))[0]
        dr, dc = aoi_offset(hh_path, args.aoi)

        images[sel] = granule_chips((gdir, row[sel], col[sel], T))

        with rasterio.open(args.aoi) as aoi:
            for k in sel:
                av = aoi.read(1, window=Window(int(col[k]) + dc, int(row[k]) + dr, T, T),
                              boundless=True, fill_value=0)
                targets[k] = (av == 1).astype(np.uint8)
        print(f"  {g}: {len(sel)} tiles", flush=True)

    coverage = targets.astype(np.float32).mean(axis=(1, 2))
    # Recomputed from the mask actually stored, not copied from aoi_pos, so shard and
    # target can never silently disagree -- same discipline NISAR's build_shard.py uses.
    # Both are `(av == 1).mean()` over the identical window, so this must be exact.
    mismatch = float(np.abs(coverage - apos).max())
    assert mismatch < 1e-6, (
        f"coverage recomputed from the stored mask disagrees with the feature table's "
        f"aoi_pos by {mismatch:.4f} -- they read the same byte mask and must match "
        f"exactly. Check the (row, col) join between --features and this shard.")
    print(f"coverage vs feature-table aoi_pos: max abs diff {mismatch:.2e} (exact)")

    np.savez(args.out, images=images, targets=targets,
             positions=np.stack([row, col], axis=1),
             x_m=xm.astype(np.float64), y_m=ym.astype(np.float64),
             coverage=coverage, aoi_pos=apos,
             granule=gran, pols=np.array(POLS), tile_size=np.int32(T))
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
