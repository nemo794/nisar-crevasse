"""Cache 4-channel (HH, HV, VH, VV) tile chips plus per-tile context, for the CNN gate.

    conda run -n biomass python code/bio_tile_cache.py
    conda run -n biomass python code/bio_tile_cache.py --chip 64

Writes data/bio_chipcache_t512_c<chip>.npz:
  chips        (n, 4, chip, chip) float16   log-intensity dB, NaN where the SAR is nodata
  y            (n,) bool                    AlphaEarth positive fraction >= pos-thresh
  idx          (n,) int32                   row index back into bio_tile_features_t512.npz
  x_m, y_m     (n,) float64                 tile upper-left in EPSG:3031, for the fold split
  vel, bcls, gfrac                          per-tile context, see below
  aoi_pos, granule, row, col                carried through for plotting

WHY THE CHIPS ARE STORED IN dB WITH NO PER-TILE STRETCH
------------------------------------------------------
The obvious move is to normalise each chip to its own p2/p98. Do not: on NISAR that exact
per-tile stretch in preprocess_sar is the documented reason amplitude came out
ANTI-correlated with the label, because it deletes the absolute level the label depends on.
Worse here, a per-CHANNEL stretch would destroy the cross-to-co offset, which is the single
thing the radar actually carries on this sensor (ratio_dB AUC 0.890 on the pilot). So chips
are raw dB, and normalisation is one shared mean/sd applied at training time from the fold's
own training tiles only.

DECIMATION IS NOT LOSSY HERE THE WAY IT LOOKS
--------------------------------------------
512 px at 5 m goes to `chip` px, so the default 128 is 20 m pixels. BIOMASS's 5 m grid is
oversampled with a ~3:1 PSF whose long axis gives an effective resolution of ~25 m, so 20 m
cells discard oversampling, not signal. Going finer buys speckle. Going coarser than 40 m
starts erasing the across-axis structure.

THE CONTEXT FIELDS ARE A CONFOUND, CARRIED DELIBERATELY
------------------------------------------------------
vel_mean / bedmap_class / grounded_frac come from data/context_grid_2560m.npz, which is on a
2560 m grid -- exactly one cell per 512 px tile at 5 m -- so this is an index lookup, not a
resample. They are cached so the CNN can be run with and without them, because on NISAR
velocity ALONE scored AUC 0.866: it tells a model WHERE it is, not what the radar saw. Any
run that uses them must be reported beside a context-only control on the same folds.
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "biomass", "gate")
POLS = ["HH", "HV", "VH", "VV"]
T = 512
CTX_PATH = os.path.join(ROOT, "data", "context_grid_2560m.npz")


def granule_chips(job):
    """Read every requested tile of one granule as (n, 4, chip, chip) float16 dB."""
    gdir, rows, cols, chip = job
    paths = {p: glob.glob(os.path.join(gdir, f"*_{p}_intensity.tif"))[0] for p in POLS}
    out = np.full((len(rows), len(POLS), chip, chip), np.nan, np.float16)
    srcs = {p: rasterio.open(paths[p]) for p in POLS}
    try:
        for k, (r, c) in enumerate(zip(rows, cols)):
            w = Window(int(c), int(r), T, T)
            for pi, p in enumerate(POLS):
                a = srcs[p].read(1, window=w, out_shape=(chip, chip),
                                 resampling=Resampling.average).astype(np.float32)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[k, pi] = 10 * np.log10(np.where(a > 0, a, np.nan))
    finally:
        for s in srcs.values():
            s.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(
        ROOT, "data", f"bio_tile_features_t{T}.npz"))
    ap.add_argument("--chip", type=int, default=128, help="chip edge in px (512 -> chip)")
    ap.add_argument("--pos-thresh", type=float, default=0.5)
    ap.add_argument("--neg-thresh", type=float, default=0.05)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--all", action="store_true",
                    help="cache every admissible tile, not just the unambiguously labelled "
                         "ones. Needed to paint a whole scene, since the ambiguous tiles the "
                         "trainer drops are still real ground a deployed gate must answer for.")
    ap.add_argument("--base", default=os.path.join(ROOT, "data", "biomass"),
                    help="directory holding granule directories")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # the labelled cache keeps its original untagged name so nothing already written moves
    tag = "_all" if args.all else ""
    out = args.out or os.path.join(
        ROOT, "data", f"bio_chipcache_t{T}_c{args.chip}{tag}.npz")

    d = np.load(args.features, allow_pickle=True)
    apos, X = d["aoi_pos"], d["X"]
    unlabelled = not np.isfinite(apos).any()
    if unlabelled and not args.all:
        raise SystemExit(
            f"{args.features} carries no labels (aoi_pos is all NaN), so there is nothing "
            "to select on. Pass --all to cache every admissible tile; the result is "
            "scoreable but not trainable.")
    keep = np.isfinite(X).all(axis=1)
    if not args.all:
        keep &= d["aoi_inbox"] & ((apos >= args.pos_thresh) | (apos <= args.neg_thresh))
    idx = np.where(keep)[0]
    with np.errstate(invalid="ignore"):
        y = apos[idx] >= args.pos_thresh          # all False on an unlabelled table
    gran, row, col = d["granule"][idx], d["row"][idx], d["col"][idx]
    xm, ym = d["x_m"][idx], d["y_m"][idx]
    print(f"caching {len(idx)} {'unlabelled' if unlabelled else 'labelled'} tiles, "
          f"{int(y.sum())} positive "
          f"({y.mean():.3f}) at {args.chip}px ({T * 5.0 / args.chip:.0f} m)", flush=True)
    mb = len(idx) * 4 * args.chip ** 2 * 2 / 1e6
    print(f"  chips will be {mb:.0f} MB float16", flush=True)

    base = args.base
    gs = sorted(set(gran.tolist()))
    jobs, slots = [], []
    for g in gs:
        s = np.where(gran == g)[0]
        jobs.append((os.path.join(base, g), row[s], col[s], args.chip))
        slots.append(s)

    chips = np.full((len(idx), 4, args.chip, args.chip), np.nan, np.float16)
    with Pool(min(args.procs, len(jobs))) as pool:
        for (g, s, part) in zip(gs, slots, pool.imap(granule_chips, jobs)):
            chips[s] = part
            print(f"  {g[:28]}  {len(s)} tiles", flush=True)

    # context: 2560 m grid, one cell per tile, so look up the tile CENTRE and index
    ctx = np.load(CTX_PATH, allow_pickle=True)
    inv = ~Affine(*ctx["transform"])
    cc, rr = inv * (xm + T * 2.5, ym - T * 2.5)
    rr, cc = np.floor(rr).astype(int), np.floor(cc).astype(int)
    h, w = ctx["grounded_frac"].shape
    assert (rr >= 0).all() and (rr < h).all() and (cc >= 0).all() and (cc < w).all(), \
        "a tile centre falls outside the continent-wide context grid"
    vel = ctx["vel_mean"][rr, cc].astype(np.float32)
    bcls = ctx["bedmap_class"][rr, cc].astype(np.int16)
    gfrac = ctx["grounded_frac"][rr, cc].astype(np.float32)

    valid = np.isfinite(chips).all(axis=1).mean(axis=(1, 2))
    print(f"\nchip validity: {valid.mean():.4f} mean, {(valid < 0.99).sum()} tiles "
          f"below 0.99 across all four pols", flush=True)
    print(f"velocity mapped on {int(np.isfinite(vel).sum())} of {len(vel)} tiles; "
          f"bedmap_class counts {np.bincount(bcls.clip(0)).tolist()}", flush=True)
    fin = np.isfinite(chips)
    print(f"dB range {np.nanmin(chips[fin]):.1f} .. {np.nanmax(chips[fin]):.1f}", flush=True)

    # which cached tiles are trainable at all; false only in an --all cache, and false
    # everywhere when the feature table carries no labels
    with np.errstate(invalid="ignore"):
        labelled = (d["aoi_inbox"][idx]
                    & ((apos[idx] >= args.pos_thresh) | (apos[idx] <= args.neg_thresh)))
    print(f"{int(labelled.sum())} of {len(idx)} cached tiles are unambiguously labelled",
          flush=True)

    np.savez(out, chips=chips, y=y, labelled=labelled, idx=idx.astype(np.int32),
             x_m=xm, y_m=ym,
             vel=vel, bcls=bcls, gfrac=gfrac, valid=valid.astype(np.float32),
             aoi_pos=apos[idx], granule=gran, row=row, col=col,
             pols=np.array(POLS), chip=args.chip)
    print(f"wrote {out}  ({os.path.getsize(out) / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
