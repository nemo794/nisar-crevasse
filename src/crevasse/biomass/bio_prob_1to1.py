"""Gate probability against AlphaEarth on one shared grid, and on a 1:1 diagonal.

    conda run -n biomass python code/bio_prob_1to1.py

Writes, per granule:
  data/bio_1to1_map_<tag>.png    label / probability / signed difference, same grid, same scale
  data/bio_1to1_cal_<tag>.png    probability vs label on a 1:1 diagonal
and one pooled data/bio_1to1_cal_POOLED.png.

WHY THE TWO GRIDS CAN BE PUT ON TOP OF EACH OTHER AT ALL
-------------------------------------------------------
AlphaEarth is grid-aligned to every BIOMASS footprint at an exactly integer pixel offset
(checked by bio_check_granule.py [5]), so the categorical mask is never resampled onto a
different grid -- it is only block-averaged, by an integer factor, onto the same grid the
probability is drawn on. Because the raster is strictly 0/1 (verified, no nodata), an average
resample IS the positive fraction, which is the quantity the label threshold is defined on.

The probability is per 512px TILE; the label is per 5 m PIXEL. To compare them 1:1 the tile
probability is replicated over its own DEC-decimated footprint (np.kron, an exact integer
expansion -- no interpolation, no smoothing), and the label is averaged down to the same
cells. Both then live on one (nr*SUB, nc*SUB) array, so the difference panel is a true
per-cell subtraction rather than two pictures placed side by side.

WHAT THE DIFFERENCE PANEL DOES AND DOES NOT MEAN
------------------------------------------------
It is P minus AlphaEarth positive fraction. Red = the gate is more confident than the label,
blue = less. It is NOT an error map:

  * The gate cannot resolve sub-tile structure by construction, so a tile that is genuinely
    half-crevassed can only be scored 0..1 for the whole 2.56 km, and the difference panel
    will show a gradient there that is arithmetic, not a mistake.
  * AlphaEarth's recall is ~0.42 at precision ~0.94, so red is the direction in which the
    label is known to be wrong. Blue is the direction in which it is trusted.

Probabilities are the buffered out-of-fold scores (82 km blocks / 41 km buffer), the only
separation at which the radar features clearly beat map coordinates. Unscored cells are left
blank; see bio_pred_maps.py's reason panel for why each one is blank.
"""
import argparse
import glob
import os
import re

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crevasse.biomass.bio_gate_controls import buffered_cv

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "biomass", "gate")
T = 512
PX_M = 5.0
DEC = 32
SUB = T // DEC
AOI_PATH = os.path.join(ROOT, "data", "aois", "new", "thwaites_mosaic_new.vrt")

BLOCK_TILES = 32
BUFFER_TILES = 16


def aoi_fraction(gdir, shape_full, tile_shape):
    """AlphaEarth positive fraction on the DEC-decimated granule grid.

    Average-resampling is legitimate here only because the raster is strictly 0/1 and the
    offset into it is an exact integer, so this is a block mean of an indicator, i.e. the
    positive fraction -- not an interpolation of category codes.
    """
    h, w = shape_full
    nr, nc = tile_shape
    oh, ow = nr * SUB, nc * SUB
    hh = glob.glob(os.path.join(gdir, "*_HH_intensity.tif"))[0]
    with rasterio.open(hh) as s:
        tf = s.transform
    with rasterio.open(AOI_PATH) as a:
        dr = (tf.f - a.transform.f) / a.transform.e
        dc = (tf.c - a.transform.c) / a.transform.a
        assert abs(dr - round(dr)) < 1e-4 and abs(dc - round(dc)) < 1e-4, \
            f"non-integer AlphaEarth offset dr={dr} dc={dc}; a 0/1 mask must not be shifted"
        frac = a.read(1, window=Window(int(round(dc)), int(round(dr)), w, h),
                      out_shape=(oh, ow), boundless=True, fill_value=0,
                      resampling=Resampling.average).astype(np.float32)
    return frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(
        ROOT, "data", f"bio_tile_features_t{T}.npz"))
    ap.add_argument("--pos-thresh", type=float, default=0.5)
    ap.add_argument("--neg-thresh", type=float, default=0.05)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X = d["X"]
    ap_ = d["aoi_pos"]
    keep = d["aoi_inbox"] & ((ap_ >= args.pos_thresh) | (ap_ <= args.neg_thresh))
    keep &= np.isfinite(X).all(axis=1)
    y = ap_[keep] >= args.pos_thresh
    block_m, buffer_m = BLOCK_TILES * T * PX_M, BUFFER_TILES * T * PX_M
    print(f"scoring out-of-fold at {block_m/1000:.0f} km blocks / "
          f"{buffer_m/1000:.0f} km buffer", flush=True)
    p, _, nf = buffered_cv(X[keep], y, d["x_m"][keep], d["y_m"][keep], block_m, buffer_m)
    print(f"  {nf} folds, {np.isfinite(p).sum()} scored", flush=True)
    P = np.full(len(ap_), np.nan)
    P[keep] = p
    gran, row, col = d["granule"], d["row"], d["col"]

    pooled = []
    base = os.path.join(ROOT, "data", "biomass")
    for g in sorted(set(gran.tolist())):
        sel = gran == g
        gdir = os.path.join(base, g)
        hh = glob.glob(os.path.join(gdir, "*_HH_intensity.tif"))
        if not hh:
            continue
        with rasterio.open(hh[0]) as s:
            h, w = s.shape
        nr, nc = h // T, w // T

        ti, tj = row[sel] // T, col[sel] // T
        ptile = np.full((nr, nc), np.nan)
        ptile[ti, tj] = P[sel]
        # exact integer expansion of the tile grid onto the decimated pixel grid
        pcell = np.kron(ptile, np.ones((SUB, SUB), np.float32))
        lcell = aoi_fraction(gdir, (h, w), (nr, nc))
        assert pcell.shape == lcell.shape, f"{pcell.shape} != {lcell.shape}"
        pcell[np.isnan(pcell)] = np.nan
        lcell = np.where(np.isfinite(pcell), lcell, np.nan)

        ok = np.isfinite(pcell)
        if not ok.any():
            print(f"  skip {g}: nothing scored")
            continue
        ri, ci = np.where(ok)
        r0, r1, c0, c1 = ri.min(), ri.max() + 1, ci.min(), ci.max() + 1
        pc, lc = pcell[r0:r1, c0:c1], lcell[r0:r1, c0:c1]

        m = re.match(r"BIO_(S\d)_SCS__1S_(\d{8})T.*_(M\d\d)_", g)
        tag = f"{m.group(1)}_{m.group(3)}_{m.group(2)}" if m else g[:24]

        fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.4))
        for a_, img, ttl, cm, vr in [
                (ax[0], lc, "AlphaEarth positive fraction", "magma", (0, 1)),
                (ax[1], pc, "gate P (out-of-fold)", "magma", (0, 1)),
                (ax[2], pc - lc, "P - AlphaEarth", "bwr", (-1, 1))]:
            im = a_.imshow(img, cmap=cm, vmin=vr[0], vmax=vr[1], interpolation="nearest")
            a_.set_title(ttl, fontsize=10)
            a_.set_xticks([])
            a_.set_yticks([])
            fig.colorbar(im, ax=a_, fraction=0.046, pad=0.02)
        md = float(np.nanmean(pc - lc))
        fig.suptitle(f"{tag}   one shared {DEC}x-decimated grid, {int(ok.sum())} cells   "
                     f"mean(P - label) {md:+.3f}\n"
                     f"P is per 2.56 km tile replicated exactly; label is the 0/1 mask "
                     f"block-averaged. Red = gate above label, where AlphaEarth's ~0.42 "
                     f"recall says it is the label that is low.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        out = os.path.join(ROOT, "data", f"bio_1to1_map_{tag}.png")
        fig.savefig(out, dpi=115)
        plt.close(fig)
        print(f"wrote {out}", flush=True)

        # tile-level 1:1, which is the resolution the model actually predicts at
        s2 = sel & np.isfinite(P)
        pooled.append((ap_[s2], P[s2], tag))
        scatter(ap_[s2], P[s2], tag,
                os.path.join(ROOT, "data", f"bio_1to1_cal_{tag}.png"), args)

    if pooled:
        A = np.concatenate([a for a, _, _ in pooled])
        B = np.concatenate([b for _, b, _ in pooled])
        scatter(A, B, f"POOLED, {len(pooled)} granules",
                os.path.join(ROOT, "data", "bio_1to1_cal_POOLED.png"), args)


def scatter(a, p, tag, out, args):
    """gate P against AlphaEarth positive fraction, one point per tile, with y=x."""
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.2))
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="1:1")
    ax[0].scatter(a, p, s=7, alpha=0.35, c="#2b6cb0", edgecolors="none")
    # binned means, the readable version of the same data
    edges = np.linspace(0, 1, 11)
    bi = np.clip(np.digitize(a, edges) - 1, 0, 9)
    bx = [a[bi == k].mean() for k in range(10) if (bi == k).sum() >= 5]
    by = [p[bi == k].mean() for k in range(10) if (bi == k).sum() >= 5]
    ax[0].plot(bx, by, "o-", color="#d94040", lw=1.6, ms=5, label="binned mean")
    ax[0].axhline(0.5, color="grey", lw=0.7, ls=":")
    ax[0].axvline(args.pos_thresh, color="grey", lw=0.7, ls=":")
    ax[0].axvline(args.neg_thresh, color="grey", lw=0.7, ls=":")
    ax[0].set_xlabel("AlphaEarth positive fraction (label)")
    ax[0].set_ylabel("gate P (out-of-fold)")
    ax[0].set_xlim(-0.03, 1.03)
    ax[0].set_ylim(-0.03, 1.03)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].set_title("1:1  (note the x gap: 0.05-0.50 is dropped as ambiguous)", fontsize=9)

    y = a >= args.pos_thresh
    ax[1].hist([p[~y], p[y]], bins=np.linspace(0, 1, 26), stacked=False,
               label=[f"label negative (n={(~y).sum()})", f"label positive (n={y.sum()})"],
               color=["#8c8c8c", "#2ea84f"])
    ax[1].set_xlabel("gate P")
    ax[1].set_ylabel("tiles")
    ax[1].legend(fontsize=8)
    ax[1].set_title("separation of the two label classes", fontsize=9)
    fig.suptitle(f"{tag}   n={len(a)} scored tiles", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
