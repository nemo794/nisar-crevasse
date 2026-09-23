"""Scene maps: gate prediction vs AlphaEarth, on the tile grid, at several thresholds.

    conda run -n biomass python code/bio_pred_maps.py
    conda run -n biomass python code/bio_pred_maps.py --thresholds 0.3,0.5,0.65,0.8

Writes one data/bio_predmap_<stack>_<foot>_<date>.png per granule.

Panels per granule:
  HH dB            decimated backdrop, for geographic context
  AlphaEarth       per-tile positive fraction, the label being compared against
  gate P           out-of-fold probability
  thr = ...        agreement at each threshold: TP / FP / FN / TN

THE PROBABILITIES ARE BUFFERED OUT-OF-FOLD, SO COVERAGE IS SPARSE ON PURPOSE
---------------------------------------------------------------------------
Scored at 82 km blocks with a 41 km buffer, which is the only separation at which the radar
features clearly beat map coordinates (model 0.889 vs (x,y) 0.601; unbuffered it is 0.954 vs
0.938, i.e. almost pure geography). Buffering costs coverage: a tile is only scored when it
falls in a test block whose buffered training set still had enough of both classes left, so
1747 of 4244 labelled tiles get a probability and the rest are drawn as UNSCORED.

Filling those gaps with a full-data-fit probability would make the maps look complete and
mean nothing -- every tile would be scored by a model that trained on it. The holes are the
honest cost of the separation.

Tiles that are ambiguous (AlphaEarth positive fraction between the two thresholds) or outside
the export box were never admissible and are also drawn as UNSCORED, since out there byte 0
means "never evaluated", not "no crevasse".
"""
import argparse
import glob
import os
import re

import numpy as np
import rasterio
from rasterio.enums import Resampling

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from rasterio.windows import from_bounds

from crevasse.biomass.bio_gate_controls import buffered_cv

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "biomass", "gate")
POLS = ["HH", "HV", "VH", "VV"]
BM_PATH = os.path.join(ROOT, "data", "aois", "bedmap3_mask.tif")
T = 512
PX_M = 5.0
DEC = 32
SUB = T // DEC          # decimated cells per tile edge
MIN_GND = 0.95
MIN_VALID = 0.99

BLOCK_TILES = 32
BUFFER_TILES = 16

# 0 unscored, 1 TN, 2 FN, 3 FP, 4 TP.
# TN must be plainly visible. It was first drawn as dark grey at 33% alpha, which over the
# HH backdrop is indistinguishable from unscored -- so correctly-quiet ground read as
# "missing data" and the maps invited exactly the wrong question.
AGREE_COLORS = ["#00000000", "#aab8c2dd", "#3b6fd4", "#d94040", "#2ea84f"]
AGREE_LABELS = ["unscored (see reason panel)", "TN", "FN (missed)", "FP (gate only)", "TP"]

# Why a tile is blank. Four different reasons look identical on an agreement panel, and
# they have completely different implications, so they get their own map.
WHY_COLORS = ["#00000000", "#5a4b8a", "#c9a227", "#8c8c8c", "#2ea84f"]
WHY_LABELS = ["no valid SAR (outside swath)", "not grounded ice (shelf/ocean)",
              "ambiguous label, dropped", "labelled, unscored by buffered CV",
              "labelled + scored"]


def coverage_reason(gdir, bounds, shape_full, tile_shape, ti, tj, apos, prob, args):
    """Per-tile code for WHY it is blank, on the same prescan rules as the feature builder.

    Recomputed here rather than read from the npz because the npz only contains tiles that
    already passed the prescan -- the interesting categories are the ones it excluded.
    """
    h, w = shape_full
    nr, nc = tile_shape
    oh, ow = nr * SUB, nc * SUB
    paths = {p: glob.glob(os.path.join(gdir, f"*_{p}_intensity.tif"))[0] for p in POLS}
    valid = np.ones((oh, ow), bool)
    for p in POLS:
        with rasterio.open(paths[p]) as s:
            a = s.read(1, out_shape=(h // DEC, w // DEC),
                       resampling=Resampling.average)[:oh, :ow]
        valid &= np.isfinite(a) & (a > 0)
    vf = valid.reshape(nr, SUB, nc, SUB).mean(axis=(1, 3))
    with rasterio.open(BM_PATH) as bm:
        b = bm.read(1, window=from_bounds(*bounds, transform=bm.transform),
                    out_shape=(h // DEC, w // DEC), boundless=True, fill_value=-9999,
                    resampling=Resampling.nearest)[:oh, :ow]
    gf = (b == 1).reshape(nr, SUB, nc, SUB).mean(axis=(1, 3))

    why = np.zeros((nr, nc), int)
    why[(vf >= MIN_VALID) & (gf < MIN_GND)] = 1
    amb = (apos > args.neg_thresh) & (apos < args.pos_thresh)
    why[ti[amb], tj[amb]] = 2
    lab_ok = ~amb
    why[ti[lab_ok], tj[lab_ok]] = 3
    sc = lab_ok & np.isfinite(prob)
    why[ti[sc], tj[sc]] = 4
    return why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(
        ROOT, "data", f"bio_tile_features_t{T}.npz"))
    ap.add_argument("--pos-thresh", type=float, default=0.5)
    ap.add_argument("--neg-thresh", type=float, default=0.05)
    ap.add_argument("--thresholds", default="0.30,0.50,0.65,0.80")
    args = ap.parse_args()
    thrs = [float(t) for t in args.thresholds.split(",")]

    d = np.load(args.features, allow_pickle=True)
    X = d["X"]
    ap_ = d["aoi_pos"]
    keep = d["aoi_inbox"] & ((ap_ >= args.pos_thresh) | (ap_ <= args.neg_thresh))
    keep &= np.isfinite(X).all(axis=1)
    y = ap_[keep] >= args.pos_thresh
    Xk = X[keep]
    xm, ym = d["x_m"][keep], d["y_m"][keep]

    block_m, buffer_m = BLOCK_TILES * T * PX_M, BUFFER_TILES * T * PX_M
    print(f"scoring out-of-fold at {block_m/1000:.0f} km blocks / "
          f"{buffer_m/1000:.0f} km buffer", flush=True)
    p, _, nf = buffered_cv(Xk, y, xm, ym, block_m, buffer_m)
    print(f"  {nf} folds, {np.isfinite(p).sum()} of {len(y)} labelled tiles scored",
          flush=True)

    # scatter the kept-tile probabilities back onto the full tile table
    P = np.full(len(ap_), np.nan)
    P[keep] = p
    gran, row, col = d["granule"], d["row"], d["col"]

    base = os.path.join(ROOT, "data", "biomass")
    for g in sorted(set(gran.tolist())):
        sel = gran == g
        gdir = os.path.join(base, g)
        hh = glob.glob(os.path.join(gdir, "*_HH_intensity.tif"))
        if not hh:
            print(f"  skip {g}: no HH")
            continue
        with rasterio.open(hh[0]) as s:
            h, w = s.shape
            bounds = s.bounds
            bg = s.read(1, out_shape=(h // DEC, w // DEC),
                        resampling=Resampling.average)
        nr, nc = h // T, w // T
        with np.errstate(invalid="ignore", divide="ignore"):
            bg = 10 * np.log10(np.where(bg > 0, bg, np.nan))

        ti, tj = row[sel] // T, col[sel] // T
        lab = np.full((nr, nc), np.nan)
        prob = np.full((nr, nc), np.nan)
        lab[ti, tj] = ap_[sel]
        prob[ti, tj] = P[sel]

        why = coverage_reason(gdir, bounds, (h, w), (nr, nc),
                              ti, tj, ap_[sel], P[sel], args)

        # crop to the labelled tiles; the swath is diagonal inside a mostly-empty
        # bounding box, and uncropped the panels are almost all nodata
        i0, i1 = int(ti.min()), int(ti.max())
        j0, j1 = int(tj.min()), int(tj.max())
        lab = lab[i0:i1 + 1, j0:j1 + 1]
        prob = prob[i0:i1 + 1, j0:j1 + 1]
        why = why[i0:i1 + 1, j0:j1 + 1]
        nr, nc = lab.shape
        bg = bg[i0 * SUB:(i1 + 1) * SUB, j0 * SUB:(j1 + 1) * SUB]

        m = re.match(r"BIO_(S\d)_SCS__1S_(\d{8})T.*_(M\d\d)_", g)
        tag = f"{m.group(1)}_{m.group(3)}_{m.group(2)}" if m else g[:24]
        ext = (0, nc * SUB, nr * SUB, 0)
        n_pan = 4 + len(thrs)
        ncols = 4
        nrows = int(np.ceil(n_pan / ncols))
        asp = (nr * SUB) / max(nc * SUB, 1)
        fig, axg = plt.subplots(nrows, ncols,
                                figsize=(4.6 * ncols, 4.6 * asp * nrows + 0.8))
        axes = axg.ravel()
        for extra in axes[n_pan:]:
            extra.axis("off")

        def backdrop(ax):
            v = bg[np.isfinite(bg)]
            lo, hi = np.percentile(v, [2, 98]) if v.size else (0, 1)
            ax.imshow(bg, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

        backdrop(axes[0])
        axes[0].set_title("HH dB", fontsize=9)

        backdrop(axes[1])
        im = axes[1].imshow(lab, cmap="magma", vmin=0, vmax=1, extent=ext,
                            interpolation="nearest", alpha=0.85)
        axes[1].set_title("AlphaEarth positive fraction", fontsize=9)
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)

        backdrop(axes[2])
        im = axes[2].imshow(prob, cmap="coolwarm", vmin=0, vmax=1, extent=ext,
                            interpolation="nearest", alpha=0.85)
        axes[2].set_title("gate P (out-of-fold)", fontsize=9)
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.02)

        wcmap = ListedColormap(WHY_COLORS)
        wnorm = BoundaryNorm(np.arange(-0.5, 5.5, 1), wcmap.N)
        backdrop(axes[3])
        axes[3].imshow(why, cmap=wcmap, norm=wnorm, extent=ext,
                       interpolation="nearest", alpha=0.9)
        axes[3].set_title("why a tile is blank", fontsize=9)
        axes[3].legend(handles=[Patch(facecolor=c, label=l)
                                for c, l in zip(WHY_COLORS, WHY_LABELS)],
                       loc="upper left", bbox_to_anchor=(0.0, -0.03), fontsize=6.5,
                       frameon=False)

        cmap = ListedColormap(AGREE_COLORS)
        norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)
        for k, thr in enumerate(thrs):
            axk = axes[4 + k]
            backdrop(axk)
            a = np.zeros((nr, nc), int)
            ok = np.isfinite(prob)
            truth = lab >= args.pos_thresh
            pred = prob >= thr
            a[ok & ~truth & ~pred] = 1
            a[ok & truth & ~pred] = 2
            a[ok & ~truth & pred] = 3
            a[ok & truth & pred] = 4
            axk.imshow(a, cmap=cmap, norm=norm, extent=ext, interpolation="nearest")
            tp, fp, fn = (a == 4).sum(), (a == 3).sum(), (a == 2).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            axk.set_title(f"thr {thr:.2f}\nprec {prec:.2f}  rec {rec:.2f}", fontsize=9)

        axes[-1].legend(handles=[Patch(facecolor=c, label=l)
                                 for c, l in zip(AGREE_COLORS, AGREE_LABELS)],
                        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7,
                        frameon=False)
        n_sc = int(np.isfinite(prob).sum())
        fig.suptitle(f"{tag}   {int(sel.sum())} tiles, {n_sc} scored out-of-fold at "
                     f"{block_m/1000:.0f} km blocks / {buffer_m/1000:.0f} km buffer",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 0.97, 0.95))
        out = os.path.join(ROOT, "data", f"bio_predmap_{tag}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}   {n_sc}/{int(sel.sum())} scored", flush=True)


if __name__ == "__main__":
    main()
