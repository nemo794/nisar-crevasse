"""Scene overlays in the style of plot_gate_comparison.py: HH, AlphaEarth, gate P.

    conda run -n biomass python code/bio_prob_scene.py
    conda run -n biomass python code/bio_prob_scene.py --dec 8

Writes data/bio_probscene_<tag>.png per granule. Probability only, no metrics on the figure.

EVERY TILE IS SCORED, WHICH MEANS THIS IS THE FULL-DATA FIT
----------------------------------------------------------
The buffered out-of-fold scores only cover 1747 of 4767 tiles -- a tile gets one only if it
lands in a test block whose 41 km-buffered training set still held both classes. To put a
probability on every tile the model has to be the one trained on all labelled tiles
(data/bio_gate_t512.joblib), so most tiles here are scored IN SAMPLE.

That is the right choice for a picture of what the gate does to a whole scene, and the wrong
one for any number. Concretely: in-sample this model separates at AUC ~0.95, buffered
out-of-fold it is 0.889 against 0.601 for map coordinates alone. Read these panels for spatial
behaviour -- where it fires, whether the field boundaries look sane, whether it bleeds onto
shelf ice -- and read bio_1to1_cal_POOLED.png for how good it actually is.

Ambiguous tiles (AlphaEarth positive fraction between 0.05 and 0.5) are scored and drawn here
even though the trainer dropped them, because they are real ground the deployed gate would
have to answer for.
"""
import argparse
import glob
import os
import re

import joblib
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "biomass", "gate")
T = 512
AOI_PATH = os.path.join(ROOT, "data", "aois", "new", "thwaites_mosaic_new.vrt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(
        ROOT, "data", f"bio_tile_features_t{T}.npz"))
    ap.add_argument("--bundle", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(ROOT))), "models", "biomass", "gate", f"bio_gate_t{T}.joblib"))
    ap.add_argument("--dec", type=int, default=16, help="basemap decimation")
    args = ap.parse_args()
    dec = args.dec
    sub = T // dec

    b = joblib.load(args.bundle)
    d = np.load(args.features, allow_pickle=True)
    X = d["X"]
    names = list(d["feature_names"])
    assert names == list(b["feature_names"]), "feature order differs from the bundle"

    ok = np.isfinite(X).all(axis=1)
    P = np.full(len(X), np.nan)
    P[ok] = b["model"].predict_proba(X[ok])[:, 1]
    print(f"scored {ok.sum()} of {len(X)} tiles "
          f"({(~ok).sum()} dropped for non-finite features)", flush=True)

    gran, row, col = d["granule"], d["row"], d["col"]
    base_dir = os.path.join(ROOT, "data", "biomass")
    for g in sorted(set(gran.tolist())):
        sel = gran == g
        gdir = os.path.join(base_dir, g)
        hh = glob.glob(os.path.join(gdir, "*_HH_intensity.tif"))
        if not hh:
            continue
        with rasterio.open(hh[0]) as s:
            h, w = s.shape
            tf = s.transform
            bg = s.read(1, out_shape=(h // dec, w // dec),
                        resampling=Resampling.average)
        with np.errstate(invalid="ignore", divide="ignore"):
            bg = 10 * np.log10(np.where(bg > 0, bg, np.nan))
        nr, nc = h // T, w // T
        oh, ow = nr * sub, nc * sub

        with rasterio.open(AOI_PATH) as a:
            dr = int(round((tf.f - a.transform.f) / a.transform.e))
            dc = int(round((tf.c - a.transform.c) / a.transform.a))
            aoi = a.read(1, window=Window(dc, dr, w, h), out_shape=(oh, ow),
                         boundless=True, fill_value=0,
                         resampling=Resampling.average).astype(np.float32)

        grid = np.full((nr, nc), np.nan, np.float32)
        grid[row[sel] // T, col[sel] // T] = P[sel]
        pcell = np.kron(grid, np.ones((sub, sub), np.float32))

        bg = bg[:oh, :ow]
        fin = np.isfinite(bg)
        ri, ci = np.where(fin)
        r0, r1, c0, c1 = ri.min(), ri.max() + 1, ci.min(), ci.max() + 1
        bg, aoi, pcell = bg[r0:r1, c0:c1], aoi[r0:r1, c0:c1], pcell[r0:r1, c0:c1]

        v = bg[np.isfinite(bg)]
        lo, hi = np.percentile(v, [2, 98])
        m = re.match(r"BIO_(S\d)_SCS__1S_(\d{8})T.*_(M\d\d)_", g)
        tag = f"{m.group(1)}_{m.group(3)}_{m.group(2)}" if m else g[:24]

        asp = bg.shape[0] / bg.shape[1]
        fig, axes = plt.subplots(1, 3, figsize=(7.0 * 3, 7.0 * asp + 1.4))
        for ax in axes:
            ax.imshow(bg, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0].set_title("HH dB", fontsize=13)

        axes[1].imshow(np.ma.masked_where(aoi <= 0, aoi), cmap="autumn",
                       vmin=0, vmax=1, alpha=0.7, interpolation="nearest")
        axes[1].set_title("AlphaEarth crevasse mask", fontsize=13)

        im = axes[2].imshow(np.ma.masked_invalid(pcell), cmap="turbo",
                            vmin=0, vmax=1, alpha=0.8, interpolation="nearest")
        axes[2].set_title("gate P(crevasse)", fontsize=13)
        cb = fig.colorbar(im, ax=axes[2], fraction=0.040, pad=0.02)
        cb.set_label("P", fontsize=11)

        fig.suptitle(f"{tag}   all {int(sel.sum())} tiles, {T}px @ 5 m   "
                     f"(full-data fit, so mostly in sample -- see the docstring)",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = os.path.join(ROOT, "data", f"bio_probscene_{tag}.png")
        fig.savefig(out, dpi=115, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
