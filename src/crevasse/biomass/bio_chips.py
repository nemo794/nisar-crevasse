"""Render BIOMASS tile chips with AlphaEarth labels and honest gate probabilities.

    conda run -n biomass python code/bio_chips.py

Writes data/bio_chips_<class>.png, one figure per confusion class.

Each row is one 512px tile. Four panels:
  HH dB         co-pol brightness
  cross dB      (HV+VH)/2, the depolarized return
  ratio dB      cross/co per pixel -- the feature the model actually leans on
  AlphaEarth    the label, red where byte==1

THE PROBABILITY SHOWN IS OUT-OF-FOLD AT 82 km BLOCKS / 41 km BUFFER
-------------------------------------------------------------------
Not the full-data fit's probability, which would be near-memorized for every tile it was
trained on, and not the unbuffered out-of-fold score either. At 41 km blocks with no buffer
the model scores 0.954 while (x,y) alone scores 0.938 -- a +0.016 margin, i.e. mostly
memorized geography. Only the buffered setting (model 0.889, (x,y) 0.601) reflects what the
radar features contribute, so those are the numbers a chip should be annotated with.

Do not read these chips as a precision/recall estimate. AlphaEarth's own recall is ~0.42 at
precision ~0.94, so a "false positive" panel may well be a real crevasse it never labelled,
and that is exactly what looking at the imagery is for.
"""
import argparse
import glob
import os

import numpy as np
import rasterio
from rasterio.windows import Window

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from crevasse.biomass.bio_gate_controls import buffered_cv

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "biomass", "gate")
POLS = ["HH", "HV", "VH", "VV"]
T = 512
PX_M = 5.0
AOI_PATH = os.path.join(ROOT, "data", "aois", "new", "thwaites_mosaic_new.vrt")

# The one setting at which the radar features clearly beat map coordinates.
BLOCK_TILES = 32
BUFFER_TILES = 16


def db(a):
    with np.errstate(invalid="ignore", divide="ignore"):
        return 10 * np.log10(np.where(a > 0, a, np.nan))


def read_tile(gdir, r, c):
    """{pol: intensity, nodata as nan} plus the AlphaEarth label patch, or None."""
    paths = {p: glob.glob(os.path.join(gdir, f"*_{p}_intensity.tif"))[0] for p in POLS}
    arr = {}
    with rasterio.open(paths["HH"]) as s:
        tf = s.transform
    for p in POLS:
        with rasterio.open(paths[p]) as s:
            a = s.read(1, window=Window(c, r, T, T)).astype(np.float64)
        arr[p] = np.where(a > 0, a, np.nan)
    with rasterio.open(AOI_PATH) as a:
        dr = int(round((tf.f - a.transform.f) / a.transform.e))
        dc = int(round((tf.c - a.transform.c) / a.transform.a))
        lab = a.read(1, window=Window(c + dc, r + dr, T, T),
                     boundless=True, fill_value=0)
    return arr, lab


def panel(ax, img, title, cmap="gray", pct=(2, 98)):
    v = img[np.isfinite(img)]
    if v.size:
        lo, hi = np.percentile(v, pct)
    else:
        lo, hi = 0, 1
    ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def figure(tiles, gdirs, out, header):
    n = len(tiles)
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.9 * n), squeeze=False)
    lab_cmap = ListedColormap([(0, 0, 0, 0), (1, 0.15, 0.15, 0.85)])
    for k, t in enumerate(tiles):
        arr, lab = read_tile(gdirs[t["granule"]], t["row"], t["col"])
        co = (arr["HH"] + arr["VV"]) / 2
        cross = (arr["HV"] + arr["VH"]) / 2
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = db(cross / np.where(co > 0, co, np.nan))
        panel(axes[k][0], db(arr["HH"]), "HH dB")
        panel(axes[k][1], db(cross), "cross dB")
        panel(axes[k][2], ratio, "cross/co dB", cmap="viridis")
        axes[k][3].imshow(db(arr["HH"]), cmap="gray", interpolation="nearest")
        axes[k][3].imshow((lab == 1).astype(int), cmap=lab_cmap, vmin=0, vmax=1,
                          interpolation="nearest")
        axes[k][3].set_title(f"AlphaEarth  {t['aoi_pos']:.2f} positive", fontsize=8)
        axes[k][3].set_xticks([])
        axes[k][3].set_yticks([])
        axes[k][0].set_ylabel(
            f"{t['stack']} {t['foot']} {t['date']}\nr{t['row']} c{t['col']}\n"
            f"P={t['p']:.2f}  ratio {t['ratio_dB']:+.1f} dB", fontsize=7)
    fig.suptitle(header, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}  ({n} tiles)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(
        ROOT, "data", f"bio_tile_features_t{T}.npz"))
    ap.add_argument("--pos-thresh", type=float, default=0.5)
    ap.add_argument("--neg-thresh", type=float, default=0.05)
    ap.add_argument("--n-per-class", type=int, default=4)
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X, names = d["X"], list(d["feature_names"])
    ap_ = d["aoi_pos"]
    keep = d["aoi_inbox"] & ((ap_ >= args.pos_thresh) | (ap_ <= args.neg_thresh))
    keep &= np.isfinite(X).all(axis=1)
    y = ap_[keep] >= args.pos_thresh
    X = X[keep]
    xm, ym = d["x_m"][keep], d["y_m"][keep]
    meta = {k: d[k][keep] for k in ("granule", "stack", "foot", "date", "row", "col")}
    aoi_pos = ap_[keep]
    ri = names.index("ratio_dB")

    block_m = BLOCK_TILES * T * PX_M
    buffer_m = BUFFER_TILES * T * PX_M
    print(f"scoring out-of-fold at {block_m/1000:.0f} km blocks / "
          f"{buffer_m/1000:.0f} km buffer", flush=True)
    p, _, nf = buffered_cv(X, y, xm, ym, block_m, buffer_m)
    m = np.isfinite(p)
    print(f"  {nf} folds, {m.sum()} of {len(y)} tiles scored", flush=True)

    base = os.path.join(ROOT, "data", "biomass")
    gdirs = {g: os.path.join(base, g) for g in os.listdir(base)
             if os.path.isdir(os.path.join(base, g))}

    def rec(i):
        return dict(granule=str(meta["granule"][i]), stack=str(meta["stack"][i]),
                    foot=str(meta["foot"][i]), date=str(meta["date"][i]),
                    row=int(meta["row"][i]), col=int(meta["col"][i]),
                    aoi_pos=float(aoi_pos[i]), p=float(p[i]),
                    ratio_dB=float(X[i, ri]))

    idx = np.where(m)[0]
    pred = p[idx] >= args.thresh
    truth = y[idx]
    classes = {
        "tp": (truth & pred, "most confident, descending P"),
        "fn": (truth & ~pred, "AlphaEarth-positive the gate missed, lowest P first"),
        "fp": (~truth & pred, "gate-positive where AlphaEarth says no -- may be real "
                              "crevasses it never labelled (its recall is ~0.42)"),
        "tn": (~truth & ~pred, "correctly quiet, lowest P first"),
    }
    for nm, (sel, note) in classes.items():
        cand = idx[sel]
        if not cand.size:
            print(f"{nm}: none")
            continue
        order = np.argsort(-p[cand]) if nm in ("tp", "fp") else np.argsort(p[cand])
        pick = [rec(i) for i in cand[order[:args.n_per_class]]]
        figure(pick, gdirs, os.path.join(ROOT, "data", f"bio_chips_{nm}.png"),
               f"BIOMASS {nm.upper()}  ({len(cand)} tiles in class)  {note}\n"
               f"out-of-fold P at {block_m/1000:.0f} km blocks / "
               f"{buffer_m/1000:.0f} km buffer, threshold {args.thresh:.2f}")


if __name__ == "__main__":
    main()
