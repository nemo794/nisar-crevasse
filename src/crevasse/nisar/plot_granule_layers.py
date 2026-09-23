"""One figure, six co-registered layers for a granule, to compare gate variants against
the inputs that could explain them.

Everything is rendered on the granule's 512-px tile grid, so a cell in any panel is the
same patch of ground in every other panel:

    SAR amplitude | AlphaEarth tile labels | Bedmap3 surface class
    ITS_LIVE speed | gate SAR P(crevasse)   | gate SAR x context prior

Needs, for the granule: the .tif, a label CSV from build_aoi_tile_labels.py, a
_context_masks_*.npz from build_context_masks.py, and two --save-scores .npz files
from map_crevasse_tiles.py.

    conda run -n nisar-gate python src/plot_granule_layers.py --granule 025_019
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from rasterio.windows import Window

import crevasse.nisar.gate_common as E
from crevasse.nisar.build_aoi_tile_labels import find_granule
from crevasse.nisar.build_context_masks import BEDMAP_CLASSES

BEDMAP_COLORS = {0: "#1f4e79", 1: "#d9d9d9", 2: "#8fd0e8",
                 3: "#4fa3d1", 4: "#8b5a2b"}


def tile_grid_basemap(path, nrow, ncol, tile):
    """Mean log-amplitude per tile, so the backdrop shares the panel grid exactly."""
    with rasterio.open(path) as src:
        a = src.read(1, window=Window(0, 0, ncol * tile, nrow * tile),
                     out_shape=(nrow, ncol),
                     resampling=rasterio.enums.Resampling.average).astype(np.float32)
    lg = np.full(a.shape, np.nan, np.float32)
    m = a > 0
    lg[m] = np.log10(a[m])
    if m.any():
        p2, p98 = np.nanpercentile(lg, [2, 98])
        lg = np.clip((lg - p2) / (p98 - p2 + 1e-9), 0, 1)
    return lg, m


def scores_to_grid(npz_path, shape, tile):
    z = np.load(npz_path, allow_pickle=True)
    g = np.full(shape, np.nan, np.float32)
    stride = int(z["stride"]) if "stride" in z else tile
    for r, c, p in zip(z["row"], z["col"], z["prob"]):
        i, j = int(r) // tile, int(c) // tile
        n = max(1, stride // tile)     # a tile may span several grid cells at 2.5 m
        if i < shape[0] and j < shape[1]:
            g[i:i + n, j:j + n] = p
    return g, float(z["gate_thresh"])


def labels_to_grid(csv_path, shape, tile):
    d = pd.read_csv(csv_path)
    g = np.full(shape, np.nan, np.float32)
    for r, c, lab in zip(d.row, d.col, d.label):
        i, j = int(r) // tile, int(c) // tile
        if i < shape[0] and j < shape[1]:
            g[i, j] = 1.0 if lab == "crevasse" else 0.0
    return g, d


def crop(arrays, valid, pad=2):
    r = np.flatnonzero(valid.any(1))
    c = np.flatnonzero(valid.any(0))
    r0, r1 = max(0, r[0] - pad), min(valid.shape[0], r[-1] + 1 + pad)
    c0, c1 = max(0, c[0] - pad), min(valid.shape[1], c[-1] + 1 + pad)
    return [a[r0:r1, c0:c1] for a in arrays]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granule", default="025_019")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--sar-scores", default=None)
    ap.add_argument("--prior-scores", default=None)
    ap.add_argument("--tile", type=int, default=E.TS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    g = args.granule
    labels = args.labels or E.ROOT / "data" / f"tile_labels_aoi_{g}_bbox.csv"
    sar_p = args.sar_scores or E.ROOT / "data" / "maps" / f"gate_sar_{g}.npz"
    pri_p = args.prior_scores or E.ROOT / "data" / "maps" / f"gate_prior_{g}.npz"
    out = args.out or E.ROOT / "data" / "maps" / f"layers_{g}.png"
    for p in (labels, sar_p, pri_p):
        if not Path(p).exists():
            raise SystemExit(f"missing {p}")

    ctx = np.load(E.ROOT / "data" / f"_context_masks_{g}_t{args.tile}.npz")
    shape = ctx["grounded_frac"].shape
    base, datamask = tile_grid_basemap(find_granule(g), *shape, args.tile)
    lab, ldf = labels_to_grid(labels, shape, args.tile)
    sar, thr_s = scores_to_grid(sar_p, shape, args.tile)
    pri, thr_p = scores_to_grid(pri_p, shape, args.tile)
    vel = np.where(ctx["vel_mean"] > 0, ctx["vel_mean"], np.nan)
    bcl = ctx["bedmap_class"].astype(float)

    base, lab, sar, pri, vel, bcl = crop([base, lab, sar, pri, vel, bcl],
                                         np.isfinite(base))
    # The context rasters cover the whole tile grid; the SAR does not. Masking them to
    # the swath keeps every panel describing the same ground.
    swath = np.isfinite(base)
    vel = np.where(swath, vel, np.nan)
    bcl = np.where(swath, bcl, np.nan)
    npos, nneg = int((lab == 1).sum()), int((lab == 0).sum())

    lr = np.flatnonzero(np.isfinite(lab).any(1))
    lc = np.flatnonzero(np.isfinite(lab).any(0))

    def label_box(ax):
        """Outline where AlphaEarth was actually evaluated -- a small part of the scene."""
        ax.add_patch(Rectangle((lc[0] - 0.5, lr[0] - 0.5), lc[-1] - lc[0] + 1,
                               lr[-1] - lr[0] + 1, fill=False, edgecolor="lime",
                               linewidth=1.2, linestyle="--"))

    fig, axes = plt.subplots(2, 3, figsize=(19, 13))
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    def backdrop(ax):
        ax.imshow(base, cmap="gray", interpolation="nearest", vmin=0, vmax=1)

    # --- SAR amplitude
    ax = axes[0, 0]
    backdrop(ax)
    label_box(ax)
    ax.set_title(f"NISAR HH amplitude — {g}\n(log, per-tile mean, "
                 f"{args.tile}px = {args.tile * 5 / 1000:.2f} km cells)\n"
                 f"green dashes = extent of the AlphaEarth label set")

    # --- AlphaEarth labels
    ax = axes[0, 1]
    backdrop(ax)
    ax.imshow(np.ma.masked_invalid(lab), cmap=ListedColormap(["#2c7bb6", "#d7191c"]),
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"AlphaEarth tile labels\n{npos} crevasse / {nneg} none "
                 f"(blank = ambiguous or too little data)")
    ax.legend(handles=[Patch(color="#d7191c", label="crevasse"),
                       Patch(color="#2c7bb6", label="none")],
              loc="lower left", fontsize=8, framealpha=0.9)

    # --- Bedmap3 class
    ax = axes[0, 2]
    present = [k for k in sorted(BEDMAP_CLASSES) if (bcl == k).any()]
    ax.imshow(np.ma.masked_invalid(bcl),
              cmap=ListedColormap([BEDMAP_COLORS[k] for k in sorted(BEDMAP_CLASSES)]),
              norm=BoundaryNorm(np.arange(-0.5, len(BEDMAP_CLASSES) + 0.5),
                                len(BEDMAP_CLASSES)),
              interpolation="nearest")
    ax.set_title("Bedmap3 surface class (500 m)")
    ax.legend(handles=[Patch(color=BEDMAP_COLORS[k], label=BEDMAP_CLASSES[k])
                       for k in present],
              loc="lower left", fontsize=8, framealpha=0.9)

    # --- ITS_LIVE speed
    ax = axes[1, 0]
    backdrop(ax)
    im = ax.imshow(np.ma.masked_invalid(np.log10(vel)), cmap="viridis",
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
    cb.set_label("log10 speed (m/yr)")
    with np.errstate(all="ignore"):
        ax.set_title(f"ITS_LIVE surface speed (120 m)\nmedian "
                     f"{np.nanmedian(vel):.0f} m/yr, max {np.nanmax(vel):.0f}")

    # --- the two gates, identical colour scale so they are comparable
    for ax, (arr, thr, name) in zip(
            (axes[1, 1], axes[1, 2]),
            ((sar, thr_s, "gate SAR only"),
             (pri, thr_p, "gate SAR x context prior"))):
        backdrop(ax)
        im = ax.imshow(np.ma.masked_invalid(arr), cmap="inferno", vmin=0, vmax=1,
                       interpolation="nearest")
        label_box(ax)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
        cb.set_label("P(crevasse)")
        n = int(np.isfinite(arr).sum())
        ax.set_title(f"{name}\n{int((arr >= thr).sum())} / {n} tiles >= {thr:.3f}"
                     f"  (median P={np.nanmedian(arr):.2f})")

    fig.suptitle(f"Granule {g}: gate output vs the layers that could explain it "
                 f"— all panels share the {args.tile}px tile grid", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
