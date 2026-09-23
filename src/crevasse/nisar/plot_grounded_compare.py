"""AlphaEarth labels vs the two gate variants, with the gates masked to grounded ice.

Shelf, ocean and rock tiles are dropped from the gate panels, so what remains is only
the surface AlphaEarth was willing to label. Everything shares the granule's 512-px tile
grid, as in plot_granule_layers.py.

    conda run -n nisar-gate python src/plot_grounded_compare.py --granule 025_019
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

import crevasse.nisar.gate_common as E
from crevasse.nisar.build_aoi_tile_labels import find_granule
from crevasse.nisar.plot_granule_layers import (crop, labels_to_grid, scores_to_grid,
                                 tile_grid_basemap)

GROUNDED = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granule", default="025_019")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--sar-scores", default=None)
    ap.add_argument("--prior-scores", default=None)
    ap.add_argument("--tile", type=int, default=E.TS)
    ap.add_argument("--thresh", type=float, default=None,
                    help="override the threshold stored in the score npz (counts only)")
    ap.add_argument("--min-p", type=float, default=None,
                    help="draw only tiles with P >= this, letting the basemap show "
                         "through elsewhere; also becomes the colour-scale floor")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    g = args.granule
    labels = args.labels or E.ROOT / "data" / f"tile_labels_aoi_{g}_bbox.csv"
    sar_p = args.sar_scores or E.ROOT / "data" / "maps" / f"gate_sar_{g}.npz"
    pri_p = args.prior_scores or E.ROOT / "data" / "maps" / f"gate_prior_{g}.npz"
    out = args.out or E.ROOT / "data" / "maps" / f"grounded_{g}.png"
    for p in (labels, sar_p, pri_p):
        if not Path(p).exists():
            raise SystemExit(f"missing {p}")

    ctx = np.load(E.ROOT / "data" / f"_context_masks_{g}_t{args.tile}.npz")
    shape = ctx["grounded_frac"].shape
    base, _ = tile_grid_basemap(find_granule(g), *shape, args.tile)
    lab, _ = labels_to_grid(labels, shape, args.tile)
    sar, thr_s = scores_to_grid(sar_p, shape, args.tile)
    pri, thr_p = scores_to_grid(pri_p, shape, args.tile)
    gnd = ctx["bedmap_class"] == GROUNDED

    base, lab, sar, pri, gnd = crop([base, lab, sar, pri, gnd], np.isfinite(base))
    if args.thresh is not None:
        thr_s = thr_p = args.thresh

    # Non-grounded data tiles are forced to P=0 rather than blanked, so the gate panels
    # keep the full swath footprint and the shelf reads as a decision, not as no-data.
    data = np.isfinite(sar)
    sar_g = np.where(data & ~gnd, 0.0, sar)
    pri_g = np.where(data & ~gnd, 0.0, pri)
    n_all = int(data.sum())
    n_gnd = int((data & gnd).sum())

    lr = np.flatnonzero(np.isfinite(lab).any(1))
    lc = np.flatnonzero(np.isfinite(lab).any(0))

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(base, cmap="gray", interpolation="nearest", vmin=0, vmax=1)
        ax.add_patch(Rectangle((lc[0] - 0.5, lr[0] - 0.5), lc[-1] - lc[0] + 1,
                               lr[-1] - lr[0] + 1, fill=False, edgecolor="lime",
                               linewidth=1.2, linestyle="--"))

    ax = axes[0]
    ax.imshow(np.ma.masked_invalid(lab), cmap=ListedColormap(["#2c7bb6", "#d7191c"]),
              vmin=0, vmax=1, interpolation="nearest")
    npos, nneg = int((lab == 1).sum()), int((lab == 0).sum())
    ax.set_title(f"AlphaEarth tile labels — {g}\n{npos} crevasse / {nneg} none"
                 f"\n(green dashes = label extent)")
    ax.legend(handles=[Patch(color="#d7191c", label="crevasse"),
                       Patch(color="#2c7bb6", label="none")],
              loc="lower left", fontsize=8, framealpha=0.9)

    lo = args.min_p if args.min_p is not None else 0.0
    for ax, (arr, thr, name) in zip(
            axes[1:], ((sar_g, thr_s, "gate SAR only"),
                       (pri_g, thr_p, "gate SAR x context prior"))):
        shown = np.where(arr >= lo, arr, np.nan) if args.min_p is not None else arr
        im = ax.imshow(np.ma.masked_invalid(shown), cmap="inferno", vmin=lo, vmax=1,
                       interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
        cb.set_label("P(crevasse)")
        cut = (f"drawn: P >= {lo:.2f}" if args.min_p is not None
               else "non-grounded forced to P=0")
        ax.set_title(f"{name} — {cut}\n"
                     f"{int((arr >= thr).sum())} / {n_all} data tiles >= {thr:.3f}"
                     f"  (median P={np.nanmedian(arr):.2f})")

    fig.suptitle(f"Granule {g}: gate output with non-grounded ice zeroed, vs the "
                 f"AlphaEarth labels — {n_all - n_gnd} of {n_all} data tiles zeroed",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
