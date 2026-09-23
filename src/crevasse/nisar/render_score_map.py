"""Re-render a crevasse gate map from a saved --save-scores .npz at a new threshold.

    conda run -n nisar-gate python src/render_score_map.py \
        data/maps/gate_sar_025_091.npz --gate-thresh 0.65

Gate probabilities do not depend on the threshold -- it only decides which tiles are
filled at render time -- so comparing operating points does not need a rescore. Scoring
a granule costs minutes; this costs seconds.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from crevasse.nisar.map_crevasse_tiles import ROOT, raster_for, build_basemap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", help="an .npz written by map_crevasse_tiles.py --save-scores")
    ap.add_argument("--gate-thresh", type=float, required=True)
    ap.add_argument("--downsample", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.scores)
    granule = str(d["granule"])
    row, col, prob = d["row"], d["col"], d["prob"]
    stride = int(d["stride"])
    grounded = d["grounded"] if "grounded" in d else np.ones(len(prob), bool)
    grounded_only = bool(d["grounded_only"]) if "grounded_only" in d else False

    raster_path = raster_for(granule)
    if raster_path is None:
        raise SystemExit(f"no raster found for granule {granule}")

    keep = prob >= args.gate_thresh
    pos = keep & grounded if grounded_only else keep
    excl = keep & ~grounded if grounded_only else np.zeros(len(prob), bool)
    print(f"{granule}: {len(prob)} scored, {int(pos.sum())} positive "
          f"(P>={args.gate_thresh}, was {d['gate_thresh']} at scoring time)")

    tag = f"{args.gate_thresh:.2f}".split(".")[1]
    out = args.out or str(Path(args.scores).with_suffix("")) + f"_p{tag}.png"

    base = build_basemap(raster_path, args.downsample)
    ds = args.downsample
    tw = stride / ds

    fig, ax = plt.subplots(figsize=(14, 14 * base.shape[0] / max(base.shape[1], 1)))
    ax.imshow(base, cmap="gray", interpolation="nearest")
    ax.add_collection(PatchCollection(
        [Rectangle((c / ds, r / ds), tw, tw) for r, c in zip(row, col)],
        facecolor="none", edgecolor="cyan", linewidth=0.15, alpha=0.25))
    if excl.any():
        ax.add_collection(PatchCollection(
            [Rectangle((c / ds, r / ds), tw, tw) for r, c in zip(row[excl], col[excl])],
            facecolor="#5b7c99", edgecolor="none", alpha=0.55))
    if pos.any():
        pc = PatchCollection(
            [Rectangle((c / ds, r / ds), tw, tw) for r, c in zip(row[pos], col[pos])],
            cmap=plt.get_cmap("autumn_r"),
            norm=plt.Normalize(vmin=args.gate_thresh, vmax=1.0),
            alpha=0.6, edgecolor="red", linewidth=0.2)
        pc.set_array(prob[pos])
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, fraction=0.03, pad=0.01).set_label("P(crevasse)")

    ax.set_title(f"Crevasse gate map — {granule}\n"
                 f"{int(pos.sum())} positive / {len(prob)} data tiles "
                 f"(P>={args.gate_thresh}, {str(d['gate_model'])})")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
