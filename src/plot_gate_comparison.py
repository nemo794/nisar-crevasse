"""Five-panel comparison on one granule: bare SAR, AlphaEarth mask, and the RF gate
trained on each label source.

    python src/plot_gate_comparison.py --granule 025_019
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as E  # noqa: E402
from map_crevasse_tiles import build_basemap, raster_for  # noqa: E402

DEFAULT_AOI = E.ROOT / "data" / "aois" / "thwaites_mosaic.vrt"
# label: (scores npz, threshold)
GATES = [
    ("human labels\n(gate_model, 950 manual)", "data/_rf_scores_025_019.npz", 0.65),
    ("AlphaEarth easy\n(755/755, far negatives)", "data/_rf_scores_025_019_aoieasy.npz", 0.702),
    ("AlphaEarth hard\n(949/949, +fringe pos, near neg)", "data/_rf_scores_025_019_aoihard.npz", 0.555),
]


def tile_grid(npz, gh, gw, ts):
    z = np.load(npz, allow_pickle=True)
    grid = np.full((gh // ts, gw // ts), np.nan, np.float32)
    grid[z["row"] // ts, z["col"] // ts] = z["prob"]
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granule", default="025_019")
    ap.add_argument("--aoi", default=str(DEFAULT_AOI))
    ap.add_argument("--downsample", type=int, default=60)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raster = raster_for(args.granule)
    if raster is None:
        raise SystemExit(f"no raster for {args.granule}")
    out = args.out or str(E.ROOT / "data" / f"gate_comparison_{args.granule}.png")
    ds, ts = args.downsample, args.tile_size

    with rasterio.open(raster) as s:
        gh, gw = s.height, s.width
    print("building basemap...")
    base = build_basemap(raster, ds)
    H, W = base.shape
    n_data = max(1, int(np.isfinite(base).sum()))

    with rasterio.open(args.aoi) as a:
        if (a.height, a.width) != (gh, gw):
            raise SystemExit("AOI is not on the granule grid")
        aoi = a.read(1, out_shape=(H, W), resampling=Resampling.average).astype(np.float32)

    panels = [("HH amplitude\n(no labels)", None, None)]
    panels.append((f"AlphaEarth mask\n{100 * (aoi > 0.25).sum() / n_data:.1f}% of data area", aoi, None))
    for name, npz, thr in GATES:
        if not Path(npz).exists():
            print(f"  missing {npz}, skipping")
            continue
        panels.append((name, None, (tile_grid(npz, gh, gw, ts), thr)))

    ext = [0, (gw // ts) * ts / ds, (gh // ts) * ts / ds, 0]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 8.4))
    for ax, (title, aoi_layer, gate) in zip(axes, panels):
        ax.imshow(base, cmap="gray", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        sub = title
        if aoi_layer is not None:
            ax.imshow(np.ma.masked_where(aoi_layer <= 0, aoi_layer), cmap="autumn",
                      vmin=0, vmax=1, alpha=0.65, interpolation="nearest")
        if gate is not None:
            grid, thr = gate
            n_sc = int(np.isfinite(grid).sum())
            n_pos = int((grid >= thr).sum())
            ax.imshow(np.ma.masked_where(~(grid >= thr), grid), cmap="autumn_r",
                      vmin=thr, vmax=1.0, alpha=0.75, extent=ext, interpolation="nearest")
            ax.set_xlim(0, W); ax.set_ylim(H, 0)
            sub += f"\nP>={thr:g}: {n_pos}/{n_sc} tiles ({100 * n_pos / max(1, n_sc):.1f}%)"
            print(f"{title.splitlines()[0]:34s} {n_pos:5d}/{n_sc} = {100 * n_pos / max(1, n_sc):5.2f}%")
        ax.set_title(sub, fontsize=11)

    fig.suptitle(f"{args.granule} — crevasse labels by source  "
                 f"({gw}x{gh} px @5 m, {ts}px tiles)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out, dpi=115, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
