"""Plot a granule run: the raster, the gate's per-tile probabilities, and a grid of
individual tiles with their U-Net segmentation, at full resolution.

    python src/plot_granule.py --granule <dir> --max-tiles 200 --out data/plot_granule.png
    python src/plot_granule.py --granule <dir> --check   # the same sample as P4, for eyeballing it

Two figures:

  1. **the overview** -- two panels sharing one downsampled HH basemap (same
     percentile-stretch convention as `bio_plot_eval._display_hh`, imported directly
     rather than duplicated): the raw raster, and the raster with the gate's per-tile
     P(crevasse) drawn as colored rectangles (flagged tiles filled by a colormap,
     unflagged tiles outlined only) -- same visual convention as
     `nisar-crevasse-pipeline/src/plot_granule.py`.
  2. **a tile grid** -- several individual tiles at full `512x512` resolution, each with
     its U-Net per-pixel probability overlaid on the raw HH. A swath-wide segmentation
     panel was tried on NISAR and dropped for the same reason it would fail here: at
     basemap scale each tile is only a few pixels across, too small for crevasse-shaped
     structure to be visible. `--tile-select random` (default) samples from the
     gate-flagged tiles so the grid isn't just the single most confident one;
     `--tile-select best` instead takes the highest mean-probability tiles.

This is a viewer, not a control -- it exists to eyeball a run, not to be pinned. It reuses
`run_granule.tile_granule` for tiling and `BiomassCrevassePipeline` for scoring, so a
figure it draws came out of exactly the code path the P1-P4 controls in
`docs/CONTRIBUTING.md` check.
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import rasterio
from rasterio.enums import Resampling

from crevasse.biomass.pipeline_predict import BiomassCrevassePipeline
from crevasse.biomass.run_granule import tile_granule, POLS, T
from crevasse.biomass.bio_plot_eval import _display_hh   # same stretch as the eval figures


def build_basemap(granule_dir, downsample):
    """Granule directory -> downsampled, percentile-stretched HH basemap for the
    overview figure. There's no existing ready-made basemap builder for BIOMASS the way
    `map_crevasse_tiles.build_basemap` is for NISAR, so this is new, but deliberately
    thin -- one band, one stretch, matching `_display_hh`'s own convention exactly so a
    zoomed tile in the grid figure reads consistently with this overview."""
    matches = glob.glob(os.path.join(granule_dir, "*_HH_intensity.tif"))
    if not matches:
        raise SystemExit(f"{granule_dir}: no *_HH_intensity.tif found")
    with rasterio.open(matches[0]) as src:
        h, w = src.height // downsample, src.width // downsample
        a = src.read(1, out_shape=(h, w), resampling=Resampling.average).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        db = 10 * np.log10(np.where(a > 0, a, np.nan))
    return _display_hh(db[None])  # (1, H, W) -> _display_hh reads channel 0


def select_tiles(unet_prob, gate_flag, n, how, seed):
    """Pick up to `n` gate-flagged tile indices for the grid figure. Same rationale as
    the NISAR sibling: `how="random"` (default) avoids showing only the top-confidence
    tiles, which alone would give an overly rosy impression of the run."""
    idx = np.flatnonzero(gate_flag)
    if len(idx) == 0:
        return np.array([], dtype=int)
    if how == "best":
        means = np.nanmean(unet_prob[idx], axis=(1, 2))
        order = np.argsort(-means)
        return idx[order[:n]]
    rng = np.random.default_rng(seed)
    return rng.choice(idx, size=min(n, len(idx)), replace=False)


def _overlay_rgba(values, cmap_name, vmin, vmax, alpha):
    cmap = plt.get_cmap(cmap_name)
    norm = np.clip((values - vmin) / (vmax - vmin + 1e-12), 0, 1)
    rgba = cmap(norm)
    rgba[..., 3] = np.where(np.isfinite(values), alpha, 0.0)
    return rgba


def plot_tile_grid(positions, inten, unet_prob, gate_prob, idx, out_path, ncols=3,
                    vis_thresh=0.65):
    """A grid of individual tiles at full `512x512` resolution, HH background + U-Net
    per-pixel probability overlaid. `vis_thresh` is display-only, same caveat as the
    NISAR sibling: it masks pixels below it to fully transparent so real ridge structure
    is visible over the low-level speckle response, and does NOT touch `unet_prob`."""
    if len(idx) == 0:
        print("No tiles to plot in the grid (nothing gate-flagged); skipping.")
        return
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for k, ax in enumerate(axes.flat):
        if k >= len(idx):
            ax.axis("off")
            continue
        i = idx[k]
        row, col = positions[i]
        p = unet_prob[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            hh_db = 10 * np.log10(np.where(inten[i, 0] > 0, inten[i, 0], np.nan))
        bg = _display_hh(hh_db[None])
        ax.imshow(bg, cmap="gray", interpolation="nearest")
        shown = np.where(p >= vis_thresh, p, np.nan)
        rgba = _overlay_rgba(shown, "hot", vmin=vis_thresh, vmax=1.0, alpha=0.85)
        ax.imshow(rgba, interpolation="nearest")
        frac_above = float(np.mean(p >= vis_thresh))
        ax.set_title(f"(r{row}, c{col})  gate P={gate_prob[i]:.2f}  "
                      f"mean P={np.nanmean(p):.2f}  "
                      f"{frac_above * 100:.1f}% >= {vis_thresh:g}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    sm = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(vis_thresh, 1.0))
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01, shrink=0.6
                 ).set_label(f"U-Net P(crevasse), pixels >= {vis_thresh:g}")
    fig.suptitle(f"{len(idx)} gate-flagged tiles — HH + U-Net segmentation "
                 f"(display threshold {vis_thresh:g})")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--granule", required=True)
    p.add_argument("--max-tiles", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check", action="store_true",
                   help="use the same --max-tiles 200 --seed 0 sample as control P4")
    p.add_argument("--gate-method", choices=["rf", "cnn"], default="rf")
    p.add_argument("--gate-thresh", type=float, default=0.65)
    p.add_argument("--basemap-downsample", type=int, default=20,
                   help="passed to build_basemap; lower = sharper background but "
                        "slower and more memory")
    p.add_argument("--out", default=None)
    p.add_argument("--n-tiles", type=int, default=6,
                   help="how many individual tiles to show in the tile-grid figure. 0 "
                        "skips it.")
    p.add_argument("--tile-select", choices=["random", "best"], default="random")
    p.add_argument("--tile-seed", type=int, default=None,
                   help="seed for --tile-select random; defaults to --seed.")
    p.add_argument("--tiles-out", default=None,
                   help="path for the tile-grid figure; defaults to <out>_tiles.png")
    p.add_argument("--unet-vis-thresh", type=float, default=0.65)
    args = p.parse_args(argv)

    max_tiles, seed = args.max_tiles, args.seed
    if args.check:
        max_tiles, seed = 200, 0

    positions, x_m, y_m, inten = tile_granule(args.granule, max_tiles=max_tiles, seed=seed)
    print(f"Tiled {len(positions)} valid tiles")

    pipe = BiomassCrevassePipeline(gate_method=args.gate_method, gate_thresh=args.gate_thresh)
    out = pipe.run(inten, x_m=x_m, y_m=y_m)
    n_flagged = int(out.gate_flag.sum())
    print(f"Gate flagged {n_flagged}/{len(positions)} "
          f"(method={args.gate_method}, thresh={args.gate_thresh})")

    ds = args.basemap_downsample
    base = build_basemap(args.granule, ds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7 * base.shape[0] / max(base.shape[1], 1)))

    axes[0].imshow(base, cmap="gray", interpolation="nearest")
    axes[0].set_title(f"Raster — {Path(args.granule).name}", fontsize=10)

    axes[1].imshow(base, cmap="gray", interpolation="nearest")
    rows = np.array([r for r, _ in positions])
    cols = np.array([c for _, c in positions])
    scored = np.isfinite(out.gate_prob)
    flagged = scored & out.gate_flag
    unflagged = scored & ~out.gate_flag
    tw = T / ds
    if unflagged.any():
        axes[1].add_collection(PatchCollection(
            [Rectangle((c / ds, r / ds), tw, tw)
             for r, c in zip(rows[unflagged], cols[unflagged])],
            facecolor="none", edgecolor="cyan", linewidth=0.3, alpha=0.4))
    if flagged.any():
        pc = PatchCollection(
            [Rectangle((c / ds, r / ds), tw, tw)
             for r, c in zip(rows[flagged], cols[flagged])],
            cmap=plt.get_cmap("autumn_r"),
            norm=plt.Normalize(vmin=args.gate_thresh, vmax=1.0),
            alpha=0.6, edgecolor="red", linewidth=0.3)
        pc.set_array(out.gate_prob[flagged])
        axes[1].add_collection(pc)
        fig.colorbar(pc, ax=axes[1], fraction=0.03, pad=0.01).set_label("gate P(crevasse)")
    axes[1].set_title(f"Gate — {n_flagged}/{int(scored.sum())} tiles flagged", fontsize=10)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()

    out_path = args.out or f"data/plot_{Path(args.granule).name}.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    if args.n_tiles > 0:
        tile_seed = args.tile_seed if args.tile_seed is not None else args.seed
        idx = select_tiles(out.unet_prob, out.gate_flag, args.n_tiles,
                            args.tile_select, tile_seed)
        tiles_out = args.tiles_out or (str(Path(out_path).with_suffix("")) + "_tiles.png")
        plot_tile_grid(positions, inten, out.unet_prob, out.gate_prob, idx, tiles_out,
                        vis_thresh=args.unet_vis_thresh)


if __name__ == "__main__":
    main()
