"""Join surface-type context onto a shard's tiles, as a sidecar the cluster can read.

WHY THIS EXISTS. The 2026-09-10 evaluation was not measuring what it claimed to.
Bedmap3 says 68.9% of the val band was floating shelf or sea ice, and both runs scored
noticeably better there than on grounded ice (pooled IoU 0.586 vs 0.550 for f4, 0.539 vs
0.424 for mean{4,2,1}). Shelf rifts and grounded-ice crevasses are different physical
processes, so a headline number dominated by rifts is not a crevasse-detection result.

WHY A SIDECAR RATHER THAN A SHARD FIELD. Bedmap3 and ITS_LIVE live in the sibling
`biomass` repo behind rasterio; the training machine has neither. This precomputes the
per-tile join into a ~20 KB npz that rsyncs beside the 1 GB shard. It also leaves the
existing shards byte-identical, so every number already quoted against them stays
reproducible.

THE JOIN IS APPROXIMATE, BY CONSTRUCTION. Labelled tiles anchor to the swath corner
(025_019 positions are 152 mod 512 in row, 4 in col) while the context grid anchors to
raster origin 0, so an exact (row, col) join matches nothing. This does an area-weighted
mean of the context grid over the cells each tile straddles -- so a tile at a grounding
line gets a fraction, not a class, which is why the consumers threshold rather than test
equality.
"""
import argparse
from pathlib import Path

import numpy as np


def overlap_mean(positions: np.ndarray, grid: np.ndarray, tile_size: int) -> np.ndarray:
    """Area-weighted mean of `grid` over the grid cells each tile straddles."""
    H, W = grid.shape
    out = np.full(len(positions), np.nan)
    for i, (r, c) in enumerate(positions):
        num = den = 0.0
        for gr in range(r // tile_size, (r + tile_size - 1) // tile_size + 1):
            for gc in range(c // tile_size, (c + tile_size - 1) // tile_size + 1):
                if not (0 <= gr < H and 0 <= gc < W):
                    continue
                oy = min(r + tile_size, (gr + 1) * tile_size) - max(r, gr * tile_size)
                ox = min(c + tile_size, (gc + 1) * tile_size) - max(c, gc * tile_size)
                if oy <= 0 or ox <= 0:
                    continue
                w = oy * ox
                num += w * float(grid[gr, gc])
                den += w
        out[i] = num / den if den else np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", required=True, help="shard whose positions define the tiles")
    ap.add_argument("--context-masks", required=True,
                    help="biomass data/_context_masks_<granule>_t512.npz "
                         "(built by code/build_context_masks.py)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.shard)
    positions = z["positions"].astype(np.int64)
    tile_size = int(z["tile_size"])
    ctx = np.load(args.context_masks, allow_pickle=True)

    print(f"{len(positions)} tiles from {Path(args.shard).name}, anchored at "
          f"{positions[0,0] % tile_size} mod {tile_size} (row), "
          f"{positions[0,1] % tile_size} (col)")
    print(f"context grid {ctx['grounded_frac'].shape} from "
          f"{Path(args.context_masks).name}")

    grounded = overlap_mean(positions, ctx["grounded_frac"], tile_size)
    # vel_mean carries NaN over ocean; zero-filling keeps the join finite and a tile that
    # is partly ocean should read slow-and-uncertain rather than propagate NaN into a
    # stratifier. Nothing gates on velocity -- it is reported only.
    vel = overlap_mean(positions, np.nan_to_num(ctx["vel_mean"], nan=0.0), tile_size)

    if not np.isfinite(grounded).all():
        n = int((~np.isfinite(grounded)).sum())
        raise SystemExit(f"{n} tiles fell entirely outside the context grid; the granule "
                         f"and the context masks do not match")

    np.savez(args.out, positions=positions, grounded_frac=grounded.astype(np.float32),
             vel_mean=vel.astype(np.float32), tile_size=np.int32(tile_size),
             source_shard=np.array(Path(args.shard).name),
             source_context=np.array(Path(args.context_masks).name))
    print(f"\nWrote {args.out}")
    for lo, hi, name in ((0.0, 0.05, "shelf / sea ice"), (0.05, 0.5, "mixed"),
                         (0.5, 0.95, "mostly grounded"), (0.95, 1.01, "grounded")):
        m = (grounded >= lo) & (grounded < hi)
        print(f"  grounded_frac [{lo:.2f}, {hi:.2f})  {name:16s} {int(m.sum()):5d} tiles"
              + (f"   mean {np.nanmean(vel[m]):6.0f} m/yr" if m.any() else ""))


if __name__ == "__main__":
    main()
