"""Pack pseudo-label results (see label_io.py) + their granule into one self-contained
training shard.

WHY. The label files are 1.8-2.4 GB each (npz; formerly pickle) and the granules they
index are 4.7-9.7 GB, so shipping a training set the obvious way is ~19 GB for two
granules -- of which the labelled tiles are only about 0.3 GB of actual pixels. This
reads each labelled tile once, stores it beside its target, and drops everything
neither the loader nor the training loop touches. Result is ~0.55 GB per 1000 tiles, no
GeoTIFF and no rasterio needed on the training machine.

WHAT IS DROPPED, and why it is safe. `binary_mask` (the superseded pre-2026-09-09
label, half the original pickle's bulk -- label_io.py's npz format never carried it in
the first place), `ridge_stats`, `texture_coverage`, `ridge_coverage`. The loader reads
only `edge_mask`, `row`, `col`; the coverage filter reads only `edge_coverage`.
`orient_conc` and `gate_prob` are kept because they are the two tile gates' scores and
are wanted for post-hoc stratification of results.

FLOAT16 IS A DELIBERATE CHOICE, NOT A COMPROMISE. The target lives in [0,1] where
float16 resolves ~5e-4 -- three orders finer than the 0.2 coverage threshold and far
finer than BCE cares about. Amplitude is stored raw (not preprocessed) so the shard
stays interchangeable with the raster path: preprocess_sar_tile runs in the loader
either way. Verify with check_labels.py, which compares shard against the label file.
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from crevasse.common.tile_utils_v2 import TileGenerator
from crevasse.nisar.label_io import load_labels


def build(results_npz: str, raster_path: str, out_path: str, tile_size: int = 512):
    results = load_labels(results_npz)
    print(f"{len(results)} labelled tiles in {Path(results_npz).name}")

    shapes = {r["edge_mask"].shape for r in results}
    if shapes != {(tile_size, tile_size)}:
        raise SystemExit(
            f"label masks are {shapes}, expected {{({tile_size}, {tile_size})}}. A mask "
            f"that does not match the tile would be silently misaligned by the "
            f"augmentation pipeline.")

    gen = TileGenerator(raster_path, tile_size=tile_size)
    n = len(results)
    images = np.zeros((n, tile_size, tile_size), dtype=np.float16)
    targets = np.zeros((n, tile_size, tile_size), dtype=np.float16)
    positions = np.zeros((n, 2), dtype=np.int64)

    for i, r in enumerate(tqdm(results, desc="Packing")):
        tile, _ = gen.read_tile(r["row"], r["col"])
        if tile.shape != (tile_size, tile_size):
            raise SystemExit(f"tile (r{r['row']}, c{r['col']}) read back as {tile.shape}")
        images[i] = tile.astype(np.float16)
        targets[i] = r["edge_mask"].astype(np.float16)
        positions[i] = (r["row"], r["col"])

    # Recomputed from the stored mask rather than copied, so the shard's own filter
    # threshold is consistent with the array it actually ships.
    coverage = (targets.astype(np.float32) > 0.2).mean(axis=(1, 2))

    meta = dict(
        images=images,
        targets=targets,
        positions=positions,
        coverage=coverage.astype(np.float32),
        orient_conc=np.array([float(r.get("orient_conc", np.nan)) for r in results],
                             dtype=np.float32),
        gate_prob=np.array([np.nan if r.get("gate_prob") is None else float(r["gate_prob"])
                            for r in results], dtype=np.float32),
        tile_size=np.int32(tile_size),
        soft_rect_t=np.float32(results[0].get("soft_rect_t") or np.nan),
        # The scale set travels with the shard: soft_rect_t alone cannot distinguish
        # the 2026-09-10 mean{4,2,1} rule (nan) from the old binary mask (also nan).
        ridge_factors=np.array(results[0].get("ridge_factors") or (4,), dtype=np.int32),
        source_npz=np.array(Path(results_npz).name),
        source_raster=np.array(Path(raster_path).name),
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **meta)
    mb = Path(out_path).stat().st_size / 1e6
    print(f"\nWrote {out_path}  ({mb:.0f} MB, {n} tiles)")
    print(f"  coverage>0.2: mean {coverage.mean()*100:.2f}%  "
          f"p10 {np.percentile(coverage,10)*100:.2f}%  "
          f"p90 {np.percentile(coverage,90)*100:.2f}%")
    print(f"  target range [{targets.min():.3f}, {targets.max():.3f}]  "
          f"soft_rect_t {float(meta['soft_rect_t'])}  "
          f"ridge_factors {tuple(int(f) for f in meta['ridge_factors'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-npz", required=True)
    ap.add_argument("--raster", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile-size", type=int, default=512)
    args = ap.parse_args()
    build(args.results_npz, args.raster, args.out, args.tile_size)


if __name__ == "__main__":
    main()
