"""Export a full granule gate+segmentation run to two Cloud-Optimized GeoTIFFs (COG)
registered on the source raster's own grid.

    python src/export_geotiff.py --granule <path> --out-dir data/export_025_019
    # -> data/export_025_019/gate_prob.tif     one band, float32, NaN nodata, COG
    # -> data/export_025_019/unet_prob.tif     one band, float32, NaN nodata, COG

Both outputs share the source raster's CRS, transform, width and height EXACTLY, so they
open lined up with the granule itself in QGIS / rasterio / any GIS tool -- no separate
transform bookkeeping needed on the reading side.

Written via GDAL's COG driver: 512px internal blocksize, DEFLATE/predictor=3 (matches a
collaborator's specified GDAL creation options -- BLOCKSIZE=512, COMPRESS=DEFLATE,
LEVEL=1, PREDICTOR=3, OVERVIEW_RESAMPLING=AVERAGE, NUM_THREADS=ALL_CPUS, BIGTIFF=YES),
with overviews and embedded per-band statistics (`-stats`'s effect, via
`dst.stats(approx=False)` right before close -- not automatic, has to be called
explicitly). GDAL's COG driver supports the same incremental windowed `write()` calls
this script always used; verified 2026-09-22 that this is a pure container-format
change, not a two-step "plain GeoTIFF then translate" pipeline.

    gate_prob.tif   each tile's SCALAR gate probability, broadcast flat over its whole
                    step x step footprint (the gate has no notion of "where inside the
                    tile"). NaN for tiles `read_amp` dropped (<50% valid) but still
                    inside the scanned swath; left at GDAL's default fill (typically 0,
                    NOT NaN -- see "Known limitation" below) outside the scanned swath
                    entirely, since nothing in this script ever touches that area.
    unet_prob.tif   the U-Net's real per-pixel probability, written only for tiles the
                    gate flagged. NaN everywhere else within the scanned swath (gate
                    said no, or `read_amp` dropped it); same GDAL-default caveat outside
                    the swath as `gate_prob.tif`.

Streams tile-by-tile via windowed writes -- nothing in this script holds more than
`--batch-size` tiles in memory at once, so it runs on a laptop regardless of granule
size. The tradeoff is wall-clock time, not memory: a full granule is thousands of tile
positions (9236 for 025_019) and most of the time is the U-Net forward pass on CPU.

**THIS IS SLOW ON A FULL GRANULE** -- budget tens of minutes to hours depending on the
machine and how many tiles the gate flags. Use --max-tiles for a quick correctness check
(a handful of tiles, seconds) before ever starting a full run. `UNetSegmenter` picks up
CUDA automatically if it's there, so a GPU machine is the real fix, not a smaller batch.

Known limitation: unwritten GeoTIFF blocks are not guaranteed to read back as the
declared NaN nodata value -- GDAL fills a block a driver never wrote with 0 by default.
This script explicitly writes NaN for every candidate position it visits (scored or
dropped), so the only genuinely ambiguous area is outside the scanned swath entirely,
which this script never visits. A reader should still treat 0 there the same as NaN.

`--crop-to-scanned` closes that gap instead of just documenting it: the output shrinks
to the bounding box of the positions actually processed this run (still on the source
grid, so it opens lined up the same way, just over a smaller extent), and every
step-aligned grid cell inside that box that was never a candidate at all also gets an
explicit NaN write. Off by default -- the full-source-extent output stays available for
callers that need exact pixel alignment with the untouched source raster.

`--edge-margin` fixes a real, separate artifact: independent non-overlapping tile
inference scores each tile with no context beyond its own edge, which shows up as a
faint but real grid of dimmer U-Net probability running along the exact tile boundaries
(confirmed 2026-09-23 on a real BIOMASS export; the same independent-tile inference
applies here too). `--edge-margin N` rescores on an overlapping, `(step - 2*N)`-pixel-
stride grid instead, and keeps only each tile's center region (the outer N-pixel ring is
discarded and covered by the *next* tile's center instead, except at the true boundary
of the scanned swath, where it's kept since nothing follows it). Costs roughly
`(step / (step - 2*N))^2` times more U-Net forward passes -- not free, but the seam is
gone. Not currently combinable with `--crop-to-scanned`. Off by default.
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

import crevasse.nisar.train_gate_classifier as T
from crevasse.nisar.find_data_swath import find_data_bounds, get_valid_tile_positions
from crevasse.nisar.pipeline_predict import CrevassePipeline
from crevasse.common.geotiff_crop import (crop_window, crop_transform, grid_gap_positions,
                                           overlap_positions, center_crop)


def _new_output(path, profile, step):
    prof = profile.copy()
    for k in ("tiled", "blockxsize", "blockysize"):
        prof.pop(k, None)
    prof.update(driver="COG", count=1, dtype="float32", nodata=np.nan,
                compress="DEFLATE", level=1, predictor=3, blocksize=min(step, 512),
                overview_resampling="average", num_threads="ALL_CPUS", bigtiff="YES")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return rasterio.open(path, "w", **prof)


def export_geotiffs(granule_path, out_dir, max_tiles=None, batch_size=16,
                     gate_thresh=None, crop_to_scanned=False, edge_margin=0):
    """Stream a full (or capped) granule run to `gate_prob.tif` and `unet_prob.tif`
    under `out_dir`, both lined up with `granule_path`'s own grid. Returns the two
    output paths. See the module docstring for the write semantics, and for what
    `crop_to_scanned`/`edge_margin` change."""
    if crop_to_scanned and edge_margin:
        raise ValueError("--crop-to-scanned and --edge-margin are not supported together yet")
    out_dir = Path(out_dir)

    with rasterio.open(granule_path) as src:
        profile = src.profile
        scale = T.tile_scale(src)
        step = T.E.TS * scale

        bounds, _ = find_data_bounds(str(granule_path), downsample=100)
        if bounds is None:
            raise SystemExit(f"{granule_path}: no data found in raster")
        positions = get_valid_tile_positions(bounds, tile_size=step)

        stride = step
        area_row0 = area_col0 = 0
        if edge_margin:
            stride = step - 2 * edge_margin
            if stride <= 0:
                raise ValueError(f"--edge-margin {edge_margin} too large for tile size {step}")
            bb = crop_window(positions, step, src.width, src.height)
            area_row0, area_col0 = bb.row_off, bb.col_off
            positions = overlap_positions(bb.row_off, bb.row_off + bb.height,
                                           bb.col_off, bb.col_off + bb.width, stride)
            print(f"--edge-margin {edge_margin}: rescoring on a {stride}px-stride "
                  f"overlapping grid ({len(positions)} positions) within the "
                  f"{bb.width}x{bb.height} px valid-data bounding box")

        if max_tiles is not None and len(positions) > max_tiles:
            positions = positions[:max_tiles]
        print(f"{len(positions)} candidate tile positions "
              f"({'capped, prefix of the scan order' if max_tiles else 'full swath'}), "
              f"step={step} native px")

        row_off = col_off = 0
        gap_positions = []
        if crop_to_scanned:
            win = crop_window(positions, step, src.width, src.height)
            row_off, col_off = win.row_off, win.col_off
            profile = profile.copy()
            profile.update(width=win.width, height=win.height,
                            transform=crop_transform(win, src.transform))
            gap_positions = grid_gap_positions(positions, step, win)
            print(f"Cropped to the scanned area: {win.width}x{win.height} px "
                  f"(was {src.width}x{src.height}); {len(gap_positions)} "
                  f"never-candidate grid cells inside that box will get an explicit "
                  f"NaN write")

        gate_path = out_dir / "gate_prob.tif"
        unet_path = out_dir / "unet_prob.tif"
        pipe = CrevassePipeline()

        def _write_window(row, col, h, w_):
            """The (array slice, output Window) to actually write for a tile read at
            (row, col) with shape (h, w_) -- the full tile unless --edge-margin trims it
            to its center (see center_crop's own docstring for the true-boundary
            exception)."""
            if not edge_margin:
                return (slice(0, h), slice(0, w_)), Window(col - col_off, row - row_off, w_, h)
            (top, left, bottom, right), (out_row, out_col) = center_crop(
                row, col, h, w_, edge_margin, area_row0, area_col0)
            return ((slice(top, bottom), slice(left, right)),
                    Window(out_col - col_off, out_row - row_off, right - left, bottom - top))

        n_scored = n_flagged = n_dropped = 0
        with _new_output(gate_path, profile, step) as dst_gate, \
             _new_output(unet_path, profile, step) as dst_unet:

            for b0 in range(0, len(positions), batch_size):
                batch_pos = positions[b0:b0 + batch_size]
                tiles, kept_pos = [], []
                for row, col in batch_pos:
                    t = T.read_amp(src, row, col)
                    w = Window(col, row, min(step, src.width - col),
                               min(step, src.height - row))
                    (rs, cs), lw = _write_window(row, col, w.height, w.width)
                    if t is None:
                        n_dropped += 1
                        nan_block = np.full((lw.height, lw.width), np.nan, dtype=np.float32)
                        dst_gate.write(nan_block, 1, window=lw)
                        dst_unet.write(nan_block, 1, window=lw)
                        continue
                    tiles.append(np.nan_to_num(t, nan=0.0).astype(np.float32))
                    kept_pos.append((row, col))

                if tiles:
                    amp = np.stack(tiles)
                    out = pipe.run(amp, gate_thresh=gate_thresh)
                    n_scored += len(kept_pos)
                    n_flagged += int(out.gate_flag.sum())

                    for i, (row, col) in enumerate(kept_pos):
                        w = Window(col, row, min(step, src.width - col),
                                   min(step, src.height - row))
                        (rs, cs), lw = _write_window(row, col, w.height, w.width)
                        gate_block = np.full((lw.height, lw.width),
                                              out.gate_prob[i], dtype=np.float32)
                        dst_gate.write(gate_block, 1, window=lw)

                        if out.gate_flag[i]:
                            p = out.unet_prob[i].astype(np.float32)[:w.height, :w.width][rs, cs]
                            dst_unet.write(p, 1, window=lw)
                        else:
                            nan_block = np.full((lw.height, lw.width), np.nan,
                                                 dtype=np.float32)
                            dst_unet.write(nan_block, 1, window=lw)

                done = min(b0 + batch_size, len(positions))
                print(f"  {done}/{len(positions)} candidates  "
                      f"({n_scored} scored, {n_flagged} flagged, {n_dropped} dropped)",
                      end="\r")

            for row, col in gap_positions:
                gw = Window(col - col_off, row - row_off,
                            min(step, col_off + profile["width"] - col),
                            min(step, row_off + profile["height"] - row))
                nan_block = np.full((gw.height, gw.width), np.nan, dtype=np.float32)
                dst_gate.write(nan_block, 1, window=gw)
                dst_unet.write(nan_block, 1, window=gw)

            dst_gate.stats(approx=False)
            dst_unet.stats(approx=False)

    print()
    print(f"Done: {n_scored} scored, {n_flagged} flagged, {n_dropped} dropped "
          f"(<50% valid)")
    print(f"Wrote {gate_path}")
    print(f"Wrote {unet_path}")
    return gate_path, unet_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--granule", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-tiles", type=int, default=None,
                   help="cap on candidate positions, taken as a PREFIX of the swath scan "
                        "order (not a random sample -- see run_granule.py for that) so a "
                        "capped run still writes a small, correctly-placed patch of both "
                        "GeoTIFFs for a quick correctness check. Omit for the full swath "
                        "-- see the module docstring for why that is slow.")
    p.add_argument("--batch-size", type=int, default=16,
                   help="tiles per U-Net forward pass / write flush. Bounds memory use, "
                        "not accuracy -- raise it on a GPU machine.")
    p.add_argument("--gate-thresh", type=float, default=None)
    p.add_argument("--crop-to-scanned", action="store_true",
                   help="Shrink the output to the bounding box of the positions actually "
                        "processed this run (still on the source grid) instead of the "
                        "full swath extent, and write explicit NaN for every "
                        "never-a-candidate grid cell inside that box. Off by default.")
    p.add_argument("--edge-margin", type=int, default=0,
                   help="Rescore on an overlapping, (step - 2*N)-pixel-stride grid and "
                        "keep only each tile's center region, removing the faint "
                        "dimmer-probability seam that independent non-overlapping tile "
                        "inference leaves along every tile boundary. Costs roughly "
                        "(step/(step-2*N))^2 times more U-Net forward passes. Not "
                        "combinable with --crop-to-scanned. 0 (off) by default.")
    args = p.parse_args(argv)
    export_geotiffs(args.granule, args.out_dir, max_tiles=args.max_tiles,
                     batch_size=args.batch_size, gate_thresh=args.gate_thresh,
                     crop_to_scanned=args.crop_to_scanned, edge_margin=args.edge_margin)


if __name__ == "__main__":
    main()
