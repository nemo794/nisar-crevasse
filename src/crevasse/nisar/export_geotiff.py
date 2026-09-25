"""Export a full granule gate+segmentation run to two Cloud-Optimized GeoTIFFs (COG)
registered on the source raster's own grid.

    python src/export_geotiff.py --granule <path> --out-dir data/export_025_019
    # -> data/export_025_019/gate_prob.tif     one band, float32, NaN nodata, COG
    # -> data/export_025_019/unet_prob.tif     one band, float32, NaN nodata, COG

Both outputs share the source raster's CRS, transform, width and height EXACTLY, so they
open lined up with the granule itself in QGIS / rasterio / any GIS tool -- no separate
transform bookkeeping needed on the reading side.

Written as a COG: 512px internal blocksize, DEFLATE/predictor=3 (matches a collaborator's
specified GDAL creation options -- BLOCKSIZE=512, COMPRESS=DEFLATE, LEVEL=1, PREDICTOR=3,
OVERVIEW_RESAMPLING=AVERAGE, NUM_THREADS=ALL_CPUS, BIGTIFF=YES), with overviews and
embedded per-band statistics (`gdal_translate -stats`). The COG is produced in two steps:
this script streams its windowed `write()` calls to a temporary *tiled plain GeoTIFF*,
then converts that to the COG in one `gdal_translate` pass at the end (see
`crevasse.common.cog_io`). Opening the output with the COG driver directly and writing
windows into it does NOT stream -- the COG driver buffers every written block in memory
until close, which OOM-kills a full-swath run (observed 2026-09-25: SIGKILL/exit 137
mid-scan on a 16 GB worker). An earlier revision of this file claimed the COG driver
"supports the same incremental windowed write() calls"; the calls succeed but are
buffered, not flushed, so that was wrong -- hence the temp-GeoTIFF-then-translate path.

    gate_prob.tif   each tile's SCALAR gate probability, broadcast flat over its whole
                    step x step footprint (the gate has no notion of "where inside the
                    tile"). NaN for tiles `read_amp` dropped (<50% valid) but still
                    inside the scanned swath, and for everywhere outside the scanned
                    swath entirely (see "NaN prefill" below).
    unet_prob.tif   the U-Net's real per-pixel probability, written only for tiles the
                    gate flagged. NaN everywhere else: within the scanned swath (gate
                    said no, or `read_amp` dropped it) and outside it.

Streams tile-by-tile via windowed writes -- nothing in this script holds more than
`--batch-size` tiles in memory at once, so it runs on a laptop regardless of granule
size. The tradeoff is wall-clock time, not memory: a full granule is thousands of tile
positions (9236 for 025_019) and most of the time is the U-Net forward pass on CPU.

**THIS IS SLOW ON A FULL GRANULE** -- budget tens of minutes to hours depending on the
machine and how many tiles the gate flags. Use --max-tiles for a quick correctness check
(a handful of tiles, seconds) before ever starting a full run. `UNetSegmenter` picks up
CUDA automatically if it's there, so a GPU machine is the real fix, not a smaller batch.

NaN prefill: unwritten GeoTIFF blocks are not guaranteed to read back as the declared
NaN nodata value -- GDAL fills a block a driver never wrote with 0 by default, which
QGIS (and most viewers) render as solid black rather than the white it uses for real
NaN nodata -- confirmed 2026-09-23 on a full, uncropped BIOMASS export as a black ring
outside the white NaN ring around the scanned footprint; the same applies here. Both
outputs are prefilled with NaN across their ENTIRE extent right after creation
(`geotiff_crop.prefill_nan`), before any real or per-position write, so no block is ever
left at GDAL's 0 default -- the area outside the scanned swath reads back as genuine
NaN, same as a dropped tile.

`--crop-to-scanned` shrinks the output to the bounding box of the positions actually
processed this run (still on the source grid, so it opens lined up the same way, just
over a smaller extent). Off by default -- the full-source-extent output stays available
for callers that need exact pixel alignment with the untouched source raster.

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

`--min-grounded` (with `--bedmap-mask`) drops candidate positions before any gate/U-Net
scoring whose Bedmap3 grounded-ice fraction is below the given cut -- e.g. `0.7` keeps
only tiles that are at least 70% grounded ice, discarding floating-shelf/sea-ice/rock/
ocean tiles up front rather than letting the gate see and score them. See
`crevasse.common.grounded_filter` for why this reads each tile's own real pixel window
rather than a separate precomputed context grid. Off by default: pass `--bedmap-mask`
with no value to use the bundled `models/bedmap3_mask.tif`, a path to use your own, or
omit it entirely to skip grounded filtering. Both flags are required together.
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

import crevasse.nisar.train_gate_classifier as T
from crevasse.nisar.find_data_swath import find_data_bounds, get_valid_tile_positions
from crevasse.nisar.pipeline_predict import CrevassePipeline
from crevasse.common.geotiff_crop import (crop_window, crop_transform, prefill_nan,
                                           overlap_positions, center_crop)
from crevasse.common.grounded_filter import grounded_vrt, filter_grounded, DEFAULT_BEDMAP_MASK
from crevasse.common.cog_io import open_temp_tiff, finalize_cog


def export_geotiffs(granule_path, out_dir, max_tiles=None, batch_size=16,
                     gate_thresh=None, crop_to_scanned=False, edge_margin=0,
                     bedmap_mask=None, min_grounded=None):
    """Stream a full (or capped) granule run to `gate_prob.tif` and `unet_prob.tif`
    under `out_dir`, both lined up with `granule_path`'s own grid. Returns the two
    output paths. See the module docstring for the write semantics, and for what
    `crop_to_scanned`/`edge_margin` change."""
    if crop_to_scanned and edge_margin:
        raise ValueError("--crop-to-scanned and --edge-margin are not supported together yet")
    if bool(bedmap_mask) != (min_grounded is not None):
        raise ValueError("--bedmap-mask and --min-grounded must be given together")
    out_dir = Path(out_dir)

    with rasterio.open(granule_path) as src:
        profile = src.profile
        scale = T.tile_scale(src)
        step = T.E.TS * scale

        bounds, _ = find_data_bounds(str(granule_path), downsample=100)
        if bounds is None:
            raise SystemExit(f"{granule_path}: no data found in raster")
        positions = get_valid_tile_positions(bounds, tile_size=step)

        if bedmap_mask:
            with grounded_vrt(bedmap_mask, src.crs, src.transform,
                               src.width, src.height) as gvrt:
                before = len(positions)
                positions = filter_grounded(
                    gvrt, positions,
                    lambda row, col: (min(step, src.width - col),
                                       min(step, src.height - row)),
                    src.width, src.height, min_grounded)
            print(f"--min-grounded {min_grounded}: kept {len(positions)} of {before} "
                  f"candidate positions")

        stride = step
        area_row0 = area_col0 = 0
        if edge_margin and positions:
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
        if not positions:
            print("No candidates -- writing all-NaN full-extent GeoTIFFs "
                  "(--edge-margin/--crop-to-scanned have nothing to rescore or crop to).")

        row_off = col_off = 0
        if crop_to_scanned and positions:
            win = crop_window(positions, step, src.width, src.height)
            row_off, col_off = win.row_off, win.col_off
            profile = profile.copy()
            profile.update(width=win.width, height=win.height,
                            transform=crop_transform(win, src.transform))
            print(f"Cropped to the scanned area: {win.width}x{win.height} px "
                  f"(was {src.width}x{src.height})")

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
        # Stream windowed writes to temporary tiled GeoTIFFs, then convert each to a COG in
        # one gdal_translate pass at the end -- see crevasse.common.cog_io for why writing
        # the COG directly (as this used to) OOMs on a full swath.
        dst_gate, gate_temp = open_temp_tiff(gate_path, profile, step)
        dst_unet, unet_temp = open_temp_tiff(unet_path, profile, step)
        with dst_gate, dst_unet:

            prefill_nan(dst_gate)
            prefill_nan(dst_unet)

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

        # Datasets are closed (temp tiled GeoTIFFs fully written); convert each to a COG,
        # embedding per-band stats via gdal_translate -stats (replaces the old in-place
        # dst.stats(approx=False) call).
        print()
        print("Converting to Cloud-Optimized GeoTIFFs...")
        finalize_cog(gate_temp, gate_path)
        finalize_cog(unet_temp, unet_path)

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
                        "full swath extent. Off by default.")
    p.add_argument("--edge-margin", type=int, default=0,
                   help="Rescore on an overlapping, (step - 2*N)-pixel-stride grid and "
                        "keep only each tile's center region, removing the faint "
                        "dimmer-probability seam that independent non-overlapping tile "
                        "inference leaves along every tile boundary. Costs roughly "
                        "(step/(step-2*N))^2 times more U-Net forward passes. Not "
                        "combinable with --crop-to-scanned. 0 (off) by default.")
    p.add_argument("--bedmap-mask", nargs="?", const=DEFAULT_BEDMAP_MASK, default=None,
                   help="Path to a Bedmap3 grounded-ice mask GeoTIFF (class 1 = "
                        "grounded). Pass with no value to use the bundled default "
                        f"({DEFAULT_BEDMAP_MASK}); pass a path to use your own; omit "
                        "the flag entirely to disable grounded filtering. Required "
                        "together with --min-grounded.")
    p.add_argument("--min-grounded", type=float, default=None,
                   help="Drop candidate positions whose Bedmap3 grounded fraction is "
                        "below this, before any gate/U-Net scoring -- e.g. 0.7 keeps "
                        "only tiles that are at least 70%% grounded ice. Off by default. "
                        "Requires --bedmap-mask.")
    args = p.parse_args(argv)
    export_geotiffs(args.granule, args.out_dir, max_tiles=args.max_tiles,
                     batch_size=args.batch_size, gate_thresh=args.gate_thresh,
                     crop_to_scanned=args.crop_to_scanned, edge_margin=args.edge_margin,
                     bedmap_mask=args.bedmap_mask, min_grounded=args.min_grounded)


if __name__ == "__main__":
    main()
