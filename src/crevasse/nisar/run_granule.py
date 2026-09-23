"""End-to-end check: point at one NISAR granule GeoTIFF, get gate + segmentation output.

`CrevassePipeline.run()` takes an in-memory tile stack; this script is what builds that
stack from a real file the way a production caller would, and it doubles as this repo's
integration test. There is no separate unit-test suite in this project (see
`docs/CONTRIBUTING.md` and `nisar-crevasse-gate/docs/CONTRIBUTING.md`'s "Known gaps") --
a pinned run against real data, with its numbers printed and checked, plays that role.

    python src/run_granule.py --granule .../GSLC_025_019_..._frequencyA_HH_amplitude.tif
    python src/run_granule.py --granule ... --max-tiles 300 --seed 0 --out-npz data/out.npz
    python src/run_granule.py --granule ... --check      # pinned control P4, see CONTRIBUTING.md

What it does, in order:
  1. `find_data_swath.find_data_bounds` -- locate the swath so empty edge tiles are
     skipped, rather than tiling the whole raster including its nodata margin.
  2. Tile the swath on the granule's OWN native grid: `train_gate_classifier.tile_scale`
     gives the native-pixel step (1 for a 5 m granule, 2 for 2.5 m, ...), and tiles are
     stepped by `512 * scale` so consecutive tiles do not overlap -- the same rule
     `map_crevasse_tiles.py` follows. This is the swath-corner-anchored grid the gate and
     U-Net were both trained against (see PIPELINE.md's tile-grid-origin note for why
     this does NOT line up with the AlphaEarth/context grid).
  3. `train_gate_classifier.read_amp` per tile -> a stack of `(512, 512)` amplitude
     arrays, NaN-filled nodata, `None` (dropped) tiles under 50% valid.
  4. `CrevassePipeline().run(stack)` -- gate, filter, U-Net, scatter back.
  5. Print a summary; optionally write `--out-npz` with positions + all four outputs.

`--max-tiles` subsamples the valid-tile list (seeded, so it's reproducible) rather than
processing a prefix of it -- the swath corner is not a representative sample of tile
content, so a prefix would bias any summary statistic.
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio

import crevasse.nisar.train_gate_classifier as T
from crevasse.nisar.find_data_swath import find_data_bounds, get_valid_tile_positions
from crevasse.nisar.pipeline_predict import CrevassePipeline


def tile_granule(granule_path, max_tiles=None, seed=0):
    """Granule GeoTIFF -> (positions, amp) covering its valid data swath.

    `positions` is `[(row, col), ...]` in native pixels, top-left corner, on the
    granule's own grid. `amp` is `(N, 512, 512)` float32 linear amplitude, NaN-filled
    nodata, in the same order. Tiles `read_amp` drops (under 50% valid) are simply
    absent from both -- this function, unlike the pipeline, is allowed to drop, because
    it is the thing that DEFINES the stack the pipeline's N-in-N-out contract is about.
    """
    with rasterio.open(granule_path) as src:
        scale = T.tile_scale(src)
        bounds, _ = find_data_bounds(str(granule_path), downsample=100)
        if bounds is None:
            raise SystemExit(f"{granule_path}: no data found in raster")
        step = T.E.TS * scale
        all_positions = get_valid_tile_positions(bounds, tile_size=step)

        if max_tiles is not None and len(all_positions) > max_tiles:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(all_positions), size=max_tiles, replace=False)
            idx.sort()
            all_positions = [all_positions[i] for i in idx]

        positions, tiles = [], []
        for row, col in all_positions:
            t = T.read_amp(src, row, col)
            if t is None:
                continue
            positions.append((row, col))
            tiles.append(np.nan_to_num(t, nan=0.0).astype(np.float32))

    if not tiles:
        raise SystemExit(f"{granule_path}: no tiles cleared read_amp's 50% valid-data cut")
    return positions, np.stack(tiles)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--granule", required=True)
    p.add_argument("--max-tiles", type=int, default=None,
                   help="Subsample the valid-tile grid to this many (seeded). Omit for "
                        "the full swath -- slow on CPU for a whole granule's worth of "
                        "U-Net calls.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate-thresh", type=float, default=None,
                   help="Defaults to the gate bundle's own threshold.")
    p.add_argument("--unet-thresh", type=float, default=None,
                   help="If given, also emit a binary unet_mask at this cut.")
    p.add_argument("--out-npz", default=None)
    p.add_argument("--check", action="store_true",
                   help="Run the pinned control P4 (see docs/CONTRIBUTING.md) instead of "
                        "an arbitrary summary: --max-tiles 300 --seed 0 on 025_019, "
                        "checked against its recorded numbers.")
    args = p.parse_args(argv)

    max_tiles, seed = args.max_tiles, args.seed
    if args.check:
        max_tiles, seed = 300, 0
        print("Running control P4: --max-tiles 300 --seed 0")

    positions, amp = tile_granule(args.granule, max_tiles=max_tiles, seed=seed)
    print(f"Granule: {Path(args.granule).name}")
    print(f"Tiled  : {len(positions)} valid tiles (of the {'full swath' if max_tiles is None else f'{max_tiles}-tile sample'})")

    pipe = CrevassePipeline()
    out = pipe.run(amp, gate_thresh=args.gate_thresh, unet_thresh=args.unet_thresh)

    n_scored = int(np.isfinite(out.gate_prob).sum())
    n_flagged = int(out.gate_flag.sum())
    flagged_idx = np.flatnonzero(out.gate_flag)
    mean_unet_p = (float(np.nanmean(out.unet_prob[flagged_idx]))
                   if len(flagged_idx) else float("nan"))

    print(f"Gate   : {n_scored}/{len(positions)} scored, {n_flagged} flagged "
          f"(thresh={pipe.gate.threshold if args.gate_thresh is None else args.gate_thresh})")
    print(f"U-Net  : ran on {len(flagged_idx)} tiles, mean P(crevasse) over their pixels "
          f"= {mean_unet_p:.4f}")

    if args.check:
        # Pinned 2026-09-16 on 025_019, --max-tiles 300 --seed 0: 300 candidate positions
        # are sampled from the swath, read_amp's own 50% valid-data cut drops some of
        # them (deterministic given the seed), and what's left is scored and gated
        # exactly -- see docs/CONTRIBUTING.md control P4 for the numbers this must
        # reproduce.
        assert len(positions) == 169, (
            f"expected 169 tiles to clear read_amp's valid-data cut, got {len(positions)}")
        assert n_scored == 169, f"expected all 169 scored, got {n_scored}"
        assert n_flagged == 43, f"expected 43 flagged, got {n_flagged}"
        print(f"\nCONTROL P4: 300 candidates -> {len(positions)} cleared read_amp, "
              f"{n_scored} scored, {n_flagged} flagged -- matches docs/CONTRIBUTING.md. PASS")

    if args.out_npz:
        Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out_npz,
            positions=np.array(positions, dtype=np.int64),
            gate_prob=out.gate_prob, gate_flag=out.gate_flag,
            unet_prob=out.unet_prob,
            unet_mask=out.unet_mask if out.unet_mask is not None else np.array([]),
        )
        print(f"\nWrote {args.out_npz}")


if __name__ == "__main__":
    main()
