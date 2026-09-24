"""End-to-end check: point at one BIOMASS granule directory, get gate + segmentation
output.

`BiomassCrevassePipeline.run()` takes an in-memory tile stack; this script is what builds
that stack from a real granule the way a production caller would, and it doubles as this
repo's integration test. There is no separate unit-test suite in this project (matching
`nisar-crevasse-pipeline`'s own `docs/CONTRIBUTING.md`) -- a pinned run against real data,
with its numbers printed and checked, plays that role.

    python src/run_granule.py --granule /path/to/BIO_S2_SCS__..._DJT7YH
    python src/run_granule.py --granule ... --max-tiles 200 --seed 0 --out-npz data/out.npz
    python src/run_granule.py --granule ... --check      # pinned control P4, see CONTRIBUTING.md

What it does, in order:
  1. A coarse, decimated valid-fraction prescan of the granule's 4 polarization GeoTIFFs
     (same `DEC=32` decimation `bio_tile_features.coarse_masks` uses, minus its
     grounded/AlphaEarth/NESZ machinery -- those are training-set-curation concerns, not
     scoring concerns for an arbitrary new granule).
  2. Full-resolution windowed reads of the surviving candidates (mirrors
     `bio_tile_cache.granule_chips`'s read pattern), kept only if >=`min_valid_frac` of
     each polarization is valid there too -- a fine-grained re-check, since a decimated
     average can differ from the true full-res fraction.
  3. `BiomassCrevassePipeline().run(inten, x_m, y_m)` -- gate, filter, U-Net, scatter back.
  4. Print a summary; optionally write `--out-npz` with positions + all four outputs.

`--max-tiles` subsamples the valid-tile list (seeded, so it's reproducible) rather than
processing a prefix of it -- the scan order is not a representative sample of tile
content, so a prefix would bias any summary statistic (same rationale as
`nisar-crevasse-pipeline/src/run_granule.py`).

`--min-grounded` (with `--bedmap-mask`) drops candidate positions whose Bedmap3
grounded-ice fraction is below the given cut, before any full-resolution read or gate
scoring -- see `crevasse.common.grounded_filter` and `export_geotiff.py`'s own
docstring. Off by default; not usable with `--check` (would change the pinned tile
counts).
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from crevasse.biomass.pipeline_predict import BiomassCrevassePipeline
from crevasse.common.grounded_filter import grounded_vrt, filter_grounded

POLS = ["HH", "HV", "VH", "VV"]
T = 512
DEC = 32                    # decimation for the coarse prescan, matches
SUB = T // DEC               # bio_tile_features.py's DEC/SUB exactly


def _granule_paths(granule_dir):
    paths = {}
    for p in POLS:
        matches = glob.glob(os.path.join(granule_dir, f"*_{p}_intensity.tif"))
        if not matches:
            raise SystemExit(f"{granule_dir}: no *_{p}_intensity.tif found")
        paths[p] = matches[0]
    return paths


def _coarse_valid_frac(paths, shape_full):
    """Tile-grid valid fraction from a DEC-decimated whole-scene read, one array (nr, nc)
    -- cheap even on a full BIOMASS scene, since it never reads at native resolution."""
    h, w = shape_full
    nr, nc = h // T, w // T
    oh, ow = nr * SUB, nc * SUB
    valid = np.ones((oh, ow), bool)
    for p in POLS:
        with rasterio.open(paths[p]) as s:
            d = s.read(1, out_shape=(h // DEC, w // DEC), resampling=Resampling.average)
        valid &= np.isfinite(d[:oh, :ow]) & (d[:oh, :ow] > 0)
    return valid.reshape(nr, SUB, nc, SUB).mean(axis=(1, 3))


def tile_granule(granule_dir, max_tiles=None, seed=0, min_valid_frac=0.99,
                  bedmap_mask=None, min_grounded=None):
    """Granule directory -> (positions, x_m, y_m, inten) covering its valid data.

    `positions` is `[(row, col), ...]` in native pixels, top-left corner. `x_m`/`y_m` are
    each tile's top-left corner in EPSG:3031 metres (same formula as
    `bio_tile_features.process_granule`: `x_m=bounds[0], y_m=bounds[3]`), needed if the
    caller wants `gate_method="cnn"`. `inten` is `(N, 4, 512, 512)` float32 LINEAR
    intensity (the raw GeoTIFF units), NaN-filled nodata, HH/HV/VH/VV order. Tiles that
    don't clear `min_valid_frac` at full resolution are simply absent from all four --
    this function, unlike the pipeline, is allowed to drop, because it is the thing that
    DEFINES the stack the pipeline's N-in-N-out contract is about.
    """
    if bool(bedmap_mask) != (min_grounded is not None):
        raise ValueError("--bedmap-mask and --min-grounded must be given together")
    paths = _granule_paths(granule_dir)
    with rasterio.open(paths["HH"]) as s:
        shape_full = s.shape
        src_crs, src_transform = s.crs, s.transform
        width, height = s.width, s.height

    vf = _coarse_valid_frac(paths, shape_full)
    cand = np.argwhere(vf >= min_valid_frac)

    if bedmap_mask:
        cand_rc = [(int(i) * T, int(j) * T) for i, j in cand]
        with grounded_vrt(bedmap_mask, src_crs, src_transform, width, height) as gvrt:
            kept = filter_grounded(
                gvrt, cand_rc, lambda row, col: (T, T), width, height, min_grounded)
        print(f"--min-grounded {min_grounded}: kept {len(kept)} of {len(cand_rc)} "
              f"candidate positions")
        cand = np.array([(r // T, c // T) for r, c in kept],
                        dtype=cand.dtype).reshape(-1, 2)

    if max_tiles is not None and len(cand) > max_tiles:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(cand), size=max_tiles, replace=False)
        idx.sort()
        cand = cand[idx]

    srcs = {p: rasterio.open(paths[p]) for p in POLS}
    positions, x_m, y_m, tiles = [], [], [], []
    try:
        for i, j in cand:
            r, c = int(i) * T, int(j) * T
            w = Window(c, r, T, T)
            arr = np.full((len(POLS), T, T), np.nan, dtype=np.float32)
            ok = True
            for pi, p in enumerate(POLS):
                a = srcs[p].read(1, window=w).astype(np.float32)
                a = np.where(a > 0, a, np.nan)
                if np.isfinite(a).mean() < min_valid_frac:
                    ok = False
                    break
                arr[pi] = a
            if not ok:
                continue
            bounds = srcs["HH"].window_bounds(w)
            positions.append((r, c))
            x_m.append(float(bounds[0]))
            y_m.append(float(bounds[3]))
            tiles.append(arr)
    finally:
        for s in srcs.values():
            s.close()

    if not tiles:
        raise SystemExit(
            f"{granule_dir}: no tiles cleared the {min_valid_frac} valid-data cut")
    return positions, np.array(x_m), np.array(y_m), np.stack(tiles)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--granule", required=True,
                   help="directory holding *_{HH,HV,VH,VV}_intensity.tif")
    p.add_argument("--max-tiles", type=int, default=None,
                   help="Subsample the valid-tile grid to this many (seeded). Omit for "
                        "the full granule -- slow on CPU for a granule's worth of U-Net "
                        "calls.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate-method", choices=["rf", "cnn"], default="rf",
                   help="'rf' (default) matches bio_build_shard_frangi.py, needs no "
                        "coordinates. 'cnn' needs x_m/y_m, provided automatically here.")
    p.add_argument("--gate-thresh", type=float, default=0.65)
    p.add_argument("--unet-thresh", type=float, default=None,
                   help="If given, also emit a binary unet_mask at this cut.")
    p.add_argument("--out-npz", default=None)
    p.add_argument("--check", action="store_true",
                   help="Run the pinned control P4 (see docs/CONTRIBUTING.md) instead of "
                        "an arbitrary summary.")
    p.add_argument("--bedmap-mask", default=None,
                   help="Path to a Bedmap3 grounded-ice mask GeoTIFF (class 1 = "
                        "grounded; not shipped in this repo -- get it from NERC BAS). "
                        "Required together with --min-grounded.")
    p.add_argument("--min-grounded", type=float, default=None,
                   help="Drop candidate positions whose Bedmap3 grounded fraction is "
                        "below this, before any gate/U-Net scoring -- e.g. 0.7 keeps "
                        "only tiles that are at least 70%% grounded ice. Off by default. "
                        "Requires --bedmap-mask. Not used by --check (would change the "
                        "pinned control numbers).")
    args = p.parse_args(argv)

    max_tiles, seed = args.max_tiles, args.seed
    if args.check:
        if args.bedmap_mask or args.min_grounded is not None:
            raise SystemExit("--check pins exact tile counts; --bedmap-mask/"
                              "--min-grounded would change them. Run without --check.")
        max_tiles, seed = 200, 0
        print("Running control P4: --max-tiles 200 --seed 0")

    positions, x_m, y_m, inten = tile_granule(
        args.granule, max_tiles=max_tiles, seed=seed,
        bedmap_mask=args.bedmap_mask, min_grounded=args.min_grounded)
    print(f"Granule: {Path(args.granule).name}")
    print(f"Tiled  : {len(positions)} valid tiles "
          f"(of the {'full granule' if max_tiles is None else f'{max_tiles}-tile sample'})")

    pipe = BiomassCrevassePipeline(gate_method=args.gate_method, gate_thresh=args.gate_thresh)
    out = pipe.run(inten, x_m=x_m, y_m=y_m, unet_thresh=args.unet_thresh)

    n_scored = int(np.isfinite(out.gate_prob).sum())
    n_flagged = int(out.gate_flag.sum())
    flagged_idx = np.flatnonzero(out.gate_flag)
    mean_unet_p = (float(np.nanmean(out.unet_prob[flagged_idx]))
                   if len(flagged_idx) else float("nan"))

    print(f"Gate   : {n_scored}/{len(positions)} scored, {n_flagged} flagged "
          f"(method={args.gate_method}, thresh={args.gate_thresh})")
    print(f"U-Net  : ran on {len(flagged_idx)} tiles, mean P(crevasse) over their pixels "
          f"= {mean_unet_p:.4f}")

    if args.check:
        # Pinned 2026-09-22 on BIO_S2_SCS__1S_20251217T135546_..._DJT7YH,
        # --max-tiles 200 --seed 0, --gate-method rf (default) -- see
        # docs/CONTRIBUTING.md control P4 for the full writeup.
        assert len(positions) == 200, f"expected 200 tiles, got {len(positions)}"
        assert n_scored == 200, f"expected all 200 scored, got {n_scored}"
        assert n_flagged == 84, f"expected 84 flagged, got {n_flagged}"
        print(f"\nCONTROL P4: 200 candidates -> {len(positions)} cleared the valid-data "
              f"cut, {n_scored} scored, {n_flagged} flagged -- matches "
              f"docs/CONTRIBUTING.md. PASS")

    if args.out_npz:
        Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out_npz,
            positions=np.array(positions, dtype=np.int64), x_m=x_m, y_m=y_m,
            gate_prob=out.gate_prob, gate_flag=out.gate_flag,
            unet_prob=out.unet_prob,
            unet_mask=out.unet_mask if out.unet_mask is not None else np.array([]),
        )
        print(f"\nWrote {args.out_npz}")


if __name__ == "__main__":
    main()
