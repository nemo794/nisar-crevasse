"""Derive tile-level crevasse/none labels for the gate from an AlphaEarth classification raster.

The AlphaEarth export for granule 025_019 is pixel-identical to the granule
(same CRS/transform/shape), so tile (row, col) indexes both rasters directly.

Labeling rule (strict, with an exclusion band):
    crevasse : aoi_pos_frac >  --pos-thresh
    none     : aoi_pos_frac == 0 AND no positive tile within --neg-buffer tiles
    dropped  : 0 < aoi_pos_frac <= --pos-thresh          (ambiguous field edge)
Tiles with data_frac < --min-data-frac are excluded (train_gate_classifier.read_amp
rejects those anyway, at 0.5).

Caveat on the negatives: the AlphaEarth ROI is smaller than the export box and EE
writes outside-ROI pixels as byte 0, indistinguishable from a true negative. So a
frac==0 tile far from the AOI means "never evaluated", not "no crevasse" — it is
only safe here because that remainder is slow interior ice. Near the AOI the zeros
are actively wrong (AlphaEarth recall is ~0.42), hence the wide --neg-buffer.

Sampling is *prefix-stable*: candidates are shuffled once with --seed and accepted
greedily, so rerunning with a larger --n-per-class extends the previous set rather
than reshuffling it. The per-tile scan is cached, so reruns skip the raster read.

    # first pass
    python src/build_aoi_tile_labels.py --n-per-class 1000
    # later, more labels (superset of the above)
    python src/build_aoi_tile_labels.py --n-per-class 4000
    # every qualifying tile
    python src/build_aoi_tile_labels.py --n-per-class 0
"""
import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

import crevasse.nisar.gate_common as E
from crevasse.nisar.gate_common import find_granule
from crevasse.nisar.build_context_masks import BEDMAP_CLASSES

DEFAULT_AOI = E.ROOT / "data" / "aois" / "thwaites_mosaic.vrt"
DATA_DECIM = 16   # granule decimation when measuring per-tile valid-data fraction
MATCH_GROUNDED_MIN = 0.5   # speed matching is only meaningful on grounded ice


def aoi_offset(g, a):
    """Pixel offset (drow, dcol) of the granule's origin inside the AOI raster.

    The AOI export is pixel-aligned to 025_019, but every 025_* granule sits on the
    same 5 m EPSG:3031 grid at a different origin, so a second granule can be indexed
    by an integer pixel shift -- no resampling, hence no interpolation of a
    categorical mask. A fractional offset would mean genuinely different grids, and
    (row, col) could no longer index both rasters.
    """
    if g.crs != a.crs:
        raise SystemExit(f"CRS mismatch: granule {g.crs} vs aoi {a.crs}")
    if abs(g.transform.a - a.transform.a) > 1e-6 or abs(g.transform.e - a.transform.e) > 1e-6:
        raise SystemExit(f"pixel size mismatch: granule {g.transform.a}/{g.transform.e} "
                         f"vs aoi {a.transform.a}/{a.transform.e}")
    dc = (g.transform.c - a.transform.c) / a.transform.a
    dr = (g.transform.f - a.transform.f) / a.transform.e
    if abs(dr - round(dr)) > 1e-4 or abs(dc - round(dc)) > 1e-4:
        raise SystemExit(f"granule origin is not on the AOI pixel grid "
                         f"(offset {dr:.4f}, {dc:.4f} px) -- cannot index both rasters")
    return int(round(dr)), int(round(dc))


def scan_grid(granule_path, aoi_path, tile):
    """Per-tile AOI positive fraction and valid-data fraction on the tile grid."""
    g = rasterio.open(granule_path)
    a = rasterio.open(aoi_path)
    drow, dcol = aoi_offset(g, a)
    if (drow, dcol) != (0, 0):
        print(f"  granule origin is offset ({drow}, {dcol}) px inside the AOI raster; "
              f"reading the AOI shifted (outside reads as 0 -- use --aoi-bbox)")

    nrow, ncol = g.height // tile, g.width // tile
    aoi_frac = np.zeros((nrow, ncol), np.float32)
    data_frac = np.zeros((nrow, ncol), np.float32)
    sub = tile // DATA_DECIM

    for i in range(nrow):
        w = Window(0, i * tile, ncol * tile, tile)
        m = a.read(1, window=Window(dcol, i * tile + drow, ncol * tile, tile),
                   boundless=True, fill_value=0) == 1
        aoi_frac[i] = m.reshape(tile, ncol, tile).mean(axis=(0, 2))

        d = g.read(1, window=w, out_shape=(sub, ncol * sub),
                   resampling=Resampling.nearest)
        v = np.isfinite(d) & (d > 0)
        data_frac[i] = v.reshape(sub, ncol, sub).mean(axis=(0, 2))

        if (i + 1) % 20 == 0 or i + 1 == nrow:
            print(f"  scanned tile-row {i + 1}/{nrow}", flush=True)

    g.close()
    a.close()
    return aoi_frac, data_frac


def tile_distance_to_positive(any_pos):
    """Euclidean distance, in tiles, from each tile to the nearest AOI-positive tile."""
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~any_pos)


def pick(cands, n, min_sep, seed):
    """Greedy prefix-stable selection: fixed shuffle, accept if >= min_sep from picks."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cands))
    taken = []
    for idx in order:
        y, x = cands[idx]
        if min_sep > 0 and any(max(abs(y - py), abs(x - px)) < min_sep for py, px in taken):
            continue
        taken.append((y, x))
        if n and len(taken) >= n:
            break
    return taken


def match_by_speed(pos, neg, vel, nbins, seed):
    """1:1 match positives to negatives inside pooled ice-speed quantile bins.

    Unmatched, ice speed alone separates these labels at AUC ~0.87, so a gate can
    score well by learning "fast ice" rather than "crevasse". Matching removes that
    by construction. Bins holding no negatives yield no pairs; their positives come
    back as `spare` -- a fast-ice regime with no counterexamples, which can be
    reported for recall but never used as a balanced test.
    """
    speeds = lambda sel: np.array([vel[y, x] for y, x in sel])
    vp, vn = speeds(pos), speeds(neg)
    edges = np.unique(np.nanquantile(np.concatenate([vp, vn]),
                                    np.linspace(0, 1, nbins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    keep_p, keep_n, spare = [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        bp = [t for t, v in zip(pos, vp) if lo <= v < hi]
        bn = [t for t, v in zip(neg, vn) if lo <= v < hi]
        k = min(len(bp), len(bn))
        sel_p = pick(bp, k, 0, seed + i) if k else []
        keep_p += sel_p
        keep_n += pick(bn, k, 0, seed + 1000 + i) if k else []
        spare += [t for t in bp if t not in set(sel_p)]
        print(f"  speed {lo:8.0f}-{hi:<8.0f} pos {len(bp):4d} neg {len(bn):4d} -> {k:4d} pairs")
    return keep_p, keep_n, spare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granule", default="025_019")
    ap.add_argument("--aoi", default=str(DEFAULT_AOI))
    ap.add_argument("--tile-size", type=int, default=E.TS)
    ap.add_argument("--pos-thresh", type=float, default=0.25,
                    help="tile is crevasse if AOI positive fraction exceeds this")
    ap.add_argument("--neg-buffer", type=int, default=8,
                    help="a negative must have no AOI-positive tile within this many tiles. "
                         "8 tiles ~= 20 km: within that band AlphaEarth's recall failures make "
                         "~46%% of frac==0 tiles actually crevassed (measured against the gate)")
    ap.add_argument("--neg-max-buffer", type=int, default=0,
                    help="0 = off. Cap negative distance to this many tiles, so negatives are "
                         "drawn from the near edge of the allowed band instead of uniformly. "
                         "Uniform sampling is dominated by far interior ice, which is trivial.")
    ap.add_argument("--amb-as-pos-min", type=float, default=0.0,
                    help="0 = off. Promote ambiguous tiles (0 < frac <= --pos-thresh) with "
                         "frac >= this to hard positives — sparse field-fringe crevasses. "
                         "Below ~0.01 the positive pixels are noise; 0.01 keeps real evidence.")
    ap.add_argument("--aoi-bbox", action="store_true",
                    help="restrict ALL candidates to the bounding box of the AOI positives. "
                         "That box is the only recoverable proxy for where AlphaEarth actually "
                         "ran, so inside it a frac==0 tile is a real negative rather than "
                         "'never evaluated'. Outside it, emit nothing.")
    ap.add_argument("--bbox-margin", type=int, default=0,
                    help="grow the --aoi-bbox by this many tiles on every side")
    ap.add_argument("--context-npz", default=None,
                    help="_context_masks_*.npz from build_context_masks.py. Adds "
                         "grounded_frac / ice_condition / vel_m_yr columns so surface-type "
                         "and speed cuts can be made downstream from the CSV. Filters "
                         "nothing by default.")
    ap.add_argument("--min-grounded", type=float, default=0.0,
                    help="0 = annotate only (default). Set to filter at build time instead.")
    ap.add_argument("--min-vel", type=float, default=0.0,
                    help="0 = annotate only (default). Set to filter at build time instead.")
    ap.add_argument("--match-neg-speed", type=int, default=0,
                    help="0 = off. Number of pooled ice-speed quantile bins to match "
                         "negatives to positives within, on grounded ice only. Removes the "
                         "speed confound (AUC 0.87 -> ~0.54) at the cost of dropping "
                         "positives in bins with no negatives. Needs --context-npz; "
                         "ignores --n-per-class and --min-sep-*.")
    ap.add_argument("--min-data-frac", type=float, default=0.5,
                    help="matches train_gate_classifier.read_amp, which rejects below 0.5")
    ap.add_argument("--n-per-class", type=int, default=1000,
                    help="0 = every qualifying tile")
    ap.add_argument("--min-sep-pos", type=int, default=0,
                    help="min tile separation between positives; keep 0, they are "
                         "clustered in the crevasse field and separation thins them hard")
    ap.add_argument("--min-sep-neg", type=int, default=2,
                    help="min tile separation between negatives (plentiful, so spread them)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--rescan", action="store_true")
    args = ap.parse_args()

    gpath = find_granule(args.granule)
    cache = Path(args.cache) if args.cache else (
        E.ROOT / "data" / f"_aoi_tile_grid_{args.granule}_t{args.tile_size}.npz")
    out = Path(args.out) if args.out else (
        E.ROOT / "data" / f"tile_labels_aoi_{args.granule}.csv")

    if cache.exists() and not args.rescan:
        z = np.load(cache)
        aoi_frac, data_frac = z["aoi_frac"], z["data_frac"]
        print(f"loaded cached scan {cache.name}  grid {aoi_frac.shape}")
    else:
        print(f"scanning {gpath.name}\n     vs {Path(args.aoi).name}")
        aoi_frac, data_frac = scan_grid(gpath, args.aoi, args.tile_size)
        np.savez_compressed(cache, aoi_frac=aoi_frac, data_frac=data_frac)
        print(f"cached scan -> {cache}")

    enough = data_frac >= args.min_data_frac
    any_pos = aoi_frac > 0
    dist = tile_distance_to_positive(any_pos)
    km = args.tile_size * 5 / 1000.0

    if args.aoi_bbox:
        r = np.flatnonzero(any_pos.any(1))
        c = np.flatnonzero(any_pos.any(0))
        m = args.bbox_margin
        box = np.zeros_like(any_pos)
        box[max(0, r[0] - m):r[-1] + 1 + m, max(0, c[0] - m):c[-1] + 1 + m] = True
        enough &= box
        print(f"AOI-positive bbox: rows {r[0]}-{r[-1]}, cols {c[0]}-{c[-1]} "
              f"(+{m} margin) = {int(box.sum())} tiles, "
              f"{(r[-1] - r[0] + 1) * km:.0f}x{(c[-1] - c[0] + 1) * km:.0f} km")

    gfrac = vel = bcls = None
    cls_names = {}
    if args.context_npz:
        cls_names = BEDMAP_CLASSES
        ctx = np.load(args.context_npz)
        gfrac, vel = ctx["grounded_frac"], ctx["vel_mean"]
        bcls = ctx["bedmap_class"] if "bedmap_class" in ctx else None
        if args.min_grounded > 0 or args.min_vel > 0:
            keep = gfrac > args.min_grounded
            if args.min_vel > 0:
                keep &= np.nan_to_num(vel, nan=-1.0) > args.min_vel
            enough &= keep
            print(f"context FILTER applied: grounded>{args.min_grounded} "
                  f"& vel>{args.min_vel:g} m/yr -> {int(keep.sum())} tiles allowed")
        else:
            print("context annotated as columns (grounded_frac, ice_condition, "
                  "vel_m_yr); no filtering")

    core_pos = (aoi_frac > args.pos_thresh) & enough
    ambig = any_pos & ~core_pos & enough
    hard_pos = (ambig & (aoi_frac >= args.amb_as_pos_min)
                if args.amb_as_pos_min > 0 else np.zeros_like(ambig))
    pos_m = core_pos | hard_pos

    neg_m = (aoi_frac == 0) & (dist >= args.neg_buffer) & enough
    if args.neg_max_buffer:
        neg_m &= dist <= args.neg_max_buffer

    print(f"\ngrid {aoi_frac.shape[0]}x{aoi_frac.shape[1]} = {aoi_frac.size} tiles")
    print(f"  data_frac >= {args.min_data_frac}      : {int(enough.sum())}")
    print(f"  positive core (frac > {args.pos_thresh}) : {int(core_pos.sum())}")
    if args.amb_as_pos_min > 0:
        print(f"  positive hard (frac >= {args.amb_as_pos_min}) : {int(hard_pos.sum())} "
              f"of {int(ambig.sum())} ambiguous")
    else:
        print(f"  dropped ambiguous              : {int(ambig.sum())}")
    band = (f"{args.neg_buffer}-{args.neg_max_buffer}" if args.neg_max_buffer
            else f">={args.neg_buffer}")
    print(f"  negative  (frac == 0, {band} tiles = "
          f"{args.neg_buffer * km:.0f}-{args.neg_max_buffer * km if args.neg_max_buffer else 999:.0f} km)"
          f" : {int(neg_m.sum())}")

    n = args.n_per_class
    spare = []
    if args.match_neg_speed:
        if vel is None:
            raise SystemExit("--match-neg-speed needs --context-npz for velocity")
        ok = (gfrac > MATCH_GROUNDED_MIN) & np.isfinite(vel)
        tiles = lambda m: [(int(y), int(x)) for y, x in np.argwhere(m & ok)]
        print(f"\nspeed-matching negatives on grounded ice "
              f"(grounded_frac > {MATCH_GROUNDED_MIN}), {args.match_neg_speed} bins:")
        pos, neg, spare = match_by_speed(tiles(pos_m), tiles(neg_m), vel,
                                        args.match_neg_speed, args.seed + 3)
    else:
        # hard positives first so they always survive an --n-per-class cap
        pos = (pick(np.argwhere(hard_pos), n, args.min_sep_pos, args.seed + 2)
               if args.amb_as_pos_min > 0 else [])
        rest = n - len(pos) if n else 0
        if not n or rest > 0:
            pos = pos + pick(np.argwhere(core_pos), rest, args.min_sep_pos, args.seed)
        # with a cap, match negatives to positives so the set is balanced; with
        # --n-per-class 0 emit every qualifying tile and let the caller rebalance
        neg = pick(np.argwhere(neg_m), n, args.min_sep_neg, args.seed + 1)
        if n and (len(pos) < n or len(neg) < n):
            print(f"\nNOTE: draw limited (pos {len(pos)}, neg {len(neg)} of {n} requested) "
                  f"by availability or --min-sep-pos/--min-sep-neg.")

    ts = datetime.now(timezone.utc).isoformat()

    def build_rows(groups):
        out_rows = []
        for lab, sel in groups:
            for y, x in sel:
                f = float(aoi_frac[y, x])
                rec = {
                    "granule": args.granule,
                    "row": int(y) * args.tile_size,
                    "col": int(x) * args.tile_size,
                    "label": lab,
                    "aoi_pos_frac": round(f, 6),
                    "data_frac": round(float(data_frac[y, x]), 4),
                    "tier": ("hard" if (lab == "crevasse" and f <= args.pos_thresh)
                             else "core"),
                    "dist_km": round(float(dist[y, x]) * km, 1),
                }
                if gfrac is not None:
                    v = float(vel[y, x])
                    rec["grounded_frac"] = round(float(gfrac[y, x]), 3)
                    if bcls is not None:
                        rec["ice_condition"] = cls_names.get(int(bcls[y, x]), "unknown")
                    rec["vel_m_yr"] = None if not np.isfinite(v) else round(v, 1)
                rec["source"] = "alphaearth"
                rec["timestamp"] = ts
                out_rows.append(rec)
        out_rows.sort(key=lambda r: (r["row"], r["col"]))
        return out_rows

    def write_csv(path, out_rows):
        with open(path, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            wr.writeheader()
            wr.writerows(out_rows)

    rows = build_rows((("crevasse", pos), ("none", neg)))

    write_csv(out, rows)
    print(f"\nwrote {out}  ({len(pos)} crevasse / {len(neg)} none)")

    if spare:
        side = out.with_name(out.stem + "_fastonly.csv")
        srows = build_rows((("crevasse", spare),))
        write_csv(side, srows)
        sv = [r["vel_m_yr"] for r in srows if r["vel_m_yr"] is not None]
        print(f"wrote {side}  ({len(spare)} positives in excess of the negatives "
              f"available in their speed bin, {min(sv):.0f}-{max(sv):.0f} m/yr)")
        print("  unbalanced by construction, so this set supports a recall number "
              "only -- never an AUC.")
    if not args.match_neg_speed:
        print("rerun with a larger --n-per-class to extend this set (prefix-stable).")


if __name__ == "__main__":
    main()
