"""Run the frozen RF binary gate over one granule's 512 tiles and render a map:
a downsampled basemap of the raster with crevasse-positive tiles colored by
P(crevasse). This is the stage-3 gate visualized across a whole scene.

    conda run -n nisar-gate python src/map_crevasse_tiles.py --granule 025_091 --grounded-only

Empty tiles are pre-filtered with the coarse data mask (from find_data_bounds)
so the gate only runs where there is actually swath data.

--grounded-only applies Bedmap3 as a hard mask on the decision: shelf, ocean and rock
tiles are still scored and saved, but never returned as positives. Using the physical
rasters this way -- a deterministic filter, not learned features -- is deliberate: a
context model fitted on ice speed was measured to be a spatial memorizer (raw row/col
reproduced its skill), so speed must not enter the decision. See --context-prior.
"""
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import joblib
import rasterio
import rasterio.enums
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from tqdm import tqdm

import train_gate_classifier as T
from find_data_swath import find_data_bounds, get_valid_tile_positions

ROOT = T.E.ROOT          # honours the GATE_ROOT override; see gate_common
NBINS = T.D * T.D

# Per-worker globals (rasterio datasets / sklearn models don't pickle well).
_SRC = None
_BUNDLE = None
_HANDKEYS = None


def _init_worker(raster_path, model_path):
    global _SRC, _BUNDLE, _HANDKEYS
    import joblib
    _SRC = rasterio.open(raster_path)
    _BUNDLE = joblib.load(model_path)
    # We already parallelize across tiles at the process level; keep the RF
    # single-threaded inside each worker (avoids the loky nested-parallel warning).
    try:
        _BUNDLE["model"].n_jobs = 1
    except Exception:
        pass
    # Bundles trained before ctx features stored no hand_keys; for those the tail
    # slice is still correct, because nothing followed the FFT block.
    _HANDKEYS = _BUNDLE.get("hand_keys") or _BUNDLE["feature_names"][:-NBINS]


def _featurize(pos):
    """Return (row, col, feature_vector) or (row, col, None) if unscoreable.

    Prediction happens in the main process so optional per-granule feature
    alignment can use whole-granule statistics before the RF runs.
    """
    row, col = pos
    amp = T.read_amp(_SRC, row, col)
    if amp is None:
        return (row, col, None)
    out = T.featurize(amp)
    if out is None:
        return (row, col, None)
    hand, ff = out
    x = np.array([hand[k] for k in _HANDKEYS] + list(ff), dtype=np.float32)
    return (row, col, x)


def raster_for(granule):
    for p in sorted((ROOT / "data" / "nisar").glob("*.tif")):
        if f"GSLC_{granule}_" in p.name:
            return str(p)
    return None


def prefilter_positions(positions, mask, downsample, tile_size, min_frac=0.3):
    """Keep only tiles whose coarse-mask footprint has >= min_frac data."""
    mh, mw = mask.shape
    step = max(1, tile_size // downsample)
    kept = []
    for row, col in positions:
        r0, c0 = row // downsample, col // downsample
        block = mask[r0:min(r0 + step, mh), c0:min(c0 + step, mw)]
        if block.size and block.mean() >= min_frac:
            kept.append((row, col))
    return kept


def build_basemap(raster_path, downsample):
    with rasterio.open(raster_path) as src:
        H, W = src.height, src.width
        data = src.read(1, out_shape=(H // downsample, W // downsample),
                        resampling=rasterio.enums.Resampling.average)
    m = data > 0
    lg = np.full(data.shape, np.nan, dtype=np.float32)
    lg[m] = np.log10(data[m] + 1e-10)
    if m.any():
        p2, p98 = np.nanpercentile(lg, [2, 98])
        lg = np.clip((lg - p2) / (p98 - p2 + 1e-9), 0, 1)
    return lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granule", default="025_091",
                    help="granule id like 025_091, or pass --raster")
    ap.add_argument("--raster", default=None, help="explicit .tif path (overrides --granule)")
    ap.add_argument("--gate-model", default=str(ROOT / "data" / "gate_5m_freqA_2gran.joblib"))
    ap.add_argument("--gate-thresh", type=float, default=0.65)
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--downsample", type=int, default=100,
                    help="basemap + prefilter downsample factor")
    ap.add_argument("--max-tiles", type=int, default=None, help="cap for a quick test")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--normalize-per-granule", action="store_true",
                    help="align this granule's feature distribution to the training "
                         "distribution before scoring (fixes cross-granule/band domain "
                         "shift; needs a bundle trained with train_feat_center/scale).")
    ap.add_argument("--allow-unseen-spacing", action="store_true",
                    help="score a granule whose pixel spacing never contributed a crevasse "
                         "label. Off by default because the resulting keep rate is not "
                         "interpretable; use only for a deliberately labelled experiment.")
    ap.add_argument("--grounded-only", action="store_true",
                    help="Bedmap3 hard mask: score every tile but return only grounded-ice "
                         "tiles as positives. Needs _context_masks_<granule>_t512.npz.")
    ap.add_argument("--context-prior", default=None,
                    help="DISCREDITED, do not use. A context-only bundle whose P(crevasse) "
                         "was multiplied into the SAR probability. Its +0.027 held-out gain "
                         "was measured to be memorized geography: binning ice speed into 8 "
                         "quantiles drops the gain to +0.001, and raw (row, col) with no "
                         "physics scores +0.031. Use --grounded-only instead.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-scores", default=None,
                    help="also write per-tile (row, col, prob) to this .npz")
    args = ap.parse_args()

    raster_path = args.raster or raster_for(args.granule)
    if raster_path is None:
        raise SystemExit(f"no raster found for granule {args.granule}")
    if not Path(args.gate_model).exists():
        raise SystemExit(f"gate model not found: {args.gate_model}")
    if args.tile_size != 512:
        raise SystemExit("gate is 512-tile-specific (corner-window FFT); use --tile-size 512")
    # Checked before featurizing, which costs minutes. A bundle trained under a different
    # looks policy scores its own training granule at 95% positive, so the mismatch is
    # silent and catastrophic rather than merely inaccurate.
    bundle = joblib.load(args.gate_model)
    if bundle.get("looks_policy") != T.LOOKS_POLICY:
        raise SystemExit(
            f"{Path(args.gate_model).name} was trained with looks_policy="
            f"{bundle.get('looks_policy')!r} but read_amp now delivers "
            f"{T.LOOKS_POLICY!r}. Retrain: conda run -n nisar-gate python "
            f"src/train_gate_classifier.py --labels-csv <csv...> --out {args.gate_model}")
    if len(bundle.get("train_granules", [])) < 2:
        print(f"WARNING: gate trained on {bundle.get('train_granules')} only. A "
              f"single-family gate does not transfer -- expect false positives on other "
              f"pixel spacings unless you pass --normalize-per-granule.")

    workers = args.num_workers or max(1, cpu_count() - 1)
    out = args.out or str(ROOT / "data" / f"crevasse_map_{args.granule}.png")

    print(f"Raster: {Path(raster_path).name}")
    print(f"Gate: {Path(args.gate_model).name}  P>={args.gate_thresh}  workers={workers}")

    # read_amp resamples to a common 5 m, so on a 2.5 m granule one tile consumes
    # 2*TS native pixels. Stride must follow, or tiles overlap and the rectangles
    # drawn below are half their true footprint.
    with rasterio.open(raster_path) as _s:
        scale = T.tile_scale(_s)
        px = _s.transform.a
    native = args.tile_size * scale
    print(f"Pixel {px:g} m -> tile stride {native} px "
          f"({native * px / 1000:.2f} km ground, scale={scale})")

    # A spacing that only ever supplied negatives taught the gate what to reject, never
    # what a crevasse looks like there. 003_064 is why: its `crevasse` labels were an
    # azimuth streak (R=0.99 scene-wide), so a 2.5 m keep rate measures the artifact.
    seen = bundle.get("train_px_m_positive")
    if seen and not any(abs(px - s) < 0.01 for s in seen):
        msg = (f"{Path(args.gate_model).name} has crevasse labels only at "
               f"{seen} m spacing; {args.granule} is {px:g} m.")
        if not args.allow_unseen_spacing:
            raise SystemExit(msg + " Refusing to score -- the keep rate would not be "
                                   "interpretable. Pass --allow-unseen-spacing to override.")
        print(f"WARNING: {msg} Scoring anyway (--allow-unseen-spacing).")

    print("Finding data swath...")
    bounds, mask = find_data_bounds(raster_path, downsample=args.downsample)
    positions = get_valid_tile_positions(bounds, native)
    positions = prefilter_positions(positions, mask, args.downsample, native)
    if args.max_tiles:
        positions = positions[:args.max_tiles]
    print(f"Scoring {len(positions)} candidate tiles with data...")

    with Pool(workers, initializer=_init_worker,
              initargs=(raster_path, args.gate_model)) as pool:
        results = list(tqdm(pool.imap(_featurize, positions, chunksize=8),
                            total=len(positions), desc="Featurize"))

    feats = [(r, c, x) for r, c, x in results if x is not None]
    if not feats:
        raise SystemExit("no scoreable tiles with data")
    X = np.stack([x for _, _, x in feats])
    rc = [(r, c) for r, c, _ in feats]

    meta = [(args.granule, r, c) for r, c in rc]
    if args.normalize_per_granule:
        if "train_feat_center" not in bundle:
            raise SystemExit(
                "--normalize-per-granule needs a bundle trained with the updated "
                "train_gate_classifier.py (missing train_feat_center/scale). Retrain.")
        # Only the SAR block is aligned. Context features are physical quantities
        # (surface class, ice speed), not speckle statistics, so mapping them onto the
        # training scene's moments would erase the very signal they carry.
        ns = X.shape[1]
        gc, gs = T.robust_stats(X)
        X = T.align_features(X, gc, gs, bundle["train_feat_center"][:ns],
                             bundle["train_feat_scale"][:ns])
        print("Per-granule feature alignment: ON (SAR block only)")
    if bundle.get("ctx_keys"):
        X = np.hstack([X, T.context_matrix(meta, bundle["ctx_keys"])])
    probs = bundle["model"].predict_proba(X)[:, 1]

    # Bedmap3 hard mask. Read through context_features so the row//TS lookup is
    # identical to the one the classifier was trained with.
    grounded = np.ones(len(rc), bool)
    if args.grounded_only:
        grounded = T.context_matrix(meta, ["ctx_is_grounded"])[:, 0] > 0.5
        print(f"Grounded mask: ON — {int((~grounded).sum())} of {len(grounded)} scored "
              f"tiles excluded as non-grounded ice")

    if args.context_prior:
        print("WARNING: --context-prior is discredited (memorized geography, not "
              "physics).\n  Its skill is reproduced by raw tile coordinates. "
              "Prefer --grounded-only.")
        cb = joblib.load(args.context_prior)
        if cb.get("feature_set") != "context":
            raise SystemExit(f"{args.context_prior} is feature_set="
                             f"{cb.get('feature_set')!r}, expected 'context'")
        pc = cb["model"].predict_proba(T.context_matrix(meta, cb["ctx_keys"]))[:, 1]
        q1, q3 = np.percentile(pc, [25, 75])
        print(f"Context prior: ON ({Path(args.context_prior).name}, "
              f"median P_ctx={np.median(pc):.3f}, IQR={q3 - q1:.3f})")
        # A forest cannot extrapolate. If this granule's speeds sit outside the range
        # the prior was fit on, every tile lands in the same leaves and P_ctx is a
        # constant -- which leaves the SAR ranking untouched and is only a relabelled
        # threshold, not evidence about this scene.
        if q3 - q1 < 0.02:
            eq = args.gate_thresh * (1 - np.median(pc)) / (
                args.gate_thresh * (1 - np.median(pc))
                + (1 - args.gate_thresh) * np.median(pc) + 1e-12)
            print(f"  WARNING: P_ctx is effectively constant on this granule, so the "
                  f"prior adds no\n  per-tile information -- it is exactly equivalent to "
                  f"--gate-thresh {eq:.3f} with no\n  prior. The context model is "
                  f"extrapolating outside its training speed range.")
        num = probs * pc
        probs = num / (num + (1 - probs) * (1 - pc) + 1e-12)

    scored = [(rc[i][0], rc[i][1], float(probs[i])) for i in range(len(rc))]
    pos_tiles = [(r, c, p) for (r, c, p), g in zip(scored, grounded)
                 if p >= args.gate_thresh and g]
    excluded = [(r, c) for (r, c, p), g in zip(scored, grounded)
                if p >= args.gate_thresh and not g]
    print(f"Scored {len(scored)} tiles; {len(pos_tiles)} crevasse-positive "
          f"(P>={args.gate_thresh})")
    if args.grounded_only:
        print(f"  {len(excluded)} tiles passed the threshold but were masked out as "
              f"non-grounded")

    if args.save_scores:
        np.savez_compressed(
            args.save_scores,
            row=np.array([r for r, _, _ in scored], np.int32),
            col=np.array([c for _, c, _ in scored], np.int32),
            prob=np.array([p for _, _, p in scored], np.float32),
            # saved per tile so downstream consumers mask consistently instead of
            # re-deriving the lookup
            grounded=grounded,
            grounded_only=bool(args.grounded_only),
            granule=args.granule, tile_size=args.tile_size,
            stride=native, pixel_m=px,
            gate_model=Path(args.gate_model).name, gate_thresh=args.gate_thresh,
            normalized=bool(args.normalize_per_granule),
            context_prior=Path(args.context_prior).name if args.context_prior else "")
        print(f"Saved scores: {args.save_scores}")

    print("Building basemap...")
    base = build_basemap(raster_path, args.downsample)
    ds, ts = args.downsample, native
    tw = ts / ds

    fig, ax = plt.subplots(figsize=(14, 14 * base.shape[0] / max(base.shape[1], 1)))
    ax.imshow(base, cmap="gray", interpolation="nearest")

    # Faint outline of every scored (has-data) tile for context.
    ctx = [Rectangle((c / ds, r / ds), tw, tw) for r, c, _ in scored]
    ax.add_collection(PatchCollection(ctx, facecolor="none", edgecolor="cyan",
                                      linewidth=0.15, alpha=0.25))

    # Tiles that cleared the threshold but sit off grounded ice: shown, so the mask is
    # visible as a decision rather than silently absent.
    if excluded:
        ex = [Rectangle((c / ds, r / ds), tw, tw) for r, c in excluded]
        ax.add_collection(PatchCollection(ex, facecolor="#5b7c99", edgecolor="none",
                                          alpha=0.55))

    # Positive tiles filled, colored by probability.
    cmap = plt.get_cmap("autumn_r")
    if pos_tiles:
        probs = np.array([p for _, _, p in pos_tiles])
        norm = plt.Normalize(vmin=args.gate_thresh, vmax=1.0)
        rects = [Rectangle((c / ds, r / ds), tw, tw) for r, c, _ in pos_tiles]
        pc = PatchCollection(rects, cmap=cmap, norm=norm, alpha=0.6,
                             edgecolor="red", linewidth=0.2)
        pc.set_array(probs)
        ax.add_collection(pc)
        cbar = fig.colorbar(pc, ax=ax, fraction=0.03, pad=0.01)
        cbar.set_label("P(crevasse)")

    ax.set_title(f"Crevasse gate map — {args.granule}\n"
                 f"{len(pos_tiles)} positive / {len(scored)} data tiles "
                 f"(P>={args.gate_thresh}, {ts}px tiles)"
                 + (f"\n{len(excluded)} masked out as non-grounded (blue-grey)"
                    if args.grounded_only else ""))
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
