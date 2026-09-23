"""Train and freeze the binary crevasse gate.

The gate is the cheap pre-filter that runs before the expensive Frangi/U-Net
stage: it passes through any tile that might contain a crevasse and skips the
rest. It is deliberately tuned for HIGH RECALL on the crevasse class so real
crevasses are rarely dropped.

Features (per tile), all on an NL-means despeckled log-amplitude image:
  * hand-14 GLCM + structure-tensor texture features
  * 4x384 corner-window FFT log-magnitude, max-pooled and downsampled to 16x16
    (256 bins) -- captures oriented ridge structure the hand features miss.
NL-means is a cheap speckle filter (~20ms/tile). PPB is stronger but ~40x
heavier, so it is reserved for the downstream ridge stage, not this gate.

Training:
  * fits RF on every labeled tile in --labels-csv
  * picks the decision threshold at a target recall using out-of-fold
    cross-validated probabilities (honest: the threshold is not chosen on the
    same predictions the RF memorized)
  * saves model + threshold + feature order + metadata to --out (joblib)

    conda run -n nisar-gate python src/train_gate_classifier.py --target-recall 0.95

Inference on a single tile:
    conda run -n nisar-gate python src/train_gate_classifier.py \
        --predict --granule 025_091 --row 56156 --col 26140
"""
import argparse, csv, json, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from numpy.fft import fft2, fftshift
import cv2
import rasterio
from rasterio.windows import Window
from scipy.ndimage import median_filter
from skimage.restoration import denoise_nl_means
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_val_predict)
from sklearn.metrics import roc_auc_score, precision_recall_curve
from joblib import Parallel, delayed
import joblib
import crevasse.nisar.gate_common as E
DEFAULT_MODEL = Path(__file__).resolve().parents[3] / "models" / "nisar" / "gate" / "gate_5m_freqA_2gran.joblib"
D, S = 16, 384
HANN_S = np.outer(np.hanning(S), np.hanning(S)).astype(np.float32)

# Every feature here is scale-dependent in PIXELS (GLCM offsets 1 and 3, the
# structure tensor, the S-px FFT window), but the archive mixes 5.0 m granules
# (025_*) with 2.5 m ones (003_064, 004_048). Tiles are resampled to this common
# ground spacing so a given physical crevasse yields the same feature vector
# regardless of which granule family it came from.
TARGET_M = 5.0

# Common ground spacing does NOT make the families comparable -- reaching 5 m from 2.5 m
# native averages 2x2 pixels (4 looks) while a 5 m granule arrives at 1 look -- but
# equalizing looks was implemented and measured not to fix it. A same-grid boxcar to 4
# looks did match log dynamic range (0.55 on all four granules) and still left the 2.5 m
# granules firing on 100% of tiles vs 30% on the 5 m ones; matching residual speckle
# correlation instead was worse (med P 0.57 -> 0.77). A 2.5 m granule genuinely has 4
# INDEPENDENT looks at 5 m resolution, while a 5 m granule can only fake looks by
# blurring, which destroys real structure along with speckle -- the asymmetry is
# informational and no resampling closes it. So looks are left at whatever the resample
# to TARGET_M delivers, and the gate is instead trained across both families
# (data/tile_labels.csv spans them). See the research repo's _looks_policy_check.py and
# _looks_vs_correlation.py probes.
LOOKS_POLICY = "native"

# Multipliers on each tile's native effective looks, added as extra TRAINING copies, so
# the gate is at least not brittle to a looks level no labelled granule happens to have.
AUGMENT_LOOKS_MULT = [4]

CTX_KEYS = ["ctx_grounded_frac", "ctx_is_grounded", "ctx_is_floating",
            "ctx_log_vel_mean", "ctx_log_vel_max", "ctx_vel_missing"]
_CTX_CACHE = {}


def normalize_log(Lg):
    Lg = np.nan_to_num(Lg, nan=np.nanmedian(Lg[np.isfinite(Lg)]))
    p2, p98 = np.percentile(Lg, [2, 98])
    return None if p98 <= p2 else np.clip((Lg - p2) / (p98 - p2), 0, 1).astype(np.float32)


def nlm_image(amp):
    """Amplitude tile (zeros->nan) -> NL-means despeckled, [0,1]-normalized log image."""
    Lg = np.log10(amp)
    med = np.nanmedian(Lg[np.isfinite(Lg)])
    Ld = np.nan_to_num(Lg, nan=med)
    d = Ld - median_filter(Ld, 3)
    sig = 1.4826 * np.median(np.abs(d - np.median(d))) + 1e-6
    Ld = denoise_nl_means(Ld, h=0.9 * sig, sigma=sig, fast_mode=True,
                          patch_size=5, patch_distance=3)
    return normalize_log(Ld)


def fft_pool(img):
    pos = [(0, 0), (0, 512 - S), (512 - S, 0), (512 - S, 512 - S)]
    specs = [cv2.resize(np.log1p(np.abs(fftshift(fft2(img[r:r + S, c:c + S] * HANN_S)))).astype(np.float32),
                        (D, D), interpolation=cv2.INTER_AREA) for r, c in pos]
    return np.stack(specs).max(axis=0).ravel()   # 256


def tile_scale(src):
    """Native pixels per resampled pixel, so a tile spans TS*TARGET_M m of ground.

    Callers that step across a granule must stride by TS*tile_scale native pixels,
    not TS, or consecutive tiles overlap.
    """
    ax, ay = src.transform.a, -src.transform.e
    if abs(ax - ay) > 1e-6:
        # One scale from transform.a alone stretches the tile: a 2.5 x 5.0 m raster came
        # out as 5 x 10 m pixels over 2.56 x 5.12 km, giving every scale-dependent feature
        # a built-in preferred direction. See data/nisar/lsar_defective/README.md.
        raise SystemExit(
            f"{Path(src.name).name} is {ax:g} x {ay:g} m -- non-square. read_amp derives "
            f"one scale from transform.a, so a tile would span {E.TS * TARGET_M:g} x "
            f"{E.TS * TARGET_M * ay / ax:g} m and every scale-dependent feature would see "
            f"a built-in preferred direction. Resample to square before use.")
    return max(1, int(round(TARGET_M / ax)))


def granule_px(granules):
    """Native pixel spacing in m for each named granule, read from its GeoTIFF."""
    out = {}
    for p in sorted(E.GRANULES.glob("*.tif")):
        g = E.GRANULE_RE.search(p.name).group(1)
        if g in granules:
            with rasterio.open(p) as s:
                out[g] = round(float(s.transform.a), 4)
    return out


def boxcar_looks(t, k):
    """nan-aware k x k mean at the SAME grid: multiplies effective looks by k*k.

    Speckle is multiplicative and roughly uncorrelated between native pixels, so
    averaging k*k of them cuts its std by k. Unlike block-averaging this keeps the
    output grid, so it changes neither the tile footprint the label CSVs are anchored
    to nor the pixel spacing the features assume. The cost is that the residual speckle
    is left spatially correlated rather than decimated.

    Edge-padded rather than wrapped, so the tile border does not mix opposite edges.
    """
    p = np.pad(t, ((0, k - 1), (0, k - 1)), mode="edge")
    h, w = t.shape
    stack = np.stack([p[i:i + h, j:j + w] for i in range(k) for j in range(k)])
    with warnings.catch_warnings():
        # all-nan windows are expected inside the nodata region; they stay nan
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=0).astype(np.float32)


def read_amp(src, row, col):
    """Amplitude tile at TARGET_M ground spacing, or None if too little valid data.

    The read window is scaled so the tile always covers TS*TARGET_M metres of ground.
    Zeros become nan and nan-aware averaging keeps nodata from bleeding into valid
    pixels. Effective looks are whatever that resample delivers (1 from a 5 m granule,
    4 from a 2.5 m one) -- see LOOKS_POLICY for why they are not equalized.
    """
    scale = tile_scale(src)
    n = E.TS * scale
    t = src.read(1, window=Window(int(col), int(row), n, n),
                 boundless=True, fill_value=0).astype(np.float32)
    m = t > 0
    if m.mean() < 0.5:
        return None
    t[~m] = np.nan
    if scale > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-nan blocks stay nan
            t = np.nanmean(t.reshape(E.TS, scale, E.TS, scale), axis=(1, 3))
    return t


def featurize(amp):
    """Amplitude tile -> (hand_dict, fft_vector) on the NL-means image, or None."""
    img = nlm_image(amp)
    if img is None:
        return None
    return E.glcm_st_features(img), fft_pool(img)


def feature_names(hand_keys):
    return list(hand_keys) + [f"fft{i}" for i in range(D * D)]


def stack_features(feats):
    hand_keys = list(feats[0][0])
    X = np.array([[h[k] for k in hand_keys] + list(ff) for h, ff in feats])
    return X, hand_keys


def context_masks(granule):
    if granule not in _CTX_CACHE:
        p = E.ROOT / "data" / f"_context_masks_{granule}_t{E.TS}.npz"
        if not p.exists():
            raise SystemExit(f"missing {p.name} -- run "
                             f"build_context_masks.py --granule {granule}")
        _CTX_CACHE[granule] = np.load(p)
    return _CTX_CACHE[granule]


def context_features(granule, row, col):
    """Bedmap3 surface class and ITS_LIVE speed for one tile, on the tile grid.

    The masks are built on TS *native* pixel cells, so on a 2.5 m granule one cell is
    1.28 km while a resampled tile spans 2.56 km: the value returned is then the
    tile's leading quadrant rather than its mean. Exact at 5 m (scale 1), and both
    fields are smooth enough at 120-500 m source resolution for the difference not to
    matter at the prior's precision.
    """
    z = context_masks(granule)
    g = z["grounded_frac"]
    i = min(int(row) // E.TS, g.shape[0] - 1)
    j = min(int(col) // E.TS, g.shape[1] - 1)
    vm, vx = float(z["vel_mean"][i, j]), float(z["vel_max"][i, j])
    cls = int(z["bedmap_class"][i, j])
    lg = lambda v: -1.0 if not np.isfinite(v) else float(np.log10(max(v, 1e-3)))
    return {
        "ctx_grounded_frac": float(g[i, j]),
        "ctx_is_grounded": float(cls == 1),
        "ctx_is_floating": float(cls in (2, 3)),
        "ctx_log_vel_mean": lg(vm),
        "ctx_log_vel_max": lg(vx),
        "ctx_vel_missing": float(not np.isfinite(vm)),
    }


def context_matrix(meta, keys=None):
    keys = keys or CTX_KEYS
    return np.array([[context_features(g, r, c)[k] for k in keys]
                     for g, r, c in meta], dtype=np.float64)


def build_matrix(feats, meta, bundle):
    """Assemble a feature matrix in the exact order a saved bundle expects.

    feats is [(hand_dict, fft_vector)], meta is [(granule, row, col)].
    """
    X = np.array([[h[k] for k in bundle["hand_keys"]] + list(ff) for h, ff in feats])
    ctx_keys = bundle.get("ctx_keys") or []
    if not ctx_keys:
        return X
    C = context_matrix(meta, ctx_keys)
    return C if bundle.get("feature_set") == "context" else np.hstack([X, C])


def read_label_rows(csv_paths):
    """crevasse/none rows from one or more label CSVs, deduped by (granule, row, col).

    The AlphaEarth-derived CSVs overlap the hand-reviewed tile_labels.csv on 025_091, so
    training on several at once would otherwise weight shared tiles twice. Earlier paths
    win, and a tile whose label disagrees between CSVs is dropped rather than silently
    resolved -- a disagreement means at least one of the two is wrong.
    """
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]
    seen, conflicts = {}, 0
    for p in csv_paths:
        for r in csv.DictReader(open(p)):
            if r["label"] not in ("crevasse", "none"):
                continue
            key = (r["granule"], int(r["row"]), int(r["col"]))
            if key in seen:
                if seen[key]["label"] != r["label"]:
                    seen[key] = None
                    conflicts += 1
                continue
            seen[key] = r
    if conflicts:
        print(f"  dropped {conflicts} tiles whose label disagrees between CSVs")
    return [r for r in seen.values() if r is not None]


def load_raw_features(csv_paths, augment_looks_mult=()):
    """feats, labels, (granule, row, col), and a native-row mask per surviving tile.

    Kept separate from load_features so several saved bundles can be scored on the
    identical row set without re-reading and re-featurizing every tile.

    `augment_looks_mult` adds an extra copy of each tile with its effective looks
    multiplied by that factor. The copies keep the ORIGINAL (granule, row, col), so
    spatial_groups puts every copy of a tile in one CV fold and out-of-fold scores stay
    honest. The returned mask is False on those copies: callers that need statistics of
    the real feature distribution (robust_stats, for align_features) must use only the
    native rows.
    """
    rows = read_label_rows(csv_paths)
    rasters = E.granule_rasters()
    amps, y, meta, native = [], [], [], []
    for r in rows:
        amp = read_amp(rasters[r["granule"]], r["row"], r["col"])
        if amp is None:
            continue
        m = (r["granule"], int(r["row"]), int(r["col"]))
        lab = 1 if r["label"] == "crevasse" else 0
        for mult in (1,) + tuple(augment_looks_mult):
            k = int(round(mult ** 0.5))
            amps.append(amp if k <= 1 else boxcar_looks(amp, k))
            y.append(lab)
            meta.append(m)
            native.append(mult == 1)
    for s in rasters.values():
        s.close()
    feats = Parallel(n_jobs=-1)(delayed(featurize)(a) for a in amps)
    keep = [(f, yi, m, nt) for f, yi, m, nt in zip(feats, y, meta, native)
            if f is not None]
    return ([t[0] for t in keep], np.array([t[1] for t in keep]),
            [t[2] for t in keep], np.array([t[3] for t in keep], bool))


def load_features(csv_paths, feature_set="sar", augment_looks_mult=()):
    """Feature matrix, labels, names, (granule, row, col), and a native-row mask.

    SAR features are computed even for feature_set="context" (where they go unused)
    so that every variant is fit on the identical row set -- read_amp and featurize
    both drop tiles, and a fair comparison needs those drops to match.
    """
    feats, y, meta, native = load_raw_features(csv_paths, augment_looks_mult)
    X, hand_keys = stack_features(feats)
    fn, ctx_keys = feature_names(hand_keys), []
    if feature_set in ("context", "fused"):
        C = context_matrix(meta)
        ctx_keys = list(CTX_KEYS)
        X, fn = (C, ctx_keys) if feature_set == "context" else (
            np.hstack([X, C]), fn + ctx_keys)
    return X, y, fn, meta, hand_keys, ctx_keys, native


def spatial_groups(meta, block_tiles=8):
    """Group id per tile for blocked CV: whole (block_tiles x block_tiles) blocks.

    Ice speed and SAR texture are both spatially smooth, so a randomly held-out
    tile usually has a training neighbour a few hundred metres away. Holding out
    whole ~20 km blocks instead makes the reported AUC mean something.
    """
    step = E.TS * block_tiles
    return np.array([f"{g}_{r // step}_{c // step}" for g, r, c in meta])


def threshold_for_recall(y, proba, target):
    """Highest threshold whose recall on the positive class is >= target."""
    prec, rec, thr = precision_recall_curve(y, proba)
    idx = np.where(rec[:-1] >= target)[0]
    if len(idx) == 0:
        return 0.0
    return float(thr[idx[-1]])


def make_rf():
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                  class_weight="balanced", random_state=E.SEED, n_jobs=-1)


def robust_stats(X):
    """Per-feature robust center (median) and scale (MAD*1.4826) over rows of X."""
    center = np.median(X, axis=0)
    scale = 1.4826 * np.median(np.abs(X - center), axis=0) + 1e-6
    return center.astype(np.float32), scale.astype(np.float32)


def align_features(X, src_center, src_scale, dst_center, dst_scale):
    """Map a test granule's features into the training feature distribution.

    Standardizes X by the *source* (test-granule) robust moments, then restores
    the *destination* (training) moments: the bulk speckle background of any
    granule is aligned onto the training background, so an LSAR scene's speckle
    stops scoring as weakly-crevasse. Genuine crevasse tiles stay outliers
    (same #MADs above center) and remain in the crevasse region after mapping.
    """
    return (X - src_center) / src_scale * dst_scale + dst_center


def train(args):
    aug = tuple(AUGMENT_LOOKS_MULT) if args.augment_looks else ()
    X, y, fn, meta, hand_keys, ctx_keys, native = load_features(
        args.labels_csv, args.feature_set, aug)
    print(f"{len(y)} tiles: {y.sum()} crevasse / {(y==0).sum()} none   "
          f"({X.shape[1]} features, set={args.feature_set})")
    if aug:
        print(f"  {int(native.sum())} native-looks + {int((~native).sum())} "
              f"augmented copies at looks x{list(aug)}")
    gran = np.array([g for g, _, _ in meta])
    for g in sorted(set(gran)):
        b = gran == g
        print(f"  {g}: {int(b.sum()):5d} rows  {int(y[b].sum()):4d} crevasse / "
              f"{int((y[b] == 0).sum()):4d} none")

    # Honest threshold: choose it on out-of-fold predictions.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=E.SEED)
    oof_rand = cross_val_predict(make_rf(), X, y, cv=cv,
                                 method="predict_proba", n_jobs=-1)[:, 1]
    auc_rand = roc_auc_score(y, oof_rand)

    groups = spatial_groups(meta, args.block_tiles)
    gcv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=E.SEED)
    oof_sp = cross_val_predict(make_rf(), X, y, cv=gcv, groups=groups,
                               method="predict_proba", n_jobs=-1)[:, 1]
    auc_sp = roc_auc_score(y, oof_sp)
    print(f"OOF AUC  random-fold {auc_rand:.3f}   spatial-block {auc_sp:.3f}   "
          f"({len(set(groups))} blocks of {args.block_tiles} tiles)")
    print(f"  leakage from spatial autocorrelation: {auc_rand - auc_sp:+.3f}")

    # Leave-one-granule-out. Spatial blocks hold out ground, not domains: a pooled AUC
    # can look healthy while the gate is unusable on an unseen granule, which is exactly
    # how the cross-granule false-positive blowup went unnoticed. Only granules that
    # carry both classes are reported -- AUC is undefined otherwise.
    logo = {}
    if len(set(gran)) > 1:
        print("leave-one-granule-out (transfer, not just unseen ground):")
        for g in sorted(set(gran)):
            te = gran == g
            if len(set(y[te])) < 2 or len(set(y[~te])) < 2:
                print(f"  {g}: single-class fold, AUC undefined")
                continue
            p = make_rf().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
            logo[g] = float(roc_auc_score(y[te], p))
            print(f"  {g}: AUC {logo[g]:.3f}   med P  pos {np.median(p[y[te] == 1]):.3f}"
                  f" / neg {np.median(p[y[te] == 0]):.3f}")

    oof, auc = ((oof_sp, auc_sp) if args.spatial_cv else (oof_rand, auc_rand))
    print(f"  threshold + reported AUC from the "
          f"{'spatial-block' if args.spatial_cv else 'random'} folds")
    if args.threshold is not None:
        thr = float(args.threshold)
        thr_mode = f"explicit={thr:.3f}"
    else:
        thr = threshold_for_recall(y, oof, args.target_recall)
        thr_mode = f"recall>={args.target_recall:.2f}"
    keep = oof >= thr
    recall = (keep & (y == 1)).sum() / max((y == 1).sum(), 1)
    skip = (~keep).mean()
    neg_disc = ((~keep) & (y == 0)).sum() / max((y == 0).sum(), 1)
    print(f"OOF AUC={auc:.3f}   threshold ({thr_mode}): {thr:.3f}")
    print(f"  -> keeps {recall:.1%} of true crevasses, skips {skip:.1%} of all tiles, "
          f"discards {neg_disc:.1%} of negatives")

    # Final model trained on everything.
    model = make_rf().fit(X, y)
    px_by_granule = granule_px(set(gran))
    # align_features maps a test granule onto the training distribution, so its target
    # moments must come from the native-looks rows only.
    feat_center, feat_scale = robust_stats(X[native])
    bundle = {
        "model": model,
        "feature_names": fn,
        # stored explicitly: callers used to recover these by slicing
        # feature_names[:-(D*D)], which silently breaks once ctx features follow
        "hand_keys": hand_keys,
        "ctx_keys": ctx_keys,
        "feature_set": args.feature_set,
        "train_feat_center": feat_center,
        "train_feat_scale": feat_scale,
        "threshold": thr,
        "threshold_mode": thr_mode,
        "target_recall": args.target_recall,
        "oof_auc": auc,
        "oof_auc_random": auc_rand,
        "oof_auc_spatial": auc_sp,
        "cv": "spatial-block" if args.spatial_cv else "random",
        "block_tiles": args.block_tiles,
        "n_train": int(len(y)),
        "target_ground_m": TARGET_M,
        "looks_policy": LOOKS_POLICY,
        "augment_looks_mult": list(aug),
        "oof_auc_leave_one_granule_out": logo,
        "train_granules": sorted(set(gran)),
        "train_px_m": px_by_granule,
        # The spacings the gate has ever seen a CREVASSE at. A granule family that only
        # contributed negatives taught it what to reject, not what to find, so scoring a
        # new granule at that spacing is extrapolation -- map_crevasse_tiles enforces this.
        "train_px_m_positive": sorted({px_by_granule[g]
                                       for g, yi in zip(gran, y) if yi == 1}),
        "labels_csv": [str(p) for p in args.labels_csv],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": f"nlm-despeckle @{TARGET_M:g}m: hand-14 GLCM+ST + "
                    f"4x{S}-pool-max FFT({D}x{D})"
                    + (f" + ctx-{len(ctx_keys)}" if ctx_keys else "") + "; TS=512",
    }
    joblib.dump(bundle, args.out)
    print(f"saved gate -> {args.out}")
    print(json.dumps({k: bundle[k] for k in
                      ("threshold", "target_recall", "oof_auc", "n_train")}, indent=2))


def predict(args):
    bundle = joblib.load(args.out)
    src = None
    for p in sorted(E.GRANULES.glob("*.tif")):
        if E.GRANULE_RE.search(p.name).group(1) == args.granule:
            src = rasterio.open(p)
            break
    if src is None:
        raise SystemExit(f"granule {args.granule} not found under data/nisar/")
    amp = read_amp(src, args.row, args.col)
    src.close()
    if amp is None:
        raise SystemExit("tile has too little valid data to score")
    hand, ff = featurize(amp)
    X = build_matrix([(hand, ff)], [(args.granule, args.row, args.col)], bundle)
    p_crev = float(bundle["model"].predict_proba(X)[0, 1])
    keep = p_crev >= bundle["threshold"]
    print(f"P(crevasse)={p_crev:.3f}   threshold={bundle['threshold']:.3f}")
    print("decision: KEEP (send to Frangi/U-Net)" if keep else "decision: SKIP (no crevasse)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", nargs="+",
                    default=[str(E.ROOT / "data" / "tile_labels.csv")],
                    help="one or more label CSVs, deduped by (granule, row, col) with "
                         "earlier files winning; pass several to train across granule "
                         "families, which is what makes the gate transfer")
    ap.add_argument("--out", default=str(DEFAULT_MODEL))
    ap.add_argument("--feature-set", choices=("sar", "context", "fused"), default="sar",
                    help="sar = SAR texture only (default, reproduces the old gate); "
                         "context = Bedmap3/ITS_LIVE only, the floor any SAR model must "
                         "beat; fused = both")
    ap.add_argument("--spatial-cv", action="store_true",
                    help="take the threshold and reported AUC from spatial-block folds "
                         "instead of random folds; both are always printed")
    ap.add_argument("--block-tiles", type=int, default=8,
                    help="spatial-block side in tiles (8 = ~20 km at 5 m)")
    ap.add_argument("--augment-looks", action="store_true", default=True,
                    help=f"also train on copies of each tile with its effective looks "
                         f"multiplied by {AUGMENT_LOOKS_MULT}, so the gate is not brittle "
                         f"to a looks level no labelled granule happens to have")
    ap.add_argument("--no-augment-looks", action="store_false", dest="augment_looks")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--threshold", type=float, default=None,
                    help="explicit keep-threshold; overrides --target-recall")
    ap.add_argument("--predict", action="store_true", help="score one tile with the saved gate")
    ap.add_argument("--granule")
    ap.add_argument("--row", type=int)
    ap.add_argument("--col", type=int)
    args = ap.parse_args()
    if args.predict:
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
