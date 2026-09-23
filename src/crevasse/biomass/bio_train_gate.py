"""Train the independent BIOMASS tile gate on AlphaEarth labels over grounded ice.

    conda run -n biomass python code/bio_train_gate.py

Reads data/bio_tile_features_t512.npz, writes data/bio_gate_t512.joblib.

INDEPENDENT OF THE NISAR GATE, DELIBERATELY
-------------------------------------------
Not a fine-tune, not a shared model. The frozen NISAR gate scored AUC 0.560 on BIOMASS
with its probabilities collapsed into 0.445-0.538 -- its 270 texture features are out of
distribution and it fell back to its prior. And mixing sensor families is what produced
the leave-one-granule-out collapse to AUC 0.59/0.35 against a pooled 0.911 on NISAR. A
joint model can be revisited once each sensor works alone.

WHAT MUST BE BEATEN, NOT JUST REPORTED
--------------------------------------
Three baselines print alongside every score, because on this project each one has at some
point matched or beaten a model that looked good:

  [ratio_dB alone]  A single feature, AUC 0.890 on the 120-tile pilot. If the 38-feature
                    model cannot clear this, the extra features are decoration.
  [(x,y) alone]     Map coordinates with no radar input at all. On NISAR a raw (row,col)
                    model scored 0.934-0.955 and TIED the real model at 8-tile blocks,
                    which is how the context prior was found to be memorised geography.
                    On the BIOMASS pilot, row alone already gave 0.770.
  [base rate]       Positive fraction, so precision numbers are readable.

SPLITS
------
Spatial blocks are cut in MAP COORDINATES (x_m, y_m), not per-granule row/col. This matters
here more than it did on NISAR: BIOMASS S1 and S2 are exact repeat passes over identical
ground (identical dr/dc at M02 and M04), so a random or per-granule split would put the
same ground in train and test through a different acquisition.

  spatial-block CV   40.96 km blocks (16 tiles x 512 px x 5 m). 16 not 8 -- at 8 tiles the
                     NISAR gate's 0.932 was tied by (row,col) alone at 0.934.
  leave-one-footprint-out  M01 / M02 / M03 held out whole. The honest cross-ground test,
                     and the one that exposed the NISAR domain shift. Still optimistic:
                     every footprint is in the same Thwaites region.

LABELS
------
AlphaEarth, recall ~0.42 at precision ~0.94, so its zeros inside the export box are real
evaluated negatives but its positives are incomplete -- recall here is a lower bound.
Ambiguous tiles (positive fraction strictly between the two thresholds) are dropped rather
than forced to a side, and tiles outside the export box are dropped entirely because there
byte 0 means "never evaluated", not "no crevasse".
"""
import argparse
import os

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

from crevasse.biomass.bio_gate_controls import auc, buffered_folds, within_fold_auc
from crevasse.biomass.bio_gate_controls import rf as rf_eval

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "biomass", "gate")
T = 512
PX_M = 5.0

# Two forests on purpose, and the difference is load-bearing. rf() with 500 trees is what
# gets SHIPPED; rf_eval() with 300 is what bio_gate_controls.py used to measure the buffered
# numbers this repo quotes (0.889 pooled, 0.621/0.762/0.918 per fold). The buffered evaluation
# below therefore calls rf_eval, so those figures reproduce exactly. Swapping in rf() here
# would move them by more than the margin being reported, which is how a pinned control
# quietly stops pinning anything.


def rf(seed=0):
    return RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=seed)


def cv_scores(X, y, groups, names, xy):
    """Out-of-fold predictions for the model and for the (x,y)-only control."""
    oof = np.full(len(y), np.nan)
    oof_xy = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if y[tr].sum() < 10 or (~y[tr]).sum() < 10 or te.sum() == 0:
            continue
        if y[te].sum() == 0 or (~y[te]).sum() == 0:
            continue
        m = rf().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        mxy = rf(1).fit(xy[tr], y[tr])
        oof_xy[te] = mxy.predict_proba(xy[te])[:, 1]
    return oof, oof_xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(
        ROOT, "data", f"bio_tile_features_t{T}.npz"))
    ap.add_argument("--pos-thresh", type=float, default=0.5,
                    help="AlphaEarth positive fraction at or above this = crevassed")
    ap.add_argument("--neg-thresh", type=float, default=0.05,
                    help="at or below this = not crevassed; between the two is dropped")
    ap.add_argument("--block-tiles", type=int, default=16,
                    help="spatial CV block edge in tiles; 16 = 40.96 km. Do not use 8, "
                         "it was tied by coordinates alone on NISAR.")
    ap.add_argument("--buffer-block-tiles", type=int, default=32,
                    help="block edge in tiles for the BUFFERED split; 32 = 81.9 km")
    ap.add_argument("--buffer-tiles", type=int, default=16,
                    help="buffer width in tiles withheld from training; 16 = 41 km")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(ROOT))), "models", "biomass", "gate", f"bio_gate_t{T}.joblib"))
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X, names = d["X"], list(d["feature_names"])
    ap_ = d["aoi_pos"]
    keep = d["aoi_inbox"] & ((ap_ >= args.pos_thresh) | (ap_ <= args.neg_thresh))
    keep &= np.isfinite(X).all(axis=1)
    y = ap_[keep] >= args.pos_thresh
    X = X[keep]
    xm, ym = d["x_m"][keep], d["y_m"][keep]
    foot, gran = d["foot"][keep], d["granule"][keep]
    xy = np.c_[xm, ym].astype(np.float32)

    print(f"tiles: {len(y)}  positive {y.sum()} ({y.mean():.3f})  negative {(~y).sum()}")
    print(f"dropped: {(~d['aoi_inbox']).sum()} outside the export box, "
          f"{((ap_ > args.neg_thresh) & (ap_ < args.pos_thresh)).sum()} ambiguous")
    print("footprints: " + ", ".join(f"{f}={int((foot==f).sum())}"
                                     for f in sorted(set(foot))))

    print("\n--- baselines that have to be beaten ---")
    ri = names.index("ratio_dB")
    print(f"  base rate (positive fraction) = {y.mean():.3f}")
    print(f"  ratio_dB alone         AUC = {auc(y, X[:, ri]):.3f}   "
          f"(pilot value 0.890)")

    blk = args.block_tiles * T * PX_M
    gb = (np.floor(xm / blk).astype(int) * 100000
          + np.floor(ym / blk).astype(int))
    print(f"\n--- spatial-block CV, {blk/1000:.2f} km blocks, "
          f"{len(np.unique(gb))} blocks ---")
    oof, oof_xy = cv_scores(X, y, gb, names, xy)
    m = np.isfinite(oof)
    print(f"  model            AUC = {auc(y[m], oof[m]):.3f}   (n={m.sum()})")
    print(f"  (x,y) only       AUC = {auc(y[m], oof_xy[m]):.3f}   <- must be clearly lower")
    print(f"  ratio_dB alone   AUC = {auc(y[m], X[m, ri]):.3f}")
    margin = auc(y[m], oof[m]) - auc(y[m], oof_xy[m])
    print(f"  margin over pure geography = {margin:+.3f}")
    if margin < 0.03:
        print("  WARNING: the model barely beats map coordinates. On NISAR this was the "
              "signature\n           of a spatial memoriser, and it is why the context "
              "prior was dropped.")

    print("\n--- leave-one-footprint-out (the honest cross-ground test) ---")
    oof_f, oof_fxy = cv_scores(X, y, foot, names, xy)
    for f in sorted(set(foot)):
        s = (foot == f) & np.isfinite(oof_f)
        if s.sum() and y[s].any() and (~y[s]).any():
            print(f"  hold out {f}  n={s.sum():4d} pos={y[s].sum():4d}   "
                  f"model {auc(y[s], oof_f[s]):.3f}   "
                  f"(x,y) {auc(y[s], oof_fxy[s]):.3f}   "
                  f"ratio_dB {auc(y[s], X[s, ri]):.3f}")
    mf = np.isfinite(oof_f)
    print(f"  pooled LOFO      AUC = {auc(y[mf], oof_f[mf]):.3f}   "
          f"(x,y) {auc(y[mf], oof_fxy[mf]):.3f}")

    # ---- the buffered split, which is the only one whose numbers may be quoted ----
    # Everything above is UNBUFFERED and is kept only as the control that condemns it: the
    # model beats pure coordinates there by +0.016, i.e. it is almost entirely memorised
    # geography, because Thwaites crevasse fields are large and contiguous enough to
    # interpolate across a block seam. A buffer is what removes that.
    blk_b = args.buffer_block_tiles * T * PX_M
    buf_b = args.buffer_tiles * T * PX_M
    folds = list(buffered_folds(y, xm, ym, blk_b, buf_b))
    print(f"\n--- BUFFERED spatial-block CV, {blk_b/1000:.0f} km blocks / "
          f"{buf_b/1000:.0f} km buffer, {len(folds)} folds ---")
    oof_b = np.full(len(y), np.nan)
    oxy_b = np.full(len(y), np.nan)
    for te, tr in folds:
        oof_b[te] = rf_eval().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oxy_b[te] = rf_eval(1).fit(xy[tr], y[tr]).predict_proba(xy[te])[:, 1]
    mb = np.isfinite(oof_b)

    print(f"  {'fold':>5} {'n':>6} {'pos':>5} {'base':>6} {'model':>7} {'(x,y)':>7} "
          f"{'pairs':>9}")
    fold_auc, fold_xy, pairs = [], [], []
    for k, (te, _) in enumerate(folds):
        n1, n0 = int(y[te].sum()), int((~y[te]).sum())
        fold_auc.append(auc(y[te], oof_b[te]))
        fold_xy.append(auc(y[te], oxy_b[te]))
        pairs.append(n1 * n0)
        print(f"  {k:>5} {int(te.sum()):>6} {n1:>5} {y[te].mean():>6.3f} "
              f"{fold_auc[-1]:>7.3f} {fold_xy[-1]:>7.3f} {pairs[-1]:>9}")
    print("  ^ read this table BEFORE any summary below. A fold with 4 positives is not "
          "evidence,\n    and the pair counts show how little the three folds are worth "
          "against each other.")

    pooled, pooled_xy = auc(y[mb], oof_b[mb]), auc(y[mb], oxy_b[mb])
    wf, wf_xy = within_fold_auc(y, oof_b, folds), within_fold_auc(y, oxy_b, folds)
    unw, unw_xy = float(np.mean(fold_auc)), float(np.mean(fold_xy))
    print(f"\n  {'summary':>16} {'model':>7} {'(x,y)':>7} {'margin':>8}")
    for nm, a, b in [("pooled", pooled, pooled_xy), ("within-fold (pair-wtd)", wf, wf_xy),
                     ("unweighted fold mean", unw, unw_xy)]:
        print(f"  {nm:>16} {a:>7.3f} {b:>7.3f} {a - b:>+8.3f}")
    print("  THREE summaries, and on this data they do not agree on the sign of the margin.\n"
          "  Quote all three or none. Pooled AUC is not a held-out metric when fold base "
          "rates\n  differ this much: it pays a model for ranking one whole block above "
          "another.")

    # ---- what a fixed threshold would actually do, per fold ----
    # Deliberately NOT a threshold search. There is no F1 to maximise here that means
    # anything: precision at a fixed cut tracks the LOCAL BASE RATE, not skill, and these
    # folds run from 0.8% to 59% positive. Lift over base rate is the quantity that is
    # comparable across folds, so that is what is printed and shipped.
    print("\n--- what a fixed operating point does, per fold (NOT a recommendation) ---")
    print(f"  {'thr':>5} {'fold':>5} {'base':>6} {'flagged':>8} {'prec':>6} {'lift':>6}")
    for thr in (0.5, 0.65, 0.8):
        for k, (te, _) in enumerate(folds):
            s, yy = oof_b[te], y[te]
            p = s >= thr
            base = float(yy.mean())
            # nothing flagged means precision is UNDEFINED, not zero, and printing 0.00x
            # would read as "flagged everything wrong". A dash is the honest cell.
            if not p.any():
                print(f"  {thr:>5.2f} {k:>5} {base:>6.3f} {0:>8} {'--':>6} {'--':>6}"
                      f"   nothing flagged")
                continue
            prec = float(yy[p].mean())
            lift = prec / base if base > 0 else float("nan")
            print(f"  {thr:>5.2f} {k:>5} {base:>6.3f} {int(p.sum()):>8} "
                  f"{prec:>6.3f} {lift:>5.2f}x")
    print("  A fold where nothing is flagged and a fold where everything is are the same\n"
          "  miscalibration: the probability scale is fitted to ~26-58% positive training\n"
          "  ground. THE RANKING TRANSFERS, THE CALIBRATION DOES NOT -- so no threshold is\n"
          "  shipped, and the bundle carries no `thresh` key. See docs/RESULTS.md.")

    print("\n--- feature importance (full-data fit) ---")
    final = rf().fit(X, y)
    imp = final.feature_importances_
    order = np.argsort(imp)[::-1]
    for i in order[:12]:
        print(f"  {names[i]:>18} {imp[i]:.4f}")
    tex = [i for i, n in enumerate(names)
           if n.startswith(("acf", "grad"))]
    print(f"\n  PSF-axis texture features are {len(tex)}/{len(names)} of the set and "
          f"carry {imp[tex].sum():.3f} of total importance")
    print("  (isotropic texture scored at chance on the pilot; this is the check on "
          "whether\n   the along-vs-across-PSF version does any better)")

    # ---- the bundle ----
    # There is deliberately NO `thresh` key. An earlier version of this file wrote the best-F1
    # threshold over the UNBUFFERED out-of-fold scores (0.25), and a bundle field named
    # `thresh` reads as a shipped operating point no matter what the docs say. The per-fold
    # lift table above is what replaces it.
    #
    # The unbuffered figures are kept, because deleting them would lose the evidence for why
    # the buffered ones are the quotable set -- but they are named so they cannot be quoted by
    # accident.
    joblib.dump(dict(
        model=final, feature_names=names, tile_size=T, px_m=PX_M,
        pos_thresh=args.pos_thresh, neg_thresh=args.neg_thresh,
        sensor="BIOMASS_P_band_quadpol",
        n_train=int(len(y)), base_rate=float(y.mean()),
        # granule IDs are derived from filenames upstream and one of them used to keep a
        # stray '.tif'; strip it here so a downstream string match cannot miss that granule.
        trained_granules=sorted({str(g).replace(".tif", "") for g in gran.tolist()}),

        # --- quotable: the buffered split ---
        eval_block_tiles=args.buffer_block_tiles, eval_buffer_tiles=args.buffer_tiles,
        eval_block_km=blk_b / 1000, eval_buffer_km=buf_b / 1000,
        n_folds=len(folds), n_eval=int(mb.sum()),
        fold_n=[int(te.sum()) for te, _ in folds],
        fold_pos=[int(y[te].sum()) for te, _ in folds],
        fold_base_rate=[float(y[te].mean()) for te, _ in folds],
        fold_pairs=pairs,
        fold_auc=[float(a) for a in fold_auc], fold_auc_xy_control=[float(a) for a in fold_xy],
        cv_auc=float(pooled), cv_auc_xy_control=float(pooled_xy),
        cv_auc_within_fold=float(wf), cv_auc_within_fold_xy_control=float(wf_xy),
        cv_auc_unweighted_fold_mean=float(unw),
        cv_auc_unweighted_fold_mean_xy_control=float(unw_xy),

        # --- kept as evidence, NOT quotable ---
        cv_auc_unbuffered_discredited=float(auc(y[m], oof[m])),
        cv_auc_xy_control_unbuffered_discredited=float(auc(y[m], oof_xy[m])),
        block_tiles_unbuffered_discredited=args.block_tiles,
        lofo_auc_discredited=float(auc(y[mf], oof_f[mf])),
        lofo_auc_xy_control_discredited=float(auc(y[mf], oof_fxy[mf])),

        eval_estimator="RandomForest(300) from bio_gate_controls.rf",
        shipped_estimator="RandomForest(500)",
        # A pickled forest is version-coupled. Recorded so a mismatch produces a message
        # naming both versions instead of sklearn's generic "use at your own risk".
        sklearn_version=sklearn.__version__,
        numpy_version=np.__version__,
        provenance_note=(
            "No operating point ships. Precision at a fixed cut tracks the local base rate, "
            "not skill: at P>=0.65 this model flags 0 tiles on two of three folds. Use "
            "fold_auc with fold_pos beside it; cv_auc is POOLED across folds whose base "
            "rates run 0.008-0.590, so it is not a held-out metric and its margin over "
            "(x,y) flips sign between the three summaries. The *_unbuffered_discredited "
            "fields come from a split with no buffer, where the margin over map coordinates "
            "alone is +0.016 -- i.e. memorised geography. lofo_* is not a cross-ground test "
            "either: all BIOMASS footprints overlap the same Thwaites region and S1/S2 are "
            "exact repeat passes, so (x,y) alone scores 0.994 there and BEATS the model. "
            "All labelled ground is Thwaites. See docs/LIMITATIONS.md."),
    ), args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
