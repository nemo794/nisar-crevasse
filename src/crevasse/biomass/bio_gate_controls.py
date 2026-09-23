"""Does the BIOMASS gate know anything the map does not?

    conda run -n biomass python code/bio_gate_controls.py

The first training run failed its own controls, and this script is the follow-up:

    spatial-block CV, 40.96 km   model 0.954   (x,y) only 0.937   margin +0.017
    leave-one-footprint-out      model 0.953   (x,y) only 0.994   <- geography WINS

Leave-one-footprint-out is not a cross-ground test on this dataset. All four BIOMASS
footprints overlap the same Thwaites region, and S1/S2 are exact repeat passes over
identical ground, so holding out a footprint holds out an ACQUISITION, not a place. An
(x,y) model trained on the others has already seen the same ground.

The spatial-block margin of +0.017 is the number that matters, and it is far too small.
This is the same failure that killed the context prior on NISAR: crevasse fields are large
and contiguous, so a held-out block's labels are predictable by smooth interpolation from
its neighbours. The recorded lesson there was that excluding exact cells is insufficient
and a BUFFER is required.

So this sweeps two knobs and asks one question: is there a train/test separation at which
the radar features still beat pure coordinates?

  block size   how big a contiguous chunk of ground is held out at once
  buffer       how much extra ground around the test block is dropped from TRAINING,
               so the model cannot interpolate across the seam

Reported per setting:
  model      all 38 radar features
  (x,y)      map coordinates only, no radar input whatsoever -- the thing to beat
  ratio_dB   the single best pilot feature, as a sanity floor
  margin     model - (x,y). This is the only honest headline number.

If margin stays near zero at every separation, the gate is a spatial memoriser and must
not be shipped, regardless of how good its AUC looks.
"""
import argparse
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "biomass", "gate")
T = 512
PX_M = 5.0


def auc(y, s):
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(s)) + 1.0
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def rf(seed=0, n=300):
    return RandomForestClassifier(
        n_estimators=n, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=seed)


def buffered_folds(y, xm, ym, block_m, buffer_m):
    """Yield (test_mask, train_mask) per spatial block, training tiles near the block cut.

    Split out of buffered_cv so a non-sklearn model can be scored on the SAME folds; any
    change here changes every buffered number this project has quoted.
    """
    bx = np.floor(xm / block_m).astype(int)
    by = np.floor(ym / block_m).astype(int)
    blocks = list({(a, b) for a, b in zip(bx, by)})
    for (a, b) in blocks:
        te = (bx == a) & (by == b)
        if te.sum() < 20 or not y[te].any() or y[te].all():
            continue
        # test block extent, grown by the buffer; anything inside is withheld from train
        x0, x1 = xm[te].min() - buffer_m, xm[te].max() + buffer_m
        y0, y1 = ym[te].min() - buffer_m, ym[te].max() + buffer_m
        near = (xm >= x0) & (xm <= x1) & (ym >= y0) & (ym <= y1)
        tr = ~near
        if y[tr].sum() < 30 or (~y[tr]).sum() < 30:
            continue
        yield te, tr


def within_fold_auc(y, o, folds):
    """AUC over positive/negative pairs drawn from the SAME fold, weighted by fold pair count.

    The pooled AUC is misleading on these folds and the numbers say so: fold base rates run
    from 4/521 to 559/947, so pooling lets the metric earn credit for ranking one whole block
    above another -- a between-block base-rate difference, not discrimination on held-out
    ground. This restricts every comparison to within a fold, which is what was actually
    held out. Report both; where they disagree, this is the honest one.

    It is not a cure. The pair weights here are 2 068 / 216 892 / 3 204, so any pair-weighted
    statistic is 97% one fold. Print the per-fold table too, and the weights.
    """
    num = den = 0.0
    for te, _ in folds:
        yy, ss = y[te], o[te]
        m = np.isfinite(ss)
        yy, ss = yy[m], ss[m]
        n1, n0 = float(yy.sum()), float((~yy).sum())
        if n1 < 1 or n0 < 1:
            continue
        num += auc(yy, ss) * n1 * n0
        den += n1 * n0
    return num / den if den else float("nan")


def buffered_cv(X, y, xm, ym, block_m, buffer_m):
    """Out-of-fold scores where training tiles within buffer_m of the test block are cut."""
    oof = np.full(len(y), np.nan)
    oof_xy = np.full(len(y), np.nan)
    xy = np.c_[xm, ym].astype(np.float32)
    n_used = 0
    for te, tr in buffered_folds(y, xm, ym, block_m, buffer_m):
        oof[te] = rf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oof_xy[te] = rf(1).fit(xy[tr], y[tr]).predict_proba(xy[te])[:, 1]
        n_used += 1
    return oof, oof_xy, n_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(
        ROOT, "data", f"bio_tile_features_t{T}.npz"))
    ap.add_argument("--pos-thresh", type=float, default=0.5)
    ap.add_argument("--neg-thresh", type=float, default=0.05)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X, names = d["X"], list(d["feature_names"])
    ap_ = d["aoi_pos"]
    keep = d["aoi_inbox"] & ((ap_ >= args.pos_thresh) | (ap_ <= args.neg_thresh))
    keep &= np.isfinite(X).all(axis=1)
    y = ap_[keep] >= args.pos_thresh
    X = X[keep]
    xm, ym = d["x_m"][keep], d["y_m"][keep]
    ri = names.index("ratio_dB")

    print(f"{len(y)} tiles, {y.sum()} positive ({y.mean():.3f})")
    print(f"extent {(xm.max()-xm.min())/1000:.0f} x {(ym.max()-ym.min())/1000:.0f} km\n")

    print(f"{'block km':>9} {'buf km':>7} {'folds':>6} {'n':>6} "
          f"{'model':>7} {'(x,y)':>7} {'ratio':>7} {'margin':>8}")
    rows = []
    for block_tiles in (16, 32, 64):
        block_m = block_tiles * T * PX_M
        for buf_tiles in (0, 16, 48):
            buffer_m = buf_tiles * T * PX_M
            oof, oxy, nf = buffered_cv(X, y, xm, ym, block_m, buffer_m)
            m = np.isfinite(oof)
            if m.sum() < 100 or nf < 2:
                print(f"{block_m/1000:>9.1f} {buffer_m/1000:>7.1f} {nf:>6} "
                      f"{m.sum():>6}   too few folds/tiles")
                continue
            a_m, a_x = auc(y[m], oof[m]), auc(y[m], oxy[m])
            a_r = auc(y[m], X[m, ri])
            print(f"{block_m/1000:>9.1f} {buffer_m/1000:>7.1f} {nf:>6} {m.sum():>6} "
                  f"{a_m:>7.3f} {a_x:>7.3f} {a_r:>7.3f} {a_m - a_x:>+8.3f}")
            rows.append((block_m / 1000, buffer_m / 1000, a_m, a_x, a_r))

    print("\nVERDICT")
    if not rows:
        print("  no setting produced enough folds -- the labelled area is too small to "
              "separate\n  train from test at these scales.")
        return
    best = max(rows, key=lambda r: r[2] - r[3])
    worst = min(rows, key=lambda r: r[2] - r[3])
    print(f"  best  margin over coordinates {best[2]-best[3]:+.3f} at "
          f"{best[0]:.0f} km blocks / {best[1]:.0f} km buffer "
          f"(model {best[2]:.3f}, (x,y) {best[3]:.3f})")
    print(f"  worst margin over coordinates {worst[2]-worst[3]:+.3f} at "
          f"{worst[0]:.0f} km blocks / {worst[1]:.0f} km buffer")
    if best[2] - best[3] < 0.03:
        print("\n  The radar features never clearly beat map coordinates. Whatever AUC "
              "this model\n  reports is mostly the shape of the Thwaites crevasse fields, "
              "not P-band physics.\n  Do NOT ship it, and do not quote its AUC. What is "
              "needed is labelled ground far\n  enough away that coordinates cannot "
              "interpolate -- i.e. a different region.")
    else:
        print("\n  A real margin survives separation at the setting above. Quote THAT "
              "number,\n  with its block size and buffer, never the unbuffered one.")


if __name__ == "__main__":
    main()
