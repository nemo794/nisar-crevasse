"""Score BIOMASS tiles with either gate, or both at once, and write a CSV.

    python src/bio_score_tiles.py --method rf  --thresh-rf 0.65 --out /tmp/tiles.csv
    python src/bio_score_tiles.py --method cnn --thresh-cnn 0.65 --out /tmp/tiles.csv \
        --weights data/bio_cnn_gate_sarvel --cache data/bio_chipcache_t512_c128_all.npz
    python src/bio_score_tiles.py --method both --thresh-rf 0.65 --thresh-cnn 0.65 \
        --weights ... --cache ... --out /tmp/tiles.csv
    python src/bio_score_tiles.py --method rf --no-thresh --out /tmp/tiles.csv   # P only

Both methods produce a probability per 512 x 512 tile (2.56 km at 5 m). The RF reads the 38
tabular features; the CNN reads 4 x 128 x 128 dB chips. They agree on ~95% of tiles at P 0.65
and their full-fit fields correlate +0.953, so this is one map ranked two ways, not two maps.

--method both DOES NOT COMBINE THE TWO SCORES
---------------------------------------------
It writes p_rf and p_cnn as separate columns and stops there. No mean, no vote, no
agreement column. The two probability scales are not comparable: at the same 0.65 cut the
RF flags ZERO tiles on two of the three buffered folds while the CNN flags some on all
three, so averaging them would be arithmetic on incommensurable numbers. If you want an
agreement rule, compute it from the two columns and own the choice.

For the same reason the thresholds are separate: --thresh-rf and --thresh-cnn, each
required when its model runs.

A THRESHOLD IS REQUIRED, AND THERE IS NO DEFAULT
-----------------------------------------------
Not an oversight and not caution for its own sake. On the buffered held-out folds, a fixed
P >= 0.65 cut gives:

    fold   base rate   flagged        precision   lift over base
    0        0.008      24 / 521        0.000        0.00x
    1        0.590     244 / 947        0.984        1.67x
    2        0.043      19 / 279        0.105        2.45x

The training folds are 26-58% positive, so the probability scale is fitted to ground that is
roughly half crevassed and floods when applied to ground that is 0.8% crevassed. The RF is
miscalibrated in the opposite direction at the same cut: it flags ZERO tiles on folds 0 and 2.

So the ranking transfers and the calibration does not. Pick a threshold against the prevalence
of the ground you are actually running on, quote LIFT rather than precision, and if you have no
basis for picking one use --no-thresh and keep the probabilities.
"""
import argparse
import csv
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "biomass", "gate")
T = 512

NO_THRESH_MSG = """error: {flag} is required. There is no defensible default for BIOMASS.
  See docs/RESULTS.md 'Operating point'.
  thr 0.65 lift: fold0 0.00x  fold1 1.67x  fold2 2.45x
  Precision at a fixed cut tracks base rate, not skill.
  The two models are cut separately because their scales are not comparable.
  Pass --no-thresh to emit probabilities with no flag column."""


def score_rf(args):
    import joblib
    import sklearn
    b = joblib.load(args.bundle)
    got, want = sklearn.__version__, b.get("sklearn_version")
    if want and got != want:
        print(f"note: bundle was fitted with scikit-learn {want}, running {got}. The forest "
              f"unpickles across this gap but sklearn will warn; if the scores move, that is "
              f"the cause. Retrain with src/bio_train_gate.py to silence it.")
    if "thresh" in b:
        raise SystemExit(
            f"{args.bundle} carries a `thresh` key. Bundles written by this repo do not; "
            "an older one wrote the best-F1 threshold over UNBUFFERED out-of-fold scores. "
            "Retrain with src/bio_train_gate.py or delete the key knowingly.")
    d = np.load(args.features, allow_pickle=True)
    X = d["X"]
    ok = np.isfinite(X).all(axis=1)
    p = np.full(len(X), np.nan, np.float32)
    p[ok] = b["model"].predict_proba(X[ok])[:, 1]
    print(f"RF: {int(ok.sum())} of {len(X)} tiles scored "
          f"({int((~ok).sum())} dropped for non-finite features)")
    return p, d, b


def score_cnn(args):
    import torch

    from crevasse.biomass.bio_cnn_gate import score_with_net
    from crevasse.common.ckpt_io import load_checkpoint
    if not args.weights:
        raise SystemExit("--method cnn needs --weights (data/bio_cnn_gate_sarvel, no "
                         "extension -- a checkpoint is a <stem>.safetensors + "
                         "<stem>.json pair)")
    if not args.cache:
        raise SystemExit(
            "--method cnn needs --cache, the 4-channel chip cache. It is NOT shipped "
            "(626 MB); rebuild it with src/bio_tile_cache.py --all from the granules. "
            "See docs/DATA.md.")
    ck = load_checkpoint(args.weights)
    z = np.load(args.cache, allow_pickle=True)
    if int(ck["chip"]) != int(z["chip"]):
        raise SystemExit(f"checkpoint chip {ck['chip']} != cache chip {int(z['chip'])}")
    ctx = None
    if ck["arm"] == "sar+vel":
        ctx = np.log10(np.maximum(z["vel"], 1e-3)).astype(np.float32)[:, None]
    dev = torch.device(args.device)
    d = np.load(args.features, allow_pickle=True)
    ps = [score_with_net(w, z["chips"], ctx, dev) for w in ck["nets"]]
    p = np.full(len(d["row"]), np.nan, np.float32)
    p[z["idx"]] = np.mean(ps, axis=0)
    print(f"CNN: {len(z['idx'])} tiles scored from {len(ck['nets'])} nets, arm {ck['arm']}")
    return p, d, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["rf", "cnn", "both"],
                    help="`both` writes p_rf and p_cnn side by side and never combines them")
    ap.add_argument("--thresh-rf", type=float,
                    help="flag tiles at or above this RF P. REQUIRED when the RF runs")
    ap.add_argument("--thresh-cnn", type=float,
                    help="flag tiles at or above this CNN P. REQUIRED when the CNN runs")
    ap.add_argument("--no-thresh", action="store_true",
                    help="emit probabilities only, with no flag column")
    ap.add_argument("--features", default=os.path.join(
        ROOT, "data", f"bio_tile_features_t{T}.npz"))
    ap.add_argument("--bundle", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(ROOT))), "models", "biomass", "gate", f"bio_gate_t{T}.joblib"))
    ap.add_argument("--weights", help="CNN checkpoint (--method cnn)")
    ap.add_argument("--cache", help="4-channel chip cache (--method cnn); not shipped")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    want = {"rf": ["rf"], "cnn": ["cnn"], "both": ["rf", "cnn"]}[args.method]
    thr = {"rf": args.thresh_rf, "cnn": args.thresh_cnn}
    if args.no_thresh and any(thr[m] is not None for m in thr):
        raise SystemExit("--no-thresh is mutually exclusive with --thresh-rf/--thresh-cnn")
    for m in thr:
        if m not in want and thr[m] is not None:
            raise SystemExit(f"--thresh-{m} given but --method {args.method} does not run "
                             f"the {m.upper()}")
    for m in want:
        if thr[m] is None and not args.no_thresh:
            raise SystemExit(NO_THRESH_MSG.format(flag=f"--thresh-{m}"))
    if args.no_thresh:
        thr = {"rf": None, "cnn": None}

    P, d = {}, None
    for m in want:
        P[m], d, _ = score_cnn(args) if m == "cnn" else score_rf(args)

    cols = ["granule", "row", "col", "x_m", "y_m"] + [f"p_{m}" for m in want]
    if not args.no_thresh:
        cols += [f"flag_{m}" for m in want]
    any_ok = np.zeros(len(d["row"]), bool)
    for m in want:
        any_ok |= np.isfinite(P[m])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in np.where(any_ok)[0]:
            r = [str(d["granule"][i]).replace(".tif", ""), int(d["row"][i]), int(d["col"][i]),
                 f"{float(d['x_m'][i]):.1f}", f"{float(d['y_m'][i]):.1f}"]
            r += [f"{P[m][i]:.4f}" if np.isfinite(P[m][i]) else "" for m in want]
            if not args.no_thresh:
                r += [int(P[m][i] >= thr[m]) if np.isfinite(P[m][i]) else "" for m in want]
            w.writerow(r)

    print(f"wrote {args.out}  ({int(any_ok.sum())} rows)")
    for m in want:
        ok = np.isfinite(P[m])
        if thr[m] is None:
            continue
        n = int((P[m][ok] >= thr[m]).sum())
        print(f"{m.upper()}: flagged {n} of {int(ok.sum())} tiles "
              f"({n / max(ok.sum(), 1):.1%}) at P >= {thr[m]}")
    if not args.no_thresh:
        print("These fractions are NOT precision estimates, and the two are not "
              "comparable to\neach other. On held-out ground a 0.65 cut gave CNN lift "
              "0.00x / 1.67x / 2.45x\nacross three folds while the RF flagged nothing on "
              "two of them. docs/LIMITATIONS.md.")


if __name__ == "__main__":
    main()
