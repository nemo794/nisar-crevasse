"""Score every admissible tile with the CNN gate, for the scene and mosaic figures.

    conda run -n biomass python code/bio_cnn_infer.py
    conda run -n biomass python code/bio_cnn_infer.py --arm sar --seeds 3

Needs data/bio_chipcache_t512_c128_all.npz (code/bio_tile_cache.py --all) and, for the
out-of-fold field, data/bio_cnn_oof_c128.npz (code/bio_cnn_gate.py). Writes
data/bio_cnn_prob_all.npz with, per row of bio_tile_features_t512.npz:
  p_full   full-data fit, every admissible tile, MOSTLY IN SAMPLE
  p_oof    buffered out-of-fold, only the 1747 tiles the buffered split could score

TWO FIELDS BECAUSE NEITHER ONE ALONE IS HONEST AND COMPLETE
---------------------------------------------------------
`p_oof` is the number you may quote; it covers 1747 of 4767 tiles, so it cannot paint a scene.
`p_full` covers everything but is scored by a model trained on those same tiles, so it is a
picture of what the gate does, not a measurement of how well it does it. The mosaic figure
draws both side by side for exactly that reason.

The arm is `sar+vel` by default because it was the better of the two measured arms on two of
three folds (fold AUCs 0.806 / 0.944 / 0.888 against sar-only's 0.731 / 0.930 / 0.895). Note
that ITS_LIVE speed is a geographic covariate: it scores 0.66-0.70 per fold alone, which is
well under the radar, but it is still the reason a deployed version of this arm would need
velocity available at inference time. `--arm sar` drops it.

Ambiguous tiles (AlphaEarth positive fraction 0.05-0.50) are scored here though the trainer
dropped them, because they are real ground the deployed gate has to answer for.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from crevasse.biomass.bio_cnn_gate import score_with_net, train_fold
from crevasse.biomass.bio_gate_controls import auc
from crevasse.common.ckpt_io import save_checkpoint, load_checkpoint

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "biomass", "gate")
T = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(
        ROOT, "data", f"bio_chipcache_t{T}_c128_all.npz"))
    ap.add_argument("--oof", default=os.path.join(ROOT, "data", "bio_cnn_oof_c128.npz"))
    ap.add_argument("--features", default=os.path.join(
        ROOT, "data", f"bio_tile_features_t{T}.npz"))
    ap.add_argument("--arm", default="sar+vel", choices=["sar", "sar+vel"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "bio_cnn_prob_all.npz"))
    ap.add_argument("--weights",
                    help="score from a saved checkpoint instead of retraining -- a stem, "
                         "no extension (e.g. data/bio_cnn_gate_sarvel), since a "
                         "checkpoint is a <stem>.safetensors + <stem>.json pair")
    ap.add_argument("--save-weights", metavar="STEM",
                    help="write the full-fit seed ensemble to <STEM>.safetensors + "
                         "<STEM>.json after training")
    args = ap.parse_args()
    dev = torch.device(args.device)

    z = np.load(args.cache, allow_pickle=True)
    chips, y, lab, idx = z["chips"], z["y"], z["labelled"], z["idx"]
    ctx = None
    if args.arm == "sar+vel":
        ctx = np.log10(np.maximum(z["vel"], 1e-3)).astype(np.float32)[:, None]
    n_tot = len(np.load(args.features, allow_pickle=True)["row"])

    if args.weights:
        ck = load_checkpoint(args.weights)
        if ck["arm"] != args.arm:
            raise SystemExit(f"checkpoint arm {ck['arm']!r} != --arm {args.arm!r}; "
                             "the head width differs, so this would be a silent mis-score")
        if int(ck["chip"]) != int(z["chip"]):
            raise SystemExit(f"checkpoint chip {ck['chip']} != cache chip {int(z['chip'])}")
        print(f"scoring {len(idx)} tiles from {len(ck['nets'])} saved nets in "
              f"{args.weights} (arm {ck['arm']}, {ck['epochs']} epochs)", flush=True)
        ps = [score_with_net(w, chips, ctx, dev) for w in ck["nets"]]
        for s, p in enumerate(ps):
            print(f"  net {s}: in-sample AUC on the labelled tiles "
                  f"{auc(y[lab], p[lab]):.3f}", flush=True)
    else:
        print(f"arm {args.arm}: fitting on {int(lab.sum())} labelled tiles "
              f"({int(y[lab].sum())} positive), scoring all {len(idx)}", flush=True)
        # train on every labelled tile, predict every cached tile. train_fold fits its
        # normalisation on the train mask only, so passing lab keeps that property here too.
        allm = np.ones(len(idx), bool)
        ps, nets = [], []
        for s in range(args.seeds):
            p, w = train_fold(chips, ctx, y, lab, allm, dev, s, args.epochs, args.lr,
                              args.bs, return_net=True)
            ps.append(p)
            nets.append(w)
            print(f"  seed {s}: in-sample AUC on the labelled tiles "
                  f"{auc(y[lab], p[lab]):.3f}", flush=True)
        if args.save_weights:
            save_checkpoint(args.save_weights, {
                "nets": nets, "arm": args.arm, "epochs": args.epochs, "seeds": args.seeds,
                "chip": int(z["chip"]), "tile_px": T, "px_m": 5.0,
                "n_train": int(lab.sum()), "train_base_rate": float(y[lab].mean()),
                # one granule ID upstream keeps a stray '.tif'; strip it so a string match
                # against the RF bundle's trained_granules cannot miss that granule.
                "train_granules": sorted({str(g).replace(".tif", "")
                                          for g in z["granule"][lab]}),
                "in_sample_auc": [float(auc(y[lab], p[lab])) for p in ps],
                # recorded, NOT reproducible from these weights: these are the buffered
                # out-of-fold AUCs from bio_cnn_gate.py, which fits a separate net per fold.
                # These weights saw every labelled tile, so they cannot be scored held-out.
                "buffered_fold_auc": [0.806, 0.944, 0.888],
                "buffered_fold_note": "82 km blocks / 41 km buffer, 3 folds, n=1747; arm "
                                      "sar+vel; base rates 4/521, 559/947, 12/279. From "
                                      "bio_cnn_gate.py, not from these full-fit weights.",
                "no_threshold_note": "There is no shipped operating point. Lift over base "
                                     "rate at P>=0.65 is 0.00x / 1.67x / 2.45x per fold. "
                                     "See docs/RESULTS.md.",
            })
            print(f"wrote {args.save_weights}.safetensors + {args.save_weights}.json",
                  flush=True)
    p_full_c = np.mean(ps, axis=0)

    p_full = np.full(n_tot, np.nan, np.float32)
    p_full[idx] = p_full_c
    p_oof = np.full(n_tot, np.nan, np.float32)
    if os.path.exists(args.oof):
        o = np.load(args.oof, allow_pickle=True)
        key = "oof_B_sar+vel" if args.arm == "sar+vel" else "oof_A_sar"
        p_oof[o["idx"]] = o[key]
        sc = np.isfinite(p_oof)
        print(f"  carried {int(sc.sum())} buffered out-of-fold scores from {key}", flush=True)
        both = sc & np.isfinite(p_full)
        print(f"  full-fit vs out-of-fold on the {int(both.sum())} shared tiles: "
              f"corr {np.corrcoef(p_full[both], p_oof[both])[0,1]:+.3f}, "
              f"mean full {p_full[both].mean():.3f} vs oof {p_oof[both].mean():.3f}",
              flush=True)
    else:
        print(f"  no {args.oof}; p_oof left all-nan", flush=True)

    np.savez(args.out, p_full=p_full, p_oof=p_oof, arm=args.arm,
             seeds=args.seeds, epochs=args.epochs)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
