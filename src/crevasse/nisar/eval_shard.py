"""
Score a trained checkpoint on a packed shard.

This is the piece that makes "training and testing" true on a cluster.
`train_unet_supervised.py` only ever reports val on its own shard's held-out band, and
`infer_from_labels.py` needs the 4.7-9.7 GB granule plus its label pickle -- neither of
which ships. This script needs only a checkpoint and a shard.

Three numbers, because they are not interchangeable and mixing them up is how IoU
figures stop being comparable:

  pooled     one intersection and one union over every pixel in the subset. The headline.
             Large empty tiles cannot inflate it, and it does not depend on batch size.
  per-tile   mean of the per-tile IoU. Weights a 20-pixel tile the same as a full one, so
             it sits below pooled and is the more pessimistic read.
  per-batch  mean of the per-batch IoU at a given --batch-size. This is the ONLY one that
             reproduces the number train_unet_supervised.py prints, which is why it is
             here: run `--subset val` on the training shard with the same --batch-size and
             it must match the checkpoint's recorded val_iou. That is the control -- if it
             does not match, the split or the preprocessing has drifted and nothing else
             this script prints is readable.

BOTH sides are thresholded: prediction at --pred-thresh (0.5), target at --target-thresh
(0.2). Targets have been continuous since 2026-09-09; comparing a binarized prediction
against a continuous mask puts a hard count and a soft mass in the same union term. 0.2 is
the level every soft-label coverage figure in this project is quoted at.

Stratification by `orient_conc` and `gate_prob` comes free -- build_shard.py already packs
both per tile. It answers the question the pooled number cannot: is the model good
everywhere, or only on the tiles the upstream gate was already confident about?

Usage:
  # the control: must reproduce the checkpoint's own val_iou
  python src/eval_shard.py --checkpoint models/.../unet_best \
      --shard data/train_025_019_soft70.npz --subset val

  # the actual test: every tile of the held-back second acquisition
  python src/eval_shard.py --checkpoint models/.../unet_best \
      --shard data/train_025_091_soft70.npz --subset all \
      --out-npz models/.../eval_025_091.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from crevasse.common.unet_model import UNet
from crevasse.common.ckpt_io import load_checkpoint
from crevasse.nisar.sar_dataset import (ShardCrevasseDataset, block_strat_values, get_val_transforms,
                         pool_mask, split_indices)


def _confusion(pred_bin, tgt_bin):
    """Per-tile tp/fp/fn over the last two axes."""
    tp = (pred_bin * tgt_bin).sum(axis=(-2, -1))
    fp = (pred_bin * (1 - tgt_bin)).sum(axis=(-2, -1))
    fn = ((1 - pred_bin) * tgt_bin).sum(axis=(-2, -1))
    return tp, fp, fn


def _iou(tp, fp, fn, eps=1e-6):
    return (tp + eps) / (tp + fp + fn + eps)


def _prf(tp, fp, fn, eps=1e-6):
    prec = (tp + eps) / (tp + fp + eps)
    rec = (tp + eps) / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return prec, rec, f1


def _stratify(name, values, tile_tp, tile_fp, tile_fn, n_bins=3):
    """Pooled IoU within quantile bins of a per-tile scalar.

    Quantile bins, not fixed cut points: a fixed threshold on a distribution that varies
    per granule would put all tiles in one bin and print a table that looks like a
    stratification but is not one. Bin edges are printed so the reader can see whether
    the bins are actually distinct.
    """
    finite = np.isfinite(values)
    if finite.sum() < n_bins * 2:
        print(f"  {name}: only {int(finite.sum())} finite values, not stratifying")
        return []

    edges = np.quantile(values[finite], np.linspace(0, 1, n_bins + 1))
    rows = []
    print(f"  by {name} (tercile edges {np.array2string(edges, precision=3)}):")
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        sel = finite & (values >= lo) & ((values <= hi) if b == n_bins - 1
                                        else (values < hi))
        if not sel.any():
            continue
        tp, fp, fn = tile_tp[sel].sum(), tile_fp[sel].sum(), tile_fn[sel].sum()
        iou = float(_iou(tp, fp, fn))
        prec, rec, _ = _prf(tp, fp, fn)
        rows.append({"bin": b, "lo": float(lo), "hi": float(hi), "n": int(sel.sum()),
                     "pooled_iou": iou, "precision": float(prec), "recall": float(rec)})
        print(f"    [{lo:7.3f}, {hi:7.3f}]  n={int(sel.sum()):5d}  "
              f"pooled IoU {iou:.4f}  P {float(prec):.4f}  R {float(rec):.4f}")
    return rows


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Score a checkpoint on a packed shard")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--shard", required=True)
    p.add_argument("--subset", default="all", choices=["all", "val", "train"],
                   help="'val'/'train' reconstruct the training split of THIS shard and "
                        "are for the reproduction control. 'all' (default) is what you "
                        "want on a shard the model never saw.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Only affects the per-batch IoU. Match the training run's value "
                        "when using --subset val as the control.")
    p.add_argument("--pred-thresh", type=float, default=0.5)
    p.add_argument("--target-thresh", type=float, default=0.2)
    p.add_argument("--min-grounded", type=float, default=None,
                   help="Defaults to the checkpoint's training value. Set 0 to score the "
                        "shelf tiles a grounded-only run never saw -- an out-of-domain "
                        "probe, not the control.")
    p.add_argument("--context-npz", default=None,
                   help="Surface context sidecar; defaults to the one recorded at training "
                        "time. Needed whenever min_grounded > 0.")
    p.add_argument("--min-coverage", type=float, default=None,
                   help="Defaults to the value recorded in the checkpoint's args, so the "
                        "subset reconstruction matches training. Override deliberately.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-npz", default=None,
                   help="Per-tile metrics + a handful of stored predictions, for "
                        "plot_eval.py. Written even on a CPU-only node.")
    p.add_argument("--save-preds", type=int, default=12,
                   help="How many tiles' predictions to store in --out-npz for plotting "
                        "(worst/median/best thirds by per-tile IoU). 0 disables.")
    p.add_argument("--sweep", action="store_true",
                   help="Also sweep the PREDICTION threshold and report the pooled IoU "
                        "curve. The target threshold stays fixed -- moving both at once "
                        "makes the curve unreadable.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint)
    train_args = ckpt.get("args", {})

    min_coverage = (args.min_coverage if args.min_coverage is not None
                    else float(train_args.get("min_coverage", 0.0)))
    min_grounded = (args.min_grounded if args.min_grounded is not None
                    else float(train_args.get("min_grounded", 0.0)))
    context_npz = args.context_npz or train_args.get("context_npz") or None

    print("=" * 80)
    print("EVAL ON SHARD")
    print("=" * 80)
    print(f"Checkpoint     : {args.checkpoint}")
    print(f"  trained on   : {train_args.get('shard', '(unrecorded)')}")
    print(f"  epoch        : {ckpt.get('epoch', '?')}   recorded val_iou: "
          f"{ckpt.get('val_iou', float('nan')):.4f}")
    print(f"Shard          : {args.shard}")
    print(f"Subset         : {args.subset}   min_coverage {min_coverage}")
    print(f"Thresholds     : pred > {args.pred_thresh}, target > {args.target_thresh}")
    print(f"Device         : {device}")

    if Path(args.shard).name == train_args.get("shard", "").split("/")[-1] \
            and args.subset == "all":
        print("\n  NOTE: this is the shard the checkpoint trained on, and --subset all "
              "\n  includes its training tiles. That number is not a generalization "
              "\n  result. Use --subset val for the control.\n")

    z = np.load(args.shard)
    images, targets = z["images"], z["targets"]
    positions = z["positions"].astype(np.int64)
    coverage = z["coverage"]
    tile_size = int(z["tile_size"])
    orient_conc = z["orient_conc"] if "orient_conc" in z else None
    gate_prob = z["gate_prob"] if "gate_prob" in z else None
    # Surface type is carried through whenever the sidecar is available, even when nothing
    # is being filtered on it: not stratifying by it is how the 2026-09-10 runs reported a
    # val IoU that was 69% floating shelf and sea ice.
    grounded_frac = None
    if context_npz:
        c = np.load(context_npz)
        matched = (c["positions"].shape == positions.shape
                   and (c["positions"].astype(np.int64) == positions).all())
        if matched:
            grounded_frac = c["grounded_frac"]
        elif min_grounded > 0:
            # The sidecar is per-granule and the checkpoint records the training
            # granule's. Scoring the held-back granule with min_grounded>0 needs its own
            # sidecar; falling back to "no filter" would score the shelf tiles the model
            # was never trained on and call it generalization.
            raise SystemExit(
                f"{Path(context_npz).name} does not match {Path(args.shard).name}'s "
                f"tiles, and min_grounded={min_grounded} needs a matching sidecar. Pass "
                f"--context-npz for THIS shard (build_tile_context.py).")
        else:
            print(f"  WARNING: {Path(context_npz).name} tiles do not match the shard; "
                  f"surface-type stratification skipped")
            context_npz = None

    # The pool filter and the split both come from the checkpoint's recorded args, via
    # the same functions training used, so `--subset val` cannot score a different set
    # than the val_iou it is being checked against.
    keep = pool_mask(positions, coverage, min_coverage, context_npz, min_grounded)
    idx_all = np.flatnonzero(keep)

    if args.subset == "all":
        sel = idx_all
    else:
        tr, va = split_indices(positions[keep], 0.8, tile_size,
                               str(train_args.get("split_mode", "spatial")),
                               int(train_args.get("split_buffer_tiles", 1)),
                               int(train_args.get("split_seed", 42)),
                               int(train_args.get("block_tiles", 3)),
                               block_strat_values(context_npz, keep))
        sel = idx_all[tr if args.subset == "train" else va]

    print(f"\nTiles: {len(sel)} of {len(coverage)} in the shard "
          f"(coverage>={min_coverage}, grounded>={min_grounded}, subset={args.subset})")

    ds = ShardCrevasseDataset(images[sel], targets[sel], positions[sel],
                              transform=get_val_transforms(), normalize=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())

    base_channels = int(train_args.get("base_channels", 64))
    model = UNet(n_channels=1, n_classes=1, base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model: base_channels={base_channels}, "
          f"{sum(q.numel() for q in model.parameters()):,} params")

    sweep_thresh = np.round(np.arange(0.05, 0.96, 0.05), 2) if args.sweep else None
    sweep_tp = np.zeros(len(sweep_thresh)) if args.sweep else None
    sweep_fp = np.zeros(len(sweep_thresh)) if args.sweep else None
    sweep_fn = np.zeros(len(sweep_thresh)) if args.sweep else None

    tile_tp, tile_fp, tile_fn, batch_ious = [], [], [], []
    keep_prob = []  # float16 probabilities, only if we are saving predictions

    for batch in loader:
        imgs = batch["image"].to(device)
        tgts = batch["mask"].to(device)

        prob = torch.sigmoid(model(imgs))
        pred_bin = (prob > args.pred_thresh).float()
        tgt_bin = (tgts > args.target_thresh).float()

        tp, fp, fn = _confusion(pred_bin, tgt_bin)
        tile_tp.append(tp.flatten().cpu().numpy())
        tile_fp.append(fp.flatten().cpu().numpy())
        tile_fn.append(fn.flatten().cpu().numpy())

        # Batch-level IoU exactly as train_unet_supervised.validate computes it.
        bi = pred_bin.sum() + tgt_bin.sum() - (pred_bin * tgt_bin).sum()
        batch_ious.append(float(((pred_bin * tgt_bin).sum() + 1e-6) / (bi + 1e-6)))

        if args.sweep:
            for i, t in enumerate(sweep_thresh):
                pb = (prob > float(t)).float()
                stp, sfp, sfn = _confusion(pb, tgt_bin)
                sweep_tp[i] += float(stp.sum())
                sweep_fp[i] += float(sfp.sum())
                sweep_fn[i] += float(sfn.sum())

        if args.save_preds > 0:
            keep_prob.append(prob.squeeze(1).cpu().numpy().astype(np.float16))

    tile_tp = np.concatenate(tile_tp).astype(np.float64)
    tile_fp = np.concatenate(tile_fp).astype(np.float64)
    tile_fn = np.concatenate(tile_fn).astype(np.float64)
    tile_iou = _iou(tile_tp, tile_fp, tile_fn)

    pooled_iou = float(_iou(tile_tp.sum(), tile_fp.sum(), tile_fn.sum()))
    prec, rec, f1 = _prf(tile_tp.sum(), tile_fp.sum(), tile_fn.sum())

    print("\n" + "-" * 80)
    print(f"pooled    IoU {pooled_iou:.4f}   P {float(prec):.4f}  R {float(rec):.4f}  "
          f"F1 {float(f1):.4f}")
    print(f"per-tile  IoU {tile_iou.mean():.4f}  (median {np.median(tile_iou):.4f}, "
          f"p10 {np.percentile(tile_iou, 10):.4f}, p90 "
          f"{np.percentile(tile_iou, 90):.4f})")
    print(f"per-batch IoU {np.mean(batch_ious):.4f}  <-- comparable to the training "
          f"script at --batch-size {args.batch_size}")
    if args.subset == "val" and "val_iou" in ckpt:
        d = abs(np.mean(batch_ious) - float(ckpt["val_iou"]))
        verdict = "MATCH" if d < 5e-3 else "MISMATCH"
        print(f"\nCONTROL: checkpoint val_iou {float(ckpt['val_iou']):.4f} vs per-batch "
              f"{np.mean(batch_ious):.4f}  (|d| {d:.4f})  -> {verdict}")
        if verdict == "MISMATCH":
            print("  The split or the preprocessing has drifted. Nothing above is "
                  "readable until this matches.")

    # Coverage of the target itself, so an IoU can be read against how much there was
    # to find. A near-empty subset makes any IoU look bad.
    tgt_frac = float((tile_tp.sum() + tile_fn.sum()) / (len(sel) * tile_size ** 2))
    pred_frac = float((tile_tp.sum() + tile_fp.sum()) / (len(sel) * tile_size ** 2))
    print(f"\ntarget coverage {tgt_frac * 100:.2f}%   predicted coverage "
          f"{pred_frac * 100:.2f}%  (ratio {pred_frac / max(tgt_frac, 1e-9):.2f})")

    print("\nStratification:")
    strata = {}
    if orient_conc is not None:
        strata["orient_conc"] = _stratify("orient_conc", orient_conc[sel].astype(float),
                                          tile_tp, tile_fp, tile_fn)
    if gate_prob is not None:
        strata["gate_prob"] = _stratify("gate_prob", gate_prob[sel].astype(float),
                                        tile_tp, tile_fp, tile_fn)
    strata["target_coverage"] = _stratify("target_coverage", coverage[sel].astype(float),
                                          tile_tp, tile_fp, tile_fn)
    if grounded_frac is not None:
        gsel = grounded_frac[sel].astype(float)
        # Fixed cut points, not terciles: the question is grounded ice vs floating shelf
        # and sea ice, which is a physical boundary, and quantiles of an almost-bimodal
        # variable would put that boundary inside a bin.
        rows = []
        print("  by surface type (Bedmap3 grounded fraction, FIXED cuts):")
        for lo, hi, name in ((0.0, 0.05, "shelf / sea ice"), (0.05, 0.95, "mixed"),
                             (0.95, 1.01, "grounded")):
            m = (gsel >= lo) & (gsel < hi)
            if not m.any():
                continue
            tp, fp, fn = tile_tp[m].sum(), tile_fp[m].sum(), tile_fn[m].sum()
            prec, rec, _ = _prf(tp, fp, fn)
            rows.append({"lo": lo, "hi": hi, "name": name, "n": int(m.sum()),
                         "pooled_iou": float(_iou(tp, fp, fn)),
                         "precision": float(prec), "recall": float(rec)})
            print(f"    {name:16s} n={int(m.sum()):5d}  pooled IoU {float(_iou(tp,fp,fn)):.4f}"
                  f"  P {float(prec):.4f}  R {float(rec):.4f}")
        strata["grounded_frac"] = rows

    sweep_rows = []
    if args.sweep:
        print("\nPrediction-threshold sweep (target fixed at "
              f"{args.target_thresh}):")
        print("   thresh  pooled IoU        P        R")
        for i, t in enumerate(sweep_thresh):
            si = float(_iou(sweep_tp[i], sweep_fp[i], sweep_fn[i]))
            sp, sr, _ = _prf(sweep_tp[i], sweep_fp[i], sweep_fn[i])
            sweep_rows.append({"thresh": float(t), "pooled_iou": si,
                               "precision": float(sp), "recall": float(sr)})
            print(f"   {t:5.2f}   {si:9.4f}  {float(sp):7.4f}  {float(sr):7.4f}")
        best = max(sweep_rows, key=lambda r: r["pooled_iou"])
        print(f"   best pooled IoU {best['pooled_iou']:.4f} at thresh "
              f"{best['thresh']:.2f}  (0.5 gives {pooled_iou:.4f})")
        print("   NOTE: picking the best threshold on this subset is fitting to it. "
              "Quote 0.5 unless the threshold was chosen elsewhere.")

    if args.out_npz:
        out = {
            "positions": positions[sel],
            "tile_iou": tile_iou,
            "tile_tp": tile_tp, "tile_fp": tile_fp, "tile_fn": tile_fn,
            "coverage": coverage[sel],
            "pooled_iou": pooled_iou,
            "per_tile_iou_mean": float(tile_iou.mean()),
            "per_batch_iou_mean": float(np.mean(batch_ious)),
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "target_coverage": tgt_frac, "predicted_coverage": pred_frac,
            "pred_thresh": args.pred_thresh, "target_thresh": args.target_thresh,
            "subset": args.subset,
            "shard": str(args.shard), "checkpoint": str(args.checkpoint),
            "strata_json": json.dumps(strata),
            "sweep_json": json.dumps(sweep_rows),
        }
        if orient_conc is not None:
            out["orient_conc"] = orient_conc[sel]
        if gate_prob is not None:
            out["gate_prob"] = gate_prob[sel]
        if grounded_frac is not None:
            out["grounded_frac"] = grounded_frac[sel]

        if args.save_preds > 0 and keep_prob:
            probs = np.concatenate(keep_prob, axis=0)
            n = min(args.save_preds, len(sel))
            order = np.argsort(tile_iou)
            k = max(1, n // 3)
            mid = len(order) // 2
            pick = np.unique(np.concatenate([
                order[:k],                                   # worst
                order[max(0, mid - k // 2):mid - k // 2 + k],  # median
                order[-k:],                                  # best
            ]))[:n]
            out["sample_idx"] = pick
            out["sample_image"] = images[sel][pick]     # RAW amplitude, float16
            out["sample_target"] = targets[sel][pick]
            out["sample_prob"] = probs[pick]
            out["sample_iou"] = tile_iou[pick]
            out["sample_position"] = positions[sel][pick]

        Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out_npz, **out)
        print(f"\nWrote {args.out_npz}")

    print("=" * 80)


if __name__ == "__main__":
    main()
