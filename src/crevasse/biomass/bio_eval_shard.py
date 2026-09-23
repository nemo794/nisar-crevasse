"""Score a BIOMASS U-Net checkpoint on a shard. No GeoTIFF, no rasterio.

Same three-IoU discipline as nisar-crevasse-unet/src/eval_shard.py -- pooled (the
headline), per-tile (the pessimistic read), per-batch (the ONLY one that reproduces the
training script's printed number, hence the `--subset val` control). Reconstructs the
checkpoint's own split via `buffered_split` with the args recorded in the checkpoint, and
uses the checkpoint's own shipped `mu`/`sd`/`channels` rather than re-fitting them --
re-deriving normalization from whatever shard is handed to this script would make the
control meaningless (a MATCH could happen for the wrong reason).

Usage:
  # the control: must reproduce the checkpoint's own val_iou
  python src/bio_eval_shard.py --checkpoint models/.../unet_best.pth \
      --shard data/train_biomass_v1.npz --subset val

  # the same shard's train split, or "all" for every tile (in-sample if this is the
  # training shard -- see the printed warning)
  python src/bio_eval_shard.py --checkpoint ... --shard ... --subset all
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
from crevasse.biomass.bio_sar_dataset import (ShardMultiPolDataset, buffered_split, channel_indices,
                             get_val_transforms, load_shard)


def _confusion(pred_bin, tgt_bin):
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
    p = argparse.ArgumentParser(description="Score a checkpoint on a BIOMASS shard")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--shard", required=True)
    p.add_argument("--subset", default="all", choices=["all", "val", "train"],
                   help="'val'/'train' reconstruct the checkpoint's own split -- 'val' "
                        "is the reproduction control. 'all' is what you want on a shard "
                        "the model never saw (still in-sample if it's the training "
                        "shard).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Only affects per-batch IoU. Match the training run's value for "
                        "the --subset val control.")
    p.add_argument("--pred-thresh", type=float, default=0.5)
    p.add_argument("--target-thresh", type=float, default=None,
                   help="Cut applied to the target before computing IoU/P/R. Default: "
                        "auto -- 0.5 for a real binary AlphaEarth shard, 0.2 for a soft "
                        "Frangi shard (matching bio_build_shard_frangi.py's own "
                        "definition of 'coverage'). 0.5 on a soft shard keeps only ~25%% "
                        "of its nonzero target pixels -- see that script's docstring.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-npz", default=None)
    p.add_argument("--save-preds", type=int, default=12)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint)
    train_args = ckpt.get("args", {})
    channels = ckpt.get("channels") or list(train_args.get("channels", "HH,HV,VH,VV"))
    mu, sd = ckpt["mu"], ckpt["sd"]

    print("=" * 80)
    print("EVAL ON SHARD")
    print("=" * 80)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"  epoch    : {ckpt.get('epoch', '?')}   recorded val_iou: "
          f"{ckpt.get('val_iou', float('nan')):.4f}")
    print(f"  channels : {channels}   mu={mu:.3f} sd={sd:.3f} (shipped, not re-fit)")
    print(f"Shard      : {args.shard}")
    print(f"Subset     : {args.subset}")
    print(f"Device     : {device}")

    if args.subset == "all" and Path(args.shard).name == \
            Path(train_args.get("shard", "")).name:
        print("\n  NOTE: this is the shard the checkpoint trained on, and --subset all "
              "\n  includes its training tiles. That number is not a generalization "
              "\n  result. Use --subset val for the control.\n")

    d = load_shard(args.shard)
    cidx = channel_indices(d["pols"], channels)

    target_thresh = args.target_thresh
    if target_thresh is None:
        target_thresh = 0.2 if d["is_soft"] else 0.5
    print(f"Target     : {'soft (Frangi)' if d['is_soft'] else 'binary (AlphaEarth)'} "
          f"-- binarizing at target_thresh={target_thresh}"
          f"{' (auto)' if args.target_thresh is None else ' (explicit)'}")

    block_tiles = int(train_args.get("block_tiles", 32))
    buffer_tiles = int(train_args.get("buffer_tiles", 16))
    val_frac = float(train_args.get("val_frac", 0.2))
    split_seed = int(train_args.get("split_seed", 42))
    tr_idx, va_idx = buffered_split(d["x_m"], d["y_m"], val_frac, block_tiles,
                                    buffer_tiles, d["tile_size"], seed=split_seed)

    if args.subset == "all":
        sel = np.arange(len(d["images"]))
    else:
        sel = tr_idx if args.subset == "train" else va_idx
    print(f"\nTiles: {len(sel)} of {len(d['images'])} in the shard (subset={args.subset})")

    ds = ShardMultiPolDataset(d["images"][sel], d["targets"][sel], d["positions"][sel],
                              cidx, mu, sd, transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=torch.cuda.is_available())

    base_channels = int(train_args.get("base_channels", 64))
    aux_decoder = bool(ckpt.get("aux_decoder", False))
    model = UNet(n_channels=len(channels), n_classes=1,
                base_channels=base_channels, aux_decoder=aux_decoder).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model: base_channels={base_channels}, aux_decoder={aux_decoder}, "
          f"{sum(q.numel() for q in model.parameters()):,} params")

    sweep_thresh = np.round(np.arange(0.05, 0.96, 0.05), 2) if args.sweep else None
    sweep_tp = np.zeros(len(sweep_thresh)) if args.sweep else None
    sweep_fp = np.zeros(len(sweep_thresh)) if args.sweep else None
    sweep_fn = np.zeros(len(sweep_thresh)) if args.sweep else None

    tile_tp, tile_fp, tile_fn, batch_ious = [], [], [], []
    keep_prob = []

    for batch in loader:
        imgs = batch["image"].to(device)
        tgts = batch["mask"].to(device)

        out = model(imgs)
        logits = out[0] if isinstance(out, tuple) else out
        prob = torch.sigmoid(logits)
        pred_bin = (prob > args.pred_thresh).float()
        tgt_bin = (tgts > target_thresh).float()

        tp, fp, fn = _confusion(pred_bin, tgt_bin)
        tile_tp.append(tp.flatten().cpu().numpy())
        tile_fp.append(fp.flatten().cpu().numpy())
        tile_fn.append(fn.flatten().cpu().numpy())

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
        dabs = abs(np.mean(batch_ious) - float(ckpt["val_iou"]))
        verdict = "MATCH" if dabs < 5e-3 else "MISMATCH"
        print(f"\nCONTROL: checkpoint val_iou {float(ckpt['val_iou']):.4f} vs per-batch "
              f"{np.mean(batch_ious):.4f}  (|d| {dabs:.4f})  -> {verdict}")
        if verdict == "MISMATCH":
            print("  The split or the normalization has drifted. Nothing above is "
                  "readable until this matches.")

    ts = d["tile_size"]
    tgt_frac = float((tile_tp.sum() + tile_fn.sum()) / (len(sel) * ts ** 2))
    pred_frac = float((tile_tp.sum() + tile_fp.sum()) / (len(sel) * ts ** 2))
    print(f"\ntarget coverage {tgt_frac * 100:.2f}%   predicted coverage "
          f"{pred_frac * 100:.2f}%  (ratio {pred_frac / max(tgt_frac, 1e-9):.2f})")

    print("\nStratification:")
    strata = {}
    strata["target_coverage"] = _stratify("target_coverage", d["coverage"][sel],
                                          tile_tp, tile_fp, tile_fn)
    strata["aoi_pos"] = _stratify("aoi_pos", d["aoi_pos"][sel],
                                  tile_tp, tile_fp, tile_fn)

    sweep_rows = []
    if args.sweep:
        print(f"\nPrediction-threshold sweep (target fixed at >{target_thresh}):")
        print("   thresh  pooled IoU        P        R")
        for i, t in enumerate(sweep_thresh):
            si = float(_iou(sweep_tp[i], sweep_fp[i], sweep_fn[i]))
            sp, sr, _ = _prf(sweep_tp[i], sweep_fp[i], sweep_fn[i])
            sweep_rows.append({"thresh": float(t), "pooled_iou": si,
                               "precision": float(sp), "recall": float(sr)})
            print(f"   {t:5.2f}   {si:9.4f}  {float(sp):7.4f}  {float(sr):7.4f}")
        best = max(sweep_rows, key=lambda r: r["pooled_iou"])
        print(f"   best pooled IoU {best['pooled_iou']:.4f} at thresh "
              f"{best['thresh']:.2f}  ({args.pred_thresh} gives {pooled_iou:.4f})")
        print("   NOTE: picking the best threshold on this subset is fitting to it. "
              f"Quote {args.pred_thresh} unless the threshold was chosen elsewhere.")

    if args.out_npz:
        out = {
            "positions": d["positions"][sel],
            "tile_iou": tile_iou,
            "tile_tp": tile_tp, "tile_fp": tile_fp, "tile_fn": tile_fn,
            "coverage": d["coverage"][sel],
            "aoi_pos": d["aoi_pos"][sel],
            "pooled_iou": pooled_iou,
            "per_tile_iou_mean": float(tile_iou.mean()),
            "per_batch_iou_mean": float(np.mean(batch_ious)),
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "target_coverage": tgt_frac, "predicted_coverage": pred_frac,
            "pred_thresh": args.pred_thresh, "target_thresh": target_thresh,
            "subset": args.subset,
            "shard": str(args.shard), "checkpoint": str(args.checkpoint),
            "strata_json": json.dumps(strata),
            "sweep_json": json.dumps(sweep_rows),
        }
        if args.save_preds > 0 and keep_prob:
            probs = np.concatenate(keep_prob, axis=0)
            n = min(args.save_preds, len(sel))
            order = np.argsort(tile_iou)
            k = max(1, n // 3)
            mid = len(order) // 2
            pick = np.unique(np.concatenate([
                order[:k], order[max(0, mid - k // 2):mid - k // 2 + k], order[-k:],
            ]))[:n]
            out["sample_idx"] = pick
            out["sample_image"] = d["images"][sel][pick]
            out["sample_target"] = d["targets"][sel][pick]
            out["sample_prob"] = probs[pick]
            out["sample_iou"] = tile_iou[pick]
            out["sample_position"] = d["positions"][sel][pick]

        Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out_npz, **out)
        print(f"\nWrote {args.out_npz}")

    print("=" * 80)


if __name__ == "__main__":
    main()
