"""Diagnose whether a checkpoint's false positives on empty-target tiles cluster on the
documented BIOMASS PSF anisotropy axis (~50 deg / ~140 deg, see docs/PIPELINE.md and the
2026-09-16 Frangi investigation) rather than looking like generic, randomly-oriented
texture the model over-fires on.

Not a verdict by itself. Two independent checks, neither conclusive alone:
  * Angle: a PSF artifact has a FIXED axis regardless of location; real crevasses vary
    with local ice flow. Tight clustering near the documented axis, especially compared
    to a control group, is evidence for artifact -- not proof, since a real crevasse
    field's flow-perpendicular orientation COULD coincidentally align with it.
  * AlphaEarth aoi_pos: reported for context only. It is coarse (field-level, not
    individual-crevasse shape) and from a DIFFERENT acquisition date/sensor -- weak
    corroboration at best, never ground truth for a specific tile's SAR-visible
    structure. Do not treat a nonzero aoi_pos here as settling the question.

Usage:
  python src/bio_diagnose_psf_fp.py --checkpoint models/RUN/unet_best.pth \
      --shard data/train_biomass_frangi_v1.npz --subset val
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from crevasse.common.unet_model import UNet
from crevasse.common.ckpt_io import load_checkpoint
from crevasse.biomass.bio_sar_dataset import (ShardMultiPolDataset, buffered_split, channel_indices,
                             get_val_transforms, load_shard)

PSF_AXES_DEG = (50.0, 140.0)


def dominant_angle_deg(img2d):
    """Structure-tensor dominant orientation, degrees in [0, 180). Invariant to the
    per-tile affine normalization (subtract mu, divide by sd) already applied to `img2d`
    -- a uniform scale/offset doesn't change gradient direction."""
    gy, gx = np.gradient(np.asarray(img2d, dtype=np.float32))
    jxx, jyy, jxy = float(np.mean(gx * gx)), float(np.mean(gy * gy)), float(np.mean(gx * gy))
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    return float(np.degrees(theta) % 180)


def angle_dist_to_psf(angle_deg):
    """Circular (mod 180) distance in degrees from `angle_deg` to the nearest PSF axis."""
    return min(min(abs(angle_deg - a), 180 - abs(angle_deg - a)) for a in PSF_AXES_DEG)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--subset", default="val", choices=["all", "val", "train"])
    ap.add_argument("--pred-thresh", type=float, default=0.25)
    ap.add_argument("--target-thresh", type=float, default=None,
                    help="Default: auto -- 0.2 soft (Frangi) / 0.5 binary (AlphaEarth).")
    ap.add_argument("--fp-coverage-thresh", type=float, default=0.05,
                    help="A tile counts as a false positive when target coverage is 0 "
                         "(gate/orientation-gate said empty) and predicted coverage "
                         "exceeds this fraction of pixels.")
    ap.add_argument("--angle-tol-deg", type=float, default=8.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.checkpoint)
    train_args = ckpt.get("args", {})
    channels = ckpt.get("channels") or list(train_args.get("channels", "HH,HV,VH,VV").split(","))
    mu, sd = ckpt["mu"], ckpt["sd"]
    if "HH" not in channels:
        raise ValueError(f"this diagnostic needs HH in the checkpoint's channels; got {channels}")
    hh_idx = channels.index("HH")

    print(f"Checkpoint: {args.checkpoint}  channels={channels}")
    print(f"Shard: {args.shard}  subset={args.subset}")

    d = load_shard(args.shard)
    cidx = channel_indices(d["pols"], channels)
    target_thresh = args.target_thresh
    if target_thresh is None:
        target_thresh = 0.2 if d["is_soft"] else 0.5
    print(f"target_thresh={target_thresh} ({'auto' if args.target_thresh is None else 'explicit'})  "
          f"pred_thresh={args.pred_thresh}  fp_coverage_thresh={args.fp_coverage_thresh}")

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
    aoi_pos_sel = d["aoi_pos"][sel]
    print(f"Tiles: {len(sel)}")

    ds = ShardMultiPolDataset(d["images"][sel], d["targets"][sel], d["positions"][sel],
                              cidx, mu, sd, transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    base_channels = int(train_args.get("base_channels", 64))
    aux_decoder = bool(ckpt.get("aux_decoder", False))
    model = UNet(n_channels=len(channels), n_classes=1, base_channels=base_channels,
                aux_decoder=aux_decoder).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    fp_angle_dist, fp_aoi = [], []
    quiet_angle_dist, quiet_aoi = [], []
    ptr = 0
    for batch in loader:
        imgs = batch["image"].to(device)
        tgts = batch["mask"].to(device)
        n = imgs.shape[0]

        out = model(imgs)
        logits = out[0] if isinstance(out, tuple) else out
        prob = torch.sigmoid(logits)
        pred_bin = (prob > args.pred_thresh).float()
        tgt_bin = (tgts > target_thresh).float()

        pred_cov = pred_bin.mean(dim=(-3, -2, -1)).cpu().numpy()
        tgt_cov = tgt_bin.mean(dim=(-3, -2, -1)).cpu().numpy()
        hh = imgs[:, hh_idx].cpu().numpy()

        for i in range(n):
            aoi = float(aoi_pos_sel[ptr + i]) if np.isfinite(aoi_pos_sel[ptr + i]) else float("nan")
            if tgt_cov[i] == 0 and pred_cov[i] > args.fp_coverage_thresh:
                ang = dominant_angle_deg(hh[i])
                fp_angle_dist.append(angle_dist_to_psf(ang))
                fp_aoi.append(aoi)
            elif tgt_cov[i] == 0 and pred_cov[i] < 1e-4:
                ang = dominant_angle_deg(hh[i])
                quiet_angle_dist.append(angle_dist_to_psf(ang))
                quiet_aoi.append(aoi)
        ptr += n

    fp_angle_dist = np.array(fp_angle_dist)
    quiet_angle_dist = np.array(quiet_angle_dist)
    fp_aoi = np.array(fp_aoi)
    quiet_aoi = np.array(quiet_aoi)

    print("\n" + "=" * 70)
    print(f"False-positive-on-empty-target tiles: n={len(fp_angle_dist)}")
    if len(fp_angle_dist):
        near = (fp_angle_dist <= args.angle_tol_deg).mean()
        print(f"  angle-to-PSF-axis: mean {fp_angle_dist.mean():.1f} deg, "
              f"median {np.median(fp_angle_dist):.1f} deg, "
              f"{near*100:.1f}% within {args.angle_tol_deg} deg of {PSF_AXES_DEG}")
        finite = np.isfinite(fp_aoi)
        if finite.any():
            print(f"  AlphaEarth aoi_pos (weak/coarse, different acquisition -- context "
                  f"only): mean {np.nanmean(fp_aoi):.3f}, median {np.nanmedian(fp_aoi):.3f}, "
                  f"{(fp_aoi[finite] > 0.05).mean()*100:.1f}% > 0.05")

    print(f"\nControl -- correctly-quiet empty tiles: n={len(quiet_angle_dist)}")
    if len(quiet_angle_dist):
        near = (quiet_angle_dist <= args.angle_tol_deg).mean()
        print(f"  angle-to-PSF-axis: mean {quiet_angle_dist.mean():.1f} deg, "
              f"median {np.median(quiet_angle_dist):.1f} deg, "
              f"{near*100:.1f}% within {args.angle_tol_deg} deg of {PSF_AXES_DEG}")
        finite = np.isfinite(quiet_aoi)
        if finite.any():
            print(f"  AlphaEarth aoi_pos: mean {np.nanmean(quiet_aoi):.3f}, "
                  f"median {np.nanmedian(quiet_aoi):.3f}")

    if len(fp_angle_dist) and len(quiet_angle_dist):
        print(f"\nFP tiles are {'CLOSER to' if fp_angle_dist.mean() < quiet_angle_dist.mean() else 'NOT closer to'} "
              f"the PSF axis than the quiet control "
              f"({fp_angle_dist.mean():.1f} deg vs {quiet_angle_dist.mean():.1f} deg mean distance).")
        print("Read this alongside the panels, not instead of them -- this is a "
              "population-level signal, not proof for any single tile.")
    print("=" * 70)


if __name__ == "__main__":
    main()
