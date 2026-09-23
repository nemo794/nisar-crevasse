"""Train the BIOMASS crevasse U-Net: 4-polarization tiles in, per-pixel probability out.

Same shape as nisar-crevasse-unet/src/train_unet_supervised.py (BCE + soft Dice + a
flip-consistency regularizer, AdamW + ReduceLROnPlateau, three checkpoints), with three
differences:

  * `--channels` selects which of the shard's 4 polarizations feed the model
    (default all 4) -- `n_channels = len(channels)` threads straight into
    `UNet(...)`, confirmed a clean, isolated architectural parameter. This is the
    mechanism for the "if 4 channels don't help, drop to fewer" fallback: a flag,
    not a rebuild.
  * No MAE pretraining path and no --raster/--results-pkl path -- BIOMASS has one
    shard format (bio_build_shard.py) and this repo does not carry the unsupervised
    pretraining code NISAR's does. Add it back if it turns out to matter.
  * The checkpoint additionally carries `mu`, `sd`, `channels` (the fitted
    normalization and channel selection) -- BIOMASS's normalization is fit ONCE
    from data, not a fixed per-tile formula, so it must travel with the weights the
    same way bio_cnn_gate.py's does, or a caller would silently score a
    differently-scaled input.

OPTIONAL AUXILIARY NOISE BRANCH (--aux-noise-branch)
------------------------------------------------------
BIOMASS's SAR data carries a confirmed, fixed-direction PSF/speckle confound (see
docs/PIPELINE.md's Frangi investigation -- the same one `bio_build_shard_frangi.py`'s
gate+orientation-gate combination was built to work around). `--aux-noise-branch` adds
a second decoder (`unet_model.UNet(aux_decoder=True)`) trained as a plain denoising
autoencoder -- reconstruct the raw input tile -- sharing the segmentation decoder's
encoder and skip connections. The idea: a bottlenecked autoencoder's cheapest way to
minimize reconstruction error is to capture the DOMINANT, repeating structure in a tile,
which on this data is the PSF/speckle confound, since genuine crevasse structure is a
comparatively sparse minority signal. No shard change is needed -- it trains on the same
raw `image` tensor already in every batch, on every tile, not just gate-rejected ones.

This is deliberately OPT-IN and defaults to off, specifically so a standard run and an
aux-branch run can be trained side by side on the identical shard/split and compared
with `bio_eval_shard.py` -- neither `unet_model.UNet`'s default behavior nor the standard
training path changes when this flag is left off.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from crevasse.common.unet_model import UNet
from crevasse.biomass.bio_sar_dataset import create_shard_dataloaders, POLS
from crevasse.common.ckpt_io import save_checkpoint, load_checkpoint


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat, target_flat = pred.view(-1), target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1 - dice


class ConsistencyLoss(nn.Module):
    def forward(self, pred1, pred2):
        return F.mse_loss(torch.sigmoid(pred1), torch.sigmoid(pred2))


def _forward(model, images):
    """Uniform (logits, noise_recon_or_None) regardless of model.aux_decoder, so the
    training/eval loops don't need an if-branch at every call site."""
    out = model(images)
    if isinstance(out, tuple):
        return out
    return out, None


def train_epoch(model, dataloader, optimizer, device, epoch,
                 bce_weight=1.0, dice_weight=1.0, consistency_weight=0.1,
                 noise_weight=0.0):
    model.train()
    bce_fn, dice_fn, cons_fn = nn.BCEWithLogitsLoss(), DiceLoss(), ConsistencyLoss()
    noise_fn = nn.MSELoss()
    totals = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "consistency": 0.0, "noise": 0.0}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits, noise_recon = _forward(model, images)
        bce_loss = bce_fn(logits, masks)
        dice_loss = dice_fn(logits, masks)

        if noise_recon is not None and noise_weight > 0:
            noise_loss = noise_fn(noise_recon, images)
        else:
            noise_loss = torch.tensor(0.0, device=device)

        if consistency_weight > 0:
            # Same axis for the flip and the un-flip -- a second independent draw
            # un-flips the wrong way ~50% of the time (the bug this repo's NISAR
            # sibling fixed 2026-09-09).
            axis = 2 if torch.rand(1).item() > 0.5 else 3
            logits_aug, _ = _forward(model, torch.flip(images, dims=[axis]))
            logits_aug = torch.flip(logits_aug, dims=[axis])
            consistency_loss = cons_fn(logits, logits_aug)
        else:
            consistency_loss = torch.tensor(0.0, device=device)

        loss = (bce_weight * bce_loss + dice_weight * dice_loss
                + consistency_weight * consistency_loss
                + noise_weight * noise_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totals["loss"] += loss.item()
        totals["bce"] += bce_loss.item()
        totals["dice"] += dice_loss.item()
        totals["consistency"] += consistency_loss.item()
        totals["noise"] += noise_loss.item()
        pbar.set_postfix({"loss": loss.item(), "bce": bce_loss.item(),
                          "dice": dice_loss.item()})

    n = len(dataloader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(model, dataloader, device, pred_thresh=0.5, target_thresh=0.5):
    """`target_thresh` must match the shard: 0.5 for AlphaEarth's genuinely binary mask,
    but 0.2 for bio_build_shard_frangi.py's continuous target (see that script's
    docstring and bio_eval_shard.py's --target-thresh, which this mirrors -- `main()`
    picks the right default from the shard's own `is_soft` flag, same auto-detection
    bio_eval_shard.py uses, so the checkpoint's own recorded val_iou/'best' selection
    means the same thing bio_eval_shard.py's numbers do)."""
    model.eval()
    bce_fn, dice_fn = nn.BCEWithLogitsLoss(), DiceLoss()
    totals = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "iou": 0.0}

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits, _ = _forward(model, images)

        bce_loss = bce_fn(logits, masks)
        dice_loss = dice_fn(logits, masks)
        loss = bce_loss + dice_loss

        preds = (torch.sigmoid(logits) > pred_thresh).float()
        tgt = (masks > target_thresh).float()
        intersection = (preds * tgt).sum()
        union = preds.sum() + tgt.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)

        totals["loss"] += loss.item()
        totals["bce"] += bce_loss.item()
        totals["dice"] += dice_loss.item()
        totals["iou"] += iou.item()

    n = len(dataloader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def visualize_predictions(model, dataloader, device, writer, epoch, num_samples=4):
    model.eval()
    batch = next(iter(dataloader))
    images = batch["image"][:num_samples].to(device)
    masks = batch["mask"][:num_samples].to(device)
    logits, noise_recon = _forward(model, images)
    preds = torch.sigmoid(logits)
    # Only the first selected channel goes to the image summary -- tensorboard's
    # add_images expects 1 or 3 channels, and our stack can be up to 4.
    writer.add_images("supervised/images_ch0", images[:, :1], epoch)
    writer.add_images("supervised/masks_true", masks, epoch)
    writer.add_images("supervised/masks_pred", preds.unsqueeze(1)
                       if preds.dim() == 3 else preds, epoch)
    if noise_recon is not None:
        writer.add_images("aux/noise_recon_ch0", noise_recon[:, :1], epoch)
        writer.add_images("aux/residual_ch0",
                          (images[:, :1] - noise_recon[:, :1]), epoch)


def main():
    parser = argparse.ArgumentParser(description="Train the BIOMASS crevasse U-Net")
    parser.add_argument("--shard", required=True,
                        help="Packed .npz shard from bio_build_shard.py")
    parser.add_argument("--channels", type=str, default=",".join(POLS),
                        help=f"Comma-separated subset of {POLS} to feed the model. "
                             f"n_channels = len(channels). Default: all 4.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-factor", type=float, default=0.5,
                        help="ReduceLROnPlateau: multiply lr by this when val loss "
                             "plateaus (default 0.5, i.e. halve it).")
    parser.add_argument("--lr-patience", type=int, default=5,
                        help="ReduceLROnPlateau: epochs of no val-loss improvement "
                             "before dropping lr (default 5). At 100 epochs this fires "
                             "several times; lower it for a faster decay schedule.")
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--pred-thresh", type=float, default=0.5,
                        help="Prediction cut used for the val_iou tracked for "
                             "checkpoint selection (best-IoU / early stopping). Purely "
                             "a monitoring/selection choice -- doesn't affect the loss. "
                             "0.25 matched bio_eval_shard.py's own --sweep-found optimum "
                             "for the first Frangi-shard standard run; 0.5 is a plain "
                             "default, not validated for this target type.")
    parser.add_argument("--target-thresh", type=float, default=None,
                        help="Target binarization cut for val_iou. Default: auto -- 0.5 "
                             "for a real binary AlphaEarth shard, 0.2 for a soft Frangi "
                             "shard (matches bio_eval_shard.py's own auto-detection).")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint (e.g. unet_last) to resume from: "
                             "loads model/optimizer state and continues the epoch count "
                             "and best-so-far trackers from it. --epochs is the TOTAL "
                             "target epoch count, not an additional number -- resuming a "
                             "run that already reached epoch 100 with --epochs 100 does "
                             "nothing. channels/base-channels/aux-noise-branch must match "
                             "the checkpoint; this is not checked, a mismatch will error "
                             "or silently produce garbage.")
    parser.add_argument("--aux-noise-branch", action="store_true",
                        help="Add the optional second decoder (denoising autoencoder, "
                             "reconstructs the raw input tile) sharing the segmentation "
                             "decoder's encoder. Off by default so a standard run and an "
                             "aux-branch run can be trained on the identical shard/split "
                             "and compared. See the module docstring.")
    parser.add_argument("--noise-weight", type=float, default=0.2,
                        help="Weight of the noise-reconstruction MSE term. Only used "
                             "when --aux-noise-branch is set.")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--block-tiles", type=int, default=32,
                        help="Spatial block side in 512px tiles (default 32 = 82 km, "
                             "biomass-crevasse-gate's validated setting).")
    parser.add_argument("--buffer-tiles", type=int, default=16,
                        help="Buffer around held-out blocks, in tiles (default 16 = "
                             "41 km).")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="../models/bio_unet")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    if not channels:
        parser.error("--channels resolved to an empty list")

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_dir / "logs")

    print("\n" + "=" * 80)
    print("BIOMASS U-NET SUPERVISED TRAINING")
    print("=" * 80)
    print(f"Shard: {Path(args.shard).name}")
    print(f"Channels: {channels}  (n_channels={len(channels)})")
    print(f"Epochs: {args.epochs}   Batch size: {args.batch_size}")
    print(f"Consistency weight: {args.consistency_weight}\n")

    train_loader, val_loader, mu, sd, cidx, is_soft = create_shard_dataloaders(
        args.shard, channels=channels, batch_size=args.batch_size,
        val_frac=args.val_frac, block_tiles=args.block_tiles,
        buffer_tiles=args.buffer_tiles, seed=args.split_seed,
        num_workers=args.num_workers)

    target_thresh = args.target_thresh
    if target_thresh is None:
        target_thresh = 0.2 if is_soft else 0.5
    print(f"Target: {'soft (Frangi)' if is_soft else 'binary (AlphaEarth)'} -- "
          f"val_iou uses pred_thresh={args.pred_thresh} target_thresh={target_thresh}"
          f"{' (auto)' if args.target_thresh is None else ' (explicit)'}")

    print("Creating U-Net model...")
    model = UNet(n_channels=len(channels), n_classes=1,
                base_channels=args.base_channels,
                aux_decoder=args.aux_noise_branch).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.aux_noise_branch:
        print(f"  aux noise branch: ON (noise_weight={args.noise_weight})\n")
    else:
        print("  aux noise branch: off (standard model)\n")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                       factor=args.lr_factor,
                                                       patience=args.lr_patience)

    start_epoch = 1
    best_val_iou, best_val_loss = 0.0, float("inf")

    if args.resume:
        print(f"Resuming from {args.resume}")
        rck = load_checkpoint(args.resume, device=device)
        model.load_state_dict(rck["model_state_dict"])
        optimizer.load_state_dict(rck["optimizer_state_dict"])
        start_epoch = int(rck["epoch"]) + 1
        best_val_iou = float(rck.get("val_iou", 0.0))
        best_val_loss = float(rck.get("val_loss", float("inf")))
        print(f"  resumed at epoch {start_epoch}  "
              f"(best_val_iou so far {best_val_iou:.4f}, best_val_loss {best_val_loss:.4f})")
        if start_epoch > args.epochs:
            parser.error(f"--epochs {args.epochs} is the TOTAL target epoch count; "
                         f"{args.resume} already reached epoch {rck['epoch']}. Raise "
                         f"--epochs past that to continue training.")

    def save_ckpt(stem, epoch, val_metrics):
        save_checkpoint(output_dir / stem, {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_iou": val_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "args": vars(args),
            "channels": channels,
            "mu": mu,
            "sd": sd,
            "aux_decoder": args.aux_noise_branch,
        })

    print("Starting training...")
    print("=" * 80 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch,
                                     bce_weight=args.bce_weight,
                                     dice_weight=args.dice_weight,
                                     consistency_weight=args.consistency_weight,
                                     noise_weight=args.noise_weight if args.aux_noise_branch else 0.0)
        val_metrics = validate(model, val_loader, device,
                                pred_thresh=args.pred_thresh,
                                target_thresh=target_thresh)

        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val IoU: {val_metrics['iou']:.4f}")

        if epoch % 10 == 0:
            visualize_predictions(model, val_loader, device, writer, epoch)

        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            save_ckpt("unet_best", epoch, val_metrics)
            print(f"  -> Saved best-IoU model (IoU={val_metrics['iou']:.4f})")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_ckpt("unet_best_loss", epoch, val_metrics)
            print(f"  -> Saved best-loss model (loss={val_metrics['loss']:.4f})")

        save_ckpt("unet_last", epoch, val_metrics)
        scheduler.step(val_metrics["loss"])

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best validation IoU:  {best_val_iou:.4f}  -> {output_dir / 'unet_best.safetensors'}")
    print(f"Best validation loss: {best_val_loss:.4f}  -> {output_dir / 'unet_best_loss.safetensors'}")
    print(f"Final epoch          : {output_dir / 'unet_last.safetensors'}")
    print("Score all three with bio_eval_shard.py before quoting one.")

    writer.close()


if __name__ == "__main__":
    main()
