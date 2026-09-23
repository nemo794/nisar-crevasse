"""
Train U-Net for crevasse segmentation with consistency regularization.
Can optionally initialize from pretrained MAE encoder.
"""
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np

from crevasse.common.unet_model import UNet
from crevasse.nisar.sar_dataset import create_supervised_dataloaders, create_shard_dataloaders
from crevasse.common.ckpt_io import save_checkpoint, load_checkpoint


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1 - dice


class ConsistencyLoss(nn.Module):
    """
    Consistency regularization loss.
    Enforces similar predictions for augmented versions of same input.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred1, pred2):
        """
        Args:
            pred1, pred2: Predictions for two augmented versions
        """
        pred1 = torch.sigmoid(pred1)
        pred2 = torch.sigmoid(pred2)

        return F.mse_loss(pred1, pred2)


def train_epoch(model, dataloader, optimizer, device, epoch,
                bce_weight=1.0, dice_weight=1.0, consistency_weight=0.1):
    """Train for one epoch with consistency regularization."""
    model.train()

    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()
    consistency_loss_fn = ConsistencyLoss()

    total_loss = 0
    total_bce = 0
    total_dice = 0
    total_consistency = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        # Forward pass
        logits = model(images)

        # Supervised losses
        bce_loss = bce_loss_fn(logits, masks)
        dice_loss = dice_loss_fn(logits, masks)

        # Consistency regularization (augment on-the-fly). The un-flip MUST reuse the
        # same axis: drawing a second independent random axis (as this did until
        # 2026-09-09) un-flips the wrong way ~50% of the time, which asks the model to be
        # invariant to a transpose it was never shown.
        if consistency_weight > 0:
            axis = 2 if torch.rand(1).item() > 0.5 else 3
            logits_aug = torch.flip(model(torch.flip(images, dims=[axis])), dims=[axis])
            consistency_loss = consistency_loss_fn(logits, logits_aug)
        else:
            consistency_loss = torch.tensor(0.0, device=device)

        # Combined loss
        loss = (bce_weight * bce_loss +
                dice_weight * dice_loss +
                consistency_weight * consistency_loss)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track losses
        total_loss += loss.item()
        total_bce += bce_loss.item()
        total_dice += dice_loss.item()
        total_consistency += consistency_loss.item()

        pbar.set_postfix({
            'loss': loss.item(),
            'bce': bce_loss.item(),
            'dice': dice_loss.item()
        })

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'bce': total_bce / n,
        'dice': total_dice / n,
        'consistency': total_consistency / n
    }


@torch.no_grad()
def validate(model, dataloader, device, target_thresh=0.2):
    """Validate model.

    IoU thresholds the TARGET as well as the prediction. Targets are continuous since
    2026-09-09 (soft B x rect(orient_agree, t)), so comparing a binarized prediction
    against a continuous mask put a hard count and a soft mass in the same union term
    and the number was not readable. target_thresh=0.2 is the level every soft-label
    coverage figure in this project is quoted at, so val IoU is comparable to them.
    """
    model.eval()

    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss()

    total_loss = 0
    total_bce = 0
    total_dice = 0
    total_iou = 0

    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        logits = model(images)

        bce_loss = bce_loss_fn(logits, masks)
        dice_loss = dice_loss_fn(logits, masks)
        loss = bce_loss + dice_loss

        # Compute IoU, both sides binarized (see docstring)
        preds = (torch.sigmoid(logits) > 0.5).float()
        tgt = (masks > target_thresh).float()
        intersection = (preds * tgt).sum()
        union = preds.sum() + tgt.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)

        total_loss += loss.item()
        total_bce += bce_loss.item()
        total_dice += dice_loss.item()
        total_iou += iou.item()

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'bce': total_bce / n,
        'dice': total_dice / n,
        'iou': total_iou / n
    }


@torch.no_grad()
def visualize_predictions(model, dataloader, device, writer, epoch, num_samples=4):
    """Visualize predictions."""
    model.eval()
    batch = next(iter(dataloader))

    images = batch['image'][:num_samples].to(device)
    masks = batch['mask'][:num_samples].to(device)

    logits = model(images)
    preds = torch.sigmoid(logits)

    writer.add_images('supervised/images', images, epoch)
    writer.add_images('supervised/masks_true', masks, epoch)
    writer.add_images('supervised/masks_pred', preds, epoch)


def load_pretrained_encoder(model, pretrained_path):
    """Load pretrained encoder weights from MAE."""
    print(f"Loading pretrained encoder from: {pretrained_path}")

    checkpoint = load_checkpoint(pretrained_path)
    mae_state = checkpoint['model_state_dict']

    # Map MAE encoder weights to U-Net encoder
    unet_state = model.state_dict()
    pretrained_dict = {}

    for k, v in mae_state.items():
        if k.startswith('encoder.'):
            # Remove 'encoder.' prefix and match to U-Net
            new_k = k.replace('encoder.', '')
            if new_k in unet_state:
                pretrained_dict[new_k] = v
                print(f"  Loaded: {k} → {new_k}")

    # Load weights
    unet_state.update(pretrained_dict)
    model.load_state_dict(unet_state)

    print(f"Loaded {len(pretrained_dict)} pretrained layers")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train U-Net for crevasse segmentation")
    parser.add_argument("--shard", type=str, default=None,
                       help="Packed .npz training shard from build_shard.py. Carries "
                            "tiles and targets together, so --raster/--results-npz are "
                            "not needed and neither is the GeoTIFF. Preferred on a "
                            "cluster: ~0.55 GB per 1000 tiles vs ~19 GB for the "
                            "raster+pickle pair.")
    parser.add_argument("--raster", type=str, default=None,
                       help="Granule GeoTIFF. Required unless --shard is given.")
    parser.add_argument("--results-npz", type=str, default=None,
                       help="Detection results for pseudo-labels. Required unless "
                            "--shard is given.")
    parser.add_argument("--pretrained", type=str, default=None,
                       help="Path to pretrained MAE checkpoint")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--min-lines", type=int, default=0,
                       help="Minimum ridge component count per tile. Default 0 (off): "
                            "the pkl is already a curated positive set and component "
                            "count is not a meaningful signal after the Frangi redesign "
                            "(the old default of 10 silently emptied the dataset).")
    parser.add_argument("--min-coverage", type=float, default=0.0,
                       help="Minimum labeled-area fraction (edge_coverage) per tile. "
                            "The meaningful density knob; default 0.0 keeps all tiles.")
    parser.add_argument("--split-mode", type=str, default="spatial",
                       choices=["spatial", "blocks", "random"],
                       help="'spatial' (default) holds out a contiguous geographic band "
                            "for a real generalization check; 'blocks' holds out whole "
                            "square blocks of tiles, which is what you want when the "
                            "difficulty is spatially clustered and a band would make val "
                            "systematically easier or harder than train; 'random' does a "
                            "seeded shuffle (mixes regions, leakier).")
    parser.add_argument("--block-tiles", type=int, default=3,
                       help="In blocks mode, block side in tiles (3 = 7.7 km at 512 px / "
                            "5 m). Smaller blocks match difficulty better but separate "
                            "train and val less. 3 was measured on 025_019's grounded "
                            "pool: f4 coverage stays within ~1 point and orient_conc "
                            "within 0.03 across all 5 folds, where 8 swung to "
                            "4.9%%/11.1%%. The cost is a 5.1 km minimum train/val "
                            "separation, which does NOT guarantee disjoint crevasse "
                            "fields -- report it as a known limit.")
    parser.add_argument("--split-buffer-tiles", type=int, default=1,
                       help="Gap in tiles between train and val. In spatial mode this is "
                            "one-sided (before the band edge); in blocks mode it is a "
                            "Chebyshev radius around every val tile.")
    parser.add_argument("--split-seed", type=int, default=42,
                       help="RNG seed for --split-mode random and blocks.")
    parser.add_argument("--context-npz", type=str, default=None,
                       help="Per-tile surface context sidecar from build_tile_context.py. "
                            "Required by --min-grounded.")
    parser.add_argument("--min-grounded", type=float, default=0.0,
                       help="Drop tiles whose Bedmap3 grounded fraction is below this. "
                            "Default 0.0 (off) reproduces the 2026-09-10 runs, whose pool "
                            "was 32%% floating shelf / sea ice -- rifts there are a "
                            "different process from grounded-ice crevasses.")
    parser.add_argument("--output-dir", type=str, default="../models/unet_supervised")
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    if args.shard is None and not (args.raster and args.results_npz):
        parser.error("give either --shard, or both --raster and --results-npz")
    if args.shard and (args.raster or args.results_npz):
        parser.error("--shard already carries the tiles and targets; drop "
                     "--raster/--results-npz so it is unambiguous which was used")
    if args.min_grounded > 0 and not args.shard:
        parser.error("--min-grounded is only wired through the shard path; the raster "
                     "path would silently ignore it and train on the shelf tiles")
    if args.min_grounded > 0 and not args.context_npz:
        parser.error("--min-grounded needs --context-npz (build_tile_context.py)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(output_dir / 'logs')

    print("\n" + "="*80)
    print("U-NET SUPERVISED TRAINING")
    print("="*80)
    if args.shard:
        print(f"Shard: {Path(args.shard).name}")
    else:
        print(f"Raster: {Path(args.raster).name}")
        print(f"Pseudo-labels: {Path(args.results_npz).name}")
    print(f"Pretrained: {args.pretrained if args.pretrained else 'None (from scratch)'}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Consistency weight: {args.consistency_weight}\n")

    # Create dataloaders
    print("Creating dataloaders...")
    if args.shard:
        train_loader, val_loader = create_shard_dataloaders(
            args.shard,
            batch_size=args.batch_size,
            train_split=0.8,
            min_coverage=args.min_coverage,
            split_mode=args.split_mode,
            split_buffer_tiles=args.split_buffer_tiles,
            seed=args.split_seed,
            num_workers=args.num_workers,
            context_npz=args.context_npz,
            min_grounded=args.min_grounded,
            block_tiles=args.block_tiles,
        )
    else:
        train_loader, val_loader = create_supervised_dataloaders(
            args.raster,
            args.results_npz,
            tile_size=args.tile_size,
            batch_size=args.batch_size,
            train_split=0.8,
            min_lines=args.min_lines,
            min_coverage=args.min_coverage,
            split_mode=args.split_mode,
            split_buffer_tiles=args.split_buffer_tiles,
            seed=args.split_seed,
            num_workers=args.num_workers
        )

    # Create model
    print("Creating U-Net model...")
    model = UNet(
        n_channels=1,
        n_classes=1,
        base_channels=args.base_channels
    ).to(device)

    # Load pretrained weights if provided
    if args.pretrained:
        model = load_pretrained_encoder(model, args.pretrained)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_iou = 0.0
    best_val_loss = float('inf')

    def save_ckpt(stem, epoch, val_metrics):
        # No optimizer_state_dict: this repo has no --resume, so it was dead weight
        # every checkpoint round-tripped for zero consumers. Dropped as part of the
        # 2026-09-21 pickle -> safetensors migration rather than carried along.
        save_checkpoint(output_dir / stem, {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_iou': val_metrics['iou'],
            'val_loss': val_metrics['loss'],
            'args': vars(args)
        })

    print("Starting training...")
    print("="*80 + "\n")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            consistency_weight=args.consistency_weight
        )

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Log
        for k, v in train_metrics.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val IoU: {val_metrics['iou']:.4f}")

        # Visualize every 10 epochs
        if epoch % 10 == 0:
            visualize_predictions(model, val_loader, device, writer, epoch)

        # Three checkpoints, because selecting on val IoU alone is not defensible: over the
        # last 10 epochs of the 2026-09-10 run val IoU was 0.504 +/- 0.039, so the recorded
        # max is largely the peak of that noise, and val loss was still improving past the
        # epoch it picked. unet_best keeps its name and meaning so existing eval commands
        # and the val_iou control still work.
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            save_ckpt('unet_best', epoch, val_metrics)
            print(f"  → Saved best-IoU model (IoU={val_metrics['iou']:.4f})")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_ckpt('unet_best_loss', epoch, val_metrics)
            print(f"  → Saved best-loss model (loss={val_metrics['loss']:.4f})")

        save_ckpt('unet_last', epoch, val_metrics)

        scheduler.step(val_metrics['loss'])

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation IoU:  {best_val_iou:.4f}  -> {output_dir / 'unet_best.safetensors'}")
    print(f"Best validation loss: {best_val_loss:.4f}  -> {output_dir / 'unet_best_loss.safetensors'}")
    print(f"Final epoch          : {output_dir / 'unet_last.safetensors'}")
    print("Selection on the noisy val IoU is optimistic; score all three with eval_shard.py "
          "before quoting one.")

    writer.close()


if __name__ == "__main__":
    main()
