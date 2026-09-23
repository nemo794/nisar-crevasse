"""
Self-supervised pretraining using Masked Autoencoder (MAE).
Learn SAR features from unlabeled tiles.
"""
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np

from crevasse.common.unet_model import MaskedAutoencoder
from crevasse.nisar.sar_dataset import create_dataloaders
from crevasse.nisar.find_data_swath import find_data_bounds, get_valid_tile_positions
from crevasse.common.ckpt_io import save_checkpoint


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        images = batch['image'].to(device)

        # Forward pass with random masking
        reconstruction, mask = model(images)

        # Compute loss only on masked regions
        loss = nn.functional.mse_loss(
            reconstruction * (1 - mask),
            images * (1 - mask)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate reconstruction."""
    model.eval()
    total_loss = 0

    for batch in dataloader:
        images = batch['image'].to(device)

        # Forward pass
        reconstruction, mask = model(images)

        # Loss on masked regions
        loss = nn.functional.mse_loss(
            reconstruction * (1 - mask),
            images * (1 - mask)
        )

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def visualize_reconstruction(model, dataloader, device, writer, epoch, num_samples=4):
    """Visualize reconstructions."""
    model.eval()
    batch = next(iter(dataloader))
    images = batch['image'][:num_samples].to(device)

    reconstruction, mask = model(images)

    # Log to tensorboard
    writer.add_images('pretraining/original', images, epoch)
    writer.add_images('pretraining/masked', images * mask, epoch)
    writer.add_images('pretraining/reconstruction', reconstruction, epoch)
    writer.add_images('pretraining/mask', mask.repeat(1, images.shape[1], 1, 1), epoch)


def main():
    parser = argparse.ArgumentParser(description="MAE pretraining on SAR tiles")
    parser.add_argument("--raster", type=str, required=True)
    parser.add_argument("--tile-size", type=int, default=512,
                       help="Tile size for training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--output-dir", type=str, default="../models/mae_pretrain")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-tiles", type=int, default=None,
                       help="Maximum tiles to use (None = all)")

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(output_dir / 'logs')

    print("\n" + "="*80)
    print("MASKED AUTOENCODER PRETRAINING")
    print("="*80)
    print(f"Raster: {Path(args.raster).name}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Output: {args.output_dir}\n")

    # Find valid tiles
    print("Finding valid tiles in data swath...")
    bounds, _ = find_data_bounds(args.raster, downsample=100)
    all_positions = get_valid_tile_positions(bounds, tile_size=args.tile_size)

    if args.max_tiles:
        all_positions = all_positions[:args.max_tiles]

    print(f"Using {len(all_positions)} tiles\n")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        args.raster,
        all_positions,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        train_split=0.9,  # Use more for pretraining
        num_workers=args.num_workers
    )

    # Create model
    print("Creating MAE model...")
    model = MaskedAutoencoder(
        n_channels=1,
        base_channels=args.base_channels,
        mask_ratio=args.mask_ratio
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float('inf')

    print("Starting training...")
    print("="*80 + "\n")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Log
        writer.add_scalar('pretraining/train_loss', train_loss, epoch)
        writer.add_scalar('pretraining/val_loss', val_loss, epoch)
        writer.add_scalar('pretraining/lr', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Visualize every 10 epochs
        if epoch % 10 == 0:
            visualize_reconstruction(model, val_loader, device, writer, epoch)

        # Save best model. No optimizer_state_dict: no --resume exists for MAE
        # pretraining either, so it was dead weight -- dropped in the 2026-09-21
        # pickle -> safetensors migration.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(output_dir / 'mae_best', {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            })
            print(f"  → Saved best model (val_loss={val_loss:.4f})")

        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            save_checkpoint(output_dir / f'mae_epoch{epoch}', {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            })

        scheduler.step()

    print("\n" + "="*80)
    print("PRETRAINING COMPLETE!")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir / 'mae_best.safetensors'} "
          f"(+ {output_dir / 'mae_best.json'})")
    print(f"\nNext step: Use pretrained encoder to initialize U-Net")
    print(f"Command: python train_unet_supervised.py --pretrained {output_dir / 'mae_best'}")

    writer.close()


if __name__ == "__main__":
    main()
