"""
Inference script to detect crevasses using trained U-Net.
"""
import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from crevasse.common.unet_model import UNet
from crevasse.common.ckpt_io import load_checkpoint
from crevasse.common.tile_utils_v2 import TileGenerator
from crevasse.nisar.find_data_swath import find_data_bounds, get_valid_tile_positions


def preprocess_tile(tile: np.ndarray) -> torch.Tensor:
    """
    Preprocess SAR tile for U-Net input.

    Args:
        tile: Raw SAR amplitude [H, W]

    Returns:
        Preprocessed tensor [1, 1, H, W]
    """
    # Mask zeros
    tile = tile.copy()
    tile[tile <= 0] = np.nan

    # Log scale
    tile_log = np.log10(tile + 1e-10)

    # Normalize to 0-1
    valid = tile_log[np.isfinite(tile_log)]
    if len(valid) > 0:
        p2, p98 = np.percentile(valid, [2, 98])
        tile_norm = np.clip(tile_log, p2, p98)
        if p98 > p2:
            tile_norm = (tile_norm - p2) / (p98 - p2)
        else:
            tile_norm = np.zeros_like(tile_log)
    else:
        tile_norm = np.zeros_like(tile_log)

    tile_norm = np.nan_to_num(tile_norm, nan=0).astype(np.float32)

    # Convert to tensor [1, 1, H, W]
    tensor = torch.from_numpy(tile_norm).unsqueeze(0).unsqueeze(0)
    return tensor


@torch.no_grad()
def predict_tile(model, tile: np.ndarray, device, threshold=0.5):
    """
    Predict crevasse mask for a single tile.

    Args:
        model: Trained U-Net
        tile: Raw SAR tile [H, W]
        device: torch device
        threshold: Probability threshold for binary mask

    Returns:
        Binary mask [H, W], probability map [H, W]
    """
    model.eval()

    # Preprocess
    tensor = preprocess_tile(tile).to(device)

    # Predict
    logits = model(tensor)
    probs = torch.sigmoid(logits)

    # Convert to numpy
    prob_map = probs.squeeze().cpu().numpy()
    binary_mask = (prob_map > threshold).astype(np.uint8)

    return binary_mask, prob_map


def visualize_prediction(tile: np.ndarray,
                        binary_mask: np.ndarray,
                        prob_map: np.ndarray,
                        row: int, col: int,
                        output_file: str = None):
    """Visualize prediction results."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original (log scale)
    tile_masked = tile.copy()
    tile_masked[tile_masked <= 0] = np.nan
    tile_log = np.log10(tile_masked + 1e-10)

    axes[0].imshow(tile_log, cmap='gray')
    axes[0].set_title('Original SAR (Log)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Probability map
    axes[1].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Crevasse Probability', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    cbar = plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046)
    cbar.set_label('Probability')

    # Binary mask
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title('Binary Mask (Threshold=0.5)', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # Overlay
    axes[3].imshow(tile_log, cmap='gray')
    axes[3].imshow(binary_mask, cmap='Reds', alpha=0.5)
    axes[3].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[3].axis('off')

    crevasse_pixels = binary_mask.sum()
    total_pixels = binary_mask.size
    coverage = crevasse_pixels / total_pixels * 100

    fig.suptitle(f'U-Net Crevasse Detection - Tile (r{row}, c{col})\n'
                 f'Crevasse coverage: {coverage:.2f}%',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def infer_single_tile(model, raster_path, row, col, device,
                     tile_size=512, threshold=0.5, output_dir=None):
    """Infer on a single tile."""
    tile_gen = TileGenerator(raster_path, tile_size=tile_size)
    tile, _ = tile_gen.read_tile(row, col)

    # Predict
    binary_mask, prob_map = predict_tile(model, tile, device, threshold)

    # Visualize
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"prediction_r{row}_c{col}.png"
        visualize_prediction(tile, binary_mask, prob_map, row, col, str(output_file))
    else:
        visualize_prediction(tile, binary_mask, prob_map, row, col)

    return binary_mask, prob_map


def infer_multiple_tiles(model, raster_path, device,
                        tile_size=512, threshold=0.5,
                        max_tiles=None, output_dir=None):
    """Infer on multiple valid tiles."""
    print("\nFinding valid tiles...")
    bounds, _ = find_data_bounds(raster_path, downsample=100)
    all_positions = get_valid_tile_positions(bounds, tile_size=tile_size)

    print(f"Found {len(all_positions)} tile positions in data swath")

    if max_tiles:
        all_positions = all_positions[:max_tiles]
        print(f"Limited to {len(all_positions)} tiles")

    print(f"Checking tiles for valid data...\n")

    tile_gen = TileGenerator(raster_path, tile_size=tile_size)

    results = []
    skipped = 0
    processed = 0

    pbar = tqdm(all_positions, desc="Inferring")
    for row, col in pbar:
        tile, _ = tile_gen.read_tile(row, col)

        # Skip if mostly empty
        valid_fraction = (tile > 0).sum() / tile.size
        if valid_fraction < 0.1:  # Lowered threshold from 0.3 to 0.1
            skipped += 1
            pbar.set_postfix({'processed': processed, 'skipped': skipped})
            continue

        processed += 1
        pbar.set_postfix({'processed': processed, 'skipped': skipped})

        # Predict
        binary_mask, prob_map = predict_tile(model, tile, device, threshold)

        # Calculate statistics
        crevasse_pixels = binary_mask.sum()
        total_pixels = binary_mask.size
        coverage = crevasse_pixels / total_pixels

        results.append({
            'row': row,
            'col': col,
            'crevasse_coverage': coverage,
            'mask': binary_mask,
            'prob_map': prob_map
        })

    print(f"\nSkipped {skipped} tiles with <10% valid data")

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save results -- flat scalar columns only; mask/prob_map are dropped here
        # same as before ("Remove large arrays for storage"), so a plain npz needs
        # no pickle at all.
        results_file = output_path / 'unet_predictions.npz'
        np.savez(
            results_file,
            row=np.array([r['row'] for r in results], dtype=np.int32),
            col=np.array([r['col'] for r in results], dtype=np.int32),
            crevasse_coverage=np.array(
                [r['crevasse_coverage'] for r in results], dtype=np.float32),
        )
        print(f"\nSaved results to: {results_file}")

        # Visualize top tiles with most crevasses
        print("\nVisualizing top 5 tiles with most crevasses...")
        top_tiles = sorted(results, key=lambda x: x['crevasse_coverage'], reverse=True)[:5]

        tile_gen = TileGenerator(raster_path, tile_size=tile_size)
        for idx, result in enumerate(top_tiles, 1):
            row, col = result['row'], result['col']
            tile, _ = tile_gen.read_tile(row, col)
            output_file = output_path / f"top{idx}_r{row}_c{col}.png"
            visualize_prediction(tile, result['mask'], result['prob_map'],
                               row, col, str(output_file))
            print(f"  {idx}. Tile (r{row}, c{col}): {result['crevasse_coverage']*100:.2f}% crevasses")

    return results


def main():
    parser = argparse.ArgumentParser(description="U-Net inference for crevasse detection")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained U-Net checkpoint")
    parser.add_argument("--raster", type=str, required=True,
                       help="Path to NISAR raster")
    parser.add_argument("--tile-size", type=int, default=512,
                       help="Tile size (should match training)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Probability threshold for binary mask")

    # Single tile or multiple
    parser.add_argument("--row", type=int, default=None,
                       help="Row position for single tile inference")
    parser.add_argument("--col", type=int, default=None,
                       help="Col position for single tile inference")
    parser.add_argument("--max-tiles", type=int, default=None,
                       help="Max tiles for batch inference (None = all)")

    parser.add_argument("--output-dir", type=str, default="../data/unet_predictions")
    parser.add_argument("--base-channels", type=int, default=64)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "="*80)
    print("U-NET INFERENCE")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Raster: {Path(args.raster).name}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")
    print(f"Threshold: {args.threshold}\n")

    # Load model
    print("Loading trained U-Net...")
    model = UNet(n_channels=1, n_classes=1, base_channels=args.base_channels)
    checkpoint = load_checkpoint(args.model, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    if 'val_iou' in checkpoint:
        print(f"Validation IoU: {checkpoint['val_iou']:.4f}")
    print()

    # Single tile or batch inference
    if args.row is not None and args.col is not None:
        print(f"Running inference on single tile (r{args.row}, c{args.col})...")
        infer_single_tile(model, args.raster, args.row, args.col, device,
                         args.tile_size, args.threshold, args.output_dir)
    else:
        print(f"Running batch inference...")
        results = infer_multiple_tiles(model, args.raster, device,
                                      args.tile_size, args.threshold,
                                      args.max_tiles, args.output_dir)

        # Summary statistics
        print("\n" + "="*80)
        print("INFERENCE SUMMARY")
        print("="*80)
        print(f"Total tiles processed: {len(results)}")

        if len(results) > 0:
            coverages = [r['crevasse_coverage'] for r in results]
            avg_coverage = np.mean(coverages) * 100
            max_coverage = np.max(coverages) * 100

            print(f"Average crevasse coverage: {avg_coverage:.2f}%")
            print(f"Maximum crevasse coverage: {max_coverage:.2f}%")
            print(f"\nResults saved to: {args.output_dir}")
        else:
            print("\n⚠ No tiles processed!")
            print("Possible reasons:")
            print("  - Tile size mismatch (model trained on different size)")
            print("  - All tiles have <30% valid data")
            print(f"\nTry:")
            print(f"  - Use --tile-size that matches training")
            print(f"  - Test on a specific tile with --row and --col")

    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
