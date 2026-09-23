"""
GPU-accelerated batch inference for crevasse detection using U-Net.
Process multiple tiles in parallel on GPU for maximum speed.
"""
import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from crevasse.common.unet_model import UNet
from crevasse.common.ckpt_io import load_checkpoint
from crevasse.common.tile_utils_v2 import TileGenerator
from crevasse.nisar.find_data_swath import find_data_bounds, get_valid_tile_positions
from crevasse.nisar.infer_unet import preprocess_tile


def batch_predict_gpu(model, tiles, device, threshold=0.5, batch_size=8):
    """
    Predict on multiple tiles in batches on GPU.

    Args:
        model: Trained U-Net
        tiles: List of numpy arrays [H, W]
        device: torch device
        threshold: Binary threshold
        batch_size: Number of tiles per batch

    Returns:
        List of (binary_mask, prob_map) tuples
    """
    model.eval()
    results = []

    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i+batch_size]

            # Preprocess all tiles in batch
            batch_tensors = []
            for tile in batch_tiles:
                tensor = preprocess_tile(tile)
                batch_tensors.append(tensor)

            # Stack into batch [B, 1, H, W]
            batch_input = torch.cat(batch_tensors, dim=0).to(device)

            # Predict entire batch at once
            logits = model(batch_input)
            probs = torch.sigmoid(logits)

            # Convert to numpy
            for j in range(len(batch_tiles)):
                prob_map = probs[j, 0].cpu().numpy()
                binary_mask = (prob_map > threshold).astype(np.uint8)
                results.append((binary_mask, prob_map))

    return results


def process_tiles_gpu(raster_path: str,
                     model,
                     device,
                     tile_size: int = 512,
                     threshold: float = 0.5,
                     batch_size: int = 8,
                     max_tiles: int = None,
                     min_valid_fraction: float = 0.3,
                     save_dir: str = None):
    """
    GPU-accelerated batch processing of tiles.

    Args:
        raster_path: Path to NISAR raster
        model: Trained U-Net
        device: torch device
        tile_size: Tile size
        threshold: Binary threshold
        batch_size: GPU batch size (higher = faster but more memory)
        max_tiles: Maximum tiles to process
        min_valid_fraction: Minimum valid pixel fraction
        save_dir: Directory to save results

    Returns:
        List of results
    """
    tile_gen = TileGenerator(raster_path, tile_size=tile_size)

    # Find valid tile positions
    print(f"\nFinding data swath bounds...")
    bounds, _ = find_data_bounds(raster_path, downsample=100)

    if bounds is None:
        print("No data found in raster!")
        return []

    all_positions = get_valid_tile_positions(bounds, tile_size)
    print(f"Data swath requires {len(all_positions)} tiles")

    # Validate tiles
    print(f"Checking tiles for valid data...")
    valid_positions = []
    for row, col in tqdm(all_positions, desc="Validating"):
        tile, _ = tile_gen.read_tile(row, col)
        valid_fraction = (tile > 0).sum() / tile.size
        if valid_fraction >= min_valid_fraction:
            valid_positions.append((row, col))
            if max_tiles and len(valid_positions) >= max_tiles:
                break

    print(f"Found {len(valid_positions)} tiles with sufficient data")
    print(f"Processing tiles on GPU with batch_size={batch_size}...\n")

    all_results = []

    # Process in batches for GPU efficiency
    pbar = tqdm(total=len(valid_positions), desc="GPU Inference")

    for batch_start in range(0, len(valid_positions), batch_size):
        batch_positions = valid_positions[batch_start:batch_start+batch_size]

        # Load batch of tiles
        batch_tiles = []
        for row, col in batch_positions:
            tile, _ = tile_gen.read_tile(row, col)
            batch_tiles.append(tile)

        # GPU batch prediction
        predictions = batch_predict_gpu(model, batch_tiles, device, threshold, batch_size)

        # Store results
        for (row, col), (binary_mask, prob_map) in zip(batch_positions, predictions):
            crevasse_pixels = binary_mask.sum()
            total_pixels = binary_mask.size
            coverage = crevasse_pixels / total_pixels

            result = {
                'row': row,
                'col': col,
                'tile_size': tile_size,
                'crevasse_coverage': coverage,
                'crevasse_pixels': int(crevasse_pixels)
            }

            all_results.append(result)

        pbar.update(len(batch_positions))

    pbar.close()

    # Save results -- flat scalar columns only, no pickle needed.
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        results_file = save_path / f'unet_gpu_results_tile{tile_size}.npz'
        np.savez(
            results_file,
            row=np.array([r['row'] for r in all_results], dtype=np.int32),
            col=np.array([r['col'] for r in all_results], dtype=np.int32),
            tile_size=np.array([r['tile_size'] for r in all_results], dtype=np.int32),
            crevasse_coverage=np.array(
                [r['crevasse_coverage'] for r in all_results], dtype=np.float32),
            crevasse_pixels=np.array(
                [r['crevasse_pixels'] for r in all_results], dtype=np.int64),
        )
        print(f"\nSaved results to: {results_file}")

    return all_results


def print_summary(results: list):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("GPU BATCH INFERENCE SUMMARY")
    print("="*80)

    if len(results) == 0:
        print("No results to summarize.")
        return

    total_tiles = len(results)
    tiles_with_crevasses = sum(1 for r in results if r['crevasse_coverage'] > 0.01)
    avg_coverage = np.mean([r['crevasse_coverage'] for r in results]) * 100
    max_coverage = max([r['crevasse_coverage'] for r in results]) * 100

    print(f"\nTotal tiles processed: {total_tiles}")
    print(f"Tiles with crevasses (>1%): {tiles_with_crevasses} ({tiles_with_crevasses/total_tiles*100:.1f}%)")
    print(f"Average crevasse coverage: {avg_coverage:.2f}%")
    print(f"Maximum crevasse coverage: {max_coverage:.2f}%")

    # Top hotspots
    if tiles_with_crevasses > 0:
        top_tiles = sorted([r for r in results if r['crevasse_coverage'] > 0],
                          key=lambda r: r['crevasse_coverage'], reverse=True)[:10]
        print(f"\nTop {min(10, len(top_tiles))} crevasse hotspots:")
        for i, tile in enumerate(top_tiles, 1):
            print(f"  {i}. Tile (r{tile['row']}, c{tile['col']}): {tile['crevasse_coverage']*100:.2f}% coverage")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated batch inference for crevasse detection"
    )
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained U-Net checkpoint")
    parser.add_argument("--raster", type=str, required=True,
                       help="Path to NISAR raster")
    parser.add_argument("--tile-size", type=int, default=512,
                       help="Tile size (must match training)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Binary threshold")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="GPU batch size (higher = faster, more memory)")
    parser.add_argument("--max-tiles", type=int, default=None,
                       help="Maximum tiles to process")
    parser.add_argument("--output-dir", type=str, default="../data/unet_gpu_predictions")
    parser.add_argument("--base-channels", type=int, default=64)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cpu':
        print("WARNING: CUDA not available, running on CPU (will be slow)")

    # Check GPU memory
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n" + "="*80)
    print("GPU BATCH CREVASSE INFERENCE")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Raster: {Path(args.raster).name}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")
    print(f"Batch size: {args.batch_size}")
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

    # Process tiles
    results = process_tiles_gpu(
        raster_path=args.raster,
        model=model,
        device=device,
        tile_size=args.tile_size,
        threshold=args.threshold,
        batch_size=args.batch_size,
        max_tiles=args.max_tiles,
        min_valid_fraction=0.1,
        save_dir=args.output_dir
    )

    # Summary
    print_summary(results)

    print(f"\nResults saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
