"""
Infer on tiles that were used for training (from pseudo-labels).
This ensures we test on tiles with known data coverage.
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
from crevasse.nisar.label_io import load_labels
from crevasse.common.tile_utils_v2 import TileGenerator
from crevasse.nisar.infer_unet import preprocess_tile, predict_tile, visualize_prediction


def main():
    parser = argparse.ArgumentParser(
        description="Infer on tiles from training set (pseudo-labels)"
    )
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained U-Net checkpoint")
    parser.add_argument("--raster", type=str, required=True,
                       help="Path to NISAR raster")
    parser.add_argument("--results-npz", type=str, required=True,
                       help="Pseudo-label file (label_io.py's npz format)")
    parser.add_argument("--tile-size", type=int, default=256,
                       help="Tile size for inference")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-tiles", type=int, default=10)
    parser.add_argument("--min-lines", type=int, default=0,
                        help="Minimum ridge component count per tile. Default 0 (off): "
                             "the label file is already a curated positive set; the old "
                             "hardcoded >10 dropped every tile on post-redesign labels.")
    parser.add_argument("--output-dir", type=str, default="../data/unet_predictions_labels")
    parser.add_argument("--base-channels", type=int, default=64)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)
    print("U-NET INFERENCE ON TRAINING TILES")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Pseudo-labels: {args.results_npz}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}\n")

    # Load model
    print("Loading U-Net...")
    model = UNet(n_channels=1, n_classes=1, base_channels=args.base_channels)
    checkpoint = load_checkpoint(args.model, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}\n")

    # Load pseudo-labels
    print("Loading pseudo-labels...")
    label_data = load_labels(args.results_npz)

    # Filter to tiles with crevasses
    tiles_with_crevasses = [r for r in label_data if r.get('n_lines', 0) >= args.min_lines]
    print(f"Found {len(tiles_with_crevasses)}/{len(label_data)} tiles "
          f"with >={args.min_lines} ridge components")

    if args.max_tiles:
        tiles_with_crevasses = tiles_with_crevasses[:args.max_tiles]
        print(f"Processing first {len(tiles_with_crevasses)} tiles\n")

    # Create tile generator
    tile_gen = TileGenerator(args.raster, tile_size=args.tile_size)

    # Output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each tile
    results = []

    for idx, label_info in enumerate(tqdm(tiles_with_crevasses, desc="Inferring")):
        row_orig = label_info['row']
        col_orig = label_info['col']
        n_lines = label_info.get('n_lines', 0)
        orig_tile_size = label_info.get('tile_size', 4096)

        print(f"\nTile {idx+1}: Original position (r{row_orig}, c{col_orig}) "
              f"at {orig_tile_size}x{orig_tile_size}, {n_lines} ridge components")

        # If tile sizes match, use directly
        if orig_tile_size == args.tile_size:
            row, col = row_orig, col_orig
        else:
            # Use center of original tile
            # Original tile spans [row_orig:row_orig+orig_tile_size]
            # Take center tile of size args.tile_size
            offset = (orig_tile_size - args.tile_size) // 2
            row = row_orig + offset
            col = col_orig + offset

        print(f"  Inferring at (r{row}, c{col}) with {args.tile_size}x{args.tile_size}")

        # Load tile
        tile, _ = tile_gen.read_tile(row, col)

        # Check valid data
        valid_fraction = (tile > 0).sum() / tile.size
        print(f"  Valid data: {valid_fraction*100:.1f}%")

        if valid_fraction < 0.1:
            print(f"  Skipping (insufficient data)")
            continue

        # Predict
        binary_mask, prob_map = predict_tile(model, tile, device, args.threshold)

        # Calculate coverage
        crevasse_coverage = binary_mask.sum() / binary_mask.size
        print(f"  Predicted crevasse coverage: {crevasse_coverage*100:.2f}%")

        # Visualize
        output_file = output_path / f"tile{idx+1}_r{row}_c{col}.png"
        visualize_prediction(tile, binary_mask, prob_map, row, col, str(output_file))

        results.append({
            'idx': idx,
            'row': row,
            'col': col,
            'original_row': row_orig,
            'original_col': col_orig,
            'n_lines_pseudolabel': n_lines,
            'crevasse_coverage': crevasse_coverage,
            'valid_fraction': valid_fraction
        })

    # Summary
    print("\n" + "="*80)
    print("INFERENCE SUMMARY")
    print("="*80)
    print(f"Tiles processed: {len(results)}")

    if len(results) > 0:
        avg_coverage = np.mean([r['crevasse_coverage'] for r in results]) * 100
        print(f"Average crevasse coverage: {avg_coverage:.2f}%")

        # Save results -- flat scalar columns only (mask/prob_map were never kept
        # here, just used for the PNG visualization above), so a plain npz needs no
        # pickle at all.
        np.savez(
            output_path / 'inference_results.npz',
            idx=np.array([r['idx'] for r in results], dtype=np.int32),
            row=np.array([r['row'] for r in results], dtype=np.int32),
            col=np.array([r['col'] for r in results], dtype=np.int32),
            original_row=np.array([r['original_row'] for r in results], dtype=np.int32),
            original_col=np.array([r['original_col'] for r in results], dtype=np.int32),
            n_lines_pseudolabel=np.array(
                [r['n_lines_pseudolabel'] for r in results], dtype=np.float32),
            crevasse_coverage=np.array(
                [r['crevasse_coverage'] for r in results], dtype=np.float32),
            valid_fraction=np.array(
                [r['valid_fraction'] for r in results], dtype=np.float32),
        )
        print(f"\nResults saved to: {output_path}")

    print("="*80)


if __name__ == "__main__":
    main()
