"""
Identify the actual data swath in the NISAR raster to avoid processing empty tiles.
"""
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def find_data_bounds(raster_path: str, downsample: int = 100):
    """
    Find the actual data bounds by downsampling and checking for non-zero pixels.

    Args:
        raster_path: Path to raster
        downsample: Downsample factor for fast checking

    Returns:
        Dictionary with bounds info
    """
    with rasterio.open(raster_path) as src:
        # Read heavily downsampled version
        data = src.read(
            1,
            out_shape=(
                src.height // downsample,
                src.width // downsample
            ),
            resampling=rasterio.enums.Resampling.nearest
        )

        height, width = src.height, src.width
        transform = src.transform

    # Find where data exists
    mask = data > 0

    # Find bounding box of data
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        print("No data found!")
        return None

    # Convert back to full resolution coordinates
    row_min = rows[0] * downsample
    row_max = (rows[-1] + 1) * downsample
    col_min = cols[0] * downsample
    col_max = (cols[-1] + 1) * downsample

    # Data coverage
    data_pixels = mask.sum()
    total_pixels = mask.size
    coverage = data_pixels / total_pixels * 100

    bounds = {
        'row_min': row_min,
        'row_max': min(row_max, height),
        'col_min': col_min,
        'col_max': min(col_max, width),
        'coverage_percent': coverage,
        'full_height': height,
        'full_width': width
    }

    return bounds, mask


def get_valid_tile_positions(bounds: dict, tile_size: int = 4096):
    """
    Generate tile positions that cover the data swath.

    Args:
        bounds: Bounds dictionary from find_data_bounds
        tile_size: Tile size

    Returns:
        List of (row, col) positions
    """
    positions = []

    for row in range(bounds['row_min'], bounds['row_max'], tile_size):
        for col in range(bounds['col_min'], bounds['col_max'], tile_size):
            positions.append((row, col))

    return positions


def visualize_tile_coverage(raster_path: str, tile_size: int = 4096,
                           output_file: str = None):
    """
    Visualize where tiles would be placed over the data.
    """
    bounds, mask = find_data_bounds(raster_path, downsample=100)

    if bounds is None:
        return

    print("\n" + "="*80)
    print("DATA SWATH ANALYSIS")
    print("="*80)
    print(f"\nFull raster: {bounds['full_height']} x {bounds['full_width']} pixels")
    print(f"Data bounds:")
    print(f"  Rows: {bounds['row_min']} to {bounds['row_max']}")
    print(f"  Cols: {bounds['col_min']} to {bounds['col_max']}")
    print(f"Data coverage: {bounds['coverage_percent']:.1f}% of full extent")

    # Get tile positions
    positions = get_valid_tile_positions(bounds, tile_size)
    print(f"\nTiles needed ({tile_size}x{tile_size}): {len(positions)}")

    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Show data mask
    ax.imshow(mask, cmap='gray', origin='upper',
              extent=[0, bounds['full_width'], bounds['full_height'], 0])

    # Overlay tile grid
    for row, col in positions:
        rect = plt.Rectangle((col, row), tile_size, tile_size,
                            linewidth=1, edgecolor='r', facecolor='none', alpha=0.5)
        ax.add_patch(rect)

    # Mark data bounds
    rect_bounds = plt.Rectangle(
        (bounds['col_min'], bounds['row_min']),
        bounds['col_max'] - bounds['col_min'],
        bounds['row_max'] - bounds['row_min'],
        linewidth=2, edgecolor='lime', facecolor='none', linestyle='--'
    )
    ax.add_patch(rect_bounds)

    ax.set_xlim(0, bounds['full_width'])
    ax.set_ylim(bounds['full_height'], 0)
    ax.set_xlabel('Column (pixels)')
    ax.set_ylabel('Row (pixels)')
    ax.set_title(f'Data Swath and Tile Coverage\n{len(positions)} tiles of {tile_size}x{tile_size}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved visualization to: {output_file}")
    else:
        plt.show()

    plt.close()

    return bounds, positions


def main():
    parser = argparse.ArgumentParser(description="Find data swath bounds")
    parser.add_argument("--raster", type=str, required=True)
    parser.add_argument("--tile-size", type=int, default=4096)
    parser.add_argument("--output", type=str, default="../data/tile_coverage.png")

    args = parser.parse_args()

    bounds, positions = visualize_tile_coverage(
        args.raster,
        tile_size=args.tile_size,
        output_file=args.output
    )

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Use these bounds to process only tiles with data:")
    print(f"  Row range: {bounds['row_min']} to {bounds['row_max']}")
    print(f"  Col range: {bounds['col_min']} to {bounds['col_max']}")
    print(f"  Total tiles: {len(positions)}")


if __name__ == "__main__":
    main()
