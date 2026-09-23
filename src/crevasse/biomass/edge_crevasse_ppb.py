"""
Edge-based crevasse detection with PPB speckle filtering.
Compares PPB vs Enhanced Lee filtering.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from crevasse.biomass.edge_crevasse_v2 import ImprovedEdgeCrevasseDetector
from crevasse.biomass.ppb_filter import ppb_filter, fast_ppb_filter, adaptive_ppb_filter
from crevasse.common.tile_utils_v2 import TileGenerator


class PPBCrevasseDetector(ImprovedEdgeCrevasseDetector):
    """
    Crevasse detector using PPB filter instead of Enhanced Lee.
    """

    def __init__(self, speckle_filter='ppb_fast', **kwargs):
        """
        Args:
            speckle_filter: 'ppb', 'ppb_fast', 'ppb_adaptive', or 'lee' (default: 'ppb_fast')
            **kwargs: Other parameters for ImprovedEdgeCrevasseDetector
        """
        super().__init__(**kwargs)
        self.speckle_filter = speckle_filter

    def apply_speckle_filter(self, image):
        """Override parent's speckle filter."""
        if self.speckle_filter == 'ppb':
            print("  Applying PPB filter (slow, high quality)...")
            return ppb_filter(image, patch_size=7, search_window=21, h=0.6)
        elif self.speckle_filter == 'ppb_fast':
            print("  Applying fast PPB filter...")
            return fast_ppb_filter(image, patch_size=7, search_window=11, h=0.6)
        elif self.speckle_filter == 'ppb_adaptive':
            print("  Applying adaptive PPB filter...")
            return adaptive_ppb_filter(image, patch_size=7, search_window=21)
        elif self.speckle_filter == 'lee':
            print("  Applying Enhanced Lee filter (baseline)...")
            return super().apply_speckle_filter(image)
        else:
            raise ValueError(f"Unknown filter: {self.speckle_filter}")


def compare_filters_on_tile(raster_path, row, col, tile_size=4096, output_dir=None):
    """
    Compare PPB vs Lee filtering on a single tile.

    Args:
        raster_path: Path to NISAR raster
        row, col: Tile position
        tile_size: Tile size
        output_dir: Where to save results
    """
    print("\n" + "="*80)
    print("COMPARING SPECKLE FILTERS FOR CREVASSE DETECTION")
    print("="*80)
    print(f"Tile position: (r{row}, c{col})")
    print(f"Tile size: {tile_size}x{tile_size}\n")

    # Test different filters
    filters = ['lee', 'ppb_fast', 'ppb', 'ppb_adaptive']
    results = {}

    for filt in filters:
        print(f"\n--- Testing {filt.upper()} filter ---")

        detector = PPBCrevasseDetector(
            speckle_filter=filt,
            speckle_window=7,
            highpass_size=15,
            min_texture_cov=0.3,
            min_ridge_size=150,
        )

        # Load tile
        tile_gen = TileGenerator(raster_path, tile_size=tile_size)
        tile, metadata = tile_gen.read_tile(row, col)

        # Detect
        result = detector.process_tile(tile)

        results[filt] = {
            'n_lines': result['n_lines'],
            'tile': tile,
            'filtered': result['tile_despeckled'],
            'ridge_response': result['ridge_response'],
            'ridge_mask': result['edges_enhanced'],
        }

        print(f"  Ridge candidates: {result['n_lines']}")

    # Visualize comparison
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create comparison figure
        fig, axes = plt.subplots(len(filters), 4, figsize=(20, 5*len(filters)))

        for i, filt in enumerate(filters):
            res = results[filt]

            # Original
            tile_log = np.log10(res['tile'] + 1e-10)
            axes[i, 0].imshow(tile_log, cmap='gray')
            axes[i, 0].set_title(f'{filt.upper()}\nOriginal', fontweight='bold')
            axes[i, 0].axis('off')

            # Filtered
            filtered_log = np.log10(res['filtered'] + 1e-10)
            axes[i, 1].imshow(filtered_log, cmap='gray')
            axes[i, 1].set_title(f'Filtered ({filt})', fontweight='bold')
            axes[i, 1].axis('off')

            # Ridge response
            axes[i, 2].imshow(res['ridge_response'], cmap='inferno')
            axes[i, 2].set_title(f'Ridge Response', fontweight='bold')
            axes[i, 2].axis('off')

            # Ridge mask overlay
            axes[i, 3].imshow(tile_log, cmap='gray')
            axes[i, 3].imshow(res['ridge_mask'], cmap='Reds', alpha=0.6)
            axes[i, 3].set_title(f'Ridge Candidates: {res["n_lines"]}', fontweight='bold')
            axes[i, 3].axis('off')

        fig.suptitle(f'Speckle Filter Comparison - Tile (r{row}, c{col})',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = output_path / f'filter_comparison_r{row}_c{col}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved comparison to: {output_file}")
        plt.close()

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for filt in filters:
        print(f"{filt.upper():15s} : {results[filt]['n_lines']:4d} lines")
    print("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare PPB vs Lee filtering for crevasse detection"
    )
    parser.add_argument("--raster", type=str, required=True,
                       help="Path to NISAR raster")
    parser.add_argument("--row", type=int, required=True,
                       help="Tile row position")
    parser.add_argument("--col", type=int, required=True,
                       help="Tile column position")
    parser.add_argument("--tile-size", type=int, default=4096,
                       help="Tile size (default 4096)")
    parser.add_argument("--output-dir", type=str, default="../data/ppb_results",
                       help="Output directory")

    args = parser.parse_args()

    results = compare_filters_on_tile(
        args.raster,
        args.row,
        args.col,
        args.tile_size,
        args.output_dir
    )


if __name__ == "__main__":
    main()
