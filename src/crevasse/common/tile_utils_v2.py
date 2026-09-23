"""
Improved utilities for loading, tiling, and visualizing NISAR SAR amplitude data.
Includes smart tile sampling that avoids nodata regions.
"""
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm


class TileGenerator:
    """Generate 512x512 tiles from large raster data with smart sampling."""

    def __init__(self, raster_path: str, tile_size: int = 512):
        """
        Initialize tile generator.

        Args:
            raster_path: Path to the raster file
            tile_size: Size of square tiles (default: 512)
        """
        self.raster_path = Path(raster_path)
        self.tile_size = tile_size

        with rasterio.open(self.raster_path) as src:
            self.width = src.width
            self.height = src.height
            self.crs = src.crs
            self.transform = src.transform
            self.dtype = src.dtypes[0]
            self.nodata = src.nodata

    def get_tile_grid(self) -> List[Tuple[int, int]]:
        """
        Get list of tile positions (row, col) in grid coordinates.

        Returns:
            List of (row_idx, col_idx) tuples
        """
        tiles = []
        for row_idx in range(0, self.height, self.tile_size):
            for col_idx in range(0, self.width, self.tile_size):
                tiles.append((row_idx, col_idx))
        return tiles

    def read_tile(self, row: int, col: int) -> Tuple[np.ndarray, Window]:
        """
        Read a single tile from the raster.

        Args:
            row: Starting row position
            col: Starting column position

        Returns:
            Tuple of (tile_data, rasterio_window)
        """
        with rasterio.open(self.raster_path) as src:
            window = Window(col, row, self.tile_size, self.tile_size)
            tile = src.read(1, window=window)
            return tile, window

    def tile_has_data(self, tile: np.ndarray, min_valid_fraction: float = 0.1) -> bool:
        """
        Check if a tile has sufficient valid data.

        Args:
            tile: Tile array
            min_valid_fraction: Minimum fraction of non-zero pixels required

        Returns:
            True if tile has enough valid data
        """
        valid_pixels = (tile > 0).sum()
        total_pixels = tile.size
        return (valid_pixels / total_pixels) >= min_valid_fraction

    def find_valid_tiles(self, max_tiles: Optional[int] = None,
                        min_valid_fraction: float = 0.1,
                        sample_rate: int = 10) -> List[Tuple[int, int]]:
        """
        Find tiles with valid data by sampling the grid.

        Args:
            max_tiles: Maximum number of valid tiles to find (None = all)
            min_valid_fraction: Minimum fraction of non-zero pixels
            sample_rate: Check every Nth tile (1 = check all)

        Returns:
            List of (row, col) positions for valid tiles
        """
        tile_positions = self.get_tile_grid()
        valid_tiles = []

        # Sample positions based on sample_rate
        sampled_positions = tile_positions[::sample_rate]

        print(f"Searching for valid tiles (checking {len(sampled_positions)}/{len(tile_positions)} positions)...")

        for row, col in tqdm(sampled_positions, desc="Checking tiles"):
            tile, _ = self.read_tile(row, col)

            if self.tile_has_data(tile, min_valid_fraction):
                valid_tiles.append((row, col))

                if max_tiles and len(valid_tiles) >= max_tiles:
                    break

        return valid_tiles

    def sample_valid_tiles(self, n_samples: int = 10,
                          min_valid_fraction: float = 0.1) -> List[Tuple[np.ndarray, int, int]]:
        """
        Sample tiles that contain valid data (avoid nodata regions).

        Args:
            n_samples: Number of valid tiles to sample
            min_valid_fraction: Minimum fraction of non-zero pixels

        Returns:
            List of (tile_data, row, col) tuples
        """
        # First, find valid tiles
        valid_positions = self.find_valid_tiles(
            max_tiles=n_samples * 3,  # Find 3x more than needed
            min_valid_fraction=min_valid_fraction,
            sample_rate=5  # Check every 5th tile for speed
        )

        if len(valid_positions) == 0:
            print("Warning: No valid tiles found! Falling back to random sampling.")
            return self.sample_random_tiles(n_samples)

        print(f"Found {len(valid_positions)} valid tiles")

        # Randomly sample from valid tiles
        n_samples = min(n_samples, len(valid_positions))
        rng = np.random.default_rng()
        sampled_indices = rng.choice(len(valid_positions), size=n_samples, replace=False)

        sampled_tiles = []
        for idx in sampled_indices:
            row, col = valid_positions[idx]
            tile, _ = self.read_tile(row, col)
            sampled_tiles.append((tile, row, col))

        return sampled_tiles

    def sample_random_tiles(self, n_samples: int = 10) -> List[Tuple[np.ndarray, int, int]]:
        """
        Sample random tiles from the raster (may include nodata tiles).

        Args:
            n_samples: Number of random tiles to sample

        Returns:
            List of (tile_data, row, col) tuples
        """
        tile_positions = self.get_tile_grid()
        n_samples = min(n_samples, len(tile_positions))

        rng = np.random.default_rng()
        sampled_positions = rng.choice(len(tile_positions), size=n_samples, replace=False)

        sampled_tiles = []
        for idx in sampled_positions:
            row, col = tile_positions[idx]
            tile, _ = self.read_tile(row, col)
            sampled_tiles.append((tile, row, col))

        return sampled_tiles

    def generate_all_tiles(self, output_dir: Optional[str] = None,
                          skip_empty: bool = True,
                          min_valid_fraction: float = 0.1) -> List[np.ndarray]:
        """
        Generate all tiles from the raster.

        Args:
            output_dir: Optional directory to save tiles as numpy arrays
            skip_empty: Skip tiles with no data
            min_valid_fraction: Minimum fraction of valid pixels for non-empty tiles

        Returns:
            List of tile arrays
        """
        tiles = []
        tile_positions = self.get_tile_grid()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        for idx, (row, col) in enumerate(tqdm(tile_positions, desc="Generating tiles")):
            tile, _ = self.read_tile(row, col)

            if skip_empty and not self.tile_has_data(tile, min_valid_fraction):
                continue

            tiles.append(tile)

            if output_dir:
                tile_name = f"tile_{idx:05d}_r{row}_c{col}.npy"
                np.save(output_path / tile_name, tile)

        return tiles


def plot_tiles(tiles: List[np.ndarray],
               nrows: int = 2,
               ncols: int = 5,
               figsize: Tuple[int, int] = (20, 8),
               cmap: str = 'gray',
               vmin: Optional[float] = None,
               vmax: Optional[float] = None,
               log_scale: bool = True):
    """
    Plot a grid of tiles.

    Args:
        tiles: List of tile arrays (can be tuples with metadata)
        nrows: Number of rows in plot grid
        ncols: Number of columns in plot grid
        figsize: Figure size
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        log_scale: Whether to use log scale for amplitude data
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    for idx, ax in enumerate(axes):
        if idx < len(tiles):
            # Handle both plain arrays and tuples with metadata
            if isinstance(tiles[idx], tuple):
                tile = tiles[idx][0]
                row, col = tiles[idx][1], tiles[idx][2]
                title = f"Tile r{row} c{col}"
            else:
                tile = tiles[idx]
                title = f"Tile {idx}"

            # Mask zeros
            tile = tile.copy()
            tile[tile <= 0] = np.nan

            # Apply log scale if requested (common for SAR amplitude)
            if log_scale:
                plot_data = np.log10(tile + 1e-10)
            else:
                plot_data = tile

            # Auto-scale if not provided
            if vmin is None or vmax is None:
                valid_data = plot_data[np.isfinite(plot_data)]
                if len(valid_data) > 0:
                    p2, p98 = np.percentile(valid_data, [2, 98])
                    vmin_use = vmin if vmin is not None else p2
                    vmax_use = vmax if vmax is not None else p98
                else:
                    vmin_use, vmax_use = 0, 1
            else:
                vmin_use, vmax_use = vmin, vmax

            im = ax.imshow(plot_data, cmap=cmap, vmin=vmin_use, vmax=vmax_use)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')

    plt.tight_layout()
    return fig


def load_raster_info(raster_path: str) -> dict:
    """
    Load and display raster metadata.

    Args:
        raster_path: Path to raster file

    Returns:
        Dictionary of metadata
    """
    with rasterio.open(raster_path) as src:
        info = {
            'width': src.width,
            'height': src.height,
            'bands': src.count,
            'dtype': src.dtypes[0],
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'nodata': src.nodata
        }
    return info


def print_raster_info(raster_path: str):
    """Print formatted raster information."""
    info = load_raster_info(raster_path)

    print(f"Raster: {Path(raster_path).name}")
    print(f"  Dimensions: {info['width']} x {info['height']} pixels")
    print(f"  Bands: {info['bands']}")
    print(f"  Data type: {info['dtype']}")
    print(f"  CRS: {info['crs']}")
    print(f"  Bounds: {info['bounds']}")
    print(f"  NoData: {info['nodata']}")
    print(f"  Size: {info['width'] * info['height'] / 1e6:.1f} megapixels")
