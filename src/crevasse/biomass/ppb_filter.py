"""
Probabilistic Patch-Based (PPB) filter for SAR speckle reduction.
Better edge preservation than traditional Lee filters.
"""
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.util import view_as_windows


def ppb_filter(image, patch_size=7, search_window=21, h=0.6, sigma=None):
    """
    Probabilistic Patch-Based filter for SAR speckle reduction.

    Based on Non-Local Means adapted for multiplicative speckle noise.

    Args:
        image: Input SAR amplitude image
        patch_size: Size of patches for comparison (odd number, default 7)
        search_window: Size of search window (odd number, default 21)
        h: Filtering parameter (higher = more smoothing, default 0.6)
        sigma: Noise standard deviation (auto-estimated if None)

    Returns:
        Filtered image with reduced speckle
    """
    # Work with intensity (amplitude squared) for multiplicative noise model
    intensity = image ** 2

    # Estimate noise if not provided
    if sigma is None:
        # Use median absolute deviation for robust estimation
        sigma = np.median(np.abs(intensity - np.median(intensity))) / 0.6745

    # Pad image
    pad_width = search_window // 2
    intensity_padded = np.pad(intensity, pad_width, mode='reflect')

    # Output array
    filtered = np.zeros_like(intensity)
    weights_sum = np.zeros_like(intensity)

    h2 = h * h * sigma * sigma

    # For each pixel in search window
    offset = search_window // 2
    for i in range(-offset, offset + 1):
        for j in range(-offset, offset + 1):
            # Shift the image
            shifted = np.roll(intensity_padded, (i, j), axis=(0, 1))

            # Extract patches
            patch_ref = intensity_padded[pad_width:-pad_width, pad_width:-pad_width]
            patch_comp = shifted[pad_width:-pad_width, pad_width:-pad_width]

            # Compute patch distances (for multiplicative noise, use ratio)
            # Using log-ratio for better multiplicative noise handling
            ratio = np.maximum(patch_ref / (patch_comp + 1e-10), 1e-10)
            log_ratio = np.log(ratio)
            distance = log_ratio ** 2

            # Smooth distances over patch
            distance_smooth = uniform_filter(distance, size=patch_size, mode='constant')

            # Compute weights
            w = np.exp(-np.maximum(distance_smooth, 0.0) / h2)

            # Accumulate weighted values
            filtered += w * patch_comp
            weights_sum += w

    # Normalize
    filtered = filtered / (weights_sum + 1e-10)

    # Back to amplitude
    filtered = np.sqrt(np.maximum(filtered, 0))

    return filtered


def fast_ppb_filter(image, patch_size=7, search_window=11, h=0.6):
    """
    Faster version using smaller search window.
    Good balance between quality and speed.

    Args:
        image: Input SAR amplitude image
        patch_size: Size of patches (default 7)
        search_window: Size of search window (default 11, smaller = faster)
        h: Filtering parameter (default 0.6)

    Returns:
        Filtered image
    """
    return ppb_filter(image, patch_size, search_window, h)


def adaptive_ppb_filter(image, patch_size=7, search_window=21):
    """
    PPB with adaptive h parameter based on local statistics.
    Better for varying speckle levels.

    Args:
        image: Input SAR amplitude image
        patch_size: Size of patches
        search_window: Size of search window

    Returns:
        Filtered image
    """
    intensity = image ** 2

    # Estimate local noise variance
    local_mean = uniform_filter(intensity, size=patch_size)
    local_var = uniform_filter(intensity ** 2, size=patch_size) - local_mean ** 2

    # Coefficient of variation (for multiplicative noise)
    cv = np.sqrt(np.maximum(local_var, 0)) / (local_mean + 1e-10)

    # Adaptive h based on local CV
    # Higher noise -> larger h -> more smoothing
    h = 0.4 + 0.6 * np.clip(cv, 0, 1)
    h_median = np.median(h)

    return ppb_filter(image, patch_size, search_window, h_median)


def compare_filters(image):
    """
    Compare PPB with different parameters.

    Args:
        image: Input SAR amplitude

    Returns:
        Dictionary with filtered results
    """
    results = {
        'original': image,
        'ppb_default': ppb_filter(image),
        'ppb_fast': fast_ppb_filter(image),
        'ppb_adaptive': adaptive_ppb_filter(image)
    }

    return results


if __name__ == "__main__":
    import glob
    import matplotlib.pyplot as plt
    import rasterio

    # Test on a small tile from whatever NISAR raster is available
    candidates = sorted(glob.glob("../data/nisar/*.tif"))
    if not candidates:
        raise FileNotFoundError("No rasters found in ../data/nisar/")
    raster_path = candidates[0]

    with rasterio.open(raster_path) as src:
        # Read small region
        window = rasterio.windows.Window(10000, 10000, 512, 512)
        tile = src.read(1, window=window).astype(np.float32)

    print("Testing PPB filters...")
    print(f"Tile shape: {tile.shape}")
    print(f"Tile range: [{tile.min():.3f}, {tile.max():.3f}]")

    # Compare filters
    results = compare_filters(tile)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    titles = ['Original', 'PPB Default', 'PPB Fast', 'PPB Adaptive']
    for ax, (name, img) in zip(axes.flat, results.items()):
        # Log scale for visualization
        img_log = np.log10(img + 1e-10)
        ax.imshow(img_log, cmap='gray')
        title = name.replace('_', ' ').title()
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('../data/ppb_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison to: ../data/ppb_comparison.png")
    plt.close()
