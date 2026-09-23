"""
SAR speckle filtering for improved feature detection.
"""
import numpy as np
from scipy.ndimage import uniform_filter, generic_filter


def lee_filter(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply Lee filter for SAR speckle reduction.

    Args:
        img: Input image
        window_size: Size of the filter window

    Returns:
        Filtered image
    """
    # Local mean
    mean = uniform_filter(img, size=window_size)

    # Local variance
    sqr_mean = uniform_filter(img**2, size=window_size)
    variance = sqr_mean - mean**2
    variance = np.maximum(variance, 0)  # Ensure non-negative

    # Estimate noise variance (using whole image statistics)
    overall_variance = np.var(img)

    # Lee filter formula
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = variance / (variance + overall_variance)
    weights = np.nan_to_num(weights, nan=0, posinf=0, neginf=0)
    weights = np.clip(weights, 0, 1)

    filtered = mean + weights * (img - mean)

    return filtered


def frost_filter(img: np.ndarray, window_size: int = 5, damping: float = 1.0) -> np.ndarray:
    """
    Apply Frost filter for SAR speckle reduction.

    Args:
        img: Input image
        window_size: Size of the filter window
        damping: Damping factor (higher = more smoothing)

    Returns:
        Filtered image
    """
    def frost_window(values):
        if len(values) == 0 or np.all(values == 0):
            return 0

        center_idx = len(values) // 2
        center_val = values[center_idx]

        mean_val = np.mean(values)
        std_val = np.std(values)

        if mean_val == 0:
            return center_val

        # Coefficient of variation
        cv = std_val / mean_val if mean_val != 0 else 0

        # Frost weight
        k = damping * cv

        # Distance weights (simplified - center weighted)
        distances = np.abs(np.arange(len(values)) - center_idx)
        weights = np.exp(-k * distances)

        return np.average(values, weights=weights)

    filtered = generic_filter(img, frost_window, size=window_size)

    return filtered


def enhanced_lee_filter(img: np.ndarray, window_size: int = 7, num_looks: int = 1) -> np.ndarray:
    """
    Enhanced Lee filter with edge preservation.

    Args:
        img: Input image (amplitude)
        window_size: Size of the filter window
        num_looks: Number of looks (affects noise model)

    Returns:
        Filtered image
    """
    # Add small epsilon to avoid log(0)
    img_safe = img + 1e-10

    # Local statistics
    mean = uniform_filter(img_safe, size=window_size)
    sqr_mean = uniform_filter(img_safe**2, size=window_size)
    variance = sqr_mean - mean**2
    variance = np.maximum(variance, 0)

    # Coefficient of variation
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = np.sqrt(variance) / mean
    cv = np.nan_to_num(cv, nan=0, posinf=0, neginf=0)

    # Theoretical CV for speckle
    cv_speckle = 1.0 / np.sqrt(num_looks)

    # Edge detection via CV
    # High CV = edge, Low CV = homogeneous
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.maximum(0, (cv**2 - cv_speckle**2) / (cv**2 + 1e-10))
    weights = np.nan_to_num(weights, nan=0, posinf=0, neginf=0)
    weights = np.clip(weights, 0, 1)

    # Apply filter
    filtered = mean + weights * (img_safe - mean)

    return filtered


def median_filter_sar(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Simple median filter for SAR (preserves edges better than mean).

    Args:
        img: Input image
        window_size: Size of the filter window

    Returns:
        Filtered image
    """
    from scipy.ndimage import median_filter
    return median_filter(img, size=window_size)


def adaptive_wiener_filter(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Adaptive Wiener filter for speckle reduction.

    Args:
        img: Input image
        window_size: Size of the filter window

    Returns:
        Filtered image
    """
    # Local mean
    mean = uniform_filter(img, size=window_size)

    # Local variance
    sqr_mean = uniform_filter(img**2, size=window_size)
    variance = sqr_mean - mean**2
    variance = np.maximum(variance, 0)

    # Estimate noise variance (minimum local variance)
    noise_var = np.percentile(variance[variance > 0], 5) if np.any(variance > 0) else 0

    # Wiener filter
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.maximum(0, (variance - noise_var) / (variance + 1e-10))
    weights = np.nan_to_num(weights, nan=0, posinf=0, neginf=0)
    weights = np.clip(weights, 0, 1)

    filtered = mean + weights * (img - mean)

    return filtered
