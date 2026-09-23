"""
Filters for identifying actual crevasses vs random texture in a ridge mask.

Crevasses are curved, elongated features, not straight lines, so filtering
operates on connected components of a binary ridge mask (from a multiscale
vesselness/ridge filter) rather than on fitted line segments and their
orientation. Filtering by orientation is deliberately avoided: Canny+Hough
line detection has a known bias toward 0/45/90/135 degrees from gradient
discretization on a square pixel grid, which previously caused pure-speckle
tiles (no real crevasses) to be misclassified as containing a dominant
parallel crevasse orientation.
"""
import numpy as np
from typing import Tuple, Dict
from skimage.measure import label, regionprops


def label_ridge_components(ridge_mask: np.ndarray) -> np.ndarray:
    """
    Label connected components of a binary ridge mask.

    Args:
        ridge_mask: Binary ridge mask

    Returns:
        Labeled array (0 = background, 1..N = component ids)
    """
    return label(ridge_mask, connectivity=2)


def compute_component_properties(labeled: np.ndarray) -> list:
    """
    Compute shape properties for each connected component.

    Args:
        labeled: Labeled array from label_ridge_components

    Returns:
        List of regionprops objects (area, eccentricity, major_axis_length, etc.)
    """
    return regionprops(labeled)


def filter_by_size(labeled: np.ndarray, min_size: float = 100.0) -> np.ndarray:
    """
    Keep only components with area >= min_size.

    Args:
        labeled: Labeled array from label_ridge_components
        min_size: Minimum component area in pixels

    Returns:
        Binary mask with small components removed
    """
    mask = np.zeros(labeled.shape, dtype=bool)
    for region in regionprops(labeled):
        if region.area >= min_size:
            mask[labeled == region.label] = True
    return mask


def filter_by_elongation(labeled: np.ndarray, min_eccentricity: float = 0.85) -> np.ndarray:
    """
    Keep only elongated (curve-like) components, rejecting blob-like false positives.

    Eccentricity is 0 for a circle and approaches 1 for an elongated ellipse/line,
    so a high minimum eccentricity favors crevasse-like curves over speckle blobs.

    Args:
        labeled: Labeled array from label_ridge_components
        min_eccentricity: Minimum eccentricity to keep a component

    Returns:
        Binary mask with non-elongated components removed
    """
    mask = np.zeros(labeled.shape, dtype=bool)
    for region in regionprops(labeled):
        if region.eccentricity >= min_eccentricity:
            mask[labeled == region.label] = True
    return mask


def filter_ridge_candidates(ridge_mask: np.ndarray,
                             min_size: float = 100.0,
                             min_eccentricity: float = 0.85) -> Tuple[np.ndarray, Dict]:
    """
    Complete filtering pipeline for ridge-based crevasse candidates.

    Args:
        ridge_mask: Binary ridge mask (post-threshold, pre-filtering)
        min_size: Minimum component area in pixels
        min_eccentricity: Minimum eccentricity (elongation) to keep a component

    Returns:
        Tuple of (filtered_mask, statistics)
    """
    labeled = label_ridge_components(ridge_mask)
    regions = compute_component_properties(labeled)

    stats = {
        'total_components': len(regions),
        'after_size_filter': 0,
        'after_elongation_filter': 0,
        'n_components': 0,
        'total_length': 0.0,
    }

    if len(regions) == 0:
        return np.zeros_like(ridge_mask, dtype=bool), stats

    kept = np.zeros(ridge_mask.shape, dtype=bool)
    n_after_size = 0
    n_after_elongation = 0
    total_length = 0.0

    for region in regions:
        if region.area < min_size:
            continue
        n_after_size += 1

        if region.eccentricity < min_eccentricity:
            continue
        n_after_elongation += 1

        kept[labeled == region.label] = True
        total_length += region.major_axis_length

    stats['after_size_filter'] = n_after_size
    stats['after_elongation_filter'] = n_after_elongation
    stats['n_components'] = n_after_elongation
    stats['total_length'] = total_length

    return kept, stats
