"""
Improved ridge-based crevasse detection with speckle filtering.
"""
import numpy as np
from skimage import morphology
from skimage.filters import frangi
from skimage.morphology import skeletonize
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from crevasse.common.tile_utils_v2 import TileGenerator
from crevasse.biomass.speckle_filters import enhanced_lee_filter, lee_filter
from crevasse.biomass.crevasse_filters import filter_ridge_candidates


class ImprovedEdgeCrevasseDetector:
    """Improved ridge-based crevasse detection with speckle filtering."""

    def __init__(self,
                 speckle_filter: str = 'lee',
                 speckle_window: int = 7,
                 highpass_size: int = 15,
                 min_texture_cov: float = 0.3,
                 ridge_sigmas: range = range(2, 12, 2),
                 black_ridges: bool = False,
                 ridge_threshold: Optional[float] = None,
                 ridge_percentile: float = 90.0,
                 min_ridge_size: float = 100.0,
                 min_eccentricity: float = 0.85,
                 ridge_downscale_factor: int = 1,
                 orient_gate_thresh: Optional[float] = None):
        """
        Initialize improved detector with speckle filtering.

        Args:
            speckle_filter: Type of speckle filter ('lee', 'enhanced_lee', 'none')
            speckle_window: Window size for speckle filter
            highpass_size: Size of Gaussian kernel for high-pass filter
            min_texture_cov: Minimum coefficient of variation for valid features
            ridge_sigmas: Scales for the multiscale Frangi ridge filter (the
                "squinting effect" — larger sigmas see coarser, more coherent
                curvilinear structure while suppressing per-pixel speckle)
            black_ridges: Ridge polarity for Frangi (False = bright ridges).
                Not yet validated against a confirmed real crevasse example;
                treat as tunable, not settled.
            ridge_threshold: Fixed threshold for the ridge response; if None,
                a per-tile percentile threshold is used (see ridge_percentile).
                Otsu was tried first but always finds *some* split point even
                on pure speckle with no real structure — the multiscale filter
                smooths noise into spatially-correlated blobs that still pass
                elongation filtering. A high percentile cut empirically
                rejected these on confirmed-flat tiles where Otsu did not.
            ridge_percentile: Percentile of the nonzero ridge response used as
                the adaptive per-tile threshold when ridge_threshold is None.
                90 was chosen because it drove n_components to 0 on 4
                confirmed pure-speckle NISAR tiles (CV ~0.52-0.53, matching
                theoretical single-look speckle) while Otsu did not. Not yet
                validated against a confirmed real crevasse example — treat
                as tunable, not settled.
            min_ridge_size: Minimum connected-component area (px, in the
                working resolution set by ridge_downscale_factor) to keep as
                a crevasse candidate
            min_eccentricity: Minimum eccentricity (elongation) to keep a
                component — rejects blob-like false positives
            ridge_downscale_factor: Block-average the tile by this factor
                (skimage.transform.downscale_local_mean) before running the
                speckle filter/Frangi pipeline, then upscale the resulting
                mask back to the original tile size (nearest-neighbor).
                Multi-look theory: averaging N pixels cuts speckle variance
                ~1/N, while genuine multi-pixel-wide crevasse bands survive
                averaging largely intact. Validated on 2 reference tiles
                (1 confirmed real crevasse field, 1 confirmed pure speckle)
                from the 025_091 granule: at factor=1 (native res), no
                percentile value cleanly separates real recall from speckle
                false positives (a sharp cliff between too-strict/misses-
                signal and too-loose/reintroduces-noise). At factor=8 with
                ridge_percentile~70-75, the cliff disappears — the speckle
                tile stays near 0 (1 small residual component) across a wide
                percentile range while the real tile recovers substantial
                coverage. Default is 1 (off) for backward compatibility;
                min_ridge_size/min_eccentricity are applied in the
                downscaled coordinate space, so their effective real-world
                meaning changes with this factor.
            orient_gate_thresh: If set, a tile-level DIRECTIONALITY gate. The
                per-tile percentile threshold always keeps the top (100-pct)%
                of Frangi response, so a pure-speckle tile still gets labeled
                (verified 2026-09-01: negatives get the same/higher label
                coverage as positives). Absolute response magnitude does NOT
                separate them (AUC ~0.6), but orientation concentration does
                (AUC ~0.89): real crevasses are directional/aligned, speckle is
                isotropic. When orientation_concentration(response) is below
                this value the whole-tile label is emptied. Operating points on
                RF-gate survivors: 0.10 -> precision 0.955, 0.12 -> 0.969,
                0.15 -> 0.991 (recall trades down 0.62/0.57/0.48). None = off
                (backward compatible).
        """
        self.speckle_filter = speckle_filter
        self.speckle_window = speckle_window
        self.highpass_size = highpass_size
        self.min_texture_cov = min_texture_cov
        self.ridge_sigmas = ridge_sigmas
        self.black_ridges = black_ridges
        self.ridge_threshold = ridge_threshold
        self.ridge_percentile = ridge_percentile
        self.min_ridge_size = min_ridge_size
        self.min_eccentricity = min_eccentricity
        self.ridge_downscale_factor = ridge_downscale_factor
        self.orient_gate_thresh = orient_gate_thresh

    @staticmethod
    def orientation_concentration(response: np.ndarray) -> float:
        """Tile-level directionality of a Frangi response in [0,1].

        Response*coherence-weighted axial resultant of the structure-tensor
        orientation: ~1 when ridges share a dominant orientation (aligned
        crevasses), ~0 when orientations are uniform (isotropic speckle).
        """
        if (response > 0).sum() < 10:
            return 0.0
        Axx, Axy, Ayy = structure_tensor(response.astype(np.float32), sigma=2.0,
                                         mode="reflect", order="rc")
        l1, l2 = structure_tensor_eigenvalues((Axx, Axy, Ayy))
        coh = (l1 - l2) / (l1 + l2 + 1e-9)
        theta = 0.5 * np.arctan2(2 * Axy, (Ayy - Axx))
        w = response * coh
        Z = w.sum() + 1e-9
        return float(np.hypot((w * np.cos(2 * theta)).sum(),
                              (w * np.sin(2 * theta)).sum()) / Z)

    def apply_speckle_filter(self, tile: np.ndarray) -> np.ndarray:
        """
        Apply speckle filter to reduce SAR noise.

        Args:
            tile: Input tile (linear amplitude)

        Returns:
            Despeckled tile
        """
        if self.speckle_filter == 'none':
            return tile
        elif self.speckle_filter == 'lee':
            return lee_filter(tile, window_size=self.speckle_window)
        elif self.speckle_filter == 'enhanced_lee':
            return enhanced_lee_filter(tile, window_size=self.speckle_window)
        else:
            return tile

    def compute_texture_mask(self, tile_norm: np.ndarray) -> np.ndarray:
        """
        Create mask for areas with significant texture (not pure speckle).

        Args:
            tile_norm: Normalized tile

        Returns:
            Binary mask of high-texture areas
        """
        from scipy.ndimage import generic_filter

        # Local statistics
        local_mean = generic_filter(tile_norm, np.mean, size=15)
        local_std = generic_filter(tile_norm, np.std, size=15)

        # Coefficient of variation
        with np.errstate(divide='ignore', invalid='ignore'):
            cov = local_std / (local_mean + 1e-6)
        cov = np.nan_to_num(cov, nan=0, posinf=0, neginf=0)

        # High texture areas
        texture_mask = cov > self.min_texture_cov

        # Clean up mask
        texture_mask = morphology.remove_small_objects(texture_mask, max_size=500)
        texture_mask = morphology.closing(texture_mask, morphology.disk(5))

        return texture_mask.astype(np.uint8)

    def apply_highpass_filter(self, tile: np.ndarray) -> np.ndarray:
        """Apply high-pass filter to enhance fine details."""
        from scipy.ndimage import gaussian_filter

        lowpass = gaussian_filter(tile, sigma=self.highpass_size)
        highpass = tile - lowpass

        return highpass

    def preprocess_sar(self, tile: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess SAR tile with speckle filtering and high-pass.

        Args:
            tile: Raw SAR amplitude tile

        Returns:
            Tuple of (despeckled, normalized, high-pass, texture_mask)
        """
        # Mask zeros
        tile = tile.copy()
        tile[tile <= 0] = np.nan

        # Apply speckle filter on linear amplitude
        tile_despeckled = self.apply_speckle_filter(tile)

        # Log scale
        tile_log = np.log10(tile_despeckled + 1e-10)

        # Normalize to 0-1
        valid_data = tile_log[np.isfinite(tile_log)]
        if len(valid_data) > 0:
            p2, p98 = np.percentile(valid_data, [2, 98])
            tile_norm = np.clip(tile_log, p2, p98)
            tile_norm = (tile_norm - p2) / (p98 - p2) if p98 > p2 else np.zeros_like(tile_log)
        else:
            tile_norm = np.zeros_like(tile_log)

        # Fill NaNs
        tile_norm = np.nan_to_num(tile_norm, nan=0)

        # High-pass filter
        tile_highpass = self.apply_highpass_filter(tile_norm)

        # Normalize high-pass
        valid_hp = tile_highpass[np.isfinite(tile_highpass)]
        if len(valid_hp) > 0:
            hp_min, hp_max = np.percentile(valid_hp, [1, 99])
            tile_highpass_norm = np.clip(tile_highpass, hp_min, hp_max)
            if hp_max > hp_min:
                tile_highpass_norm = (tile_highpass_norm - hp_min) / (hp_max - hp_min)
            else:
                tile_highpass_norm = np.zeros_like(tile_highpass)
        else:
            tile_highpass_norm = np.zeros_like(tile_highpass)

        tile_highpass_norm = np.nan_to_num(tile_highpass_norm, nan=0)

        # Compute texture mask
        texture_mask = self.compute_texture_mask(tile_norm)

        return tile_despeckled, tile_norm, tile_highpass_norm, texture_mask

    def detect_ridges(self, tile_norm: np.ndarray, texture_mask: np.ndarray) -> np.ndarray:
        """
        Compute a multiscale curvilinear ridge response, masked to high-texture areas.

        Uses skimage's Frangi vesselness filter across `ridge_sigmas` — this is
        the "squinting effect": larger sigmas respond to coarse, coherent
        curved structure while individual speckle pixels are suppressed,
        without assuming the feature is straight (unlike Canny+Hough).

        Args:
            tile_norm: Normalized tile
            texture_mask: Mask of valid texture areas

        Returns:
            Continuous ridge response array, masked to texture areas
        """
        response = frangi(tile_norm, sigmas=self.ridge_sigmas, black_ridges=self.black_ridges)
        return response * texture_mask

    def clean_ridge_mask(self, ridge_response: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Threshold and clean the ridge response into a binary crevasse-candidate mask.

        Args:
            ridge_response: Continuous ridge response from detect_ridges

        Returns:
            Tuple of (cleaned binary mask, ridge statistics dict)
        """
        if self.ridge_threshold is not None:
            thresh = self.ridge_threshold
        else:
            nonzero = ridge_response[ridge_response > 0]
            if len(nonzero) > 10 and nonzero.std() > 1e-8:
                thresh = np.percentile(nonzero, self.ridge_percentile)
            else:
                # No real structure to threshold against (e.g. fully masked
                # or perfectly flat) -> nothing survives.
                thresh = np.inf

        binary = ridge_response > thresh
        binary = morphology.binary_closing(binary, morphology.disk(2))
        binary = morphology.remove_small_objects(binary, min_size=9)

        cleaned, stats = filter_ridge_candidates(
            binary,
            min_size=self.min_ridge_size,
            min_eccentricity=self.min_eccentricity
        )

        stats['coverage'] = cleaned.sum() / cleaned.size
        stats['mean_response'] = float(ridge_response[cleaned].mean()) if cleaned.any() else 0.0
        stats['total_length'] = float(skeletonize(cleaned).sum()) if cleaned.any() else 0.0

        return cleaned.astype(np.uint8), stats

    def process_tile(self, tile: np.ndarray) -> Dict:
        """
        Complete processing pipeline for one tile.

        Args:
            tile: Raw SAR amplitude tile

        Returns:
            Dictionary with results, all arrays at the original tile resolution
            regardless of ridge_downscale_factor (downscaling is an internal
            detail; upscaling back happens before returning).
        """
        factor = self.ridge_downscale_factor
        if factor > 1:
            from skimage.transform import downscale_local_mean, resize
            valid = tile > 0
            working_tile = downscale_local_mean(tile, (factor, factor))
            valid_small = downscale_local_mean(valid.astype(float), (factor, factor))
            working_tile[valid_small < 0.5] = 0
        else:
            working_tile = tile

        # Preprocess with speckle filtering
        tile_despeckled, tile_norm, tile_highpass, texture_mask = self.preprocess_sar(working_tile)

        # Multiscale ridge detection, masked to texture areas
        ridge_response = self.detect_ridges(tile_norm, texture_mask)
        edges_enhanced, ridge_stats = self.clean_ridge_mask(ridge_response)

        # Statistics
        texture_coverage = texture_mask.sum() / texture_mask.size

        if factor > 1:
            full_shape = tile.shape
            edges_enhanced = (resize(edges_enhanced.astype(float), full_shape, order=0,
                                      preserve_range=True, anti_aliasing=False) > 0.5).astype(np.uint8)
            texture_mask = (resize(texture_mask.astype(float), full_shape, order=0,
                                    preserve_range=True, anti_aliasing=False) > 0.5).astype(np.uint8)
            ridge_response = resize(ridge_response, full_shape, order=1,
                                     preserve_range=True, anti_aliasing=False)
            tile_despeckled = resize(tile_despeckled, full_shape, order=1,
                                      preserve_range=True, anti_aliasing=False)
            tile_norm = resize(tile_norm, full_shape, order=1,
                                preserve_range=True, anti_aliasing=False)
            tile_highpass = resize(tile_highpass, full_shape, order=1,
                                    preserve_range=True, anti_aliasing=False)

        # Tile-level directionality gate: computed on the final (full-res)
        # response so its calibration matches the offline sweep. Below the
        # cut the tile is isotropic speckle, not a crevasse -> empty label.
        orient_conc = self.orientation_concentration(ridge_response)
        if self.orient_gate_thresh is not None and orient_conc < self.orient_gate_thresh:
            edges_enhanced = np.zeros_like(edges_enhanced)
            ridge_stats = {**ridge_stats, 'n_components': 0, 'coverage': 0.0,
                           'mean_response': 0.0, 'total_length': 0.0}
        ridge_stats['orient_conc'] = orient_conc

        return {
            'tile_despeckled': tile_despeckled,
            'tile_norm': tile_norm,
            'tile_highpass': tile_highpass,
            'texture_mask': texture_mask,
            'ridge_response': ridge_response,
            'edges_enhanced': edges_enhanced,
            'n_lines': ridge_stats['n_components'],
            'ridge_coverage': ridge_stats['coverage'],
            'orient_conc': orient_conc,
            'ridge_stats': ridge_stats,
            'texture_coverage': texture_coverage
        }


def visualize_improved_detection(tile: np.ndarray,
                                results: Dict,
                                figsize: Tuple[int, int] = (20, 12)) -> plt.Figure:
    """Visualize improved detection results."""
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    axes = axes.flatten()

    # Original (log scale)
    tile_masked = tile.copy()
    tile_masked[tile_masked <= 0] = np.nan
    tile_log = np.log10(tile_masked + 1e-10)
    axes[0].imshow(tile_log, cmap='gray')
    axes[0].set_title("Original (log scale)")
    axes[0].axis('off')

    # Despeckled
    despeckled_log = np.log10(results['tile_despeckled'] + 1e-10)
    axes[1].imshow(despeckled_log, cmap='gray')
    axes[1].set_title("Despeckled")
    axes[1].axis('off')

    # Normalized
    axes[2].imshow(results['tile_norm'], cmap='gray')
    axes[2].set_title("Normalized")
    axes[2].axis('off')

    # High-pass
    axes[3].imshow(results['tile_highpass'], cmap='gray')
    axes[3].set_title("High-pass Filtered")
    axes[3].axis('off')

    # Texture mask
    axes[4].imshow(results['tile_norm'], cmap='gray', alpha=0.7)
    axes[4].imshow(results['texture_mask'], cmap='Reds', alpha=0.5)
    axes[4].set_title(f"Texture Mask ({results['texture_coverage']*100:.1f}%)")
    axes[4].axis('off')

    # Ridge response heatmap
    axes[5].imshow(results['ridge_response'], cmap='inferno')
    axes[5].set_title("Multiscale Ridge Response")
    axes[5].axis('off')

    # Cleaned ridge mask
    axes[6].imshow(results['edges_enhanced'], cmap='gray')
    axes[6].set_title("Ridge Mask (cleaned)")
    axes[6].axis('off')

    # Ridge mask overlay on normalized tile
    axes[7].imshow(results['tile_norm'], cmap='gray')
    axes[7].imshow(results['edges_enhanced'], cmap='Reds', alpha=0.6)
    stats = results['ridge_stats']
    axes[7].set_title(
        f"Crevasse Candidates ({results['n_lines']})\n"
        f"mean resp: {stats['mean_response']:.3f}, length: {stats['total_length']:.0f}px",
        fontsize=9
    )
    axes[7].axis('off')

    # Texture + ridge mask overlay
    axes[8].imshow(results['tile_norm'], cmap='gray')
    axes[8].imshow(results['texture_mask'], cmap='Reds', alpha=0.3)
    axes[8].imshow(results['edges_enhanced'], cmap='Blues', alpha=0.5)
    axes[8].set_title("Texture + Ridge Mask")
    axes[8].axis('off')

    # Hide unused
    axes[9].axis('off')
    axes[10].axis('off')
    axes[11].axis('off')

    plt.tight_layout()
    return fig


def batch_process_tiles(raster_path: str,
                       detector: ImprovedEdgeCrevasseDetector,
                       n_samples: int = 5,
                       output_dir: Optional[str] = None,
                       min_valid_fraction: float = 0.3) -> List[Dict]:
    """Process multiple tiles with improved detection."""
    tile_gen = TileGenerator(raster_path, tile_size=512)

    print(f"\nSearching for {n_samples} tiles with valid data...")
    sampled_tiles = tile_gen.sample_valid_tiles(
        n_samples=n_samples,
        min_valid_fraction=min_valid_fraction
    )

    results = []

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    for idx, (tile, row, col) in enumerate(sampled_tiles):
        print(f"Processing tile {idx+1}/{n_samples} (r{row}, c{col})...")

        result = detector.process_tile(tile)
        result['row'] = row
        result['col'] = col
        result['tile'] = tile

        print(f"  Ridge candidates: {result['n_lines']}")
        print(f"  Texture coverage: {result['texture_coverage']*100:.1f}%")
        print(f"  Ridge coverage: {result['ridge_coverage']*100:.2f}%")

        fig = visualize_improved_detection(tile, result)

        if output_dir:
            fig.savefig(output_path / f"improved_detection_tile_{idx:03d}_r{row}_c{col}.png",
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

        results.append(result)

    return results
