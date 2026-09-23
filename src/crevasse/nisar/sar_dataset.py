"""
Dataset loaders for SAR tiles and crevasse labels.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

from crevasse.nisar.label_io import load_labels


def _open_tile_generator(*args, **kwargs):
    """Construct a TileGenerator, importing rasterio only at that moment.

    Only the two raster-backed datasets in this module need it; the shard path does not.
    A top-level `from tile_utils_v2 import TileGenerator` made every training run depend
    on a working GDAL stack regardless, which is not hypothetical tidiness: on gattaca,
    conda-forge's libgdal.so.38 needs GCC_12.0.0 symbols that the system
    /lib64/libgcc_s.so.1 does not export and no available gcc module provides, so
    `import rasterio` raises there while shard training is otherwise fine.
    """
    try:
        from crevasse.common.tile_utils_v2 import TileGenerator
    except ImportError as e:
        raise ImportError(
            f"opening a GeoTIFF needs rasterio, which failed to import: {e}\n"
            "Training and eval from a .npz shard need neither -- pass --shard."
        ) from e
    return TileGenerator(*args, **kwargs)


def preprocess_sar_tile(tile: np.ndarray) -> np.ndarray:
    """Raw amplitude -> [0, 1]: mask zeros, log10, per-tile p2/p98 contrast stretch.

    Single definition for every dataset in this module. NOTE the per-tile stretch is
    load-bearing and slightly perverse: it amplifies residual speckle hardest on the
    emptiest tiles, which is exactly why the pseudo-label needs its orientation factor
    (see soft_labels.py). Changing it invalidates the label calibration.
    """
    tile = tile.copy()
    tile[tile <= 0] = np.nan
    tile_log = np.log10(tile + 1e-10)

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

    return np.nan_to_num(tile_norm, nan=0).astype(np.float32)


class SARTileDataset(Dataset):
    """
    Dataset for loading SAR tiles.
    For self-supervised pretraining (no labels needed).
    """

    def __init__(self,
                 raster_path: str,
                 tile_positions: List[Tuple[int, int]],
                 tile_size: int = 512,
                 transform=None,
                 normalize: bool = True):
        """
        Args:
            raster_path: Path to NISAR raster
            tile_positions: List of (row, col) positions
            tile_size: Size of tiles to load
            transform: Albumentations transform
            normalize: Whether to normalize tiles to [0, 1]
        """
        self.raster_path = raster_path
        self.tile_positions = tile_positions
        self.tile_size = tile_size
        self.transform = transform
        self.normalize = normalize

        self.tile_gen = _open_tile_generator(raster_path, tile_size=tile_size)

    def __len__(self):
        return len(self.tile_positions)

    def _preprocess_tile(self, tile: np.ndarray) -> np.ndarray:
        return preprocess_sar_tile(tile)

    def __getitem__(self, idx):
        row, col = self.tile_positions[idx]

        # Load tile
        tile, _ = self.tile_gen.read_tile(row, col)

        # Preprocess
        if self.normalize:
            tile = self._preprocess_tile(tile)

        # Add channel dimension
        tile = tile[np.newaxis, ...]  # [1, H, W]

        # Apply transforms
        if self.transform:
            # Albumentations expects HWC format
            tile_hwc = np.transpose(tile, (1, 2, 0))  # [H, W, 1]
            transformed = self.transform(image=tile_hwc)
            tile = transformed['image']
        else:
            tile = torch.from_numpy(tile)

        return {
            'image': tile,
            'position': (row, col)
        }


class SARCrevasseDataset(Dataset):
    """
    Dataset for supervised training with crevasse labels.
    """

    def __init__(self,
                 raster_path: str,
                 label_data: List[dict],
                 tile_size: int = 512,
                 transform=None,
                 normalize: bool = True):
        """
        Args:
            raster_path: Path to NISAR raster
            label_data: List of dicts with 'row', 'col', 'edge_mask' (ridge mask, from detection results)
            tile_size: Size of tiles
            transform: Albumentations transform
            normalize: Whether to normalize
        """
        self.raster_path = raster_path
        self.label_data = label_data
        self.tile_size = tile_size
        self.transform = transform
        self.normalize = normalize

        self.tile_gen = _open_tile_generator(raster_path, tile_size=tile_size)

    def __len__(self):
        return len(self.label_data)

    def _preprocess_tile(self, tile: np.ndarray) -> np.ndarray:
        return preprocess_sar_tile(tile)

    def _lines_to_mask(self, lines: List[Tuple], tile_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert line coordinates to binary segmentation mask.

        Args:
            lines: List of ((x0,y0), (x1,y1)) line coordinates
            tile_shape: (H, W) shape of tile

        Returns:
            Binary mask [H, W]
        """
        import cv2

        mask = np.zeros(tile_shape, dtype=np.uint8)

        for line in lines:
            (x0, y0), (x1, y1) = line
            # Convert to integer coordinates
            pt1 = (int(x0), int(y0))
            pt2 = (int(x1), int(y1))
            # Draw line with thickness
            cv2.line(mask, pt1, pt2, color=1, thickness=3)

        return mask.astype(np.float32)

    def __getitem__(self, idx):
        label_info = self.label_data[idx]
        row = label_info['row']
        col = label_info['col']

        # Load tile
        tile, _ = self.tile_gen.read_tile(row, col)

        # Preprocess
        if self.normalize:
            tile = self._preprocess_tile(tile)

        # Get mask - either from edge_mask (new) or lines (old)
        if 'edge_mask' in label_info:
            mask = label_info['edge_mask'].astype(np.float32)
            n_lines = label_info.get('n_lines', 0)
        else:
            # Fallback to old line-based labels
            lines = label_info['lines']
            mask = self._lines_to_mask(lines, tile.shape)
            n_lines = len(lines)

        # Apply transforms (to both image and mask)
        if self.transform:
            transformed = self.transform(image=tile, mask=mask)
            tile = transformed['image']
            mask = transformed['mask']
            # Ensure mask has channel dimension [1, H, W]
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            tile = torch.from_numpy(tile).unsqueeze(0)  # Add channel
            mask = torch.from_numpy(mask).unsqueeze(0)

        return {
            'image': tile,
            'mask': mask,
            'position': (row, col),
            'n_lines': n_lines
        }


def get_train_transforms(tile_size: int = 512):
    """
    Training augmentations for SAR data.
    """
    return A.Compose([
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            border_mode=0,
            p=0.5
        ),

        # Intensity (careful with SAR!)
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3
        ),
        # std_range, NOT var_limit: albumentations 2.x renamed it and silently
        # DROPS the unknown kwarg, falling back to its default std_range=(0.2,0.44).
        # On these [0,1]-normalized tiles that is noise at up to half the dynamic
        # range, which buries the speckle texture the model has to read. These
        # values are the old var_limit (0.001, 0.01) expressed as std.
        A.GaussNoise(std_range=(0.032, 0.10), p=0.3),

        # Convert to tensor
        ToTensorV2()
    ])


def get_val_transforms():
    """Validation transforms (no augmentation)."""
    return A.Compose([
        ToTensorV2()
    ])


def create_dataloaders(raster_path: str,
                       tile_positions: List[Tuple[int, int]],
                       tile_size: int = 512,
                       batch_size: int = 4,
                       train_split: float = 0.8,
                       num_workers: int = 4):
    """
    Create train/val dataloaders for self-supervised pretraining.

    Args:
        raster_path: Path to NISAR raster
        tile_positions: List of valid tile positions
        tile_size: Tile size
        batch_size: Batch size
        train_split: Fraction for training
        num_workers: Number of dataloader workers

    Returns:
        train_loader, val_loader
    """
    # Split positions
    n_train = int(len(tile_positions) * train_split)
    train_positions = tile_positions[:n_train]
    val_positions = tile_positions[n_train:]

    print(f"Train tiles: {len(train_positions)}, Val tiles: {len(val_positions)}")

    # Create datasets
    train_dataset = SARTileDataset(
        raster_path,
        train_positions,
        tile_size=tile_size,
        transform=get_train_transforms(tile_size),
        normalize=True
    )

    val_dataset = SARTileDataset(
        raster_path,
        val_positions,
        tile_size=tile_size,
        transform=get_val_transforms(),
        normalize=True
    )

    # Create dataloaders
    # Disable pin_memory on MPS (Apple Silicon)
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    return train_loader, val_loader


def _tile_label_coverage(r):
    """Fraction of the tile that is labeled crevasse, robust across pkl formats.

    Prefers the stored edge_coverage (final dilated mask density), falls back to
    ridge_coverage, then computes from edge_mask, and returns None when a tile
    carries no coverage info (old line-based labels) so it isn't filtered out.
    """
    if 'edge_coverage' in r:
        return float(r['edge_coverage'])
    if 'ridge_coverage' in r:
        return float(r['ridge_coverage'])
    if 'edge_mask' in r:
        m = r['edge_mask']
        return float((m > 0).sum()) / m.size
    return None


def pool_mask(positions: np.ndarray,
              coverage: np.ndarray,
              min_coverage: float = 0.0,
              context_npz: str = None,
              min_grounded: float = 0.0):
    """Which tiles enter the experiment at all, before any train/val split.

    Factored out so training and eval_shard.py cannot disagree about the pool -- if they
    do, `--subset val` silently scores a different set than the checkpoint's val_iou.

    `min_grounded` gates on Bedmap3 grounded fraction from a build_tile_context.py
    sidecar. It exists because the 2026-09-10 runs trained and scored on a pool that was
    32% floating shelf / sea ice; rifts there are a different process from grounded-ice
    crevasses and the Frangi label rule was never validated on them.
    """
    keep = coverage >= min_coverage
    if min_grounded > 0.0:
        if context_npz is None:
            raise ValueError(f"min_grounded={min_grounded} needs --context-npz "
                             f"(build it with build_tile_context.py)")
        c = np.load(context_npz)
        cpos = c["positions"].astype(np.int64)
        if cpos.shape != positions.shape or not (cpos == positions).all():
            raise ValueError(
                f"{Path(context_npz).name} was built for a different tile set "
                f"({len(cpos)} tiles vs {len(positions)}); rebuild it against this shard "
                f"or the grounded filter would be applied to the wrong tiles.")
        keep &= c["grounded_frac"] >= min_grounded
        print(f"  grounded filter: {int((c['grounded_frac'] >= min_grounded).sum())}"
              f"/{len(positions)} tiles at grounded_frac >= {min_grounded}")
    if not keep.any():
        raise ValueError(
            f"No tiles passed min_coverage={min_coverage} / min_grounded={min_grounded} "
            f"out of {len(coverage)}. Observed coverage max={coverage.max():.4f}.")
    return keep


def block_strat_values(context_npz: str, keep: np.ndarray):
    """log10 surface speed per kept tile, for balancing a "blocks" split.

    log because speed spans 5-4000 m/yr on this footprint, so a linear mean would let one
    ice-stream block dominate a block's difficulty score.
    """
    if context_npz is None:
        return None
    v = np.load(context_npz)["vel_mean"][keep]
    return np.log10(np.maximum(v, 1.0))


def split_indices(positions: np.ndarray,
                  train_split: float,
                  tile_size: int,
                  split_mode: str = "spatial",
                  split_buffer_tiles: int = 1,
                  seed: int = 42,
                  block_tiles: int = 3,
                  strat_values: np.ndarray = None):
    """Train/val index split over tile (row, col) positions.

    Factored out so the raster-backed and shard-backed loaders cannot drift apart.
    The spatial band logic is subtle: the original code sliced label_data[:n] on a
    raster-scan-ordered list, which made val a spatially-adjacent leaky slice.

    `strat_values` is a per-tile difficulty proxy used only by "blocks" mode, to balance
    it across train and val. It must be independent of the label rule, or the split
    itself would encode the thing under test -- ITS_LIVE surface speed is what this
    project uses (r = +0.45 with both rules' coverage, and it is not derived from the
    imagery at all).
    """
    n_train = int(len(positions) * train_split)

    if split_mode == "random":
        perm = np.random.default_rng(seed).permutation(len(positions))
        train_idx, val_idx = perm[:n_train], perm[n_train:]
        print(f"Split: random (seed={seed}) -> train {len(train_idx)}, val {len(val_idx)}")
    elif split_mode == "spatial":
        rows, cols = positions[:, 0], positions[:, 1]
        axis = 0 if (rows.max() - rows.min()) >= (cols.max() - cols.min()) else 1
        coord = positions[:, axis]
        order = np.argsort(coord, kind="stable")
        train_idx, val_idx = order[:n_train], order[n_train:]
        dropped = 0
        if len(val_idx) > 0 and split_buffer_tiles > 0:
            val_edge = coord[val_idx].min()
            gap = split_buffer_tiles * tile_size
            keep = coord[train_idx] < (val_edge - gap)
            dropped = int((~keep).sum())
            train_idx = train_idx[keep]
        axis_name = "row" if axis == 0 else "col"
        print(f"Split: spatial band on {axis_name} axis -> train {len(train_idx)}, "
              f"val {len(val_idx)} (buffer dropped {dropped} train tiles)")
        if len(train_idx):
            print(f"  train {axis_name} range: "
                  f"[{coord[train_idx].min()}, {coord[train_idx].max()}]")
        if len(val_idx):
            print(f"  val   {axis_name} range: "
                  f"[{coord[val_idx].min()}, {coord[val_idx].max()}]")
    elif split_mode == "blocks":
        # A single contiguous band cannot be both spatially separated and
        # difficulty-matched here: on 025_019 the fast crevassed grounded ice is
        # clustered, so every band split lands val on slow interior ice (f4 coverage
        # 2.0% vs train 8.1%) -- and IoU on this data is largely a proxy for
        # orient_conc (r = +0.84), so that mismatch IS the score. Holding out whole
        # blocks samples the same difficulty mix from disjoint geography.
        rows, cols = positions[:, 0], positions[:, 1]
        span = block_tiles * tile_size
        block_id = (rows // span) * (2 + cols.max() // span) + (cols // span)
        blocks = np.unique(block_id)
        if strat_values is None:
            # Unstratified is a trap here, kept only so the mode works without a
            # stratifier: with ~40 blocks and 8 held out, WHICH 8 swings train/val
            # coverage from 5.2%/9.3% to 6.3%/7.0% on 025_019. Picking the seed that
            # looks best is fitting the split to the metric.
            print("  WARNING: blocks mode without strat_values -- train/val difficulty "
                  "is left to the seed, and on this data that swings the val IoU more "
                  "than the label rule does")
            order = np.random.default_rng(seed).permutation(blocks)
        else:
            # Systematic sampling down a difficulty ordering: every k-th block is held
            # out, so the val blocks span the whole range of the stratifier by
            # construction and `seed` only chooses which of the k folds you get.
            key = np.array([np.mean(strat_values[block_id == b]) for b in blocks])
            order = blocks[np.argsort(key, kind="stable")]
            k = max(int(round(1.0 / max(1.0 - train_split, 1e-9))), 2)
            order = np.concatenate([order[seed % k::k], order[np.arange(len(order))
                                                             % k != seed % k]])
        val_blocks, held = [], 0
        for b in order:
            if held >= (1.0 - train_split) * len(positions):
                break
            val_blocks.append(b)
            held += int((block_id == b).sum())
        is_val = np.isin(block_id, val_blocks)
        val_idx = np.flatnonzero(is_val)
        cand = np.flatnonzero(~is_val)
        # Chebyshev buffer in tile units, against every val tile -- a band's
        # one-sided "before the edge" test does not generalize to blocks, which have
        # a train neighbour on all four sides.
        if split_buffer_tiles > 0 and len(val_idx):
            gap = split_buffer_tiles * tile_size
            vr, vc = rows[val_idx], cols[val_idx]
            near = np.array([np.any((np.abs(vr - rows[i]) <= gap)
                                    & (np.abs(vc - cols[i]) <= gap)) for i in cand])
            train_idx, dropped = cand[~near], int(near.sum())
        else:
            train_idx, dropped = cand, 0
        print(f"Split: {block_tiles}x{block_tiles}-tile blocks "
              f"({block_tiles * tile_size * 5 / 1000:.0f} km at 5 m px), "
              f"{len(val_blocks)} of {len(np.unique(block_id))} blocks held out "
              f"-> train {len(train_idx)}, val {len(val_idx)} "
              f"(buffer dropped {dropped} train tiles)")
    else:
        raise ValueError(f"unknown split_mode {split_mode!r} "
                         f"(expected 'spatial', 'blocks' or 'random')")

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"Split produced an empty set (train={len(train_idx)}, val={len(val_idx)}) "
            f"from {len(positions)} tiles with train_split={train_split}, "
            f"split_mode={split_mode!r}. Adjust train_split or split_buffer_tiles.")

    return train_idx, val_idx


class ShardCrevasseDataset(Dataset):
    """Supervised dataset backed by a packed .npz shard instead of a GeoTIFF.

    The shard stores the already-read amplitude tile next to its target, so training
    on a cluster needs neither the 5-10 GB granule nor rasterio. `images` are RAW
    amplitude and preprocessing stays here, identical to the raster path, so the two
    loaders are interchangeable. Both arrays are float16 on disk, promoted to float32
    per item.
    """

    def __init__(self, images, targets, positions, transform=None, normalize: bool = True):
        self.images = images
        self.targets = targets
        self.positions = positions
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        tile = np.asarray(self.images[idx], dtype=np.float32)
        mask = np.asarray(self.targets[idx], dtype=np.float32)

        if self.normalize:
            tile = preprocess_sar_tile(tile)

        if self.transform:
            transformed = self.transform(image=tile, mask=mask)
            tile = transformed['image']
            mask = transformed['mask']
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            tile = torch.from_numpy(tile).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0)

        row, col = self.positions[idx]
        return {'image': tile, 'mask': mask, 'position': (int(row), int(col))}


def create_shard_dataloaders(shard_path: str,
                             batch_size: int = 4,
                             train_split: float = 0.8,
                             min_coverage: float = 0.0,
                             split_mode: str = "spatial",
                             split_buffer_tiles: int = 1,
                             seed: int = 42,
                             num_workers: int = 4,
                             context_npz: str = None,
                             min_grounded: float = 0.0,
                             block_tiles: int = 3):
    """Train/val loaders from a shard built by build_shard.py.

    Same split semantics as create_supervised_dataloaders (both call split_indices).
    """
    z = np.load(shard_path)
    images, targets = z['images'], z['targets']
    positions = z['positions'].astype(np.int64)
    coverage = z['coverage']
    tile_size = int(z['tile_size'])

    keep = pool_mask(positions, coverage, min_coverage, context_npz, min_grounded)
    images, targets, positions = images[keep], targets[keep], positions[keep]
    print(f"Shard {Path(shard_path).name}: {int(keep.sum())}/{len(coverage)} tiles "
          f"(coverage>={min_coverage}, grounded>={min_grounded}), tile_size {tile_size}")

    train_idx, val_idx = split_indices(positions, train_split, tile_size, split_mode,
                                       split_buffer_tiles, seed, block_tiles,
                                       block_strat_values(context_npz, keep))

    train_dataset = ShardCrevasseDataset(
        images[train_idx], targets[train_idx], positions[train_idx],
        transform=get_train_transforms(tile_size), normalize=True)
    val_dataset = ShardCrevasseDataset(
        images[val_idx], targets[val_idx], positions[val_idx],
        transform=get_val_transforms(), normalize=True)

    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=use_pin_memory)
    return train_loader, val_loader


def create_supervised_dataloaders(raster_path: str,
                                  results_npz: str,
                                  tile_size: int = 512,
                                  batch_size: int = 4,
                                  train_split: float = 0.8,
                                  min_lines: int = 0,
                                  min_coverage: float = 0.0,
                                  split_mode: str = "spatial",
                                  split_buffer_tiles: int = 1,
                                  seed: int = 42,
                                  num_workers: int = 4):
    """
    Create dataloaders for supervised training with pseudo-labels.

    Args:
        raster_path: Path to NISAR raster
        results_npz: Path to detection results (label_io.py's npz format)
        tile_size: Tile size
        batch_size: Batch size
        train_split: Train/val split
        min_lines: Minimum ridge connected-component count per tile. Default 0
            (off): the label file is already a curated positive set (label generation
            drops n_lines==0 and applies the RF/orientation gates), and component
            count is not a meaningful "how much crevasse" signal after the
            Frangi/downscale redesign (real fields yield only 2-8 merged
            components, so the old default of 10 silently emptied the dataset).
        min_coverage: Minimum labeled-area fraction per tile (edge_coverage).
            The meaningful density knob; default 0.0 keeps every curated tile.
        split_mode: How to split train/val.
            "spatial" (default) holds out a contiguous geographic band along the
            tile grid's longer axis, with a buffer gap so train and val regions
            don't touch — a real generalization check for the continent-wide end
            goal. The old code sliced label_data[:n_train] directly; since label
            files are in raster-scan order that made val a spatially-adjacent slice
            (leaky, over-optimistic). "random" does a seeded shuffle instead
            (mixes regions — use only if you don't need spatial generalization).
        split_buffer_tiles: In "spatial" mode, drop train tiles within this many
            tiles of the val band boundary so the two sets are physically
            separated. 0 disables the buffer.
        seed: RNG seed for "random" split_mode.
        num_workers: Number of workers

    Returns:
        train_loader, val_loader
    """
    # Load detection results
    results = load_labels(results_npz)

    # Filter by component count AND labeled-area fraction. Coverage is the
    # meaningful signal; min_lines is kept only for backward compatibility.
    def keep(r):
        cov = _tile_label_coverage(r)
        return (r.get('n_lines', 0) >= min_lines
                and (cov is None or cov >= min_coverage))

    label_data = [r for r in results if keep(r)]
    print(f"Tiles kept (n_lines>={min_lines}, coverage>={min_coverage}): "
          f"{len(label_data)}/{len(results)}")

    if len(label_data) == 0:
        covs = [c for c in (_tile_label_coverage(r) for r in results) if c is not None]
        nls = [r.get('n_lines', 0) for r in results]
        raise ValueError(
            f"No tiles passed the label filter (min_lines={min_lines}, "
            f"min_coverage={min_coverage}) out of {len(results)} pseudo-labels. "
            f"This would train on an empty dataset. "
            f"Observed n_lines max={max(nls) if nls else 0}, "
            f"coverage max={max(covs) if covs else 0:.4f}. "
            f"Lower --min-lines/--min-coverage.")

    # Split train/val. See split_mode docstring for why the old raster-scan slice
    # was leaky.
    positions = np.array([(r['row'], r['col']) for r in label_data], dtype=np.int64)
    train_idx, val_idx = split_indices(positions, train_split, tile_size,
                                       split_mode, split_buffer_tiles, seed)

    train_data = [label_data[i] for i in train_idx]
    val_data = [label_data[i] for i in val_idx]

    # Create datasets
    train_dataset = SARCrevasseDataset(
        raster_path,
        train_data,
        tile_size=tile_size,
        transform=get_train_transforms(tile_size),
        normalize=True
    )

    val_dataset = SARCrevasseDataset(
        raster_path,
        val_data,
        tile_size=tile_size,
        transform=get_val_transforms(),
        normalize=True
    )

    # Create dataloaders
    # Disable pin_memory on MPS (Apple Silicon)
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    return train_loader, val_loader
