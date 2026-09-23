"""Dataset, split and normalization for the BIOMASS crevasse U-Net.

Deliberately NOT a straight port of nisar-crevasse-unet/src/sar_dataset.py in two places:

1. **Normalization.** NISAR's `preprocess_sar_tile` is a per-tile log/p2/p98 stretch --
   load-bearing there because the Frangi label calibration depends on it. BIOMASS's
   target is a real label, not a manufactured response, so nothing depends on that
   stretch, and a per-tile OR per-channel stretch would actively destroy the cross/co
   offset that is the one thing this radar carries (see `bio_tile_cache.py`'s own
   docstring). Instead: ONE shared mean/sd over all 4 channels, fit once on the training
   split, applied identically to train and val -- the exact scheme already validated for
   the CNN gate (`biomass-crevasse-gate/src/bio_cnn_gate.py:107-117`).
2. **Spatial split.** NISAR's "blocks" mode uses a grounded-only, 3x3-tile scheme built
   to fix a grounding-line confound specific to Thwaites/NISAR's validation band. BIOMASS
   already has its own validated separation -- 32-tile (82 km) blocks with a 16-tile
   (41 km) buffer is the setting at which the radar features clearly beat map coordinates
   (`bio_gate_controls.py`: model 0.889 vs (x,y) 0.601 buffered). `buffered_split` below
   reuses that geometry, adapted from a k-fold generator (`buffered_folds`) to a single
   train/val split.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

POLS = ["HH", "HV", "VH", "VV"]
BLOCK_TILES = 32   # 82 km at 512 px / 5 m -- biomass-crevasse-gate's validated setting
BUFFER_TILES = 16  # 41 km


def fit_normalization(images, idx, channel_idx):
    """One shared mean/sd over the given channels, fit on `images[idx]` only.

    `dtype=np.float64` is load-bearing: images are float16 and numpy reduces in the
    input dtype by default, which overflows on ~1e8 values and silently yields sd=nan
    (an all-zero input) -- the exact bug `bio_cnn_gate.py:110` documents and guards.
    """
    sub = np.asarray(images[idx])[:, channel_idx].astype(np.float64)
    mu = float(np.nanmean(sub))
    sd = float(np.nanstd(sub)) or 1.0
    assert np.isfinite(mu) and np.isfinite(sd) and sd > 0, \
        f"bad normalization mu={mu} sd={sd}"
    return mu, sd


def normalize_stack(stack_chw, mu, sd):
    """(C, H, W) raw dB, NaN-nodata -> zero-mean/unit-sd, NaN -> 0."""
    x = (np.asarray(stack_chw, dtype=np.float32) - mu) / sd
    return np.nan_to_num(x, nan=0.0).astype(np.float32)


def buffered_split(x_m, y_m, val_frac=0.2, block_tiles=BLOCK_TILES,
                    buffer_tiles=BUFFER_TILES, tile_size=512, px_m=5.0, seed=42):
    """One train/val split on BIOMASS's own validated block+buffer geometry.

    Unlike `bio_gate_controls.buffered_folds` (a k-fold generator for cross-validation),
    this returns ONE split: blocks are visited in a seeded random order and added to val
    until `val_frac` of tiles is covered, then every remaining tile within `buffer_tiles`
    (in block-edge units, i.e. `buffer_tiles * tile_size * px_m` metres) of any val
    block's extent is dropped from train -- mirroring `buffered_folds`'s own "grow the
    test block by the buffer" rule, applied once instead of per fold.
    """
    block_m = block_tiles * tile_size * px_m
    buffer_m = buffer_tiles * tile_size * px_m
    bx = np.floor(np.asarray(x_m) / block_m).astype(int)
    by = np.floor(np.asarray(y_m) / block_m).astype(int)
    blocks = sorted(set(zip(bx.tolist(), by.tolist())))
    rng = np.random.default_rng(seed)
    order = [blocks[i] for i in rng.permutation(len(blocks))]

    n = len(x_m)
    val_mask = np.zeros(n, dtype=bool)
    val_blocks = []
    target_n = int(round(val_frac * n))
    for (a, b) in order:
        if val_mask.sum() >= target_n:
            break
        m = (bx == a) & (by == b)
        val_mask |= m
        val_blocks.append((a, b))

    near = np.zeros(n, dtype=bool)
    for (a, b) in val_blocks:
        x0, x1 = a * block_m - buffer_m, (a + 1) * block_m + buffer_m
        y0, y1 = b * block_m - buffer_m, (b + 1) * block_m + buffer_m
        near |= (x_m >= x0) & (x_m <= x1) & (y_m >= y0) & (y_m <= y1)

    val_idx = np.flatnonzero(val_mask)
    train_idx = np.flatnonzero(~val_mask & ~near)
    dropped = int((near & ~val_mask).sum())

    print(f"Split: {block_tiles}-tile blocks ({block_m/1000:.0f} km) / "
          f"{buffer_tiles}-tile buffer ({buffer_m/1000:.0f} km), "
          f"{len(val_blocks)}/{len(blocks)} blocks held out "
          f"-> train {len(train_idx)}, val {len(val_idx)} (buffer dropped {dropped})")
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"Split produced an empty set (train={len(train_idx)}, val={len(val_idx)}) "
            f"from {n} tiles. Adjust val_frac/block_tiles/buffer_tiles.")
    return train_idx, val_idx


def get_train_transforms():
    """Training augmentations. Applied AFTER normalization (see ShardMultiPolDataset),
    so images entering here are ~zero-mean/unit-sd, not [0,1] like NISAR's -- the noise
    and brightness/contrast magnitudes are scaled for that, not copied from sar_dataset.py.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45,
                            border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([ToTensorV2()])


class ShardMultiPolDataset(Dataset):
    """4-polarization dB tiles + AlphaEarth pixel mask, backed by a packed shard.

    `images` is `(N, 4, H, W)` raw dB (float16 on disk), `targets` is `(N, H, W)` uint8
    0/1. `channel_idx` selects a subset of the 4 stored channels -- the mechanism behind
    `--channels` in bio_train_unet_supervised.py, so a later pivot to fewer channels is a
    training flag, not a shard rebuild. `mu`/`sd` must be fit ONCE (fit_normalization, on
    the train split only) and passed to both the train and val dataset so they share one
    normalization, then shipped in the checkpoint for inference.
    """

    def __init__(self, images, targets, positions, channel_idx, mu, sd, transform=None):
        self.images = images
        self.targets = targets
        self.positions = positions
        self.channel_idx = list(channel_idx)
        self.mu = mu
        self.sd = sd
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        stack = np.asarray(self.images[idx])[self.channel_idx]
        stack = normalize_stack(stack, self.mu, self.sd)          # (C, H, W)
        mask = np.asarray(self.targets[idx], dtype=np.float32)    # (H, W)

        if self.transform:
            hwc = np.transpose(stack, (1, 2, 0))
            transformed = self.transform(image=hwc, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        else:
            image = torch.from_numpy(stack)
            mask = torch.from_numpy(mask).unsqueeze(0)

        row, col = self.positions[idx]
        return {"image": image, "mask": mask, "position": (int(row), int(col))}


def load_shard(shard_path):
    z = np.load(shard_path, allow_pickle=True)
    # Soft (continuous [0,1]) Frangi targets from bio_build_shard_frangi.py carry these
    # keys; the real AlphaEarth binary shard from bio_build_shard.py does not. Consumers
    # use this to pick a sane default target-binarization threshold: 0.5 is right for a
    # genuinely binary {0,1} mask, but wrong for a soft target -- see bio_eval_shard.py.
    is_soft = "orient_conc" in z.files and "ridge_factors" in z.files
    return dict(
        images=z["images"], targets=z["targets"],
        positions=z["positions"].astype(np.int64),
        x_m=z["x_m"], y_m=z["y_m"], coverage=z["coverage"], aoi_pos=z["aoi_pos"],
        granule=z["granule"], pols=[str(p) for p in z["pols"]],
        tile_size=int(z["tile_size"]), is_soft=is_soft,
    )


def channel_indices(pols, channels):
    """`channels` (e.g. ["HH","HV","VH","VV"] or a subset) -> indices into `pols`."""
    bad = [c for c in channels if c not in pols]
    if bad:
        raise ValueError(f"unknown channel(s) {bad}; shard carries {pols}")
    return [pols.index(c) for c in channels]


def create_shard_dataloaders(shard_path, channels=None, batch_size=8, val_frac=0.2,
                              block_tiles=BLOCK_TILES, buffer_tiles=BUFFER_TILES,
                              seed=42, num_workers=4, mu=None, sd=None):
    """Train/val loaders from a shard built by bio_build_shard.py or
    bio_build_shard_frangi.py.

    Returns (train_loader, val_loader, mu, sd, channel_idx, is_soft) -- `mu`/`sd` are
    returned so the caller can ship them in the checkpoint; pass them back in (e.g. from a
    checkpoint) to reproduce the exact val split's normalization without re-fitting, which
    is what bio_eval_shard.py's `--subset val` control needs. `is_soft` says whether the
    shard's target is continuous (Frangi) rather than binary (AlphaEarth) -- see
    `load_shard`.
    """
    d = load_shard(shard_path)
    channels = channels or POLS
    cidx = channel_indices(d["pols"], channels)
    print(f"Shard {shard_path}: {len(d['images'])} tiles, channels {channels}")

    train_idx, val_idx = buffered_split(d["x_m"], d["y_m"], val_frac, block_tiles,
                                        buffer_tiles, d["tile_size"], seed=seed)

    if mu is None or sd is None:
        mu, sd = fit_normalization(d["images"], train_idx, cidx)
    print(f"Normalization: mu={mu:.3f} sd={sd:.3f} (dB, fit on train split only)")

    train_ds = ShardMultiPolDataset(
        d["images"][train_idx], d["targets"][train_idx], d["positions"][train_idx],
        cidx, mu, sd, transform=get_train_transforms())
    val_ds = ShardMultiPolDataset(
        d["images"][val_idx], d["targets"][val_idx], d["positions"][val_idx],
        cidx, mu, sd, transform=get_val_transforms())

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, mu, sd, cidx, d["is_soft"]
