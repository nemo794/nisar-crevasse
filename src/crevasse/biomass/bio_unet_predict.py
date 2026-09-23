"""Array-in Python API for the BIOMASS crevasse U-Net: N tiles in, N probability masks out.

    from crevasse.biomass.bio_unet_predict import BiomassUNetSegmenter
    seg = BiomassUNetSegmenter()               # models/bio_unet_4ch_frangi_aux_nw0p02/unet_best
    p = seg.predict_proba(stack)              # (N, C, 512, 512) dB -> (N, 512, 512) float32
    mask = seg.predict(stack, thresh=0.5)      # -> (N, 512, 512) bool

Mirrors `nisar-crevasse-unet/src/unet_predict.py`'s shape. The one BIOMASS-specific
difference: the checkpoint ships its own `channels` (which of HH/HV/VH/VV it was trained
on, and in what order) and `mu`/`sd` (the one shared-across-channels normalization, fit
once on the training split -- see `bio_sar_dataset.py`). Both travel with the weights and
are applied automatically; the caller does not supply or re-derive them, the same way
`bio_cnn_gate.py`'s `score_with_net` insists on using the shipped `mu`/`sd` rather than
whatever the scoring batch happens to look like.

Input contract
---------------
* `stack` is raw **dB** (what `bio_build_shard.py` / `bio_tile_cache.granule_chips`
  produce: `10*log10(linear intensity)`), NOT linear intensity, shape
  `(N, C, 512, 512)` or `(C, 512, 512)`, where `C == len(checkpoint["channels"])` and the
  channel order must match `checkpoint["channels"]` (default `["HH","HV","VH","VV"]`).
* NaN means nodata (the SAR's own nodata code, propagated through the dB conversion).
  There is no 0-as-nodata convention here, unlike NISAR's amplitude -- 0 dB is a normal,
  finite value.

Output contract
-----------------
N in, N out, same order. Like `UNetSegmenter`, every tile gets a real prediction -- there
is no per-tile NaN path, because there is no dynamic-range degeneracy check on this path.

Optional aux noise branch
---------------------------
A checkpoint trained with `--aux-noise-branch` (see `bio_train_unet_supervised.py`)
carries a second decoder that reconstructs the raw input tile -- meant to capture the
data's documented PSF/speckle confound, since a bottlenecked autoencoder's cheapest way
to minimize reconstruction error is to reproduce the dominant repeating structure.
`predict_noise` exposes that reconstruction; `self.aux_decoder` says whether a given
checkpoint has one. Calling `predict_noise` on a checkpoint without one raises rather
than silently returning nothing.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from crevasse.common.unet_model import UNet
from crevasse.common.ckpt_io import load_checkpoint
from crevasse.biomass.bio_sar_dataset import POLS, normalize_stack

ROOT = Path(__file__).resolve().parents[3]
# 2026-09-21 noise-weight sweep on the Frangi soft-label shard (aux-noise-branch, see
# unet_model.py / bio_train_unet_supervised.py --aux-noise-branch): nw=0.02 chosen over
# nw=0.15/0.17 (higher pooled IoU, 0.342/0.301 vs 0.308) because it is the most
# conservative of the three -- highest precision (0.484 vs 0.446/0.374) and the
# closest in behavior to the plain standard model in side-by-side comparison, at a time
# when whether the others' extra recall is real crevasse signal or PSF-confound-chasing
# is still an open question (see docs/PIPELINE.md). Provisional, not a claim of
# finality -- pass `checkpoint=` to use any other run.
DEFAULT_CHECKPOINT = ROOT / "models" / "biomass" / "unet" / "bio_unet_4ch_frangi_aux_nw0p02" / "unet_best"


class BiomassUNetSegmenter:
    """The trained BIOMASS U-Net, scoring in-memory 4-pol dB tiles."""

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device=None, tile_size=512):
        self.device = (torch.device(device) if device is not None else
                       torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu"))
        ck = load_checkpoint(str(checkpoint), device=self.device)
        train_args = ck.get("args", {}) or {}
        self.channels = ck.get("channels") or list(
            (train_args.get("channels") or ",".join(POLS)).split(","))
        self.mu, self.sd = float(ck["mu"]), float(ck["sd"])
        base_channels = int(train_args.get("base_channels", 64))
        self.aux_decoder = bool(ck.get("aux_decoder", False))

        model = UNet(n_channels=len(self.channels), n_classes=1,
                    base_channels=base_channels, aux_decoder=self.aux_decoder)
        model.load_state_dict(ck["model_state_dict"])
        model.to(self.device)
        model.eval()

        self._model = model
        self.checkpoint = str(checkpoint)
        self.tile_size = int(tile_size)
        self.epoch = ck.get("epoch")
        self.val_iou = ck.get("val_iou")
        self.val_loss = ck.get("val_loss")

    # -- internals ---------------------------------------------------------------

    def _as_batch(self, stack):
        a = np.asarray(stack)
        if a.ndim == 3:
            a = a[None]
        if a.ndim != 4:
            raise ValueError(
                f"stack must be (N, {len(self.channels)}, {self.tile_size}, "
                f"{self.tile_size}) or ({len(self.channels)}, {self.tile_size}, "
                f"{self.tile_size}); got shape {a.shape}")
        if a.shape[1] != len(self.channels):
            raise ValueError(
                f"stack has {a.shape[1]} channels but this checkpoint was trained on "
                f"{len(self.channels)} ({self.channels}); got shape {a.shape}")
        if a.shape[2:] != (self.tile_size, self.tile_size):
            raise ValueError(
                f"tiles must be {self.tile_size}x{self.tile_size}; got {a.shape[2:]}")
        return a.astype(np.float32, copy=True)

    # -- public ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, stack, batch_size=8):
        """(N, C, 512, 512) dB -> (N, 512, 512) float32 P(crevasse) per pixel."""
        a = self._as_batch(stack)
        out = np.empty((len(a), self.tile_size, self.tile_size), dtype=np.float32)
        for i in range(0, len(a), batch_size):
            chunk = a[i:i + batch_size]
            pre = np.stack([normalize_stack(t, self.mu, self.sd) for t in chunk])
            x = torch.from_numpy(pre).to(self.device)
            logits = self._model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            out[i:i + batch_size] = probs
        return out

    def predict(self, stack, thresh=0.5, batch_size=8):
        """(N, ...) -> (N, 512, 512) bool, P >= thresh.

        `thresh=0.5` is a plain default, not a calibrated operating point -- there is no
        bundle-shipped threshold for this model.
        """
        p = self.predict_proba(stack, batch_size=batch_size)
        return p >= float(thresh)

    @torch.no_grad()
    def predict_noise(self, stack, batch_size=8):
        """(N, C, 512, 512) dB -> (N, C, 512, 512) float32 reconstructed "noise" tile.

        Only valid for a checkpoint trained with `--aux-noise-branch` (check
        `self.aux_decoder` first). This is the "capture the noise" half of the
        capture-and-remove idea; `stack - predict_noise(stack)` is the residual, in the
        checkpoint's own normalized (mu/sd) space, not raw dB -- denormalize with
        `predict_noise(stack) * self.sd + self.mu` before comparing to raw dB input.
        """
        if not self.aux_decoder:
            raise ValueError(
                f"checkpoint {self.checkpoint} has no aux noise branch "
                f"(aux_decoder=False) -- trained without --aux-noise-branch")
        a = self._as_batch(stack)
        out = np.empty_like(a, dtype=np.float32)
        for i in range(0, len(a), batch_size):
            chunk = a[i:i + batch_size]
            pre = np.stack([normalize_stack(t, self.mu, self.sd) for t in chunk])
            x = torch.from_numpy(pre).to(self.device)
            _, noise_recon = self._model(x)
            out[i:i + batch_size] = noise_recon.cpu().numpy()
        return out
