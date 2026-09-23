"""Array-in Python API for the crevasse U-Net: N tiles in, N probability masks out.

    from crevasse.nisar.unet_predict import UNetSegmenter
    seg = UNetSegmenter()                 # models/unet_025_019_f421_meansoft_g3/unet_best
    p = seg.predict_proba(amp)            # amp (N, 1, 512, 512) -> (N, 512, 512) float32 in [0,1]
    mask = seg.predict(amp, thresh=0.5)   # -> (N, 512, 512) bool

This exists for the same reason `gate_predict.py` does in `nisar-crevasse-gate`:
`infer_unet.py` / `infer_unet_batch_gpu.py` can only be entered with a GeoTIFF on disk.
A deployed pipeline that already holds tiles in memory (typically the survivors of the
stage-4 gate) would otherwise have to write rasters to a temp directory to call its own
model.

Input contract
---------------
* `amp` is raw **linear amplitude** (what the GSLC GeoTIFFs and `gate_predict.NisarGate`
  both take), not dB, shape `(N, 1, 512, 512)` or `(1, 512, 512)` -- the explicit
  size-1 channel axis matches `biomass-crevasse-unet`'s `(N, 4, 512, 512)` convention
  rather than NISAR's own older `(N, 512, 512)` shape. 512 must match the checkpoint's
  training tile size -- the architecture is fully convolutional so other sizes would
  technically run, but nothing here has ever been evaluated off 512.
* 0 or NaN means nodata, same convention as `gate_predict.py`.
* Preprocessing is `sar_dataset.preprocess_sar_tile` -- the SAME function every training
  and eval path in this repo uses (log10 -> per-tile p2/p98 contrast stretch -> [0,1]).
  `infer_unet.py` carries its own copy of this logic; importing the real one here is
  deliberate, so this module cannot drift from what the checkpoint was trained against
  the way `infer_unet.py`'s copy silently could. Control U1 is what checks this stays true.

Output contract
----------------
N in, N out, same order, same shape convention as `gate_predict.py`. Unlike the gate,
`predict_proba` has no "unscoreable tile" path: a fully-nodata tile still gets a
prediction, because the U-Net (unlike the 270-feature gate) has no dynamic-range
degeneracy check to refuse on. Deciding whether a tile was worth segmenting at all is
the caller's job -- in production that's the gate, upstream of this module, or the
coordinating pipeline repo that calls both.

There is no bundle-shipped operating threshold the way the gate has 0.33301 at
`recall>=0.95`. `predict`'s `thresh=0.5` default matches `docs/PIPELINE.md`'s own eval
convention ("quote 0.5 unless a threshold was chosen elsewhere") -- it is a plain
default, not a tuned or calibrated one.
"""
from pathlib import Path

import numpy as np
import torch

from crevasse.common.unet_model import UNet
from crevasse.nisar.sar_dataset import preprocess_sar_tile
from crevasse.common.ckpt_io import load_checkpoint

ROOT = Path(__file__).resolve().parents[3]
# Newest run (2026-09-14). Provisional, not a claim of finality: open task #41
# ("retrain U-Net f4 and mean{4,2,1} on the grounded-only block split", see
# biomass/.summary/CONTEXT-2026-09-16-full-resume.md) is not confirmed closed against
# this specific checkpoint. Pass `checkpoint=` to use any other run.
DEFAULT_CHECKPOINT = ROOT / "models" / "nisar" / "unet" / "unet_025_019_f421_meansoft_g3" / "unet_best"


class UNetSegmenter:
    """The trained U-Net, scoring in-memory SAR tiles."""

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, base_channels=None, device=None,
                 tile_size=512):
        """`base_channels=None` reads the value the checkpoint was trained with (recorded
        in `ckpt["args"]["base_channels"]` the way `eval_shard.py` reads it) rather than
        assuming 64 -- a mismatch there loads silently and scores garbage, it does not
        raise."""
        self.device = (torch.device(device) if device is not None else
                        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        ck = load_checkpoint(str(checkpoint), device=self.device)
        train_args = ck.get("args", {}) or {}
        bc = base_channels if base_channels is not None else int(
            train_args.get("base_channels", 64))

        model = UNet(n_channels=1, n_classes=1, base_channels=bc)
        model.load_state_dict(ck["model_state_dict"])
        model.to(self.device)
        model.eval()

        self._model = model
        self.checkpoint = str(checkpoint)
        self.tile_size = int(tile_size)
        self.base_channels = bc
        self.epoch = ck.get("epoch")
        self.val_iou = ck.get("val_iou")
        self.val_loss = ck.get("val_loss")

    # -- internals ---------------------------------------------------------------

    def _as_batch(self, amp):
        a = np.asarray(amp)
        if a.ndim == 3:
            a = a[None]
        if a.ndim != 4:
            raise ValueError(
                f"amp must be (N, 1, {self.tile_size}, {self.tile_size}) or "
                f"(1, {self.tile_size}, {self.tile_size}); got shape {a.shape}")
        if a.shape[1] != 1:
            raise ValueError(
                f"amp must have a single channel (shape (N, 1, {self.tile_size}, "
                f"{self.tile_size})); got {a.shape[1]} channels")
        if a.shape[2:] != (self.tile_size, self.tile_size):
            raise ValueError(
                f"tiles must be {self.tile_size}x{self.tile_size} to match the "
                f"checkpoint's training size; got {a.shape[2:]}")
        a = a.astype(np.float32, copy=True)
        finite = a[np.isfinite(a)]
        # dB would give plausible garbage rather than an error, which is the worst
        # failure mode here. Amplitude is non-negative; 0/NaN are the nodata codes.
        if finite.size and np.any(finite < 0):
            raise ValueError(
                "amp holds negative values -- this API takes LINEAR amplitude, not dB. "
                "Convert with 10 ** (dB / 20) before calling.")
        return a

    # -- public ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, amp, batch_size=8):
        """(N, 1, 512, 512) linear amplitude -> (N, 512, 512) float32 P(crevasse) per
        pixel. Output has no channel axis -- there is exactly one probability per pixel
        regardless of how many input channels there were, same as
        `BiomassUNetSegmenter.predict_proba`'s (N, 4, 512, 512) -> (N, 512, 512).

        Every tile gets a real prediction -- see the module docstring for why there is
        no NaN path here, unlike `gate_predict.NisarGate.predict_proba`.
        """
        a = self._as_batch(amp)
        out = np.empty((a.shape[0], a.shape[2], a.shape[3]), dtype=np.float32)
        for i in range(0, len(a), batch_size):
            chunk = a[i:i + batch_size, 0]
            pre = np.stack([preprocess_sar_tile(t) for t in chunk])
            x = torch.from_numpy(pre).unsqueeze(1).to(self.device)   # (B, 1, H, W)
            probs = torch.sigmoid(self._model(x)).squeeze(1).cpu().numpy()
            out[i:i + batch_size] = probs
        return out

    def predict(self, amp, thresh=0.5, batch_size=8):
        """(N, ...) -> (N, 512, 512) bool, P >= thresh.

        `thresh=0.5` is a plain default, not a calibrated operating point -- there is no
        bundle-shipped threshold for this model the way the gate has one. See the module
        docstring.
        """
        p = self.predict_proba(amp, batch_size=batch_size)
        return p >= float(thresh)
