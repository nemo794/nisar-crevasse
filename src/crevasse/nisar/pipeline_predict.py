"""Array-in Python API coordinating the NISAR gate and U-Net: N tiles in, N results out,
in the original order, with a per-pixel segmentation only where the gate says to look.

    from crevasse.nisar.pipeline_predict import CrevassePipeline
    pipe = CrevassePipeline()
    out = pipe.run(amp)          # amp (N, 512, 512) linear amplitude at 5 m

    out.gate_prob     # (N,)             float32  -- NisarGate.predict_proba
    out.gate_flag     # (N,)             bool     -- NisarGate.predict
    out.unet_prob     # (N, 512, 512)    float32  -- NaN where gate_flag is False
    out.unet_mask     # (N, 512, 512) | None bool  -- only populated if unet_thresh is given

This is pure plumbing over two sibling modules' array APIs, in this same `src/nisar/`
directory. It does not retrain, refit, or re-derive anything:

* `gate_predict.py` (`NisarGate`) -- stage 4, the tile gate.
* `unet_predict.py` (`UNetSegmenter`) -- stage 5, the U-Net.

Why the gate filter is not optional: the Frangi pseudo-labeller the U-Net was trained
against fires on pure speckle, so it is only ever run on tiles the gate has already
accepted (`docs/nisar/UNET_PIPELINE.md`, "why the gate exists"). Calling `UNetSegmenter`
on a gate-rejected tile would be running it out of the distribution it was trained and
evaluated on. `CrevassePipeline.run` enforces that ordering; there is no argument to skip
it.

Restoring the stack
--------------------
`unet_prob` is always the same length and order as the input `amp`. Tiles the gate
rejected (or that were unscoreable -- gate_prob is NaN, gate_flag is False) get an
all-NaN mask at their position rather than being dropped, so a caller can zip `unet_prob`
back against its own tile positions without re-deriving which indices were skipped.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from crevasse.nisar.gate_predict import NisarGate
from crevasse.nisar.unet_predict import UNetSegmenter


@dataclass
class PipelineResult:
    gate_prob: np.ndarray
    gate_flag: np.ndarray
    unet_prob: np.ndarray
    unet_mask: Optional[np.ndarray]


class CrevassePipeline:
    """Coordinates `NisarGate` -> filter -> `UNetSegmenter` -> scatter back to the stack."""

    def __init__(self, gate=None, segmenter=None, gate_thresh=None):
        """`gate=` / `segmenter=` accept pre-built `NisarGate`/`UNetSegmenter` instances,
        for a caller that needs e.g. `NisarGate(px_m=2.5, allow_unseen_spacing=True)` or a
        non-default checkpoint. `gate_thresh=None` defers to the gate bundle's own
        threshold at call time; see `run`."""
        self.gate = gate if gate is not None else NisarGate()
        self.segmenter = segmenter if segmenter is not None else UNetSegmenter()
        self.gate_thresh = gate_thresh

    def run(self, amp, gate_thresh=None, unet_thresh=None, batch_size=8):
        """(N, 512, 512) linear amplitude -> `PipelineResult`, N in, N out, same order.

        `gate_thresh`: overrides the constructor's `gate_thresh` for this call only.
        `None` (the default, at both levels) lets `NisarGate.predict` fall back to the
        bundle's own threshold (0.33301 at `recall>=0.95` for the shipped bundle) -- this
        module never re-implements that default-resolution logic.

        `unet_thresh`: if given, also returns a binarized `unet_mask` at `P >= unet_thresh`
        for the flagged tiles. Left `None` by default -- soft probability masks are the
        primary output, matching this project's preference for continuous targets over
        premature binarization (`soft_labels.py`'s target is continuous for the same
        reason).
        """
        a = np.asarray(amp)
        if a.ndim == 2:
            a = a[None]

        gate_prob = self.gate.predict_proba(a)
        thresh = gate_thresh if gate_thresh is not None else self.gate_thresh
        gate_flag = self.gate.predict(a, thresh=thresh)

        unet_prob = np.full(a.shape, np.nan, dtype=np.float32)
        idx = np.flatnonzero(gate_flag)
        if len(idx):
            # NisarGate takes (N, 512, 512); UNetSegmenter takes (N, 1, 512, 512) -- add
            # the explicit channel axis only for this call, output stays (N, 512, 512).
            unet_prob[idx] = self.segmenter.predict_proba(
                a[idx][:, None], batch_size=batch_size)

        unet_mask = None
        if unet_thresh is not None:
            unet_mask = np.zeros(a.shape, dtype=bool)
            if len(idx):
                unet_mask[idx] = unet_prob[idx] >= float(unet_thresh)

        return PipelineResult(gate_prob=gate_prob, gate_flag=gate_flag,
                               unet_prob=unet_prob, unet_mask=unet_mask)
