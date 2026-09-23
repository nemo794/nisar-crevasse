"""Array-in Python API coordinating the BIOMASS gate and U-Net: N tiles in, N results
out, in the original order, with a per-pixel segmentation only where the gate says to
look.

    from crevasse.biomass.pipeline_predict import BiomassCrevassePipeline
    pipe = BiomassCrevassePipeline()
    out = pipe.run(inten)          # inten (N, 4, 512, 512) linear intensity at 5 m

    out.gate_prob     # (N,)             float32  -- BiomassGate.predict_proba(...)[gate_method]
    out.gate_flag     # (N,)             bool     -- gate_prob >= gate_thresh
    out.unet_prob     # (N, 512, 512)    float32  -- NaN where gate_flag is False
    out.unet_mask     # (N, 512, 512) | None bool  -- only populated if unet_thresh is given

This is pure plumbing over two sibling modules' array APIs, in this same `src/biomass/`
directory. It does not retrain, refit, or re-derive anything:

* `bio_predict.py` (`BiomassGate`) -- stage 4, the tile gate.
* `bio_unet_predict.py` (`BiomassUNetSegmenter`) -- stage 5, the U-Net (defaults to the
  `nw=0.02` aux-noise-branch checkpoint -- see `docs/biomass/UNET_PIPELINE.md` for why).

Why the gate filter is not optional: `bio_build_shard_frangi.py`'s soft labels were
generated exclusively on `BiomassGate` (RF, `P>=0.65`) survivors -- the Frangi step never
ran on a tile the gate rejected. Calling `BiomassUNetSegmenter` on a gate-rejected tile
would be running it out of the distribution it was trained and evaluated on. `run` enforces
that ordering; there is no argument to skip it.

Why the default gate method is RF, not CNN: `bio_build_shard_frangi.py` (the actual label
generator) uses `method="rf"` at `gate_thresh=0.65` -- reproducing that is the point, not
an arbitrary choice. `method="cnn"`/`"both"` need `x_m`/`y_m` (EPSG:3031 tile top-left,
for the CNN's ice-velocity lookup); `BiomassGate` itself raises a clear error if they're
required and missing, so this module does not duplicate that check.

Restoring the stack
--------------------
`unet_prob` is always the same length and order as the input `inten`. Tiles the gate
rejected (or that were unscoreable -- `gate_prob` is NaN, `gate_flag` is False) get an
all-NaN mask at their position rather than being dropped, so a caller can zip `unet_prob`
back against its own tile positions without re-deriving which indices were skipped.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from crevasse.biomass.bio_predict import BiomassGate
from crevasse.biomass.bio_unet_predict import BiomassUNetSegmenter


@dataclass
class PipelineResult:
    gate_prob: np.ndarray
    gate_flag: np.ndarray
    unet_prob: np.ndarray
    unet_mask: Optional[np.ndarray]


class BiomassCrevassePipeline:
    """Coordinates `BiomassGate` -> filter -> `BiomassUNetSegmenter` -> scatter back."""

    def __init__(self, gate=None, segmenter=None, gate_method="rf", gate_thresh=0.65):
        """`gate=`/`segmenter=` accept pre-built `BiomassGate`/`BiomassUNetSegmenter`
        instances, for a caller that needs a non-default bundle/checkpoint.
        `gate_method`: "rf" (default, matches `bio_build_shard_frangi.py`) or "cnn" --
        NOT "both", since `run` has one gate_prob/gate_flag output to threshold; use
        `BiomassGate` directly for a two-model comparison, this class does not merge or
        choose between them any more than `BiomassGate.predict` does. `gate_thresh=0.65`:
        `BiomassGate` ships no default threshold at all (unlike `NisarGate`) -- `0.65` is
        the value already validated and used in production by `bio_build_shard_frangi.py`,
        not an arbitrary pick.
        """
        if gate_method not in ("rf", "cnn"):
            raise ValueError(
                f"gate_method must be 'rf' or 'cnn', got {gate_method!r} -- 'both' isn't "
                f"supported here because run() has one gate_prob/gate_flag output; call "
                f"BiomassGate.predict_proba(..., method='both') directly for a two-model "
                f"comparison, this class does not merge or choose between them.")
        self.gate = gate if gate is not None else BiomassGate()
        self.segmenter = segmenter if segmenter is not None else BiomassUNetSegmenter()
        self.gate_method = gate_method
        self.gate_thresh = gate_thresh

    def run(self, inten, x_m=None, y_m=None, gate_thresh=None, unet_thresh=None,
            batch_size=8):
        """(N, 4, 512, 512) linear intensity -> `PipelineResult`, N in, N out, same order.

        `x_m`/`y_m`: EPSG:3031 tile top-left metres, one per tile. Only required if this
        pipeline's `gate_method` is "cnn"/"both" -- `BiomassGate.predict_proba` raises its
        own clear error if they're needed and missing.

        `gate_thresh`: overrides the constructor's `gate_thresh` for this call only.

        `unet_thresh`: if given, also returns a binarized `unet_mask` at `P >= unet_thresh`
        for the flagged tiles. Left `None` by default -- soft probability masks are the
        primary output, matching this project's preference for continuous targets over
        premature binarization.
        """
        a = np.asarray(inten)
        if a.ndim == 3:
            a = a[None]

        gate_out = self.gate.predict_proba(a, x_m, y_m, method=self.gate_method)
        gate_prob = gate_out[self.gate_method]
        thresh = gate_thresh if gate_thresh is not None else self.gate_thresh
        gate_flag = np.where(np.isfinite(gate_prob), gate_prob, -np.inf) >= float(thresh)

        unet_prob = np.full((len(a), a.shape[2], a.shape[3]), np.nan, dtype=np.float32)
        idx = np.flatnonzero(gate_flag)
        if len(idx):
            db = 10 * np.log10(np.where(a[idx] > 0, a[idx], np.nan)).astype(np.float32)
            unet_prob[idx] = self.segmenter.predict_proba(db, batch_size=batch_size)

        unet_mask = None
        if unet_thresh is not None:
            unet_mask = np.zeros(unet_prob.shape, dtype=bool)
            if len(idx):
                unet_mask[idx] = unet_prob[idx] >= float(unet_thresh)

        return PipelineResult(gate_prob=gate_prob, gate_flag=gate_flag,
                               unet_prob=unet_prob, unet_mask=unet_mask)
