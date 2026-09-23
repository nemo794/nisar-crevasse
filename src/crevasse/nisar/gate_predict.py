"""Array-in Python API for the frozen crevasse gate: N tiles in, N labels out.

    from crevasse.nisar.gate_predict import NisarGate
    g = NisarGate()
    p = g.predict_proba(amp)          # amp (N, 512, 512) -> (N,) float32
    keep = g.predict(amp)             # (N,) bool at the bundle's own threshold

This exists because `map_crevasse_tiles.py` can only be entered with a GeoTIFF on
disk. A deployed pipeline that already holds tiles in memory would otherwise have to
write rasters to a temp directory to call its own model.

Input contract
--------------
* `amp` is **linear amplitude** (what the GSLC GeoTIFFs hold), not dB and not power,
  at **5.0 m** ground spacing, shape `(N, 512, 512)` or `(512, 512)`.
* **0 or NaN means nodata.** A tile with less than half its pixels valid is not scored.
* 512 is not negotiable: the FFT block reads four fixed 384 px corner windows.

Output contract
---------------
**N in, N out, same order.** An unscoreable tile is `NaN` in `predict_proba` and
`False` in `predict` -- never dropped, never reordered. Index the result against your
own tile list directly.

Two guards the CLI gets from the raster and an array cannot supply are moved into the
constructor: the looks policy and the pixel spacing. `px_m` is the **native** spacing of
the granule the tiles were cut from, not the spacing of the array -- the array is always
5 m, but a tile block-averaged down from a 2.5 m granule arrives with 4 looks where the
gate was trained on 1, and a spacing the bundle never saw a crevasse label at produces an
uninterpretable keep rate rather than an error (see docs/LIMITATIONS.md on 003_064). It
has to be passed because a bare array cannot be asked where it came from.
"""
from pathlib import Path

import numpy as np
import joblib

import crevasse.nisar.train_gate_classifier as T
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = _REPO_ROOT / "models" / "nisar" / "gate" / "gate_5m_freqA_2gran.joblib"
NBINS = T.D * T.D


class NisarGate:
    """The frozen RF gate, scoring in-memory amplitude tiles."""

    def __init__(self, model=DEFAULT_MODEL, px_m=5.0, allow_unseen_spacing=False):
        """`px_m` is the NATIVE spacing of the granule the tiles came from; the tiles
        themselves are always at 5 m. See the module docstring."""
        b = joblib.load(str(model))
        # A bundle trained under a different looks policy scores its own training
        # granule at 95% positive, so the mismatch is silent rather than merely
        # inaccurate. Same refusal as map_crevasse_tiles.py.
        if b.get("looks_policy") != T.LOOKS_POLICY:
            raise ValueError(
                f"bundle was trained with looks_policy={b.get('looks_policy')!r} but "
                f"this code delivers {T.LOOKS_POLICY!r}")
        # This API has no way to build context features (they need a granule id and the
        # per-granule mask npz), so a context bundle must be refused, not silently
        # scored on a short vector.
        if b.get("ctx_keys"):
            raise ValueError(
                f"bundle carries ctx_keys={list(b['ctx_keys'])} -- context features need "
                f"a granule id and _context_masks_*.npz, which an array cannot supply. "
                f"Use src/map_crevasse_tiles.py for a context bundle.")
        seen = b.get("train_px_m_positive")
        if seen and not any(abs(px_m - s) < 0.01 for s in seen):
            msg = (f"bundle has crevasse labels only at train_px_m_positive={list(seen)} m "
                   f"spacing; you passed px_m={px_m:g}")
            if not allow_unseen_spacing:
                raise ValueError(
                    msg + ". Refusing -- the keep rate would not be interpretable. Pass "
                          "allow_unseen_spacing=True to override.")

        self._b = b
        self._hand_keys = b.get("hand_keys") or b["feature_names"][:-NBINS]
        self.px_m = float(px_m)
        self.model_id = str(model)
        self.n_features = len(b["feature_names"])
        # Not a tuned optimum: chosen at a target recall on out-of-fold probabilities.
        self.threshold = b.get("threshold")
        self.threshold_mode = b.get("threshold_mode")

    # -- internals ---------------------------------------------------------------

    def _as_batch(self, amp):
        a = np.asarray(amp)
        if a.ndim == 2:
            a = a[None]
        if a.ndim != 3:
            raise ValueError(f"amp must be (N, {T.E.TS}, {T.E.TS}) or "
                             f"({T.E.TS}, {T.E.TS}); got shape {a.shape}")
        if a.shape[1:] != (T.E.TS, T.E.TS):
            raise ValueError(
                f"tiles must be {T.E.TS}x{T.E.TS} (the gate reads four fixed "
                f"{T.S} px corner FFT windows); got {a.shape[1:]}")
        a = a.astype(np.float32, copy=True)
        # dB would give plausible garbage rather than an error, which is the worst
        # failure mode here. Amplitude is non-negative; 0/NaN are the nodata codes.
        if np.any(a[np.isfinite(a)] < 0):
            raise ValueError(
                "amp holds negative values -- this API takes LINEAR amplitude, not dB. "
                "Convert with 10 ** (dB / 20) before calling.")
        return a

    def _featurize(self, a):
        """One tile -> feature vector in bundle order, or None if unscoreable."""
        m = a > 0                      # NaN compares False, so NaN is nodata too
        if m.mean() < 0.5:             # same cut as train_gate_classifier.read_amp
            return None
        a = a.copy()
        a[~m] = np.nan
        out = T.featurize(a)
        if out is None:
            return None
        hand, ff = out
        x = np.array([hand[k] for k in self._hand_keys] + list(ff), dtype=np.float32)
        return x if np.all(np.isfinite(x)) else None

    # -- public ------------------------------------------------------------------

    def predict_proba(self, amp):
        """(N, 512, 512) linear amplitude at 5 m -> (N,) float32 P(crevasse).

        NaN marks a tile that was not scored (under half its pixels valid, or a
        degenerate tile whose log-amplitude has no dynamic range).
        """
        a = self._as_batch(amp)
        p = np.full(len(a), np.nan, dtype=np.float32)
        feats = [(i, x) for i, x in ((i, self._featurize(a[i])) for i in range(len(a)))
                 if x is not None]
        if feats:
            X = np.stack([x for _, x in feats])
            p[[i for i, _ in feats]] = self._b["model"].predict_proba(X)[:, 1]
        return p

    def predict(self, amp, thresh=None):
        """(N, ...) -> (N,) bool, P >= thresh.

        `thresh=None` uses the bundle's own `threshold` (see `threshold_mode`, which
        for the shipped bundle is `recall>=0.95` -- a recall target, not a tuned
        optimum). An unscoreable tile is False.
        """
        if thresh is None:
            thresh = self.threshold
            if thresh is None:
                raise ValueError("bundle carries no 'threshold'; pass thresh= explicitly")
        p = self.predict_proba(amp)
        return np.where(np.isfinite(p), p, -np.inf) >= float(thresh)
