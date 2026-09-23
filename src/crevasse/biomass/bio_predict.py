"""Array-in Python API for the BIOMASS gates: N tiles in, N labels out.

    from crevasse.biomass.bio_predict import BiomassGate
    g = BiomassGate()
    p = g.predict_proba(inten)["rf"]                        # (N,4,512,512) -> (N,)
    p = g.predict_proba(inten, x_m, y_m, method="both")     # {"rf": (N,), "cnn": (N,)}

This exists because `bio_score_tiles.py` can only be entered with a features npz built
from GeoTIFFs on disk. A deployed pipeline that already holds tiles in memory would
otherwise have to write rasters to a temp directory to call its own model.

Input contract
--------------
* `inten` is **linear intensity** (what the delivered `*_intensity.tif` files hold), not
  dB and not amplitude, at **5.0 m**, shape `(N, 4, 512, 512)` or `(4, 512, 512)`.
* Channel order is **HH, HV, VH, VV** -- `bio_tile_features.POLS`. Getting it wrong is
  silent: HV and VH correlate at 0.997-0.999, but HH/VV vs cross-pol is the whole signal.
* **0 or NaN means nodata.** A tile with under 99% of any pol valid is not scored -- the
  same `min_valid` cut the shipped feature table was built with, and applied to both
  models so `method="both"` answers for one population.
* `x_m`/`y_m` are the tile's **top-left** corner in EPSG:3031 metres, one per tile. They
  are optional and only needed for the CNN.

Why coordinates enable the CNN
------------------------------
The shipped checkpoint is `arm="sar+vel"`: its second input is per-tile ice speed, which
this module looks up in `data/context_grid_2560m.npz` (one cell per tile, so it is index
arithmetic, not a resample). There is no way to reconstruct that from pixels, so
`method="cnn"` without coordinates raises rather than guessing. Speed is carried as a
deliberate confound and is safe under the buffered split (0.518 alone); see
docs/LIMITATIONS.md 5.

The two probabilities are never combined
----------------------------------------
`method="both"` returns two arrays and derives nothing from the pair -- no mean, no vote,
no agreement. At the same 0.65 cut the RF flags ZERO tiles on two of three buffered folds
while the CNN flags some on all three, so averaging the scales would be arithmetic on
incommensurable numbers. `predict` requires the two thresholds separately for the same
reason, and there is no default for either.

Output contract
---------------
**N in, N out, same order.** An unscoreable tile is `NaN` in `predict_proba` and `False`
in `predict` -- never dropped, never reordered. Index the result against your own tile
list directly.
"""
import os
import warnings

import numpy as np

from crevasse.biomass.bio_tile_features import POLS, T, tile_features

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_BUNDLE = os.path.join(ROOT, "models", "biomass", "gate", f"bio_gate_t{T}.joblib")
DEFAULT_WEIGHTS = os.path.join(ROOT, "models", "biomass", "gate", "bio_cnn_gate_sarvel")
CTX_PATH = os.path.join(ROOT, "data", "biomass", "gate", "context_grid_2560m.npz")

NO_THRESH_MSG = (
    "thresh_{m} is required. There is no defensible default for BIOMASS.\n"
    "  See docs/RESULTS.md 'Operating point'.\n"
    "  thr 0.65 CNN lift: fold0 0.00x  fold1 1.67x  fold2 2.45x; the RF flags ZERO "
    "tiles on folds 0 and 2.\n"
    "  Precision at a fixed cut tracks base rate, not skill. Use predict_proba and pick "
    "a cut against the prevalence of the ground you are actually on.")


class BiomassGate:
    """The shipped RF and CNN gates, scoring in-memory 4-pol intensity tiles."""

    def __init__(self, bundle=DEFAULT_BUNDLE, weights=DEFAULT_WEIGHTS, px_m=5.0,
                 device="cpu", min_valid=0.99):
        if abs(px_m - 5.0) > 1e-6:
            raise ValueError(
                f"px_m={px_m:g}: every BIOMASS granule this gate was fitted on is "
                f"EPSG:3031 at exactly 5.0 m with bounds on 2560 m multiples, and every "
                f"scale-dependent feature assumes it. Resample before calling; "
                f"see docs/DATA.md.")
        self._bundle_path = bundle
        self._weights_path = weights
        self.px_m = float(px_m)
        self.device = device
        self.min_valid = float(min_valid)
        self._rf = None
        self._ck = None

    # -- lazy loading ------------------------------------------------------------
    # The RF path must not import torch (invariant 1), so the checkpoint is loaded
    # only when the CNN is actually asked for, exactly as bio_score_tiles.score_cnn does.

    def _load_rf(self):
        if self._rf is None:
            import joblib
            import sklearn
            b = joblib.load(self._bundle_path)
            if "thresh" in b:
                raise ValueError(
                    f"{self._bundle_path} carries a `thresh` key. Bundles written by this "
                    "repo do not; an older one wrote the best-F1 threshold over UNBUFFERED "
                    "out-of-fold scores. Retrain with src/bio_train_gate.py or delete the "
                    "key knowingly.")
            want = b.get("sklearn_version")
            if want and sklearn.__version__ != want:
                warnings.warn(
                    f"bundle was fitted with scikit-learn {want}, running "
                    f"{sklearn.__version__}. The forest unpickles across this gap; if the "
                    f"scores move, that is the cause.", RuntimeWarning, stacklevel=3)
            self._rf = b
        return self._rf

    def _load_cnn(self):
        if self._ck is None:
            from crevasse.common.ckpt_io import load_checkpoint  # deferred: keep this class importable
                                                  # without torch/safetensors until a
                                                  # CNN score is actually requested
            if not os.path.exists(str(self._weights_path) + ".safetensors"):
                raise ValueError(
                    f"CNN checkpoint not found: {self._weights_path}.safetensors. Pass "
                    f"weights=<stem, e.g. path to bio_cnn_gate_sarvel (no extension)>.")
            ck = load_checkpoint(self._weights_path)
            if int(ck["chip"]) != 128:
                raise ValueError(
                    f"checkpoint was trained at chip={ck['chip']} px; this API block-means "
                    f"512 -> 128. Chip size is not a free parameter: the head width "
                    f"depends on it.")
            if ck["arm"] != "sar+vel":
                raise ValueError(
                    f"checkpoint arm is {ck['arm']!r}; this API only builds the sar+vel "
                    f"input (4 chips + log ice speed).")
            self._ck = ck
        return self._ck

    # -- internals ---------------------------------------------------------------

    def _as_batch(self, inten):
        a = np.asarray(inten)
        if a.ndim == 3:
            a = a[None]
        if a.ndim != 4:
            raise ValueError(f"inten must be (N, 4, {T}, {T}) or (4, {T}, {T}); "
                             f"got shape {a.shape}")
        if a.shape[1] != len(POLS):
            raise ValueError(f"inten must have {len(POLS)} channels in the order "
                             f"{POLS}; got {a.shape[1]}")
        if a.shape[2:] != (T, T):
            raise ValueError(f"tiles must be {T}x{T} px (2.56 km at 5 m); got {a.shape[2:]}")
        a = a.astype(np.float64)
        # dB would give plausible garbage rather than an error, which is the worst failure
        # mode here. Intensity is non-negative; 0/NaN are the nodata codes.
        f = np.isfinite(a)
        if np.any(a[f] < 0):
            raise ValueError(
                "inten holds negative values -- this API takes LINEAR intensity, not dB. "
                "Convert with 10 ** (dB / 10) before calling.")
        return np.where(f & (a > 0), a, np.nan)

    def _valid(self, a):
        """Per-tile min-over-pols valid fraction >= min_valid.

        Applied to BOTH models, not just the RF. In the CLI the CNN can only ever see
        tiles that already cleared this cut at feature-build time, because the chip cache
        selects on finite features. Letting the array API score a mostly-nodata tile with
        the CNN would answer for ground the CLI refuses, and would make `method="both"`
        return two different populations.
        """
        return np.isfinite(a).mean(axis=(2, 3)).min(axis=1) >= self.min_valid

    def _coords(self, x_m, y_m, n, why):
        if x_m is None or y_m is None:
            raise ValueError(
                f"{why} needs x_m and y_m, the tile top-left corners in EPSG:3031 metres. "
                f"The shipped checkpoint is arm='sar+vel': its second input is per-tile ice "
                f"speed, read out of data/context_grid_2560m.npz at the tile centre. That "
                f"cannot be reconstructed from pixels. Pass coordinates, or use "
                f"method='rf'.")
        x = np.asarray(x_m, np.float64).ravel()
        y = np.asarray(y_m, np.float64).ravel()
        if len(x) != n or len(y) != n:
            raise ValueError(f"x_m/y_m must be one per tile: got {len(x)}/{len(y)} "
                             f"for {n} tiles")
        return x, y

    def _vel(self, x, y):
        """Per-tile ice speed from the 2560 m grid: one cell per tile, index arithmetic."""
        from rasterio.transform import Affine
        ctx = np.load(CTX_PATH, allow_pickle=True)
        inv = ~Affine(*ctx["transform"])
        cc, rr = inv * (x + T * 2.5, y - T * 2.5)          # tile CENTRE
        rr, cc = np.floor(rr).astype(int), np.floor(cc).astype(int)
        h, w = ctx["vel_mean"].shape
        if not ((rr >= 0).all() and (rr < h).all() and (cc >= 0).all() and (cc < w).all()):
            raise ValueError(
                "a tile centre falls outside the continent-wide context grid -- check that "
                "x_m/y_m are EPSG:3031 metres at the tile TOP-LEFT corner.")
        return ctx["vel_mean"][rr, cc].astype(np.float32)

    def _score_rf(self, a, ok):
        b = self._load_rf()
        names = list(b["feature_names"])
        p = np.full(len(a), np.nan, np.float32)
        rows = []
        for i in np.where(ok)[0]:
            arr = {q: a[i, k] for k, q in enumerate(POLS)}
            f = tile_features(arr)
            if f is None:
                continue
            x = np.array([f[k] for k in names], np.float32)
            if np.all(np.isfinite(x)):
                rows.append((int(i), x))
        if rows:
            X = np.stack([x for _, x in rows])
            p[[i for i, _ in rows]] = b["model"].predict_proba(X)[:, 1]
        return p

    def _chips(self, a, chip=128):
        """(N,4,512,512) intensity with nodata as nan -> (N,4,chip,chip) float16 dB.

        A plain block mean, which is what rasterio's Resampling.average delivers on these
        rasters: measured bit-identical to the shipped chip cache on fully-valid blocks
        (np.nanmean is NOT -- it is 1.17 dB out on a tile with interior nodata). A 4x4
        block straddling nodata is the one disagreement: GDAL emits a value there and this
        emits NaN, which score_with_net then zeroes after normalising. The float16 cast is
        required -- it is what the cache stores and what the net's mu/sd were fitted to.
        """
        f = T // chip
        blk = a.reshape(len(a), len(POLS), chip, f, chip, f).mean(axis=(3, 5))
        with np.errstate(invalid="ignore", divide="ignore"):
            return (10 * np.log10(np.where(blk > 0, blk, np.nan))).astype(np.float16)

    def _score_cnn(self, a, x, y, ok):
        import torch

        from crevasse.biomass.bio_cnn_gate import score_with_net
        ck = self._load_cnn()
        vel = self._vel(x, y)
        ctx = np.log10(np.maximum(vel, 1e-3)).astype(np.float32)[:, None]
        chips = self._chips(a, int(ck["chip"]))
        ok = ok & np.isfinite(ctx[:, 0])
        p = np.full(len(a), np.nan, np.float32)
        if ok.any():
            dev = torch.device(self.device)
            ps = [score_with_net(w, chips[ok], ctx[ok], dev) for w in ck["nets"]]
            p[ok] = np.mean(ps, axis=0)
        return p

    # -- public ------------------------------------------------------------------

    def predict_proba(self, inten, x_m=None, y_m=None, method="rf"):
        """(N,4,512,512) linear intensity at 5 m -> {name: (N,) float32 P(crevasse)}.

        `method` is "rf", "cnn" or "both". "both" returns both arrays and combines
        nothing. NaN marks a tile that was not scored (under `min_valid` of some pol
        valid, or degenerate).
        """
        if method not in ("rf", "cnn", "both"):
            raise ValueError(f"method must be 'rf', 'cnn' or 'both'; got {method!r}")
        a = self._as_batch(inten)
        ok = self._valid(a)
        out = {}
        if method in ("rf", "both"):
            out["rf"] = self._score_rf(a, ok)
        if method in ("cnn", "both"):
            x, y = self._coords(x_m, y_m, len(a), f"method={method!r}")
            out["cnn"] = self._score_cnn(a, x, y, ok)
        return out

    def predict(self, inten, x_m=None, y_m=None, method="rf",
                thresh_rf=None, thresh_cnn=None):
        """Same inputs -> {name: (N,) bool}, P >= the model's own threshold.

        Both thresholds are required when their model runs and neither has a default:
        the probability scale is fitted to 26-58% positive training ground and floods on
        0.8% positive ground. The ranking transfers; the calibration does not.
        An unscoreable tile is False.
        """
        thr = {"rf": thresh_rf, "cnn": thresh_cnn}
        p = self.predict_proba(inten, x_m, y_m, method=method)
        for m in p:
            if thr[m] is None:
                raise ValueError(NO_THRESH_MSG.format(m=m))
        return {m: np.where(np.isfinite(v), v, -np.inf) >= float(thr[m])
                for m, v in p.items()}
