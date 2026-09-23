"""All six BIOMASS footprints in one EPSG:3031 mosaic: HH, AlphaEarth, gate P.

    conda run -n biomass python code/bio_prob_mosaic.py
    conda run -n biomass python code/bio_prob_mosaic.py --dec 8 --thresh 0.65

Writes data/bio_probmosaic.png (and _thr<t>.png when --thresh is given).

THE MOSAIC IS EXACT, NOT WARPED
-------------------------------
Every granule is EPSG:3031 at 5 m with bounds on exact 2560 m (= 512 px) multiples, so all
six tile grids coincide with each other and with the mosaic grid. Placement is integer index
arithmetic; nothing is reprojected or interpolated. The assert on that is load-bearing -- if a
future granule breaks alignment the panels would silently smear.

REPEAT PASSES OVERLAP, SO P IS AVERAGED WHERE THEY DO
----------------------------------------------------
S1/S2/S3 are repeat passes over the same Thwaites ground, so a tile can be scored up to six
times. The probability panel shows the MEAN over the passes that cover it, and the count panel
shows how many. Averaging repeat passes is the reason this figure is smoother than any single
scene -- do not read that smoothness as confidence.

SAME FULL-DATA-FIT CAVEAT AS bio_prob_scene.py
----------------------------------------------
Every admissible tile is scored, which requires the model trained on all of them
(data/bio_gate_t512.joblib), so most tiles here are IN SAMPLE. In sample it separates at
AUC ~0.95; buffered out-of-fold it is 0.889 against 0.601 for map coordinates alone. Read this
for geography -- which fields fire, whether the pattern is consistent between passes -- and
read bio_1to1_cal_POOLED.png for how good it is.

Blank in the probability panel is "no admissible tile": off-swath nodata, or ground bedmap3
calls shelf/ocean, both excluded by the feature builder's prescan before any model ran.
"""
import argparse
import glob
import os

import joblib
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "biomass", "gate")
T = 512
TILE_M = T * 5.0
AOI_PATH = os.path.join(ROOT, "data", "aois", "new", "thwaites_mosaic_new.vrt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(
        ROOT, "data", f"bio_tile_features_t{T}.npz"))
    ap.add_argument("--bundle", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(ROOT))), "models", "biomass", "gate", f"bio_gate_t{T}.joblib"))
    ap.add_argument("--dec", type=int, default=16, help="basemap decimation")
    ap.add_argument("--thresh", type=float, default=None,
                    help="if set, also write a masked-below-threshold version")
    ap.add_argument("--prob-npz", default=None,
                    help="score from this npz's p_full/p_oof (e.g. bio_cnn_prob_all.npz) "
                         "instead of the RF bundle. Draws both fields, since the out-of-fold "
                         "one is the quotable one and the full fit is the only complete one.")
    ap.add_argument("--tag", default=None, help="filename suffix, defaults to rf or the npz")
    args = ap.parse_args()
    dec = args.dec
    px = 5.0 * dec
    sub = T // dec

    d = np.load(args.features, allow_pickle=True)
    X = d["X"]
    gran, xm, ym = d["granule"], d["x_m"], d["y_m"]
    if args.prob_npz:
        pz = np.load(args.prob_npz, allow_pickle=True)
        fields = [("gate P, full-data fit", pz["p_full"].astype(float)),
                  ("gate P, buffered out-of-fold", pz["p_oof"].astype(float))]
        tag = args.tag or "cnn"
        src = f"CNN arm {pz['arm']}, {int(pz['seeds'])} seeds averaged"
    else:
        b = joblib.load(args.bundle)
        assert list(d["feature_names"]) == list(b["feature_names"]), "feature order differs"
        fields = [("gate P, full-data fit", b["model"].predict_proba(X)[:, 1])]
        tag = args.tag or "rf"
        src = "feature RF on the 38 merged features"
    for nm, P in fields:
        assert len(P) == len(xm), f"{nm}: {len(P)} scores for {len(xm)} tiles"

    base = os.path.join(ROOT, "data", "biomass")
    gs = sorted(set(gran.tolist()))
    hhs = {}
    for g in gs:
        f = glob.glob(os.path.join(base, g, "*_HH_intensity.tif"))
        if f:
            hhs[g] = f[0]

    bnds = []
    for g, f in hhs.items():
        with rasterio.open(f) as s:
            bnds.append(s.bounds)
            assert s.crs.to_epsg() == 3031 and s.transform.a == 5.0
    x0 = min(bb.left for bb in bnds)
    x1 = max(bb.right for bb in bnds)
    y0 = max(bb.top for bb in bnds)
    y1 = min(bb.bottom for bb in bnds)
    for v in (x0, x1, y0, y1):
        assert abs(v / TILE_M - round(v / TILE_M)) < 1e-6, f"{v} is off the 2560 m tile grid"
    W = int(round((x1 - x0) / px))
    H = int(round((y0 - y1) / px))
    print(f"mosaic {H} x {W} at {px:.0f} m  "
          f"({(x1-x0)/1000:.0f} x {(y0-y1)/1000:.0f} km)", flush=True)

    # HH: mean over passes, in linear intensity, then dB once at the end
    acc = np.zeros((H, W), np.float64)
    cnt = np.zeros((H, W), np.int32)
    for g, f in hhs.items():
        with rasterio.open(f) as s:
            h, w = s.shape
            a = s.read(1, out_shape=(h // dec, w // dec), resampling=Resampling.average)
            r0 = int(round((y0 - s.bounds.top) / px))
            c0 = int(round((s.bounds.left - x0) / px))
        m = np.isfinite(a) & (a > 0)
        sl = (slice(r0, r0 + a.shape[0]), slice(c0, c0 + a.shape[1]))
        acc[sl] += np.where(m, a, 0)
        cnt[sl] += m
    with np.errstate(invalid="ignore", divide="ignore"):
        bg = 10 * np.log10(np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan))

    with rasterio.open(AOI_PATH) as a:
        dr = (y0 - a.transform.f) / a.transform.e
        dc = (x0 - a.transform.c) / a.transform.a
        assert abs(dr - round(dr)) < 1e-4 and abs(dc - round(dc)) < 1e-4, \
            f"non-integer AlphaEarth offset dr={dr} dc={dc}"
        aoi = a.read(1, window=Window(int(round(dc)), int(round(dr)), W * dec, H * dec),
                     out_shape=(H, W), boundless=True, fill_value=0,
                     resampling=Resampling.average).astype(np.float32)

    # gate P on the shared 2560 m tile grid, averaged over repeat passes
    nr, nc = int(round((y0 - y1) / TILE_M)), int(round((x1 - x0) / TILE_M))
    ti = np.rint((y0 - ym) / TILE_M).astype(int)
    tj = np.rint((xm - x0) / TILE_M).astype(int)
    assert ti.min() >= 0 and tj.min() >= 0 and ti.max() < nr and tj.max() < nc
    pn = np.zeros((nr, nc), np.int32)
    np.add.at(pn, (ti, tj), 1)
    print(f"{len(xm)} tile slots over {int((pn > 0).sum())} distinct ground tiles; "
          f"passes per tile: {pn[pn > 0].mean():.2f} mean, {pn.max()} max", flush=True)

    def to_cells(P):
        """Mean over the passes that scored a tile, then exact integer expansion to cells."""
        ok = np.isfinite(P)
        psum = np.zeros((nr, nc), np.float64)
        pcnt = np.zeros((nr, nc), np.int32)
        np.add.at(psum, (ti[ok], tj[ok]), P[ok])
        np.add.at(pcnt, (ti[ok], tj[ok]), 1)
        ptile = np.where(pcnt > 0, psum / np.maximum(pcnt, 1), np.nan)
        cells = np.kron(ptile, np.ones((sub, sub), np.float32))
        assert cells.shape == bg.shape, f"{cells.shape} != {bg.shape}"
        return cells, int((pcnt > 0).sum())

    cells = []
    for nm, P in fields:
        c, ngt = to_cells(P)
        cells.append((nm, c, ngt))
        print(f"  {nm}: {int(np.isfinite(P).sum())} tile scores on {ngt} ground tiles",
              flush=True)
    ncell = np.kron(pn.astype(np.float32), np.ones((sub, sub), np.float32))

    v = bg[np.isfinite(bg)]
    lo, hi = np.percentile(v, [2, 98])
    ext = [x0 / 1000, x1 / 1000, y1 / 1000, y0 / 1000]

    def draw(panels, out, note):
        n = 2 + len(panels) + 1
        asp = H / W
        fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 6.2 * asp + 1.5))
        for ax in axes:
            ax.imshow(bg, cmap="gray", vmin=lo, vmax=hi, extent=ext,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0].set_title("HH dB, six passes averaged", fontsize=13)

        axes[1].imshow(np.ma.masked_where(aoi <= 0, aoi), cmap="autumn", vmin=0, vmax=1,
                       alpha=0.7, extent=ext, interpolation="nearest")
        axes[1].set_title("AlphaEarth crevasse mask", fontsize=13)

        for k, (nm, c, ngt) in enumerate(panels):
            ax = axes[2 + k]
            im = ax.imshow(np.ma.masked_invalid(c), cmap="turbo", vmin=0, vmax=1,
                           alpha=0.85, extent=ext, interpolation="nearest")
            ax.set_title(f"{nm}{note}\n{ngt} ground tiles", fontsize=12)
            fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02).set_label("P", fontsize=11)

        im = axes[-1].imshow(np.ma.masked_where(ncell < 1, ncell), cmap="viridis",
                             vmin=1, vmax=max(pn.max(), 2), alpha=0.85, extent=ext,
                             interpolation="nearest")
        axes[-1].set_title("passes covering each tile", fontsize=13)
        fig.colorbar(im, ax=axes[-1], fraction=0.040, pad=0.02).set_label("n", fontsize=11)

        fig.suptitle(
            f"BIOMASS Thwaites, {len(hhs)} footprints in one EPSG:3031 mosaic   "
            f"{src}   {int((pn > 0).sum())} ground tiles, {T}px @ 5 m, shown at {px:.0f} m   "
            f"(the full-data-fit panel is mostly in sample -- see the docstring)", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out, dpi=115, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}", flush=True)

    draw(cells, os.path.join(ROOT, "data", f"bio_probmosaic_{tag}.png"), "")
    if args.thresh is not None:
        t = args.thresh
        draw([(nm, np.where(c >= t, c, np.nan), ngt) for nm, c, ngt in cells],
             os.path.join(ROOT, "data", f"bio_probmosaic_{tag}_thr{t:.2f}.png"),
             f", masked below {t:.2f}")


if __name__ == "__main__":
    main()
