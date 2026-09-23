"""Build the BIOMASS tile table: radiometric features + AlphaEarth labels, per 512px tile.

    conda run -n biomass python code/bio_tile_features.py
    conda run -n biomass python code/bio_tile_features.py --only 20260308 --max-tiles 50

Writes data/bio_tile_features_t512.npz.

WHY THESE FEATURES AND NOT THE NISAR ONES
-----------------------------------------
The NISAR gate's 270 features are 14 hand GLCM/statistical + 256 pooled-FFT magnitude
bins on an NL-means despeckled image -- all texture. Measured 2026-09-14 on 120
AlphaEarth-arbitrated grounded BIOMASS tiles, every texture feature came back at chance
(grad_HH 0.373, grad_HV 0.558, grad_ratio 0.511, std_HH 0.506) and the frozen NISAR gate
itself scored AUC 0.560 with its probabilities collapsed into 0.445-0.538 -- out of
distribution, falling back to its prior.

What DID separate was radiometry, and specifically the cross/co ratio:

    cross/co ratio_dB   AUC 0.890     corr with HH_dB  +0.125
    HV_dB               AUC 0.874     corr with HH_dB  +0.897   <- mostly brightness
    HH_dB               AUC 0.710

The ratio is the one to build on because it is near-orthogonal to co-pol brightness and
it survived every control: within row bands 0.871, residual after regressing out row and
col 0.817, per-granule 0.908/0.900, and on the 7 tiles sampled in both repeat-pass
granules three weeks apart its repeat correlation was +0.946 at mean |diff| 1.33 dB --
MORE stable than HH itself (+0.935, 2.82 dB).

So: no NL-means (the expensive despeckle is not earned if texture is at chance), and the
feature set is per-channel radiometry plus the ratio.

The one texture family kept is PSF-AXIS-ALIGNED. BIOMASS's point spread function is
elongated ~3:1 with its long axis at ~50 deg (lag-1 autocorrelation 0.78 along 45 deg vs
0.13 along 135 deg, against NISAR's 0.03 isotropic at the same declared 5 m). Isotropic
texture measures mostly that PSF, which is identical in every tile -- which is exactly why
it scored at chance. Measuring along and across the PSF axes separately, and taking the
difference, is the one way structural content could still be visible. It is included so
the trained model can reject it on the evidence rather than on my assumption.

CHANNELS
--------
BIOMASS is two independent channels, not four: HV~=VH (r 0.997-0.999) and HH~=VV
(r 0.990-0.994). So HV and VH are averaged into `cross` and HH/VV into `co`, and HH and VV
are also kept separately only so their difference can be checked for residual information.

LABELS
------
AlphaEarth, which is grid-aligned to every BIOMASS footprint at an exactly integer pixel
offset, so a categorical mask is never resampled. Carried forward from the NISAR work:
recall ~0.42 at precision ~0.94, and byte 0 OUTSIDE the export box means "never
evaluated", not "no crevasse". `aoi_inbox` records whether the tile is inside the box;
tiles outside it are not admissible negatives and the trainer must drop them.

Labels are OPTIONAL. No feature reads the mosaic, so when it is absent (the ordinary case
for a new granule -- it is 7.1 GB and not shipped) the table is written with `aoi_pos` NaN
and `aoi_inbox` False. Such a table can be scored and cannot be trained on. The grounded
prescan reads data/context_grid_2560m.npz rather than bedmap3_mask.tif for the same reason.
"""
import argparse
import glob
import os
import re

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "biomass", "gate")
POLS = ["HH", "HV", "VH", "VV"]
T = 512
DEC = 32                     # decimation for the coarse tile prescan; T/DEC = 16 cells
SUB = T // DEC
NESZ_DB = -27.0

AOI_PATH = os.path.join(ROOT, "data", "aois", "new", "thwaites_mosaic_new.vrt")
CTX_PATH = os.path.join(ROOT, "data", "context_grid_2560m.npz")


def _dir_acf(L):
    """lag-1 autocorrelation along the PSF axes (45 deg) and across them (135 deg)."""
    out = {}
    for nm, (dr, dc) in [("45", (1, 1)), ("135", (1, -1))]:
        x = L[1:-1, 1:-1].ravel()
        y = L[1 + dr:L.shape[0] - 1 + dr, 1 + dc:L.shape[1] - 1 + dc].ravel()
        m = np.isfinite(x) & np.isfinite(y)
        out[nm] = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 100 else np.nan
    return out


def _dir_grad(L):
    """mean |first difference| along and across the PSF axes."""
    d45 = np.abs(L[1:, 1:] - L[:-1, :-1])
    d135 = np.abs(L[1:, :-1] - L[:-1, 1:])
    return float(np.nanmean(d45)), float(np.nanmean(d135))


def tile_features(arr):
    """dict of features from {pol: intensity tile with nodata as nan}."""
    f = {}
    co = (arr["HH"] + arr["VV"]) / 2
    cross = (arr["HV"] + arr["VH"]) / 2
    chans = {"HH": arr["HH"], "VV": arr["VV"], "co": co, "cross": cross}

    for nm, a in chans.items():
        v = a[np.isfinite(a) & (a > 0)]
        if v.size < 0.5 * a.size:
            return None
        lg = np.log10(v)
        f[f"{nm}_dB"] = 10 * float(np.log10(np.median(v)))
        f[f"{nm}_logstd"] = float(lg.std())
        p10, p90 = np.percentile(lg, [10, 90])
        f[f"{nm}_logspread"] = float(p90 - p10)
        f[f"{nm}_cv"] = float(v.std() / v.mean())

    # the strongest single feature: cross/co depolarization ratio
    f["ratio_dB"] = f["cross_dB"] - f["co_dB"]
    f["hh_vv_dB"] = f["HH_dB"] - f["VV_dB"]

    # per-pixel ratio, so its spread is a real distribution not a difference of medians
    with np.errstate(invalid="ignore", divide="ignore"):
        rp = cross / np.where(co > 0, co, np.nan)
    rv = rp[np.isfinite(rp) & (rp > 0)]
    if rv.size < 100:
        return None
    lr = np.log10(rv)
    f["ratio_logstd"] = float(lr.std())
    f["ratio_logspread"] = float(np.subtract(*np.percentile(lr, [90, 10])))

    # PSF-axis-aligned texture. Isotropic texture is at chance because it measures the
    # PSF, which is identical everywhere; the along-vs-across DIFFERENCE is the part that
    # could still carry structure.
    for nm, a in [("HH", arr["HH"]), ("cross", cross), ("ratio", rp)]:
        with np.errstate(invalid="ignore", divide="ignore"):
            L = np.log10(np.where(a > 0, a, np.nan))
        L = L - np.nanmean(L)
        ac = _dir_acf(L)
        g45, g135 = _dir_grad(L)
        f[f"acf45_{nm}"] = ac["45"]
        f[f"acf135_{nm}"] = ac["135"]
        f[f"acfaniso_{nm}"] = ac["45"] - ac["135"]
        f[f"grad45_{nm}"] = g45
        f[f"grad135_{nm}"] = g135
        f[f"gradaniso_{nm}"] = g135 - g45
    return f


def ctx_grounded(tf, nr, nc):
    """Per-tile grounded fraction, looked up in the 2560 m context grid.

    One grid cell per 512 px tile at 5 m, so this is index arithmetic on the tile centre --
    the same lookup as bio_tile_cache.py. It stands in for a native-resolution
    bedmap3_mask.tif read, which is 7.1 GB and not shipped, so a tile straddling the
    grounding line can be admitted or rejected differently than in the shipped table.
    """
    ctx = np.load(CTX_PATH, allow_pickle=True)
    i, j = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    xm = tf.c + (j * T + T / 2) * tf.a
    ym = tf.f + (i * T + T / 2) * tf.e
    inv = ~Affine(*ctx["transform"])
    cc, rr = inv * (xm, ym)
    rr, cc = np.floor(rr).astype(int), np.floor(cc).astype(int)
    h, w = ctx["grounded_frac"].shape
    assert (rr >= 0).all() and (rr < h).all() and (cc >= 0).all() and (cc < w).all(), \
        "a tile centre falls outside the continent-wide context grid"
    return ctx["grounded_frac"][rr, cc].astype(np.float32)


def coarse_masks(paths, tf, shape_full):
    """Tile-grid grounded and valid fractions, from a DEC-decimated whole-scene read."""
    h, w = shape_full
    nr, nc = h // T, w // T
    oh, ow = nr * SUB, nc * SUB

    valid = np.ones((oh, ow), bool)
    for p in POLS:
        with rasterio.open(paths[p]) as s:
            d = s.read(1, out_shape=(h // DEC, w // DEC),
                       resampling=Resampling.average)[:oh, :ow]
        valid &= np.isfinite(d) & (d > 0)
    vf = valid.reshape(nr, SUB, nc, SUB).mean(axis=(1, 3))
    return vf, ctx_grounded(tf, nr, nc)


def process_granule(d, min_gnd, min_valid, max_tiles, skip_nesz_fail=True,
                    aoi_path=AOI_PATH):
    name = os.path.basename(d.rstrip("/"))
    m = re.match(r"BIO_(S\d)_SCS__1S_(\d{8})T.*_(M\d\d)_", name)
    if m is None:
        raise SystemExit(
            f"cannot read stack/date/footprint out of {name!r} (in {d}).\n"
            r"  expected BIO_(S\d)_SCS__1S_(\d{8})T..._(M\d\d)_ ." "\n"
            "  Rename the directory to the delivered granule ID rather than relaxing "
            "the pattern -- stack and footprint are what the fold split is cut on.")
    stack, date, foot = m.group(1), m.group(2), m.group(3)

    paths = {p: glob.glob(os.path.join(d, f"*_{p}_intensity.tif"))[0] for p in POLS}
    with rasterio.open(paths["HH"]) as s:
        shape_full, tf = s.shape, s.transform
    labels = os.path.exists(aoi_path)
    dr = dc = 0
    abox = None
    if labels:
        with rasterio.open(aoi_path) as a:
            dr = int(round((tf.f - a.transform.f) / a.transform.e))
            dc = int(round((tf.c - a.transform.c) / a.transform.a))
            abox = a.bounds

    vf, gf = coarse_masks(paths, tf, shape_full)
    cand = np.argwhere((vf >= min_valid) & (gf >= min_gnd))
    print(f"  {stack} {foot} {date}: {len(cand)} candidate grounded tiles", flush=True)
    if max_tiles:
        rng = np.random.default_rng(0)
        cand = cand[rng.choice(len(cand), min(max_tiles, len(cand)), replace=False)]

    srcs = {p: rasterio.open(paths[p]) for p in POLS}
    aoi = rasterio.open(aoi_path) if labels else None
    rows = []
    for k, (i, j) in enumerate(cand):
        r, c = int(i) * T, int(j) * T
        w = Window(c, r, T, T)
        arr = {}
        bad = False
        for p in POLS:
            a = srcs[p].read(1, window=w).astype(np.float64)
            a = np.where(a > 0, a, np.nan)
            if np.isfinite(a).mean() < 0.99:
                bad = True
                break
            arr[p] = a
        if bad:
            continue
        f = tile_features(arr)
        if f is None:
            continue
        if labels:
            av = aoi.read(1, window=Window(c + dc, r + dr, T, T),
                          boundless=True, fill_value=0)
            apos = float((av == 1).mean())
        else:
            apos = np.nan
        tb = srcs["HH"].window_bounds(w)
        f.update(granule=name, stack=stack, foot=foot, date=date, row=r, col=c,
                 grounded_frac=float(gf[i, j]), valid_frac=float(vf[i, j]),
                 aoi_pos=apos,
                 aoi_inbox=bool(labels and tb[0] >= abox[0] and tb[2] <= abox[2]
                                and tb[1] >= abox[1] and tb[3] <= abox[3]),
                 x_m=float(tb[0]), y_m=float(tb[3]))
        rows.append(f)
        if (k + 1) % 200 == 0:
            print(f"    {k+1}/{len(cand)} scanned, {len(rows)} kept", flush=True)
    for s in srcs.values():
        s.close()
    if aoi is not None:
        aoi.close()

    # NESZ gate, measured on the tiles actually kept rather than a hardcoded list.
    # Below the noise floor the cross/co ratio is noise-over-co-pol, i.e. a disguised
    # monotone function of HH -- it would look informative and mean nothing.
    if rows:
        xdb = float(np.median([r["cross_dB"] for r in rows]))
        if xdb <= NESZ_DB:
            msg = (f"    cross-pol median {xdb:+.2f} dB <= NESZ {NESZ_DB:+.0f} dB")
            if skip_nesz_fail:
                print(f"{msg} -- SKIPPING granule", flush=True)
                return []
            print(f"{msg} -- keeping anyway (--no-skip-nesz-fail)", flush=True)
    print(f"    -> {len(rows)} tiles kept", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="substring filter on granule name")
    ap.add_argument("--min-grounded", type=float, default=0.95)
    ap.add_argument("--min-valid", type=float, default=0.99)
    ap.add_argument("--max-tiles", type=int, default=None,
                    help="cap per granule, for a quick test")
    ap.add_argument("--no-skip-nesz-fail", dest="skip_nesz_fail", action="store_false",
                    help="keep granules whose grounded cross-pol median is at or below "
                         "NESZ. Default is to skip them (3 of 10 as of 2026-09-14) "
                         "because their ratio features are noise-over-co-pol.")
    ap.add_argument("--aoi", default=AOI_PATH,
                    help="AlphaEarth crevasse mosaic. Labels only -- no feature reads it. "
                         "If it is absent the table is written with aoi_pos NaN and "
                         "aoi_inbox False, which is scoreable but not trainable.")
    ap.add_argument("--base", default=os.path.join(ROOT, "data", "biomass"),
                    help="directory holding granule directories")
    ap.add_argument("--out", default=os.path.join(ROOT, "data",
                                                  f"bio_tile_features_t{T}.npz"))
    args = ap.parse_args()

    if not os.path.exists(args.aoi):
        print(f"WARN  no AlphaEarth mosaic at {args.aoi}\n"
              "      writing a SCORING-ONLY table: aoi_pos NaN, aoi_inbox False.\n"
              "      The 38 features are unaffected; bio_train_gate.py will refuse it.")

    base = args.base
    dirs = [os.path.join(base, g) for g in sorted(os.listdir(base))
            if os.path.isdir(os.path.join(base, g))
            and len(glob.glob(os.path.join(base, g, "*.tif"))) >= 4]
    if args.only:
        dirs = [d for d in dirs if args.only in os.path.basename(d)]
    if not dirs:
        raise SystemExit(f"no granule directory under {base} with at least four .tif files"
                         + (f" matching --only {args.only!r}" if args.only else ""))

    allrows = []
    for d in dirs:
        allrows += process_granule(d, args.min_grounded, args.min_valid,
                                   args.max_tiles, args.skip_nesz_fail, args.aoi)

    if not allrows:
        raise SystemExit("no tiles")
    keys = [k for k in allrows[0] if isinstance(allrows[0][k], (int, float, bool, np.floating))
            and k not in ("aoi_inbox",)]
    feat_keys = [k for k in keys if k not in ("row", "col", "grounded_frac",
                                              "valid_frac", "aoi_pos", "x_m", "y_m")]
    X = np.array([[r[k] for k in feat_keys] for r in allrows], np.float32)
    np.savez(
        args.out,
        X=X,
        feature_names=np.array(feat_keys),
        aoi_pos=np.array([r["aoi_pos"] for r in allrows], np.float32),
        aoi_inbox=np.array([r["aoi_inbox"] for r in allrows], bool),
        grounded_frac=np.array([r["grounded_frac"] for r in allrows], np.float32),
        row=np.array([r["row"] for r in allrows], np.int32),
        col=np.array([r["col"] for r in allrows], np.int32),
        x_m=np.array([r["x_m"] for r in allrows], np.float64),
        y_m=np.array([r["y_m"] for r in allrows], np.float64),
        granule=np.array([r["granule"] for r in allrows]),
        stack=np.array([r["stack"] for r in allrows]),
        foot=np.array([r["foot"] for r in allrows]),
        date=np.array([r["date"] for r in allrows]),
    )
    print(f"\nwrote {args.out}")
    print(f"  {X.shape[0]} tiles x {X.shape[1]} features")
    ap_ = np.array([r["aoi_pos"] for r in allrows])
    if not np.isfinite(ap_).any():
        print("  no labels: scoring-only table. Score it with src/bio_score_tiles.py.")
        return
    print(f"  {sum(r['aoi_inbox'] for r in allrows)} inside the AlphaEarth export box")
    print(f"  AlphaEarth positive fraction: >=0.5 in {(ap_ >= 0.5).sum()}, "
          f"<=0.05 in {(ap_ <= 0.05).sum()}, ambiguous in {((ap_ > 0.05) & (ap_ < 0.5)).sum()}")


if __name__ == "__main__":
    main()
