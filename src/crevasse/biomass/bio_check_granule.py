"""Acceptance test for a BIOMASS quad-pol granule, before it is labelled or trained on.

    conda run -n biomass python code/bio_check_granule.py
    conda run -n biomass python code/bio_check_granule.py --granule-dir data/biomass/BIO_S2_...

This is the BIOMASS counterpart of nisar-crevasse-gate/src/check_granule.py, and it is
a DIFFERENT test on purpose. Two of the NISAR checks do not transfer:

  * The orientation-concentration screen convicts every BIOMASS granule by construction.
    Measured 2026-09-14 on all 10 granules: R = 1.000 at 139.8-140.7 deg, per-tile |z|
    p50 ~= 0.72 against NISAR 025_019's 0.0120 (60x). The angle drifts <1 deg across four
    months and four footprints. That is not an artifact like 003_064 (R=0.948) was -- it is
    BIOMASS's intrinsic P-band SCS point spread function, elongated ~3:1 with its long axis
    at ~50 deg, resampled onto a square 5 m grid. Confirmed by directional lag-1
    autocorrelation: 0.78 along 45 deg vs 0.13 along 135 deg, where NISAR at the same
    declared 5 m is 0.03 isotropic. So R is REPORTED here, never used as a gate.

  * Texture screens are pointless. Every texture feature scored at chance against
    AlphaEarth on grounded ice (grad_HH 0.373, std_HH 0.506, grad_ratio 0.511, n=120).
    The frozen NISAR gate, whose 270 features are all texture, returned AUC 0.560 with
    probabilities collapsed into 0.445-0.538 -- out of distribution, falling back to its
    prior. BIOMASS carries crevasse signal radiometrically, not structurally.

What IS checked, and why each one earned its place:

  [1] All four polarizations present, same shape and transform.
      A partial download is a real failure mode -- one was caught mid-session as an empty
      granule dir that crashed a survey with KeyError: 'HH'.
  [2] 5.0 m square pixels, EPSG:3031.
      Non-square pixels broke the NISAR pipeline in several places; 2.5 m turned out to be
      a different sensor family entirely, and equalizing looks was built and made it worse.
  [3] Reciprocity: HV ~= VH and HH ~= VV.
      Measured r 0.997-0.999 and 0.990-0.994. This is what makes BIOMASS two channels, not
      four. If it breaks, the product is miscalibrated and the cross/co ratio is unsafe.
  [4] Grounded-ice cross-pol median against the ~-27 dB NESZ.
      3 of 10 granules sit at or below it (S3 M03 -30.3, S3 M02 -27.7, S2 M04 -27.2), with
      CV_HV climbing 1.43 -> 3.11 as it darkens. In those scenes the cross/co ratio -- the
      single best feature, AUC 0.890 -- degenerates into noise-over-co-pol. This is a HARD
      FAIL for any cross-pol-dependent model.
  [5] Integer pixel offset into the AlphaEarth raster.
      All 10 footprints pass, so existing Thwaites labels index BIOMASS tiles with no
      resampling of a categorical mask.
  [6] Valid data fraction on grounded ice.

Exit code 0 = usable, 1 = at least one hard failure.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "biomass", "gate")
POLS = ["HH", "HV", "VH", "VV"]
BM_PATH = os.path.join(ROOT, "data", "aois", "bedmap3_mask.tif")
AOI_PATH = os.path.join(ROOT, "data", "aois", "new", "thwaites_mosaic_new.vrt")

# BIOMASS spec noise-equivalent sigma-zero. A grounded-ice cross-pol median at or below
# this means the cross-pol channel is measuring the instrument, not the ice.
NESZ_DB = -27.0
EXPECT_PX_M = 5.0
EXPECT_EPSG = 3031

DEC = 32          # decimation for whole-scene radiometry; average-resampled
T = 512           # tile size, matching the rest of the pipeline
# Native-resolution grounded tiles sampled for the noise-floor and PSF measurements.
# 16 rather than a handful because the NESZ verdict is a median over these and it has to
# agree with bio_tile_features.py's, which takes its median over several hundred tiles.
ACF_TILES = 16


def _fail(msg):
    print(f"  FAIL  {msg}")
    return False


def _warn(msg):
    print(f"  WARN  {msg}")
    return True


def _ok(msg):
    print(f"  ok    {msg}")
    return True


def check_files(d):
    """[1] all four pols present, same shape and transform."""
    print("[1] product completeness")
    paths = {}
    for p in POLS:
        g = glob.glob(os.path.join(d, f"*_{p}_intensity.tif"))
        if not g:
            return _fail(f"no *_{p}_intensity.tif -- incomplete download?"), None
        if len(g) > 1:
            return _fail(f"{len(g)} files match *_{p}_intensity.tif"), None
        paths[p] = g[0]

    meta = {}
    for p, f in paths.items():
        with rasterio.open(f) as s:
            meta[p] = (s.shape, tuple(np.round(s.transform[:6], 6)), s.crs, s.res)
    ref = meta["HH"]
    for p in POLS[1:]:
        if meta[p][0] != ref[0]:
            return _fail(f"{p} shape {meta[p][0]} != HH {ref[0]}"), None
        if meta[p][1] != ref[1]:
            return _fail(f"{p} transform differs from HH"), None
    _ok(f"4 pols, shape {ref[0]}, identical transforms")
    return True, paths


def check_grid(paths):
    """[2] 5.0 m square pixels in EPSG:3031."""
    print("[2] grid")
    with rasterio.open(paths["HH"]) as s:
        res, crs, bounds = s.res, s.crs, s.bounds
    good = True
    if abs(res[0] - res[1]) > 1e-6:
        good = _fail(f"non-square pixels {res} -- several pipeline stages assume square")
    if abs(res[0] - EXPECT_PX_M) > 1e-6:
        good = _fail(f"pixel size {res[0]} m, expected {EXPECT_PX_M} m "
                     f"(2.5 m is a different sensor family -- do not mix)")
    if crs is None or crs.to_epsg() != EXPECT_EPSG:
        good = _fail(f"CRS {crs} , expected EPSG:{EXPECT_EPSG}")
    if good:
        _ok(f"{res[0]} m square, EPSG:{crs.to_epsg()}")
    return good, bounds


def read_decimated(paths):
    """Whole-scene decimated intensity per pol, average-resampled."""
    dat = {}
    for p in POLS:
        with rasterio.open(paths[p]) as s:
            h, w = s.shape
            dat[p] = s.read(1, out_shape=(h // DEC, w // DEC),
                            resampling=Resampling.average)
    return dat


def grounded_mask(bounds, shape):
    """bedmap3 grounded-ice mask resampled onto the decimated grid."""
    p = BM_PATH
    if not os.path.exists(p):
        return None
    with rasterio.open(p) as bm:
        v = bm.read(1, window=from_bounds(*bounds, transform=bm.transform),
                    out_shape=shape, boundless=True, fill_value=-9999,
                    resampling=Resampling.nearest)
    return v == 1


def sample_grounded_tiles(paths, n_want, tries=300, seed=0):
    """Full-resolution grounded tiles, {pol: intensity with nodata as nan}.

    Sampled at NATIVE resolution on purpose. Measuring the noise floor on
    average-decimated pixels biases the median upward -- block-averaging suppresses
    speckle variance -- and that divergence was real: S1 M04 read -25.46 dB from a
    32x-decimated scene (a pass) but -27.24 dB from per-tile medians (a fail), so it
    passed acceptance and was then silently dropped by bio_tile_features.py. Both now
    measure the same way.
    """
    with rasterio.open(paths["HH"]) as s:
        nr, nc = s.height // T, s.width // T
    srcs = {p: rasterio.open(paths[p]) for p in POLS}
    bm = rasterio.open(BM_PATH) if os.path.exists(BM_PATH) else None
    rng = np.random.default_rng(seed)
    out = []
    try:
        for _ in range(tries):
            if len(out) >= n_want:
                break
            i, j = int(rng.integers(nr)), int(rng.integers(nc))
            w = Window(j * T, i * T, T, T)
            if bm is not None:
                v = bm.read(1, window=from_bounds(*srcs["HH"].window_bounds(w),
                                                  transform=bm.transform),
                            out_shape=(16, 16), boundless=True, fill_value=-9999)
                if (v == 1).mean() < 0.95:
                    continue
            arr, bad = {}, False
            for p in POLS:
                a = srcs[p].read(1, window=w).astype(np.float64)
                a = np.where(a > 0, a, np.nan)
                if np.isfinite(a).mean() < 0.99:
                    bad = True
                    break
                arr[p] = a
            if not bad:
                out.append(arr)
    finally:
        for s in srcs.values():
            s.close()
        if bm is not None:
            bm.close()
    return out


def check_reciprocity(dat, gnd):
    """[3] HV ~= VH and HH ~= VV -- this is why BIOMASS is two channels, not four."""
    print("[3] reciprocity (BIOMASS has two independent channels, not four)")
    good = True
    for a, b, lo in [("HV", "VH", 0.99), ("HH", "VV", 0.95)]:
        x, y = dat[a][gnd], dat[b][gnd]
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if m.sum() < 100:
            good = _fail(f"{a}/{b}: only {m.sum()} valid grounded pixels")
            continue
        r = float(np.corrcoef(np.log10(x[m]), np.log10(y[m]))[0, 1])
        dd = 10 * np.log10(np.median(x[m])) - 10 * np.log10(np.median(y[m]))
        line = f"{a}/{b}: r={r:.4f}, median diff {dd:+.2f} dB"
        if r < lo:
            good = _fail(f"{line} -- expected r>={lo}; product may be miscalibrated")
        else:
            _ok(line)
    return good


def check_nesz(tiles):
    """[4] is cross-pol above the noise floor? HARD FAIL if not.

    Measured as the median of per-tile cross-pol medians at native resolution, which is
    exactly what bio_tile_features.py's skip rule uses, so the two cannot disagree.
    """
    print(f"[4] cross-pol vs NESZ ({NESZ_DB:+.0f} dB) on grounded ice")
    if len(tiles) < 3:
        return _fail("too few clean grounded tiles to judge the noise floor")
    xs, cs, cvs = [], [], []
    for a in tiles:
        cross = (a["HV"] + a["VH"]) / 2
        co = (a["HH"] + a["VV"]) / 2
        cross = cross[np.isfinite(cross) & (cross > 0)]
        co = co[np.isfinite(co) & (co > 0)]
        if cross.size < 100 or co.size < 100:
            continue
        xs.append(10 * np.log10(np.median(cross)))
        cs.append(10 * np.log10(np.median(co)))
        cvs.append(cross.std() / cross.mean())
    if not xs:
        return _fail("no tile had enough valid pixels")
    x_db, co_db, cv = float(np.median(xs)), float(np.median(cs)), float(np.median(cvs))
    print(f"        n={len(xs)} native-resolution grounded tiles")
    print(f"        cross {x_db:+.2f} dB   co {co_db:+.2f} dB   "
          f"ratio {x_db - co_db:+.2f} dB   CV_cross {cv:.3f}")
    if x_db <= NESZ_DB:
        return _fail(f"cross-pol median {x_db:+.2f} dB is at or below NESZ -- the "
                     f"cross/co ratio (the strongest feature, AUC 0.890) degenerates "
                     f"into noise-over-co-pol here. Do NOT use this granule in a "
                     f"cross-pol model.")
    if x_db <= NESZ_DB + 3:
        return _warn(f"cross-pol median {x_db:+.2f} dB is within 3 dB of NESZ -- "
                     f"ratio features will be noisy")
    return _ok(f"cross-pol {x_db:+.2f} dB, {x_db - NESZ_DB:.1f} dB clear of NESZ")


def check_aoi_alignment(bounds, paths):
    """[5] integer pixel offset into the AlphaEarth label raster."""
    print("[5] AlphaEarth grid alignment")
    a_path = AOI_PATH
    if not os.path.exists(a_path):
        return _warn(f"no AlphaEarth raster at {a_path} -- cannot label this granule")
    with rasterio.open(paths["HH"]) as s:
        tf, crs = s.transform, s.crs
    with rasterio.open(a_path) as a:
        if a.crs != crs:
            return _fail(f"AlphaEarth CRS {a.crs} != granule {crs}")
        if abs(a.res[0] - abs(tf.a)) > 1e-6 or abs(a.res[1] - abs(tf.e)) > 1e-6:
            return _fail(f"AlphaEarth res {a.res} != granule {(abs(tf.a), abs(tf.e))}")
        dc = (tf.c - a.transform.c) / a.transform.a
        dr = (tf.f - a.transform.f) / a.transform.e
        ab = a.bounds
    if abs(dr - round(dr)) > 1e-4 or abs(dc - round(dc)) > 1e-4:
        return _fail(f"fractional pixel offset dr={dr:.6f} dc={dc:.6f} -- a categorical "
                     f"mask must not be resampled")
    inside = (bounds[0] >= ab[0] and bounds[2] <= ab[2]
              and bounds[1] >= ab[1] and bounds[3] <= ab[3])
    _ok(f"integer offset dr={int(round(dr))} dc={int(round(dc))}"
        f"{'' if inside else '  (granule extends beyond the export box)'}")
    if not inside:
        _warn("outside the export box, byte 0 means NEVER EVALUATED, not "
              "'no crevasse' -- those tiles are not admissible negatives")
    return True


def check_coverage(dat, gnd):
    """[6] how much usable grounded ice is there?"""
    print("[6] coverage")
    valid = np.ones_like(dat["HH"], bool)
    for p in POLS:
        valid &= np.isfinite(dat[p]) & (dat[p] > 0)
    g = valid & gnd
    frac_valid = float(valid.mean())
    frac_gnd = float(g.sum() / max(valid.sum(), 1))
    n_tiles = int(g.sum() * (DEC / T) ** 2)
    print(f"        valid (all 4 pols) {frac_valid:.3f} of the bounding box")
    print(f"        grounded fraction of valid {frac_gnd:.3f}   "
          f"~{n_tiles} usable {T}px tiles")
    if n_tiles < 50:
        return _fail(f"only ~{n_tiles} grounded tiles -- not enough to train or score on")
    return _ok(f"~{n_tiles} grounded {T}px tiles")


def report_psf(tiles):
    """Reported, never a gate. See the module docstring for why."""
    print("[i] point spread function (REPORTED, not a pass/fail)")
    acfs = []
    for a in tiles:
        L = np.log10(a["HH"])
        L = L - np.nanmean(L)
        row = {}
        for nm, (dr, dc) in [("45", (1, 1)), ("135", (1, -1))]:
            x = L[1:-1, 1:-1].ravel()
            y = L[1 + dr:L.shape[0] - 1 + dr, 1 + dc:L.shape[1] - 1 + dc].ravel()
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() > 100:
                row[nm] = float(np.corrcoef(x[m], y[m])[0, 1])
        if len(row) == 2:
            acfs.append(row)
    if not acfs:
        print("        no clean grounded tile found to measure")
        return
    a45 = np.mean([r["45"] for r in acfs])
    a135 = np.mean([r["135"] for r in acfs])
    print(f"        lag-1 autocorrelation, n={len(acfs)} grounded tiles")
    print(f"          along  45 deg = {a45:.3f}")
    print(f"          along 135 deg = {a135:.3f}")
    print(f"          anisotropy    = {a45 - a135:+.3f}   "
          f"(NISAR 5 m reference: 0.03 / 0.03, isotropic)")
    if max(a45, a135) > 0.5:
        print("        -> oversampled: the 5 m grid carries fewer independent samples "
              "than it claims.")
        print("           Individual 10-30 m crevasses are not resolvable along the "
              "smooth axis, so")
        print("           ridge-based (Frangi) pseudo-labels are not admissible on this "
              "product.")


def check_granule(d):
    name = os.path.basename(d.rstrip("/"))
    m = re.match(r"BIO_(S\d)_SCS__1S_(\d{8})T.*_(M\d\d)_", name)
    tag = f"{m.group(1)} {m.group(3)} {m.group(2)}" if m else name[:40]
    print(f"\n{'=' * 74}\n{tag}   {name[:60]}\n{'=' * 74}")

    good, paths = check_files(d)
    if not good:
        return False
    g2, bounds = check_grid(paths)

    dat = read_decimated(paths)
    gnd = grounded_mask(bounds, dat["HH"].shape)
    if gnd is None:
        print("  WARN  no bedmap3_mask.tif -- checks [3][4][6] fall back to all pixels")
        gnd = np.ones_like(dat["HH"], bool)

    g3 = check_reciprocity(dat, gnd)
    tiles = sample_grounded_tiles(paths, ACF_TILES)
    g4 = check_nesz(tiles)
    g5 = check_aoi_alignment(bounds, paths)
    g6 = check_coverage(dat, gnd)
    report_psf(tiles)

    ok = all([g2, g3, g4, g5, g6])
    print(f"\n  VERDICT: {'USABLE' if ok else 'REJECTED'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--granule-dir", default=None,
                    help="one granule directory; default is every dir under data/biomass")
    ap.add_argument("--only", default=None, help="substring filter on the granule name")
    args = ap.parse_args()

    if args.granule_dir:
        dirs = [args.granule_dir]
    else:
        base = os.path.join(ROOT, "data", "biomass")
        dirs = [os.path.join(base, g) for g in sorted(os.listdir(base))
                if os.path.isdir(os.path.join(base, g))]
    if args.only:
        dirs = [d for d in dirs if args.only in os.path.basename(d)]
    if not dirs:
        sys.exit("no granule directories found")

    results = {os.path.basename(d): check_granule(d) for d in dirs}
    print(f"\n{'=' * 74}\nSUMMARY")
    for k, v in results.items():
        print(f"  {'USABLE  ' if v else 'REJECTED'}  {k[:64]}")
    n_bad = sum(not v for v in results.values())
    print(f"\n{len(results) - n_bad}/{len(results)} usable")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
