"""Acceptance test for a granule, to be run BEFORE spending effort labelling it.

Why this exists. Granule 025_048 was labelled and added to training before anyone asked
whether it behaved like the granules already in the set. It does not: on 359 tiles of
ground it shares with 025_019, with identical labels and identical code, the gate scores
blocked-OOF 0.904 reading from 025_019 and 0.627 reading from 025_048. The cause is still
unknown after ruling out geolocation, speckle correlation, temporal change, label
placement, coverage and pixel geometry. Finding out cost a full label-plus-retrain cycle.
The tests below are what would have caught it in twenty minutes.

Subcommands, cheapest first:

  orientation GRAN         Oriented-artifact screen. Catches the LSAR-type scene-wide
                           product defect ONLY. Necessary, nowhere near sufficient.
  register GRAN_A GRAN_B   Are the two products co-registered? Cross-correlates identical
                           world boxes.
  speckle GRAN_A GRAN_B    Do they have the same number of looks and the same speckle
                           correlation, measured on identical ground?
  ab GRAN_NEW GRAN_KNOWN   THE DECISIVE ONE. Same tiles, same labels, same code, only the
                           source granule varies. Requires labels for both.

Every subcommand prints a control next to its measurement. A number without its control is
not readable -- that is the single most expensive lesson in this project's history.

Run `ab` if you possibly can. The three cheap subcommands are screens: they can convict a
granule but they cannot clear one. Measured, on the granule that actually fails:
`orientation` gives it a clean 0.490 scene-wide (better than known-good 025_019's 0.531),
`register` finds it perfectly co-registered (0 px), and `speckle` finds a real 1.6x azimuth
difference that a causal test then REFUTED as the cause. All three pass or mislead. Only
`ab` returns 0.627 against 0.904.

Two naming hazards, stated because both have already caused confusion:

  * "orientation R" has meant two different statistics here. This module reports the
    CROSS-TILE resultant (concentration of per-tile dominant orientations, ~0.46 vs 0.86
    for 025_019 vs 025_048) and labels it as such. A WITHIN-TILE gradient resultant over
    all pixels gives ~0.013 vs 0.026 for the same granules -- same direction, magnitudes
    not comparable.
  * A swath principal-axis fit is deliberately NOT offered. It is the natural way to guess
    look direction from a geocoded product, but these footprints are near-square (aspect
    1.04 and 1.01), so the axis is unstable and the apparent 8-11 degree separation it
    reports is mostly noise. Look direction needs orbit metadata from the original HDF5.

    conda run -n nisar-gate python src/check_granule.py orientation 025_048
    conda run -n nisar-gate python src/check_granule.py ab 025_048 025_019
"""
import argparse
import csv
import sys

import numpy as np
import rasterio
import rasterio.enums
from rasterio.windows import from_bounds
from scipy.ndimage import uniform_filter
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

import crevasse.nisar.gate_common as E
import crevasse.nisar.train_gate_classifier as T
NMAX = 120          # matched tiles sampled for the speckle comparison
BOX = 4096          # native px per side of a registration window (20.5 km at 5 m)
DS = 2              # registration downsample: correlate on a 10 m grid
ORIENT_SAMPLE = 150  # tiles sampled for the orientation screen
ORIENT_REGIONS = 3   # split the swath into this many x this many regions


# ---------------------------------------------------------------- shared helpers

def labels_csv(granule):
    """Conventional label-CSV path for a granule, or exit naming what is missing."""
    p = E.ROOT / "data" / f"tile_labels_aoi_{granule}_speedmatched.csv"
    if not p.exists():
        raise SystemExit(f"no labels for {granule}: expected {p}\n"
                         f"Build them with build_aoi_tile_labels.py, or use the "
                         f"orientation/register/speckle subcommands, which need none.")
    return p


def keyed_labels(granule):
    """{world-grid key: (label, row, col)} so tiles can be matched across granules.

    Keying on world coordinates rather than (row, col) is the whole point: two granules on
    the same 5 m grid have different origins, so identical ground has different indices.
    """
    with rasterio.open(E.find_granule(granule)) as s:
        tr = s.transform
    out = {}
    for r in csv.DictReader(open(labels_csv(granule))):
        if r["label"] not in ("crevasse", "none"):
            continue
        row, col = int(r["row"]), int(r["col"])
        x, y = tr * (col + E.TS / 2, row + E.TS / 2)
        out[(round(x / (E.TS * 5)), round(y / (E.TS * 5)))] = (r["label"], row, col)
    return out


def read_tile(granule, row, col):
    with rasterio.open(E.find_granule(granule)) as s:
        return T.read_amp(s, row, col)


def enl(a):
    """Equivalent number of looks, via the (mean/std)^2 proxy. More looks => smoother."""
    return float((a.mean() / a.std()) ** 2)


def rho1(a, axis):
    """Lag-1 autocorrelation of the residual after removing a 9-px smooth background.

    Subtracting the background is what makes this measure speckle rather than scene
    structure. Independent speckle gives ~0; interpolation or oversampling correlates
    neighbours and pushes it up, anisotropically if the resampling was one-dimensional.
    """
    r = a.astype(np.float64) - uniform_filter(a.astype(np.float64), 9)
    r -= r.mean()
    den = (r * r).mean()
    if den < 1e-20:
        return 0.0
    if axis == 0:
        return float((r[:-1, :] * r[1:, :]).mean() / den)
    return float((r[:, :-1] * r[:, 1:]).mean() / den)


def dominant_orientation(amp):
    """Per-tile dominant gradient orientation (radians, doubled-angle mean) and weight."""
    img = T.nlm_image(amp)
    if img is None:
        return None
    gy, gx = np.gradient(img.astype(np.float64))
    ang = np.arctan2(gy, gx)
    w = np.hypot(gy, gx)
    z = (w * np.exp(2j * ang)).sum() / (w.sum() + 1e-12)
    return z


def cross_tile_R(zs):
    """Concentration of per-tile dominant orientations. 0 = isotropic, 1 = all aligned."""
    if not len(zs):
        return float("nan")
    u = np.array(zs)
    u = u / (np.abs(u) + 1e-12)          # direction only, so tile contrast cannot dominate
    return float(np.abs(u.mean()))


# ---------------------------------------------------------------- orientation

def cmd_orientation(args):
    """Screen for an oriented product artifact, scene-wide AND per region."""
    from crevasse.nisar.find_data_swath import find_data_bounds, get_valid_tile_positions
    from crevasse.nisar.map_crevasse_tiles import prefilter_positions

    print(f"=== orientation screen: {args.granule} ===")
    print("statistic: CROSS-TILE resultant of per-tile dominant orientations")
    print("(not the within-tile gradient resultant, which runs ~30x smaller)\n")

    path = str(E.find_granule(args.granule))
    with rasterio.open(path) as src:
        native = E.TS * T.tile_scale(src)   # positions are in native pixels
    ds = 200
    bounds, mask = find_data_bounds(path, downsample=ds)
    positions = prefilter_positions(get_valid_tile_positions(bounds, native),
                                    mask, ds, native)
    if len(positions) < ORIENT_SAMPLE:
        sample = positions
    else:
        rng = np.random.default_rng(E.SEED)
        idx = rng.choice(len(positions), ORIENT_SAMPLE, replace=False)
        sample = [positions[i] for i in idx]
    print(f"{len(positions)} valid tiles, sampling {len(sample)}")

    amps = Parallel(n_jobs=-1)(delayed(read_tile)(args.granule, r, c) for r, c in sample)
    got = [(p, a) for p, a in zip(sample, amps) if a is not None]
    zs = Parallel(n_jobs=-1)(delayed(dominant_orientation)(a) for _, a in got)
    keep = [(p, z) for (p, _), z in zip(got, zs) if z is not None]
    if not keep:
        raise SystemExit("no usable tiles -- check the granule has data")

    R_all = cross_tile_R([z for _, z in keep])
    print(f"\nscene-wide  R = {R_all:.3f}   (n={len(keep)})")

    # Per region, reported for context only -- measured NOT to discriminate. See the
    # closing text: the known-good granule reaches a regional 0.971 too.
    rows = np.array([p[0] for p, _ in keep])
    cols = np.array([p[1] for p, _ in keep])
    rq = np.quantile(rows, np.linspace(0, 1, ORIENT_REGIONS + 1))
    cq = np.quantile(cols, np.linspace(0, 1, ORIENT_REGIONS + 1))
    print(f"\nper region ({ORIENT_REGIONS}x{ORIENT_REGIONS} over the swath):")
    regional = []
    for i in range(ORIENT_REGIONS):
        cells = []
        for j in range(ORIENT_REGIONS):
            sel = ((rows >= rq[i]) & (rows <= rq[i + 1]) &
                   (cols >= cq[j]) & (cols <= cq[j + 1]))
            if sel.sum() < 5:
                cells.append("   n/a")
                continue
            r = cross_tile_R([z for (_, z), s in zip(keep, sel) if s])
            regional.append(r)
            cells.append(f"  {r:.3f}")
        print("   " + "".join(cells))

    print(f"\nREAD THIS CAREFULLY. Scene-wide R above ~0.7 with the same orientation")
    print(f"everywhere is the signature of a product-level oriented artifact: the")
    print(f"quarantined LSAR_HH product scores 0.89-0.95 and its 'crevasse' labels turned")
    print(f"out to be the artifact itself. That is the only thing this screen detects.")
    print(f"\nThe regional table is CONTEXT, NOT A VERDICT. It was added hoping it would")
    print(f"catch the regional pathology that scene-wide R misses, and it was measured")
    print(f"NOT to: known-good 025_019 reaches a regional 0.971 (scene-wide 0.531) and")
    print(f"failing 025_048 reaches 0.984 (scene-wide 0.490). High regional values are")
    print(f"normal -- real crevasse fields are locally aligned. Do not reject a granule")
    print(f"on this table.")
    print(f"\nHere: scene-wide {R_all:.3f}, highest region {max(regional):.3f}.")
    print(f"\nPassing this screen means only that the granule is not an LSAR-type oriented")
    print(f"artifact. It says NOTHING about whether the gate transfers to it. Run `ab`")
    print(f"against a known-good granule before labelling. That is the only test that has")
    print(f"ever caught a bad granule that this screen passed.")


# ---------------------------------------------------------------- registration

def read_world(granule, bounds):
    """Read a world-coordinate box, log-scaled, NaN where there is no data."""
    with rasterio.open(E.find_granule(granule)) as s:
        w = from_bounds(*bounds, transform=s.transform)
        a = s.read(1, window=w, boundless=True, fill_value=0).astype(np.float32)
    a = a[::DS, ::DS]
    m = a > 0
    out = np.full(a.shape, np.nan, np.float32)
    out[m] = np.log10(a[m] + 1e-10)
    return out


def prep_box(a):
    """Zero-mean unit-variance, holes filled with the mean. None if too sparse."""
    m = np.isfinite(a)
    if m.sum() < 0.5 * a.size:
        return None, m.mean()
    b = np.zeros(a.shape, np.float32)
    b[m] = a[m] - a[m].mean()
    s = b[m].std()
    if s < 1e-6:
        return None, m.mean()
    return b / s, m.mean()


def overlap_bounds(ga, gb):
    with rasterio.open(E.find_granule(ga)) as a, rasterio.open(E.find_granule(gb)) as b:
        b1, b2 = a.bounds, b.bounds
    return (max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3]))


def grid_boxes(ov, n=3):
    """n x n boxes spread over the overlap, inset to avoid the swath edges."""
    x0, y0, x1, y1 = ov
    w, h, side = x1 - x0, y1 - y0, BOX * 5.0
    return [(x0 + w * (i + 1) / (n + 1) - side / 2, y0 + h * (j + 1) / (n + 1) - side / 2,
             x0 + w * (i + 1) / (n + 1) + side / 2, y0 + h * (j + 1) / (n + 1) + side / 2)
            for i in range(n) for j in range(n)]


def registration_pair(ga, gb, label):
    from skimage.registration import phase_cross_correlation

    ov = overlap_bounds(ga, gb)
    if ov[2] <= ov[0] or ov[3] <= ov[1]:
        print(f"{label}: no overlap -- cannot test")
        return
    print(f"\n=== {label} ===")
    print(f"overlap {(ov[2]-ov[0])/1000:.1f} x {(ov[3]-ov[1])/1000:.1f} km")
    print(f"{'box':>4} {'validA':>7} {'validB':>7} {'dy_px':>7} {'dx_px':>7} "
          f"{'dy_m':>8} {'dx_m':>8} {'peak':>6}")
    shifts = []
    for k, bx in enumerate(grid_boxes(ov)):
        A, va = prep_box(read_world(ga, bx))
        B, vb = prep_box(read_world(gb, bx))
        if A is None or B is None:
            print(f"{k:>4} {va:>7.2f} {vb:>7.2f}   skipped (insufficient data)")
            continue
        n = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1])
        A, B = A[:n[0], :n[1]], B[:n[0], :n[1]]
        sh, _, _ = phase_cross_correlation(A, B, upsample_factor=4, normalization=None)
        peak = float(np.abs(np.fft.ifft2(np.fft.fft2(A) * np.conj(np.fft.fft2(B))).max())
                     / A.size)
        dy, dx = sh[0] * DS, sh[1] * DS
        print(f"{k:>4} {va:>7.2f} {vb:>7.2f} {dy:>7.2f} {dx:>7.2f} "
              f"{dy*5:>8.1f} {dx*5:>8.1f} {peak:>6.3f}")
        shifts.append((dy, dx, peak))
    if not shifts:
        return
    s = np.array(shifts)
    print(f"median shift  dy={np.median(s[:,0]):+.2f} px  dx={np.median(s[:,1]):+.2f} px"
          f"   ({np.median(s[:,0])*5:+.0f} m, {np.median(s[:,1])*5:+.0f} m)")
    strong = s[s[:, 2] >= np.median(s[:, 2])]
    print(f"strong-peak boxes only: dy={np.median(strong[:,0]):+.2f} "
          f"dx={np.median(strong[:,1]):+.2f} px")


def cmd_register(args):
    print("READ THE PEAK STRENGTH, NOT JUST THE SHIFT. Low-overlap boxes return")
    print("nonsense shifts with weak peaks in good and bad pairs alike.")
    registration_pair(args.granule_a, args.granule_b, f"TEST: {args.granule_a} vs {args.granule_b}")
    if args.control:
        registration_pair(args.granule_a, args.control,
                          f"CONTROL: {args.granule_a} vs {args.control} (known-good pair)")
        print("\nIf the control does not come back near (0,0) with a strong peak, the")
        print("harness is wrong and the TEST number above is void.")


# ---------------------------------------------------------------- speckle

def speckle_pair(ga, gb, tag):
    ka, kb = keyed_labels(ga), keyed_labels(gb)
    shared = sorted(set(ka) & set(kb))
    if not shared:
        print(f"{tag}: no shared labelled ground -- cannot compare")
        return
    rng = np.random.default_rng(E.SEED)
    if len(shared) > NMAX:
        shared = [shared[i] for i in rng.choice(len(shared), NMAX, replace=False)]
    print(f"\n=== {tag} === {len(shared)} matched tiles")

    acc = {ga: [], gb: []}
    for key in shared:
        pair = {}
        for g, kk in ((ga, ka), (gb, kb)):
            _, row, col = kk[key]
            amp = read_tile(g, row, col)
            if amp is None or not np.isfinite(amp).all():
                pair = None
                break
            z = dominant_orientation(amp)
            pair[g] = (enl(amp), rho1(amp, 0), rho1(amp, 1), abs(z) if z is not None else np.nan)
        if pair:
            for g in pair:
                acc[g].append(pair[g])

    names = ["ENL", "rho1_azim(y)", "rho1_range(x)", "within-tile R"]
    print(f"{'granule':>10} " + " ".join(f"{n:>14}" for n in names) + f"  n={len(acc[gb])}")
    for g in (ga, gb):
        v = np.array(acc[g])
        if not len(v):
            print(f"{g:>10}  no usable tiles")
            return
        print(f"{g:>10} " + " ".join(f"{np.median(v[:, i]):>14.3f}" for i in range(4)))
    a, b = np.array(acc[ga]), np.array(acc[gb])
    print(f"{'ratio b/a':>10} " + " ".join(
        f"{np.median(b[:, i]) / (np.median(a[:, i]) + 1e-12):>14.2f}" for i in range(4)))


def cmd_speckle(args):
    speckle_pair(args.granule_a, args.granule_b, f"TEST: {args.granule_a} vs {args.granule_b}")
    if args.control:
        speckle_pair(args.granule_a, args.control,
                     f"CONTROL: {args.granule_a} vs {args.control} (known-good pair)")
    print("\nEqual ENL means the same number of looks. A rho1 ratio the control pair does")
    print("not reproduce is a real product difference -- but note that on 025_048 a 1.59x")
    print("azimuth ratio was measured and then REFUTED as the cause by a causal test")
    print("(degrading 025_019 to match left its AUC at 0.907 vs 0.904). Treat a difference")
    print("here as a lead, not a verdict; only `ab` measures the thing you care about.")


# ---------------------------------------------------------------- matched-ground A/B

def blocked_auc(X, y, meta):
    """Spatially blocked out-of-fold AUC -- whole ~20 km blocks held out, not tiles."""
    g = T.spatial_groups(meta)
    oof = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=E.SEED)
    for tr, te in cv.split(X, y, g):
        oof[te] = T.make_rf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof), len(set(g))


def score_from(granule, keys, keyed, y, tag):
    """Featurize and score the matched tile set as read from one granule."""
    amps = Parallel(n_jobs=-1)(delayed(read_tile)(granule, keyed[k][1], keyed[k][2])
                               for k in keys)
    ok = [i for i, a in enumerate(amps) if a is not None]
    feats = Parallel(n_jobs=-1)(delayed(T.featurize)(amps[i]) for i in ok)
    keep = [ok[j] for j, f in enumerate(feats) if f is not None]
    X, _ = T.stack_features([f for f in feats if f is not None])
    meta = [(granule, keyed[keys[i]][1], keyed[keys[i]][2]) for i in keep]
    auc, nb = blocked_auc(X, y[keep], meta)
    print(f"  read from {tag:<22} n={len(keep):4d} blocks={nb:3d}  blockedOOF={auc:.3f}")
    return auc


def cmd_ab(args):
    """The test that caught 025_048. Same tiles, same labels, same code, one variable."""
    new, known = args.granule_new, args.granule_known
    kn, kk = keyed_labels(new), keyed_labels(known)
    shared = sorted(set(kn) & set(kk))
    if len(shared) < 100:
        raise SystemExit(
            f"only {len(shared)} tiles of shared labelled ground between {new} and "
            f"{known} -- not enough for a readable A/B. This test only works where "
            f"footprints overlap; if they do not, there is no substitute available "
            f"from the GeoTIFFs alone.")

    # All label sets derive from the same AlphaEarth mosaic, so agreement is expected.
    # Asserting it is what proves aoi_offset placed both on the correct ground.
    disagree = [k for k in shared if kn[k][0] != kk[k][0]]
    if disagree:
        raise SystemExit(
            f"{len(disagree)} of {len(shared)} matched tiles carry DIFFERENT labels in the "
            f"two CSVs. The premise of this test is that only the source granule varies, "
            f"so it cannot be run. Check aoi_offset and the label provenance first.")

    y = np.array([1 if kk[k][0] == "crevasse" else 0 for k in shared])
    print(f"=== matched-ground A/B: {new} vs {known} ===")
    print(f"{len(shared)} tiles of shared ground, identical labels "
          f"({int(y.sum())} crevasse / {int((1-y).sum())} none)")
    print("Same footprint, same read_amp, same featurize, same folds. Only the source")
    print("granule varies.\n")

    auc_known = score_from(known, shared, kk, y, f"{known} (CONTROL)")
    auc_new = score_from(new, shared, kn, y, f"{new} (TEST)")

    print(f"\ncontrol {auc_known:.3f}  test {auc_new:.3f}  delta {auc_new - auc_known:+.3f}")
    if auc_known < 0.85:
        print(f"\nCONTROL FAILED. {known} scores only {auc_known:.3f} on its own ground, so")
        print("this harness is not measuring the new granule -- fix that before reading the")
        print("test arm. A known-good granule should land near 0.90.")
    elif auc_new < auc_known - 0.10:
        print(f"\nFAIL. {new} loses {auc_known - auc_new:.3f} AUC on ground where {known}")
        print("succeeds, with labels and code held fixed. Do not train on it.")
        print("Reference: 025_048 loses 0.277 (0.904 -> 0.627) and its cause is unknown.")
    else:
        print(f"\nPASS. {new} behaves like {known} on shared ground.")


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("orientation", help="oriented-artifact screen, scene-wide + regional")
    p.add_argument("granule")
    p.set_defaults(fn=cmd_orientation)

    p = sub.add_parser("register", help="are two products co-registered?")
    p.add_argument("granule_a")
    p.add_argument("granule_b")
    p.add_argument("--control", help="third granule known to co-register with granule_a")
    p.set_defaults(fn=cmd_register)

    p = sub.add_parser("speckle", help="ENL and speckle correlation on matched ground")
    p.add_argument("granule_a")
    p.add_argument("granule_b")
    p.add_argument("--control", help="granule known to behave like granule_a")
    p.set_defaults(fn=cmd_speckle)

    p = sub.add_parser("ab", help="matched-ground A/B -- the decisive test")
    p.add_argument("granule_new")
    p.add_argument("granule_known")
    p.set_defaults(fn=cmd_ab)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
