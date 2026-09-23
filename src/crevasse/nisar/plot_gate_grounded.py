"""SAR-only gate predictions for every scored granule, with Bedmap3 applied as a hard
mask and ITS_LIVE used only to stratify the report.

This is the gate as it should be run: ONE random forest on SAR texture, physical rasters
used as a deterministic filter and as reporting strata, never as learned features. A
context prior fitted on ice speed was measured to be a spatial memorizer (raw row/col
scores as well), so it is deliberately absent here.

Non-grounded tiles are drawn in blue-grey and excluded from the decision -- they are not
"P=0 crevasse-free", they are surface the gate was never meant to judge.

    conda run -n nisar-gate python src/plot_gate_grounded.py
    conda run -n nisar-gate python src/plot_gate_grounded.py --thresh 0.85
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import crevasse.nisar.gate_common as E
from crevasse.nisar.build_aoi_tile_labels import find_granule
from crevasse.nisar.plot_granule_layers import crop, scores_to_grid, tile_grid_basemap

GROUNDED = 1
BANDS = [(0, 50), (50, 250), (250, 1000), (1000, np.inf)]


def band_label(lo, hi):
    return f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"


def one(granule, thresh, tile):
    """Grids and stats for a granule, or None if it has not been scored."""
    npz = E.ROOT / "data" / "maps" / f"gate_sar_{granule}.npz"
    ctxp = E.ROOT / "data" / f"_context_masks_{granule}_t{tile}.npz"
    if not npz.exists() or not ctxp.exists():
        return None
    ctx = np.load(ctxp)
    shape = ctx["grounded_frac"].shape
    base, _ = tile_grid_basemap(find_granule(granule), *shape, tile)
    sar, thr = scores_to_grid(npz, shape, tile)
    thr = thresh if thresh is not None else thr
    gnd = ctx["bedmap_class"] == GROUNDED
    vel = ctx["vel_mean"].astype(np.float32)
    base, sar, gnd, vel = crop([base, sar, gnd, vel], np.isfinite(base))

    data = np.isfinite(sar)
    keep = data & gnd & (sar >= thr)
    stats = {"thr": thr, "n_data": int(data.sum()), "n_gnd": int((data & gnd).sum()),
             "n_keep": int(keep.sum()), "bands": []}
    for lo, hi in BANDS:
        with np.errstate(invalid="ignore"):
            sel = data & gnd & (vel >= lo) & (vel < hi)
        stats["bands"].append((band_label(lo, hi), int(sel.sum()),
                               int((sel & keep).sum())))
    with np.errstate(invalid="ignore"):
        nm = data & gnd & ~np.isfinite(vel)
    stats["bands"].append(("unmapped", int(nm.sum()), int((nm & keep).sum())))
    return base, sar, gnd, data, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--granules", nargs="+",
                    default=["025_019", "025_091", "004_048", "003_064"])
    ap.add_argument("--tile", type=int, default=E.TS)
    ap.add_argument("--thresh", type=float, default=None,
                    help="override the threshold stored in each npz")
    ap.add_argument("--min-p", type=float, default=None,
                    help="only draw tiles with P >= this; also the colour-scale floor")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    got = [(g, one(g, args.thresh, args.tile)) for g in args.granules]
    missing = [g for g, r in got if r is None]
    got = [(g, r) for g, r in got if r is not None]
    if missing:
        print(f"skipped (no gate_sar_<g>.npz or context mask): {', '.join(missing)}")
    if not got:
        raise SystemExit("nothing to plot")

    n = len(got)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9 * ncol, 7.5 * nrow),
                             squeeze=False)
    lo = args.min_p if args.min_p is not None else 0.0

    print(f"\n{'granule':10s} {'thr':>6s} {'data':>7s} {'grounded':>9s} "
          f"{'kept':>7s} {'rate':>7s}")
    print("-" * 52)
    for ax, (g, (base, sar, gnd, data, st)) in zip(axes.ravel(), got):
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(base, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        # Excluded surface: shown, but visibly not a probability.
        ax.imshow(np.ma.masked_where(~(data & ~gnd), np.ones_like(sar)),
                  cmap=matplotlib.colors.ListedColormap(["#5b7c99"]),
                  vmin=0, vmax=1, alpha=0.55, interpolation="nearest")
        shown = np.where((sar >= lo) & gnd, sar, np.nan)
        im = ax.imshow(np.ma.masked_invalid(shown), cmap="inferno", vmin=lo, vmax=1,
                       interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
        cb.set_label("P(crevasse)  [grounded ice only]")
        rate = st["n_keep"] / max(st["n_gnd"], 1)
        ax.set_title(f"{g} — {st['n_keep']} kept of {st['n_gnd']} grounded "
                     f"({rate:.0%})  P>={st['thr']:.3f}\n"
                     f"{st['n_data'] - st['n_gnd']} of {st['n_data']} data tiles "
                     f"excluded as non-grounded", fontsize=10)
        print(f"{g:10s} {st['thr']:6.3f} {st['n_data']:7d} {st['n_gnd']:9d} "
              f"{st['n_keep']:7d} {rate:6.0%}")

    for ax in axes.ravel()[len(got):]:
        ax.axis("off")
    axes.ravel()[0].legend(
        handles=[Patch(color="#5b7c99", label="excluded (not grounded ice)")],
        loc="lower left", fontsize=8, framealpha=0.9)

    print(f"\nkeep rate by ITS_LIVE speed band (grounded tiles only, m/yr):")
    hdr = [b[0] for b in got[0][1][4]["bands"]]
    print(f"{'granule':10s} " + "".join(f"{h:>16s}" for h in hdr))
    for g, (_, _, _, _, st) in got:
        cells = []
        for _, tot, kept in st["bands"]:
            cells.append(f"{kept}/{tot}" + (f" {kept/tot:.0%}" if tot else " -"))
        print(f"{g:10s} " + "".join(f"{c:>16s}" for c in cells))

    fig.suptitle("SAR-only gate (one RF on texture), Bedmap3 as a hard mask — "
                 "no learned context prior", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = args.out or E.ROOT / "data" / "maps" / "gate_grounded_all.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
