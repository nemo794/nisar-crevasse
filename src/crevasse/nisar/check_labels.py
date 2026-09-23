"""Acceptance test for a training set: run this after rsync, before training.

Every check prints its measurement AND the bound it is judged against, because a
number without its control is not readable. Failures are listed at the end; exit
code is nonzero if any FAIL fires.

WHAT THIS IS FOR. Three of the four things that have gone wrong at this stage were
silent, not crashes: a filter that kept 0 of 510 tiles and trained on nothing; a
train/val split that put adjacent tiles on both sides; and an augmentation kwarg that
albumentations 2.x renamed, so GaussNoise ran at its default std_range=(0.2, 0.44) --
noise at half the dynamic range of a [0,1] tile -- while the code read as if it were
asking for 0.03. None of those raise. All three are checked here.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from crevasse.nisar.label_io import load_labels

RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def load_shard(path):
    z = np.load(path, allow_pickle=False)
    return dict(
        targets=z["targets"], positions=z["positions"].astype(np.int64),
        coverage=z["coverage"], orient_conc=z["orient_conc"],
        tile_size=int(z["tile_size"]), soft_rect_t=float(z["soft_rect_t"]),
        ridge_factors=(tuple(int(f) for f in z["ridge_factors"])
                       if "ridge_factors" in z.files else None),
        images=z["images"], n=len(z["positions"]),
    )


def is_shard_npz(path):
    """Both a packed shard and a raw label file are `.npz` now (label_io.py replaced
    the old label pickle) -- disambiguate by key set rather than extension."""
    return "images" in np.load(path, allow_pickle=False).files


def load_labels_npz(path):
    R = load_labels(path)
    return dict(
        targets=np.stack([r["edge_mask"] for r in R]),
        positions=np.array([(r["row"], r["col"]) for r in R], dtype=np.int64),
        coverage=np.array([r["edge_coverage"] for r in R], dtype=np.float32),
        orient_conc=np.array([r.get("orient_conc", np.nan) for r in R], dtype=np.float32),
        tile_size=int(R[0]["tile_size"]), soft_rect_t=float(R[0].get("soft_rect_t") or np.nan),
        ridge_factors=R[0].get("ridge_factors"),
        images=None, n=len(R),
    )


def check_labels(d, orient_gate):
    ts = d["tile_size"]
    t = d["targets"].astype(np.float32)

    check("tile count", d["n"] > 0, f"{d['n']} tiles (need > 0)")
    check("mask shape matches tile", t.shape[1:] == (ts, ts),
          f"{t.shape[1:]} vs tile_size {ts} -- a mismatch is silently misaligned by "
          f"the augmentation pipeline, not an error")
    check("no NaN in targets", not np.isnan(t).any(),
          f"{int(np.isnan(t).sum())} NaN (need 0; BCE would return nan loss)")
    check("targets in [0,1]", t.min() >= 0.0 and t.max() <= 1.0,
          f"[{t.min():.4f}, {t.max():.4f}] (BCEWithLogitsLoss requires [0,1])")

    # Bound unchanged at 5% for the 2026-09-10 mean{4,2,1} rule: it averages three
    # soft() maps, so if anything it is MORE continuous than soft*rect was. Only the
    # wording changed -- soft_rect_t is nan under that rule, so it cannot be the
    # explanation for a binary-looking target any more.
    interior = float(((t > 0) & (t < 1)).mean())
    check("target is genuinely continuous", interior > 0.05,
          f"{interior*100:.2f}% of pixels strictly between 0 and 1 (need > 5%; a soft "
          f"target that is really binary means the soft rule never took effect)")

    cov = d["coverage"]
    check("no all-empty tiles", (cov > 0).all(),
          f"{int((cov == 0).sum())} tiles with 0 coverage at >0.2 (need 0; an empty "
          f"target teaches the model to predict nothing)")
    print(f"         coverage>0.2: mean {cov.mean()*100:.2f}%  "
          f"p10 {np.percentile(cov,10)*100:.2f}%  p50 {np.percentile(cov,50)*100:.2f}%  "
          f"p90 {np.percentile(cov,90)*100:.2f}%  max {cov.max()*100:.2f}%")

    oc = d["orient_conc"]
    if np.isfinite(oc).any():
        check("orientation gate respected", np.nanmin(oc) >= orient_gate - 1e-6,
              f"min orient_conc {np.nanmin(oc):.4f} vs gate {orient_gate} (a tile below "
              f"the gate should never have reached the label file)")

    pos = d["positions"]
    uniq = len({tuple(p) for p in pos})
    check("no duplicate tile positions", uniq == len(pos),
          f"{len(pos) - uniq} duplicates (need 0; a repeated tile is silent "
          f"oversampling)")

    rf = d.get("ridge_factors")
    if rf and len(rf) > 1:
        # Under mean_s soft(resp_s) the target's MAGNITUDE counts how many scales agree,
        # so mass just above the 0.2 cut is single-scale-only structure -- the population
        # the dropped orientation factor used to remove. REPORTED, not bounded: a bound
        # fitted to this rule would only agree with whatever it is handed, and the
        # far-FP recovery it trades against is the test, not an objective.
        one = 1.0 / len(rf) + 1e-6
        seen = t[t > 0.2]
        frac = float((seen <= one).mean()) * 100 if seen.size else float("nan")
        print(f"         target rule = mean soft over f{rf}, no orientation factor")
        print(f"         of the labelled area, {frac:.1f}% is single-scale-only "
              f"(<= {one:.3f}); REF_NEG sits at 9.78% coverage under this rule vs 0.53% "
              f"under f4 soft*rect -- the accepted risk, priced by the training run")
    else:
        print(f"         target rule = f{rf or '?'} soft x rect(orient_agree, "
              f"{d['soft_rect_t']})")


def check_split(d):
    from crevasse.nisar.sar_dataset import split_indices
    tr, va = split_indices(d["positions"], 0.8, d["tile_size"],
                           "spatial", 1, 42)
    pos = d["positions"]
    rows, cols = pos[:, 0], pos[:, 1]
    axis = 0 if (rows.max() - rows.min()) >= (cols.max() - cols.min()) else 1
    coord = pos[:, axis]
    gap = int(coord[va].min() - coord[tr].max())
    check("train/val spatially separated", gap >= d["tile_size"],
          f"{gap} px between the closest train and val tile (need >= {d['tile_size']}, "
          f"one tile; overlapping bands are the leaky split this replaced)")
    check("both split sides non-empty", len(tr) > 0 and len(va) > 0,
          f"train {len(tr)}, val {len(va)}")


def check_augmentation():
    """The GaussNoise kwarg rename is the silent failure this pins."""
    import warnings
    import albumentations as A
    print(f"         albumentations {A.__version__}")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from crevasse.nisar.sar_dataset import get_train_transforms
        get_train_transforms(512)
    bad = [str(x.message) for x in w if "not valid for transform" in str(x.message)]
    check("no dropped augmentation kwargs", not bad,
          f"{bad if bad else 'none'} -- albumentations silently IGNORES unknown kwargs "
          f"and falls back to its own defaults, so a rename reads as working code")

    flat = np.full((256, 256), 0.5, np.float32)
    noisy = A.Compose([A.GaussNoise(std_range=(0.032, 0.10), p=1.0)])(image=flat)["image"]
    std = float(noisy.std())
    check("gaussian noise magnitude sane", 0.01 < std < 0.15,
          f"std {std:.4f} on a flat 0.5 image (need 0.01-0.15; the 2.x default "
          f"std_range=(0.2,0.44) would bury the speckle texture)")


def check_batch(d, shard_path):
    from crevasse.nisar.sar_dataset import create_shard_dataloaders
    tr, va = create_shard_dataloaders(shard_path, batch_size=4, num_workers=0)
    for name, dl in (("train", tr), ("val", va)):
        b = next(iter(dl))
        i, m = b["image"], b["mask"]
        ts = d["tile_size"]
        ok = (tuple(i.shape) == (4, 1, ts, ts) and tuple(m.shape) == (4, 1, ts, ts)
              and float(m.min()) >= 0 and float(m.max()) <= 1
              and 0.0 <= float(i.min()) and float(i.max()) <= 1.0)
        check(f"{name} batch well-formed", ok,
              f"image {tuple(i.shape)} [{i.min():.3f},{i.max():.3f}]  "
              f"mask {tuple(m.shape)} [{m.min():.3f},{m.max():.3f}]  "
              f"(need (4,1,{ts},{ts}) and both in [0,1])")


def compare(shard, labels):
    """A shard is only trustworthy if its targets equal the label file's to float16."""
    ok_n = shard["n"] == labels["n"]
    check("shard tile count matches labels", ok_n, f"{shard['n']} vs {labels['n']}")
    if not ok_n:
        return
    same_pos = (shard["positions"] == labels["positions"]).all()
    check("shard positions match labels", same_pos,
          "identical (row, col) in identical order" if same_pos else "MISMATCH")
    err = float(np.abs(shard["targets"].astype(np.float32)
                       - labels["targets"].astype(np.float32)).max())
    check("shard targets match labels", err < 1e-3,
          f"max abs diff {err:.2e} (need < 1e-3; float16 resolves ~5e-4 in [0,1])")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", help="a packed .npz shard or a raw label .npz (label_io.py)")
    ap.add_argument("--compare-labels", default=None,
                    help="when checking a shard, also verify it against its source "
                         "label .npz")
    ap.add_argument("--orient-gate", type=float, default=0.12,
                    help="the orient_conc gate the labels were generated behind")
    args = ap.parse_args()

    is_shard = is_shard_npz(args.input)
    print(f"\n=== {Path(args.input).name} ({'shard' if is_shard else 'labels'})\n")
    d = load_shard(args.input) if is_shard else load_labels_npz(args.input)

    print("LABELS")
    check_labels(d, args.orient_gate)
    print("\nSPLIT")
    check_split(d)
    print("\nAUGMENTATION")
    check_augmentation()
    if is_shard:
        print("\nDATALOADER")
        check_batch(d, args.input)
    if args.compare_labels:
        print("\nSHARD vs LABELS")
        compare(d, load_labels_npz(args.compare_labels))

    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{'=' * 70}")
    if failed:
        print(f"{len(failed)}/{len(RESULTS)} checks FAILED:")
        for n in failed:
            print(f"  - {n}")
        sys.exit(1)
    print(f"All {len(RESULTS)} checks passed. Data is fit for training.")


if __name__ == "__main__":
    main()
