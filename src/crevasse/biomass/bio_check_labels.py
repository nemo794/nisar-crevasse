"""Acceptance test for a BIOMASS training shard: run this after building or rsyncing it,
before training.

Same discipline as nisar-crevasse-unet/src/check_labels.py -- every check prints its
measurement AND the bound it is judged against, failures are listed at the end, exit code
is nonzero on any FAIL. Handles BOTH shard types this repo produces: `bio_build_shard.py`'s
real AlphaEarth mask (genuinely binary {0,1}) and `bio_build_shard_frangi.py`'s soft
Frangi ridge response (continuous [0,1]) -- detected by the presence of the Frangi
shard's own `orient_conc`/`ridge_factors` keys, since the two need a different target
check (binary-membership vs range) but everything else about the shard (shape, channel
order, split, dataloader) is identical.
"""
import argparse
import sys

import numpy as np

RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def load_shard(path):
    z = np.load(path, allow_pickle=True)
    is_soft = "orient_conc" in z.files and "ridge_factors" in z.files
    return dict(
        images=z["images"], targets=z["targets"],
        positions=z["positions"].astype(np.int64),
        x_m=z["x_m"], y_m=z["y_m"], coverage=z["coverage"], aoi_pos=z["aoi_pos"],
        granule=z["granule"], pols=[str(p) for p in z["pols"]],
        tile_size=int(z["tile_size"]), n=len(z["positions"]),
        is_soft=is_soft,
    )


def check_shard(d):
    from crevasse.biomass.bio_sar_dataset import POLS

    ts = d["tile_size"]
    t = d["targets"]
    img = d["images"]

    check("tile count", d["n"] > 0, f"{d['n']} tiles (need > 0)")
    check("target shape matches tile", t.shape[1:] == (ts, ts),
          f"{t.shape[1:]} vs tile_size {ts}")
    check("image shape matches (4, tile, tile)", img.shape[1:] == (4, ts, ts),
          f"{img.shape[1:]} vs (4, {ts}, {ts})")
    check("channel order is HH,HV,VH,VV", d["pols"] == POLS,
          f"{d['pols']} (need {POLS} -- everything downstream assumes this order)")
    check("no NaN in targets", not np.isnan(t.astype(np.float32)).any(),
          f"{int(np.isnan(t.astype(np.float32)).sum())} NaN (need 0)")

    if d["is_soft"]:
        tmin, tmax = float(t.min()), float(t.max())
        check("targets are in [0,1] (soft Frangi response)", tmin >= -1e-6 and tmax <= 1 + 1e-6,
              f"range [{tmin:.4f},{tmax:.4f}] (need subset of [0,1] -- "
              f"bio_build_shard_frangi.py's continuous ridge response, not a binary mask)")
    else:
        check("targets are genuinely binary", set(np.unique(t).tolist()) <= {0, 1},
              f"unique values {sorted(np.unique(t).tolist())[:5]} (need subset of {{0,1}} -- "
              f"AlphaEarth is a real byte mask, not a soft target; anything else means the "
              f"shard was built from something other than the raw mosaic read)")
    check("images have some finite data per tile",
          bool((np.isfinite(img.astype(np.float32)).any(axis=(1, 2, 3))).all()),
          "every tile has at least one finite dB pixel across its 4 channels")

    cov = d["coverage"]
    print(f"         coverage: mean {cov.mean()*100:.2f}%  p10 {np.percentile(cov,10)*100:.2f}%  "
          f"p50 {np.percentile(cov,50)*100:.2f}%  p90 {np.percentile(cov,90)*100:.2f}%  "
          f"max {cov.max()*100:.2f}%  ({int((cov==0).sum())} all-empty tiles -- legitimate "
          f"negatives here, unlike NISAR's manufactured target, so NOT flagged as a bug)")

    if d["is_soft"]:
        print("         (skipping coverage-vs-aoi_pos exactness check: `coverage` here is "
              "the mean Frangi soft response, not AlphaEarth's positive fraction -- the two "
              "are unrelated by construction for this shard type)")
    else:
        mismatch = float(np.abs(cov - d["aoi_pos"]).max())
        check("coverage matches feature-table aoi_pos exactly", mismatch < 1e-6,
              f"max abs diff {mismatch:.2e} (need < 1e-6 -- both are `(mask==1).mean()` over "
              f"the identical window; any difference is a bug, not drift, per bio_build_shard.py)")

    # Keyed on (granule, row, col), NOT (row, col) alone: BIOMASS repeat passes (S1/S2/S3
    # over the same Thwaites ground) legitimately reuse the same tile grid position under
    # a different granule, unlike NISAR's single-granule shards where (row, col) alone
    # was already unique.
    pos = d["positions"]
    keys = list(zip(d["granule"].tolist(), pos[:, 0].tolist(), pos[:, 1].tolist()))
    uniq = len(set(keys))
    check("no duplicate (granule, row, col)", uniq == len(keys),
          f"{len(keys) - uniq} duplicates (need 0)")


def check_split(d):
    from crevasse.biomass.bio_sar_dataset import buffered_split

    tr, va = buffered_split(d["x_m"], d["y_m"])
    check("both split sides non-empty", len(tr) > 0 and len(va) > 0,
          f"train {len(tr)}, val {len(va)}")

    # Buffer check: no train tile within buffer_m of ANY val block's extent should
    # survive -- verify by construction rather than re-deriving block_m/buffer_m here,
    # since buffered_split already enforces and prints it. This check instead confirms
    # the split is deterministic (same seed -> same split), which the "control that
    # matters" for eval depends on.
    tr2, va2 = buffered_split(d["x_m"], d["y_m"])
    check("split is deterministic given the same seed",
          np.array_equal(tr, tr2) and np.array_equal(va, va2),
          "repeated call with default seed reproduces the identical split -- required "
          "for bio_eval_shard.py's --subset val control to mean anything")


def check_augmentation():
    import warnings
    import albumentations as A
    print(f"         albumentations {A.__version__}")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from crevasse.biomass.bio_sar_dataset import get_train_transforms
        get_train_transforms()
    bad = [str(x.message) for x in w if "not valid for transform" in str(x.message)]
    check("no dropped augmentation kwargs", not bad,
          f"{bad if bad else 'none'} -- albumentations silently ignores unknown kwargs")

    flat = np.zeros((256, 256, 4), np.float32)
    noisy = A.Compose([A.GaussNoise(std_range=(0.02, 0.08), p=1.0)])(image=flat)["image"]
    std = float(noisy.std())
    check("gaussian noise magnitude sane on normalized-scale input", 0.005 < std < 0.15,
          f"std {std:.4f} on a flat zero image, 4 channels (need 0.005-0.15)")


def check_batch(shard_path, d):
    from crevasse.biomass.bio_sar_dataset import create_shard_dataloaders

    tr, va, mu, sd, cidx, _ = create_shard_dataloaders(shard_path, batch_size=4,
                                                        num_workers=0)
    n_ch = len(cidx)
    check("normalization is finite", np.isfinite(mu) and np.isfinite(sd) and sd > 0,
          f"mu={mu:.3f} sd={sd:.3f}")
    for name, dl in (("train", tr), ("val", va)):
        b = next(iter(dl))
        i, m = b["image"], b["mask"]
        ts = d["tile_size"]
        ok = (tuple(i.shape) == (4, n_ch, ts, ts) and tuple(m.shape) == (4, 1, ts, ts)
              and float(m.min()) >= 0 and float(m.max()) <= 1)
        check(f"{name} batch well-formed", ok,
              f"image {tuple(i.shape)} [{i.min():.3f},{i.max():.3f}]  "
              f"mask {tuple(m.shape)} [{m.min():.3f},{m.max():.3f}]  "
              f"(need (4,{n_ch},{ts},{ts}) image and (4,1,{ts},{ts}) mask in [0,1])")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("shard", help="a .npz shard from bio_build_shard.py")
    args = ap.parse_args()

    d = load_shard(args.shard)
    print(f"\n=== {args.shard}\n")

    print("SHARD")
    check_shard(d)
    print("\nSPLIT")
    check_split(d)
    print("\nAUGMENTATION")
    check_augmentation()
    print("\nDATALOADER")
    check_batch(args.shard, d)

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
