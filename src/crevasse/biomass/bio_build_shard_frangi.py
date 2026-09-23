"""Build a BIOMASS training shard with Frangi-derived soft targets, instead of the
AlphaEarth pixel mask `bio_build_shard.py` uses.

    conda run -n biomass python src/bio_build_shard_frangi.py \
        --features ../biomass-crevasse-gate/data/bio_tile_features_t512.npz \
        --base ../biomass/data/biomass \
        --out data/train_biomass_frangi_v1.npz

WHY THIS EXISTS, GIVEN docs/PIPELINE.md SAYS FRANGI ISN'T NEEDED HERE
------------------------------------------------------------------------
`bio_build_shard.py`'s target (the real AlphaEarth pixel mask) only tells you WHERE a
tile reads as crevassed at a coarse, cross-sensor, embedding-classifier resolution -- it
was never meant to trace individual crevasse SHAPE. A live investigation (2026-09-16,
see docs/PIPELINE.md's Frangi section) tested whether a ridge filter can recover
per-pixel crevasse shape on this sensor and found the honest answer is "not cleanly":
every setting swept (4 raw polarizations, cross/ratio merges, both averaging orders, 2
despeckle filters, scales f1/f2/f4/f8) converges on a confound tied to BIOMASS's
documented ~3:1 anisotropic point-spread function, confirmed three independent ways
(the response's dominant angle matches the PSF's ~50/140 deg axis; it is nearly
channel-invariant; it collapses when a common-mode-canceling ratio is used). So this is
a SOFT, approximate label source, explicitly not required to be perfect -- the U-Net is
expected to average out what Frangi can't cleanly separate, the same tolerance NISAR's
own soft-label pipeline was built around.

THE RECIPE, AND WHY EACH PIECE IS THERE
-----------------------------------------
Mirrors `biomass/code/build_crevasse_labels.py`'s two-gate structure:

  1. **`BiomassGate` (RF) at P >= 0.65** runs BEFORE Frangi, exactly NISAR's own
     gate-before-Frangi order. This is NOT the NISAR RF gate (that one is documented
     to be chance-level on BIOMASS) -- it's BIOMASS's own radiometric gate, verified
     (2026-09-16, n=80 real tiles) to separate genuine crevasse ground from the PSF
     artifact with corr(gate P, AlphaEarth coverage)=0.959, including correctly
     rejecting the exact tile whose Frangi response we later proved sits at the PSF's
     own 52-degree axis.
  2. **`avg(Frangi(HH), Frangi(HV))`** -- Frangi run separately on HH and HV, THEN the
     two RESPONSES averaged (not the images first -- a different, noisier operation,
     see docs/PIPELINE.md's order-of-operations check). HH+HV only, not all 4
     polarizations: measured to give statistically indistinguishable results from the
     4-band average (orientation concentration within 0.02-0.05 of each other across
     every setting tested) at half the compute.
  3. **`ppb_fast` despeckle**, NISAR's real production filter (not the `enhanced_lee`
     stand-in used for most of the exploration). Its extra speckle relative to a
     heavier despeckler is accepted deliberately -- the U-Net is expected to learn past
     it, the same tolerance applied to using an imperfect label source at all.
  4. **Averaged across ridge_downscale_factor {1, 2, 4}** -- more sensitive to finer
     structure than f4 alone, at the cost of more fragments (every multiscale-average
     test showed 2-4x the components of any single scale). Accepted for the same
     reason as (3): this is a soft label, not a metric to be quoted.
  5. **Tile-level orientation gate at 0.12** (NISAR's own default) -- if the FINAL
     (post scale-averaging) response's `orientation_concentration` is below this, the
     whole tile's target is zeroed rather than kept. This is the simple, tile-level
     "if it's noise, return an empty mask" gate -- NOT NISAR's per-pixel
     `rect(orient_agree, t)` soft-target rectifier, which was calibrated against a
     specific anti-correlation in NISAR's per-tile contrast-stretch preprocessing that
     does not apply here (BIOMASS chips are raw dB with no per-tile stretch, by
     `bio_tile_cache.py`'s own design). Keeping the gate simple rather than porting
     that rectifier unvalidated.

WHAT IS NOT DONE HERE, ON PURPOSE, AND SHOULD BE DONE BEFORE THIS IS TRUSTED
-------------------------------------------------------------------------------
The raw Frangi response has no calibrated soft-target scaling the way NISAR's
`soft_labels.HI`/`LO` do (those were measured off a 341-tile calibration sample with a
known REF_POS/REF_NEG anchor check). This script instead computes a single global
`response_p99` from whatever tiles it actually processes and divides by it -- a
placeholder normalization, not a validated calibration. `targets_raw` (the un-normalized
response) is also kept in the shard so a real calibration can be redone later without
re-running Frangi. Treat any absolute coverage number from this shard as provisional
until that calibration exists.
"""
import argparse
import glob
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from crevasse.biomass.bio_predict import BiomassGate
from crevasse.biomass.bio_tile_cache import granule_chips, POLS, T
from crevasse.biomass.multiscale_labels import make_detector
from crevasse.biomass.edge_crevasse_v2 import ImprovedEdgeCrevasseDetector

GATE_THRESH = 0.65
ORIENT_GATE = 0.12
FACTORS = (1, 2, 4)
BANDS = ("HH", "HV")
SPECKLE = "ppb_fast"
RIDGE_PCT = 75.0
DETECTOR_PARAMS = dict(ridge_sigmas=range(2, 12, 2), ridge_percentile=RIDGE_PCT,
                        orient_gate_thresh=None,  # applied by hand below, on the FINAL response
                        min_ridge_size=200.0, min_eccentricity=0.85)

_ORIENT_DETECTOR = ImprovedEdgeCrevasseDetector(speckle_filter="none",
                                                ridge_downscale_factor=4, **DETECTOR_PARAMS)


def db_to_lin(db):
    return 10 ** (np.asarray(db, dtype=np.float64) / 10.0)


def frangi_soft_response(hh_lin, hv_lin):
    """avg(Frangi(HH), Frangi(HV)), averaged again over FACTORS, component-cleaned,
    orientation-gated.

    `clean_ridge_mask` computes a binary mask that has already been thresholded at
    `ridge_percentile` AND filtered by `min_ridge_size`/`min_eccentricity` (rejects
    small, blob-shaped, non-elongated components -- exactly the speckle-noise rejection
    this whole pipeline depends on). An earlier version of this function discarded that
    mask and kept only its summary stats, so the saved target was the raw, UNFILTERED
    response -- every blobby noise fragment the size/eccentricity filter exists to
    reject survived into the label. Fixed by multiplying the continuous response by the
    cleaned binary mask before returning it: a pixel only keeps a nonzero soft value if
    it belongs to a component that actually passed the elongation/size test.
    """
    per_factor = []
    for fac in FACTORS:
        sub = []
        for band_img in (hh_lin, hv_lin):
            det = make_detector(SPECKLE, dict(DETECTOR_PARAMS, ridge_downscale_factor=fac))
            r = det.process_tile(np.nan_to_num(band_img, nan=0.0))
            sub.append(r["ridge_response"])
        per_factor.append(np.mean(sub, axis=0))
    response = np.mean(per_factor, axis=0)

    orient_conc = _ORIENT_DETECTOR.orientation_concentration(response)
    cleaned, stats = _ORIENT_DETECTOR.clean_ridge_mask(response)
    response = response * cleaned.astype(response.dtype)

    if orient_conc < ORIENT_GATE:
        response = np.zeros_like(response)

    return response, {"orient_conc": float(orient_conc), "n_lines": int(stats["n_components"])}


def _frangi_worker(pair):
    """Top-level (picklable) wrapper for multiprocessing.Pool -- the Frangi step is the
    dominant cost (~12s/gate-passed tile measured, single-threaded: 2 bands x 3 scales
    of ppb_fast + Frangi each), so it is the one part of this script worth parallelizing.
    A bare Pool call would need the `if __name__ == "__main__":` guard this repo already
    uses -- see biomass/lessons/ for the standing macOS trap where a top-level (unguarded)
    Pool call recurses and hangs forever.
    """
    hh, hv = pair
    return frangi_soft_response(hh, hv)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True,
                     help="bio_tile_features_t512.npz from biomass-crevasse-gate")
    ap.add_argument("--base", required=True, help="directory holding granule directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tiles", type=int, default=None,
                     help="cap on tiles, a seeded random sample -- for a quick "
                          "correctness check before the real run. Omit for every "
                          "candidate tile in --features.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=None,
                     help="Frangi is the dominant cost (~12s/gate-passed tile "
                          "single-threaded); defaults to cpu_count()-1.")
    args = ap.parse_args()
    num_workers = args.num_workers or max(1, cpu_count() - 1)

    d = np.load(args.features, allow_pickle=True)
    keep = np.isfinite(d["X"]).all(axis=1)   # every scanned tile, NOT just aoi_inbox --
    # Frangi needs no AlphaEarth label, so restricting to aoi_inbox here would throw
    # away real ground this label source can actually score.
    idx = np.flatnonzero(keep)
    print(f"{len(idx)}/{len(keep)} scanned tiles have finite features")
    if args.max_tiles is not None and len(idx) > args.max_tiles:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(idx, size=args.max_tiles, replace=False))
        print(f"  capped to {len(idx)} tiles (seed={args.seed})")

    gran = d["granule"][idx]
    row, col = d["row"][idx].astype(np.int64), d["col"][idx].astype(np.int64)
    xm, ym = d["x_m"][idx], d["y_m"][idx]
    aoi_pos = d["aoi_pos"][idx]   # NaN where unlabelled -- carried through for comparison
                                  # only, never used to gate or filter here

    # 1. cache the 4-pol dB stacks, same reusable path bio_build_shard.py uses --
    # parallelized across granules, same pattern bio_tile_cache.py's own main() uses.
    images = np.full((len(idx), len(POLS), T, T), np.nan, np.float16)
    gs = sorted(set(gran.tolist()))
    jobs, slots = [], []
    for g in gs:
        sel = np.where(gran == g)[0]
        jobs.append((os.path.join(args.base, g), row[sel], col[sel], T))
        slots.append(sel)
    with Pool(min(num_workers, len(jobs))) as pool:
        for g, sel, part in zip(gs, slots, pool.imap(granule_chips, jobs)):
            images[sel] = part
            print(f"  cached {g}: {len(sel)} tiles", flush=True)

    # 2. gate BEFORE Frangi -- chunked, and NEVER materializing a full-dataset linear
    # copy. `images` (dB) is already ~10GB at N=4767; a one-shot
    # `db_to_lin(images.astype(np.float32))` over the whole array needs the fp32 astype
    # temp (~20GB) AND its fp32 output (~20GB) alive at once on top of that -- ~50GB
    # before `BiomassGate.predict_proba` (which does its own internal float64 cast,
    # another ~40GB) is even called. That combination OOM-killed a laptop twice. Fix:
    # convert dB->linear per GATE_CHUNK-sized slice, score it, discard it -- peak memory
    # is bounded by chunk size, not N, for this whole step.
    gate = BiomassGate()
    GATE_CHUNK = 200
    gate_prob = np.full(len(idx), np.nan, dtype=np.float32)
    for s in range(0, len(idx), GATE_CHUNK):
        e = min(s + GATE_CHUNK, len(idx))
        lin_chunk = db_to_lin(images[s:e].astype(np.float32))
        gate_prob[s:e] = gate.predict_proba(lin_chunk)["rf"]
        print(f"  gate {e}/{len(idx)} tiles scored", flush=True)
    passed = np.isfinite(gate_prob) & (gate_prob >= GATE_THRESH)
    print(f"gate: {int(passed.sum())}/{len(idx)} tiles pass P>={GATE_THRESH}")

    # 3. Frangi only on gate survivors -- same reasoning: convert dB->linear only for
    # the (far fewer) gate-passed tiles, not the full N, and only the 2 bands Frangi
    # actually uses (HH, HV), not all 4 polarizations.
    targets_raw = np.zeros((len(idx), T, T), dtype=np.float32)
    orient_conc = np.full(len(idx), np.nan, dtype=np.float32)
    n_lines = np.zeros(len(idx), dtype=np.int32)

    pass_idx = np.flatnonzero(passed)
    pairs = [(db_to_lin(images[i, 0].astype(np.float32)),
              db_to_lin(images[i, 1].astype(np.float32))) for i in pass_idx]
    print(f"Frangi: {len(pairs)} gate-passed tiles, {num_workers} workers "
          f"(~12s/tile single-threaded -> ~{len(pairs) * 12 / max(num_workers, 1) / 60:.0f} "
          f"min estimated)", flush=True)
    with Pool(num_workers) as pool:
        for j, (i, (resp, stats)) in enumerate(zip(pass_idx, pool.imap(_frangi_worker, pairs))):
            targets_raw[i] = resp
            orient_conc[i] = stats["orient_conc"]
            n_lines[i] = stats["n_lines"]
            if (j + 1) % 25 == 0:
                print(f"  frangi {j + 1}/{len(pairs)} gate-passed tiles processed", flush=True)

    emptied = int(((orient_conc < ORIENT_GATE) & passed).sum())
    print(f"orientation gate emptied {emptied}/{int(passed.sum())} gate-passed tiles "
          f"(orient_conc < {ORIENT_GATE})")

    # 4. provisional soft-target scaling -- see module docstring. NOT a validated
    # calibration; kept as a shard-level scalar so it can be redone without rerunning
    # Frangi.
    nonzero = targets_raw[targets_raw > 0]
    response_p99 = float(np.percentile(nonzero, 99)) if nonzero.size else 1.0
    targets_soft = np.clip(targets_raw / max(response_p99, 1e-9), 0, 1).astype(np.float16)

    coverage = (targets_soft.astype(np.float32) > 0.2).mean(axis=(1, 2))
    print(f"response_p99={response_p99:.4f} (provisional normalization constant)")
    print(f"coverage>0.2: mean {coverage.mean()*100:.2f}%  "
          f"nonzero tiles {int((coverage > 0).sum())}/{len(idx)}")

    np.savez(args.out,
             images=images, targets=targets_soft, targets_raw=targets_raw.astype(np.float16),
             positions=np.stack([row, col], axis=1),
             x_m=xm.astype(np.float64), y_m=ym.astype(np.float64),
             coverage=coverage, gate_prob=gate_prob.astype(np.float32),
             orient_conc=orient_conc, n_lines=n_lines, aoi_pos=aoi_pos.astype(np.float32),
             granule=gran, pols=np.array(POLS), tile_size=np.int32(T),
             gate_thresh=np.float32(GATE_THRESH), orient_gate=np.float32(ORIENT_GATE),
             ridge_factors=np.array(FACTORS, dtype=np.int32),
             ridge_bands=np.array(BANDS), speckle_filter=SPECKLE,
             ridge_percentile=np.float32(RIDGE_PCT), response_p99=np.float32(response_p99))
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
