---
date: 2026-08-29
topic: Binary-classifier feature search for crevasse pre-gate, and discovering the validated Frangi fix isn't actually wired into production
tags: [fft, glcm, structure-tensor, random-forest, frangi, ridge-detection, sar, speckle, reproducibility, cohens-d, crevasse-labeling]
status: ongoing
difficulty: hard
---

# Lesson: A visual sanity check on "balanced" training tiles uncovered that the validated Frangi fix from two days earlier was never actually reachable from the production script

## Objective

Design a cheap binary classifier (`tile -> features -> Random Forest`) to gate tiles *before* running the expensive multiscale Frangi ridge detector. Needed features that are informative about crevasse-likeness but computable without running Frangi first (a classifier that depends on Frangi's own output isn't a gate, it's just moving the expensive step earlier — caught by the user mid-session). Along the way, tried to fix a severe class imbalance (545 positive / 51 negative) in the labeled tile set used to evaluate candidate features, which is what actually surfaced the real problem.

## Context

- Working from a known-good validated fix from the prior session (`2026-08-27-crevasse-ridge-detection-recall.md`): `ridge_downscale_factor=8` + `ridge_percentile≈70` fixes a speckle-vs-real signal inversion in the Frangi detector.
- The labeled 596-tile set used for feature evaluation this session came entirely from one granule (`025_091`), one ROI (the known braided crevasse field) — negatives were tiles within/adjacent to that ROI, not representative of true negatives elsewhere.
- Ad hoc script convention: scripts prefixed `_name.py`, deleted after confirming output artifacts (pickles/PNGs) are saved.

## The Journey

### Attempt 1: FFT angular-energy-profile feature
**What we tried**: `tile -> FFT -> angular energy profile -> peak/mean ratio` as a classifier feature. Benchmarked cost first (FFT is cheap, especially after downscaling — 8.8-16.4x speedup vs. full-res).
**Why we thought it would work**: On 2 curated reference tiles, peak/mean ratio was 5.16 (positive) vs 1.24 (negative) — looked like a clean signal.
**What happened**: On a real 596-tile sample, no separation at all (2.56 vs 2.60).
**Why it failed**: Unwindowed 2D-FFT edge-discontinuity leakage caused a systematic 0°/90°-axis-aligned dominant-angle artifact in 74.5% of ALL tiles, positive or negative. Hann windowing reduced but didn't eliminate the artifact or restore separation.
**Time spent**: ~45 min (benchmark + full-sample test + windowing A/B).

### Correction (user-caught, not a failed attempt but pivotal): Frangi-derived features are disqualified by construction
**What happened**: After the FFT failure, the natural next idea was "use Frangi's own `ridge_stats` as classifier features." The user immediately flagged this as circular — the classifier's whole point is to run *before* Frangi. Pivoted to GLCM texture + structure-tensor coherence, both computable independently of Frangi.

### Attempt 2: GLCM + structure-tensor features — looked promising, but on a bad label set
**What we tried**: GLCM contrast/homogeneity/energy/correlation (4-angle averaged) + structure-tensor local coherence, evaluated via Cohen's d against the same 596-tile set.
**What happened**: Moderate effect sizes (`glcm_correlation` d=-0.55, `glcm_contrast` d=0.51, `coherence_mean` d=0.40). Looked like real progress.
**Why this needs revisiting**: The 51 "negative" examples this was validated against turned out (see Attempt 4) to not be trustworthy as a representative negative population — they were all near the known crevasse field, and the *positive* labels' own detector wasn't confirmed reproducible either. The Cohen's d numbers may still hold, but the ground truth under them is now suspect.

### Attempt 3: "Balance tiles" via broader sampling — revealed a data-availability problem
**What we tried**: Random-sampled 900 positions across the full `025_091` raster (excluding the known ROI) to find genuinely diverse negatives.
**What happened**: 88.6% nodata. The 103 valid tiles found were mostly still positive (90/103) and clustered right at the edge of the excluded ROI.
**Why it failed**: This granule's actual valid-data swath is narrow and stays close to the known crevasse field — it structurally cannot supply diverse negative terrain, no matter how much you sample it.
**Time spent**: ~20 min (script + background run + inline coordinate inspection).

### Attempt 4: Sampling from two other granules — the real breakthrough (via a visual gut-check, not a metric)
**What we tried**: Coarse reconnaissance confirmed granule `003_064` has a much broader valid-data footprint (97/400 sampled positions valid, spread across 3 quadrants) than `025_091`. Densely sampled it and `004_048` (500 each), ran the full detector for labels, extracted the same GLCM/structure-tensor features.
**What happened**: 132/143 (92%) valid tiles came back positive — worse imbalance, not better. Before trusting these as new training data, rendered a 3x3 grid of the "positive" tiles and looked at them.
**Breakthrough**: They're pure speckle. No crevasses, no ridges, no coherent structure — visually indistinguishable from the "negative" tiles in the same batch. The metric (n_lines>0) said positive; the eyes said noise. **This is the second time in this pipeline's history that a quantitative-looking signal was contradicted by a two-minute visual check** (the first was the original 45°-Hough artifact). The lesson repeats: always look at a sample of what you're about to trust, especially right after a surprising number (92% positive rate should have been the trigger to look sooner).
**Time spent**: ~15 min (visualize script + inspection).

### Root-causing the false positives: three candidate explanations, only one ruled out
**Bug A — checked and ruled out**: My ad hoc scripts this session instantiated `ImprovedEdgeCrevasseDetector` directly with `speckle_filter='ppb_fast'`. The base class's `apply_speckle_filter()` silently no-ops for anything other than `'lee'/'enhanced_lee'/'none'` — real PPB despeckling only happens via the `PPBCrevasseDetector` subclass override. Re-ran the cross-granule check with the correct subclass: results barely changed (despeckled CV 0.0835 → 0.0830). Not the cause.
**Bug B — confirmed real, in checked-in production code**: `build_crevasse_labels.py`'s `detector_params` dict never sets `ridge_percentile` or `ridge_downscale_factor`, and there's no CLI flag for either — meaning the actual production script silently falls back to the class defaults (`ridge_percentile=90.0`, `ridge_downscale_factor=1`), the *pre-fix* settings. This gap was explicitly named as a "Next Step" in the 2026-08-27 session summary and never closed in the two days since. The existing validated pickle's `_8x` filename suffix doesn't even match the current filename-generation code — it must have come from a one-off script outside the checked-in pipeline.
**Bug C — found, not yet resolved**: Calling the detector directly with the exact documented fix params on the *lesson's own reference tiles* did not reproduce the lesson's numbers — it reproduced the *pre-fix* signal inversion (negative reference tile: claimed n=1/0.87% coverage, got n=3/8.47%; positive reference tile: claimed n=8/17.5%, got n=1/1.52%). Possible cause: `FutureWarning`s observed this session (deprecated `remove_small_objects`/`binary_closing`/`major_axis_length` skimage APIs) weren't mentioned in the 08-27 lesson, suggesting a library version bump between then and now that could shift results across the sharp "cliff" the original lesson describes. Not yet confirmed — the original ad hoc validation script was already deleted per convention, so there's no direct diff available.

## Key Lessons

### What We Learned
1. **A surprising aggregate statistic (92% positive rate) is itself a trigger to look at raw examples, not just a number to report.** The visual check that found the real bug took 15 minutes and came *after* several rounds of feature engineering on data that turned out to be mislabeled — doing the visual check first would have saved most of that time.
2. **"Wired into the code" and "reachable from the production entry point" are different claims — verify the second, not just the first.** The 08-27 memory said the fix was "wired into `edge_crevasse_v2.py`," which was true at the constructor-parameter level, but `build_crevasse_labels.py` never passed those parameters through. A constructor supporting an option isn't the same as any caller actually using it.
3. **A "resolved" investigation from a prior session can silently stop reproducing** if the environment changes underneath it (e.g., a library version bump) and there's no regression test pinning the reference-tile behavior. The 08-27 lesson's two reference tiles (`r44000/c26000`, `r52000/c24000`) were validation evidence, not a permanent regression check — nothing re-ran them until this session, two days later, by accident.
4. Restated from 08-27, confirmed again here: a classifier feature or label that looks clean on a curated 2-tile sample needs a real-distribution test before being trusted — this happened twice more this session (FFT angular ratio; the 003_064/004_048 "positive" tiles).

### What Didn't Work and Why
- FFT angular-profile ratio → unwindowed FFT edge leakage creates a label-independent 0°/90° artifact; Hann windowing helps but doesn't fix it.
- Using Frangi's own outputs as classifier features → architecturally circular given the gating use case (user-caught, not empirically tested).
- Fixing imbalance via denser in-granule random sampling → the granule's valid-data footprint is geographically narrow and doesn't extend to diverse terrain, no amount of sampling fixes that.
- Trusting the cross-granule sample without a visual check first → would have propagated mislabeled speckle into the classifier's training data.

### What To Do Next Time
1. When a detection/labeling pipeline is producing an unexpectedly high or low positive rate on new data, render a handful of examples before doing any further feature engineering on top of those labels.
2. When resuming or extending a previously "validated" pipeline result, re-run its original reference-case check first, especially if any dependency (library versions, environment) could have changed since — don't assume yesterday's validated numbers still hold today.
3. Before extending a fix into "pipeline-wide default," grep the actual production entry point's argument-parsing and default dict, don't just confirm the underlying function/class supports the parameter.

### Patterns Recognized
- This is the second time visual inspection has overturned a metric-driven conclusion in this exact pipeline (first: 45°-Hough artifact on speckle; now: Frangi false positives on speckle from new granules). Visual spot-checks earn their cost repeatedly in this specific SAR/speckle domain — worth defaulting to "render a few examples" as step 1 whenever a new granule or ROI is introduced, not step 5.
- "Validated two days ago" is a weaker guarantee than it sounds when the validation depended on specific library versions and an ad hoc script that no longer exists to diff against.

## When to Apply This Lesson

**Similar problems**:
- Extending any detector/classifier validated on one region/dataset to a new region/dataset — re-run the original reference-case sanity check before trusting new labels, don't just apply the same code and assume it transfers.
- Any pipeline where a "fix" was validated via an ad hoc script that was then deleted (per this project's convention) — the fix's default wiring into the production entry point needs its own explicit check, since ad hoc validation and production usage can silently diverge.
- Whenever a metric-based label set produces a surprising class-imbalance shift after a change (sampling strategy, new data source, etc.) — visualize before trusting.

**Keywords**: FFT windowing artifact, GLCM texture features, structure tensor coherence, Cohen's d, class imbalance, Frangi ridge detector, reproducibility drift, skimage version, production defaults vs validated defaults, visual sanity check

## References
- Files touched: none in production code this session (investigation only)
- Related lessons: [[2026-08-27-crevasse-ridge-detection-recall]] (the original fix this session tried to build on, and whose reference-tile numbers no longer reproduce)
- Related memory: `ridge_detection_pipeline_redesign.md`, `edge_pipeline_45deg_artifact.md` (the original visual-check-beats-metric precedent), `unet_pipeline_review.md`
- Key data: reference tiles r44000_c26000 (positive) and r52000_c24000 (negative) in granule `025_091`, from the 08-27 lesson
- Output artifacts kept: `data/improved_detections_025_091_braided/fft_classifier_feature_check_results.pkl`, `classifier_feature_candidates_results.pkl`, `balanced_tiles_broad_sample.pkl`, `diverse_negatives_003_064_004_048.pkl`, `new_positives_sanity_check.png`, `new_negatives_sanity_check.png`

## Time Investment
- Total time: ~3 hours across FFT feature dead-end, GLCM/structure-tensor feature search, balance-tiles sampling attempts, and the bug hunt
- Could have been: ~1.5-2 hours if the visual sanity check on the cross-granule sample had been done immediately after seeing the 92% positive rate, rather than after computing full feature sets on it
- Savings for next time: default to rendering a handful of examples immediately after any surprising aggregate statistic, before further downstream computation

---
*Captured from conversation on 2026-08-29*
