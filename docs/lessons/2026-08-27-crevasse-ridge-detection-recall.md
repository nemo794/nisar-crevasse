---
date: 2026-08-27
topic: Fixing recall on the Frangi ridge-based crevasse detector without reintroducing speckle false positives
tags: [frangi, ridge-detection, sar, speckle, downscaling, multi-look, thresholding, crevasse-labeling]
status: resolved
difficulty: hard
---

# Lesson: The recall ceiling on the Frangi crevasse detector wasn't a threshold problem — it was a signal problem, and averaging (not thresholding) was the fix

## Objective

The pseudo-labeling pipeline for crevasse detection (`edge_crevasse_v2.py` + `build_crevasse_labels.py`) uses a multiscale Frangi ridge/vesselness filter, thresholded per-tile at a percentile of the nonzero response, to turn continuous ridge response into a binary crevasse-candidate mask. On a confirmed real crevasse field (NISAR granule `025_091`, "braided region" ROI), the default settings (`ridge_percentile=90`) were badly under-detecting — user feedback: the mask "is removing most of them," visually confirmed against a tile where obvious curved dark/bright banding covered far more of the tile than the ~2.5% the detector was flagging. Goal: recover that missing recall without reintroducing the false positives on pure-speckle tiles that the percentile threshold was originally added to suppress.

## Context

- Two reference tiles were used throughout: a confirmed-positive tile (r44000, c26000 — visible curved crevasse banding) and a confirmed-negative tile (r52000, c24000 — near-pure speckle, no visible linear structure, used as a specificity control).
- The percentile threshold itself was a fix from an *earlier* session (Otsu thresholding always found some bisection point even on pure noise; per-tile percentile of nonzero response fixed that — see `ridge-detection-pipeline-redesign` memory).
- macOS multiprocessing `spawn` start method requires all `Pool()` usage to sit behind `if __name__ == "__main__":` — got bitten by this earlier in the session on an unrelated ad hoc script; carried the lesson forward into every rerun script this session.

## The Journey

### Attempt 1: Lower the percentile ("middle ground" at pct=80)
**What we tried**: Sweep `ridge_percentile` from 90 down to 60 on both reference tiles.
**Why we thought it would work**: Lower percentile = less strict threshold = more of the real ridge response survives.
**What happened**: Positive tile recall did improve smoothly (2.4% coverage at pct=90 → 18% at pct=60). But the negative/speckle tile's false-positive count did NOT fall off smoothly — it cliffed sharply between pct=85 (81 false components) and pct=90 (8 components). At pct=80, the "obvious" middle ground, the negative tile still had 229 false components / 3.48% coverage.
**Why it failed**: A single global percentile cannot satisfy both goals simultaneously *at native resolution* — this looked structural, not a tuning miss.
**Time spent**: ~30-45 min across several sweep scripts.

### Attempt 2: Combine both Frangi polarities (`max(bright, dark)`)
**What we tried**: Hypothesis was that real SAR crevasses show as a bright-rim/dark-shadow pair, so a single polarity (`black_ridges=True` or `False`) only ever captures half of each real feature — combining both should recover the missing half.
**Why we thought it would work**: Made physical sense for SAR crevasse shadowing, and zoomed visual inspection showed bright-only and dark-only masks picking out different, only-partially-overlapping subsets of the same visible curved pattern.
**What happened**: Combining did raise positive-tile coverage (2.52% → 8.06%), but it blew up the negative tile from 8 false components to 232 (4.32% coverage) — a regression of similar magnitude to just lowering the percentile.
**Why it failed**: The two polarities' false-positive populations aren't independent noise that averages out — taking the max of two independently-thresholded fields just gives noise twice as many chances to clear the bar.
**Time spent**: ~20 min.

### Breakthrough #1 (diagnostic, not a fix): raw Frangi amplitude doesn't separate signal from noise at all
**Key insight**: Directly compared the nonzero Frangi response distributions between the positive and negative reference tiles (percentiles p50 through p99.9). The **negative (pure speckle) tile had a *higher* response than the positive (real crevasse) tile at every percentile from p50 to p90** (e.g. p90: negative=0.186 vs positive=0.156). This flipped the naive assumption on its head.
**Source**: A quick ad hoc distribution-comparison script, prompted by wanting to actually answer "why isn't it picking up more lines" with evidence instead of theorizing.
**Why this mattered**: It explained *why* attempts 1 and 2 both failed the same way — there is no threshold, percentile, or polarity combination that can cleanly separate two populations that aren't separated in amplitude to begin with. This reframed the problem from "find a better threshold" to "the discriminating variable itself is broken."

### Attempt 3: Multiscale scale-consistency filtering
**What we tried**: Hypothesized that requiring a pixel's ridge response to be strong across *multiple individual* Frangi sigma scales (not just the single max-over-scales output) would separate real curvilinear structure from noise, since real ridges should be scale-coherent while noise shouldn't be.
**Why we thought it would work**: Seemed like a principled way to add a second, amplitude-independent discriminating signal (spatial/scale coherence) on top of the broken amplitude signal.
**What happened**: Both tiles lost detections at *roughly the same rate* as the required scale-agreement count `k` increased (negative: n=287→8 from k=1→4; positive: n=178→61 from k=1→4). At the strictest setting, positive-tile recall (61 components) was actually worse than the plain percentile-90 baseline (158 components).
**Why it failed**: The multiscale Gaussian smoothing inside Frangi that creates the noise-correlated-blob problem in the first place is itself consistent across scales — so noise is just as "scale-consistent" as real signal. This filter added no new information; it was strictly dominated by the existing threshold.
**Time spent**: ~15 min (script + one run).

### Breakthrough #2: the user's downscaling intuition, extended past where it was first tested
An *earlier* part of this session had already validated that block-averaging (`downscale_local_mean`) the raw tile before running Frangi, at a fixed strict `pct=90`, cleanly improved specificity (negative tile: 8→0 false components from 1x→4x downscale) without touching recall much. This session picked that back up and pushed further: swept downscale factors 1x-8x **crossed with** percentile 60-90, instead of holding percentile fixed.

**What changed**: At native resolution (1x), lowering percentile from 90→80 blew the negative tile up to 216 false components / 3.18% coverage (the same cliff as Attempt 1). At **8x downscale**, the identical percentile sweep gave a *qualitatively different, well-behaved curve*: pct=70 → positive n=8/17.5% coverage, negative n=1/0.87% coverage. The negative tile's false-positive count stayed at 1-2 small components across the entire pct=65-80 range, instead of cliffing.

### Final Solution: 8x block-averaging + `ridge_percentile≈70`, wired in as `ridge_downscale_factor`
**What we did**: Added a `ridge_downscale_factor` constructor parameter to `ImprovedEdgeCrevasseDetector`. Internally: block-average the *raw* tile (dropping partially-invalid blocks via a valid-fraction mask) before running the *entire* existing pipeline (speckle filter → normalize → texture mask → Frangi → threshold/clean) at the reduced resolution, then upscale the resulting binary mask and continuous arrays back to the original tile shape (nearest-neighbor for masks, bilinear for continuous fields) before returning — so every downstream consumer (`build_crevasse_labels.py`, `visualize_improved_detection`) sees the exact same schema/shape as before. Downscaling is entirely an internal implementation detail.
**Why it worked**: This is multi-look theory, not thresholding — averaging N=64 pixels (8x8 block) cuts speckle variance by ~64x (1/N scaling), pushing the noise-driven Frangi response down near the true noise floor, while genuine multi-pixel-wide crevasse bands survive the averaging largely intact. It fixes the actual broken signal (amplitude separation) instead of trying to compensate for it with a smarter threshold.
**Validation**: Reproduced the exact ad hoc experiment numbers bit-for-bit after wiring it in (sanity check that the resize/upscale logic didn't subtly change anything). Then re-ran the full 610-tile ROI scan: tiles flagged went from 184→510 (30%→84% of valid tiles), average coverage 0.78%→9.54%. Because that jump was large enough to be suspicious on its own, did two independent sanity checks before trusting it: (1) the known-real hotspot rows still scored highest (13.8% vs 8.7% mean coverage), so the signal wasn't diluted; (2) visually spot-checked 6 newly-flagged tiles specifically *outside* the known hotspot — 5 of 6 showed clearly visible real crevasse banding the old detector had simply missed, only 1 of 6 looked ambiguous. This confirmed the recall gain generalizes past the 2 tiles it was originally tuned on.

## Key Lessons

### What We Learned
1. **When every threshold-tuning attempt fails the same way (recall gain always costs an equal-or-worse specificity loss), stop tuning the threshold and go check whether the underlying signal actually separates the classes at all.** The direct amplitude-distribution comparison (Breakthrough #1) should have been the *first* experiment, not the third.
2. **A filter that adds a "smarter" criterion on top of a broken base signal doesn't help if the criterion is correlated with the same thing that broke the base signal.** Scale-consistency failed because Frangi's own multiscale smoothing already makes noise scale-consistent — the new filter wasn't independent information, just a repackaging of the same information.
3. **Fixing a signal-separation problem by averaging/denoising the input is more robust than fixing it by combining or re-weighting the broken derived signal.** Both dual-polarity combination and scale-consistency tried to extract more discriminating power from the *already-computed* Frangi response; downscaling instead fixed the raw input the response was computed from. It's the difference between "compensate for noisy measurements" and "make fewer, better measurements."
4. **A large, suspicious-looking jump in a metric (184→510 tiles) is exactly when you most need an independent validation check, not exactly when you should feel good and stop.** Two cheap checks (spatial-hotspot consistency, 6-tile visual spot check) turned "this number looks too good" into "this is actually a validated result."

### What Didn't Work and Why
- Lowering the percentile at native resolution → recall improves, but false positives cliff back in at a similar magnitude — no clean middle ground exists at native resolution.
- Combining Frangi polarities (`max(bright, dark)`) → same magnitude of false-positive regression as just lowering the percentile; the two polarities' false positives aren't independent.
- Multiscale scale-consistency filtering → strictly worse than the existing baseline threshold; noise is just as scale-consistent as signal because of how Frangi's own smoothing works.

### What To Do Next Time
1. Before tuning a threshold on a derived/continuous signal, check whether the signal's *distribution* actually separates your positive and negative reference cases — a quick percentile/histogram comparison is cheap and can save several failed tuning attempts.
2. When a proposed fix only re-derives information from the same upstream computation (e.g. multiple scales of the same filter, multiple polarities of the same filter), be suspicious that it's correlated with the existing failure mode rather than orthogonal to it.
3. Keep a matched pair of reference tiles (one confirmed-positive, one confirmed-negative/pure-noise) on hand for every tuning experiment — nearly every experiment in this session depended on having both, not just a positive example.
4. When a change produces a much bigger effect than expected, budget time for an independent sanity check (spatial consistency, visual spot-check, etc.) before treating it as validated — do this immediately, while the context/scripts are still warm, not as an afterthought.

### Patterns Recognized
- "Structural tradeoff, not a tunable knob" is a useful phrase/mental flag for when a whole family of threshold values has been swept and none of them work — it's a signal to stop varying the threshold and go vary something upstream instead.
- Averaging/multi-look approaches are a recurring, physically-grounded lever for SAR speckle problems specifically (this is the second time in this pipeline's history that a speckle-driven artifact was fixed by changing how the signal is computed, rather than how it's thresholded — the first was the Otsu→percentile switch, see [[ridge-detection-pipeline-redesign]] memory).

## When to Apply This Lesson

**Similar problems**:
- Any detector/classifier tuning session where every threshold value tried trades recall for precision at roughly the same rate, with no clean middle ground — suspect the underlying score doesn't separate the classes, and check the raw score distributions directly before continuing to sweep the threshold.
- Any SAR or other speckle/noise-limited imagery problem where a derived per-pixel statistic is noisy — multi-look/block-averaging the raw input before computing the statistic is worth trying before trying to denoise or re-threshold the statistic itself.
- Any "combine two related signals" fix (multiple polarities, multiple scales, multiple filters) — check whether the two signals' *failure modes* are actually independent before expecting the combination to help.

**Keywords**: Frangi, ridge detection, vesselness filter, SAR speckle, multi-look, downscale_local_mean, percentile threshold, precision-recall tradeoff, false positive cliff, scale consistency, dual polarity, crevasse detection

## References
- Files touched: `code/edge_crevasse_v2.py` (added `ridge_downscale_factor` param and downscale/upscale logic in `process_tile`)
- Related lessons: [[macos-multiprocessing-spawn-hang]] (encountered earlier in the same session, informed how the 610-tile rerun script was written)
- Related memory: `ridge_detection_pipeline_redesign.md` (full quantitative history of this detector's evolution, including the Canny+Hough→Frangi switch and the Otsu→percentile threshold fix that preceded this session's work)
- Key data: reference tiles r44000_c26000 (positive) and r52000_c24000 (negative), NISAR granule `025_091`
- Output artifacts: `data/crevasse_labels/crevasse_labels_ppb_fast_ridge_tile1024_braided_region_8x.pkl`, `data/improved_detections_025_091_braided/old_vs_new_run_spatial.png`, `.../spotcheck_new_far_detections.png`

## Time Investment
- Total time: ~2-3 hours across the failed attempts (percentile sweep, polarity combination, scale-consistency) plus the successful downscale extension and full-ROI validation
- Could have been: ~45-60 min if the raw amplitude-distribution comparison (Breakthrough #1) had been done first, before the polarity-combination and scale-consistency detours
- Savings for next time: ~1-1.5 hours — check signal separation before tuning thresholds or combining derived signals

---
*Captured from conversation on 2026-08-27*
