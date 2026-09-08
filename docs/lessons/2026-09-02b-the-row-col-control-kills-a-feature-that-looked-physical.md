---
date: 2026-09-02
topic: A smooth physical covariate can act as a tile ID; the (row, col) control is the only thing that catches it
tags: [spatial-leakage, memorization, feature-validation, holdout-design, random-forest, context-features, deployment, raster-resampling]
status: resolved
difficulty: hard
---

# Lesson: The feature was physics. The model was using it as a coordinate.

## Objective

Explain to the user how a new 512×512 tile obtains the context features (Bedmap3 surface
class, ITS_LIVE ice speed) for the second random forest — the "context prior" that the
previous session had measured at **+0.027 held-out AUC** and adopted as the preferred
fusion route.

The task was expository. It ended with the feature deleted.

## Context

- Two-model gate: RF #1 on 270 SAR texture features, RF #2 on 6 context features, combined
  at scoring time by a parameter-free naive-Bayes `fuse()`. Nothing is fitted on the
  combination.
- Training set: 910 speed-matched tiles from granule 025_019. Held-out: 183 tiles of
  "genuinely unseen ground" on 025_091, isolated with `--exclude-ground-from`.
- The prior had already survived a lot of scrutiny the same day: a velocity-only AUC floor
  (0.843 vs SAR 0.954), spatial-block CV, a ground-overlap check that caught 73% of the
  "held-out" granule sitting on training ground, and a degeneracy warning for when the
  forest extrapolates. It was not adopted carelessly.
- Constraint that mattered: **ice speed is spatially smooth and near-unique per tile.**
  Nobody had asked what that implies.

## The Journey

### Attempt 1: Just trace the lookup and answer the question

**What we tried**: Walk `context_features(granule, row, col)` — warp Bedmap3 and ITS_LIVE
onto the granule tile grid, index by `row // TS`, emit 6 numbers.

**What happened**: The trace was fine, but reading the feature importances to explain which
ones mattered showed **three of six at exactly 0.0000**: `ctx_grounded_frac`,
`ctx_is_floating`, `ctx_vel_missing`.

**Why**: The speed-matched training set is 100% grounded ice with velocity everywhere. Those
features are *constant in training*. So the "6-feature context model" is a 2-feature model
on `log_vel_{mean,max}`.

**Time spent**: ~15 min. Not a failure — the first crack.

### Attempt 2: Probe the untrained code paths

**What we tried**: If `ctx_vel_missing` is constant in training but *does* occur at scoring
time, what does the model do with it? Checked its frequency in the 910 training tiles and
traced the resulting probability through `fuse()`.

**What happened**: `ctx_vel_missing=1` appears in **0 of 910** training tiles. At scoring
time it maps to P_ctx = 0.031, which drags a confident P_sar = 0.90 down to **0.224**.

**Why it's wrong**: The model has never seen the value, so its response is an artifact of
leaf geometry. Operationally, **missing velocity was being read as evidence of absence** —
an unmapped tile got argued out of being a crevasse.

Two more non-physical responses fell out of the same probe: floating shelf scores **0.984**,
*boosted above* grounded ice at 0.772; and the speed response is a non-monotonic staircase,
not a trend.

**Time spent**: ~20 min.

### Breakthrough

**Key insight**: `ctx_log_vel_mean` has **910 unique values across 910 tiles**.

That is not a feature distribution. That is a primary key. A forest given a near-unique,
spatially smooth continuous covariate can partition tiles individually and then *interpolate*
to neighbours — which is exactly what a coordinate system does. The staircase response wasn't
noise; it was the shape of memorization.

**Source**: Not an error message and not the literature. It came from asking a mundane
explanatory question — "where does this number come from for a new tile?" — with enough
attention to actually look at the values.

### Final Solution: the four-variant control (`_ctx_prior_binned.py`)

Retrain the identical context bundle four ways, 025_019 → 025_091:

| variant | what it keeps | ctx-only unseen AUC | fused gain over SAR |
|---|---|---|---|
| `continuous` (as shipped) | physics + resolution | 0.843 | **+0.027** |
| `binned8` | physics, resolution destroyed | ~0.72 | **+0.001** |
| `randid` | nothing (control) | 0.525 | — |
| `coords` | **position only, zero physics** | **0.955** | **+0.031** |

**Why it's decisive** — the two variants disagree in opposite directions:

- **Coarsening the values while keeping the physics destroys the gain.** If the model were
  using "fast ice crevasses more," 8 quantile bins would carry that. They carry ~nothing.
- **Removing the physics entirely and keeping only position beats it.** Raw `(row, col)`
  with no geophysical input at all scores *higher* than the physical model.
- **`randid` transfers at chance (0.525)**, which rules out plain label leakage. A random
  per-tile ID memorizes in-sample and transfers at nothing; speed transfers *because it is
  smooth*. That is the signature of spatial interpolation specifically.

Verdict: the prior is a map, not a mechanism. **Dropped from the decision path.**

Replacement (`--grounded-only` in `map_crevasse_tiles.py`): Bedmap3 as a **deterministic
hard mask** — every tile still scored and saved, non-grounded ones simply never returned as
positives — and ITS_LIVE demoted to reporting strata. The mask is read through the same
`T.context_matrix(meta, ["ctx_is_grounded"])` lookup the classifier trained with, so the
indexing cannot silently diverge. 31% of raw detections were off grounded ice.

### Collateral finding: the holdout that validated it was also broken

If a pure coordinate model scores 0.955 on the 183 "unseen ground" tiles, then that subset
does not test transfer for *any* smooth spatial feature. `--exclude-ground-from` drops test
tiles whose **exact** `TS*5 m` cell appears in training — which leaves every training tile's
immediate neighbours in the test set, and a smooth model just interpolates across the
one-cell gap.

The 183-tile subset was the entire basis for the previous session's decision. **The control
invalidated both the feature and the protocol that blessed it, in one run.**

## Key Lessons

### What We Learned

1. **A feature being physically meaningful says nothing about how the model uses it.** Ice
   speed is real physics. The forest used it as a tile index. Provenance is not a defense.
2. **`(row, col)` is the control for any smooth spatial covariate.** It is trivial to
   implement — swap two columns of the design matrix — and it is the *only* test here that
   caught the problem. Nothing else in a fairly rigorous stack did.
3. **Two controls, not one, and they must point opposite ways.** *Coarsen the values* (does
   the physics survive?) and *strip the physics* (does position alone suffice?). Either
   alone is inconclusive; together they pin it.
4. **`randid` distinguishes memorization from leakage.** Both memorize in-sample. Only a
   *smooth* covariate transfers. If your suspect feature beats `randid` on held-out but a
   coarsened version of it doesn't, the mechanism is spatial interpolation.
5. **Feature importance of exactly 0.0000 means "constant in training," not "unimportant."**
   And a constant-in-training feature that varies at scoring time is a **live untrained code
   path**. Grep for those before deploying.
6. **Count unique values per training row.** 910 unique values in 910 rows is a red flag
   with a number attached — a cheap, mechanical check that would have caught this on day one.
7. **Feature values that are unphysical are the tell.** Floating shelf ranked above grounded
   ice for crevassing. Sanity-checking the *response surface* against domain knowledge is
   faster than any AUC.
8. **"Held out" must be measured in projected metres, with a buffer.** Exact-cell exclusion
   is not spatial separation.

### What Didn't Work and Why

- **A velocity-only AUC floor (0.843)** → measures whether the feature is confounded with
  the label. Says nothing about *how* the model exploits it. The prior cleared this bar and
  was still memorizing.
- **Spatial-block CV** (`StratifiedGroupKFold`, 8-tile ≈ 20 km blocks) → catches leakage
  *within* a granule. The coordinate model still scored 0.982 blockOOF. Blocks were too
  small relative to the smoothness.
- **`--exclude-ground-from`** → exact-cell only, so neighbours leak. Insufficient by
  construction, not by bug.
- **The degenerate-prior IQR warning** → real and useful (a forest returns a constant
  outside its training range, which is a relabelled threshold), but it fires on
  *extrapolation*. It is silent in-range, which is exactly where memorization lives.

Every one of these was a reasonable check that passed. **A stack of reasonable checks can
all miss the same failure mode if none of them is the control.**

### What To Do Next Time

1. Before adding any gridded, continuous, spatially smooth feature: run the **`coords`
   variant first**, as a baseline, not as a post-hoc audit. If coordinates match your
   feature, stop.
2. Add `randid` alongside it — it costs one line and separates memorization from leakage.
3. Print unique-value-count / n_rows and the per-feature importance table for every bundle.
   Zeros and near-1.0 ratios are both alarms.
4. Buffer spatial holdouts by several tiles in projected metres. Report the buffer width
   with the AUC.
5. When a physical raster genuinely belongs in the pipeline, prefer a **deterministic filter
   over a learned feature**. A hard mask cannot memorize.
6. When the goal is exposition, slow down rather than speed up. This entire finding came out
   of explaining existing code to someone.

### Patterns Recognized

- **Third consecutive session where "the thing that looked better was the confound."**
  08-31: a cleaner threshold was class overlap. 09-02a: the prior's nicer-looking maps were
  the velocity field in colour (rank correlation with speed 0.30 → 0.61). 09-02b: the prior's
  *held-out gain* was geography. Same failure, escalating subtlety — appearance, then
  correlation, then a validated metric on a held-out split.
- **Connects to `2026-09-01` (cross-band domain shift)**: both are "the model is responding
  to scene identity, not to the target." There it was texture domain shift making a gate
  fire everywhere; here it is position dressed as physics. Ask *what varies between scenes
  that the model could latch onto* — the answer is rarely the physics you intended.
- **Inverse of `2026-08-30` (raw FFT beat spectral summaries)**: there, giving the model the
  raw representation *helped* because the summary discarded signal. Here, giving the model
  full-resolution values *hurt* because the resolution was identifying information. The
  general question is the same — **what does resolution buy?** — and the answer depends
  entirely on whether the fine structure is signal or an index.
- **Explaining code to someone is an underrated audit.** Two of the day's three findings
  (dead features, unique-value count) surfaced while narrating a lookup, not while testing it.

## Also Settled This Session

Not part of the main journey, but worth finding here:

- **The gate's RF fit costs 0.46 s; featurizing 910 tiles costs 33.1 s.** Featurization is
  ~100% of training cost. At 1/4 PB (~125M data-bearing tiles) that is ~13,000 core-hours
  ≈ 1 day on 500 cores, and a 270-float32 feature cache is ~135 GB — **1/1800th of the
  input.** Cache features, not models.
- **One continent-wide 2560 m context grid (7.2 MB, `data/context_grid_2560m.npz`) replaces
  per-granule warping**, which re-read a 7.49 GB ITS_LIVE source per granule. Total static
  worker state under 11 MB.
- **Building it falsified a docstring.** `context_features` claimed both context fields are
  "smooth enough for the difference not to matter" when cell size mismatches tile footprint.
  On 2.5 m granules `res = t.a * tile` yields a 1.28 km cell against a 2.56 km tile, so the
  lookup returns the tile's leading quadrant. `vel_mean` correlation grid-vs-per-granule is
  **0.221** on 004_048 and 0.628 on 003_064. Grounded masking agrees 100%, so no decision
  made so far is wrong — but the speed strata on those two granules are not trustworthy.
  **A "this is fine" comment in the code is a hypothesis, and this one had never been tested.**
- **Only genuinely stateful deployment dependency:** the `--normalize-per-granule` moments
  (~2 KB/granule) must be reduced over a granule's tiles before any can be scored. Not the
  rasters — that was the assumption going in, and it was wrong.

## Open

**003_064 keeps 22% of grounded slow-ice tiles at P≥0.85, versus 6% / 6% / 2% on the other
three granules in the same speed band.** Speed cannot explain a rate that granule-specific,
so this points at residual domain shift surviving `--normalize-per-granule`. Next step:
compare its post-alignment feature distributions against the training scene, feature by
feature.

And the SAR gate's 0.954 has not been re-earned on a buffered holdout. Texture is local and
much less exposed to this failure than the context model was — GLCM offsets and a 384-px FFT
window cannot easily encode absolute position — but "less exposed" is a prediction, not a
measurement.
