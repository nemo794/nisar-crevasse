---
date: 2026-08-30
topic: FFT-as-feature done right, faithful PPB despeckle, and a "hung" job that wasn't
tags: [fft, features, ppb, speckle, despeckle, random-forest, sar, conda, joblib, debugging, research]
status: resolved
difficulty: medium
---

# Lesson: Raw spectra beat spectral summaries, PPB is easy to get subtly wrong, and `conda run` hides progress

## Objective

Continue the "cheap gate" feature search: (a) settle whether FFT can contribute
features to the binary crevasse gate after the 08-29 conclusion that it was dead,
and (b) evaluate whether a proper PPB speckle filter is worth using.

## Context

- Gate = a Frangi-independent RF classifier that pre-filters tiles before the
  expensive ridge stage. Tuned for high recall; must stay cheap.
- 08-29 had declared "FFT is dead for the gate" based on an FFT angular-energy
  profile feature with Cohen's d=0.079.
- Env: conda `biomass`; no PPB library is pip-installable.

## The Journey

### Attempt 1: FFT angular-energy profile as a feature (inherited, already failed)
**What we tried**: Collapse the 2D FFT magnitude to a 1D profile of mean energy
vs angle, extract peak/mean ratios.
**Why we thought it would work**: Oriented crevasse ridges should show a
directional spectral streak.
**What happened**: d=0.079, zero separation on real tiles. Dead.
**Why it failed**: Collapsing the 2D spectrum to a 1-D angular summary throws away
almost all the discriminative structure, and the profile is too noisy per-tile to
locate a consistent orientation (verified visually this session: peaks at
166/94/26/26 deg across four windows of the *same* confident-crevasse tile).

### Breakthrough: use the raw spectrum, not a summary of it
**Key insight**: The failure wasn't "FFT doesn't help" — it was "hand-crafted
spectral *summaries* don't help." Feed the RF the **raw downsampled magnitude
bins** and let it find the structure itself.
**Source**: User's suggestion — "use the values instead" of computing min/max/peak.

### Final Solution (FFT branch): raw pooled magnitude bins + light despeckle
**What we did**: NL-means despeckle -> 4 overlapping 384px corner windows ->
Hann-windowed FFT -> log-magnitude -> resize to 16x16 -> max-pool over the 4
windows = 256 bins. Concatenate with the 14 hand features = 270.
**Why it worked**: Preserves the full 2D oriented-ridge structure; max-pool over
overlapping windows is a shift-ensemble that tolerates ridge position; despeckle
drops the speckle noise floor so ridge lines stand out (visually obvious in
`_viz_ppb_fft.py`: FFT of raw = featureless DC blob; FFT of despeckled = clear
low-freq oriented lobes).
**Result**: OOF 0.901 -> 0.915; held-out 0.844 -> 0.869. Shipped into
`train_gate_classifier.py`.

### Attempt 2: hand-rolled PPB filter
**What we tried**: Implement PPB (Deledalle 2009) from memory — NL-means with a
SAR amplitude likelihood weight + iterative refinement.
**What happened**: Ran, gave combo 0.921. Looked fine. But when compared to a
reference implementation the user supplied, two real bugs surfaced:
1. **WML estimator averaged amplitude, not intensity.** For Nakagami speckle the
   ML reflectivity estimate is the weighted mean of *intensity* (A^2), then sqrt.
   Averaging amplitude directly is biased.
2. **Refinement term was broken.** It compared amplitude ratios inside the *same*
   `exp` with the *same* bandwidth `h` as the data term. The correct form uses
   reflectivity (A^2) with a *separate, much smaller* bandwidth `T` (~5x stronger).
   Result: my "iterations" did almost nothing, so I wrongly concluded "iteration
   doesn't help."
**Why it failed to be correct**: The data-similarity term (the GLR
`log(A/A' + A'/A)`) was right, so single-iteration numbers looked plausible and
masked the estimator/refinement bugs.

### Final Solution (PPB): use the reference impl, then decide on cost
**What we did**: Saved the user's clean reference as `code/ppb.py`
(`ppb_nakagami`). Re-benchmarked. Correct PPB: combo **0.924** (best of all), and
the refinement *does* help (fft 0.899->0.905, i1->i4). BUT PPB *hurts* the hand
GLCM features (0.906->0.885, over-smooths texture) and is **~40x heavier** than
NL-means (~9s/tile vs ~21ms/tile).
**Decision**: PPB stays out of the gate (not worth +0.009 OOF at 40x cost for a
"cheap filter"); reserved for the downstream Frangi ridge stage where compute is
affordable and clean ridges are exactly what's wanted.

### Side-quest: a background job that looked hung for 30 minutes
**What happened**: The i4 PPB benchmark produced an empty output file after ~30
min; a first `ps | grep python` showed only the parent at low CPU and no workers.
Looked like the known macOS multiprocessing spawn hang.
**Why it wasn't**: (1) `conda run` **buffers all child stdout until the process
exits** — an empty log is not evidence of no progress. (2) The first `ps`
snapshot caught a gap between joblib batches. A second check showed 8 loky workers
at 60-80% CPU with ~23 min CPU time each. It was genuinely compute-bound (PPB i4 =
~9s/tile x 950 tiles / 8 workers ~ 19 min per config). Verified by timing one
`ppb_nakagami` call in isolation.

## Key Lessons

### What We Learned
1. **"Feature X is dead" often means "my *summary* of X is dead."** Before
   abandoning a signal, try handing the raw representation to the model instead of
   hand-crafted statistics. The 2D FFT bins worked where every 1D summary failed.
2. **Despeckle-then-FFT is the pattern.** Speckle spreads energy across all
   frequencies (flat noise floor) and buries oriented structure; denoising first
   is what makes the spectrum informative.
3. **PPB is easy to get subtly wrong.** The data-similarity term is the famous
   part and easy to copy; the WML estimator (average *intensity*, not amplitude)
   and the refinement bandwidth (`T` << `h`, on reflectivity) are the parts that
   quietly determine whether iteration does anything. Validate against a reference.
4. **A stronger filter is not automatically the right filter.** PPB won on AUC but
   lost on cost-per-gain and even *hurt* the texture features. Match the filter to
   the stage: light (NL-means) for the cheap gate, strong (PPB) for the expensive
   ridge stage.

### What Didn't Work and Why
- FFT angular-energy profile -> collapses 2D structure to a noisy 1D summary.
- Hand-rolled PPB -> amplitude-averaging estimator + mis-scaled refinement.
- PPB in the gate -> 40x cost for +0.009, and it degrades the GLCM branch.

### What To Do Next Time
1. When a spectral/transform feature fails, test the raw binned transform as a
   feature vector before concluding the transform is uninformative.
2. When reimplementing a published filter, get a reference implementation and
   diff behavior — especially the estimator and any per-term bandwidths, not just
   the headline distance metric.
3. Before calling a `conda run` background job "hung": remember stdout is buffered
   till exit; check **per-worker CPU time** (`ps -o time`), not the log. (Same
   check as the `macos_multiprocessing_spawn_hang` memory, opposite conclusion.)

### Patterns Recognized
- Same "summary throws away the signal" family as the 08-29 angular-ratio failure
  and the original 45° Hough artifact — repeatedly, aggregating away spatial/2D
  structure is where crevasse features die.
- "Looks hung" debugging mirrors the memory note: always distinguish *idle* from
  *busy* by CPU time, but also account for output buffering before reading intent
  into an empty log.

## When to Apply This Lesson

**Similar problems**:
- A transform-based feature (FFT/wavelet/etc.) shows no class separation.
- Reimplementing a speckle/denoise filter and unsure it's faithful.
- A long parallel job under `conda run` appears stalled with no output.

**Keywords**: FFT feature vector, spectral summary, angular energy profile,
Nakagami, PPB, Deledalle, WML estimator, weighted maximum likelihood,
denoise speckle, NL-means, joblib loky workers, conda run buffering, stdout
buffering, job hung, per-worker CPU time.

## References
- Files touched: `code/ppb.py` (new), `code/train_gate_classifier.py` (upgraded),
  `code/_ppb_test.py`, `code/_viz_subtile_fft.py`, `code/_viz_ppb_fft.py`,
  `data/gate_model.joblib` (re-saved), `data/gate_model.hand14.joblib.bak`.
- Related lessons: [[2026-08-29-classifier-feature-search-and-detector-drift]],
  [[2026-08-27-crevasse-ridge-detection-recall]]
- Related memory: [[gate-feature-upgrade]], [[ppb-despeckle-decision]],
  [[macos-multiprocessing-spawn-hang]]

## Time Investment
- Total time: ~1 session
- Biggest avoidable cost: ~40 min waiting on / re-running an i4 PPB benchmark and
  second-guessing whether it was hung.
- Savings for next time: the raw-bins-not-summaries reflex and the
  reference-diff-your-filter habit should each save a full false-negative
  investigation.

---
*Captured from conversation on 2026-08-30*
