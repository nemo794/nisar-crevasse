# 2026-09-01: A single-granule gate silently blows up across SAR bands; fix it with per-granule feature alignment (and verify FP hypotheses by looking, not guessing)

## The situation

A gate trained only on granule 025_091 (SAR band `frequencyA`) was mapped across
all three granules. On 025_091 it looked great — tight, high-precision, saturated-red
on the real branching crevasse field. But on 004_048 (`LSAR` band) it flagged
**2127 / 2906 tiles (73%)** as crevasse, and on 003_064 it went nearly silent. Same
model, wildly different behavior per scene.

## The journey

### 1. The tell was in the confidence, not the count
Before looking at a single tile: on 004_048 the max probability over the whole scene
was only **0.74**, with the bulk sitting just above the 0.65 threshold. On 025_091 the
positives hit **P=1.00**. A model that is "positive everywhere but never confident" is
not detecting anything — it's a distribution that has drifted so the whole scene sits
in a low-but-over-threshold band. That's the fingerprint of **domain shift**, not
detection.

### 2. Confirm by rendering the actual tiles (don't argue from features)
Rendered a montage of the highest-P 004_048 "positives" (the NL-means image the gate
sees) next to 025_091's. 004_048: pure featureless speckle, zero ridge structure.
025_091: unmistakable oriented crevasse ridges. This took one throwaway script and
settled it instantly — the 73% were all false positives.

### 3. Name the cause from the filenames
025_091 is `..._frequencyA_HH_...`; 003_064 and 004_048 are `..._LSAR_HH_...`.
Different SAR band → different speckle statistics. The gate's features (GLCM +
FFT-log-magnitude) are computed on *per-tile* percentile-normalized images, which
removes brightness but NOT the texture/speckle-stat difference. Foreign-band speckle
therefore lands in a weakly-crevasse pocket of feature space.

### 4. The fix: unsupervised per-granule feature alignment (CORAL-lite)
Store the training set's robust per-feature moments in the model bundle
(`train_feat_center` = median, `train_feat_scale` = 1.4826·MAD). At inference on a
new granule, estimate that granule's own robust moments from all its tiles and remap:

```
x' = (x - gran_center) / gran_scale * train_scale + train_center
```

The assumption: the bulk of any granule's tiles are background speckle, so aligning
the granule's overall feature distribution onto the training distribution drags the
speckle background onto the training background (→ scores as "none"), while genuine
crevasse tiles stay outliers (same #MADs above center) and survive the mapping.
Robust median/MAD (not mean/std) so a crevasse-rich scene doesn't skew the alignment.

Result: 004_048 **73%→5%** (2127→154), 003_064 stays sparse, and the training
granule 025_091 is essentially unchanged (886→927, Pmax stays 1.0, structure intact).
Exactly the desired behavior — kill the foreign-scene blowout, don't touch real signal.

### 5. Then I got the residual-FP cause wrong, and the data corrected me
The user noticed the leftover positives cluster near the swath edges and guessed a
sensor edge artifact. I had earlier hand-waved "partial-data / nodata-boundary tiles"
as the cause. I wrote a diagnostic that split positives by valid-data fraction and
rendered them. **My guess was wrong:** only 3-4% of positives were partial-data tiles.
The real picture was two *different* causes:
- **004_048**: structureless speckle residual, edge and interior fire at the same ~5%
  rate — no edge effect, just alignment leftover.
- **003_064**: faint **oriented diagonal streaks**, and edge tiles fire at **~2x**
  the interior rate (11% vs 6%) — a directional sensor/processing artifact (azimuth
  streaking) that intensifies toward the swath edge. The user's intuition was right
  for *this* granule, for a reason neither of us had stated precisely.

### 6. Threshold as the cheap precision knob
Raising the operating threshold 0.65→0.75 (a pure scoring knob, no retrain): 025_091
−15% (core fully intact), 004_048 −54%, 003_064 −63%. It removes the low-confidence
speckle residual but can't touch the 0.9+ oriented-streak artifacts on 003_064.

## What worked

- Reading **confidence distribution** (Pmax, median) before counts — it diagnosed
  "domain shift" vs "detection" before any tile was viewed.
- **Rendering the tiles** to confirm FPs, every time.
- CORAL-lite alignment with **robust** moments — theoretically sound and it held up
  empirically across a crevasse-rich AND two crevasse-poor granules.
- A **valid-fraction split + montage** diagnostic that falsified my own hypothesis
  cheaply.

## What didn't

- My first, unverified explanation of the edge FPs ("partial-data boundary") was
  wrong. I'd stated it in a summary before checking. The 3-4% number killed it.

## Meta-lessons

1. **A classifier that fires everywhere but never confidently isn't detecting — it's
   out of distribution.** Check Pmax / score distribution per scene before trusting
   any positive rate.
2. **Per-tile normalization removes brightness, not texture-domain shift.** For SAR,
   a different band/product is a genuine domain change; expect single-granule models
   NOT to transfer, and reach for domain adaptation (feature-moment alignment) or
   multi-domain training data.
3. **Feature-moment alignment (CORAL-lite) is a strong, cheap, unsupervised patch**
   when most of a scene is background — but it's a patch. The durable fix is labeled
   negatives from the other domains folded into training.
4. **Verify false-positive causes by looking and by a statistical split — never state
   a cause you haven't checked.** The intuitive explanation (nodata boundary) and the
   real one (residual speckle; directional streak artifact) were different, and only
   one was on the swath edge for the reason assumed.
5. **Threshold trims low-confidence residuals but is blind to high-confidence
   artifacts.** Oriented-streak artifacts score as high as real crevasses because the
   detector keys on oriented linear structure; those need masking or training data,
   not a threshold.
