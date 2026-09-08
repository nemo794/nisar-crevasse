---
date: 2026-08-31
topic: "Discard all negatives" is a classifier-separability wish, not a threshold you can dial
tags: [threshold, random-forest, precision-recall, class-overlap, gate, sar, operating-point, petabyte-scale]
status: resolved
difficulty: easy
---

# Lesson: You cannot threshold your way to "discard all negatives, keep all positives"

## Objective

Set the operating threshold for the cheap binary crevasse "gate" now that the
pipeline is going to run at petabyte scale. New priority: discard as much negative
volume as possible (throughput/cost), accepting that extra false positives pass
downstream. User's initial ask: "just discard all negatives and gate the positives."

## Context

- Gate = RandomForest (400 trees) outputting `p = predict_proba[:,1]` ∈ [0,1],
  the fraction of trees voting crevasse. Held-out AUC 0.87, OOF 0.915.
- Shipped threshold was 0.320 (chosen for recall≥0.95, skips ~38%).
- Held-out eval set: 152 fresh tiles, 67 crevasse / 85 none.

## The Journey

### Attempt 1: aggressive single cut — keep only P≥0.85
**What we tried**: Treat 0.85 as the keep bar, discard everything below.
**What happened**: Dropped 95.3% of negatives (goal!) but kept only 43.3% of
crevasses — lost 38 of 67.
**Why it wasn't enough**: The "almost all negatives gone" win came entirely at the
expense of more than half the crevasses.

### Attempt 2: sweep the keep threshold 0.50→0.99, watch negatives-discarded
**What happened**: 100% of negatives discarded only at keep_p≥0.95 — where recall
collapses to ~10% (7 tiles kept). And that 0-false-positive point is partly a
small-sample artifact (only 85 negatives).
**The knee**: keep_p≈0.65 — 87% of negatives discarded, 73% recall retained.
0.65→0.85 buys only 8 more points of negative-discard but costs 30 points of recall.

### Breakthrough: the user's model of `p` was the real issue
**User asked**: "so p is not probability? I thought we might just discard all
negatives and gate the positives."
**Key insight**: `p` *is* the probability. The flaw is the assumption that a
threshold can cleanly separate classes. It can't, because the two classes
**overlap in score** — some real negatives score 0.7–0.9, some real crevasses
score 0.3–0.5. "Discard all negatives AND keep all positives" describes a
*perfectly separating classifier*, not a choice of cut point. The entire sweep
table is just a readout of that overlap.

### Final Solution: pick the knee, name the backstop
**What we did**: Locked in keep_p=0.65 (added a reproducible `--threshold`
override to `train_gate_classifier.py`, retrained, backed up the 0.32 model).
On 950-tile OOF: keeps 72.8% crevasses, discards 90.5% negatives, skips 60.8%.
**Why it worked as a decision**: 0.65 is the efficient point — beyond it you pay
recall much faster than you gain discard. To discard *more* without losing recall
you need a better-separated classifier (better features / more data), not tuning.
The downstream Frangi ridge stage remains the precision backstop, which matters
because at realistic low prevalence even a clean 88% precision (P≥0.85 on the
balanced labeled set) falls to ~22% at 3% prevalence.

## Key Lessons

### What We Learned
1. **"Discard all negatives, keep all positives" is not a threshold — it's a wish
   for a perfect classifier.** When someone asks for it, show the class overlap in
   the score distribution; the answer lives there, not in the cut point.
2. **Find the knee, not the extreme.** Where the precision/recall (or
   discard/recall) curve bends is the efficient operating point. Past it you trade
   a lot of the scarce quantity for a little of the plentiful one.
3. **At low prevalence, recall is expensive and negative-discard is cheap.** Rare
   positives mean each point of recall lost throws away scarce signal, while
   over-keeping negatives is comparatively harmless — so *don't* chase the last few
   percent of negative-discard.
4. **A 100%/0-error corner on a small eval set is usually an artifact.** keep_p≥0.95
   showed "100% negatives discarded / precision 1.0" on just 85 negatives — it
   won't hold on a real scan. Distrust perfect corners from small samples.

### What Didn't Work and Why
- Single aggressive cut at 0.85 → dropped >half the crevasses.
- Chasing 100% negative-discard → recall collapse (classes overlap in score).

### What To Do Next Time
1. When asked to "just discard all the negatives," first plot/print the score
   distributions of both classes to expose the overlap, then discuss the knee.
2. Store operating-point choices reproducibly (an explicit `--threshold` flag with
   provenance in the model bundle), not as a hand-edited scalar.
3. Push separability (features, data) when you need more discard *and* more recall;
   thresholding can only move you along the existing curve.

### Patterns Recognized
- Same shape as the 08-27 lesson: "when every threshold trades recall for precision
  at the same rate, the signal may not separate the classes" — here it separates
  *partially*, and the knee is where that partial separation is best spent.

## When to Apply This Lesson

**Similar problems**:
- Setting a decision threshold for any binary classifier under a cost asymmetry.
- A stakeholder expects a threshold to perfectly split two classes.
- Choosing an operating point for a high-volume pre-filter / gate.

**Keywords**: predict_proba, decision threshold, operating point, class overlap,
precision-recall knee, negatives discarded, recall cliff, low prevalence,
base rate, petabyte scale, pre-filter gate, RandomForest score.

## References
- Files touched: `code/train_gate_classifier.py` (new `--threshold` override),
  `code/_threshold_sweep.py` (discard sweep + aggressive experiment),
  `data/gate_model.joblib` (re-saved @0.65), `data/gate_model.thr032.joblib.bak`.
- Related lessons: [[2026-08-30-fft-features-and-faithful-ppb]],
  [[2026-08-27-crevasse-ridge-detection-recall]]
- Related memory: [[gate-feature-upgrade]], [[ppb-despeckle-decision]]

## Time Investment
- Total time: ~short session.
- Biggest avoidable cost: none major — the conceptual clarification about `p` was
  the whole point and was quick.
- Savings for next time: skip straight to showing class-score overlap when someone
  asks to "discard all negatives."

---
*Captured from conversation on 2026-08-31*
