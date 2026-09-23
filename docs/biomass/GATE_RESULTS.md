# Results

Every number here is from the **buffered** spatial-block split: 81.92 km blocks, 40.96 km
buffer withheld from training, 3 folds, n = 1747. Nothing from the unbuffered or
leave-one-footprint-out splits is quotable; see [METHOD.md](METHOD.md) for why they are kept
anyway. Read [LIMITATIONS.md](LIMITATIONS.md) alongside this file.

## Read the folds before the summaries

This table comes first on purpose. A fold with 4 positives is not evidence, and the pair
counts show how little the three folds are worth against each other.

| fold | n | positives | base rate | pairs (n1 x n0) |
|---:|---:|---:|---:|---:|
| 0 | 521 | 4 | 0.008 | 2 068 |
| 1 | 947 | 559 | 0.590 | 216 892 |
| 2 | 279 | 12 | 0.043 | 3 204 |

**Fold 1 is 97% of the total pair weight.** Any pair-weighted statistic over this data is
substantially a statement about fold 1, and fold 1 trains on 446 tiles.

## Per-fold AUC, all arms

| arm | fold 0 | fold 1 | fold 2 |
|---|---:|---:|---:|
| **CNN B** `sar+vel` (shipped) | **0.806** | **0.944** | 0.888 |
| CNN A `sar` | 0.731 | 0.930 | **0.895** |
| **RF** 38 features (shipped) | 0.621 | 0.762 | **0.918** |
| CNN C `vel` alone (control) | 0.703 | 0.672 | 0.660 |
| `(x,y)` alone (control) | 0.084 | 0.847 | 0.276 |

The CNN beats the RF on folds 0 and 1 and loses fold 2. **The honest claim is "wins 2 of 3
folds"**, not a single figure. Note also that the `(x,y)` control is *below chance* on folds 0
and 2 and at 0.847 on fold 1 — that is what a smooth spatial memoriser looks like when it is
extrapolated off its training block, and it is exactly why the buffer is there.

## Three summaries, which disagree

| summary | CNN B | RF | `(x,y)` | RF margin over `(x,y)` |
|---|---:|---:|---:|---:|
| pooled across folds | 0.914 | 0.889 | 0.601 | **+0.287** |
| within-fold, pair-weighted | 0.942 | 0.763 | 0.832 | **-0.069** |
| unweighted fold mean | 0.879 | 0.767 | 0.402 | **+0.365** |

**Quote all three or none.** The RF's margin over pure geography *changes sign* between them.
Pooled AUC is not a held-out metric when fold base rates run from 0.008 to 0.590: it pays a
model for ranking one whole block above another, which the base rates alone already do. The
within-fold figure removes that, but it is 97% fold 1 (see above), so it is not a cure either.

The CNN is ahead of the RF under all three, which is the strongest thing that can be said for
it here. It is *not* clean evidence that a CNN is the right architecture — the comparison
changed two variables at once, input representation *and* model class. See
[LIMITATIONS.md](LIMITATIONS.md).

## Operating point: none ships, and here is why

**`--thresh-rf` and `--thresh-cnn` are required arguments to `src/bio_score_tiles.py`**, each when
its model runs. There is no default, adding one would be wrong, and they are two flags rather
than one because the table below is what a *shared* number would be averaging over. What a fixed
cut actually does on held-out ground:

| model | thr | fold 0 (base 0.008) | fold 1 (base 0.590) | fold 2 (base 0.043) |
|---|---:|---|---|---|
| CNN B | 0.50 | 35 flagged, prec 0.000, **lift 0.00x** | 299 flagged, prec 0.987, **1.67x** | 36 flagged, prec 0.139, **3.23x** |
| CNN B | 0.65 | 24 flagged, prec 0.000, **0.00x** | 244 flagged, prec 0.984, **1.67x** | 19 flagged, prec 0.105, **2.45x** |
| CNN B | 0.80 | 14 flagged, prec 0.000, **0.00x** | 175 flagged, prec 0.977, **1.66x** | 11 flagged, prec 0.091, **2.11x** |
| RF | 0.50 | **0 flagged** — precision undefined | 572 flagged, prec 0.781, **1.32x** | **0 flagged** |
| RF | 0.65 | **0 flagged** | 460 flagged, prec 0.791, **1.34x** | **0 flagged** |
| RF | 0.80 | **0 flagged** | 271 flagged, prec 0.801, **1.36x** | **0 flagged** |

Three things to take from this table:

1. **Precision at a fixed cut tracks the local base rate, not skill.** Training folds are
   26-58% positive, so the probability scale is fitted to ground that is roughly half crevassed
   and floods when applied to ground that is 0.8% crevassed. **The ranking transfers; the
   calibration does not.**
2. **Lift over base rate is the comparable quantity**, and it is what the training script
   prints. On the two low-prevalence folds the CNN's lift is 0.00x and 2.45x — one useless,
   one genuinely useful, at the same threshold.
3. **The two models fail differently, in a way AUC hides.** On low-prevalence folds the CNN
   flags a handful of tiles and gets them all wrong; the RF flags **nothing at all**. Both are
   the same miscalibration pointing in opposite directions. A cell reading "0 flagged" is
   precision *undefined*, not zero — `bio_train_gate.py` prints a dash there deliberately,
   because `0.00x` would read as "flagged everything wrong".

The frequently repeated "P 0.84 / R 0.42 at 0.65" figure is **fold 1's number**, and fold 1 is
59% positive. Do not carry it to new ground.

Consequently `data/bio_gate_t512.joblib` **carries no `thresh` key**, and
`src/bio_score_tiles.py` hard-exits if it finds one in a bundle you hand it — an earlier
version of the trainer wrote the best-F1 threshold (0.25) over *unbuffered* out-of-fold scores,
and a bundle field named `thresh` reads as a shipped operating point no matter what the docs
say.

## The two models rank the same map

On a full-data fit over all 4767 tiles:

- correlation **+0.953**
- agreement at P >= 0.65: **94.46%** (RF flags 2025 tiles, CNN 1969)

On the 1747 held-out tiles the agreement is looser — correlation 0.616, 80.9% agreement at
0.65 — which is the more honest figure and is what the per-fold differences above are made of.
Still: this is **one map ranked two ways**, and the CNN's contribution is a better *ordering*,
not a different set of detections.

`--method both` writes both orderings — `p_rf` and `p_cnn`, separate columns, cut at separate
thresholds, **with nothing derived from the pair**. No mean, no vote, no agreement column. The
+0.953 above is a full-fit number and 0.616 is the held-out one; combining two scales that
disagree that much between fits, and that flag 2025 vs 1969 tiles at the same nominal cut, would
be arithmetic on incommensurable numbers. Compute an agreement rule from the two columns if you
want one, and own the choice.

## Baselines, for calibration of expectations

| baseline | AUC |
|---|---:|
| `ratio_dB` alone, buffered pooled | 0.639 |
| `ratio_dB` alone, over six granules pooled unbuffered | 0.764 |
| `ratio_dB` alone, within a single granule | 0.866 |
| `ratio_dB` alone, on the original 120-tile pilot | 0.890 |

That descent from 0.890 to 0.639 as the evaluation gets stricter and the granule count grows
is **per-scene radiometric calibration drift**, not a modelling failure — it is the single
strongest reason to be sceptical of any absolute number here. It is recorded as an open item
in [LIMITATIONS.md](LIMITATIONS.md) rather than fixed.

## Reproducing all of it

```bash
scripts/run_rf_training.sh                 # reprints every RF number above, ~4 min
```

The CNN's per-fold AUCs and the whole lift table can be recomputed **without torch and without
the unshipped chip cache**, from the held-out scores in `data/bio_cnn_oof_c128.npz`; the recipe
is control 4 in [CONTRIBUTING.md](CONTRIBUTING.md).
