# Method

Two gates, side by side, measured on byte-identical folds. Read [RESULTS.md](RESULTS.md) for
the numbers and [LIMITATIONS.md](LIMITATIONS.md) for what they do not cover.

## The sensor, and why it dictates the design

BIOMASS is **P-band (~435 MHz) quad-pol**. Four things about it drove every choice here, and
all four were measured on this data rather than assumed:

1. **It is effectively two channels, not four.** HV and VH correlate at r 0.997-0.999 and HH
   with VV at r 0.990-0.994. So the feature builder merges by hand — `co = (HH+VV)/2`,
   `cross = (HV+VH)/2`, `ratio = cross/co` — and the CNN exists partly to test whether that
   fixed 50/50 merge costs anything.
2. **These are intensity products.** No HH-VV phase difference, so no coherent polarimetric
   decomposition is available. Whatever is here is radiometric or spatial.
3. **The grid is oversampled with a strongly anisotropic PSF.** ~3:1 elongated, long axis near
   50 deg: lag-1 autocorrelation is **0.78 along 45 deg against 0.13 along 135 deg**. This is
   an instrument property, not ground structure, and it is why texture features are computed
   **along and across the PSF axes** rather than isotropically.
4. **The signal is radiometric, not textural.** Every isotropic texture feature scored at
   chance. The frozen NISAR gate, applied unchanged, scored AUC **0.560** with its
   probabilities collapsed into 0.445-0.538 — its 270 GLCM/FFT texture features are out of
   distribution and it fell back to its prior. The discriminating quantity is the
   **cross/co ratio in dB**.

That last point is why this repo trains an **independent** model rather than fine-tuning the
NISAR one. Mixing sensor families is also what produced the leave-one-granule-out collapse to
AUC 0.59/0.35 against a pooled 0.911 on NISAR. A joint model can be revisited once each
sensor works alone.

## Method 1: the 38-feature random forest

`src/bio_tile_features.py` builds the vector; `src/bio_train_gate.py` fits it.

Per tile, on four derived channels (`HH`, `VV`, `co`, `cross`) plus `ratio`:

- **Radiometric** (20): mean dB, log-domain sd, log spread, coefficient of variation for each
  of HH / VV / co / cross; then `ratio_dB`, `hh_vv_dB`, `ratio_logstd`, `ratio_logspread`.
- **PSF-axis texture** (18): for each of HH / cross / ratio, the lag-1 autocorrelation along
  45 deg and 135 deg plus their anisotropy, and the same triple for gradient magnitude —
  `acf45_*`, `acf135_*`, `acfaniso_*`, `grad45_*`, `grad135_*`, `gradaniso_*`.

RandomForest, 500 trees, `min_samples_leaf=2`, `max_features='sqrt'`,
`class_weight='balanced_subsample'`. Importance on the full-data fit:

```
   grad135_cross 0.1205      cross_logspread 0.0836      ratio_dB 0.0645
gradaniso_cross 0.0992         cross_logstd 0.0701
```

The split matters more than the ranking: **cross- and ratio-derived features carry 0.744 of
total importance against 0.095 for HH-only ones**, which is the same conclusion as point 4
above arrived at independently. The 18 PSF-axis texture features carry 0.433 — so the
along-vs-across version *does* do better than the isotropic texture that scored at chance,
but it is not what the model leans on hardest.

## Method 2: the 4-polarization CNN

`src/bio_cnn_gate.py` (five arms, buffered folds), `src/bio_cnn_infer.py` (full-data fit and
the shipped checkpoint).

Input is a **4 x 128 x 128 dB chip** — the four raw polarizations, centre-cropped from the
512 px tile, in dB. Four stride-2 conv blocks, global average pooling, then a linear head.
The `sar+vel` arm concatenates log10 ITS_LIVE speed **after** pooling. 30 epochs, 3 seeds,
Adam, BCE. Reported score is the mean over seeds.

Arms, all on identical folds: `A sar` (chips only), `B sar+vel` (shipped), `C vel` (speed
alone, the control that prices B), `RF` (the incumbent), `XY` (map coordinates, the standing
confound control).

### Three constraints that are easy to get wrong

- **180 degree rotation is the only admissible augmentation.** The usual flip / 90-degree set
  is wrong on this sensor: with a 3:1 PSF at ~50 deg, a horizontal flip or a quarter turn maps
  the along-axis direction onto the across-axis direction and teaches the net that a fixed
  instrument asymmetry is arbitrary. A 180-degree rotation maps both axes onto themselves.
- **Reduce float16 chips with an explicit `dtype=np.float64`.** The cache is float16 to fit in
  memory; a float16 `nanmean` over ~10^8 values overflows and silently returns `sd = nan`,
  which then propagates into every normalised chip without raising anything.
- **No early stopping.** There are three folds. Stopping on the test block leaks it, and there
  is no room to carve a third split out of blocked spatial data, so the epoch count is fixed
  in advance and every arm gets the same one. Run-to-run spread across seeds is reported
  instead, which on three folds is the number that decides whether a difference is real.

### Normalisation travels with the weights

The checkpoint stores per-net `mu`, `sd`, `ctx_mean`, `ctx_std` alongside the `state_dict`,
and `score_with_net` uses *those* rather than re-deriving statistics from whatever array is
being scored. Re-deriving would make a tile's probability depend on which other tiles it
happened to be scored beside. A single shared mean/sd across the four channels is used, not
per-channel — that is what preserves the cross-to-co offset the model actually reads.

## Splits

Blocks are cut in **map coordinates** (`x_m`, `y_m`), never per-granule row/col. This matters
more here than on NISAR: BIOMASS S1 and S2 are exact repeat passes over identical ground, so a
random or per-granule split puts the same ground in train and test through a different
acquisition.

The **only quotable split is the buffered one**: 81.92 km blocks (32 tiles) with a 40.96 km
buffer (16 tiles) withheld from training, giving 3 folds over n = 1747. Two other splits are
computed and kept, but only as the controls that condemn themselves:

- **Unbuffered 40.96 km blocks** — model 0.954 against `(x,y)`-only 0.937, a margin of
  **+0.016**. Thwaites crevasse fields are large and contiguous enough to interpolate across a
  block seam, so this is very nearly pure memorised geography. Kept in the bundle under
  `*_unbuffered_discredited` names so it cannot be quoted by accident.
- **Leave-one-footprint-out** — model 0.953, `(x,y)`-only **0.994**. Coordinates *beat* the
  model, because all three footprints overlap the same region. Not a cross-ground test.

## Labels

AlphaEarth, recall ~0.42 at precision ~0.94. Consequences that are structural rather than
fixable here:

- Zeros **inside** the export box are real evaluated negatives. Byte 0 **outside** it means
  "never evaluated", so those tiles are dropped entirely rather than treated as negative.
- Recall figures from this repo are **lower bounds**, because the positives are incomplete.
- Tiles whose AlphaEarth positive fraction falls strictly between 0.05 and 0.5 are **dropped
  as ambiguous** rather than forced to a side. That takes 4767 tile slots down to 4244
  labelled, 1948 of them positive.

## Three controls that print beside every score

Each one has, at some point on this project, matched or beaten a model that looked good.

| control | why it is there |
|---|---|
| `ratio_dB` alone | one feature. If 38 cannot clear it, the other 37 are decoration. |
| `(x,y)` alone | zero radar input. On NISAR a raw `(row,col)` model tied the real model at 8-tile blocks, which is how the context prior was exposed as memorised geography. |
| base rate | so precision numbers are readable at all. |

A fourth is specific to the CNN: **ice speed alone**. On BIOMASS it scores 0.518 buffered
despite 0.924 pooled, so the buffer already removes the velocity shortcut and the `sar+vel`
arm is not inflated by it. Note the direction — that *reverses* the warning inherited from
NISAR, where velocity was a genuine confound. A control has to be re-run per sensor, not
cited.
