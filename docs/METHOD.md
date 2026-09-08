# Method

## The problem

Per-pixel crevasse segmentation over a full NISAR swath is expensive, and most of a swath
is not crevassed. The gate is a cheap binary filter applied at tile granularity: keep the
tiles that might contain crevasses, discard the rest, and let the expensive stage run only
on what survives.

That framing sets the operating point. A false negative is a crevasse field never looked
at; a false positive is wasted compute. So the threshold is calibrated for **recall**, not
accuracy or F1.

## Tiling

Tiles are 512 × 512 samples at a 5.0 m target ground spacing = **2.56 × 2.56 km**
(`TS = 512`, `TARGET_M = 5.0`).

`read_amp()` scales the read window so a tile always covers the same *ground* distance
regardless of the product's native pixel spacing, then block-averages down to 512. A 5 m
granule is read at 1 look; a 2.5 m granule at 4 looks. Those are not equalised
(`LOOKS_POLICY = "native"`) — an attempt to equalise them was built and made things worse,
because the arms then carried different amounts of information rather than the same
information at different smoothness.

Zeros are nodata, turned to NaN, and NaN-aware averaging keeps nodata from bleeding into
valid pixels. A tile with under 50% valid data is dropped.

**Square pixels are required.** A guard in `tile_scale()` refuses non-square products,
because `read_amp` derives one scale from `transform.a` and a 2.5 × 5.0 m product would
silently produce 2.56 × 5.12 km tiles — giving every scale-dependent feature a built-in
preferred direction. This bit us for real before the guard existed.

## Features — 270 total

The amplitude tile is despeckled before any feature is computed: log10, then NL-means with
its `h` and `sigma` set from a robust (MAD) noise estimate off a median-filter residual,
then normalised to [0, 1]. Speckle *is* texture as far as a GLCM is concerned, so
despeckling first is what makes the texture features measure the scene.

**14 hand features** (`gate_common.glcm_st_features`):
- coefficient of variation
- GLCM contrast, correlation, homogeneity, energy, dissimilarity — each as a mean over
  4 angles × 2 distances, **and** as an anisotropy (std across angles). Crevasse fields
  are directional; the anisotropy terms are where that shows up.
- structure-tensor coherence (σ = 2): mean, 90th percentile, and fraction above 0.5

**256 FFT features** (`fft_pool`): four 384-px Hann-windowed corner windows, each
transformed, binned into a 16 × 16 grid of the magnitude spectrum, then **max-pooled**
across the four windows. Pooling over sub-windows rather than transforming the whole tile
is what makes this work — a crevasse field occupying a quarter of the tile still registers.

Use **raw magnitude bins**, not angular summaries. An earlier version summarised the
spectrum into orientation statistics and concluded "FFT features are dead"; that conclusion
was an artifact of the summarisation, and the raw bins added real skill.

## Classifier

`RandomForestClassifier`, balanced class weights, fixed seed 42. Deliberately not a neural
network:

- 3692 training tiles is small.
- An ImageNet-pretrained ResNet18 was evaluated head-to-head. It beat the 14 hand features
  alone but lost to the 270-feature random forest, and it needed a GPU and PyTorch. It is
  not in this repo, and dropping it is why the environment is torch-free.
- Feature importances are readable, which mattered repeatedly for catching confounds.

## Cross-validation

Two numbers are always printed, and they mean different things:

- **Random-fold OOF** — optimistic. Ice texture is spatially smooth, so a randomly held-out
  tile usually has a training neighbour a few hundred metres away.
- **Spatial-block OOF** — whole 8 × 8 tile (~20 km) blocks held out. This is the number to
  quote.
- **Leave-one-granule-out** — per-granule, also printed. Read the caveat in
  LIMITATIONS.md before believing it: the two training granules are the same frame five
  days apart, so LOGO here is closer to a repeat-pass consistency check than a transfer
  test.

## Threshold

0.333, set as the highest threshold achieving recall ≥ 0.95 on out-of-fold predictions.

Two honest caveats. First, 0.95 was chosen as a plausible round number and **never priced
against the actual downstream cost** of a missed crevasse field versus a wasted
segmentation. Second, the threshold is a property of the *bundle*, not the scores — every
tile's probability is computed once and stored, so changing the threshold is a re-render,
not a re-score.

## Why context is not a feature

Ice type (Bedmap3) and surface velocity (ITS_LIVE) are strong predictors of crevassing, and
adding them as features raises AUC. **They are still not used as features here**, on
purpose.

The reason is a confound that took a while to find. Negatives were originally drawn with a
spatial buffer from the positives, which made them systematically slower-moving ice —
velocity *alone* scored AUC 0.866. Fixing the sampling to speed-match the negatives killed
most of that. But the context prior survived even then, so it was tested against the
control that settles it: **raw `(row, col)` tile indices as features**. Geography alone
scored +0.031, more than the context prior's +0.027. The prior was memorising *where*
crevasses are in this scene, not learning what they look like — which is worthless on a new
scene and actively misleading in a transfer test.

So context is used for two things only: constructing labels (speed matching, grounded-ice
filtering) and optional hard masking at scoring time (`--grounded-only`).

**Run the `(row, col)` control on any new label set.** It is cheap and it is the only thing
that reliably catches this class of error.

## Downstream

The gate's output feeds a Frangi ridge-filter stage that generates per-pixel pseudo-labels,
which train a U-Net. Neither is shipped here — the ridge stage's parameters are still
sensitive to tile size and percentile choice, and the U-Net has not had a real training
run. `docs/lessons/` covers what is known about both.
