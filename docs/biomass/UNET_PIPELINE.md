# The BIOMASS U-Net, end to end

Written the way `nisar-crevasse-unet/docs/PIPELINE.md` is — for a cold reader, human or
LLM — but shorter, because BIOMASS's version of this pipeline has one fewer genuinely
hard stage than NISAR's.

## The shape of the problem, and why it differs from NISAR

NISAR's U-Net is trained on a **manufactured** pixel target: nobody has hand-annotated
crevasses, so a Frangi ridge filter on despeckled amplitude stands in for ground truth,
gated by an orientation-agreement rectifier to cancel a documented inversion (empty tiles
otherwise get *more* response than real fields). That whole apparatus —
`soft_labels.py`'s `HI`/`LO`/`RECT_T` constants, the `SCALE_FACTORS` multiscale band,
the orientation structure-tensor gate — exists because there is no real label.

BIOMASS does not have that problem. The AlphaEarth mosaic
(`biomass/data/aois/new/thwaites_mosaic_new.vrt`) is **already a per-pixel byte 0/1
mask**, exactly grid-aligned to every BIOMASS SAR tile at an integer pixel offset
(verified: `bio_build_shard.py`'s target read matches
`biomass-crevasse-gate`'s own `aoi_pos` aggregate to `0.0` max abs diff). So the
target here is real, if noisy (AlphaEarth recall ~0.42 at precision ~0.94) — there is no
soft rectifier, no discriminator term, no scale band to calibrate.

This is also a documented, deliberate choice, not an oversight: `bio_check_granule.py`'s
`report_psf()` explicitly states that ridge-based (Frangi) pseudo-labels are **not
admissible** on BIOMASS, because its 5 m grid is oversampled relative to its true
~3:1-anisotropic PSF resolution (lag-1 autocorrelation 0.78 along 45° vs 0.13 along 135°,
against NISAR's 0.03 isotropic at the same declared spacing) — individual 10-30 m
crevasses simply aren't resolvable along the smooth axis. A ridge detector here would
mostly encode the instrument, not the ice.

```
  BIOMASS granule (4 pol GeoTIFFs, quad-pol SCS)
        |
   [0]  bio_tile_features.py (biomass-crevasse-gate) -- per-tile radiometric features +
        AlphaEarth per-tile positive fraction (aoi_pos) + aoi_inbox flag
        |
   [1]  bio_build_shard.py -- for every aoi_inbox tile: bio_tile_cache.granule_chips at
        chip=512 (4-pol dB, no resample) + the AlphaEarth mask itself, read pixel-for-
        pixel at the tile's own footprint -> one packed .npz shard
        |
   [2]  bio_check_labels.py -- RUN THIS before trusting any training number
        |
   [3]  bio_train_unet_supervised.py -- buffered block/buffer split (82 km / 41 km,
        biomass-crevasse-gate's own validated geometry), shared mean/sd normalization
        fit once on train, BCE + soft Dice + flip-consistency
        |
   [4]  bio_eval_shard.py, with its --subset val control; bio_plot_eval.py for figures
```

Stage 0 already exists and is not duplicated here — this repo consumes its output
(`bio_tile_features_t512.npz`) and reads the AlphaEarth mosaic itself for the per-pixel
target, exactly the way `biomass-crevasse-gate` reads it for the per-tile `aoi_pos`.

## [0]/[1] Building a shard

```bash
python src/crevasse/biomass/bio_build_shard.py \
  --features ../biomass-crevasse-gate/data/bio_tile_features_t512.npz \
  --aoi ../biomass/data/aois/new/thwaites_mosaic_new.vrt \
  --base ../biomass/data/biomass \
  --out data/train_biomass_v1.npz
```

### Only `aoi_inbox` tiles, and why that removes the need for a mask

AlphaEarth's export box is smaller than its raster extent — byte `0` **outside** the box
means "never evaluated," not "no crevasse" (`biomass-crevasse-gate/docs/DATA.md`). Rather
than carry a per-pixel in/out-of-box flag through training and mask the loss with it,
`bio_build_shard.py` builds the shard **only** from tiles whose full 512×512 footprint
lies inside the box (`aoi_inbox` in the feature table). Every kept tile's mask is then
trustworthy across its whole area — up to AlphaEarth's own recall/precision, which is a
property of the label, not something a mask fixes. Verified on real data (700-tile
sample, 2026-09-16): every kept tile's mask coverage matched `bio_tile_features.py`'s own
`aoi_pos` to **exactly 0.0** max abs diff, since both read the identical byte-mask window.

### No gate-selection, unlike NISAR

NISAR's gate runs *before* Frangi to avoid manufacturing a pseudo-label out of pure
speckle. BIOMASS's target is real, so there is no equivalent risk, and gate-selection
would only throw away legitimate training tiles (including hard negatives the gate is
unsure about). `bio_build_shard.py` therefore uses every `aoi_inbox` tile regardless of
`biomass-crevasse-gate`'s RF/CNN score.

### Channels

Reuses `biomass-crevasse-gate/src/bio_tile_cache.py`'s `granule_chips` at `chip=512` (no
block-average resampling) — 4 channels, HH/HV/VH/VV order, raw dB, NaN-nodata, no
per-tile or per-channel stretch. A per-channel stretch would destroy the cross/co offset,
which is the one thing this radar carries (`ratio_dB` AUC 0.890 vs `HH_dB` alone 0.710) —
the same reason `bio_tile_cache.py` itself avoids one.

## [2] `bio_check_labels.py` — run this every time

16 checks, each printing its measurement beside the bound it's judged against, nonzero
exit on failure. Two are BIOMASS-specific and were not carried over unchanged from the
NISAR sibling:

- **Targets must be genuinely binary** (`{0, 1}`), not "genuinely continuous" — the
  inverse of NISAR's check, because this target is a real mask, not a soft manufactured
  one.
- **Duplicate tiles are keyed on `(granule, row, col)`**, not `(row, col)` alone. BIOMASS
  has repeat passes (S1/S2/S3 over the same Thwaites ground) that legitimately reuse the
  same tile-grid position under a different granule — treating that as a duplicate would
  be a false positive. (This was caught by the check itself during a 120-tile smoke test
  and fixed before the real control run.)

## [3] Training: the split

`bio_sar_dataset.buffered_split` reuses `biomass-crevasse-gate/src/bio_gate_controls.py`'s
`buffered_folds` geometry — 32-tile (82 km) blocks with a 16-tile (41 km) buffer — rather
than NISAR's grounded-only 3×3-tile scheme, which fixes a grounding-line confound
specific to Thwaites/NISAR's own validation band and has no BIOMASS analogue. 82 km/41 km
is the separation at which BIOMASS's radar features were measured to clearly beat map
coordinates alone (model AUC 0.889 vs `(x,y)`-only 0.601, buffered). Unlike
`buffered_folds` (a k-fold generator), `buffered_split` returns one split: blocks are
added to val (seeded random order) until `--val-frac` of tiles is covered, then every
remaining tile within the buffer distance of any val block is dropped from train.

**Known limitation, carried over from `bio_gate_controls.py`'s own finding:** at this
block scale the labelled BIOMASS ground only supports a handful of blocks (the gate's own
sweep found just 3 usable cross-validation folds at 82 km/41 km). A single train/val
split at whole-block granularity means `--val-frac` is a target, not something the split
can hit exactly, and a small `--max-tiles`-style smoke sample can land almost all its
tiles in one block by chance (observed on a 700-tile test sample: train 9 / val 314).
This gets better on the full shard, where the label density per block is far less lumpy,
but report the actual train/val counts printed at split time rather than assuming
`--val-frac` was achieved.

### Normalization

One shared mean/sd over the selected channels, fit **once** on the training split only
(`bio_sar_dataset.fit_normalization`, `dtype=np.float64` to avoid the overflow-to-nan bug
`bio_cnn_gate.py:110` documents), applied identically to train and val, and shipped in
the checkpoint (`mu`, `sd` keys) so inference never re-derives it from whatever batch
happens to be scored — the same discipline `bio_cnn_gate.py`'s `score_with_net` already
uses for the CNN gate.

### Channels are a training-time flag

`--channels HH,HV,VH,VV` (default, all 4) threads `n_channels=len(channels)` straight
into `UNet(...)`, confirmed a clean, isolated architectural parameter (its only use is
`DoubleConv(n_channels, base_channels)`). If 4 channels don't help, `--channels HH,cross`
or any other subset of the shard's 4 stored channels is a training flag, not a shard
rebuild — this was a deliberate design requirement, not an afterthought.

## [4] Evaluation

Same three-IoU discipline as NISAR (pooled / per-tile / per-batch — only the last
reproduces the training script's printed number, hence the `--subset val` control). One
difference: since the target is a fixed 0/1 mask rather than a continuous soft target,
there is no `target_thresh` to choose — `--sweep` only ever moves the *prediction*
threshold, at a plain `>0.5` cut on the target.

**Verified end to end, 2026-09-16, on a 700-tile real-data sample:** a 2-epoch smoke
training run's `unet_best` recorded `val_iou=0.4109`; `bio_eval_shard.py --subset val
--batch-size 2` (matching the training batch size) reproduced it exactly:
`per-batch IoU 0.4109`, `|d| 0.0000`, **MATCH**. `BiomassUNetSegmenter.predict_proba`
(the array API) reproduced `bio_eval_shard.py`'s own inference on the same 16 tiles to
**max abs diff 0.0** — the equivalent of NISAR's N1/U1 controls, and "the control that
matters" here for the same reason: `bio_unet_predict.py` is additive, not a second
implementation the training path also has to agree with, so exact agreement is the only
thing tying the two together.

## Frangi soft labels, the aux noise branch, and the current default checkpoint

`bio_build_shard_frangi.py` is a second label source (see the README's "Two label
sources" section) — `BiomassGate` pre-filter + `avg(Frangi(HH), Frangi(HV))` soft
targets — built after confirming the plain AlphaEarth path's known ceiling. Training on
it exposed BIOMASS's documented PSF/speckle confound directly in the loss, which
motivated `unet_model.UNet(aux_decoder=True)`: an optional second decoder, sharing the
segmentation encoder, trained as a denoising autoencoder on the theory that a
bottlenecked autoencoder's cheapest way to reduce error is to reproduce the *dominant*
repeating structure (the PSF confound) rather than sparse real crevasse signal.

A `--noise-weight` sweep (2026-09-21) on that branch found:

| noise_weight | pooled IoU @ pred_thresh=0.25 | precision | recall |
|---|---|---|---|
| standard (no aux branch) | 0.277 | 0.361 | 0.544 |
| 0.02 | 0.308 | **0.484** | 0.459 |
| 0.15 | **0.342** | 0.446 | 0.564 |
| 0.17 | 0.301 | 0.374 | **0.604** |

`nw=0.15`/`0.17` win on pooled IoU and recall, but a side-by-side comparison across many
tiles (not just each run's own worst/median/best picks — those barely overlap between
runs) showed **all variants, including the plain standard model, confidently firing on
tiles where the training target is empty but the raw imagery shows crisp, well-defined
linear structure** — consistently across independently-trained models, which argues
against it being any one model's idiosyncratic overfit, but does NOT settle whether that
shared structure is real crevasse ground the label pipeline's gates missed, or a
manifestation of the PSF confound the aux branch was built to work around. AlphaEarth
context was tried as a tie-breaker and retired: it only reports whether a tile contains
crevasse ground at all, not where, and is a different acquisition, so it cannot
adjudicate a specific-pixel question like this one.

**Current default (`bio_unet_predict.DEFAULT_CHECKPOINT`): `nw=0.02`.** Chosen for being
the most conservative of the three aux variants — closest in behavior to the plain
standard model, highest precision — while that PSF-vs-real-crevasse question about the
others' extra recall is still open. Not a claim that `0.02` is the "true" best model;
just the one with the least unverified upside baked into its number. Revisit if the
open question above gets resolved either way.

## Reading any number from this pipeline honestly

AlphaEarth's own precision/recall (~0.94 / ~0.42) sets a ceiling on what any IoU here can
mean — read the per-tile panels (`bio_plot_eval.py`'s `_panels` figure) before trusting a
pooled number, and remember that a "false positive" against this target might be a real
crevasse AlphaEarth simply missed, not a model error.

## What is genuinely unresolved

- **The channel-count question the user asked to explore.** 4 channels is the starting
  point; whether it beats a smaller channel set (or plain HH, or `ratio_dB`-only) is an
  open empirical question this repo is built to make cheap to answer (`--channels`), not
  one this document answers.
- **No second acquisition yet.** Unlike NISAR's `025_091` held-back-acquisition check,
  there is currently one shard spanning every available granule; a genuine
  generalization test needs new BIOMASS ground, the same standing limitation
  `biomass-crevasse-gate` already documents.
- **No MAE pretraining path.** NISAR's `train_mae_pretrain.py` wasn't ported here; add it
  back if self-supervised pretraining turns out to matter for this smaller, noisier label
  set.
- **The few-blocks-at-82km limitation** above is a property of the labelled area, not
  something this repo's code can fix by itself.
- **Whether the aux-branch noise-weight variants' extra recall is real crevasse signal
  or PSF-confound-chasing.** See "Frangi soft labels, the aux noise branch, and the
  current default checkpoint" above — genuinely unresolved, not just under-documented.

## Where the reasoning is written down

- `biomass-crevasse-gate/docs/METHOD.md`, `docs/DATA.md` — the radiometric-not-textural
  finding, the PSF anisotropy measurement, the AlphaEarth label contract.
- `biomass-crevasse-gate/src/bio_check_granule.py` — the Frangi-inadmissibility finding,
  in code, with its measured numbers.
- `nisar-crevasse-unet/docs/PIPELINE.md` — the fuller walkthrough this document is
  deliberately shorter than; read it for anything about the U-Net architecture, the
  training loop shape, or the SLURM job mechanics that transferred unchanged.
