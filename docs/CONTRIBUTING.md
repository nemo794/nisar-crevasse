# Adding to this repo

Written so that a change made months from now lands in the same shape as the existing code,
and so the numbers in [RESULTS.md](RESULTS.md) never quietly stop being true.

Read [LIMITATIONS.md](LIMITATIONS.md) first. Most of the rules below exist because something
listed there cost real time.

## Four invariants

Break these and the repo stops meaning what its docs say.

1. **No torch, no GPU.** The lean environment is a deliberate property, not an accident —
   the ResNet18 arm was measured and lost to the 270-feature forest. If a change needs
   torch, it belongs in the research repo. The check: `python -c "import torch"` must keep
   failing in `nisar-gate`.
2. **Context data is never a model feature.** Ice type and velocity may build labels
   (`build_context_masks.py`, `build_aoi_tile_labels.py`) and may hard-mask a decision
   (`--grounded-only`). They may not enter `X`. `--feature-set context|fused` exists to
   reproduce the refutation, not to be used. Reason in METHOD.md.
3. **Every measurement ships with its control.** A number without one is not readable and
   should not be printed, committed, or quoted. This is the convention every existing
   script follows — see `check_granule.py:391` (`cmd_ab`), which prints the known-good pair
   next to the test pair and has an explicit CONTROL-FAILED branch.
4. **Never overwrite `data/gate_5m_freqA_2gran.joblib`.** Train to a new path and compare.
   `scripts/run_training.sh` already defaults to `data/gate_retrained.joblib` for this
   reason.

## Where things go

| you are adding | put it in | notes |
|---|---|---|
| a shared constant, or a feature used by more than one script | `src/gate_common.py` | it is the shared-constants-and-features module; it must stay import-light |
| a new granule check | a subcommand in `src/check_granule.py` | do not make a new top-level script |
| anything reading/writing tiles | reuse `read_amp` / `tile_scale` from `train_gate_classifier.py` | do not re-derive the scale from `transform.a` yourself |
| a library entry point for a caller holding arrays | `src/gate_predict.py` | it reuses `featurize` verbatim and takes column order from the bundle's `hand_keys`, never from a literal. A guard the CLI gets from the raster must be moved here explicitly — that is what `px_m` is for |
| a figure | `src/plot_<thing>.py` | one figure per script, `--out` argument |
| a one-command workflow | `scripts/*.sh` | thin wrapper only — no logic that isn't in `src/` |
| a number | `docs/RESULTS.md`, with its control | |
| a reason a number is narrower than it looks | `docs/LIMITATIONS.md` | |
| a thing you tried that did not work | `docs/lessons/` | see below — these are the most valuable files here |

`src/` is deliberately flat. There are 15 modules; a package hierarchy would buy nothing and
would break the `sys.path.insert(0, 'src')` idiom the scripts use.

**Never hardcode an absolute path.** Everything resolves off `gate_common.ROOT`, which is
`GATE_ROOT` if set and the repo root otherwise (`gate_common.py:28`). A new module gets its
paths the same way:

```python
import gate_common as E
p = E.ROOT / "data" / "something.npz"
```

`map_crevasse_tiles.py` sets `ROOT = T.E.ROOT` rather than deriving its own, so the override
reaches inference. Keep it that way.

## Recipe: adding a granule

**Run the acceptance test before labelling, not after.** This is the whole lesson of
`025_048` — it looked fine, passed every cheap screen, was labelled, was trained on, and
scores 0.627 where a known-good granule scores 0.904 on identical tiles. The label cycle was
wasted.

```bash
ln -s /path/to/NISAR_..._frequencyA_HH_amplitude.tif data/nisar/   # keep the filename
scripts/run_acceptance.sh <new_granule> 025_019
```

Then read the output the way it is written: **steps 1-3 are screens, step 4 is the verdict.**
Screens can convict a granule; they cannot clear one. If step 4 could not run (no overlapping
ground, or no labels yet) you do not know whether the granule works, and passing 1-3 does not
tell you.

Requirements enforced in code, so you will hear about them: square pixels
(`tile_scale`), unique granule ID within `data/nisar/` (`gate_common.granule_rasters`), and
5.0 m spacing unless you pass `--allow-unseen-spacing` (`map_crevasse_tiles.py:186`).

## Recipe: adding a label set

1. Same CSV schema: `granule, row, col, label` with `label` in `{crevasse, none}` and
   `row`/`col` the tile's top-left corner in **native raster pixels**. Rows are deduped by
   `(granule, row, col)` at load.
2. **Speed-match the negatives.** Sampling with a spatial buffer instead makes negatives
   systematically slower ice, and velocity alone then scores AUC 0.866. `--neg-buffer` is
   what *creates* that confound; `--match-neg-speed` is the fix.
3. **Run the `(row, col)` control.** Mandatory, and the only cheap thing that reliably
   catches this class of error. Recipe (there is no subcommand for it yet — see Known gaps):

   ```python
   # scratch script, run from the repo root. Verified to run 2026-09-08.
   import sys; sys.path.insert(0, "src")
   import numpy as np, train_gate_classifier as T, gate_common as E
   from sklearn.metrics import roc_auc_score
   from sklearn.model_selection import StratifiedGroupKFold

   # load_features returns 7 values; meta is a list of (granule, row, col) TUPLES
   X, y, _, meta, _, _, _ = T.load_features(["data/your_new_labels.csv"])
   R = np.array([[r, c] for _, r, c in meta], float)   # geography alone, zero physics
   g = T.spatial_groups(meta)
   for name, M in (("sar", X), ("row,col only", R)):
       oof = np.zeros(len(y))
       for tr, te in StratifiedGroupKFold(5, shuffle=True,
                                          random_state=E.SEED).split(M, y, g):
           oof[te] = T.make_rf().fit(M[tr], y[tr]).predict_proba(M[te])[:, 1]
       print(f"{name:14s} spatial-block AUC {roc_auc_score(y, oof):.3f}")
   ```

   **Read the margin, not the absolute number.** Interpret it this way:

   - **Position near chance** → the CV number means what it says.
   - **Position tying the SAR features** → your blocks are probably smaller than the thing
     you are predicting. Sweep `block_tiles` before concluding anything (below).
   - **Position beating the SAR features** → stop; the labels encode *where* rather than
     *what* and no number from them is readable.

   **The default 8×8 blocking fails this on the shipped label set** — `sar 0.935` vs
   `row,col only` 0.934 — because a 20 km block is smaller than the crevasse field, so a
   held-out block's label is still predictable from where it is. That is a property of the
   labels, not of the forest, which never sees `row`/`col`. So always sweep the block width
   too (swap `T.spatial_groups(meta)` for `T.spatial_groups(meta, bt)` and loop `bt` over
   8, 16, 32, 64). On the shipped set the margin opens at 16 (+0.051) and holds at 32 and 64,
   which is what makes it believable; see the table in RESULTS.md.

   The control is mandatory because it is the only cheap thing that distinguishes those three
   cases — and it is what killed the context prior (+0.031 as a gain). But note what it
   cannot do: it varies the *features* while holding the pixels fixed. To vary the pixels
   while holding position fixed you need the matched-ground A/B (`check_granule.py ab`).
4. Add the CSV to `scripts/run_training.sh` and say in the comment there why it is included
   — that file is where the training set is defined, and `025_048`'s absence is documented
   in it rather than left implicit.

## Recipe: adding or changing a feature

The feature vector is 270 = 14 hand + 256 FFT, assembled in `train_gate_classifier.py`:

- `featurize(amp)` (`:178`) → `(hand_dict, fft_vector)`, both computed on the NL-means
  despeckled image. **Despeckle first** — speckle is texture as far as a GLCM is concerned.
- `E.glcm_st_features(img)` (`gate_common.py:79`) is the 14 hand features.
- `fft_pool(img)` (`:98`) is the 256 FFT bins: four 384-px Hann corner windows,
  **max-pooled**. Pooling over sub-windows is what makes a crevasse field occupying a
  quarter of the tile register. Use raw magnitude bins — an earlier angular *summary*
  produced the false conclusion "FFT features are dead".
- `stack_features` / `feature_names` fix the column order: hand keys first (dict insertion
  order), then `fft0..fft255`.

Two consequences:

- **Changing the feature set invalidates every shipped bundle.** There is no version check
  on feature count, so a stale bundle plus new features fails late and confusingly. Retrain
  to a new path and treat the old bundle as archived.
- **Adding a hand feature means adding a key to the dict in `glcm_st_features`.** That
  shifts nothing else, because the FFT block is appended after. Removing or reordering keys
  does shift things.

Whatever you add, report it against the current 270 on the **spatial-block** number, not the
random-fold one, and keep both in the output — the gap between them is the honest cost of
spatial autocorrelation and it is information.

## Recipe: adding a check to `check_granule.py`

Follow the existing shape: a `cmd_<name>(args)` plus a `sub.add_parser` block in `main()`
(`:437`). Reusable pieces already there: `keyed_labels` (matches tiles across granules by
world grid), `read_tile`, `enl`, `rho1`, `cross_tile_R`, `overlap_bounds`, `blocked_auc`.

Two conventions that are not optional:

- **Print the control beside the measurement**, and if the control fails, say the result is
  unreadable rather than reporting a verdict.
- **Say which statistic you mean when you print "R".** Two different things have been called
  orientation R in this project — cross-tile resultant vs within-tile gradient resultant,
  ~0.46/0.86 vs ~0.013/0.026 for the same two granules.

And know what has already been refuted so you do not rebuild it: a **per-region** orientation
breakdown does *not* discriminate. Known-good 025_019 reaches regional R 0.971 against
failing 025_048's 0.984. The subcommand still prints the table, labelled as context, not a
verdict.

## Bundle protocol

Train to a new path, then compare before promoting anything:

```bash
scripts/run_training.sh                       # -> data/gate_retrained.joblib
python -c "
import joblib
for n in ('gate_5m_freqA_2gran','gate_retrained'):
    b = joblib.load(f'data/{n}.joblib')
    print(n, b['oof_auc_spatial'], b['n_train'], b['threshold'], b['features'])"
```

The bundle carries its own provenance — `labels_csv`, `train_granules`, `train_px_m`,
`train_px_m_positive`, `looks_policy`, `trained_at`, `features`, all three AUCs. Two of
those keys are load-bearing at scoring time and must keep being written:

- `looks_policy` — mismatch against `LOOKS_POLICY` is a hard exit
  (`map_crevasse_tiles.py:156`).
- `train_px_m_positive` — the spacings a *crevasse* was ever seen at. Scoring outside them
  is refused. A granule family that only ever supplied negatives taught the gate what to
  reject, not what to find.

If you promote a new bundle, update the model filename in `map_crevasse_tiles.py:116`,
`scripts/run_inference.sh`, `docs/DATA.md` and `docs/RESULTS.md` together.

## Before you call a change done

Re-run the controls. Each one has a pinned number, and each was verified against the shipped
tree on 2026-09-08:

| # | command | must produce |
|---|---|---|
| 1 | `conda run -n nisar-gate python -c "import torch"` | **fails** |
| 2 | import all of `src/` in `nisar-gate` | no ImportError; if a package is missing the env is wrong, not the code — add it to `environment.yml` |
| 3 | `scripts/run_inference.sh 025_019` | 9236 tiles scored, **2327** at P≥0.333 (also 1476 at 0.65, 1084 at 0.80) |
| 4 | `scripts/run_training.sh` | 3692 tiles, random-fold 0.951, **spatial-block 0.932** (124 blocks), threshold 0.333, LOGO 0.921/0.943 |
| 5 | `python src/check_granule.py ab 025_048 025_019` | 359 tiles, 0.904 vs **0.627** |
| 5b | `python src/check_granule.py ab 025_091 025_019` (control) | both arms near 0.9, delta ≈ +0.004 — a run that fails this is measuring the harness, not the granule |
| 6 | inference from a tree holding only the bundle + one GeoTIFF | works; and `--grounded-only` exits naming `_context_masks_<granule>_t512.npz` |
| 7 | every flag named in `docs/` and `scripts/` | resolves to a real `add_argument` |
| N1 | cut 40 of `025_019`'s tiles with `read_amp`, score them through `NisarGate.predict_proba` | **exactly equal** to `data/maps/gate_sar_025_019.npz` — same `featurize`, same bundle, no resample at 5 m, so any deviation at all is a bug rather than a tolerance. Verified 2026-09-15: maxdiff **0.0** |
| N2 | `NisarGate().threshold`, and `predict()` with no `thresh` | **0.33301181457431456**, mode `recall>=0.95`; over the full `025_019` tiling that cut flags **2327 of 9236**, the same number as control 3 |
| N3 | an all-zero tile and a 60%-nodata tile mixed into a batch of real ones | output length preserved, both unscoreable tiles `NaN` (`False` from `predict`), the real tiles' scores unchanged; `NisarGate(px_m=2.5)` raises naming `train_px_m_positive`, and a negative (dB) array raises rather than scoring |

Steps 3 and 4 are slow (~7 min and ~5 min); run them in the background rather than trimming
them, and don't run them at the same time — they will fight over cores. N1-N3 take under a
minute, but they read the granule, so `data/nisar/` must be present.

**N1 is the control that matters** for `src/gate_predict.py`. The API is additive — the CLI
was deliberately not refactored onto it, so the pinned controls above keep testing the CLI
rather than testing new code twice — which makes exact agreement the only thing tying the two
paths together. It is exact rather than approximate because there is no resample and no
re-derived normalisation anywhere between them.

## Writing a lesson

`docs/lessons/` is the record of what was tried and refuted, and for anyone porting this it
is worth more than the code. Add one when a result **surprised** you — especially when
something that looked physical turned out not to be.

Match the existing files: dated filename `YYYY-MM-DD-short-slug.md`, and an entry in
`docs/lessons/README.md` under both the category list and *Recent Lessons*. Write the
refutation, not just the conclusion — `2026-09-02b-the-row-col-control-kills-a-feature-that-looked-physical.md`
is the model for this. A lesson that only records what worked is half a lesson.

## Porting to BIOMASS

The framework is the deliverable; NISAR is the pathfinder. So when weighing a change, ask
whether it survives a sensor change rather than whether it raises AUC on one Thwaites frame.

Things expected to port: the controls, `check_granule.py`, the tiling-at-fixed-ground-distance
idea, the despeckle-then-texture ordering, the label-construction discipline. Things expected
*not* to port: the specific 0.333 threshold, the 5.0 m spacing guard, the trained forest.

Sensor-specific assumptions live in a small number of places, and these are what to change
first: `TARGET_M`, `TS`, `LOOKS_POLICY` (`train_gate_classifier.py:55-69`), `GRANULE_RE`
(`gate_common.py:29`), and the 384-px window size behind `fft_pool`.

## Known gaps

Honest list, so nobody assumes these exist:

- **The `(row, col)` control has no subcommand**, only the recipe above, even though three
  documents call it mandatory. It belongs in `check_granule.py`.
- **No tests, no CI.** The pinned controls above are the test suite. A synthetic unit-test
  suite would add files without adding evidence.
- **No LICENSE, no git history.** Deliberate — publication is a separate decision.
- Two open research items are recorded in LIMITATIONS.md §2 and §7 rather than fixed: the
  cause of `025_048`'s failure, and why high-confidence positives are enriched *off*
  grounded ice.
