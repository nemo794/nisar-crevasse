# Adding to this repo

Written so that a change made months from now lands in the same shape as the existing code, and
so the numbers in [RESULTS.md](RESULTS.md) never quietly stop being true.

Read [LIMITATIONS.md](LIMITATIONS.md) first. Most of the rules below exist because something
listed there cost real time.

## Five invariants

Break these and the repo stops meaning what its docs say.

1. **The RF path never imports torch.** This is *deliberately divergent* from the NISAR sibling
   repo, whose invariant 1 is the stronger "no torch, no GPU". That cannot hold here, because
   the better of the two BIOMASS gates is a CNN. Rather than drop the guarantee, it is narrowed
   to a **testable env boundary**:

   ```bash
   conda run -n biomass-gate     python -c "import torch"   # must FAIL
   conda run -n biomass-gate-cnn python -c "import torch"    # must succeed
   ```

   If the RF path ever needs `biomass-gate-cnn` to run, something has leaked — find it rather
   than merging the two environment files. Concretely: `bio_gate_controls.py` must stay
   torch-free, which is why `within_fold_auc` lives there and not in `bio_cnn_gate.py`.
2. **No operating point ships, and the two models are never cut with one number.** No `thresh`
   key in any bundle, no default for `--thresh-rf` or `--thresh-cnn`, and `--method both` writes
   `p_rf` and `p_cnn` as separate columns with **no combined score, no vote, no agreement
   column**. At 0.65 the RF flags zero tiles on two of three folds while the CNN flags some on
   all three; averaging those scales is arithmetic on incommensurable numbers. If
   you find yourself adding one, re-read [RESULTS.md](RESULTS.md) "Operating point" — a previous
   version of the trainer did exactly this (best-F1 over unbuffered out-of-fold scores, 0.25)
   and `bio_score_tiles.py` now hard-exits on any bundle carrying the key.
3. **Only the buffered split is quotable.** Unbuffered and leave-one-footprint-out numbers may
   be computed and stored, but only under `*_discredited` names. They exist as the evidence for
   why the buffered ones are the quotable set.
4. **Every measurement ships with its control.** A number without one is not readable and should
   not be printed, committed, or quoted. `bio_train_gate.py` prints the `(x,y)`-only control,
   the `ratio_dB`-alone baseline and the base rate beside every score; `bio_cnn_gate.py` prints
   five arms on identical folds. Follow that.
5. **Never overwrite `data/bio_gate_t512.joblib` or `data/bio_cnn_gate_sarvel`.** Train to a
   new path and compare. `scripts/run_rf_training.sh` defaults to
   `data/bio_gate_retrained.joblib` for this reason. The pinned controls below are pinned to the
   shipped artifacts.

## Where things go

| you are adding | put it in | notes |
|---|---|---|
| anything shared that must stay torch-free | `src/bio_gate_controls.py` | `auc`, `buffered_folds`, `within_fold_auc`, the 300-tree eval forest |
| anything torch | `src/bio_cnn_gate.py` or `src/bio_cnn_infer.py` | nowhere else. Invariant 1. |
| a new granule check | a subcommand in `src/bio_check_granule.py` | do not make a new top-level script |
| a feature | `tile_features()` in `src/bio_tile_features.py` | see the recipe below — column order is a contract |
| a library entry point for a caller holding arrays | `src/bio_predict.py` | it reuses `tile_features` verbatim and takes column order from the bundle's `feature_names`, never from a literal. **`import torch` stays inside the CNN branch** — invariant 1 is about the RF path, so `import bio_predict` must keep working in `biomass-gate`; control B3 pins this |
| a figure | `src/bio_<thing>.py` | one figure per script, `--out` argument |
| a one-command workflow | `scripts/*.sh` | thin wrapper only — no logic that isn't in `src/` |
| anything that reads the labels | behind an `os.path.exists` guard | the mosaic is 7.1 GB and not shipped; a new granule must build features and score without it |
| a number | `docs/RESULTS.md`, with its control | |
| a reason a number is narrower than it looks | `docs/LIMITATIONS.md` | |

`src/crevasse/biomass/` is deliberately flat, every module keeping its `bio_` prefix. The
prefix is not decoration: this directory sits beside `src/crevasse/nisar/` in the same
installed package, and the prefix is what keeps every filename but the four coordinator
files (`pipeline_predict.py`, `run_granule.py`, `plot_granule.py`, `export_geotiff.py`,
which stay bare on purpose) unambiguous between the two. **Do not rename them.** Import
as `from crevasse.biomass.bio_predict import BiomassGate` etc. -- no `sys.path.insert`
needed, this repo is a real installed package (`pip install -e .`, see the top-level
README).

**Never hardcode an absolute path.** Every module resolves off its own location:

```python
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "data", "biomass", "gate")
```

The repo must run end-to-end from a fresh clone in a scratch directory. That is the path a
stranger takes and it is what catches an absolute path nobody noticed.

## Recipe: changing a feature

The vector is 38, assembled in `tile_features()`: 20 radiometric then 18 PSF-axis texture, and
`feature_names` fixes the column order.

- **Texture must be computed along and across the PSF axes** (45 deg / 135 deg), never
  isotropically. Isotropic GLCM texture scored at chance on this sensor and the scikit-image
  dependency went with it. `_dir_acf` and `_dir_grad` are the two helpers; both take a
  log-domain tile.
- **Changing the feature set invalidates every shipped bundle.** There is no version check on
  feature count, so a stale bundle plus new features fails late and confusingly. Train to a new
  path and treat the old bundle as archived.
- Report any addition against the current 38 on the **buffered** number, with all three
  summaries, and keep the `(x,y)` control beside it.

## Recipe: changing the CNN

- **180 degree rotation is the only admissible augmentation.** Reason in
  [METHOD.md](METHOD.md); a flip or quarter turn teaches the net that the 3:1 PSF asymmetry is
  arbitrary.
- **No early stopping.** Three folds; stopping on the test block leaks it. Fix the epoch count
  in advance and give every arm the same one. Report seed spread instead.
- **Reduce float16 chips with an explicit `dtype=np.float64`.** A float16 `nanmean` over ~10^8
  values overflows and silently returns `sd = nan`.
- **Normalisation constants travel with the weights.** If you change the head or the input
  channels, bump what `train_fold(..., return_net=True)` returns and make `score_with_net`
  read it. Re-deriving `mu`/`sd` from the array being scored would make a tile's probability
  depend on what it was scored beside.
- A checkpoint records `arm` and `chip`, and `bio_score_tiles.py` refuses a mismatch on either,
  because the head width differs by arm and a silent mis-score is the failure mode.

## Bundle protocol

Train to a new path, then compare before promoting:

```bash
scripts/run_rf_training.sh                        # -> data/bio_gate_retrained.joblib
conda run -n biomass-gate python -c "
import joblib
for n in ('bio_gate_t512','bio_gate_retrained'):
    b = joblib.load(f'data/{n}.joblib')
    print(n, b['fold_auc'], b['cv_auc'], b['n_train'], 'thresh' in b)"
```

The bundle carries its own provenance. Keys that must keep being written:

- **`fold_auc`, `fold_pos`, `fold_n`, `fold_base_rate`, `fold_pairs`** — the per-fold table is
  the primary result; the pooled number is not readable without it.
- **`cv_auc_within_fold`, `cv_auc_unweighted_fold_mean`** and both `*_xy_control` partners — all
  three summaries, because they disagree on the sign of the margin.
- **`*_unbuffered_discredited`, `lofo_*_discredited`** — named so they cannot be quoted by
  accident.
- **`eval_estimator` / `shipped_estimator`** — the eval forest is 300 trees
  (`bio_gate_controls.rf`), the shipped one is 500. Swapping them moves the quoted numbers by
  more than the margin being reported.
- **`sklearn_version` / `numpy_version`** — a pickled forest is version-coupled;
  `bio_score_tiles.py` prints a note naming both versions on a mismatch.
- **`trained_granules`** — with `.tif` stripped. One granule ID upstream keeps a stray suffix,
  and both the RF bundle and the CNN checkpoint strip it so a string match cannot miss that
  granule.

There must be **no `thresh` key**. Invariant 2.

## Before you call a change done

Re-run the controls. Each has a pinned number, and each was verified against the shipped tree
on 2026-09-15.

| # | command | must produce |
|---|---|---|
| 1 | `conda run -n biomass-gate python -c "import torch"` | **ModuleNotFoundError** |
| 2 | `conda run -n biomass-gate-cnn python -c "import torch"` | succeeds |
| 3 | `scripts/run_rf_training.sh` | 4244 tiles, base 0.459; buffered 3 folds n=1747; per-fold **0.621 / 0.762 / 0.918** with `(x,y)` at 0.084 / 0.847 / 0.276; pooled **0.889** vs 0.601; within-fold 0.763 vs 0.832; fold mean 0.767 vs 0.402; **0 tiles flagged on folds 0 and 2** at 0.5/0.65/0.8 |
| 4 | control 4 recipe below (numpy only, no torch, no chip cache) | CNN B per-fold **0.806 / 0.944 / 0.888**; lift at 0.65 **0.00x / 1.67x / 2.45x** |
| 5 | `scripts/run_rf_inference.sh 0.65 /tmp/t.csv` | 4767 rows, **2025** flagged (42.5%), header `granule,row,col,x_m,y_m,p_rf,flag_rf` |
| 6 | `python src/crevasse/biomass/bio_score_tiles.py --method rf --out /tmp/x.csv` (no threshold) | exits **1** with `error: --thresh-rf is required. There is no defensible default for BIOMASS.` plus the lift line. The wrapper stops earlier, with its own usage error — check the Python level, since that is what a caller bypassing `scripts/` hits |
| 7 | bundle keys (recipe under Bundle protocol) | `'thresh' in b` is **False**; `cv_auc` is 0.889, not 0.954; discredited numbers present only under their `*_discredited` names |
| 8 | checkpoint loads in `biomass-gate-cnn` | `arm=sar+vel`, `chip=128`, 3 nets, `in_sample_auc` ≈ 0.991/0.991/0.993, 6 granule IDs with **no `.tif`** |
| 9 | rebuild features for the `M03` granule with `--aoi /nonexistent/...`, join to the shipped table on `(granule,row,col)` | `aoi_pos` all NaN, `aoi_inbox` all False; **399** tiles rebuilt vs **398** shipped, intersection **395**; the **38 feature columns bit-identical** on the intersection, `x_m`/`y_m` exact. `grounded_frac` differs by up to **0.046875** — the prescan now reads the 2560 m grid, so set equality is *not* the claim, column equality is |
| 10 | `--method both --thresh-rf 0.65 --thresh-cnn 0.65` on the shipped features + chip cache | header `granule,row,col,x_m,y_m,p_rf,p_cnn,flag_rf,flag_cnn`; `p_rf` **identical** to the `--method rf` CSV; `max abs(p_cnn - p_full)` = **5.0e-5** over 4767 tiles; RF **2025** / CNN **1969** flagged, agreement **0.9446**, corr **0.9534** |
| 11 | `--method both --thresh-rf 0.65` alone; `--method rf --thresh-cnn 0.65`; `--no-thresh --thresh-rf 0.65` | each exits **1**, naming `--thresh-cnn` as required / `--thresh-cnn given but --method rf does not run the CNN` / the mutually-exclusive error. `--no-thresh` writes `p_*` and **no `flag_*` columns** |
| B1 | cut 48 real tiles (8 per granule x 6) into `(N,4,512,512)` linear intensity, `BiomassGate().predict_proba(inten)["rf"]` | **bit-identical** to the shipped `p_rf`, maxdiff **0.000e+00**. Verified 2026-09-15 in **both** envs — the RF half must pass in torch-free `biomass-gate` too |
| B2 | same tiles, `method="cnn"` with `x_m`/`y_m` from the features npz | vs `data/bio_cnn_prob_all.npz` `p_full`: maxdiff **1.290e-04**, median **5.774e-08**, corr **1.000000**. Not exact, and cannot be: the chips are a numpy block mean, GDAL's `Resampling.average` is not, on 4x4 blocks straddling nodata. See LIMITATIONS.md 12 |
| B3 | `conda run -n biomass-gate python -c "import bio_predict, sys; assert 'torch' not in sys.modules"` | succeeds, **and** the RF scores in that env with torch still unloaded. This is invariant 1 for the library path |
| B4 | eleven error paths: `predict` with no thresholds; `both` with only `thresh_rf`; `cnn` with no `x_m`; `both` with no `y_m`; wrong-length coords; a negative (dB) array; 3 channels; 256 px tiles; `px_m=2.5`; `method="rf+cnn"`; coords off the context grid | each a **`ValueError` naming the specific thing**, never a `KeyError` or an `IndexError`. Plus N-in-N-out: an all-zero and a 78%-nodata tile mixed into a real batch come back `NaN` (`False` from `predict`) with the real tiles' scores unchanged, and the same tiles are unscoreable under **both** methods |

Control 3 takes ~4 min; run it in the background. B1-B4 take under a minute but read the
granule GeoTIFFs, so they need `data/biomass/`; B2 additionally needs the chip cache.

**B1 is the control that matters** for `src/bio_predict.py`. The API is additive — the CLI was
deliberately not refactored onto it, so controls 1-11 keep testing the CLI rather than testing
new code twice — which makes bit-identity on the RF the only thing tying the two paths together.
It is exact where B2 is not because the RF path shares `tile_features` verbatim and resamples
nothing.

### Two controls that cannot run from a clean checkout, and are listed anyway

Being explicit is the point — pinning a control nobody can run is worse than admitting the gap.

- **CNN scoring from the shipped weights.** Needs the 626 MB chip cache. With it, the
  round-trip is exact: `bio_score_tiles.py --method cnn --weights ... --cache ...` reproduces
  `data/bio_cnn_prob_all.npz`'s `p_full` to within 5e-5 over all 4767 tiles. That is the control
  that proves the weights *and their normalisation constants* both shipped — if the weights load
  but the scores move, the normalisation did not.
- **CNN retrain from scratch.** Same cache requirement, ~20 min. Folds are byte-identical to the
  RF's because both call `buffered_folds` from `bio_gate_controls.py`.
- **The whole `scripts/run_new_granule.sh <dir> both 0.65 ...` path**, which needs the granule
  itself. Verified 2026-09-15 on the `M03` granule: 399 labels-free tiles cached with `--all`,
  then `max abs(p_cnn - p_full)` = **4.97e-5** against the shipped scores on the 395 tiles the
  two tile sets share. That is the control proving a granule can go from delivery to
  probabilities with no label data present.

### Control 4 recipe

Recomputes the CNN's whole held-out story from shipped files only:

```python
# from the repo root, in biomass-gate (no torch needed). Verified 2026-09-15.
import sys; sys.path.insert(0, "src")
import numpy as np
from bio_gate_controls import auc, buffered_folds, within_fold_auc

z = np.load("data/bio_cnn_oof_c128.npz", allow_pickle=True)
y = z["y"].astype(bool)
folds = list(buffered_folds(y, z["x_m"], z["y_m"], 32 * 512 * 5.0, 16 * 512 * 5.0))
for arm in ("oof_B_sar+vel", "oof_A_sar", "oof_RF_merged", "oof_C_vel", "oof_XY_coords"):
    s = z[arm]; m = np.isfinite(s)
    per = [auc(y[te], s[te]) for te, _ in folds]
    print(f"{arm:16s}", " ".join(f"{v:.3f}" for v in per),
          f" pooled {auc(y[m], s[m]):.3f}  within {within_fold_auc(y, s, folds):.3f}"
          f"  mean {np.mean(per):.3f}")

s = z["oof_B_sar+vel"]
for k, (te, _) in enumerate(folds):
    p, yy = s[te] >= 0.65, y[te]
    print(f"fold {k} base {yy.mean():.3f} flagged {p.sum():4d} "
          + (f"lift {yy[p].mean() / yy.mean():.2f}x" if p.any() else "lift -- (undefined)"))
```

## Adding a granule

**Run the acceptance test before labelling, not after.** The whole lesson of `025_048` on the
NISAR sibling: it passed every cheap screen, was labelled, was trained on, and scores 0.627
where a known-good granule scores 0.904 on identical tiles. The label cycle was wasted.

```bash
python src/crevasse/biomass/bio_check_granule.py <granule_dir>
```

Read the output the way it is written: screens can **convict** a granule; they cannot clear one.
Requirements enforced in code, so you will hear about them: EPSG:3031, 5.0 m square pixels,
bounds on exact 2560 m multiples, all four polarizations present.

And do not relax an assert to admit a granule — resample it onto the grid instead.
[DATA.md](DATA.md) says why.

## Adding a control

Two conventions that are not optional:

- **Print the control beside the measurement**, and if the control fails, say the result is
  unreadable rather than reporting a verdict.
- **A control from another sensor is a hypothesis, not a finding.** The clearest case here:
  velocity was a genuine confound on NISAR (AUC 0.866 alone) and is not one on BIOMASS
  (0.518 buffered). Re-run it; do not cite it.

Things already refuted on this sensor, so nobody rebuilds them: isotropic GLCM texture
(chance), the frozen NISAR gate applied directly (AUC 0.560, probabilities collapsed into
0.445-0.538), and `bedmap_class` as a feature (constant by construction).

## Porting from NISAR

Things that ported: the controls, the tiling-at-fixed-ground-distance idea, the
label-construction discipline, `buffered_folds`, the `(x,y)`-only control, the
acceptance-test-before-labelling rule.

Things that did **not**: the trained forest, the 270 texture features, the despeckle-then-GLCM
ordering, the 0.333 threshold, the "no torch" invariant, and the velocity warning.

That ratio is itself the finding — roughly, the *methodology* transferred and none of the
*measurements* did.

## Known gaps

Honest list, so nobody assumes these exist:

- **No tests, no CI.** The pinned controls above are the test suite. A synthetic unit-test suite
  would add files without adding evidence.
- **No LICENSE, no git history.** Deliberate; publication is a separate decision.
- **The CNN path is not runnable from a clean checkout** (chip caches, 558/626 MB).
- **The AlphaEarth export recipe has not been recovered**, so the label set cannot be
  regenerated or extended.
- **Per-granule `ratio_dB` normalisation is not implemented**, and the 0.890 → 0.639 drift in
  [RESULTS.md](RESULTS.md) is the cost.
- Two open research items live in [LIMITATIONS.md](LIMITATIONS.md) §3 and §4 rather than being
  fixed: separating the raw-4-pol effect from the spatial-structure effect, and the fact that
  ice type cannot be tested at all under the current prescan.
