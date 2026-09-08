# 2026-09-02: A label confound usually has a *cause* you can name and delete — and the cleaner-looking map is often just the confound rendered in colour

## The situation

The task was ordinary: add Bedmap3 surface class and ITS_LIVE ice speed to a SAR crevasse
gate, because crevasses genuinely form in fast-flowing grounded ice. The gate already
scored OOF AUC 0.98 on AlphaEarth-derived labels.

The problem: **ice speed alone, with no SAR at all, separated those labels at AUC 0.866.**
Adding velocity as a feature would have raised every headline number while proving nothing.

## The journey

### 1. Measure the trivial baseline before adding the feature

The single most valuable thing done all session took ten lines: score one feature,
velocity, against the labels. 0.866. That number reframed the whole task from "add a
feature" to "figure out why the labels already know the answer."

**A feature you are about to add is also a baseline you can test first.** If it works
alone, you have found a confound, not a feature.

### 2. Don't stop at "the labels are confounded" — find what *makes* them confounded

The obvious next move is to filter: drop slow tiles, set `--min-vel 100`. That was tried
and it does not work — `vel>100` just moves the boundary (0.777), and `vel>250` makes it
*worse* (0.862), because it strips proportionally more negatives than positives.

The move that worked was to ask which knob in the label builder produced this. Sweeping
`--neg-buffer` (the exclusion radius around positives when drawing negatives):

| buffer (tiles) | negatives | negatives >300 m/yr | AUC(velocity alone) |
|---|---|---|---|
| 0 | 1121 | 186 | 0.866 |
| 4 | 486 | 47 | 0.900 |
| 8 | 147 | **2** | 0.986 |
| 12 | 39 | 0 | **1.000** |

Monotonic, and it reaches *exactly* 1.000. That is not a correlation, it is a mechanism:
**fast uncrevassed ice exists only immediately adjacent to the crevasse field.** Any buffer
wide enough to keep AlphaEarth's fuzzy field edges out of the negatives also deletes every
fast negative in the scene, leaving "fast" and "crevassed" perfectly coextensive.

The buffer had been widened in a *previous* session to fix a recall problem. So the two
problems are in direct tension, and the earlier note "AUC 0.984 but the negatives got too
easy" was this same effect seen from the other side without the cause attached.

**A confound introduced by a parameter can be deleted by that parameter.** Filtering after
the fact only reshuffles it.

### 3. Verify the "held-out" set is actually held out — before reporting the number

With speed-matched labels (AUC(vel) 0.51), training on granule 025_019 and testing on
025_091 gave a clean-looking result. Before writing it down: the two granules are the same
orbit five days apart. Measured the ground overlap — **73.4% of the test tiles sit on
training ground.**

The conclusion survived on the 183 genuinely-unseen tiles, so the result stood. But it
would have been reported as stronger than it was.

**"Different granule" is not "different data."** Same-orbit repeats share most of their
footprint. Compare absolute ground coordinates, not file names.

### 4. When the user says a result "looks better," that is a hypothesis, not a conclusion

Two fusion routes were built: a fused random forest, and a scoring-time naive-Bayes prior.
On genuinely unseen ground, against a context-only floor of 0.843: SAR-only 0.954, fused
0.956, prior **0.981**. The prior wins, and its maps look visibly cleaner.

The user observed exactly that — the prior images look better. The tempting response is to
agree, because the AUC agrees. Instead: rank-correlate each map against the velocity field.

| granule | rho(P_sar, log vel) | rho(P_prior, log vel) | delta |
|---|---|---|---|
| 025_019 | +0.296 | **+0.612** | +0.317 |
| 025_091 | +0.418 | +0.628 | +0.211 |
| 003_064 | +0.484 | +0.484 | **+0.000** |

The prior roughly **doubles** the map's agreement with ice speed. So the prior map looks
cleaner substantially because it looks more like the velocity plume — which is precisely
the quantity the whole confound investigation established you cannot use as a crevasse
proxy. The prior is still the right choice, but the AUC is the reason; the appearance is
the confound wearing a different colourmap.

### 5. A model that cannot extrapolate produces a *constant*, and a constant prior is a lie

On 003_064 the prior map looked dramatically cleaner than the SAR map. It was not working
at all. A random forest fit on 025_019 speeds (12–1986 m/yr) applied to a scene whose
maximum speed is 14 m/yr puts every tile in the same leaves: P_ctx collapses to a constant
0.031 (IQR 0.000, vs 0.464 on the training scene).

With P_ctx constant, the naive-Bayes transform is monotonic in P_sar — it is
*algebraically identical* to a stricter SAR threshold. Verified exactly: rank correlation
**1.000000**, and the identical 168 tiles as `--gate-thresh 0.920` with no prior at all.

So the improvement was a scene-level veto whose magnitude was extrapolated from Thwaites,
not evidence about Dronning Maud. It would suppress a genuine crevasse on slow ice just as
hard. `map_crevasse_tiles.py` now warns when the P_ctx IQR is below 0.02 and prints the
equivalent plain threshold, so this cannot masquerade again.

The delta of +0.000 in the table above is the same finding arriving independently, which is
what a real mechanism looks like.

### 6. Sweep the threshold — "stricter" and "looser" may swap places

The prior was believed to be the stricter of the two gates. Counting tiles above a range of
cuts:

| cut | SAR n | prior n |
|---|---|---|
| 0.35 | 5297 | 2711 |
| 0.65 | 2361 | 1545 |
| 0.85 | 1086 | **1084** |
| 0.95 | **294** | **774** |

They cross at 0.85. Below it the prior is stricter; above it it is 2.6× *looser*, because
the naive-Bayes transform inflates fast ice as hard as it suppresses slow ice — it
manufactures high-confidence tiles on the trunk. At 0.85 the SAR gate collapses onto the
shear margins, where crevassing physically belongs; the prior fills the trunk interior.

**One threshold is one point on a curve.** A comparison at a single cut can invert two
tiles over.

## What made the difference

- Scoring the confounding variable **alone**, as a floor, before adding it as a feature.
- Sweeping the label-builder parameter instead of filtering the labels.
- Checking ground overlap in projected coordinates before using the word "held out".
- Treating the sentence "this looks better" as something to measure.
- Sweeping the operating threshold rather than comparing at one.

## Meta-lessons

1. **Any feature worth adding is worth testing alone first.** Its solo score is the floor
   your combined model must beat, and if the solo score is high you have found a confound.
2. **Confounds have causes, and causes are usually parameters.** Trace the pipeline knob
   that produced it. Post-hoc filtering moves a confound around; deleting its cause removes it.
3. **Two fixes can be in direct tension.** The buffer that fixed recall created the speed
   confound. When a past fix looks arbitrary, check what it broke before removing it.
4. **A cleaner map is not evidence.** Held-out metrics are evidence. If a map looks better,
   correlate it against the variable you are worried about before believing it.
5. **Tree models cannot extrapolate; outside their training range they return a constant.**
   Check the interquartile spread of any model output applied to a new domain. Zero spread
   means you have a relabelled threshold, not a prediction.
6. **Compare models across a threshold range, not at one operating point.** Ordering can invert.
7. **Verify "held out" means held out.** For remote sensing: same orbit, days apart, is the
   same ground.

## Where this leaves the mask

Honest loose end, recorded rather than glossed: after exporting the 0.85 prior mask to
GeoTIFF, a two-chip texture comparison came out *backwards* — the mask-positive chip was
**less** oriented than the mask-negative one (structure-tensor coherence 0.37 vs 0.52,
every GLCM anisotropy feature lower), and the pooled FFT summaries were nearly identical.
Two chips is anecdote, and the negative was drawn from a visibly banded part of the scene.
`code/_mask_feature_audit.py` (200 tiles per class, single-feature AUC per interpretable
feature) exists to settle it and has not been run. Until it is, the mask is not verified —
which, given lesson 4, is the whole point.
