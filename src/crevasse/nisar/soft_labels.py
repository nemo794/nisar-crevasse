"""Soft (continuous) crevasse pseudo-label target. Production home of the formulation.

    target = clip((frangi_response - LO)/(HI - LO), 0, 1) * rect(orient_agree, t)
             \-------------- "soft B", the localizer --------/   \- the discriminator -/

WHY NOT A BINARY MASK, WHICH IS WHAT THIS REPLACES. `clean_ridge_mask` thresholded the
Frangi response at the 70th percentile of its OWN nonzero values, which keeps ~30% of it
by construction -- measured coverage came out 23.4/24.4/23.9% on three tiles with nothing
in common. It was a per-tile quota, so all discrimination lived in the two tile gates and
none in the mask. The 09-08 absolute-floor sweep confirmed no fixed floor repairs this:
mask-coverage AUC as a crevasse-vs-none discriminator peaked at 0.639 hand / 0.556 AE at
floor=0 and FELL as the floor rose.

WHY THE SCALE IS GLOBAL AND NOT PER-TILE. HI=0.48 is the measured median p99 of the
response over 341 tiles (IQR 0.430-0.562, ratio 1.22), so one divisor is coherent across
tiles despite the two stacked per-tile normalizations upstream. Per-tile max-normalizing
was deliberately rejected: it would reinstate the quota in a new costume, guaranteeing
every tile contains a maximal-value crevasse. LO=0.08 suppresses the speckle baseline.

WHY THE ORIENTATION FACTOR IS NOT OPTIONAL. Amplitude alone is ANTI-correlated with the
label. On 238 pool tiles, tile-mean AUC:
    soft A  resp/HI                 0.320 hand / 0.221 AE    INVERTED
    soft B  (resp-LO)/(HI-LO)       0.384      / 0.242       INVERTED
    binary label (the old target)   0.646      / 0.606
Below 0.5 means `none` tiles carry MORE target mass than `crevasse` tiles. Cause:
`preprocess_sar` contrast-stretches each tile by its own log-amplitude p2/p98, so the
emptiest tiles get their residual speckle amplified hardest and end up with more Frangi
mass than a real crevasse field. The old binary label's min_size/eccentricity filters
were quietly repairing that, which is why it beat every amplitude-only soft variant.

A TILE GATE DOES NOT FIX THE INVERSION -- only this per-pixel factor does. Measured
09-09: behind the production `orient_conc >= 0.12` gate, soft-B mass PER KEPT TILE is
0.074 on `crevasse` vs 0.091 on `none` (hand) and 0.058 vs 0.082 (AE). The inversion
survives gating; gating only limits how many inverted tiles enter. Multiplying by
rect(orient_agree, t) reverses it: t=0.6 gives 4.06x (hand) / 1.53x (AE) the right way
round, t=0.8 gives 16.15x / 2.26x.

WHY t=0.7 IS THE DEFAULT. The cost of t is thinning of real lines, and it is small --
REF_POS coverage at >0.2 runs 26.24% (t=0) -> 24.38 (0.6) -> 23.74 (0.7) -> 22.53 (0.8),
against the old binary label's 23.72%. So t=0.7 matches the production label's density
almost exactly while REF_NEG falls from 10.26% to 0.53% (the old binary label put 11.61%
there) and the three visually-empty tiles that defeated plain soft B fall to 1.57/0.94/
0.54%. t is a genuine dial, not a knife-edge: pool AUC is flat over 0.5-0.8.

WHAT THIS IS NOT. rect(orient_agree, t) ALONE scores best of everything measured
(0.953 hand / 0.806 AE) and is NOT a label -- it covers 73% of REF_POS. It is a smooth
tile-level discriminator re-measuring what orient_conc already gates on. Tile-mean AUC is
therefore a necessary-not-sufficient criterion: it rewards smooth tile-level separation,
not localization. Amplitude is what keeps the target thin.

CALIBRATION ANCHORS, for any change to this file. REF_POS 025_091 r44256 c26256 and
REF_NEG 025_091 r52256 c24256. At t=0.7 they must come out ~23.7% and ~0.5% coverage at
>0.2. If REF_POS drops much below the old binary label's 23.72% the target is deleting
crevasses, and if REF_NEG rises the orientation factor is not doing its job.

THE SECOND RULE, `multiscale_soft_target` (adopted 2026-09-10). mean of soft(resp_s) over
s in {4,2,1}, NO orientation factor. Measured on the same two anchors, beside the f4
values above, at >0.2:
    REF_POS   23.74%  ->  21.86%
    REF_NEG    0.53%  ->   9.78%      <-- reported, NOT a bound. See below.
REF_NEG is back near the 10.26% it sat at before the orientation factor existed. That is
the KNOWN AND ACCEPTED cost of dropping rect: the factor was introduced precisely to fix
this, and the decision was taken anyway because on sparse ground it also over-suppresses
real structure -- on the two low-orient_conc tiles AlphaEarth calls crevassed at 0.94 and
0.65 positive fraction, the f4 rule puts 2.19% and 1.08% coverage and mean{4,2} soft*rect
adds nothing, while mean{4,2,1} soft-only recovers 32.4% / 29.9% of the U-Net's
far-from-label detections. AlphaEarth is independent of Frangi, of amplitude and of the
U-Net, which is what made it admissible. The retraining run prices the cost; a precision
drop is a finding to report, not a reason to silently re-add rect.
WHERE THAT FALSE AREA SITS, which is the lever if it has to be paid down: the mean over
three scales makes the target's MAGNITUDE a scale-agreement count, and on REF_NEG 6.88 of
the 9.78 points lie in (0.2, 1/3] -- structure only one scale of three sees, clearing the
cut only just. At a >1/3 cut REF_NEG would be 2.91% and REF_POS 14.73%. COV_T is
deliberately NOT moved: it is the level every coverage figure and every validate() IoU in
this project is quoted at, and changing the rule and the threshold together would leave
nothing comparable.
"""
import numpy as np
from scipy.ndimage import uniform_filter
from skimage.feature import structure_tensor, structure_tensor_eigenvalues

# HI = measured median p99 of the Frangi response over 341 tiles.
# LO = speckle-baseline suppression point; sits inside the observed spread of the
# per-tile p70 cuts it replaces (0.043 / 0.085 / 0.188 on three decomposed tiles).
HI, LO = 0.48, 0.08

# Rectification point for the orientation factor. See module docstring for the sweep.
RECT_T = 0.7


def soft(resp, lo=LO, hi=HI):
    """Amplitude term ("soft B"). Global scale, deliberately not per-tile."""
    return np.clip((resp - lo) / (hi - lo), 0.0, 1.0)


def rect(x, t):
    """Rectify above a baseline.

    orient_agree sits at mean 0.44-0.49 on `none` tiles and 0.55-0.61 on `crevasse`
    tiles, so the gap is only usable once that floor is removed.
    """
    return np.clip((x - t) / (1.0 - t), 0.0, 1.0)


def orient_agree(R, sigma=2.0, win=41):
    """Per-pixel analogue of orient_conc: the (R*coh)-weighted axial resultant of
    structure-tensor orientation inside a local window, normalized by the weight in that
    same window. 1 = every nearby responding pixel shares one orientation, 0 = random.

    WHY THIS AND NOT COHERENCE. Per-pixel coherence is HIGH on an isolated speckle worm --
    a worm is locally elongated. What distinguishes a crevasse is that its orientation
    AGREES with its neighbours' over distance. That agreement is exactly what orient_conc
    aggregates at tile level (AUC 0.895 as a crevasse-vs-none discriminator, vs 0.49-0.64
    for mask coverage), so this is the same statistic evaluated locally, not a new one.

    Normalizing by the local weight rather than the window area is deliberate: a lone
    genuine lineament sits in a mostly-empty window, and dividing by area would penalize
    it for the emptiness instead of judging its orientation consistency.
    win=41 native px = 205 m, several crevasse widths.
    """
    Axx, Axy, Ayy = structure_tensor(R.astype(np.float32), sigma=sigma,
                                     mode="reflect", order="rc")
    l1, l2 = structure_tensor_eigenvalues((Axx, Axy, Ayy))
    coh = (l1 - l2) / (l1 + l2 + 1e-9)
    theta = 0.5 * np.arctan2(2 * Axy, (Ayy - Axx))
    w = R * coh
    f = lambda x: uniform_filter(x.astype(np.float32), win, mode="reflect")
    num = np.hypot(f(w * np.cos(2 * theta)), f(w * np.sin(2 * theta)))
    return num / (f(w) + 1e-9)


def soft_target(resp, t=RECT_T):
    """The pseudo-label the U-Net regresses. `resp` is the raw Frangi response."""
    return soft(resp) * rect(orient_agree(resp), t)


# ridge_sigmas = [2,4,6,8,10] px are on the DOWNSCALED grid, so the factor slides the
# ridge-WIDTH band at 5 m native pixels: f4 = 40-200 m, f2 = 20-100 m, f1 = 10-50 m.
SCALE_FACTORS = (4, 2, 1)


def multiscale_soft_target(resps):
    """Mean of soft(resp_s) over scales, with NO orientation factor.

    `resps` maps downscale factor -> Frangi response already upsampled to native
    resolution (process_tile does that), so the scales are directly averageable.

    A ridge seen at every scale approaches 1.0; one seen at a single scale caps near
    1/len(resps), which sits just above COV_T -- single-scale structure is deliberately
    left at the binarization edge rather than renormalized up.
    """
    if not resps:
        raise ValueError("multiscale_soft_target needs at least one response")
    return np.mean([soft(r) for r in resps.values()], axis=0)


# Coverage is quoted at >0.2 everywhere in this project's figures and sweeps; the
# dataset's density filter must use the same level or the two are not comparable.
COV_T = 0.2


def coverage(target, thresh=COV_T):
    return float((target > thresh).mean())
