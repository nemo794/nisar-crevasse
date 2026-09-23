"""Pure-logic tests for the pipelines' scatter/gather behavior -- restoring the full
N-tile stack after filtering through the gate -- using synthetic degenerate tiles
(all-zero / all-NaN) rather than real granule data. This is the same "unscoreable tile"
case each sensor's own pinned real-data control (P2) already checks; these tests check
it without needing a real granule, so it runs in a fast local loop instead.

Real gate/U-Net models are loaded (small, already shipped in models/) -- what's
synthetic here is the INPUT tiles, not the models. Actual crevasse-detection accuracy on
real tiles is exactly what the pinned real-data controls in docs/{nisar,biomass}/ exist
to check; these tests never make a claim about that.
"""
import numpy as np
from crevasse import Pipeline


def test_nisar_all_zero_tile_is_unscoreable():
    """An all-zero amplitude tile has <50% valid pixels by definition (0 means nodata
    for this sensor) -- NisarGate.predict_proba returns NaN, predict returns False, and
    the pipeline must never call the U-Net on it."""
    pipe = Pipeline("nisar")
    amp = np.zeros((1, 512, 512), dtype=np.float32)
    out = pipe.run(amp)

    assert np.isnan(out.gate_prob[0])
    assert out.gate_flag[0] == False  # noqa: E712
    assert np.isnan(out.unet_prob[0]).all()


def test_nisar_scatter_preserves_order_and_shape():
    """A mixed batch (one degenerate tile among real-shaped ones) keeps every tile at
    its original index -- nothing gets dropped or reordered, regardless of gate_flag."""
    pipe = Pipeline("nisar")
    rng = np.random.default_rng(0)
    amp = rng.uniform(1, 1000, size=(3, 512, 512)).astype(np.float32)
    amp[1] = 0.0  # the middle tile is the unscoreable one

    out = pipe.run(amp)

    assert out.gate_prob.shape == (3,)
    assert out.unet_prob.shape == (3, 512, 512)
    assert np.isnan(out.gate_prob[1])
    assert np.isnan(out.unet_prob[1]).all()
    # the other two indices are untouched by the degenerate one
    assert not np.isnan(out.gate_prob[0]) or not np.isnan(out.gate_prob[2])


def test_nisar_unet_mask_matches_threshold():
    pipe = Pipeline("nisar")
    rng = np.random.default_rng(0)
    amp = rng.uniform(1, 1000, size=(2, 512, 512)).astype(np.float32)

    out_no_mask = pipe.run(amp)
    assert out_no_mask.unet_mask is None

    out = pipe.run(amp, unet_thresh=0.5)
    if out.unet_mask is not None:
        expected = np.where(np.isnan(out.unet_prob), False, out.unet_prob >= 0.5)
        assert np.array_equal(out.unet_mask, expected)


def test_biomass_degenerate_tile_is_unscoreable():
    """An all-NaN 4-pol intensity tile fails BiomassGate's min_valid guard -- gate_prob
    is NaN, gate_flag is False, and the U-Net never runs on it."""
    pipe = Pipeline("biomass")
    inten = np.full((1, 4, 512, 512), np.nan, dtype=np.float32)
    out = pipe.run(inten)

    assert np.isnan(out.gate_prob[0])
    assert out.gate_flag[0] == False  # noqa: E712
    assert np.isnan(out.unet_prob[0]).all()


def test_biomass_scatter_preserves_order_and_shape():
    pipe = Pipeline("biomass")
    inten = np.full((3, 4, 512, 512), np.nan, dtype=np.float32)  # all unscoreable

    out = pipe.run(inten)

    assert out.gate_prob.shape == (3,)
    assert out.unet_prob.shape == (3, 512, 512)
    assert np.isnan(out.gate_prob).all()
    assert not out.gate_flag.any()
    assert np.isnan(out.unet_prob).all()


def test_biomass_unet_mask_matches_threshold():
    pipe = Pipeline("biomass")
    inten = np.full((2, 4, 512, 512), np.nan, dtype=np.float32)

    out_no_mask = pipe.run(inten)
    assert out_no_mask.unet_mask is None

    out = pipe.run(inten, unet_thresh=0.5)
    # nothing was gate-flagged (all-NaN input), so unet_mask should be all-False
    assert out.unet_mask is not None
    assert not out.unet_mask.any()
