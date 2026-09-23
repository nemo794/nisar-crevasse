"""Pure-logic tests for `UNetSegmenter`/`BiomassUNetSegmenter`'s input-shape contract --
the `(N, 1, 512, 512)` / `(N, 4, 512, 512)` convention and its guard rails. Synthetic
tiles only; loads the real shipped default checkpoint (small, local, no granule needed).
"""
import numpy as np
import pytest
from crevasse.nisar.unet_predict import UNetSegmenter
from crevasse.biomass.bio_unet_predict import BiomassUNetSegmenter


def test_nisar_single_tile_promotes_to_batch():
    seg = UNetSegmenter()
    single = np.zeros((1, 512, 512), dtype=np.float32)  # (1, 512, 512), no batch dim
    out = seg.predict_proba(single)
    assert out.shape == (1, 512, 512)


def test_nisar_batch_shape_round_trips():
    seg = UNetSegmenter()
    batch = np.zeros((3, 1, 512, 512), dtype=np.float32)
    out = seg.predict_proba(batch)
    assert out.shape == (3, 512, 512)  # output has no channel axis


def test_nisar_wrong_channel_count_raises():
    seg = UNetSegmenter()
    bad = np.zeros((1, 2, 512, 512), dtype=np.float32)
    with pytest.raises(ValueError, match="single channel"):
        seg.predict_proba(bad)


def test_nisar_wrong_tile_size_raises():
    seg = UNetSegmenter()
    bad = np.zeros((1, 1, 256, 256), dtype=np.float32)
    with pytest.raises(ValueError, match="512"):
        seg.predict_proba(bad)


def test_nisar_negative_value_raises():
    seg = UNetSegmenter()
    bad = -np.ones((1, 1, 512, 512), dtype=np.float32)
    with pytest.raises(ValueError, match="LINEAR amplitude"):
        seg.predict_proba(bad)


def test_biomass_batch_shape_round_trips():
    seg = BiomassUNetSegmenter()
    batch = np.zeros((2, 4, 512, 512), dtype=np.float32)
    out = seg.predict_proba(batch)
    assert out.shape == (2, 512, 512)
