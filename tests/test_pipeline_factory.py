"""Pure-logic tests for `crevasse.common.pipeline.Pipeline` -- construction, error
handling, and the module-isolation guarantee that lets both sensors be used in one
process. Uses the real shipped default models (small, already in `models/`), never a
real granule -- these tests never touch `data/`.

Requires `pip install -e .` from the repo root first (see `tests/README.md`).
"""
import pytest
from crevasse import Pipeline


def test_invalid_sensor_raises():
    with pytest.raises(ValueError, match="sensor must be 'nisar' or 'biomass'"):
        Pipeline("bogus")


def test_nisar_pipeline_constructs():
    pipe = Pipeline("nisar")
    assert type(pipe).__name__ == "CrevassePipeline"
    assert hasattr(pipe, "run")


def test_biomass_pipeline_constructs():
    pipe = Pipeline("biomass")
    assert type(pipe).__name__ == "BiomassCrevassePipeline"
    assert hasattr(pipe, "run")


def test_kwargs_pass_through():
    """gate_thresh (NISAR) / gate_method (BIOMASS) reach the underlying pipeline's own
    constructor unchanged -- Pipeline() does not intercept or rename anything."""
    pipe_n = Pipeline("nisar", gate_thresh=0.5)
    assert pipe_n.gate_thresh == 0.5

    pipe_b = Pipeline("biomass", gate_method="cnn")
    assert pipe_b.gate_method == "cnn"


def test_both_sensors_in_one_process_no_collision():
    """`nisar/` and `biomass/` share coordinator filenames (`pipeline_predict.py`, etc.)
    by design. Before this repo was a real installed package, that meant a naive bare
    import of both in one process would have the second shadow the first in
    `sys.modules` -- `Pipeline()` used to work around that with a private-module-name
    loader. Now both live under distinct dotted package names
    (`crevasse.nisar.pipeline_predict` vs `crevasse.biomass.pipeline_predict`), so the
    collision cannot happen at all; this just confirms that."""
    n1 = Pipeline("nisar")
    b1 = Pipeline("biomass")
    n2 = Pipeline("nisar")
    b2 = Pipeline("biomass")

    assert type(n1).__name__ == type(n2).__name__ == "CrevassePipeline"
    assert type(b1).__name__ == type(b2).__name__ == "BiomassCrevassePipeline"
    assert type(n1).__module__ == "crevasse.nisar.pipeline_predict"
    assert type(b1).__module__ == "crevasse.biomass.pipeline_predict"
