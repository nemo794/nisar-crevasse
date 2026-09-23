"""The one exposed entry point for either sensor's crevasse-detection pipeline.

    from crevasse import Pipeline
    pipe = Pipeline("nisar")             # -> crevasse.nisar.pipeline_predict.CrevassePipeline()
    pipe = Pipeline("biomass")           # -> crevasse.biomass.pipeline_predict.BiomassCrevassePipeline()

    out = pipe.run(amp)                  # NISAR: amp (N, 512, 512) linear amplitude
    out = pipe.run(inten, x_m=x_m, y_m=y_m)   # BIOMASS: inten (N, 4, 512, 512) linear intensity

Both return the same `PipelineResult` shape (`gate_prob`, `gate_flag`, `unet_prob`,
`unet_mask`) -- see `crevasse/nisar/pipeline_predict.py` / `crevasse/biomass/pipeline_predict.py`'s
own docstrings for the per-sensor `run()` signature, since the two sensors' inputs are
genuinely different (single amplitude band vs 4-pol intensity + coordinates). This
factory unifies *construction*, not the call signature -- unifying `.run()` itself would
mean re-deriving both pipelines' controls, and is a possible future pass, not this one.

`kwargs` pass straight through to the sensor's own `__init__` (e.g. `Pipeline("biomass",
gate_method="cnn")`).

`nisar/` and `biomass/` share four filenames by design (`pipeline_predict.py`,
`run_granule.py`, `plot_granule.py`, `export_geotiff.py` -- each sensor's own code was
merged as a flat, self-contained directory so every file's existing same-directory
imports kept working). Before this repo became a real installed package, that meant both
directories couldn't be on a flat `sys.path` and bare-imported at once without one
shadowing the other in `sys.modules` -- this module used to work around that with a
private-module-name `importlib` loader. Now that both live under proper dotted package
names (`crevasse.nisar.pipeline_predict` vs `crevasse.biomass.pipeline_predict`), that
collision cannot happen and the workaround is gone -- these are plain imports.
"""
_CLASS_NAME = {"nisar": "CrevassePipeline", "biomass": "BiomassCrevassePipeline"}


def Pipeline(sensor, **kwargs):
    """`Pipeline("nisar", **kwargs)` or `Pipeline("biomass", **kwargs)` -> a ready
    pipeline instance. `kwargs` pass through to the sensor's own constructor."""
    if sensor == "nisar":
        from crevasse.nisar.pipeline_predict import CrevassePipeline
        return CrevassePipeline(**kwargs)
    if sensor == "biomass":
        from crevasse.biomass.pipeline_predict import BiomassCrevassePipeline
        return BiomassCrevassePipeline(**kwargs)
    raise ValueError(f"sensor must be 'nisar' or 'biomass', got {sensor!r}")
