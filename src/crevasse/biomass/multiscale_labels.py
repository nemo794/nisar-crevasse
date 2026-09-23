"""Run the ridge detector at several downscale factors on one tile.

`ridge_sigmas = [2,4,6,8,10]` px are on the DOWNSCALED grid, so
`ridge_downscale_factor` slides the ridge-WIDTH band, not the smoothing: at 5 m native
pixels f4 = 40-200 m, f2 = 20-100 m, f1 = 10-50 m. The production label was f4 only, so
it was blind BY CONSTRUCTION to individual 10-30 m crevasses -- arithmetic, not a
hypothesis. See soft_labels.multiscale_soft_target for what is done with the responses.

`process_tile` upsamples every array it returns back to native resolution, so responses
from different factors are the same shape and directly averageable.

Cost measured 2026-09-10: ~4 s/tile for all three factors together, i.e. the factor does
not dominate the cost. An earlier claim that it did was wrong.
"""
from crevasse.biomass.edge_crevasse_v2 import ImprovedEdgeCrevasseDetector
from crevasse.biomass.edge_crevasse_ppb import PPBCrevasseDetector


def make_detector(speckle_filter: str, detector_params: dict):
    """PPB-family filters need PPBCrevasseDetector; everything else uses the base detector."""
    if speckle_filter.startswith("ppb"):
        return PPBCrevasseDetector(speckle_filter=speckle_filter, **detector_params)
    return ImprovedEdgeCrevasseDetector(speckle_filter=speckle_filter, **detector_params)


def scale_results(tile, factors, speckle_filter, detector_params):
    """{factor: full process_tile result}, one detector run per factor.

    `detector_params` must NOT contain ridge_downscale_factor; it is set per factor here.
    """
    if "ridge_downscale_factor" in detector_params:
        raise ValueError("ridge_downscale_factor is set per factor; drop it from "
                         "detector_params")
    out = {}
    for f in factors:
        det = make_detector(speckle_filter,
                            dict(detector_params, ridge_downscale_factor=f))
        out[f] = det.process_tile(tile)
    return out


def scale_responses(tile, factors, speckle_filter, detector_params):
    """{factor: ridge_response at native resolution}."""
    return {f: r["ridge_response"]
            for f, r in scale_results(tile, factors, speckle_filter,
                                      detector_params).items()}
