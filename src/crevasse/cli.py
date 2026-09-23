"""Top-level console-script entry points, one per coordinator script that used to be a
sensor-specific CLI (`run_granule.py`, `export_geotiff.py`, `plot_granule.py`) -- each
still exists exactly as before, at `crevasse/{nisar,biomass}/<name>.py`, with the exact
same flags. This module is a thin dispatcher only: the first positional argument picks
the sensor, everything after it is that sensor's own script's normal argv, unchanged.

    crevasse-run-granule    nisar    --granule <path> --check
    crevasse-run-granule    biomass  --granule <dir>  --check
    crevasse-export-geotiff nisar    --granule <path> --out-dir <dir>
    crevasse-export-geotiff biomass  --granule <dir>  --out-dir <dir>
    crevasse-plot-granule   nisar    --granule <path> --check
    crevasse-plot-granule   biomass  --granule <dir>  --check

Registered in `pyproject.toml`'s `[project.scripts]`. Calling
`crevasse.nisar.run_granule.main(argv)` / `crevasse.biomass.run_granule.main(argv)`
(etc.) directly, with an explicit argv list, still works exactly as it did as a plain
script -- this dispatcher doesn't touch `sys.argv` itself, it just slices the sensor
argument off before handing the rest along.
"""
import sys

_SENSORS = ("nisar", "biomass")


def _dispatch(prog, nisar_main, biomass_main):
    argv = sys.argv[1:]
    if not argv or argv[0] not in _SENSORS:
        print(f"usage: {prog} {{nisar,biomass}} [args ...]\n"
              f"       {prog} {{nisar,biomass}} --help   for that sensor's own flags",
              file=sys.stderr)
        raise SystemExit(2)
    sensor, rest = argv[0], argv[1:]
    main = nisar_main if sensor == "nisar" else biomass_main
    main(rest)


def run_granule():
    from crevasse.nisar.run_granule import main as nisar_main
    from crevasse.biomass.run_granule import main as biomass_main
    _dispatch("crevasse-run-granule", nisar_main, biomass_main)


def export_geotiff():
    from crevasse.nisar.export_geotiff import main as nisar_main
    from crevasse.biomass.export_geotiff import main as biomass_main
    _dispatch("crevasse-export-geotiff", nisar_main, biomass_main)


def plot_granule():
    from crevasse.nisar.plot_granule import main as nisar_main
    from crevasse.biomass.plot_granule import main as biomass_main
    _dispatch("crevasse-plot-granule", nisar_main, biomass_main)
