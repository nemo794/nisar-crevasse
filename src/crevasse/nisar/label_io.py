"""Pickle-free I/O for Frangi pseudo-label results: a plain `.npz` (columnar arrays,
`allow_pickle=False`-safe), no pickle anywhere in this pipeline. `save_labels`/
`load_labels` reconstruct the exact `list[dict]` shape the generator
(build_crevasse_labels.py, a different repo) produces and every consumer in this repo
already expects from the old `pickle.load(f)` -- so consumer code changes are limited
to the load/save call itself, same principle as `ckpt_io.py`'s checkpoint migration.

LOSSLESS BY DESIGN. Earlier versions of this module only kept the fields
build_shard.py's downstream readers use (row/col/edge_mask/n_lines/edge_coverage/
orient_conc/gate_prob + the three per-file globals). That's fine for a shard-building
consumer, but wrong for the GENERATOR itself: dropping `binary_mask`/`ridge_stats`/
`texture_coverage`/`ridge_coverage` at the point of generation would make them
permanently unrecoverable without re-running the expensive Frangi step. This version
stores everything build_crevasse_labels.py's `process_tile` produces. Existing
consumers are unaffected -- they only ever read a subset of the returned dict's keys.

Schema: per-tile columns row/col/n_lines/edge_coverage/orient_conc/gate_prob/
texture_coverage/ridge_coverage (one entry per tile) + edge_mask and (optional)
binary_mask (each a stacked (N, H, W) array) + ridge_stats' own 8 scalar fields
(total_components/after_size_filter/after_elongation_filter/n_components/
total_length/coverage/mean_response/orient_conc -- see crevasse_filters.py and
edge_crevasse_v2.py for where each is computed) + GLOBAL scalars constant across the
whole file -- tile_size, soft_rect_t, ridge_factors, orient_rect -- every current
consumer reads tile_size/soft_rect_t/ridge_factors only from element 0 and assumes
they're the same for every tile (a direct consequence of one generator run using one
fixed CLI configuration for every tile it processes), so they're stored once here
instead of repeated per tile.

edge_mask/binary_mask are stored float16: build_shard.py casts edge_mask to float16 on
load anyway ("three orders finer than the 0.2 coverage threshold"), so storing at
float16 costs no consumer any precision it was already discarding, and roughly halves
the file size.

gate_prob/soft_rect_t/texture_coverage/ridge_coverage/binary_mask can be legitimately
absent (None) in the original format; stored as NaN sentinels (or, for binary_mask, a
per-file `has_binary_mask` flag) and restored to None on load, matching what every
consumer's `.get(...)`/`is None` check already expects.
"""
import numpy as np

_RIDGE_STATS_KEYS = ("total_components", "after_size_filter", "after_elongation_filter",
                     "n_components", "total_length", "coverage", "mean_response",
                     "orient_conc")
_RIDGE_STATS_INT_KEYS = {"total_components", "after_size_filter",
                         "after_elongation_filter", "n_components"}


def save_labels(path, results):
    """`results`: list[dict], the shape the (external) Frangi pseudo-label generator
    produces -- see this module's docstring for the full field list."""
    n = len(results)
    if n == 0:
        raise ValueError("save_labels: results is empty, nothing to write")

    row = np.array([r["row"] for r in results], dtype=np.int32)
    col = np.array([r["col"] for r in results], dtype=np.int32)
    edge_mask = np.stack(
        [np.asarray(r["edge_mask"], dtype=np.float16) for r in results], axis=0)
    n_lines = np.array([r.get("n_lines", 0) for r in results], dtype=np.int32)
    edge_coverage = np.array(
        [r.get("edge_coverage", np.nan) for r in results], dtype=np.float32)
    orient_conc = np.array(
        [r.get("orient_conc", np.nan) for r in results], dtype=np.float32)
    gate_prob = np.array(
        [np.nan if r.get("gate_prob") is None else r["gate_prob"] for r in results],
        dtype=np.float32)
    texture_coverage = np.array(
        [r.get("texture_coverage", np.nan) for r in results], dtype=np.float32)
    ridge_coverage = np.array(
        [r.get("ridge_coverage", np.nan) for r in results], dtype=np.float32)

    # binary_mask (and gate_prob) are tracked per-FILE, not per-tile: this module
    # assumes every tile in one `results` list came from one generator run with one
    # fixed CLI config, so binary_mask is either present for every tile (--soft-rect-t
    # given) or None for every tile (binary target requested directly) -- never a mix.
    # True for every real build_crevasse_labels.py run (`alt = binary_mask` is set
    # uniformly from the run's own soft_t, not per-tile). A file that DID mix the two
    # would silently load real per-tile zero-masks as if they were "no mask" for the
    # (any-positive) tiles that came before the first non-None one -- not a case this
    # format needs to support, since it can't occur from the actual generator.
    has_binary_mask = any(r.get("binary_mask") is not None for r in results)
    if has_binary_mask:
        binary_mask = np.stack(
            [np.asarray(r["binary_mask"], dtype=np.float16)
             if r.get("binary_mask") is not None else np.zeros_like(edge_mask[0])
             for r in results], axis=0)
    else:
        binary_mask = np.zeros((0,), dtype=np.float16)

    ridge_stats_cols = {}
    for k in _RIDGE_STATS_KEYS:
        dtype = np.int32 if k in _RIDGE_STATS_INT_KEYS else np.float32
        sentinel = -1 if k in _RIDGE_STATS_INT_KEYS else np.nan
        ridge_stats_cols[f"ridge_stats_{k}"] = np.array(
            [r.get("ridge_stats", {}).get(k, sentinel) for r in results], dtype=dtype)

    tile_size = int(results[0].get("tile_size", 4096))
    soft_rect_t = results[0].get("soft_rect_t")
    ridge_factors = results[0].get("ridge_factors")
    orient_rect = results[0].get("orient_rect")

    np.savez(
        path,
        row=row, col=col, edge_mask=edge_mask, n_lines=n_lines,
        edge_coverage=edge_coverage, orient_conc=orient_conc, gate_prob=gate_prob,
        texture_coverage=texture_coverage, ridge_coverage=ridge_coverage,
        has_binary_mask=np.bool_(has_binary_mask), binary_mask=binary_mask,
        tile_size=np.int32(tile_size),
        soft_rect_t=np.float32(np.nan if soft_rect_t is None else soft_rect_t),
        ridge_factors=np.array(
            ridge_factors if ridge_factors is not None else [], dtype=np.int32),
        orient_rect=np.bool_(bool(orient_rect)) if orient_rect is not None
                    else np.array(np.nan),
        **ridge_stats_cols,
    )


def load_labels(path):
    """Returns `list[dict]`, the same shape every consumer in this repo already
    expects from `pickle.load` (plus the previously-dropped fields -- extra keys are
    harmless to a consumer that only reads a subset)."""
    z = np.load(path, allow_pickle=False)

    # Pull each array out of the NpzFile ONCE -- indexing z["key"] inside the per-tile
    # loop below would re-decompress that entire array from the zip on every single
    # access (N times per key, not once), which is both very slow and, for a
    # multi-thousand-tile file, enough repeated large allocation to exhaust memory.
    row, col = z["row"], z["col"]
    edge_mask = z["edge_mask"]
    n_lines = z["n_lines"]
    edge_coverage = z["edge_coverage"]
    orient_conc = z["orient_conc"]
    gate_prob = z["gate_prob"]
    texture_coverage = z["texture_coverage"] if "texture_coverage" in z.files else None
    ridge_coverage = z["ridge_coverage"] if "ridge_coverage" in z.files else None
    has_binary_mask = bool(z["has_binary_mask"]) if "has_binary_mask" in z.files else False
    binary_mask = z["binary_mask"] if has_binary_mask else None
    n = len(row)

    ridge_stats_cols = {}
    for k in _RIDGE_STATS_KEYS:
        key = f"ridge_stats_{k}"
        ridge_stats_cols[k] = z[key] if key in z.files else None

    tile_size = int(z["tile_size"])
    soft_rect_t = float(z["soft_rect_t"])
    soft_rect_t = None if np.isnan(soft_rect_t) else soft_rect_t
    ridge_factors = z["ridge_factors"]
    ridge_factors = tuple(ridge_factors.tolist()) if ridge_factors.size else None
    orient_rect = None
    if "orient_rect" in z.files:
        orv = z["orient_rect"]
        orient_rect = None if orv.dtype == np.float64 and np.isnan(orv) else bool(orv)

    results = []
    for i in range(n):
        gp = float(gate_prob[i])
        ridge_stats = None
        if all(v is not None for v in ridge_stats_cols.values()):
            ridge_stats = {}
            for k in _RIDGE_STATS_KEYS:
                col_arr = ridge_stats_cols[k]
                v = col_arr[i]
                if k in _RIDGE_STATS_INT_KEYS:
                    ridge_stats[k] = int(v)
                else:
                    ridge_stats[k] = float(v)
        results.append({
            "row": int(row[i]),
            "col": int(col[i]),
            "edge_mask": edge_mask[i],
            "n_lines": int(n_lines[i]),
            "edge_coverage": float(edge_coverage[i]),
            "orient_conc": float(orient_conc[i]),
            "gate_prob": None if np.isnan(gp) else gp,
            "texture_coverage": (None if texture_coverage is None
                                 or np.isnan(texture_coverage[i])
                                 else float(texture_coverage[i])),
            "ridge_coverage": (None if ridge_coverage is None or np.isnan(ridge_coverage[i])
                              else float(ridge_coverage[i])),
            "ridge_stats": ridge_stats,
            "binary_mask": None if binary_mask is None else binary_mask[i],
            "tile_size": tile_size,
            "soft_rect_t": soft_rect_t,
            "ridge_factors": ridge_factors,
            "orient_rect": orient_rect,
        })
    return results
