"""Shared constants, granule lookup, and the hand GLCM/structure-tensor features.

Every other module in src/ keys off this one: the tile size, the random seed, the
granule-id regex, and the 14 hand features are defined here and nowhere else. The
`--labels-csv` entry point at the bottom is a 5-fold RF baseline, useful as a sanity
check on a fresh label set:

    conda run -n nisar-gate python src/gate_common.py --labels-csv data/my_labels.csv

An ImageNet-pretrained ResNet18 arm was also evaluated during development (it beat the
14 hand features but lost to the 270-feature RF in train_gate_classifier.py). It is not
carried here, because it was the only reason this module imported PyTorch and dropping
it makes the whole inference path torch-free. See the research repo for that code.

Data is resolved relative to `ROOT` (defaults to `<repo>/data/nisar/gate`, this sensor's
own data root within the merged `crevasse` repo -- mirrors the pre-merge
`nisar-crevasse-gate/data` layout one level deeper). Set `GATE_ROOT` to point at a
directory tree holding its own `data/` elsewhere -- the 48 GB of granules usually will
not live in the repo. Raw granule GeoTIFFs live in the sibling `data/nisar/granules/`
directory, not under `ROOT` -- see `find_data_swath`/callers for that path.
"""
import argparse, csv, os, re
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.feature import graycomatrix, graycoprops, structure_tensor, structure_tensor_eigenvalues
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve

ROOT = Path(os.environ.get(
    "GATE_ROOT", Path(__file__).resolve().parents[3] / "data" / "nisar" / "gate"))
GRANULES = Path(__file__).resolve().parents[3] / "data" / "nisar" / "granules"
GRANULE_RE = re.compile(r"GSLC_(\d{3}_\d{3})_")
TS, LEVELS = 512, 32
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
DISTS = [1, 3]
SEED = 42


def granule_rasters():
    """Open every granule under data/nisar/, keyed by GRANULE_RE, refusing ambiguity.

    GRANULE_RE captures only the 3-digit pair, so two acquisitions of the same pair on
    different tracks collide. A plain dict comprehension resolves that silently -- and
    inconsistently across call sites, which meant training and scoring could read
    different ground for the same label row. Fail instead.
    """
    out, src = {}, {}
    for p in sorted(GRANULES.glob("*.tif")):
        g = GRANULE_RE.search(p.name).group(1)
        if g in out:
            raise SystemExit(
                f"granule id {g} is ambiguous -- two files match it:\n  {src[g].name}\n"
                f"  {p.name}\nMove one out of data/nisar/ (see lsar_defective/README.md) "
                f"or widen GRANULE_RE; do not leave it to glob order.")
        out[g], src[g] = rasterio.open(p), p
    return out


def find_granule(granule_id):
    """Path to the one raster matching granule_id, or exit."""
    for p in sorted(GRANULES.glob("*.tif")):
        m = GRANULE_RE.search(p.name)
        if m and m.group(1) == granule_id:
            return p
    raise SystemExit(f"granule {granule_id} not found under {GRANULES}/")


def preprocess(tile):
    t = tile.astype(np.float32)
    m = t > 0
    if m.mean() < 0.5:
        return None
    t[~m] = np.nan
    t = np.log10(t)
    t = np.nan_to_num(t, nan=np.nanmedian(t))
    p2, p98 = np.percentile(t, [2, 98])
    if p98 <= p2:
        return None
    return np.clip((t - p2) / (p98 - p2), 0, 1).astype(np.float32)


def glcm_st_features(img):
    feats = {"cv": float(img.std() / (img.mean() + 1e-9))}
    q = (img * (LEVELS - 1)).astype(np.uint8)
    glcm = graycomatrix(q, distances=DISTS, angles=ANGLES, levels=LEVELS,
                        symmetric=True, normed=True)
    for prop in ("contrast", "correlation", "homogeneity", "energy", "dissimilarity"):
        vals = graycoprops(glcm, prop)
        feats[f"glcm_{prop}"] = float(vals.mean())
        feats[f"glcm_{prop}_aniso"] = float(vals.std(axis=1).mean())
    Axx, Axy, Ayy = structure_tensor(img.astype(np.float32), sigma=2.0, mode="reflect", order="rc")
    l1, l2 = structure_tensor_eigenvalues((Axx, Axy, Ayy))
    coh = (l1 - l2) / (l1 + l2 + 1e-9)
    feats["coherence_mean"] = float(coh.mean())
    feats["coherence_p90"] = float(np.percentile(coh, 90))
    feats["coherence_frac_hi"] = float((coh > 0.5).mean())
    return feats


def load_data(csv_path):
    rasters = granule_rasters()
    rows = [r for r in csv.DictReader(open(csv_path)) if r["label"] in ("crevasse", "none")]
    imgs, feats, y = [], [], []
    for r in rows:
        tile = rasters[r["granule"]].read(1, window=Window(int(r["col"]), int(r["row"]), TS, TS),
                                          boundless=True, fill_value=0)
        img = preprocess(tile)
        if img is None:
            continue
        imgs.append(img)
        feats.append(glcm_st_features(img))
        y.append(1 if r["label"] == "crevasse" else 0)
    for s in rasters.values():
        s.close()
    return np.stack(imgs), feats, np.array(y, dtype=np.int64)


def report(name, y, proba):
    auc = roc_auc_score(y, proba)
    acc = accuracy_score(y, (proba >= 0.5).astype(int))
    prec, rec, thr = precision_recall_curve(y, proba)
    line = f"{name:30s} AUC={auc:.3f}  acc={acc:.3f}"
    for target in (0.90, 0.95):
        idx = np.where(rec[:-1] >= target)[0]
        if len(idx):
            line += f"  skip@r{target:.2f}={(proba < thr[idx[-1]]).mean():.1%}"
    print(line)
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", default=str(ROOT / "data" / "tile_labels.csv"))
    args = ap.parse_args()

    np.random.seed(SEED)
    X512, feats, y = load_data(args.labels_csv)
    print(f"{len(y)} tiles: {y.sum()} crevasse / {(y==0).sum()} none")
    print("-" * 78)

    fn = list(feats[0])
    Xf = np.array([[f[k] for k in fn] for f in feats])
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rf_proba = cross_val_predict(rf, Xf, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    report("RF (14 hand features)", y, rf_proba)


if __name__ == "__main__":
    main()
