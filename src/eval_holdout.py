"""Held-out validation for the crevasse gate against confirmation bias.

The relabeling pass was guided by the model's own predictions, so cross-validated
AUC on that set can be optimistic. This script instead trains on a frozen
snapshot (the model-reviewed tiles) and tests ONLY on tiles labeled afterward
with the plain labeling GUI -- tiles the model never surfaced. If RF still scores
near its CV AUC here, the gate is genuinely good, not just self-consistent.

    conda run -n nisar-gate python src/eval_holdout.py

Train set  = rows in --snapshot-csv
Test set   = rows in --labels-csv NOT present in the snapshot (the fresh batch)
"""
import argparse, csv
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
import gate_common as E


def load_keyed(csv_path):
    """Return dict {(granule,row,col): (label, feats)} for valid tiles."""
    rows = [r for r in csv.DictReader(open(csv_path))
            if r["label"] in ("crevasse", "none")]
    rasters = E.granule_rasters()
    out = {}
    for r in rows:
        g = r["granule"]
        tile = rasters[g].read(1, window=Window(int(r["col"]), int(r["row"]), E.TS, E.TS),
                               boundless=True, fill_value=0)
        img = E.preprocess(tile)
        if img is None:
            continue
        out[(g, int(r["row"]), int(r["col"]))] = (
            1 if r["label"] == "crevasse" else 0, E.glcm_st_features(img))
    for s in rasters.values():
        s.close()
    return out


def to_matrix(items, fn):
    X = np.array([[f[k] for k in fn] for (_, f) in items])
    y = np.array([lab for (lab, _) in items])
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", default=str(E.ROOT / "data" / "tile_labels.csv"))
    ap.add_argument("--snapshot-csv", default=str(E.ROOT / "data" / "tile_labels.review_snapshot.csv"))
    args = ap.parse_args()

    full = load_keyed(args.labels_csv)
    snap = load_keyed(args.snapshot_csv)
    train_keys = [k for k in full if k in snap]
    test_keys = [k for k in full if k not in snap]

    if not test_keys:
        raise SystemExit("No fresh tiles found. Label a new batch with "
                         "label_tiles_gui.py first (they must not be in the snapshot).")

    fn = list(next(iter(full.values()))[1])
    Xtr, ytr = to_matrix([full[k] for k in train_keys], fn)
    Xte, yte = to_matrix([full[k] for k in test_keys], fn)

    print(f"train (snapshot): {len(ytr)} tiles  {ytr.sum()} crev / {(ytr==0).sum()} none")
    print(f"test  (fresh):    {len(yte)} tiles  {yte.sum()} crev / {(yte==0).sum()} none")
    print("-" * 78)

    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                class_weight="balanced", random_state=E.SEED, n_jobs=-1)
    rf.fit(Xtr, ytr)
    proba = rf.predict_proba(Xte)[:, 1]
    E.report("RF held-out (fresh tiles)", yte, proba)


if __name__ == "__main__":
    main()
