"""
Figures from an eval_shard.py .npz. No GPU, no model, no shard needed.

Split from eval_shard.py deliberately: the eval runs inside the GPU job, this runs
afterwards on whatever node is free, and it can be re-run with different panel choices
without paying for inference again.

Two figures:

  <stem>_panels.png    per-tile strips: amplitude | target | predicted probability |
                       overlay, for the worst / median / best tiles by IoU. Every panel
                       carries its own IoU and (row, col), because a panel without its
                       score invites reading the prettiest tile as typical.
  <stem>_summary.png   per-tile IoU distribution, IoU against target coverage, the
                       stratification bars, and the prediction-threshold sweep if
                       --sweep was used.

Optionally also plots the training curves straight from the run's tensorboard events:

  <stem>_curves.png    train/val loss and val IoU per epoch.

Usage:
  python src/plot_eval.py --eval-npz models/.../eval_025_091.npz \
      --tb-logdir models/.../logs
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crevasse.nisar.sar_dataset import preprocess_sar_tile


def _panels(d, out_path, target_thresh, pred_thresh):
    if "sample_image" not in d:
        print("No stored samples in the npz (eval_shard.py --save-preds 0?), "
              "skipping panels")
        return

    imgs = d["sample_image"]
    tgts = d["sample_target"]
    probs = d["sample_prob"]
    ious = d["sample_iou"]
    poss = d["sample_position"]

    n = len(imgs)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.15 * n), squeeze=False)

    for i in range(n):
        # Same preprocessing the model saw, so the panel is not a different image from
        # the one that produced the prediction beside it.
        amp = preprocess_sar_tile(np.asarray(imgs[i], dtype=np.float32))
        tgt = np.asarray(tgts[i], dtype=np.float32)
        prob = np.asarray(probs[i], dtype=np.float32)

        axes[i][0].imshow(amp, cmap="gray", vmin=0, vmax=1)
        axes[i][0].set_ylabel(f"({int(poss[i][0])}, {int(poss[i][1])})\n"
                              f"IoU {ious[i]:.3f}", fontsize=8)
        axes[i][1].imshow(tgt, cmap="magma", vmin=0, vmax=1)
        axes[i][2].imshow(prob, cmap="magma", vmin=0, vmax=1)

        axes[i][3].imshow(amp, cmap="gray", vmin=0, vmax=1)
        # Binary overlay at the SAME thresholds the reported IoU used, otherwise the
        # picture and the number disagree.
        t_bin = tgt > target_thresh
        p_bin = prob > pred_thresh
        rgba = np.zeros((*t_bin.shape, 4))
        rgba[t_bin & ~p_bin] = [0.0, 0.6, 1.0, 0.55]   # missed
        rgba[p_bin & ~t_bin] = [1.0, 0.2, 0.1, 0.55]   # false positive
        rgba[p_bin & t_bin] = [0.2, 1.0, 0.2, 0.65]    # hit
        axes[i][3].imshow(rgba)

        for j in range(4):
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

    for j, t in enumerate(["amplitude (as model sees it)",
                           "soft target",
                           "predicted probability",
                           f"overlay: green hit, blue miss,\nred false pos "
                           f"(t>{target_thresh}, p>{pred_thresh})"]):
        axes[0][j].set_title(t, fontsize=9)

    fig.suptitle(f"{Path(str(d['shard'])).name}  subset={d['subset']}  "
                 f"pooled IoU {float(d['pooled_iou']):.4f}  "
                 f"(rows: worst / median / best by per-tile IoU)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _summary(d, out_path):
    tile_iou = d["tile_iou"]
    coverage = d["coverage"]
    strata = json.loads(str(d["strata_json"])) if "strata_json" in d else {}
    sweep = json.loads(str(d["sweep_json"])) if "sweep_json" in d else []

    n_strata = sum(1 for v in strata.values() if v)
    ncols = 2 + n_strata + (1 if sweep else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 3.6))
    axes = np.atleast_1d(axes)
    k = 0

    axes[k].hist(tile_iou, bins=40, color="steelblue")
    axes[k].axvline(float(d["pooled_iou"]), color="crimson", lw=2,
                    label=f"pooled {float(d['pooled_iou']):.3f}")
    axes[k].axvline(tile_iou.mean(), color="k", ls="--", lw=1.5,
                    label=f"per-tile mean {tile_iou.mean():.3f}")
    axes[k].set_xlabel("per-tile IoU")
    axes[k].set_ylabel("tiles")
    axes[k].legend(fontsize=7)
    axes[k].set_title(f"n={len(tile_iou)}", fontsize=9)
    k += 1

    # The control for reading any IoU: an empty tile cannot score well, so check whether
    # low IoU tracks low target coverage before calling it a model failure.
    axes[k].scatter(coverage, tile_iou, s=6, alpha=0.35, color="darkslategray")
    axes[k].set_xlabel("target coverage (fraction > 0.2)")
    axes[k].set_ylabel("per-tile IoU")
    if len(coverage) > 2 and np.std(coverage) > 0:
        r = np.corrcoef(coverage, tile_iou)[0, 1]
        axes[k].set_title(f"IoU vs how much there was to find\nPearson r = {r:.3f}",
                          fontsize=9)
    k += 1

    for name, rows in strata.items():
        if not rows:
            continue
        labels = [f"{r['lo']:.2f}-\n{r['hi']:.2f}" for r in rows]
        axes[k].bar(range(len(rows)), [r["pooled_iou"] for r in rows],
                    color="tab:orange")
        axes[k].set_xticks(range(len(rows)))
        axes[k].set_xticklabels(labels, fontsize=7)
        axes[k].axhline(float(d["pooled_iou"]), color="crimson", ls="--", lw=1,
                        label="overall pooled")
        for i, r in enumerate(rows):
            axes[k].text(i, r["pooled_iou"], f"n={r['n']}", ha="center",
                         va="bottom", fontsize=7)
        axes[k].set_ylabel("pooled IoU")
        axes[k].set_title(f"by {name} (terciles)", fontsize=9)
        axes[k].legend(fontsize=7)
        k += 1

    if sweep:
        th = [r["thresh"] for r in sweep]
        axes[k].plot(th, [r["pooled_iou"] for r in sweep], "o-", label="pooled IoU")
        axes[k].plot(th, [r["precision"] for r in sweep], "s--", ms=3,
                     label="precision")
        axes[k].plot(th, [r["recall"] for r in sweep], "^--", ms=3, label="recall")
        axes[k].axvline(float(d["pred_thresh"]), color="k", ls=":",
                        label=f"reported {float(d['pred_thresh'])}")
        axes[k].set_xlabel("prediction threshold")
        axes[k].legend(fontsize=7)
        axes[k].set_title("target threshold held fixed", fontsize=9)
        k += 1

    fig.suptitle(f"{Path(str(d['checkpoint'])).parent.name} on "
                 f"{Path(str(d['shard'])).name}  subset={d['subset']}  |  "
                 f"P {float(d['precision']):.3f}  R {float(d['recall']):.3f}  "
                 f"target coverage {float(d['target_coverage']) * 100:.2f}%  "
                 f"predicted {float(d['predicted_coverage']) * 100:.2f}%", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def _curves(tb_logdir, out_path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not importable, skipping training curves")
        return

    acc = EventAccumulator(str(tb_logdir))
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))
    if not tags:
        print(f"No scalars in {tb_logdir}, skipping training curves")
        return

    def series(tag):
        ev = acc.Scalars(tag)
        return [e.step for e in ev], [e.value for e in ev]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for tag, style in [("train/loss", "-"), ("val/loss", "--")]:
        if tag in tags:
            s, v = series(tag)
            axes[0].plot(s, v, style, label=tag)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=8)

    if "val/iou" in tags:
        s, v = series("val/iou")
        axes[1].plot(s, v, "-o", ms=3, color="tab:green")
        axes[1].axhline(max(v), color="k", ls=":", lw=1,
                        label=f"best {max(v):.4f} @ epoch {s[int(np.argmax(v))]}")
        axes[1].legend(fontsize=8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val IoU (both sides thresholded)")

    for tag in ["train/bce", "train/dice", "train/consistency"]:
        if tag in tags:
            s, v = series(tag)
            axes[2].plot(s, v, label=tag)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("loss term")
    axes[2].legend(fontsize=8)
    axes[2].set_title("consistency is flip-equivariance only", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Figures from an eval_shard.py npz")
    p.add_argument("--eval-npz", required=True)
    p.add_argument("--tb-logdir", default=None,
                   help="Training run's tensorboard log dir, for the loss/IoU curves.")
    p.add_argument("--out-prefix", default=None,
                   help="Defaults to the eval npz path without its extension.")
    args = p.parse_args()

    d = np.load(args.eval_npz, allow_pickle=False)
    stem = Path(args.out_prefix) if args.out_prefix \
        else Path(args.eval_npz).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)

    _panels(d, f"{stem}_panels.png", float(d["target_thresh"]),
            float(d["pred_thresh"]))
    _summary(d, f"{stem}_summary.png")
    if args.tb_logdir:
        _curves(args.tb_logdir, f"{stem}_curves.png")


if __name__ == "__main__":
    main()
