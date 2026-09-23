"""Figures from a bio_eval_shard.py .npz. No GPU, no model, no shard needed.

Same split as nisar-crevasse-unet/src/plot_eval.py: this runs after the GPU eval job, on
whatever node is free, and can be re-run without paying for inference again.

  <stem>_panels.png    per-tile strips: HH (dB, display-stretched) | target | predicted
                       probability | overlay, for worst/median/best tiles by IoU.
  <stem>_summary.png   per-tile IoU distribution, IoU vs target coverage, stratification
                       bars, prediction-threshold sweep if --sweep was used.
  <stem>_curves.png    training curves from tensorboard, if --tb-logdir is given.

Only the HH channel is shown in the panels regardless of how many channels the model
trained on -- a 4-panel-per-channel strip would be unreadable, and HH is present in every
configuration this repo expects to train (see bio_train_unet_supervised.py --channels).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _display_hh(stack):
    """(C, H, W) raw dB -> HH channel, percentile-stretched to [0, 1] for display only.
    Not the same transform as bio_sar_dataset.normalize_stack, which is model-input
    oriented (zero-mean/unit-sd) rather than display oriented."""
    hh = np.asarray(stack[0], dtype=np.float32)
    valid = hh[np.isfinite(hh)]
    if valid.size == 0:
        return np.zeros_like(hh)
    p2, p98 = np.percentile(valid, [2, 98])
    out = np.clip(hh, p2, p98)
    out = (out - p2) / max(p98 - p2, 1e-6)
    return np.nan_to_num(out, nan=0.0)


def _panels(d, out_path, pred_thresh, target_thresh):
    if "sample_image" not in d:
        print("No stored samples in the npz (bio_eval_shard.py --save-preds 0?), "
              "skipping panels")
        return

    imgs, tgts, probs = d["sample_image"], d["sample_target"], d["sample_prob"]
    ious, poss = d["sample_iou"], d["sample_position"]

    n = len(imgs)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.15 * n), squeeze=False)

    for i in range(n):
        hh = _display_hh(np.asarray(imgs[i], dtype=np.float32))
        tgt = np.asarray(tgts[i], dtype=np.float32)
        prob = np.asarray(probs[i], dtype=np.float32)

        axes[i][0].imshow(hh, cmap="gray", vmin=0, vmax=1)
        axes[i][0].set_ylabel(f"({int(poss[i][0])}, {int(poss[i][1])})\n"
                              f"IoU {ious[i]:.3f}", fontsize=8)
        axes[i][1].imshow(tgt, cmap="magma", vmin=0, vmax=1)
        axes[i][2].imshow(prob, cmap="magma", vmin=0, vmax=1)

        axes[i][3].imshow(hh, cmap="gray", vmin=0, vmax=1)
        t_bin = tgt > target_thresh
        p_bin = prob > pred_thresh
        rgba = np.zeros((*t_bin.shape, 4))
        rgba[t_bin & ~p_bin] = [0.0, 0.6, 1.0, 0.55]
        rgba[p_bin & ~t_bin] = [1.0, 0.2, 0.1, 0.55]
        rgba[p_bin & t_bin] = [0.2, 1.0, 0.2, 0.65]
        axes[i][3].imshow(rgba)

        for j in range(4):
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

    for j, t in enumerate(["HH (dB, display-stretched)", "target",
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

    axes[k].scatter(coverage, tile_iou, s=6, alpha=0.35, color="darkslategray")
    axes[k].set_xlabel("target coverage")
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
        axes[k].bar(range(len(rows)), [r["pooled_iou"] for r in rows], color="tab:orange")
        axes[k].set_xticks(range(len(rows)))
        axes[k].set_xticklabels(labels, fontsize=7)
        axes[k].axhline(float(d["pooled_iou"]), color="crimson", ls="--", lw=1,
                        label="overall pooled")
        for i, r in enumerate(rows):
            axes[k].text(i, r["pooled_iou"], f"n={r['n']}", ha="center", va="bottom",
                        fontsize=7)
        axes[k].set_ylabel("pooled IoU")
        axes[k].set_title(f"by {name} (terciles)", fontsize=9)
        axes[k].legend(fontsize=7)
        k += 1

    if sweep:
        th = [r["thresh"] for r in sweep]
        axes[k].plot(th, [r["pooled_iou"] for r in sweep], "o-", label="pooled IoU")
        axes[k].plot(th, [r["precision"] for r in sweep], "s--", ms=3, label="precision")
        axes[k].plot(th, [r["recall"] for r in sweep], "^--", ms=3, label="recall")
        axes[k].axvline(float(d["pred_thresh"]), color="k", ls=":",
                        label=f"reported {float(d['pred_thresh'])}")
        axes[k].set_xlabel("prediction threshold")
        axes[k].legend(fontsize=7)
        axes[k].set_title(f"target fixed at >{float(d['target_thresh'])}", fontsize=9)
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
    axes[1].set_ylabel("val IoU")

    for tag in ["train/bce", "train/dice", "train/consistency"]:
        if tag in tags:
            s, v = series(tag)
            axes[2].plot(s, v, label=tag)
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("loss term")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description="Figures from a bio_eval_shard.py npz")
    p.add_argument("--eval-npz", required=True)
    p.add_argument("--tb-logdir", default=None)
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args()

    d = np.load(args.eval_npz, allow_pickle=False)
    stem = Path(args.out_prefix) if args.out_prefix else Path(args.eval_npz).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)

    _panels(d, f"{stem}_panels.png", float(d["pred_thresh"]), float(d["target_thresh"]))
    _summary(d, f"{stem}_summary.png")
    if args.tb_logdir:
        _curves(args.tb_logdir, f"{stem}_curves.png")


if __name__ == "__main__":
    main()
