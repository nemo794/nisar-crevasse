"""One-time migration: an existing pickle `.pt`/`.pth` torch checkpoint (a U-Net training
checkpoint or a CNN gate checkpoint -- anything `ckpt_io.save_checkpoint` can serialize) ->
`<stem>.safetensors` + `<stem>.json` (see `ckpt_io.py`). Non-destructive -- the old file is
left in place.

This is the one script in this repo still allowed to call `torch.load(...,
weights_only=False)`: converting a checkpoint YOU already trained is a deliberate,
one-time, scoped use of pickle on a trusted, self-produced file, not a permanent
dependency in the training/inference pipeline (which no longer touches pickle at all
after this conversion -- an RF-only gate path never did, being joblib/sklearn only).

    python src/convert_pth_to_safetensors.py models/RUN/unet_best.pth [more.pth ...]
    python src/convert_pth_to_safetensors.py data/bio_cnn_gate_sarvel.pt [more.pt ...]
"""
import argparse
from pathlib import Path

import torch

from crevasse.common.ckpt_io import save_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ckpt_paths", nargs="+", help="one or more existing .pt/.pth checkpoints")
    args = ap.parse_args()

    for p in args.ckpt_paths:
        p = Path(p)
        stem = p.with_suffix("")
        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        save_checkpoint(stem, ckpt)
        print(f"{p} -> {stem}.safetensors + {stem}.json")


if __name__ == "__main__":
    main()
