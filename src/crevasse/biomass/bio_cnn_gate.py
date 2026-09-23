"""A small 4-polarization CNN gate, scored against the feature RF on identical folds.

    conda run -n biomass python code/bio_cnn_gate.py
    conda run -n biomass python code/bio_cnn_gate.py --epochs 40 --seeds 5

Needs data/bio_chipcache_t512_c128.npz from code/bio_tile_cache.py. Writes
data/bio_cnn_oof_c<chip>.npz with the out-of-fold scores of every arm.

THE QUESTION THIS ANSWERS
-------------------------
The feature gate merges the four intensity channels by hand -- co = (HH+VV)/2,
cross = (HV+VH)/2, ratio = cross/co -- on the grounds that HV vs VH correlate at r 0.997-0.999
and HH vs VV at r 0.990-0.994. This feeds all four channels raw and lets the net choose the
combination, so a gain here is evidence the fixed 50/50 merge costs something.

Set expectations honestly: because the four channels are near rank 2, the raw input carries
almost the same information, and these are intensity-only products so there is no HH-VV phase
difference to recover. The plausible gain is from spatial structure the 38 tabular features
throw away, not from the polarimetry.

ARMS, ALL ON THE SAME BUFFERED FOLDS (82 km blocks / 41 km buffer)
-----------------------------------------------------------------
  A  sar          4-pol chips only
  B  sar+vel      chips plus log ITS_LIVE speed, concatenated after the pooling layer
  C  vel          speed alone, the control that prices B
  RF  the 38 merged features, the incumbent
  XY  map coordinates alone, the standing confound control

Reference numbers from code/bio_gate_controls.py at this setting: RF 0.889, (x,y) 0.601,
ratio_dB alone 0.639. Speed alone is 0.518 buffered despite 0.924 pooled, so the buffer
already removes the velocity shortcut -- B is not expected to be inflated by it.

bedmap_class is NOT an arm: it is 1 (grounded) for all 4244 labelled tiles, because the
feature builder admits a tile only at grounded_frac >= 0.95. Ice type has zero variance here
and cannot contribute; testing it needs the prescan relaxed, which is a separate decision.

ONLY 180 deg ROTATION IS AN ADMISSIBLE AUGMENTATION
--------------------------------------------------
The usual flip/90 deg-rotate set is WRONG on this sensor. The PSF is ~3:1 elongated with its
long axis near 50 deg (lag-1 acf 0.78 along 45 deg vs 0.13 along 135 deg), so a horizontal
flip or a 90 deg turn maps the along-axis direction onto the across-axis direction and teaches
the net that a real, fixed instrument asymmetry is arbitrary. A 180 deg rotation maps both axes
onto themselves, so it is the only one used.

NO EARLY STOPPING, AND THAT IS DELIBERATE
-----------------------------------------
There are 3 folds. Stopping on the test block would leak it, and there is no room to carve a
third split out of blocked spatial data, so the epoch count is fixed in advance and every arm
gets the same one. Run-to-run noise is reported instead, from --seeds independent inits, which
on 3 folds is the number that decides whether any difference is real.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from crevasse.biomass.bio_gate_controls import auc, buffered_folds, rf, within_fold_auc

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "biomass", "gate")
T = 512
PX_M = 5.0
BLOCK_TILES = 32
BUFFER_TILES = 16


class Net(nn.Module):
    """4 stride-2 conv blocks, global average pool, optional context scalars at the head.

    Global pooling rather than a flatten so the score cannot depend on WHERE in the tile the
    structure sits -- the label is a per-tile positive fraction, so position within the tile
    carries no supervision and a flatten would only give it room to memorise.
    """

    def __init__(self, in_ch=4, n_ctx=0, width=16):
        super().__init__()
        c = [in_ch, width, width * 2, width * 4, width * 4]
        self.body = nn.Sequential(*[
            layer for k in range(4) for layer in (
                nn.Conv2d(c[k], c[k + 1], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c[k + 1]),
                nn.ReLU(inplace=True),
            )])
        self.head = nn.Sequential(
            nn.Linear(c[-1] + n_ctx, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.n_ctx = n_ctx

    def forward(self, x, ctx=None):
        h = self.body(x).mean(dim=(2, 3))
        if self.n_ctx:
            h = torch.cat([h, ctx], dim=1)
        return self.head(h).squeeze(1)


def train_fold(chips, ctx, y, tr, te, dev, seed, epochs, lr, bs, return_net=False):
    """Fit on tr, return probabilities on te. Normalisation is fitted on tr only.

    With return_net, also returns everything needed to reproduce those probabilities later:
    the weights AND the four normalisation constants. Shipping weights without mu/sd would be
    shipping a model that silently scores a differently-scaled input -- see score_with_net.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_ctx = 0 if ctx is None else ctx.shape[1]

    # ONE shared mean/sd over all four channels, not per channel: a per-channel standardisation
    # would remove the cross-to-co offset, which is the signal on this sensor.
    # dtype=float64 is load-bearing: chips are float16 and numpy reduces in the input dtype,
    # which overflows on ~10^8 values and silently yields sd = nan, i.e. an all-zero input.
    sub = chips[tr]
    mu = float(np.nanmean(sub, dtype=np.float64))
    sd = float(np.nanstd(sub, dtype=np.float64)) or 1.0
    assert np.isfinite(mu) and np.isfinite(sd) and sd > 0, f"bad normalisation mu={mu} sd={sd}"

    def prep(a):
        x = (a.astype(np.float32) - mu) / sd
        return np.nan_to_num(x, nan=0.0)

    cm = cs = None
    if n_ctx:
        cm, cs = ctx[tr].mean(0), ctx[tr].std(0) + 1e-6

    def ctensor(sl):
        if not n_ctx:
            return None
        return torch.from_numpy(((ctx[sl] - cm) / cs).astype(np.float32)).to(dev)

    ytr = torch.from_numpy(y[tr].astype(np.float32)).to(dev)
    pos = float(y[tr].sum())
    pw = torch.tensor([(len(ytr) - pos) / max(pos, 1.0)], device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    net = Net(in_ch=chips.shape[1], n_ctx=n_ctx).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    itr = np.where(tr)[0]
    ctr = ctensor(tr)

    net.train()
    for _ in range(epochs):
        order = np.random.permutation(len(itr))
        for k in range(0, len(order), bs):
            sl = order[k:k + bs]
            xb = prep(chips[itr[sl]])
            if np.random.rand() < 0.5:                      # 180 deg only, see the docstring
                xb = xb[:, :, ::-1, ::-1].copy()
            xb = torch.from_numpy(xb).to(dev)
            cb = None if not n_ctx else ctr[sl]
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(xb, cb), ytr[sl])
            loss.backward()
            opt.step()
        sched.step()

    net.eval()
    ite = np.where(te)[0]
    cte = ctensor(te)
    out = np.empty(len(ite), np.float32)
    with torch.no_grad():
        for k in range(0, len(ite), 256):
            xb = torch.from_numpy(prep(chips[ite[k:k + 256]])).to(dev)
            cb = None if not n_ctx else cte[k:k + 256]
            out[k:k + 256] = torch.sigmoid(net(xb, cb)).cpu().numpy()
    if return_net:
        return out, {
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "mu": mu, "sd": sd,
            "ctx_mean": None if cm is None else np.asarray(cm, np.float64),
            "ctx_std": None if cs is None else np.asarray(cs, np.float64),
            "in_ch": int(chips.shape[1]), "n_ctx": int(n_ctx), "seed": int(seed),
        }
    return out


def score_with_net(w, chips, ctx, dev, bs=256):
    """Reproduce a saved net's probabilities on new chips, using ITS normalisation constants.

    The constants travel with the weights on purpose. Re-deriving mu/sd from whatever array is
    being scored would make the score depend on the composition of the batch, so a tile would
    get a different answer depending on what it was scored alongside.
    """
    net = Net(in_ch=w["in_ch"], n_ctx=w["n_ctx"]).to(dev)
    net.load_state_dict(w["state_dict"])
    net.eval()
    if w["n_ctx"] and ctx is None:
        raise SystemExit("this checkpoint expects context (arm sar+vel) but none was given")
    out = np.empty(len(chips), np.float32)
    with torch.no_grad():
        for k in range(0, len(chips), bs):
            x = (chips[k:k + bs].astype(np.float32) - w["mu"]) / w["sd"]
            xb = torch.from_numpy(np.nan_to_num(x, nan=0.0)).to(dev)
            cb = None
            if w["n_ctx"]:
                c = (ctx[k:k + bs] - w["ctx_mean"]) / w["ctx_std"]
                cb = torch.from_numpy(c.astype(np.float32)).to(dev)
            out[k:k + bs] = torch.sigmoid(net(xb, cb)).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(
        ROOT, "data", f"bio_chipcache_t{T}_c128.npz"))
    ap.add_argument("--features", default=os.path.join(
        ROOT, "data", f"bio_tile_features_t{T}.npz"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    z = np.load(args.cache, allow_pickle=True)
    chips, y, xm, ym = z["chips"], z["y"], z["x_m"], z["y_m"]
    vel = np.log10(np.maximum(z["vel"], 1e-3)).astype(np.float32)[:, None]
    block_m, buffer_m = BLOCK_TILES * T * PX_M, BUFFER_TILES * T * PX_M
    folds = list(buffered_folds(y, xm, ym, block_m, buffer_m))
    print(f"{len(y)} tiles, {int(y.sum())} positive; {len(folds)} folds at "
          f"{block_m/1000:.0f} km blocks / {buffer_m/1000:.0f} km buffer", flush=True)
    print(f"device {dev}, {args.epochs} epochs, {args.seeds} seeds, "
          f"chips {tuple(chips.shape)}\n", flush=True)
    for k, (te, tr) in enumerate(folds):
        print(f"  fold {k}: train {int(tr.sum())} ({int(y[tr].sum())} pos)  "
              f"test {int(te.sum())} ({int(y[te].sum())} pos)", flush=True)

    arms = {"A sar": (chips, None), "B sar+vel": (chips, vel)}
    oof = {}
    for nm, (C, X2) in arms.items():
        per_seed = []
        for s in range(args.seeds):
            o = np.full(len(y), np.nan)
            for te, tr in folds:
                o[te] = train_fold(C, X2, y, tr, te, dev, s, args.epochs, args.lr, args.bs)
            m = np.isfinite(o)
            per_seed.append(o)
            print(f"  {nm} seed {s}: pooled AUC {auc(y[m], o[m]):.3f}", flush=True)
        sc = np.isfinite(per_seed[0])
        ens = np.full(len(y), np.nan)
        ens[sc] = np.mean([o[sc] for o in per_seed], axis=0)
        oof[nm] = ens
        a = [auc(y[sc], o[sc]) for o in per_seed]
        w = [within_fold_auc(y, o, folds) for o in per_seed]
        print(f"{nm}: pooled {np.mean(a):.3f} +- {np.std(a):.3f}, "
              f"within-fold {np.mean(w):.3f} +- {np.std(w):.3f} over {args.seeds} seeds\n",
              flush=True)

    # the incumbent and the two confound controls, on exactly these folds
    d = np.load(args.features, allow_pickle=True)
    XF = d["X"][z["idx"]]
    XY = np.c_[xm, ym].astype(np.float32)
    for nm, F, seed in [("RF merged", XF, 0), ("C vel", vel, 1), ("XY coords", XY, 1)]:
        o = np.full(len(y), np.nan)
        for te, tr in folds:
            o[te] = rf(seed).fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
        oof[nm] = o
        m = np.isfinite(o)
        print(f"{nm}: pooled {auc(y[m], o[m]):.3f}, "
              f"within-fold {within_fold_auc(y, o, folds):.3f}", flush=True)

    hdr = "  ".join(f"fold{k} (n1={int(y[te].sum())}/{int(te.sum())})"
                    for k, (te, _) in enumerate(folds))
    print(f"\n{'arm':>12}  pooled  within  {hdr}")
    for nm, o in oof.items():
        m = np.isfinite(o)
        pf = [f"{auc(y[te], o[te]):18.3f}" for te, _ in folds]
        print(f"{nm:>12}  {auc(y[m], o[m]):6.3f}  {within_fold_auc(y, o, folds):6.3f}  "
              + "  ".join(pf), flush=True)

    out = os.path.join(ROOT, "data", f"bio_cnn_oof_c{int(z['chip'])}.npz")
    np.savez(out, y=y, idx=z["idx"], x_m=xm, y_m=ym,
             **{f"oof_{k.replace(' ', '_')}": v for k, v in oof.items()})
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
