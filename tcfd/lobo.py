"""Bearing-level (leave-group-of-bearings-out) conformal calibration scores.

Why: split conformal is only valid if calibration and test points are exchangeable. In bearing
diagnosis the dominant shift is between PHYSICAL BEARINGS, so calibrating on windows of 1 bearing per
class (val bearings) does not cover unseen bearings. Here every real non-test bearing of the fold is
scored by a model that never saw it:

  N = real train bearings + val bearings of the fold (test bearings are never touched)
  split N into K class-stratified groups; for each group g:
      train the dense model on (N \\ g) + train-only bearings at the source condition, for the same number of
      epochs as the deployed model's best epoch (no early stopping -> g is never used for model selection);
      for every pruning ratio r: prune + fine-tune exactly like the deployed model (fixed epochs);
      save logits of g's windows at ALL operating conditions.
The pooled out-of-bearing scores (|N| bearings instead of 3) calibrate the deployed model's conformal
sets (variants lobo_src / lobo_pooled / lobo_mondrian in tcfd.evaluate). This is a cross-conformal /
jackknife+-style construction at the asset level; calibration scores come from FP32 PyTorch models.

Run:  python -m tcfd.lobo -c config.yaml [--folds A] [--groups 4]
Out:  <runs_dir>/<fold>/<model_id>/lobo/<Cx>_{logits,y,bearing}.npy
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from .config import (all_ratios, archs, base_parser, folds, get_device, load_config, model_id, path,
                     seed_everything)
from .data_pu import n_features
from .models import build_model
from .prune import prune_model
from .splits import all_bearings, fold_bearings, get_bearings
from .trainer import fit, predict_logits


def stratified_groups(bearings: list[str], labels: dict[str, str], k: int) -> list[list[str]]:
    groups = [[] for _ in range(k)]
    i = 0
    for cls in sorted(set(labels[b] for b in bearings)):
        for b in sorted(b for b in bearings if labels[b] == cls):
            groups[i % k].append(b)
            i += 1
    return [g for g in groups if g]


def best_epoch(info_path) -> int:
    info = json.load(open(info_path))
    if info.get("best_epoch"):
        return int(info["best_epoch"])
    h = info.get("history", [])
    with_val = [r for r in h if "val_nll" in r]
    return int(min(with_val, key=lambda r: r["val_nll"])["epoch"]) if with_val else max(1, len(h))


def main():
    p = base_parser("Bearing-level cross-conformal calibration scores")
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    dev = get_device(cfg)
    src, t, pr = cfg["source_condition"], cfg["train"], cfg["prune"]
    nC, F = len(cfg["classes"]), n_features(cfg)
    runs = path(cfg, t.get("runs_dir", "runs"))
    lab = all_bearings(cfg)
    for fold in folds(cfg, a.folds):
        fb = fold_bearings(cfg, fold)
        pool = fb["train_real"] + fb["val"]
        extra = [b for b in fb["train"] if b not in fb["train_real"]]
        groups = stratified_groups(pool, lab, a.groups)
        print(f"\n=== Fold {fold}: calibration pool {len(pool)} bearings in {len(groups)} groups: {groups}"
              f"  (+{len(extra)} train-only bearings)")
        for arch in archs(cfg, a.archs):
            mids = [model_id(arch, r) for r in all_ratios(cfg)]
            if not a.force and all((runs / fold / m / "lobo" / f"{src}_logits.npy").exists() for m in mids):
                print(f"[skip] {fold}/{arch} done")
                continue
            ep0 = best_epoch(runs / fold / model_id(arch, 0.0) / "info.json")
            ep_r = {r: best_epoch(runs / fold / model_id(arch, r) / "info.json") for r in pr["ratios"]}
            out = {m: {c: ([], [], []) for c in cfg["conditions"]} for m in mids}
            for gi, g in enumerate(groups):
                t0 = time.time()
                tr = [b for b in pool if b not in g] + extra
                Xtr, ytr, _, _ = get_bearings(cfg, tr, src, overlap=True)
                seed_everything(cfg["seed"] + 100 + gi)
                base = build_model(arch, nC)
                fit(base, Xtr, ytr, None, None, epochs=ep0, lr=t["lr"], wd=t["weight_decay"], patience=10 ** 9,
                    bs=t["batch_size"], device=dev, n_classes=nC, seed=cfg["seed"] + gi,
                    label_smoothing=t.get("label_smoothing", 0.0), tag=f"[lobo {fold}/{arch}/g{gi}]")
                models = {model_id(arch, 0.0): base}
                for r in pr["ratios"]:
                    m = prune_model(base, float(r), F)
                    fit(m, Xtr, ytr, None, None, epochs=ep_r[r], lr=pr["finetune_lr"], wd=t["weight_decay"],
                        patience=10 ** 9, bs=t["batch_size"], device=dev, n_classes=nC, seed=cfg["seed"] + gi,
                        label_smoothing=t.get("label_smoothing", 0.0), tag=f"[lobo {fold}/{arch}/g{gi}/r{r}]")
                    models[model_id(arch, r)] = m
                for c in cfg["conditions"]:
                    X, y, b, _ = get_bearings(cfg, g, c, overlap=False)
                    for mid, m in models.items():
                        out[mid][c][0].append(predict_logits(m.to(dev), X, dev))
                        out[mid][c][1].append(y)
                        out[mid][c][2].append(b)
                print(f"  {fold}/{arch} group {gi + 1}/{len(groups)} {g}: {len(models)} models in "
                      f"{time.time() - t0:.0f}s", flush=True)
            for mid in mids:
                d = runs / fold / mid / "lobo"
                d.mkdir(parents=True, exist_ok=True)
                for c, (L, Y, B) in out[mid].items():
                    np.save(d / f"{c}_logits.npy", np.concatenate(L))
                    np.save(d / f"{c}_y.npy", np.concatenate(Y))
                    np.save(d / f"{c}_bearing.npy", np.concatenate(B))
            print(f"[saved] {fold}/{arch}: out-of-bearing logits for {len(pool)} bearings")


if __name__ == "__main__":
    main()
