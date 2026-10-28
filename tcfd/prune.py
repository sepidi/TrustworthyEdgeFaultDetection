"""Structured channel pruning (L1 magnitude, uniform per layer, classifier kept) + fine-tuning.

Uses torch-pruning (DepGraph, Fang et al. CVPR 2023), which handles residual adds.
Run:  python -m tcfd.prune -c config.yaml [--folds A] [--archs resnet1d]
Out:  runs/<fold>/<arch>_pXX/{model.pt, info.json}
"""
from __future__ import annotations

import copy
import json
import time

import torch
import torch_pruning as tp

from .config import (archs, base_parser, folds, get_device, load_config, model_id, path, save_json,
                     seed_everything)
from .data_pu import n_features
from .models import count_params
from .splits import get_split
from .trainer import fit, nll_acc, predict_logits


def prune_model(model: torch.nn.Module, ratio: float, n_feat: int) -> torch.nn.Module:
    model = copy.deepcopy(model).cpu().eval()
    example = torch.randn(1, 1, n_feat)
    imp = tp.importance.MagnitudeImportance(p=1)
    kw = dict(importance=imp, ignored_layers=[model.fc])
    try:
        pruner = tp.pruner.MetaPruner(model, example, pruning_ratio=ratio, **kw)
    except TypeError:  # torch-pruning < 1.3
        pruner = tp.pruner.MetaPruner(model, example, ch_sparsity=ratio, **kw)
    pruner.step()
    return model


def macs_params(model, n_feat):
    macs, params = tp.utils.count_ops_and_params(model.cpu().eval(), torch.randn(1, 1, n_feat))
    return int(macs), int(params)


def main():
    p = base_parser("Prune + fine-tune")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    dev = get_device(cfg)
    src = cfg["source_condition"]
    pr, t = cfg["prune"], cfg["train"]
    nC, F = len(cfg["classes"]), n_features(cfg)
    for fold in folds(cfg, a.folds):
        Xtr, ytr, _, _ = get_split(cfg, fold, "train", src)
        Xva, yva, _, _ = get_split(cfg, fold, "valfit", src)
        for arch in archs(cfg, a.archs):
            base_dir = path(cfg, cfg["train"].get("runs_dir", "runs")) / fold / model_id(arch, 0.0)
            base = torch.load(base_dir / "model.pt", weights_only=False)
            bm, bp = macs_params(base, F)
            info0 = path(cfg, cfg["train"].get("runs_dir", "runs")) / fold / model_id(arch, 0.0) / "info.json"
            d0 = json.load(open(info0))
            d0.update({"macs": bm, "params": bp})
            save_json(d0, info0)
            for r in pr["ratios"]:
                out = path(cfg, cfg["train"].get("runs_dir", "runs")) / fold / model_id(arch, r)
                if (out / "model.pt").exists() and not a.force:
                    print(f"[skip] {out} exists")
                    continue
                seed_everything(cfg["seed"])
                m = prune_model(base, float(r), F)
                macs, params = macs_params(m, F)
                pre_nll, pre_acc = nll_acc(predict_logits(m.to(dev), Xva, dev), yva)
                t0 = time.time()
                res = fit(m, Xtr, ytr, Xva, yva, epochs=pr["finetune_epochs"], lr=pr["finetune_lr"],
                          wd=t["weight_decay"], patience=t["patience"], bs=t["batch_size"], device=dev,
                          n_classes=nC, seed=cfg["seed"], label_smoothing=t.get("label_smoothing", 0.0), min_epochs=pr.get("min_epochs", 1), tag=f"[{fold}/{arch}/r{r}]")
                vnll, vacc = nll_acc(predict_logits(m, Xva, dev), yva)
                out.mkdir(parents=True, exist_ok=True)
                torch.save(m.cpu(), out / "model.pt")
                save_json({"fold": fold, "arch": arch, "ratio": float(r), "params": params, "macs": macs,
                           "params_rel": params / bp, "macs_rel": macs / bm,
                           "acc_before_ft": pre_acc, "nll_before_ft": pre_nll,
                           "val_nll": vnll, "val_acc": vacc, "ft_sec": round(time.time() - t0, 1), **res},
                          out / "info.json")
                print(f"[saved] {out} params={params} ({params / bp:.1%}) MACs={macs / bm:.1%} "
                      f"valfit acc {pre_acc:.3f}->{vacc:.3f}")


if __name__ == "__main__":
    main()
