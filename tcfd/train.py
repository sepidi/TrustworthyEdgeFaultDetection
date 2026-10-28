"""Train dense base models (ratio 0) on the source condition for every fold x arch.

Run:  python -m tcfd.train -c config.yaml [--folds A B] [--archs wdcnn]
Out:  runs/<fold>/<arch>_p00/{model.pt, info.json}
"""
from __future__ import annotations

import time

import torch

from .config import (archs, base_parser, folds, get_device, load_config, model_id, path, save_json,
                     seed_everything)
from .models import build_model, count_params
from .splits import fold_bearings, get_split
from .trainer import fit, nll_acc, predict_logits


def main():
    p = base_parser("Train dense base models")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    dev = get_device(cfg)
    src = cfg["source_condition"]
    t = cfg["train"]
    nC = len(cfg["classes"])
    print(f"device={dev}")
    for fold in folds(cfg, a.folds):
        fb = fold_bearings(cfg, fold)
        print(f"\n=== Fold {fold}: train={fb['train']} val={fb['val']} test={fb['test']}")
        Xtr, ytr, _, _ = get_split(cfg, fold, "train", src)
        Xva, yva, _, _ = get_split(cfg, fold, "valfit", src)
        print(f"train windows={len(Xtr)}  valfit windows={len(Xva)}")
        for arch in archs(cfg, a.archs):
            out = path(cfg, cfg["train"].get("runs_dir", "runs")) / fold / model_id(arch, 0.0)
            if (out / "model.pt").exists() and not a.force:
                print(f"[skip] {out} exists (use --force)")
                continue
            seed_everything(cfg["seed"])
            model = build_model(arch, nC)
            t0 = time.time()
            res = fit(model, Xtr, ytr, Xva, yva, epochs=t["epochs"], lr=t["lr"], wd=t["weight_decay"],
                      patience=t["patience"], bs=t["batch_size"], device=dev, n_classes=nC,
                      seed=cfg["seed"], label_smoothing=t.get("label_smoothing", 0.0), min_epochs=t.get("min_epochs", 1), tag=f"[{fold}/{arch}]")
            vnll, vacc = nll_acc(predict_logits(model, Xva, dev), yva)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.cpu(), out / "model.pt")
            save_json({"fold": fold, "arch": arch, "ratio": 0.0, "params": count_params(model),
                       "val_nll": vnll, "val_acc": vacc, "train_sec": round(time.time() - t0, 1), **res},
                      out / "info.json")
            print(f"[saved] {out}  params={count_params(model)}  valfit/{src} acc={vacc:.4f} nll={vnll:.4f}")


if __name__ == "__main__":
    main()
