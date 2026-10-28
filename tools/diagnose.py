"""Why is accuracy low? Run after the workstation stage for ONE fold (minutes, CPU is fine).

Prints, for the dense models of the fold:
  1. accuracy on TRAIN bearings vs VAL bearings vs TEST bearings (source condition)
     -> train >> val/test means the CNN memorises bearing "fingerprints" (overfitting to bearings)
  2. per-test-bearing accuracy and the confusion matrix at the source condition
     -> one bad bearing vs. a systematic class confusion
  3. a classical baseline (logistic regression + random forest on band-pooled spectra, same split)
     -> if the classical baseline is also near chance, the INPUT FEATURE is the problem, not the CNN

  python tools/diagnose.py -c config.yaml --fold A
  python tools/diagnose.py -c config_env.yaml --fold A        # envelope-spectrum variant
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcfd.config import load_config, model_id, path  # noqa: E402
from tcfd.splits import fold_bearings, get_split  # noqa: E402


def bacc(y, p, k):
    return float(np.mean([(p[y == c] == c).mean() for c in range(k) if (y == c).any()]))


def pooled(X, bands=128):
    X = X[:, 0, :].astype(np.float32)
    return X.reshape(len(X), bands, -1).mean(axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--fold", default="A")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    K, src, fold = len(cfg["classes"]), cfg["source_condition"], a.fold
    conds = list(cfg["conditions"])
    fb = fold_bearings(cfg, fold)
    print(f"Fold {fold}  feature={cfg['data'].get('feature', 'fft')}  train={fb['train']}  val={fb['val']}  "
          f"test={fb['test']}\nclasses={cfg['classes']}  chance balanced acc = {1 / K:.3f}\n")

    data = {s: get_split(cfg, fold, s, src, overlap=False) for s in ("train", "valfit", "test")}
    shifted = {c: get_split(cfg, fold, "test", c) for c in conds}

    # ---------------------------------------------------------------- 1+2: dense CNNs
    try:
        import torch
        from tcfd.trainer import predict_logits
        for arch in cfg["train"]["archs"]:
            rd = path(cfg, cfg["train"].get("runs_dir", "runs")) / fold / model_id(arch, 0.0)
            if not (rd / "model.pt").exists():
                print(f"[{arch}] no model at {rd}")
                continue
            m = torch.load(rd / "model.pt", weights_only=False)
            info = json.load(open(rd / "info.json"))
            h = info.get("history", [])
            best = min(h, key=lambda r: r["val_nll"]) if h else {}
            print(f"== {arch}: stopped after {len(h)} epochs, best epoch {best.get('epoch')} "
                  f"(val_nll {best.get('val_nll', float('nan')):.3f})")
            for s in ("train", "valfit", "test"):
                X, y, _, _ = data[s]
                p = predict_logits(m, X, "cpu").argmax(1)
                print(f"   {s:7s} bearings @ {src}: balanced acc {bacc(y, p, K):.3f}  (n={len(y)})")
            for c in [c for c in conds if c != src]:
                X, y, _, _ = shifted[c]
                print(f"   test    bearings @ {c}: balanced acc {bacc(y, predict_logits(m, X, 'cpu').argmax(1), K):.3f}")
            X, y, b, _ = data["test"]
            p = predict_logits(m, X, "cpu").argmax(1)
            print("   per test bearing @ source:  " + "  ".join(
                f"{bb}({cfg['bearings'][bb][:1].upper()})={(p[b == bb] == y[b == bb]).mean():.2f}" for bb in fb["test"]))
            cm = np.zeros((K, K), int)
            np.add.at(cm, (y, p), 1)
            print("   confusion (rows=true " + "/".join(cfg["classes"]) + ", cols=pred):")
            for i, row in enumerate(cm):
                print(f"      {cfg['classes'][i]:8s} {row}")
            print()
    except ImportError:
        print("(torch not available - skipping CNN diagnostics)\n")

    # ---------------------------------------------------------------- 3: classical baseline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xtr, ytr, _, _ = get_split(cfg, fold, "train", src)
    Ftr = pooled(Xtr)
    print("== classical baseline on band-pooled spectra (same bearing-wise split)")
    for name, clf in (("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1,
                                                                                    class_weight="balanced"))),
                      ("rforest", RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced",
                                                         n_jobs=-1, random_state=0))):
        clf.fit(Ftr, ytr)
        tr = bacc(ytr, clf.predict(Ftr), K)
        res = [f"{c}={bacc(shifted[c][1], clf.predict(pooled(shifted[c][0])), K):.3f}" for c in conds]
        print(f"   {name:8s} train={tr:.3f}  test: " + "  ".join(res))
    print("\nReading guide: train >> test for both CNN and baseline -> bearings are not separable by this "
          "feature;\ntry the envelope variant (config_env.yaml). Baseline OK but CNN poor -> CNN overfits; "
          "try --set train.weight_decay=0.01 or fewer epochs.")


if __name__ == "__main__":
    main()
