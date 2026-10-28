"""Compute every metric from saved logits (Jetson TensorRT + workstation PyTorch reference).

For each fold x model x precision:
  * temperature T fitted on valfit / source condition
  * metrics (raw and TS probabilities) on test and valfit, per operating condition
  * agreement with the dense FP32 model of the same fold/arch (prediction "loyalty")
  * conformal sets (LAC & APS) with three calibration variants:
        src      : q from valcal / source condition only (what a naive deployment does)
        pooled   : q from valcal pooled over all conditions
        mondrian : one q per operating condition (drive reports speed/torque -> condition known)
Run:  python -m tcfd.evaluate -c config.yaml
Out:  results/metrics.csv, results/cp.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import base_parser, load_config, path
from .conformal import conformal_quantile, predict_sets, set_metrics, true_scores
from .metrics import all_metrics, fit_temperature, softmax

WS = "ws_fp32"


def discover(cfg, man) -> list[tuple[str, dict, str, Path]]:
    jd = path(cfg, cfg["jetson"]["out_dir"]) / "logits"
    out = []
    for fold, d in man["folds"].items():
        for m in d["models"]:
            ws = path(cfg, cfg["train"].get("runs_dir", "runs")) / fold / m["id"] / "logits_ws"
            if ws.exists():
                out.append((fold, m, WS, ws))
            md = jd / fold / m["id"]
            if md.exists():
                for p in sorted(md.iterdir()):
                    if p.is_dir():
                        out.append((fold, m, p.name, p))
    return out


LOBO_VARIANTS = ("lobo_src", "lobo_pooled", "lobo_mondrian")


def lobo_rows(cfg, fold, m, base, L, labels, conds, src, alpha) -> list[dict]:
    """Conformal sets calibrated on out-of-bearing scores (tcfd.lobo), raw softmax.
    split='test'  : deployed model (this precision) on the test bearings
    split='valfit': bearing-level validation estimate = leave-one-bearing-out inside the calibration pool
                    (used by the deployment rule; never touches test bearings)."""
    d = path(cfg, cfg["train"].get("runs_dir", "runs")) / fold / m["id"] / "lobo"
    if not (d / f"{src}_logits.npy").exists():
        return []
    P = {c: softmax(np.load(d / f"{c}_logits.npy")) for c in conds}
    Y = {c: np.load(d / f"{c}_y.npy") for c in conds}
    B = {c: np.load(d / f"{c}_bearing.npy") for c in conds}
    rows = []
    for kind in ("lac", "aps"):
        sc = {c: true_scores(P[c], Y[c], kind) for c in conds}

        def cal_scores(variant, c, exclude=None):
            def keep(cc):
                return sc[cc] if exclude is None else sc[cc][B[cc] != exclude]
            if variant == "lobo_src":
                return keep(src)
            if variant == "lobo_pooled":
                return np.concatenate([keep(cc) for cc in conds])
            return keep(c)

        for variant in LOBO_VARIANTS:
            for c in conds:
                q = conformal_quantile(cal_scores(variant, c), alpha)
                sets = predict_sets(softmax(L[f"test_{c}"]), q, kind)
                rows.append({**base, "score": kind, "variant": variant, "split": "test", "cond": c, "q": q,
                             **set_metrics(sets, labels[(fold, f"test_{c}")])})
                all_sets, all_y = [], []
                for b in np.unique(B[c]):
                    mb = B[c] == b
                    qb = conformal_quantile(cal_scores(variant, c, exclude=b), alpha)
                    all_sets.append(predict_sets(P[c][mb], qb, kind))
                    all_y.append(Y[c][mb])
                rows.append({**base, "score": kind, "variant": variant, "split": "valfit", "cond": c, "q": q,
                             **set_metrics(np.concatenate(all_sets), np.concatenate(all_y))})
    return rows


def main():
    p = base_parser("Evaluate all logits")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    exp = path(cfg, cfg["export"]["dir"])
    man = json.load(open(exp / "manifest.json"))
    conds = list(cfg["conditions"].keys())
    src = cfg["source_condition"]
    K = len(cfg["classes"])
    an = cfg["analysis"]
    alpha, bins = an["alpha"], an["ece_bins"]
    res = path(cfg, an["results_dir"])
    res.mkdir(parents=True, exist_ok=True)

    labels = {}
    for fold in man["folds"]:
        for s in man["eval_sets"]:
            labels[(fold, s)] = np.load(exp / fold / "data" / f"{s}_y.npy")

    runs = discover(cfg, man)
    if not runs:
        raise SystemExit("No logits found. Run tcfd.export (workstation) and/or copy jetson_out/logits back.")
    print(f"Evaluating {len(runs)} (fold, model, precision) combinations")
    # reference predictions for agreement: dense model, Jetson fp32 if available else workstation
    ref_pred = {}
    for fold, m, prec, d in runs:
        if m["ratio"] == 0.0 and prec in ("fp32", WS):
            key = (fold, m["arch"])
            if prec == "fp32" or key not in ref_pred:
                ref_pred[key] = {s: np.load(d / f"{s}.npy").argmax(1) for s in man["eval_sets"]
                                 if (d / f"{s}.npy").exists()}

    mrows, crows = [], []
    for i, (fold, m, prec, d) in enumerate(runs, 1):
        L = {s: np.load(d / f"{s}.npy") for s in man["eval_sets"] if (d / f"{s}.npy").exists()}
        if len(L) != len(man["eval_sets"]):
            print(f"[WARN] incomplete logits for {fold}/{m['id']}/{prec} ({len(L)} sets) - skipped")
            continue
        T = fit_temperature(L[f"valfit_{src}"], labels[(fold, f"valfit_{src}")])
        base = {"fold": fold, "model": m["id"], "arch": m["arch"], "ratio": m["ratio"], "precision": prec,
                "params": m.get("params"), "macs": m.get("macs"), "T": T}
        ref = ref_pred.get((fold, m["arch"]), {})
        probs_ts = {s: softmax(L[s], T) for s in L}
        for split in ("test", "valfit"):
            for c in conds:
                s = f"{split}_{c}"
                y = labels[(fold, s)]
                for cal, P in (("raw", softmax(L[s])), ("ts", probs_ts[s])):
                    row = {**base, "split": split, "cond": c, "calib": cal, "n": len(y),
                           **all_metrics(P, y, K, bins)}
                    row["agree_dense"] = float((L[s].argmax(1) == ref[s]).mean()) if s in ref else np.nan
                    mrows.append(row)
        for kind in ("lac", "aps"):
            sc = {c: true_scores(probs_ts[f"valcal_{c}"], labels[(fold, f"valcal_{c}")], kind) for c in conds}
            q = {"src": {c: conformal_quantile(sc[src], alpha) for c in conds},
                 "pooled": {c: conformal_quantile(np.concatenate(list(sc.values())), alpha) for c in conds},
                 "mondrian": {c: conformal_quantile(sc[c], alpha) for c in conds}}
            for variant, qs in q.items():
                for split in ("test", "valfit"):
                    for c in conds:
                        s = f"{split}_{c}"
                        sets = predict_sets(probs_ts[s], qs[c], kind)
                        crows.append({**base, "score": kind, "variant": variant, "split": split, "cond": c,
                                      "q": qs[c], **set_metrics(sets, labels[(fold, s)])})
        crows += lobo_rows(cfg, fold, m, base, L, labels, conds, src, alpha)
        if i % 10 == 0 or i == len(runs):
            print(f"  {i}/{len(runs)} done", flush=True)
    pd.DataFrame(mrows).to_csv(res / "metrics.csv", index=False)
    pd.DataFrame(crows).to_csv(res / "cp.csv", index=False)
    print(f"Wrote {res/'metrics.csv'} ({len(mrows)} rows) and {res/'cp.csv'} ({len(crows)} rows)")


if __name__ == "__main__":
    main()
