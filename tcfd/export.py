"""Export everything the Jetson needs + workstation FP32 reference logits.

For each fold:
  export/<fold>/data/<split>_<Cx>.npy        float16 inputs (n,1,F)    split in {valfit, valcal, test}
  export/<fold>/data/<split>_<Cx>_y.npy      labels (kept for analysis; Jetson does not need them)
  export/<fold>/data/<split>_<Cx>_bearing.npy
  export/<fold>/calib/{src,all}.npy          INT8 calibration windows (class-balanced, valfit only)
  export/<fold>/onnx/<model_id>.onnx         dynamic batch, input "x", output "logits"
  export/<fold>/ref/<model_id>_test_C0.npy   workstation FP32 logits for the Jetson smoke test
  export/manifest.json
and runs/<fold>/<model_id>/logits_ws/<split>_<Cx>.npy (PyTorch FP32 logits on the workstation).

Run:  python -m tcfd.export -c config.yaml
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .config import (all_ratios, archs, base_parser, folds, get_device, load_config, model_id, path,
                     save_json)
from .data_pu import n_features
from .splits import get_split
from .trainer import predict_logits

EVAL_SPLITS = ("valfit", "valcal", "test")


def balanced_sample(X, y, n, rng):
    classes = np.unique(y)
    per = max(1, n // len(classes))
    idx = np.concatenate([rng.choice(np.where(y == c)[0], size=min(per, (y == c).sum()), replace=False)
                          for c in classes])
    rng.shuffle(idx)
    return X[idx]


def export_onnx(model, out_path, n_feat, opset):
    model = model.cpu().eval()
    dummy = torch.randn(1, 1, n_feat)
    kw = dict(input_names=["x"], output_names=["logits"],
              dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(model, dummy, str(out_path), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo kwarg
        torch.onnx.export(model, dummy, str(out_path), **kw)
    import onnx
    onnx.checker.check_model(onnx.load(str(out_path)))


def check_ort(onnx_path, X, ref, atol=1e-3):
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = s.run(["logits"], {"x": X.astype(np.float32)})[0]
    return float(np.abs(out - ref).max())


def main():
    p = base_parser("Export ONNX models, eval data, calibration sets and workstation logits")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    dev = get_device(cfg)
    F = n_features(cfg)
    exp = path(cfg, cfg["export"]["dir"])
    rng = np.random.default_rng(cfg["seed"])
    conds = list(cfg["conditions"].keys())
    src = cfg["source_condition"]
    manifest = {"n_features": F, "input": "x", "output": "logits", "classes": cfg["classes"],
                "conditions": cfg["conditions"], "source_condition": src,
                "eval_sets": [f"{s}_{c}" for s in EVAL_SPLITS for c in conds],
                "precisions": cfg["jetson"]["precisions"], "infer_batch": cfg["jetson"]["infer_batch"],
                "bench_fold": cfg["jetson"]["bench_fold"], "folds": {}}
    old = exp / "manifest.json"
    if old.exists():  # keep folds exported in earlier runs
        manifest["folds"].update(json.load(open(old)).get("folds", {}))
    for fold in folds(cfg, a.folds):
        print(f"\n=== Fold {fold}")
        data = {}
        for s in EVAL_SPLITS:
            for c in conds:
                X, y, b, _ = get_split(cfg, fold, s, c)
                data[f"{s}_{c}"] = (X, y)
                d = exp / fold / "data"
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / f"{s}_{c}.npy", X.astype(np.float16))
                np.save(d / f"{s}_{c}_y.npy", y)
                np.save(d / f"{s}_{c}_bearing.npy", b)
                print(f"  {s}_{c}: {len(y)} windows, class counts {np.bincount(y, minlength=len(cfg['classes']))}")
        n = cfg["export"]["calib_windows"]
        cal = exp / fold / "calib"
        cal.mkdir(parents=True, exist_ok=True)
        Xs, ys = data[f"valfit_{src}"]
        np.save(cal / "src.npy", balanced_sample(Xs, ys, n, rng).astype(np.float16))
        parts = [balanced_sample(*data[f"valfit_{c}"], max(1, n // len(conds)), rng) for c in conds]
        allX = np.concatenate(parts)
        rng.shuffle(allX)
        np.save(cal / "all.npy", allX.astype(np.float16))
        models = []
        for arch in archs(cfg, a.archs):
            for r in all_ratios(cfg):
                mid = model_id(arch, r)
                rd = path(cfg, cfg["train"].get("runs_dir", "runs")) / fold / mid
                if not (rd / "model.pt").exists():
                    print(f"  [WARN] missing {rd}/model.pt - skipped")
                    continue
                m = torch.load(rd / "model.pt", weights_only=False)
                info = json.load(open(rd / "info.json"))
                onnx_path = exp / fold / "onnx" / f"{mid}.onnx"
                export_onnx(m, onnx_path, F, cfg["export"]["opset"])
                m = m.to(dev)
                (rd / "logits_ws").mkdir(exist_ok=True)
                for k, (X, _) in data.items():
                    np.save(rd / "logits_ws" / f"{k}.npy", predict_logits(m, X, dev))
                ref = np.load(rd / "logits_ws" / f"test_{src}.npy")
                (exp / fold / "ref").mkdir(exist_ok=True)
                np.save(exp / fold / "ref" / f"{mid}_test_{src}.npy", ref)
                diff = check_ort(onnx_path, data[f"test_{src}"][0][:256], ref[:256])
                models.append({"id": mid, "arch": arch, "ratio": r, "params": info.get("params"),
                               "macs": info.get("macs")})
                print(f"  exported {mid}  params={info.get('params')}  ORT|diff|max={diff}")
        manifest["folds"][fold] = {"models": models}
    save_json(manifest, exp / "manifest.json")
    print(f"\nWrote {exp/'manifest.json'}.  Copy '{exp}' to the Jetson (scripts/sync_to_jetson.sh).")


if __name__ == "__main__":
    main()
