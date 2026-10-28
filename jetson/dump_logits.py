"""Run every engine on every evaluation set and save logits (the only thing analysis needs).

Run on the Jetson:
  python3 jetson/dump_logits.py --export-dir export --out-dir jetson_out [--folds A]
Dry run anywhere (ONNX Runtime CPU, fp32 only):
  python3 jetson/dump_logits.py --backend ort --precisions fp32
Out: jetson_out/logits/<fold>/<model_id>/<precision>/<split>_<Cx>.npy
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import Runner, engine_path, iter_models, load_manifest, logits_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default="export")
    ap.add_argument("--out-dir", default="jetson_out")
    ap.add_argument("--backend", choices=["trt", "ort"], default="trt")
    ap.add_argument("--folds", nargs="*")
    ap.add_argument("--archs", nargs="*")
    ap.add_argument("--ratios", nargs="*")
    ap.add_argument("--precisions", nargs="*")
    ap.add_argument("--sets", nargs="*", help="subset of eval sets, e.g. test_C0")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    exp, out = Path(a.export_dir), Path(a.out_dir)
    man = load_manifest(exp)
    precs = a.precisions or man["precisions"]
    sets = a.sets or man["eval_sets"]
    bs = man["infer_batch"]
    data_cache: dict = {}
    for fold, m in iter_models(man, a.folds, a.archs, a.ratios):
        for prec in precs:
            od = logits_dir(out, fold, m["id"], prec)
            if not a.force and all((od / f"{s}.npy").exists() for s in sets):
                print(f"skip {fold}/{m['id']}/{prec} (done)")
                continue
            mf = engine_path(out, fold, m["id"], prec) if a.backend == "trt" else exp / fold / "onnx" / f"{m['id']}.onnx"
            if not mf.exists():
                print(f"[WARN] missing {mf} - skipped")
                continue
            od.mkdir(parents=True, exist_ok=True)
            t0, n = time.time(), 0
            with Runner(a.backend, mf, man["input"], man["output"]) as r:
                for s in sets:
                    key = (fold, s)
                    if key not in data_cache:
                        if len(data_cache) > 24:
                            data_cache.clear()
                        data_cache[key] = np.load(exp / fold / "data" / f"{s}.npy")
                    X = data_cache[key]
                    np.save(od / f"{s}.npy", r.infer_all(X, bs))
                    n += len(X)
            print(f"{fold}/{m['id']}/{prec}: {n} windows in {time.time() - t0:.1f}s", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
