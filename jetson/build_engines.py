"""Build TensorRT engines (FP32 / FP16 / INT8 with two calibration sets) from the exported ONNX.

  fp32      : no reduced-precision flags
  fp16      : FP16 enabled
  int8_src  : INT8 (entropy calibrator 2), calibration windows from the SOURCE condition only
  int8_all  : INT8, calibration windows spanning ALL four operating conditions
INT8 engines are built without the FP16 flag, so non-INT8 layers fall back to FP32
(isolates the INT8 effect).

Run on the Jetson:
  python3 jetson/build_engines.py --export-dir export --out-dir jetson_out [--folds A] [--force]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import engine_path, iter_models, load_manifest


def build(onnx_path: Path, eng_path: Path, prec: str, n_feat: int, calib: np.ndarray | None,
          cache: Path | None, max_batch: int):
    from polygraphy.backend.trt import (Calibrator, CreateConfig, EngineFromNetwork,
                                        NetworkFromOnnxPath, Profile, SaveEngine)
    profiles = [Profile().add("x", min=(1, 1, n_feat), opt=(64, 1, n_feat), max=(max_batch, 1, n_feat))]
    kw = {"profiles": profiles}
    if prec == "fp16":
        kw["fp16"] = True
    elif prec.startswith("int8"):
        def loader():  # full batches only: TensorRT calibrates at the profile's opt shape (64)
            for i in range(0, len(calib) - len(calib) % 64, 64):
                yield {"x": np.ascontiguousarray(calib[i:i + 64], dtype=np.float32)}
        kw["int8"] = True
        kw["calibrator"] = Calibrator(data_loader=loader(), cache=str(cache))
    elif prec != "fp32":
        raise ValueError(prec)
    eng_path.parent.mkdir(parents=True, exist_ok=True)
    engine = EngineFromNetwork(NetworkFromOnnxPath(str(onnx_path)), config=CreateConfig(**kw))
    SaveEngine(engine, str(eng_path))()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default="export")
    ap.add_argument("--out-dir", default="jetson_out")
    ap.add_argument("--folds", nargs="*")
    ap.add_argument("--archs", nargs="*")
    ap.add_argument("--ratios", nargs="*")
    ap.add_argument("--precisions", nargs="*")
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    from polygraphy.logger import G_LOGGER
    try:
        G_LOGGER.module_severity = G_LOGGER.WARNING
    except Exception:  # noqa: BLE001 - older polygraphy
        G_LOGGER.severity = G_LOGGER.WARNING
    exp, out = Path(a.export_dir), Path(a.out_dir)
    man = load_manifest(exp)
    precs = a.precisions or man["precisions"]
    F = man["n_features"]
    todo = [(f, m, p) for f, m in iter_models(man, a.folds, a.archs, a.ratios) for p in precs]
    print(f"{len(todo)} engines to build")
    calib_cache = {}
    fails = []
    for k, (fold, m, prec) in enumerate(todo, 1):
        ep = engine_path(out, fold, m["id"], prec)
        if ep.exists() and not a.force:
            print(f"[{k}/{len(todo)}] skip {ep} (exists)")
            continue
        calib, cache = None, None
        if prec.startswith("int8"):
            which = prec.split("_", 1)[1]  # src | all
            key = (fold, which)
            if key not in calib_cache:
                calib_cache[key] = np.load(exp / fold / "calib" / f"{which}.npy").astype(np.float32)
            calib = calib_cache[key]
            cache = ep.with_suffix(".calib_cache")
            if a.force and cache.exists():
                cache.unlink()
        t0 = time.time()
        try:
            build(exp / fold / "onnx" / f"{m['id']}.onnx", ep, prec, F, calib, cache, a.max_batch)
            print(f"[{k}/{len(todo)}] built {fold}/{m['id']}/{prec} in {time.time() - t0:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            fails.append((fold, m["id"], prec, repr(e)))
            print(f"[{k}/{len(todo)}] FAILED {fold}/{m['id']}/{prec}: {e}", flush=True)
    if fails:
        print(f"\n{len(fails)} builds failed:")
        for f in fails:
            print("  ", f)
        raise SystemExit(1)
    print("All engines built.")


if __name__ == "__main__":
    main()
