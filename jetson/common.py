"""Shared helpers for the Jetson-side scripts (no PyTorch needed on the board).

Backends
  trt : TensorRT engines via Polygraphy (the real thing, on the Jetson)
  ort : ONNX Runtime on CPU (dry run on any machine; only 'fp32' is meaningful)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_manifest(export_dir: Path) -> dict:
    p = Path(export_dir) / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - copy the workstation 'export/' folder first")
    return json.load(open(p))


def iter_models(man: dict, folds=None, archs=None, ratios=None):
    for fold, d in man["folds"].items():
        if folds and fold not in folds:
            continue
        for m in d["models"]:
            if archs and m["arch"] not in archs:
                continue
            if ratios and not any(abs(m["ratio"] - float(r)) < 1e-9 for r in ratios):
                continue
            yield fold, m


def engine_path(out_dir: Path, fold: str, mid: str, prec: str) -> Path:
    return Path(out_dir) / "engines" / fold / mid / f"{prec}.engine"


def logits_dir(out_dir: Path, fold: str, mid: str, prec: str) -> Path:
    return Path(out_dir) / "logits" / fold / mid / prec


class Runner:
    """Uniform infer(X float32 (n,1,F)) -> logits, for TensorRT or ONNX Runtime."""

    def __init__(self, backend: str, model_file: Path, input_name="x", output_name="logits"):
        self.backend, self.inp, self.out = backend, input_name, output_name
        if backend == "trt":
            from polygraphy.backend.common import BytesFromPath
            from polygraphy.backend.trt import EngineFromBytes, TrtRunner
            self._r = TrtRunner(EngineFromBytes(BytesFromPath(str(model_file))))
        elif backend == "ort":
            import onnxruntime as ort
            self._s = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        else:
            raise ValueError(backend)

    def __enter__(self):
        if self.backend == "trt":
            self._r.activate()
        return self

    def __exit__(self, *exc):
        if self.backend == "trt":
            self._r.deactivate()

    def infer(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        if self.backend == "trt":
            return np.array(self._r.infer({self.inp: X})[self.out], dtype=np.float32, copy=True)
        return self._s.run([self.out], {self.inp: X})[0].astype(np.float32)

    def infer_all(self, X: np.ndarray, bs: int) -> np.ndarray:
        return np.concatenate([self.infer(X[i:i + bs]) for i in range(0, len(X), bs)])
