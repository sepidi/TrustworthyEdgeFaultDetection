"""5-minute sanity check on the Jetson BEFORE launching the full build.

Builds all precisions for the dense models of one fold, runs test_C0, and compares with
the workstation PyTorch FP32 logits shipped in export/<fold>/ref/.
Expected: fp32 max|diff| < 1e-2 and agreement ~1.000; fp16 agreement > 0.99;
int8 agreement typically > 0.95 (lower is a finding, not a bug, if fp32 matches).

  python3 jetson/smoke_test.py --export-dir export --out-dir jetson_out
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import load_manifest, logits_dir

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default="export")
    ap.add_argument("--out-dir", default="jetson_out")
    ap.add_argument("--fold", default=None)
    ap.add_argument("--backend", choices=["trt", "ort"], default="trt")
    a = ap.parse_args()
    exp, out = Path(a.export_dir), Path(a.out_dir)
    man = load_manifest(exp)
    fold = a.fold or man["bench_fold"]
    src = man["source_condition"]
    precs = man["precisions"] if a.backend == "trt" else ["fp32"]
    common = ["--export-dir", str(exp), "--out-dir", str(out), "--folds", fold, "--ratios", "0.0"]
    if a.backend == "trt":
        subprocess.run([sys.executable, str(HERE / "build_engines.py"), *common], check=True)
    subprocess.run([sys.executable, str(HERE / "dump_logits.py"), *common, "--backend", a.backend,
                    "--precisions", *precs, "--sets", f"test_{src}"], check=True)
    ok = True
    for m in man["folds"][fold]["models"]:
        if m["ratio"] != 0.0:
            continue
        ref = np.load(exp / fold / "ref" / f"{m['id']}_test_{src}.npy")
        y = np.load(exp / fold / "data" / f"test_{src}_y.npy")
        for p in precs:
            f = logits_dir(out, fold, m["id"], p) / f"test_{src}.npy"
            lg = np.load(f)
            diff = float(np.abs(lg - ref).max())
            agree = float((lg.argmax(1) == ref.argmax(1)).mean())
            acc = float((lg.argmax(1) == y).mean())
            flag = ""
            if p == "fp32":
                # A prediction flip on a window whose top-two logits are closer than the
                # logit error is a tie broken by rounding, not an export bug. Only count
                # flips where the workstation top-two margin exceeds 1e-2.
                m2 = np.sort(ref, axis=1)
                margin = m2[:, -1] - m2[:, -2]
                real = int(((lg.argmax(1) != ref.argmax(1)) & (margin > 1e-2)).sum())
                if diff > 1e-2 or real > 0:
                    flag, ok = "  <-- CHECK (fp32 should match the workstation)", False
                elif agree < 0.999:
                    flag = f"  (ok: {int(round((1 - agree) * len(y)))} tie-flip(s) within {diff:.0e} noise)"
            print(f"{fold}/{m['id']}/{p:9s} max|diff|={diff:.4g} agreement={agree:.4f} acc={acc:.4f}{flag}")
    print("\nSMOKE TEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
