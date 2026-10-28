"""Compare two trees of logits (.npy) file by file, e.g. shipped vs. regenerated.

    python tools/compare_logits.py runs_shipped runs --pattern "*/*/logits_ws/*.npy"
    python tools/compare_logits.py jetson_out_shipped/logits jetson_out/logits --pattern "*/*/*/*.npy"

Reports, per file and overall, the largest absolute logit difference and the label agreement. Small
differences are expected between GPUs, drivers, cuDNN/TensorRT versions (non-deterministic kernels,
different tactic selection); see README section "Expected tolerances".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--pattern", default="**/*.npy")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    A, B = Path(a.a), Path(a.b)
    worst, agrees, n, missing = 0.0, [], 0, 0
    for fa in sorted(A.glob(a.pattern)):
        rel = fa.relative_to(A)
        fb = B / rel
        if not fb.exists():
            missing += 1
            continue
        x, y = np.load(fa), np.load(fb)
        if x.ndim != 2 or x.shape != y.shape:  # labels / bearing ids: must be identical
            if not np.array_equal(x, y):
                print(f"[DIFF] {rel}: arrays differ")
            continue
        d = float(np.abs(x.astype(np.float64) - y).max())
        ag = float((x.argmax(1) == y.argmax(1)).mean())
        worst, n = max(worst, d), n + 1
        agrees.append(ag)
        if a.verbose or ag < 0.99:
            print(f"{rel}: max|diff|={d:.2e} agreement={ag:.4f}")
    if not n:
        raise SystemExit("no matching logit files found")
    print(f"{n} logit files compared ({missing} missing in {B}): max|diff|={worst:.3e}, "
          f"label agreement mean={np.mean(agrees):.4f} min={np.min(agrees):.4f}")


if __name__ == "__main__":
    main()
