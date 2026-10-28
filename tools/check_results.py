"""Compare regenerated results with the shipped ones (Tier 1 check).

    cp -r results results_shipped        # before running the analysis
    bash scripts/run_analysis.sh
    python tools/check_results.py results_shipped results

LaTeX tables and Markdown files must be byte-identical. CSV files are compared numerically (relative
tolerance 1e-9), because different pandas/NumPy versions may print the last floating-point digit
differently. PDF figures are only checked for presence (their bytes depend on the matplotlib version).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    A, B = Path(sys.argv[1]), Path(sys.argv[2])
    bad = 0
    for fa in sorted(A.rglob("*")):
        if not fa.is_file():
            continue
        rel = fa.relative_to(A)
        fb = B / rel
        if not fb.exists():
            print(f"MISSING  {rel}")
            bad += 1
            continue
        if fa.suffix in (".tex", ".md"):
            ok = fa.read_bytes() == fb.read_bytes()
        elif fa.suffix == ".csv":
            x, y = pd.read_csv(fa), pd.read_csv(fb)
            ok = list(x.columns) == list(y.columns) and x.shape == y.shape
            if ok:
                num = x.select_dtypes("number").columns
                oth = [c for c in x.columns if c not in num]
                ok = (np.allclose(x[num].to_numpy(float), y[num].to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True)
                      and x[oth].astype(str).equals(y[oth].astype(str)))
        else:
            ok = True  # figures: presence only
        print(f"{'ok      ' if ok else 'DIFFERS '} {rel}")
        bad += not ok
    print("ALL RESULTS REPRODUCED" if not bad else f"{bad} file(s) differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
