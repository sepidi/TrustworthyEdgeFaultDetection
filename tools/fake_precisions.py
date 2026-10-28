"""TESTING ONLY: clone fp32 logits into fp16/int8_src/int8_all dirs with small perturbations,
so the analysis/figure code paths for reduced precision can be exercised without a Jetson.
  python tools/fake_precisions.py --out-dir /tmp/tcfd_selftest/jetson_out
"""
import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
rng = np.random.default_rng(1)
noise = {"fp16": 0.001, "int8_src": 0.4, "int8_all": 0.25}
for d in Path(a.out_dir, "logits").glob("*/*/fp32"):
    for prec, s in noise.items():
        od = d.parent / prec
        od.mkdir(exist_ok=True)
        for f in d.glob("*.npy"):
            x = np.load(f)
            np.save(od / f.name, (x * (1 + s * 0.3) + rng.normal(0, s, x.shape)).astype(np.float32))
print("fake precisions written (TEST ONLY)")
