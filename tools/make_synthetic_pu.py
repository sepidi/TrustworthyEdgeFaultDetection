"""Generate a tiny synthetic dataset with the SAME file layout and .mat structure as Paderborn.

Only for testing the pipeline end-to-end (tools/selftest.sh). Not for any result!
  python tools/make_synthetic_pu.py --out /tmp/tcfd_selftest/data/raw --len 70000
--len must cover at least one feature window: order/envelope features use 4096 samples after x16
decimation, i.e. >= 65,536 raw samples (the real files have ~256,000).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io
import yaml

COND_SPEED = {"N15": 1500, "N09": 900}
COND_TORQUE = {"M07": 0.7, "M01": 0.1}
COND_FORCE = {"F10": 1000, "F04": 400}


def signal(cls, cond, rng, n, fs, sev):
    t = np.arange(n) / fs
    sp, tq, fo = cond.split("_")
    fr = COND_SPEED[sp] / 60.0
    amp = 0.6 + 0.6 * COND_TORQUE[tq] + 0.0004 * COND_FORCE[fo]
    x = 0.3 * np.sin(2 * np.pi * fr * t) + 0.1 * np.sin(2 * np.pi * 2 * fr * t)
    x += rng.normal(0, 0.5, n)
    if cls != "healthy":
        f_def = (3.05 if cls == "outer" else 4.95) * fr
        imp = (np.sin(2 * np.pi * f_def * t) > 0.995).astype(float)
        ring = np.exp(-t[:200] * 3000) * np.sin(2 * np.pi * 3000 * t[:200])
        burst = np.convolve(imp, ring, mode="same")
        if cls == "inner":
            burst *= 1 + 0.5 * np.sin(2 * np.pi * fr * t)
        x += sev * amp * 6 * burst
    current = np.sin(2 * np.pi * 50 * t) + 0.05 * rng.normal(0, 1, n)
    return x.astype(np.float32), current.astype(np.float32)


def write_mat(path, stem, vib, cur):
    dt = [("Name", "O"), ("Data", "O"), ("Raster", "O"), ("Unit", "O")]
    names = [("force", np.zeros(16)), ("phase_current_1", cur), ("phase_current_2", -cur), ("speed", np.zeros(16)),
             ("temp_2_bearing_module", np.zeros(4)), ("torque", np.zeros(16)), ("vibration_1", vib)]
    Y = np.zeros((1, len(names)), dtype=dt)
    for i, (n, d) in enumerate(names):
        Y[0, i] = (n, d[None, :], "HostService", "-")
    scipy.io.savemat(str(path), {stem: {"Info": "synthetic", "X": np.zeros((1, 1)), "Y": Y, "Description": stem}},
                     do_compression=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--len", type=int, default=70000)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    rng = np.random.default_rng(0)
    out = Path(a.out)
    bearings = {**cfg["bearings"], **(cfg.get("train_only_bearings") or {})}
    for b, cls in bearings.items():
        sev = rng.uniform(0.4, 1.0)
        (out / b).mkdir(parents=True, exist_ok=True)
        for code in cfg["conditions"].values():
            for i in range(1, cfg["data"]["n_files"] + 1):
                stem = f"{code}_{b}_{i}"
                vib, cur = signal(cls, code, rng, a.len, cfg["data"]["fs"], sev)
                write_mat(out / b / f"{stem}.mat", stem, vib, cur)
        print(f"wrote {b} ({cls})", flush=True)


if __name__ == "__main__":
    main()
