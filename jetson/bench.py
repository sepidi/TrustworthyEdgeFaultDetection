"""Latency / throughput / energy benchmark of every engine (one fold is enough: shapes are fold-independent).

Per engine:
  * trtexec, batch 1, --duration D  -> GPU-compute & end-to-end latency percentiles, throughput (qps)
    while tegrastats logs the VDD_IN rail  -> mean power -> energy per inference
  * trtexec, batch 32               -> throughput (windows/s)
  * Python (Polygraphy) loop, batch 1 -> "application" latency incl. host<->device copies
Idle power is measured at the start and every --idle-every engines; dynamic energy subtracts it.

Run on the Jetson (lock clocks first, see README):
  python3 jetson/bench.py --export-dir export --out-dir jetson_out
Dry run anywhere:  python3 jetson/bench.py --backend ort --precisions fp32 --duration 2
Out: jetson_out/bench/bench.csv, jetson_out/bench/env.txt, raw logs in jetson_out/bench/logs/
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from common import Runner, engine_path, iter_models, load_manifest

TRTEXEC_CANDIDATES = ["/usr/src/tensorrt/bin/trtexec", "trtexec"]
RAIL = re.compile(r"(?:VDD_IN|POM_5V_IN) (\d+)(?:mW)?")


def find_trtexec():
    for c in TRTEXEC_CANDIDATES:
        p = shutil.which(c) or (c if os.path.exists(c) else None)
        if p:
            return p
    return None


class PowerLog:
    """Context manager logging tegrastats; .mean_w() returns mean VDD_IN in watts (NaN if unavailable)."""

    def __init__(self, logfile: Path, interval_ms: int = 100):
        self.f, self.iv, self.proc = logfile, interval_ms, None

    def __enter__(self):
        if shutil.which("tegrastats"):
            self.f.parent.mkdir(parents=True, exist_ok=True)
            if self.f.exists():
                self.f.unlink()
            self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.iv), "--logfile", str(self.f)],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.3)
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            subprocess.run(["tegrastats", "--stop"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def mean_w(self, skip: int = 5) -> float:
        if not self.f.exists():
            return float("nan")
        vals = [int(m.group(1)) for line in open(self.f) for m in [RAIL.search(line)] if m]
        vals = vals[skip:] if len(vals) > skip + 2 else vals
        return float(np.mean(vals)) / 1000.0 if vals else float("nan")


def idle_power(logdir: Path, seconds: float) -> float:
    with PowerLog(logdir / "idle.log") as pl:
        time.sleep(seconds)
    return pl.mean_w()


def parse_trtexec(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        for tag, key in (("GPU Compute Time:", "gpu"), ("Latency:", "lat")):
            if tag in line and "min =" in line:
                for name, val in re.findall(r"(min|max|mean|median|percentile\(\d+%\)) = ([\d.]+) ms", line):
                    name = name.replace("percentile(", "p").replace("%)", "")
                    out[f"trt_{key}_{name}_ms"] = float(val)
        m = re.search(r"Throughput: ([\d.]+) qps", line)
        if m:
            out["trt_qps"] = float(m.group(1))
    return out


def run_trtexec(trtexec: str, eng: Path, batch: int, F: int, duration: float, log: Path) -> dict:
    cmd = [trtexec, f"--loadEngine={eng}", f"--shapes=x:{batch}x1x{F}", "--warmUp=500",
           f"--duration={int(max(1, duration))}", "--percentile=99"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(cmd) + "\n\n" + r.stdout + "\n" + r.stderr)
    if r.returncode != 0:
        print(f"[WARN] trtexec failed for {eng} (see {log})")
        return {}
    return parse_trtexec(r.stdout)


def python_latency(runner: Runner, x1: np.ndarray, n: int, warm: int = 200) -> dict:
    for _ in range(warm):
        runner.infer(x1)
    t = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter()
        runner.infer(x1)
        t[i] = (time.perf_counter() - t0) * 1e3
    return {"py_lat_p50_ms": float(np.median(t)), "py_lat_p99_ms": float(np.percentile(t, 99)),
            "py_lat_mean_ms": float(t.mean())}


def python_energy_loop(runner: Runner, x1: np.ndarray, seconds: float):
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        runner.infer(x1)
        n += 1
    return n / (time.perf_counter() - t0)


def env_info(out: Path, backend: str):
    lines = [f"backend={backend}", f"time={time.ctime()}"]
    for cmd in (["cat", "/etc/nv_tegra_release"], ["nvpmodel", "-q"], ["uname", "-a"],
                ["python3", "-c", "import tensorrt as t; print('TensorRT', t.__version__)"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            lines.append(f"$ {' '.join(cmd)}\n{r.stdout.strip()}{r.stderr.strip()}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"$ {' '.join(cmd)} -> {e}")
    out.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default="export")
    ap.add_argument("--out-dir", default="jetson_out")
    ap.add_argument("--backend", choices=["trt", "ort"], default="trt")
    ap.add_argument("--folds", nargs="*", help="default: manifest bench_fold")
    ap.add_argument("--archs", nargs="*")
    ap.add_argument("--ratios", nargs="*")
    ap.add_argument("--precisions", nargs="*")
    ap.add_argument("--duration", type=float, default=10.0, help="seconds per energy/latency run")
    ap.add_argument("--py-iters", type=int, default=2000)
    ap.add_argument("--idle-seconds", type=float, default=8.0)
    ap.add_argument("--idle-every", type=int, default=10)
    ap.add_argument("--no-trtexec", action="store_true")
    a = ap.parse_args()
    exp, out = Path(a.export_dir), Path(a.out_dir)
    man = load_manifest(exp)
    F = man["n_features"]
    precs = a.precisions or man["precisions"]
    folds = a.folds or [man["bench_fold"]]
    bdir = out / "bench"
    (bdir / "logs").mkdir(parents=True, exist_ok=True)
    env_info(bdir / "env.txt", a.backend)
    trtexec = None if (a.no_trtexec or a.backend != "trt") else find_trtexec()
    if a.backend == "trt" and trtexec is None and not a.no_trtexec:
        print("[WARN] trtexec not found; only Python-loop latency/energy will be recorded")
    x1 = None
    rows = []
    jobs = [(f, m, p) for f, m in iter_models(man, folds, a.archs, a.ratios) for p in precs]
    p_idle = idle_power(bdir / "logs", a.idle_seconds)
    print(f"idle VDD_IN = {p_idle:.3f} W")
    for k, (fold, m, prec) in enumerate(jobs, 1):
        if k > 1 and (k - 1) % a.idle_every == 0:
            p_idle = idle_power(bdir / "logs", a.idle_seconds)
            print(f"idle VDD_IN = {p_idle:.3f} W")
        mf = engine_path(out, fold, m["id"], prec) if a.backend == "trt" else exp / fold / "onnx" / f"{m['id']}.onnx"
        if not mf.exists():
            print(f"[WARN] missing {mf}")
            continue
        if x1 is None:
            x1 = np.load(exp / fold / "data" / f"test_{man['source_condition']}.npy")[:1].astype(np.float32)
        row = {"fold": fold, "model": m["id"], "arch": m["arch"], "ratio": m["ratio"], "precision": prec,
               "params": m.get("params"), "macs": m.get("macs"), "engine_bytes": mf.stat().st_size,
               "p_idle_w": p_idle}
        tag = f"{fold}_{m['id']}_{prec}"
        if trtexec:
            with PowerLog(bdir / "logs" / f"{tag}_b1.tegra") as pl:
                row.update(run_trtexec(trtexec, mf, 1, F, a.duration, bdir / "logs" / f"{tag}_b1.txt"))
            row["p_load_w"] = pl.mean_w()
            row["qps_b1"] = row.pop("trt_qps", float("nan"))
            b32 = run_trtexec(trtexec, mf, 32, F, max(2, a.duration / 2), bdir / "logs" / f"{tag}_b32.txt")
            row["windows_per_s_b32"] = b32.get("trt_qps", float("nan")) * 32
        with Runner(a.backend, mf, man["input"], man["output"]) as r:
            row.update(python_latency(r, x1, a.py_iters))
            if not trtexec:
                with PowerLog(bdir / "logs" / f"{tag}_py.tegra") as pl:
                    row["qps_b1"] = python_energy_loop(r, x1, a.duration)
                row["p_load_w"] = pl.mean_w()
        qps = row.get("qps_b1", float("nan"))
        row["energy_mj_per_inf"] = row["p_load_w"] / qps * 1e3 if qps == qps and qps > 0 else float("nan")
        row["dyn_energy_mj_per_inf"] = ((row["p_load_w"] - p_idle) / qps * 1e3
                                        if qps == qps and qps > 0 else float("nan"))
        rows.append(row)
        print(f"[{k}/{len(jobs)}] {tag}: " + ", ".join(
            f"{key}={row[key]:.4g}" for key in ("trt_gpu_median_ms", "py_lat_p50_ms", "qps_b1", "p_load_w",
                                                "dyn_energy_mj_per_inf")
            if isinstance(row.get(key), float)), flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda s: (s not in rows[0], s)) if rows else []
    with open(bdir / "bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {bdir/'bench.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
