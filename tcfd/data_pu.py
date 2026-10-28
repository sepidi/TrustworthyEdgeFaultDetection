"""Paderborn (PU / KAt DataCenter) loading and feature caching.

Each PU measurement file  <raw_dir>/<BEARING>/<COND>_<BEARING>_<i>.mat  holds one
MATLAB struct (variable named like the file stem) whose field ``Y`` is a struct
array of channels with fields ``Name``/``Data``/``Raster``/``Unit``. Channel names
include ``vibration_1``, ``phase_current_1``, ``phase_current_2``, ``speed``,
``torque``, ``force``, ``temp_2_bearing_module``.

Features (data.feature):
  fft      : windowed raw signal -> Hann -> |rFFT| (DC dropped) -> log1p -> per-window z-score
  envelope : high-pass -> Hilbert envelope -> decimate (64 kHz -> 4 kHz) -> same spectral pipeline,
             i.e. a log envelope spectrum (classic bearing-fault feature, ~1 Hz resolution at 4096/4 kHz)
  order    : envelope spectrum resampled to shaft orders 0..max_order (n_order_bins), i.e. speed-normalised  Cached per (bearing, condition) as float16 in
<proc_dir>/<signal>/<BEARING>/<Cx>.npz with arrays X (n, F), file_id, win_idx.

Run:  python -m tcfd.data_pu -c config.yaml [--workers 8] [--force]
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.signal

from .config import base_parser, load_config, path

CHANNEL = {"vibration": "vibration_1", "current": "phase_current_1"}


# ----------------------------------------------------------------------------- I/O
def find_mat(raw_dir: Path, bearing: str, cond_code: str, i: int) -> Path | None:
    name = f"{cond_code}_{bearing}_{i}.mat"
    direct = raw_dir / bearing / name
    if direct.exists():
        return direct
    hits = glob.glob(str(raw_dir / bearing / "**" / name), recursive=True)
    return Path(hits[0]) if hits else None


def load_channel(mat_path: Path, channel: str) -> np.ndarray:
    m = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    keys = [k for k in m if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No variables in {mat_path}")
    rec = m[keys[0]]
    Y = np.atleast_1d(rec.Y)
    names = []
    for ch in Y:
        name = str(ch.Name).strip()
        names.append(name)
        if name == channel:
            return np.asarray(ch.Data, dtype=np.float64).ravel()
    raise KeyError(f"{channel!r} not in {mat_path.name}; channels: {names}")


# ------------------------------------------------------------------------ features
def frame(x: np.ndarray, window: int, stride: int) -> np.ndarray:
    if len(x) < window:
        return np.empty((0, window))
    n = 1 + (len(x) - window) // stride
    idx = np.arange(window)[None, :] + stride * np.arange(n)[:, None]
    return x[idx]


def fft_features(frames: np.ndarray) -> np.ndarray:
    W = frames.shape[1]
    w = frames - frames.mean(axis=1, keepdims=True)
    w = w * np.hanning(W)[None, :]
    mag = np.abs(np.fft.rfft(w, axis=1))[:, 1:W // 2 + 1]
    f = np.log1p(mag)
    f = (f - f.mean(axis=1, keepdims=True)) / (f.std(axis=1, keepdims=True) + 1e-8)
    return f.astype(np.float16)


def n_features(cfg: dict) -> int:
    d = cfg["data"]
    return d["n_order_bins"] if d.get("feature", "fft") == "order" else d["window"] // 2


def shaft_hz(cond_code: str) -> float:
    """Nominal shaft frequency from a PU condition code, e.g. N15_M07_F10 -> 1500 rpm -> 25 Hz."""
    return int(cond_code.split("_")[0][1:]) * 100 / 60.0


def order_features(frames: np.ndarray, fs: float, f_shaft: float, max_order: float, n_bins: int) -> np.ndarray:
    """Speed-normalised (order) log spectrum: the spectrum is resampled on a fixed grid of shaft orders,
    so bearing fault lines (BPFO ~3.1, BPFI ~4.9 orders and harmonics) sit at the same input index at every
    speed. Uses the operating point the drive already knows (shaft speed)."""
    W = frames.shape[1]
    w = frames - frames.mean(axis=1, keepdims=True)
    w = w * np.hanning(W)[None, :]
    mag = np.abs(np.fft.rfft(w, axis=1))
    freqs = np.fft.rfftfreq(W, 1.0 / fs)
    fq = np.linspace(max_order / n_bins, max_order, n_bins) * f_shaft
    if fq[-1] > freqs[-1]:
        raise ValueError(f"max_order {max_order} x {f_shaft:.1f} Hz exceeds Nyquist {freqs[-1]:.0f} Hz")
    f = np.log1p(np.stack([np.interp(fq, freqs, m) for m in mag]))
    f = (f - f.mean(axis=1, keepdims=True)) / (f.std(axis=1, keepdims=True) + 1e-8)
    return f.astype(np.float16)


def cache_tag(cfg: dict) -> str:
    feat = cfg["data"].get("feature", "fft")
    return cfg["data"]["signal"] if feat == "fft" else f"{cfg['data']['signal']}_{feat}"


def cache_file(cfg: dict, bearing: str, cond_key: str) -> Path:
    return path(cfg, cfg["data"]["proc_dir"]) / cache_tag(cfg) / bearing / f"{cond_key}.npz"


def envelope_signal(x: np.ndarray, fs: float, highpass: float, decim: int) -> np.ndarray:
    """Classic bearing envelope analysis: high-pass (keep structural resonances excited by impacts),
    Hilbert envelope, remove mean, low-pass + decimate. Fault repetition rates (BPFO/BPFI and harmonics,
    tens to hundreds of Hz) then appear as lines in the spectrum of the returned signal."""
    sos = scipy.signal.butter(4, highpass, btype="highpass", fs=fs, output="sos")
    env = np.abs(scipy.signal.hilbert(scipy.signal.sosfiltfilt(sos, x)))
    env = env - env.mean()
    while decim > 1:  # decimate in stages of <= 8 for numerical stability
        q = 8 if decim % 8 == 0 else 4 if decim % 4 == 0 else 2 if decim % 2 == 0 else decim
        env = scipy.signal.decimate(env, q, ftype="fir", zero_phase=True)
        decim //= q
    return env


def process_one(cfg: dict, bearing: str, cond_key: str, force: bool = False) -> str:
    out = cache_file(cfg, bearing, cond_key)
    if out.exists() and not force:
        return f"skip {bearing}/{cond_key} (cached)"
    d = cfg["data"]
    raw_dir = path(cfg, d["raw_dir"])
    cond_code = cfg["conditions"][cond_key]
    Xs, fids, widx, missing = [], [], [], []
    for i in range(1, d["n_files"] + 1):
        p = find_mat(raw_dir, bearing, cond_code, i)
        if p is None:
            missing.append(i)
            continue
        try:
            x = load_channel(p, CHANNEL[d["signal"]])
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"[WARN] {p.name}: {e}", file=sys.stderr)
            missing.append(i)
            continue
        feat = d.get("feature", "fft")
        if feat in ("envelope", "order"):
            x = envelope_signal(x, d["fs"], d.get("env_highpass", 1000.0), d.get("env_decimate", 16))
        elif d.get("decimate", 1) > 1:
            x = scipy.signal.decimate(x, d["decimate"], ftype="fir", zero_phase=True)
        fr = frame(x, d["window"], d["stride"])
        if feat == "order":
            Xs.append(order_features(fr, d["fs"] / d.get("env_decimate", 16), shaft_hz(cond_code),
                                     d.get("max_order", 32), d["n_order_bins"]))
        else:
            Xs.append(fft_features(fr))
        fids.append(np.full(len(fr), i, dtype=np.int16))
        widx.append(np.arange(len(fr), dtype=np.int16))
    if not Xs:
        return f"[ERROR] {bearing}/{cond_key}: no readable files (expected under {raw_dir/bearing})"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=np.concatenate(Xs), file_id=np.concatenate(fids), win_idx=np.concatenate(widx))
    msg = f"ok   {bearing}/{cond_key}: {sum(len(a) for a in fids)} windows from {len(Xs)} files"
    if missing:
        msg += f"  [WARN missing files {missing}]"
    return msg


def main():
    p = base_parser("Preprocess PU .mat files into cached FFT features")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    from .splits import all_bearings
    jobs = [(b, c) for b in all_bearings(cfg) for c in cfg["conditions"]]
    print(f"Preprocessing {len(jobs)} bearing x condition sets, signal={cfg['data']['signal']}, "
          f"feature={cfg['data'].get('feature', 'fft')}")
    errors = 0
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process_one, cfg, b, c, a.force) for b, c in jobs]
        for f in cf.as_completed(futs):
            msg = f.result()
            errors += msg.startswith("[ERROR]")
            print(msg, flush=True)
    if errors:
        print(f"\n{errors} sets failed - check download/extraction (see README, 'Data').")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
