"""Bearing-wise splits. Leakage rule: a bearing is in exactly one of train/val/test per fold."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .data_pu import cache_file

SPLITS = ("train", "valfit", "valcal", "test")


def class_index(cfg: dict) -> dict[str, int]:
    return {c: i for i, c in enumerate(cfg["classes"])}


def all_bearings(cfg: dict) -> dict[str, str]:
    """Evaluation bearings (real damage) + bearings used ONLY for training (e.g. artificial damage)."""
    return {**cfg["bearings"], **(cfg.get("train_only_bearings") or {})}


def fold_bearings(cfg: dict, fold: str) -> dict[str, list[str]]:
    f = cfg["folds"][fold]
    test, val = list(f["test"]), list(f["val"])
    overlap = set(test) & set(val)
    if overlap:
        raise ValueError(f"Fold {fold}: bearings in both test and val: {overlap}")
    extra = list((cfg.get("train_only_bearings") or {}).keys())
    bad = set(extra) & (set(test) | set(val))
    if bad:
        raise ValueError(f"train_only_bearings must not be test/val bearings: {bad}")
    train = [b for b in cfg["bearings"] if b not in test and b not in val] + extra
    return {"train": train, "val": val, "test": test, "train_real": [b for b in train if b not in extra]}


@lru_cache(maxsize=256)
def _load(cache_path: str):
    z = np.load(cache_path)
    return z["X"], z["file_id"], z["win_idx"]


def get_split(cfg: dict, fold: str, split: str, cond_key: str, overlap: bool | None = None):
    """Return X (n,1,F) float16, y (n,) int64, bearing (n,) str, file_id (n,) int.

    overlap=None -> overlapping windows for 'train', non-overlapping otherwise.
    """
    if split not in SPLITS:
        raise ValueError(split)
    fb = fold_bearings(cfg, fold)
    bearings = fb["train"] if split == "train" else fb["val"] if split.startswith("val") else fb["test"]
    files = None
    if split == "valfit":
        files = set(cfg["valfit_files"])
    elif split == "valcal":
        files = set(cfg["valcal_files"])
    if overlap is None:
        overlap = split == "train"
    return get_bearings(cfg, bearings, cond_key, overlap, files)


def get_bearings(cfg: dict, bearings: list[str], cond_key: str, overlap: bool = False, files=None):
    """X (n,1,F) float16, y, bearing, file_id for an explicit list of bearings."""
    step = max(1, cfg["data"]["window"] // cfg["data"]["stride"])
    cidx = class_index(cfg)
    lab = all_bearings(cfg)
    Xs, ys, bs, fs = [], [], [], []
    for b in bearings:
        p = cache_file(cfg, b, cond_key)
        if not p.exists():
            raise FileNotFoundError(f"Missing cache {p}. Run: python -m tcfd.data_pu")
        X, fid, widx = _load(str(p))
        m = np.ones(len(X), dtype=bool)
        if files is not None:
            m &= np.isin(fid, list(files))
        if not overlap:
            m &= (widx % step) == 0
        Xs.append(X[m])
        ys.append(np.full(m.sum(), cidx[lab[b]], dtype=np.int64))
        bs.append(np.full(m.sum(), b))
        fs.append(fid[m])
    X = np.concatenate(Xs)[:, None, :]
    return X, np.concatenate(ys), np.concatenate(bs), np.concatenate(fs)
