"""Config loading, CLI overrides and small shared helpers."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml


def _set_key(cfg: dict, dotted: str, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _set_key(cfg, k, yaml.safe_load(v))
    cfg = copy.deepcopy(cfg)
    cfg["_root"] = str(Path(path).resolve().parent)
    return cfg


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override config keys, e.g. train.epochs=5")
    p.add_argument("--folds", nargs="*", default=None, help="subset of folds (default: all)")
    p.add_argument("--archs", nargs="*", default=None, help="subset of archs (default: config)")
    return p


def path(cfg: dict, rel: str) -> Path:
    """Resolve a config path relative to the repo root (where config.yaml lives)."""
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(cfg: dict):
    import torch
    want = cfg["train"].get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def model_id(arch: str, ratio: float) -> str:
    return f"{arch}_p{int(round(ratio * 100)):02d}"


def all_ratios(cfg: dict) -> list[float]:
    return [0.0] + [float(r) for r in cfg["prune"]["ratios"]]


def folds(cfg: dict, subset=None) -> list[str]:
    fs = list(cfg["folds"].keys())
    return [f for f in fs if subset is None or f in subset]


def archs(cfg: dict, subset=None) -> list[str]:
    a = list(cfg["train"]["archs"])
    return [x for x in a if subset is None or x in subset]


def save_json(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
