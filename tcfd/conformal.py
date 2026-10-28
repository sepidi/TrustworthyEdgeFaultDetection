"""Split conformal classification (LAC and APS scores) incl. Mondrian (group-conditional) calibration.

LAC: s(x,y) = 1 - p_y                       (Sadinle, Lei & Wasserman, JASA 2019)
APS: s(x,y) = sum of p_j ranked at or above y (non-randomised; Romano, Sesia & Candes, NeurIPS 2020)
Set:  C(x) = {y : s(x,y) <= q},  q = ceil((n+1)(1-alpha))-th smallest calibration score.
Mondrian: one q per group (here: drive operating condition), giving group-conditional coverage
(Vovk et al. 2005) under within-group exchangeability.
"""
from __future__ import annotations

import math

import numpy as np


def class_scores(probs: np.ndarray, kind: str) -> np.ndarray:
    """Score of every candidate label, shape (n, K)."""
    if kind == "lac":
        return 1.0 - probs
    if kind == "aps":
        order = np.argsort(-probs, axis=1, kind="stable")
        sorted_p = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        out = np.empty_like(probs)
        np.put_along_axis(out, order, cum, axis=1)
        return out
    raise ValueError(kind)


def true_scores(probs: np.ndarray, y: np.ndarray, kind: str) -> np.ndarray:
    return class_scores(probs, kind)[np.arange(len(y)), y]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def predict_sets(probs: np.ndarray, q: float, kind: str) -> np.ndarray:
    return class_scores(probs, kind) <= q + 1e-12


def set_metrics(sets: np.ndarray, y: np.ndarray) -> dict:
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(axis=1)
    per_class = [covered[y == c].mean() for c in np.unique(y)]
    return {"coverage": float(covered.mean()), "set_size": float(size.mean()),
            "singleton": float((size == 1).mean()), "empty": float((size == 0).mean()),
            "min_class_cov": float(np.min(per_class)) if per_class else float("nan")}
