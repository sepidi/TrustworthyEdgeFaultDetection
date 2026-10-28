"""Classification + calibration metrics and temperature scaling (numpy only)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import balanced_accuracy_score, f1_score


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    oh = np.eye(probs.shape[1])[y]
    return float(((probs - oh) ** 2).sum(axis=1).mean())


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Top-label expected calibration error, equal-width bins (Naeini et al. 2015; Guo et al. 2017)."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def reliability(probs: np.ndarray, y: np.ndarray, bins: int = 15):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    c, a, n = [], [], []
    for b in range(bins):
        m = idx == b
        c.append(conf[m].mean() if m.any() else np.nan)
        a.append(correct[m].mean() if m.any() else np.nan)
        n.append(int(m.sum()))
    return np.array(c), np.array(a), np.array(n), edges


def classification(probs: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    pred = probs.argmax(axis=1)
    return {
        "acc": float((pred == y).mean()),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(n_classes)), zero_division=0)),
        "conf": float(probs.max(axis=1).mean()),
    }


def all_metrics(probs: np.ndarray, y: np.ndarray, n_classes: int, bins: int) -> dict:
    d = classification(probs, y, n_classes)
    d.update({"ece": ece(probs, y, bins), "nll": nll(probs, y), "brier": brier(probs, y)})
    return d


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Single-parameter temperature scaling (Guo et al. 2017): minimise NLL over log T."""
    def f(logt):
        return nll(softmax(logits, float(np.exp(logt))), y)
    r = minimize_scalar(f, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(r.x))
