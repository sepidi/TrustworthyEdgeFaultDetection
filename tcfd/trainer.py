"""Training / evaluation loops shared by train.py and prune.py."""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def batches(X: np.ndarray, y: np.ndarray | None, bs: int, shuffle: bool, rng: np.random.Generator | None = None):
    idx = np.arange(len(X))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield X[j], (None if y is None else y[j])


@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray, device, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in batches(X, None, bs, False):
        out.append(model(torch.from_numpy(xb.astype(np.float32)).to(device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.empty((0,))


def nll_acc(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    lt = torch.from_numpy(logits)
    nll = F.cross_entropy(lt, torch.from_numpy(y)).item()
    acc = float((logits.argmax(1) == y).mean())
    return nll, acc


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


def fit(model: nn.Module, Xtr, ytr, Xva, yva, *, epochs: int, lr: float, wd: float, patience: int,
        bs: int, device, n_classes: int, seed: int, tag: str = "", label_smoothing: float = 0.0,
        min_epochs: int = 1) -> dict:
    """Train with AdamW + cosine LR, class-weighted CE, early stopping on val NLL.
    Model selection only considers epochs >= min_epochs: with few validation bearings the val NLL of an
    almost untrained network can be deceptively low (fold B, envelope: ResNet 'best' at epoch 1).
    Restores the best-val-NLL weights before returning."""
    rng = np.random.default_rng(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(len(Xtr) / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    crit = nn.CrossEntropyLoss(weight=class_weights(ytr, n_classes).to(device), label_smoothing=label_smoothing)
    best, best_state, bad, hist, best_ep = float("inf"), copy.deepcopy(model.state_dict()), 0, [], 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for xb, yb in batches(Xtr, ytr, bs, True, rng):
            xb = torch.from_numpy(xb.astype(np.float32)).to(device)
            yb = torch.from_numpy(yb).to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        if Xva is None:  # fixed-epoch training (no early stopping)
            hist.append({"epoch": ep, "train_loss": tot / max(n, 1), "sec": round(time.time() - t0, 1)})
            best_state, best_ep = copy.deepcopy(model.state_dict()), ep
            continue
        vnll, vacc = nll_acc(predict_logits(model, Xva, device), yva)
        hist.append({"epoch": ep, "train_loss": tot / max(n, 1), "val_nll": vnll, "val_acc": vacc,
                     "sec": round(time.time() - t0, 1)})
        print(f"  {tag} ep{ep:03d} loss={tot / max(n, 1):.4f} val_nll={vnll:.4f} val_acc={vacc:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)
        if ep < min_epochs:
            continue
        if vnll < best - 1e-4:
            best, best_state, bad, best_ep = vnll, copy.deepcopy(model.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= patience:
                print(f"  {tag} early stop at epoch {ep}")
                break
    if best_ep == 0:  # never reached min_epochs (epochs < min_epochs): keep the last state
        best_state, best_ep = copy.deepcopy(model.state_dict()), len(hist)
    model.load_state_dict(best_state)
    return {"history": hist, "best_val_nll": best, "best_epoch": best_ep}
