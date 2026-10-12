"""Evaluation helpers: top-k accuracy and confusion matrix."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def predict_all(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, targets = [], []
    for x, y in loader:
        logits.append(model(x).cpu())
        targets.append(torch.as_tensor(y))
    if not logits:
        return np.zeros((0, 1)), np.zeros((0,), dtype=int)
    return torch.cat(logits).numpy(), torch.cat(targets).numpy()


def topk_accuracy(logits: np.ndarray, targets: np.ndarray, k: int) -> float:
    if len(targets) == 0:
        return 0.0
    k = min(k, logits.shape[1])
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float((topk == targets[:, None]).any(axis=1).mean())


def confusion(logits: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    pred = logits.argmax(axis=1)
    for t, p in zip(targets, pred, strict=True):
        cm[t, p] += 1
    return cm


def evaluate(model, loader) -> dict:
    logits, targets = predict_all(model, loader)
    return {
        "top1": topk_accuracy(logits, targets, 1),
        "top5": topk_accuracy(logits, targets, 5),
        "n": int(len(targets)),
    }
