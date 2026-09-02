from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def binary_labels(values, positive: str) -> np.ndarray:
    return np.asarray([1 if str(v) == positive else 0 for v in values], dtype=np.int64)


def mean_difference_direction(acts: np.ndarray, labels: np.ndarray):
    labels = np.asarray(labels, dtype=np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Direction discovery requires both classes.")
    mean_pos = acts[labels == 1].mean(axis=0)
    mean_neg = acts[labels == 0].mean(axis=0)
    delta = mean_pos - mean_neg
    norm = np.linalg.norm(delta)
    if norm <= 1e-12:
        raise ValueError("Mean-difference direction has near-zero norm.")
    direction = delta / norm
    pos_proj = acts[labels == 1] @ direction
    neg_proj = acts[labels == 0] @ direction
    center = float((pos_proj.mean() + neg_proj.mean()) / 2.0)
    gap = float(pos_proj.mean() - neg_proj.mean())
    return direction.astype(np.float32), center, gap


def projection_scores(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return np.asarray(acts, dtype=np.float32) @ np.asarray(direction, dtype=np.float32)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def bootstrap_auc_ci(labels, scores, samples: int = 1000, seed: int = 42, alpha: float = 0.05):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    values = []
    n = len(labels)
    for _ in range(samples):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        if np.unique(y).size < 2:
            continue
        values.append(roc_auc_score(y, scores[idx]))
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def random_direction_null(acts: np.ndarray, labels: np.ndarray, n: int, seed: int):
    rng = np.random.default_rng(seed)
    values = []
    dim = acts.shape[-1]
    for _ in range(n):
        direction = rng.normal(size=dim)
        direction /= np.linalg.norm(direction)
        values.append(auc(labels, projection_scores(acts, direction)))
    return np.asarray(values)
