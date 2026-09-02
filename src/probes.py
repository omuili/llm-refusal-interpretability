from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_probe(x, y, c: float = 1.0, max_iter: int = 2000):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=max_iter, solver="liblinear", random_state=0),
    )
    model.fit(x, y)
    return model


def probe_auc(model, x, y) -> float:
    scores = model.predict_proba(x)[:, 1]
    return float(roc_auc_score(y, scores))


def shuffled_label_null(x_train, y_train, x_test, y_test, c: float, max_iter: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        shuffled = rng.permutation(y_train)
        model = fit_probe(x_train, shuffled, c=c, max_iter=max_iter)
        values.append(probe_auc(model, x_test, y_test))
    return np.asarray(values)
