from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def bootstrap_metric_summary(y_true, y_pred, n_bootstrap: int = 1000, seed: int = 42) -> dict[str, dict[str, float]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {key: [] for key in ["accuracy", "precision", "recall", "f1"]}
    n = len(y_true)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        scores = classification_metrics(y_true[indices], y_pred[indices])
        for key, value in scores.items():
            values[key].append(value)

    return {
        key: {
            "mean": float(np.mean(metric_values)),
            "std": float(np.std(metric_values, ddof=1)),
        }
        for key, metric_values in values.items()
    }
