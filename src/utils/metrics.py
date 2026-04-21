"""BirdCLEF+ 2026 — Metrics utilities."""

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    """Compute macro-averaged ROC-AUC, skipping non-evaluable classes."""
    aucs = []
    for i in range(labels.shape[1]):
        col = labels[:, i]
        if col.sum() > 0 and col.sum() < len(col):
            try:
                aucs.append(roc_auc_score(col, preds[:, i]))
            except ValueError:
                pass
    return np.mean(aucs) if aucs else 0.0


def per_species_auc(labels: np.ndarray, preds: np.ndarray,
                    species_names: list[str]) -> dict:
    """Compute per-species AUC with species names."""
    results = {}
    for i, sp in enumerate(species_names):
        col = labels[:, i]
        if col.sum() > 0 and col.sum() < len(col):
            try:
                results[sp] = roc_auc_score(col, preds[:, i])
            except ValueError:
                results[sp] = None
        else:
            results[sp] = None
    return results
