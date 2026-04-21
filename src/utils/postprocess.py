"""
BirdCLEF+ 2026 — Post-Processing
==================================

Free improvements that don't require retraining:

1. Prediction clipping: Floor very low predictions to prevent extreme
   confident negatives from catastrophically hurting ROC-AUC when wrong.

2. Temperature scaling: Calibrate prediction confidence by fitting a
   single temperature parameter on OOF predictions. Sharper/softer
   probabilities can improve ranking quality.

3. Power transform: Apply p^alpha to sharpen or smooth predictions.

Usage:
    # Fit temperature on OOF predictions
    from src.utils.postprocess import fit_temperature, apply_postprocessing

    temperature = fit_temperature(oof_labels, oof_logits)

    # Apply at inference time
    probs = apply_postprocessing(raw_probs, clip_min=0.001, temperature=temperature)
"""

import numpy as np
from typing import Optional


def clip_predictions(
    probs: np.ndarray,
    clip_min: float = 0.001,
    clip_max: float = 0.999,
) -> np.ndarray:
    """Clip predictions to prevent extreme confidence.

    For ROC-AUC, an extreme confident negative (p ≈ 0.0) that's actually
    positive is heavily penalized. Clipping to a floor like 0.001
    provides a safety net with negligible cost when correct.

    Args:
        probs: Prediction probabilities, shape (N, C).
        clip_min: Minimum probability floor.
        clip_max: Maximum probability ceiling.

    Returns:
        Clipped probabilities.
    """
    return np.clip(probs, clip_min, clip_max)


def apply_temperature(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Apply temperature scaling to logits before sigmoid.

    temperature > 1.0 → softer (more uncertain) predictions
    temperature < 1.0 → sharper (more confident) predictions

    Args:
        logits: Raw model logits, shape (N, C).
        temperature: Temperature parameter.

    Returns:
        Calibrated probabilities after sigmoid.
    """
    scaled = logits / max(temperature, 1e-6)
    return 1.0 / (1.0 + np.exp(-scaled))


def fit_temperature(
    labels: np.ndarray,
    logits: np.ndarray,
    t_range: tuple[float, float] = (0.1, 5.0),
    n_steps: int = 200,
) -> float:
    """Fit optimal temperature on OOF predictions using grid search.

    Optimizes macro-averaged ROC-AUC by searching over temperature values.

    Args:
        labels: Ground truth labels, shape (N, C).
        logits: Raw model logits (pre-sigmoid), shape (N, C).
        t_range: (min_temp, max_temp) to search.
        n_steps: Number of grid points.

    Returns:
        Optimal temperature value.
    """
    from sklearn.metrics import roc_auc_score

    def macro_auc_at_temp(t):
        probs = apply_temperature(logits, temperature=t)
        aucs = []
        for i in range(labels.shape[1]):
            col = labels[:, i]
            if col.sum() > 0 and col.sum() < len(col):
                try:
                    aucs.append(roc_auc_score(col, probs[:, i]))
                except ValueError:
                    pass
        return np.mean(aucs) if aucs else 0.0

    temperatures = np.linspace(t_range[0], t_range[1], n_steps)
    best_t = 1.0
    best_auc = 0.0

    for t in temperatures:
        auc = macro_auc_at_temp(t)
        if auc > best_auc:
            best_auc = auc
            best_t = t

    print(f"  Temperature scaling: best_t={best_t:.3f}, AUC improvement: "
          f"{macro_auc_at_temp(1.0):.4f} → {best_auc:.4f}")

    return best_t


def power_transform(
    probs: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Apply power transform to sharpen or smooth predictions.

    alpha < 1.0 → boost low predictions (more uniform)
    alpha > 1.0 → suppress low predictions (more peaky)

    Args:
        probs: Prediction probabilities.
        alpha: Power exponent.

    Returns:
        Transformed probabilities.
    """
    return np.power(probs, alpha)


def apply_postprocessing(
    probs: np.ndarray,
    clip_min: float = 0.001,
    clip_max: float = 0.999,
    temperature: Optional[float] = None,
    logits: Optional[np.ndarray] = None,
    power_alpha: float = 1.0,
) -> np.ndarray:
    """Apply all post-processing steps.

    If temperature is set and logits are provided, applies temperature
    scaling to logits first, then clips. Otherwise clips probs directly.

    Args:
        probs: Sigmoid probabilities (used if no temperature/logits).
        clip_min: Minimum prediction floor.
        clip_max: Maximum prediction ceiling.
        temperature: Temperature scaling parameter.
        logits: Raw logits (needed for temperature scaling).
        power_alpha: Power transform exponent (1.0 = no change).

    Returns:
        Post-processed probabilities.
    """
    # Temperature scaling (requires logits)
    if temperature is not None and logits is not None:
        probs = apply_temperature(logits, temperature=temperature)

    # Power transform
    if power_alpha != 1.0:
        probs = power_transform(probs, alpha=power_alpha)

    # Clipping
    probs = clip_predictions(probs, clip_min=clip_min, clip_max=clip_max)

    return probs


def evaluate_postprocessing_grid(
    labels: np.ndarray,
    logits: np.ndarray,
) -> dict:
    """Evaluate a grid of post-processing configurations.

    Tests combinations of clipping, temperature, and power transforms
    to find the best configuration for the OOF data.

    Args:
        labels: Ground truth labels.
        logits: Raw model logits.

    Returns:
        Dict with best configuration and results table.
    """
    from sklearn.metrics import roc_auc_score

    def macro_auc(probs):
        aucs = []
        for i in range(labels.shape[1]):
            col = labels[:, i]
            if col.sum() > 0 and col.sum() < len(col):
                try:
                    aucs.append(roc_auc_score(col, probs[:, i]))
                except ValueError:
                    pass
        return np.mean(aucs) if aucs else 0.0

    results = []

    # Baseline: raw sigmoid
    raw_probs = 1.0 / (1.0 + np.exp(-logits))
    base_auc = macro_auc(raw_probs)
    results.append({"config": "raw sigmoid", "auc": base_auc})

    # Clipping only
    for clip_min in [0.0001, 0.001, 0.005, 0.01]:
        clipped = clip_predictions(raw_probs, clip_min=clip_min)
        auc = macro_auc(clipped)
        results.append({"config": f"clip_min={clip_min}", "auc": auc})

    # Temperature only
    for t in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        t_probs = apply_temperature(logits, temperature=t)
        auc = macro_auc(t_probs)
        results.append({"config": f"temp={t}", "auc": auc})

    # Temperature + clipping
    best_t = fit_temperature(labels, logits)
    t_probs = apply_temperature(logits, temperature=best_t)
    clipped = clip_predictions(t_probs, clip_min=0.001)
    auc = macro_auc(clipped)
    results.append({"config": f"temp={best_t:.2f}+clip=0.001", "auc": auc})

    # Power transform
    for alpha in [0.5, 0.7, 0.9, 1.0, 1.5, 2.0]:
        p_probs = power_transform(raw_probs, alpha=alpha)
        auc = macro_auc(p_probs)
        results.append({"config": f"power={alpha}", "auc": auc})

    # Sort by AUC
    results.sort(key=lambda x: x["auc"], reverse=True)

    best = results[0]
    print(f"\n  Best post-processing: {best['config']} → AUC={best['auc']:.4f} "
          f"(baseline: {base_auc:.4f}, delta: {best['auc'] - base_auc:+.4f})")

    return {"best": best, "all_results": results, "baseline_auc": base_auc}
