from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


EPS = 1e-12


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = np.clip(arr, EPS, None)
    return arr / arr.sum(axis=1, keepdims=True)


def probability_temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature scale probabilities without requiring original logits."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    arr = normalize_probabilities(probs)
    logits = np.log(np.clip(arr, EPS, 1.0))
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def multiclass_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    arr = normalize_probabilities(probs)
    y = np.asarray(labels, dtype=int)
    if arr.shape[0] != y.shape[0]:
        raise ValueError("probs and labels length mismatch")
    return float(-np.mean(np.log(np.clip(arr[np.arange(y.shape[0]), y], EPS, 1.0))))


def fit_temperature(probs: np.ndarray, labels: np.ndarray, bounds: tuple[float, float] = (0.25, 8.0)) -> float:
    """Fit a scalar temperature by minimizing validation NLL."""

    def objective(temp: float) -> float:
        return multiclass_nll(probability_temperature_scale(probs, temp), labels)

    result = minimize_scalar(objective, bounds=bounds, method="bounded")
    if not result.success:
        return 1.0
    return float(result.x)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    arr = normalize_probabilities(probs)
    y = np.asarray(labels, dtype=int)
    confidences = arr.max(axis=1)
    predictions = arr.argmax(axis=1)
    correct = (predictions == y).astype(float)
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > low) & (confidences <= high)
        if not np.any(mask):
            continue
        weight = mask.mean()
        ece += weight * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))
    return float(ece)


