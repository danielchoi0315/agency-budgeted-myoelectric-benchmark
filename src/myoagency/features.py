from __future__ import annotations

import numpy as np


def window_signal(signal: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive")
    if arr.shape[0] < window_size:
        return np.empty((0, window_size, arr.shape[1]))
    starts = np.arange(0, arr.shape[0] - window_size + 1, step_size)
    return np.stack([arr[start : start + window_size] for start in starts], axis=0)


def time_domain_features(windows: np.ndarray) -> np.ndarray:
    """Classic myoelectric features: MAV, RMS, WL, ZC, SSC per channel."""
    x = np.asarray(windows, dtype=float)
    if x.ndim != 3:
        raise ValueError("windows must have shape (n_windows, window_size, n_channels)")
    mav = np.mean(np.abs(x), axis=1)
    rms = np.sqrt(np.mean(x**2, axis=1))
    wl = np.sum(np.abs(np.diff(x, axis=1)), axis=1)
    diff = np.diff(x, axis=1)
    zc = np.sum(np.diff(np.signbit(x), axis=1), axis=1)
    ssc = np.sum(np.diff(np.signbit(diff), axis=1), axis=1)
    return np.concatenate([mav, rms, wl, zc, ssc], axis=1)


def inject_channel_dropout(features: np.ndarray, dropout_rate: float, seed: int = 0) -> np.ndarray:
    if not 0 <= dropout_rate <= 1:
        raise ValueError("dropout_rate must be in [0, 1]")
    rng = np.random.default_rng(seed)
    arr = np.asarray(features, dtype=float).copy()
    mask = rng.random(arr.shape) < dropout_rate
    arr[mask] = 0.0
    return arr


