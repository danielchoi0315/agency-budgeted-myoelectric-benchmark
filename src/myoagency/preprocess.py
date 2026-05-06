from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_emg(signal: np.ndarray, fs: float, low_hz: float = 20.0, high_hz: float = 450.0, order: int = 4) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    nyq = 0.5 * fs
    high = min(high_hz, nyq * 0.95)
    if low_hz >= high:
        raise ValueError("low_hz must be below high_hz after Nyquist adjustment")
    b, a = butter(order, [low_hz / nyq, high / nyq], btype="band")
    return filtfilt(b, a, arr, axis=0)


def notch_filter(signal: np.ndarray, fs: float, notch_hz: float = 60.0, quality: float = 30.0) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if notch_hz >= fs / 2:
        return arr
    b, a = iirnotch(notch_hz / (fs / 2), quality)
    return filtfilt(b, a, arr, axis=0)


def robust_zscore(features: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(features, dtype=float)
    median = np.median(arr, axis=0, keepdims=True)
    mad = np.median(np.abs(arr - median), axis=0, keepdims=True)
    return (arr - median) / (1.4826 * mad + eps)


