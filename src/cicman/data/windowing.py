"""Resampling, normalization, and windowing utilities."""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal


def resample_signal(x: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if source_rate == target_rate:
        return x
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError(f"Invalid sample rates: source={source_rate}, target={target_rate}")
    # resample_poly is stable and deterministic for the integer-rate pairs used here.
    gcd = np.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    y = scipy_signal.resample_poly(x, up=up, down=down)
    return np.asarray(y, dtype=np.float32)


def robust_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < eps:
        scale = np.std(x)
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    return ((x - median) / (scale + eps)).astype(np.float32)


def make_windows(x: np.ndarray, window_length: int, hop_length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) < window_length:
        return np.empty((0, window_length), dtype=np.float32)
    starts = np.arange(0, len(x) - window_length + 1, hop_length)
    windows = np.stack([x[start : start + window_length] for start in starts], axis=0)
    return windows.astype(np.float32)


def estimate_window_count(num_samples: int, source_rate: int, target_rate: int, window_length: int, hop_length: int) -> int:
    resampled_len = int(round(num_samples * target_rate / source_rate))
    if resampled_len < window_length:
        return 0
    return (resampled_len - window_length) // hop_length + 1

