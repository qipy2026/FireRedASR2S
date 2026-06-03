"""Audio helpers for tests (SNR estimation, etc.)."""

from __future__ import annotations

import math

import numpy as np


def estimate_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Return SNR in dB given parallel ``clean`` and ``noisy = clean + noise``."""
    clean = clean.astype(np.float64)
    noisy = noisy.astype(np.float64)
    noise = noisy[: len(clean)] - clean[: len(noisy)]
    p_signal = float(np.mean(clean[: len(noise)] ** 2))
    p_noise = float(np.mean(noise ** 2))
    if p_noise <= 0:
        return float("inf")
    if p_signal <= 0:
        return float("-inf")
    return 10.0 * math.log10(p_signal / p_noise)


__all__ = ["estimate_snr"]
