# Copyright 2026 Xiaohongshu.
"""Lightweight mono NLMS adaptive filter for playback echo suppression (software AEC-lite).

This is **not** a substitute for production WebRTC/OS AEC: performance depends on filter length,
playback–mic delay alignment, and room acoustics. Use **headphones** when quality matters.

Reference ``ref`` should be the **same** signal (or aligned delayed copy) fed to the speaker;
``mic`` is the microphone capture in the same sample rate, float32 roughly in ``[-1, 1]``.
"""

from __future__ import annotations

import numpy as np


class NlmsMonoAec:
    """Sample-wise normalized LMS adaptive echo canceller (mono)."""

    def __init__(
        self,
        filter_len: int = 2048,
        mu: float = 0.25,
        eps: float = 1e-3,
        ref_delay_samples: int = 0,
    ) -> None:
        if filter_len < 32:
            raise ValueError("filter_len should be at least 32")
        self.L = int(filter_len)
        self.mu = float(mu)
        self.eps = float(eps)
        self.ref_delay_samples = max(0, int(ref_delay_samples))
        self._w = np.zeros(self.L, dtype=np.float64)
        self._x = np.zeros(self.L, dtype=np.float64)
        self._ref_ring: list[float] = (
            [0.0] * self.ref_delay_samples if self.ref_delay_samples > 0 else []
        )

    def reset(self) -> None:
        self._w.fill(0.0)
        self._x.fill(0.0)
        self._ref_ring = (
            [0.0] * self.ref_delay_samples if self.ref_delay_samples > 0 else []
        )

    def process_block(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Return near-end (echo-reduced) signal, same shape as ``mic``."""
        mic = np.asarray(mic, dtype=np.float64).reshape(-1)
        ref = np.asarray(ref, dtype=np.float64).reshape(-1)
        if mic.shape != ref.shape:
            raise ValueError("mic and ref must have the same shape")
        out = np.empty_like(mic)
        for i in range(mic.size):
            r = float(ref[i])
            if self._ref_ring:
                self._ref_ring.append(r)
                r = float(self._ref_ring.pop(0))
            self._x = np.roll(self._x, -1)
            self._x[-1] = r
            y = float(np.dot(self._w, self._x))
            e = mic[i] - y
            out[i] = e
            denom = float(np.dot(self._x, self._x)) + self.eps
            self._w += (self.mu * e / denom) * self._x
        return out.astype(np.float32)
