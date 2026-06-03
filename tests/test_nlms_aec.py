# Copyright 2026 Xiaohongshu.

"""Synthetic tests for ``NlmsMonoAec`` (software echo path, not production AEC)."""

from __future__ import annotations

import numpy as np
import pytest

from fireredasr2s.duplex import NlmsMonoAec


def test_nlms_reduces_instantaneous_linear_echo() -> None:
    """Mic = scalar * ref (+ noise); late residual RMS should stay well below raw mic RMS."""
    rng = np.random.default_rng(42)
    aec = NlmsMonoAec(filter_len=128, mu=0.35, eps=1e-3, ref_delay_samples=0)
    late_err: list[float] = []
    late_mic: list[float] = []
    for t in range(12000):
        r = float(rng.normal(0.0, 0.25))
        m = 0.55 * r + float(rng.normal(0.0, 0.02))
        e = float(aec.process_block(np.array([m], dtype=np.float32), np.array([r], dtype=np.float32))[0])
        if t >= 10000:
            late_err.append(abs(e))
            late_mic.append(abs(m))
    assert np.sqrt(np.mean(np.square(late_err))) < 0.45 * np.sqrt(np.mean(np.square(late_mic)))


def test_nlms_ref_delay_aligns_echo() -> None:
    """Echo is ref delayed by D samples; ``ref_delay_samples=D`` aligns reference into taps."""
    rng = np.random.default_rng(7)
    D = 40
    block = 64
    n_blocks = 400
    aec = NlmsMonoAec(filter_len=256, mu=0.4, ref_delay_samples=D)
    total = n_blocks * block
    ref_all = rng.standard_normal(total).astype(np.float32) * 0.2
    mic_all = np.zeros(total, dtype=np.float32)
    for t in range(D, total):
        mic_all[t] = 0.6 * ref_all[t - D] + float(rng.normal(0.0, 0.015))

    early: list[float] = []
    late: list[float] = []
    late_mic: list[float] = []
    for b in range(n_blocks):
        sl = slice(b * block, (b + 1) * block)
        out = aec.process_block(mic_all[sl], ref_all[sl])
        if 8 <= b < 40:
            early.extend(np.abs(out).tolist())
        if b >= 320:
            late.extend(np.abs(out).tolist())
            late_mic.extend(np.abs(mic_all[sl]).tolist())
    assert np.sqrt(np.mean(np.square(late))) < 0.45 * np.sqrt(np.mean(np.square(late_mic)))
    assert np.sqrt(np.mean(np.square(late))) < np.sqrt(np.mean(np.square(early)))


def test_nlms_requires_matching_shapes() -> None:
    aec = NlmsMonoAec(filter_len=64)
    with pytest.raises(ValueError, match="same shape"):
        aec.process_block(np.zeros(10, dtype=np.float32), np.zeros(8, dtype=np.float32))


def test_nlms_mono_aec_package_export() -> None:
    from fireredasr2s import NlmsMonoAec as N_pkg
    from fireredasr2s.duplex import NlmsMonoAec as N_mod

    assert N_pkg is N_mod


def test_nlms_reset_clears_state() -> None:
    rng = np.random.default_rng(1)
    aec = NlmsMonoAec(filter_len=96, mu=0.5)
    for _ in range(500):
        r = rng.standard_normal(32).astype(np.float32) * 0.1
        m = 0.4 * r + rng.standard_normal(32).astype(np.float32) * 0.01
        aec.process_block(m, r)
    w_before = aec._w.copy()
    aec.reset()
    assert np.allclose(aec._w, 0.0)
    assert not np.allclose(w_before, 0.0)
