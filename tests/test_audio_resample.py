"""Mono + 16 kHz normalization for production stacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fireredasr2s.firereddiar.audio import (
    load_pcm_int16_mono,
    prepare_asr_stack_audio,
    resample_int16_pcm,
    to_mono_int16,
)


def test_to_mono_int16_stereo_mean():
    stereo = np.array([[1000, -1000], [2000, 0]], dtype=np.int16)
    m = to_mono_int16(stereo)
    assert m.shape == (2,)
    assert m.dtype == np.int16


def test_prepare_16k_noop():
    x = np.ones(800, dtype=np.int16)
    y, sr = prepare_asr_stack_audio(x, 16000)
    assert sr == 16000
    assert np.array_equal(y, x)


def test_load_pcm_int16_mono_from_fixture_wav(fixtures_dir):
    p = fixtures_dir / "clean_zh_short.wav"
    x, sr = load_pcm_int16_mono(p)
    assert x.ndim == 1 and x.size > 0 and sr > 0


def test_metting_0507_mp3_decode_when_file_present():
    repo = Path(__file__).resolve().parent.parent
    mp3 = repo / "assets" / "metting_0507.mp3"
    if not mp3.is_file():
        pytest.skip("未找到 assets/metting_0507.mp3，放入该文件后重跑本用例")
    x, sr = load_pcm_int16_mono(mp3)
    y, sr16 = prepare_asr_stack_audio(x, sr)
    assert sr16 == 16000 and y.size > 0


def test_resample_48k_to_16k_length_approx():
    sr_in = 48000
    n = int(sr_in * 0.25)
    t = np.arange(n, dtype=np.float64)
    x = (np.sin(2.0 * np.pi * 180.0 * t / sr_in) * 12000.0).astype(np.int16)
    y = resample_int16_pcm(x, sr_in, 16000)
    expected = int(round(n * 16000 / sr_in))
    assert abs(y.shape[0] - expected) <= 8
