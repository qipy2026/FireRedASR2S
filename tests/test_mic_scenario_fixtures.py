"""麦克风场景代理素材（近讲 + 轻噪）与栈音频约定 — 无模型。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fireredasr2s.firereddiar.audio import prepare_asr_stack_audio


def test_noisy_proxy_mixed_int16_prepares_to_16k_mono(fixtures_dir: Path) -> None:
    p = fixtures_dir / "clean_zh_short.wav"
    if not p.is_file():
        pytest.skip("run scripts/generate_test_fixtures.py for clean_zh_short.wav")
    pcm, sr = sf.read(str(p), dtype="int16")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1).astype(np.int16)
    rng = np.random.default_rng(0)
    noise = rng.integers(-2000, 2000, size=pcm.shape, dtype=np.int32)
    mix = np.clip(pcm.astype(np.int32) + noise, -32768, 32767).astype(np.int16)
    out, sr2 = prepare_asr_stack_audio(mix, int(sr))
    assert int(sr2) == 16000
    assert out.dtype == np.int16
    assert out.ndim == 1
