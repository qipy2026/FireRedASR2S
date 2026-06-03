"""T0 smoke tests: validate the test infrastructure itself.

These tests must pass on any machine that has installed ``requirements-dev.txt``
and run ``scripts/generate_test_fixtures.py``. Real-model tests live elsewhere
and are gated by the ``xpu`` / ``slow`` markers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.utils import estimate_snr, read_rttm, wer, write_rttm


def test_fixtures_present(fixtures_dir: Path):
    required = [
        "clean_zh_short.wav",
        "mixed_zh_en_short.wav",
        "noisy_short.wav",
        "e2e_vad_speech_proxy.wav",
        "dialog_2spk_30s.wav",
        "dialog_2spk_30s.rttm",
        "numbers_text_samples.json",
        "hotword_30sentences.json",
        "dialog_2spk_30s_speakers.json",
        "enroll_spkA_1.wav",
        "enroll_spkA_2.wav",
        "enroll_spkB_1.wav",
        "enroll_spkB_2.wav",
    ]
    missing = [n for n in required if not (fixtures_dir / n).exists()]
    assert not missing, (
        f"Missing fixtures: {missing}. "
        "Run: .venv/Scripts/python.exe scripts/generate_test_fixtures.py"
    )


def test_clean_audio_loads(fixtures_dir: Path):
    wav, sr = sf.read(str(fixtures_dir / "clean_zh_short.wav"), dtype="int16")
    assert sr == 16000
    assert wav.ndim == 1
    assert 2.5 * sr <= len(wav) <= 3.5 * sr


def test_wer_basic_metrics():
    assert wer("我有三百二十块钱", "我有三百二十块钱") == 0.0
    assert wer("a b c d", "a b c d") == 0.0
    assert 0.0 < wer("我有三百二十块钱", "我有320块钱") < 1.0


def test_estimate_snr_reasonable(fixtures_dir: Path):
    clean, _ = sf.read(str(fixtures_dir / "clean_zh_short.wav"), dtype="int16")
    noisy, _ = sf.read(str(fixtures_dir / "noisy_short.wav"), dtype="int16")
    snr = estimate_snr(clean.astype(np.float32), noisy.astype(np.float32))
    assert 1.0 <= snr <= 10.0, f"expected SNR ~5dB, got {snr:.2f}dB"


def test_rttm_roundtrip(tmp_path: Path):
    spans = [(0.0, 6.0, "spkA"), (6.0, 12.0, "spkB"), (12.0, 18.0, "spkA")]
    p = tmp_path / "x.rttm"
    write_rttm(str(p), spans, file_id="x")
    got = read_rttm(str(p))
    assert got == spans


def test_dialog_rttm_aligns_with_audio(fixtures_dir: Path):
    wav, sr = sf.read(str(fixtures_dir / "dialog_2spk_30s.wav"), dtype="int16")
    spans = read_rttm(str(fixtures_dir / "dialog_2spk_30s.rttm"))
    total = max(e for _, e, _ in spans)
    assert math.isclose(len(wav) / sr, total, rel_tol=0.0, abs_tol=0.05)
    speakers = {spk for _, _, spk in spans}
    assert speakers == {"spkA", "spkB"}


def test_numbers_samples_schema(fixtures_dir: Path):
    data = json.loads((fixtures_dir / "numbers_text_samples.json").read_text(encoding="utf-8"))
    for section in ("chinese_numbers", "decimals_and_units", "idempotent"):
        assert section in data and isinstance(data[section], list) and data[section]
        for item in data[section]:
            assert "in" in item and "out" in item


def test_record_metric_works(metric, request: pytest.FixtureRequest):
    metric("smoke_dummy", 0.123)
    props = dict(request.node.user_properties)
    assert props.get("metric_smoke_dummy") == 0.123


@pytest.mark.xpu
def test_xpu_runtime_is_real():
    """Hard-asserts XPU is alive when the marker actually runs (i.e. not skipped)."""
    from fireredasr2s.torch_device import xpu_runtime_available

    assert xpu_runtime_available()
