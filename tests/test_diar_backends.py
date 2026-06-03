"""T7: diarization backend factory (ModelScope default, pyannote optional, speakerlab stub)."""

from __future__ import annotations

import numpy as np
import pytest

from fireredasr2s.firereddiar.backends import run_diarization_backend
from fireredasr2s.firereddiar.backends.types import DiarSpan, legacy_to_spans, spans_to_legacy


def test_unknown_backend_raises():
    wav = np.zeros(16000, dtype=np.int16)
    with pytest.raises(ValueError, match="Unknown"):
        run_diarization_backend(
            "no_such_backend",
            wav,
            16000,
            [(0.0, 1.0)],
            model_id="x",
            model_revision=None,
        )


def test_speakerlab_raises_not_implemented():
    wav = np.zeros(16000, dtype=np.int16)
    with pytest.raises(NotImplementedError):
        run_diarization_backend(
            "speakerlab",
            wav,
            16000,
            [(0.0, 1.0)],
            model_id="x",
            model_revision=None,
        )


def test_modelscope_backend_short_audio_returns_none():
    """Short audio should not build diar input (same rule as build_diar_input_full)."""
    wav = np.zeros(8000, dtype=np.int16)
    out = run_diarization_backend(
        "modelscope_campplus",
        wav,
        16000,
        [(0.0, 0.5)],
        model_id="damo/speech_campplus_speaker-diarization_common",
        model_revision=None,
        diar_input_mode="full",
    )
    assert out is None


def test_pyannote_without_token_returns_none():
    wav = np.zeros(32000, dtype=np.int16)
    out = run_diarization_backend(
        "pyannote",
        wav,
        16000,
        [(0.0, 1.0)],
        model_id="ignored",
        model_revision=None,
        hf_token="",
    )
    assert out is None


def test_rttm_sidecar_loads_adjacent_rttm(tmp_path):
    wav_path = tmp_path / "clip.wav"
    wav_path.write_bytes(b"")
    rttm_path = tmp_path / "clip.rttm"
    rttm_path.write_text(
        "SPEAKER clip 1 0.000 6.000 <NA> <NA> spkA <NA> <NA>\n"
        "SPEAKER clip 1 6.000 6.000 <NA> <NA> spkB <NA> <NA>\n",
        encoding="utf-8",
    )
    wav = np.zeros(16000 * 12, dtype=np.int16)
    out = run_diarization_backend(
        "rttm_sidecar",
        wav,
        16000,
        [(0.0, 12.0)],
        model_id="ignored",
        model_revision=None,
        wav_path=str(wav_path),
    )
    assert out == [(0.0, 6.0, 0), (6.0, 12.0, 1)]


def test_rttm_sidecar_without_wav_path_returns_none():
    wav = np.zeros(16000, dtype=np.int16)
    out = run_diarization_backend(
        "rttm_sidecar",
        wav,
        16000,
        [(0.0, 1.0)],
        model_id="",
        model_revision=None,
        wav_path="",
    )
    assert out is None


def test_spectral_tone_pair_on_dialog_fixture(fixtures_dir):
    import soundfile as sf

    w, sr = sf.read(str(fixtures_dir / "dialog_2spk_30s.wav"), dtype="int16")
    out = run_diarization_backend(
        "spectral_tone_pair",
        w,
        sr,
        [(0.0, 30.0)],
        model_id="ignored",
        model_revision=None,
        spectral_f0_hz=220.0,
        spectral_f1_hz=330.0,
    )
    assert out is not None
    assert len({x[2] for x in out}) == 2
    assert len(out) >= 4


def test_diar_span_roundtrip():
    spans = [DiarSpan(0.0, 1.0, 2), DiarSpan(1.0, 2.0, 3)]
    leg = spans_to_legacy(spans)
    assert leg == [(0.0, 1.0, 2), (1.0, 2.0, 3)]
    assert legacy_to_spans(leg) == spans
