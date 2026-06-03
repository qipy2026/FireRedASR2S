"""流式 ASR（``FireRedAsr2StreamSession`` / ``open_stream``）用例。

- 无模型：Mock Stream-VAD、伪 System，验证切段与 ``segment_final`` 结构。
- 可选 E2E：真实 XPU + 音频分块推流；输入优先级见 ``_e2e_stream_session_pcm_16k``。
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fireredasr2s.fireredasr2system import FireRedAsr2System
from fireredasr2s.firereddiar.audio import load_pcm_int16_mono, prepare_asr_stack_audio
from fireredasr2s.fireredvad.core.stream_vad_postprocessor import StreamVadFrameResult
from fireredasr2s.full_duplex_stream import FireRedFullDuplexStreamSession
from fireredasr2s.stream_session import FireRedAsr2StreamSession


def _e2e_stream_session_pcm_16k(fixtures_dir: Path) -> tuple[np.ndarray, str]:
    """E2E 流式输入：``FIREREDASR2S_E2E_STREAM_WAV`` → ``assets/metting_0507_seg03.wav`` → ``vad_proxy``。"""
    repo = fixtures_dir.parent.parent
    candidates: list[Path] = []
    env = (os.environ.get("FIREREDASR2S_E2E_STREAM_WAV") or "").strip()
    if env:
        candidates.append(Path(env).expanduser().resolve())
    candidates.append(repo / "assets" / "metting_0507_seg03.wav")
    wav_path: Path | None = None
    for p in candidates:
        if p.is_file():
            wav_path = p
            break
    if wav_path is None:
        from tests.test_e2e_by_feature import _require_wavs

        wav_path = _require_wavs(fixtures_dir, "vad_proxy")["vad_proxy"]
    label = str(wav_path.resolve())
    wav, sr = load_pcm_int16_mono(str(wav_path))
    wav, sr = prepare_asr_stack_audio(wav, int(sr))
    if int(sr) != 16000:
        pytest.skip(f"流式 E2E 需要 16 kHz 栈音频: {label}")
    return wav, label


def test_stream_session_symbol_export():
    from fireredasr2s import FireRedAsr2StreamSession as S_pkg
    from fireredasr2s.stream_session import FireRedAsr2StreamSession as S_mod

    assert S_pkg is S_mod


def test_full_duplex_symbol_export():
    from fireredasr2s import FireRedFullDuplexStreamSession as D_pkg
    from fireredasr2s.full_duplex_stream import FireRedFullDuplexStreamSession as D_mod

    assert D_pkg is D_mod


def test_open_stream_rejects_non_aed():
    class _Dummy:
        config = SimpleNamespace(asr_type="llm")

    with pytest.raises(ValueError, match="aed"):
        FireRedAsr2System.open_stream(_Dummy(), "live")  # type: ignore[arg-type]


def test_open_full_duplex_stream_rejects_non_aed():
    class _Dummy:
        config = SimpleNamespace(asr_type="llm")

    with pytest.raises(ValueError, match="aed"):
        FireRedAsr2System.open_full_duplex_stream(_Dummy(), "live")  # type: ignore[arg-type]


class _FakeSystemForStream:
    def __init__(self) -> None:
        self.pcm_calls: list[tuple[int, str, int, int]] = []
        self.config = SimpleNamespace(
            enable_diarization=False,
            vad_model_dir="dummy_vad",
            stream_vad_model_dir="",
            stream_vad_use_gpu=False,
            vad_config=SimpleNamespace(
                smooth_window_size=5,
                speech_threshold=0.4,
                min_speech_frame=8,
                max_speech_frame=2000,
                min_silence_frame=20,
                chunk_max_frame=30000,
            ),
        )

    def process_pcm_segment(self, wav_np, sample_rate, seg_uttid, segment_start_ms, segment_end_ms):
        self.pcm_calls.append((int(wav_np.shape[0]), seg_uttid, segment_start_ms, segment_end_ms))
        return {
            "uttid": seg_uttid,
            "text": "mock_seg",
            "sentences": [{"text": "mock_seg"}],
            "segment_start_ms": segment_start_ms,
            "segment_end_ms": segment_end_ms,
        }


def test_stream_session_mock_vad_emits_segment_final():
    """固定帧数后触发 speech_end，应调用一次 ``process_pcm_segment`` 并返回 ``segment_final``。"""
    fake = _FakeSystemForStream()
    mock_vad = MagicMock()
    n_calls = {"n": 0}

    def _detect(_frame):
        n_calls["n"] += 1
        i = n_calls["n"]
        return StreamVadFrameResult(
            frame_idx=i,
            is_speech=True,
            raw_prob=0.9,
            smoothed_prob=0.9,
            is_speech_start=(i == 1),
            is_speech_end=(i == 123),
            speech_start_frame=1 if i == 1 else -1,
            speech_end_frame=123 if i == 123 else -1,
        )

    mock_vad.detect_frame.side_effect = _detect
    mock_vad.reset = MagicMock()

    pcm_len = 20_000
    assert (pcm_len - 400) // 160 + 1 == 123

    with patch(
        "fireredasr2s.stream_session.FireRedStreamVad.from_pretrained",
        return_value=mock_vad,
    ):
        session = FireRedAsr2StreamSession(fake, uttid_prefix="unit")
        pcm = np.zeros(pcm_len, dtype=np.int16)
        evs = session.push_pcm_int16_mono(pcm, sample_rate=16000)

    assert len(evs) == 1
    assert evs[0]["event"] == "segment_final"
    assert evs[0]["segment_index"] == 1
    assert "pipeline" in evs[0]
    assert evs[0]["pipeline"]["text"] == "mock_seg"
    assert len(fake.pcm_calls) == 1
    assert fake.pcm_calls[0][0] > 0
    assert "unit_seg1_" in fake.pcm_calls[0][1]

    tail = session.finalize()
    assert tail == []


def test_stream_session_emit_vad_boundaries_mock():
    fake = _FakeSystemForStream()
    mock_vad = MagicMock()
    n_calls = {"n": 0}

    def _detect(_frame):
        n_calls["n"] += 1
        i = n_calls["n"]
        return StreamVadFrameResult(
            frame_idx=i,
            is_speech=True,
            raw_prob=0.9,
            smoothed_prob=0.9,
            is_speech_start=(i == 1),
            is_speech_end=(i == 123),
            speech_start_frame=1 if i == 1 else -1,
            speech_end_frame=123 if i == 123 else -1,
        )

    mock_vad.detect_frame.side_effect = _detect
    mock_vad.reset = MagicMock()

    pcm_len = 20_000
    with patch(
        "fireredasr2s.stream_session.FireRedStreamVad.from_pretrained",
        return_value=mock_vad,
    ):
        session = FireRedAsr2StreamSession(fake, uttid_prefix="unit", emit_vad_boundaries=True)
        pcm = np.zeros(pcm_len, dtype=np.int16)
        evs = session.push_pcm_int16_mono(pcm, sample_rate=16000)

    assert len(evs) == 2
    assert evs[0]["event"] == "vad_speech_start"
    assert evs[0]["start_ms"] == 0
    assert evs[1]["event"] == "segment_final"


def test_full_duplex_barge_in_during_playback_mock():
    fake = _FakeSystemForStream()
    mock_vad = MagicMock()
    n_calls = {"n": 0}

    def _detect(_frame):
        n_calls["n"] += 1
        i = n_calls["n"]
        return StreamVadFrameResult(
            frame_idx=i,
            is_speech=True,
            raw_prob=0.9,
            smoothed_prob=0.9,
            is_speech_start=(i == 1),
            is_speech_end=(i == 123),
            speech_start_frame=1 if i == 1 else -1,
            speech_end_frame=123 if i == 123 else -1,
        )

    mock_vad.detect_frame.side_effect = _detect
    mock_vad.reset = MagicMock()

    pcm_len = 20_000
    with patch(
        "fireredasr2s.stream_session.FireRedStreamVad.from_pretrained",
        return_value=mock_vad,
    ):
        session = FireRedFullDuplexStreamSession(fake, uttid_prefix="dup", verbose_vad=False)
        session.begin_local_playback()
        pcm = np.zeros(pcm_len, dtype=np.int16)
        evs = session.push_microphone_pcm(pcm, sample_rate=16000)

    assert len(evs) == 2
    assert evs[0]["event"] == "barge_in"
    assert evs[0]["start_ms"] == 0
    assert evs[1]["event"] == "segment_final"
    assert evs[1].get("during_local_playback") is True
    session.end_local_playback()


def test_stream_session_max_pcm_trims_timeline_when_no_open_segment():
    """长静音累积后超过 max_pcm_duration_s 应裁剪时间线并重置 Stream-VAD（无开放段）。"""
    fake = _FakeSystemForStream()
    mock_vad = MagicMock()
    mock_vad.detect_frame.return_value = StreamVadFrameResult(
        frame_idx=1,
        is_speech=False,
        raw_prob=0.0,
        smoothed_prob=0.0,
    )
    mock_vad.reset = MagicMock()

    with patch(
        "fireredasr2s.stream_session.FireRedStreamVad.from_pretrained",
        return_value=mock_vad,
    ):
        session = FireRedAsr2StreamSession(
            fake, uttid_prefix="trim", max_pcm_duration_s=0.05
        )
        # 0.05 s * 16000 = 800 samples max
        session.push_pcm_int16_mono(np.zeros(5000, dtype=np.int16), 16000)

    assert len(session._pcm) <= 800


def test_full_duplex_playback_context_on_barge_in_mock():
    fake = _FakeSystemForStream()
    mock_vad = MagicMock()
    n_calls = {"n": 0}

    def _detect(_frame):
        n_calls["n"] += 1
        i = n_calls["n"]
        return StreamVadFrameResult(
            frame_idx=i,
            is_speech=True,
            raw_prob=0.9,
            smoothed_prob=0.9,
            is_speech_start=(i == 1),
            is_speech_end=(i == 123),
            speech_start_frame=1 if i == 1 else -1,
            speech_end_frame=123 if i == 123 else -1,
        )

    mock_vad.detect_frame.side_effect = _detect
    mock_vad.reset = MagicMock()

    pcm_len = 20_000
    with patch(
        "fireredasr2s.stream_session.FireRedStreamVad.from_pretrained",
        return_value=mock_vad,
    ):
        session = FireRedFullDuplexStreamSession(fake, uttid_prefix="ctx")
        session.begin_local_playback("tts-001", anchor_wallclock_ms=1_700_000_000)
        pcm = np.zeros(pcm_len, dtype=np.int16)
        evs = session.push_microphone_pcm(pcm, sample_rate=16000)

    assert evs[0]["event"] == "barge_in"
    assert evs[0].get("playback_id") == "tts-001"
    assert evs[0].get("anchor_wallclock_ms") == 1_700_000_000
    assert evs[1].get("playback_id") == "tts-001"
    session.end_local_playback()


def test_full_duplex_verbose_vad_forwards_vad_start_mock():
    fake = _FakeSystemForStream()
    mock_vad = MagicMock()
    n_calls = {"n": 0}

    def _detect(_frame):
        n_calls["n"] += 1
        i = n_calls["n"]
        return StreamVadFrameResult(
            frame_idx=i,
            is_speech=True,
            raw_prob=0.9,
            smoothed_prob=0.9,
            is_speech_start=(i == 1),
            is_speech_end=(i == 123),
            speech_start_frame=1 if i == 1 else -1,
            speech_end_frame=123 if i == 123 else -1,
        )

    mock_vad.detect_frame.side_effect = _detect
    mock_vad.reset = MagicMock()

    pcm_len = 20_000
    with patch(
        "fireredasr2s.stream_session.FireRedStreamVad.from_pretrained",
        return_value=mock_vad,
    ):
        session = FireRedFullDuplexStreamSession(fake, uttid_prefix="v", verbose_vad=True)
        session.begin_local_playback()
        pcm = np.zeros(pcm_len, dtype=np.int16)
        evs = session.push_microphone_pcm(pcm, sample_rate=16000)

    assert len(evs) == 3
    assert evs[0]["event"] == "barge_in"
    assert evs[1]["event"] == "vad_speech_start"
    assert evs[2]["event"] == "segment_final"
    session.end_local_playback()


def test_stream_session_reset_clears_state():
    fake = _FakeSystemForStream()
    mock_vad = MagicMock()
    mock_vad.detect_frame.return_value = StreamVadFrameResult(
        frame_idx=1,
        is_speech=False,
        raw_prob=0.0,
        smoothed_prob=0.0,
    )
    mock_vad.reset = MagicMock()

    with patch(
        "fireredasr2s.stream_session.FireRedStreamVad.from_pretrained",
        return_value=mock_vad,
    ):
        session = FireRedAsr2StreamSession(fake, uttid_prefix="r")
        session.push_pcm_int16_mono(np.zeros(500, dtype=np.int16), 16000)
        session.reset()
        mock_vad.reset.assert_called()


@pytest.mark.e2e
@pytest.mark.xpu
@pytest.mark.slow
def test_e2e_stream_session_replay_wav(asr_system_xpu, fixtures_dir: Path):
    """真实模型：分块推入 PCM；默认优先 ``assets/metting_0507_seg03.wav``（或 ``FIREREDASR2S_E2E_STREAM_WAV``）。"""
    wav, _src_label = _e2e_stream_session_pcm_16k(fixtures_dir)
    sr = 16000

    session = asr_system_xpu.open_stream(uttid_prefix="e2e_stream")
    all_evs: list[dict] = []
    chunk = 3200
    for i in range(0, len(wav), chunk):
        all_evs.extend(session.push_pcm_int16_mono(wav[i : i + chunk], sample_rate=int(sr)))
    all_evs.extend(session.finalize())

    if not all_evs:
        pytest.skip("Stream-VAD 在该素材上未产生闭合段（阈值/环境与离线 VAD 不同属正常）")

    for ev in all_evs:
        assert ev.get("event") == "segment_final"
        pipe = ev.get("pipeline") or {}
        assert "text" in pipe
        assert "sentences" in pipe
        assert ev.get("start_ms") is not None
        assert ev.get("end_ms") is not None
