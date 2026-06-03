# Copyright 2026 Xiaohongshu.
"""Online streaming ASR session on top of ``FireRedStreamVad`` + ``FireRedAsr2System.process_pcm_segment``."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from fireredasr2s.fireredvad import FireRedStreamVad, FireRedStreamVadConfig
from fireredasr2s.fireredvad.core.constants import FRAME_LENGTH_SAMPLE, FRAME_SHIFT_SAMPLE
from fireredasr2s.firereddiar.audio import prepare_asr_stack_audio

if TYPE_CHECKING:
    from fireredasr2s.fireredasr2system import FireRedAsr2System

logger = logging.getLogger("fireredasr2s.stream_session")


class FireRedAsr2StreamSession:
    """Push 16 kHz mono PCM chunks; receive ``segment_final`` events when Stream-VAD ends a speech span.

    Holds the full PCM timeline in memory for the session (MVP). Call ``reset()`` between calls.
    """

    def __init__(
        self,
        system: FireRedAsr2System,
        uttid_prefix: str = "live",
        *,
        emit_vad_boundaries: bool = False,
        max_pcm_duration_s: float | None = None,
        telemetry: bool = False,
    ) -> None:
        self._system = system
        self._utt_prefix = (uttid_prefix or "live").strip() or "live"
        self._emit_vad_boundaries = bool(emit_vad_boundaries)
        self._telemetry = bool(telemetry)
        if max_pcm_duration_s is not None and float(max_pcm_duration_s) <= 0:
            raise ValueError("max_pcm_duration_s must be positive when set")
        self._max_pcm_samples: int | None = (
            int(float(max_pcm_duration_s) * 16000.0) if max_pcm_duration_s is not None else None
        )
        cfg = system.config
        if cfg.enable_diarization:
            logger.warning("Streaming session: diarization is not applied online; use offline ``process`` for diar.")
        sv_dir = (cfg.stream_vad_model_dir or "").strip() or cfg.vad_model_dir
        vcfg = cfg.vad_config
        sv_cfg = FireRedStreamVadConfig(
            use_gpu=bool(cfg.stream_vad_use_gpu),
            smooth_window_size=vcfg.smooth_window_size,
            speech_threshold=vcfg.speech_threshold,
            pad_start_frame=5,
            min_speech_frame=max(1, int(vcfg.min_speech_frame)),
            max_speech_frame=vcfg.max_speech_frame,
            min_silence_frame=vcfg.min_silence_frame,
            chunk_max_frame=vcfg.chunk_max_frame,
        )
        self._stream_vad = FireRedStreamVad.from_pretrained(sv_dir, sv_cfg)
        self._pcm = np.zeros(0, dtype=np.int16)
        self._next_frame_j = 0
        self._seg_start_j: int | None = None
        self._segment_count = 0
        self._stream_vad.reset()

    def reset(self) -> None:
        self._pcm = np.zeros(0, dtype=np.int16)
        self._next_frame_j = 0
        self._seg_start_j = None
        self._segment_count = 0
        self._stream_vad.reset()

    @property
    def emit_vad_boundaries(self) -> bool:
        return self._emit_vad_boundaries

    def push_pcm_int16_mono(self, chunk: np.ndarray, sample_rate: int = 16000) -> list[dict[str, Any]]:
        """Append PCM and run online VAD + ASR on completed speech segments. Returns new ``segment_final`` dicts."""
        chunk = np.asarray(chunk, dtype=np.int16)
        if chunk.ndim != 1:
            raise ValueError("push_pcm_int16_mono expects 1-D int16 mono PCM")
        wav, sr = prepare_asr_stack_audio(chunk, int(sample_rate))
        if int(sr) != 16000:
            raise ValueError(f"streaming expects 16 kHz after prepare_asr_stack_audio, got {sr}")
        self._pcm = np.concatenate([self._pcm, wav])
        self._maybe_trim_pcm_timeline()
        return self._drain_vad_frames()

    def finalize(self) -> list[dict[str, Any]]:
        """Flush an open speech span at end-of-stream (if any)."""
        out: list[dict[str, Any]] = []
        if self._seg_start_j is not None:
            end_j = len(self._pcm)
            if end_j > self._seg_start_j:
                ev = self._emit_segment(self._seg_start_j, end_j)
                if ev is not None:
                    out.append(ev)
            self._seg_start_j = None
        return out

    def _maybe_trim_pcm_timeline(self) -> None:
        """Drop oldest PCM when over ``max_pcm_duration_s`` (no open VAD segment).

        Resets Stream-VAD and frame cursor so only **recent** audio is interpreted; early
        timeline context is discarded. Use for long sessions to cap RAM.
        """
        if self._max_pcm_samples is None:
            return
        if len(self._pcm) <= self._max_pcm_samples:
            return
        if self._seg_start_j is not None:
            logger.warning(
                "stream_session: PCM length %d exceeds max %d samples with an open speech span; "
                "skipping trim until VAD closes the segment",
                len(self._pcm),
                self._max_pcm_samples,
            )
            return
        excess = len(self._pcm) - self._max_pcm_samples
        self._pcm = self._pcm[excess:].copy()
        self._next_frame_j = 0
        self._stream_vad.reset()

    def _drain_vad_frames(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while self._next_frame_j <= len(self._pcm) - FRAME_LENGTH_SAMPLE:
            j = self._next_frame_j
            frame = self._pcm[j : j + FRAME_LENGTH_SAMPLE]
            r = self._stream_vad.detect_frame(frame)
            self._next_frame_j += FRAME_SHIFT_SAMPLE
            if self._emit_vad_boundaries and r.is_speech_start:
                events.append(
                    {
                        "event": "vad_speech_start",
                        "sample_index": j,
                        "start_ms": int(round(j / 16.0)),
                    }
                )
            if r.is_speech_start:
                self._seg_start_j = j
            if r.is_speech_end and self._seg_start_j is not None:
                end_j = j + FRAME_LENGTH_SAMPLE
                if end_j > self._seg_start_j:
                    ev = self._emit_segment(self._seg_start_j, end_j)
                    if ev is not None:
                        events.append(ev)
                self._seg_start_j = None
        return events

    def _emit_segment(self, start_j: int, end_j: int) -> dict[str, Any] | None:
        seg = self._pcm[start_j:end_j].copy()
        if seg.size < int(0.02 * 16000):
            return None
        start_ms = int(round(start_j / 16.0))
        end_ms = int(round(end_j / 16.0))
        self._segment_count += 1
        seg_uttid = f"{self._utt_prefix}_seg{self._segment_count}_s{start_ms}_e{end_ms}"
        t0 = time.perf_counter()
        res = self._system.process_pcm_segment(
            seg, 16000, seg_uttid, segment_start_ms=start_ms, segment_end_ms=end_ms
        )
        if self._telemetry:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "fireredasr2s.stream.telemetry segment_infer_ms=%.2f uttid=%s start_ms=%d end_ms=%d",
                dt_ms,
                seg_uttid,
                start_ms,
                end_ms,
            )
        return {
            "event": "segment_final",
            "segment_index": self._segment_count,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "pipeline": res,
        }
