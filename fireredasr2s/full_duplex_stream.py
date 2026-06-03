# Copyright 2026 Xiaohongshu.
"""Full-duplex orchestration: streaming ASR + local-playback window + barge-in hints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from fireredasr2s.stream_session import FireRedAsr2StreamSession

if TYPE_CHECKING:
    from fireredasr2s.fireredasr2system import FireRedAsr2System


class FireRedFullDuplexStreamSession:
    """Wraps ``FireRedAsr2StreamSession`` with a **local playback** flag (e.g. TTS playing).

    When playback is active and Stream-VAD reports ``vad_speech_start``, this layer emits
    ``barge_in`` so the application can stop TTS and prioritize user speech.

    Not thread-safe: call ``push_microphone_pcm`` / ``begin_local_playback`` / ``end_local_playback``
    from a single thread unless you add your own locking.
    """

    def __init__(
        self,
        system: FireRedAsr2System,
        uttid_prefix: str = "live",
        *,
        verbose_vad: bool = False,
        max_pcm_duration_s: float | None = None,
        telemetry: bool = False,
    ) -> None:
        self._playback_stack: list[dict[str, Any]] = []
        self._verbose_vad = bool(verbose_vad)
        self._inner = FireRedAsr2StreamSession(
            system,
            uttid_prefix=uttid_prefix,
            emit_vad_boundaries=True,
            max_pcm_duration_s=max_pcm_duration_s,
            telemetry=telemetry,
        )

    @property
    def local_playback_active(self) -> bool:
        return len(self._playback_stack) > 0

    def begin_local_playback(
        self,
        playback_id: str | None = None,
        *,
        anchor_wallclock_ms: int | None = None,
    ) -> None:
        """Mark the start of local audio playback (TTS, prompt, ringtone, etc.). Supports nesting.

        ``playback_id`` / ``anchor_wallclock_ms`` are copied onto ``barge_in`` and ``segment_final``
        (when ``during_local_playback``) for correlation with an external TTS pipeline.
        """
        self._playback_stack.append(
            {
                "playback_id": (playback_id or "").strip(),
                "anchor_wallclock_ms": anchor_wallclock_ms,
            }
        )

    def end_local_playback(self) -> None:
        """Pair with ``begin_local_playback``; nested calls unwind in LIFO order."""
        if self._playback_stack:
            self._playback_stack.pop()

    def push_microphone_pcm(
        self, chunk: np.ndarray, sample_rate: int = 16000
    ) -> list[dict[str, Any]]:
        raw = self._inner.push_pcm_int16_mono(chunk, sample_rate=sample_rate)
        return self._postprocess(raw)

    def finalize(self) -> list[dict[str, Any]]:
        return self._postprocess(self._inner.finalize())

    def reset(self) -> None:
        self._inner.reset()
        self._playback_stack.clear()

    def _playback_context(self) -> dict[str, Any]:
        if not self._playback_stack:
            return {}
        return dict(self._playback_stack[-1])

    def _postprocess(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        playback = len(self._playback_stack) > 0
        ctx = self._playback_context()
        for ev in events:
            et = ev.get("event")
            if et == "vad_speech_start":
                if playback:
                    bi: dict[str, Any] = {
                        "event": "barge_in",
                        "start_ms": ev.get("start_ms"),
                        "sample_index": ev.get("sample_index"),
                    }
                    if ctx.get("playback_id"):
                        bi["playback_id"] = ctx["playback_id"]
                    if ctx.get("anchor_wallclock_ms") is not None:
                        bi["anchor_wallclock_ms"] = ctx["anchor_wallclock_ms"]
                    out.append(bi)
                if self._verbose_vad:
                    out.append(ev)
                continue
            if et == "segment_final" and playback:
                ev = dict(ev)
                ev["during_local_playback"] = True
                if ctx.get("playback_id"):
                    ev["playback_id"] = ctx["playback_id"]
                if ctx.get("anchor_wallclock_ms") is not None:
                    ev["anchor_wallclock_ms"] = ctx["anchor_wallclock_ms"]
            out.append(ev)
        return out
