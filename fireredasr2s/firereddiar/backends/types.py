# Copyright 2026 Xiaohongshu.
"""Shared types for diarization backends."""

from __future__ import annotations

from typing import NamedTuple


class DiarSpan(NamedTuple):
    """Single diarization interval in seconds."""

    start_s: float
    end_s: float
    speaker_id: int


def spans_to_legacy(
    spans: list[DiarSpan],
) -> list[tuple[float, float, int]]:
    return [(s.start_s, s.end_s, s.speaker_id) for s in spans]


def legacy_to_spans(
    legacy: list[tuple[float, float, int]],
) -> list[DiarSpan]:
    return [DiarSpan(t0, t1, int(spk)) for t0, t1, spk in legacy]
