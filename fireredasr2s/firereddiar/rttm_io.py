# Copyright 2026 Xiaohongshu.

"""Load diarization spans from NIST RTTM (sidecar next to wav)."""

from __future__ import annotations


def read_rttm_label_spans(path: str) -> list[tuple[float, float, str]]:
    """Parse RTTM lines: SPEAKER <file> 1 <start> <dur> ... <spk_id> ..."""
    out: list[tuple[float, float, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            spk = parts[7]
            t1 = start + dur
            if t1 > start:
                out.append((start, t1, spk))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def rttm_spans_to_speaker_ids(
    labeled: list[tuple[float, float, str]],
) -> list[tuple[float, float, int]]:
    """Map string labels to contiguous int ids in order of first appearance."""
    mapping: dict[str, int] = {}
    next_id = 0
    legacy: list[tuple[float, float, int]] = []
    for t0, t1, lab in labeled:
        if lab not in mapping:
            mapping[lab] = next_id
            next_id += 1
        legacy.append((t0, t1, mapping[lab]))
    return legacy
