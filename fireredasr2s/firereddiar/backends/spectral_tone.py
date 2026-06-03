# Copyright 2026 Xiaohongshu.

"""Frame-level dual-tone energy diarization (no external models).

Designed for alternating single-frequency segments (e.g. synthetic dialog
fixtures at 220 Hz vs 330 Hz). Not a replacement for neural diarization on
natural speech — use ``modelscope_campplus`` or ``pyannote`` for that.
"""

from __future__ import annotations

import numpy as np


def _tone_frame_energy(frame: np.ndarray, sample_rate: int, f_hz: float) -> float:
    """Narrow-band energy via quadrature projection (pure tone discriminator)."""
    n = frame.shape[0]
    if n <= 2:
        return 0.0
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    c = np.cos(2.0 * np.pi * f_hz * t)
    s = np.sin(2.0 * np.pi * f_hz * t)
    x = frame.astype(np.float64)
    a = float(np.dot(x, c))
    b = float(np.dot(x, s))
    return float(np.hypot(a, b) / (n + 1e-8))


def run_spectral_tone_pair_diar(
    wav_np: np.ndarray,
    sample_rate: int,
    *,
    f0_hz: float = 220.0,
    f1_hz: float = 330.0,
    frame_ms: float = 40.0,
    hop_ms: float = 10.0,
    dominance_ratio: float = 1.12,
) -> list[tuple[float, float, int]] | None:
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    if wav_np.size < int(0.05 * sample_rate):
        return None
    x = wav_np.astype(np.float64) / 32768.0
    frame = max(1, int(frame_ms * sample_rate / 1000.0))
    hop = max(1, int(hop_ms * sample_rate / 1000.0))
    labels: list[tuple[float, int]] = []
    pos = 0
    prev_lab: int | None = None
    while pos + frame <= x.shape[0]:
        seg = x[pos : pos + frame]
        e0 = _tone_frame_energy(seg, sample_rate, f0_hz)
        e1 = _tone_frame_energy(seg, sample_rate, f1_hz)
        if prev_lab is not None:
            if prev_lab == 0 and e0 > e1 / dominance_ratio:
                lab = 0
            elif prev_lab == 1 and e1 > e0 / dominance_ratio:
                lab = 1
            else:
                lab = 0 if e0 > e1 * dominance_ratio else 1
        else:
            lab = 0 if e0 > e1 * dominance_ratio else 1
        center_t = (pos + frame / 2.0) / float(sample_rate)
        labels.append((center_t, lab))
        prev_lab = lab
        pos += hop

    if not labels:
        return None

    raw_spans: list[tuple[float, float, int]] = []
    t0_frame = 0.0
    cur = labels[0][1]
    for i in range(1, len(labels)):
        t_mid, lab = labels[i]
        if lab != cur:
            t_split = (labels[i - 1][0] + t_mid) / 2.0
            if t_split > t0_frame + 1e-4:
                raw_spans.append((t0_frame, t_split, cur))
            t0_frame = t_split
            cur = lab
    dur_s = float(x.shape[0]) / float(sample_rate)
    if dur_s > t0_frame + 1e-4:
        raw_spans.append((t0_frame, dur_s, cur))

    def _merge_consecutive(
        spans: list[tuple[float, float, int]],
    ) -> list[tuple[float, float, int]]:
        out: list[tuple[float, float, int]] = []
        for t0, t1, spk in spans:
            if t1 <= t0 + 1e-6:
                continue
            if out and out[-1][2] == spk:
                out[-1] = (out[-1][0], t1, spk)
            else:
                out.append((t0, t1, spk))
        return out

    merged = _merge_consecutive(raw_spans)

    seen: dict[int, int] = {}
    next_id = 0
    norm: list[tuple[float, float, int]] = []
    for t0, t1, spk in merged:
        if spk not in seen:
            seen[spk] = next_id
            next_id += 1
        norm.append((t0, t1, seen[spk]))

    if len({s[2] for s in norm}) < 2:
        return None
    return norm
