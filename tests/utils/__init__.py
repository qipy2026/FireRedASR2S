"""Test utilities: WER / DER / SNR helpers.

These avoid heavy deps where possible; ``jiwer`` / ``pyannote.metrics`` are
imported lazily and failures bubble up as ``ImportError`` for the caller to
``pytest.skip`` on.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from .audio import estimate_snr

# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------


_PUNC_RE = re.compile(r"[，。！？、；：,.\!\?;:\u3000\s]+")


def _normalize_zh(text: str) -> str:
    text = text.strip().lower()
    return _PUNC_RE.sub(" ", text).strip()


def _tokens_zh(text: str) -> list[str]:
    """Char-level tokens for CN, whitespace-aware for ASCII letters/digits."""
    text = _normalize_zh(text)
    out: list[str] = []
    buf: list[str] = []
    for ch in text:
        if ch == " ":
            if buf:
                out.append("".join(buf))
                buf = []
            continue
        if ord(ch) < 128 and ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
            out.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def wer(reference: str, hypothesis: str) -> float:
    """Word/char level WER. Char-tokenized for CN; falls back to whitespace for EN."""
    ref = _tokens_zh(reference)
    hyp = _tokens_zh(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n


# ---------------------------------------------------------------------------
# RTTM
# ---------------------------------------------------------------------------


def write_rttm(path: str, spans: Iterable[tuple[float, float, str]], file_id: str = "audio") -> None:
    """Write minimal RTTM rows: SPEAKER <file> 1 <start> <dur> <NA> <NA> <spk> <NA> <NA>."""
    with open(path, "w", encoding="utf-8") as f:
        for start, end, spk in spans:
            dur = max(0.0, float(end) - float(start))
            f.write(
                f"SPEAKER {file_id} 1 {float(start):.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n"
            )


def read_rttm(path: str) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            spk = parts[7]
            out.append((start, start + dur, spk))
    return out


def der(reference_rttm: str, hypothesis_rttm: str, file_id: str = "audio") -> float:
    """Diarization Error Rate via pyannote.metrics. Raises ImportError if missing."""
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def _to_annotation(spans: list[tuple[float, float, str]]) -> Annotation:
        ann = Annotation(uri=file_id)
        for s, e, spk in spans:
            if e > s:
                ann[Segment(s, e)] = spk
        return ann

    ref = _to_annotation(read_rttm(reference_rttm))
    hyp = _to_annotation(read_rttm(hypothesis_rttm))
    metric = DiarizationErrorRate()
    return float(metric(ref, hyp))
