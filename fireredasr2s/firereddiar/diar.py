# Copyright 2026 Xiaohongshu.

"""Optional speaker diarization via ModelScope SegmentationClusteringPipeline."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ModelScope list-input path asserts total effective speech > 5 seconds
DIAR_MIN_EFFECTIVE_SECONDS = 5.0

_pipeline_cache: dict[tuple[str, str | None], Any] = {}


def clear_diarization_pipeline_cache() -> None:
    _pipeline_cache.clear()


def _get_pipeline(model: str, model_revision: str | None) -> Any:
    key = (model, model_revision)
    if key in _pipeline_cache:
        return _pipeline_cache[key]
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
    except ImportError as e:
        raise ImportError(
            "ModelScope diarization needs `modelscope` and transitive imports "
            "(`addict`, `datasets`/`pyarrow`, `simplejson`, `sortedcontainers`, "
            "`scikit-learn`, `hdbscan`, `umap-learn`, …). Try: "
            "pip install 'fireredasr2s[modelscope]' ; on Windows if importing "
            "`pyarrow` crashes, fix/reinstall pyarrow or run on Linux/WSL."
        ) from e
    kwargs: dict[str, Any] = {"task": Tasks.speaker_diarization, "model": model}
    if model_revision:
        kwargs["model_revision"] = model_revision
    pipe = pipeline(**kwargs)
    _pipeline_cache[key] = pipe
    return pipe


def total_vad_speech_seconds(vad_segments: list[tuple[float, float]]) -> float:
    return float(sum(max(0.0, e - s) for s, e in vad_segments))


def build_diar_input_from_vad(
    wav_np: np.ndarray,
    sample_rate: int,
    vad_segments: list[tuple[float, float]],
) -> list | None:
    """Build [[start_s, end_s, float32_mono], ...] for ModelScope diarization."""
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    if wav_np.dtype != np.int16:
        wav_np = wav_np.astype(np.int16, copy=False)

    tot = total_vad_speech_seconds(vad_segments)
    if tot <= DIAR_MIN_EFFECTIVE_SECONDS:
        logger.warning(
            "Diarization skipped: effective VAD speech %.2fs <= %.2fs (ModelScope requirement).",
            tot,
            DIAR_MIN_EFFECTIVE_SECONDS,
        )
        return None

    out: list = []
    for start_s, end_s in vad_segments:
        i0 = int(start_s * sample_rate)
        i1 = int(end_s * sample_rate)
        if i1 <= i0:
            continue
        seg = wav_np[i0:i1].astype(np.float32) / 32768.0
        seg = np.clip(seg, -1.0, 1.0)
        expected = i1 - i0
        if seg.shape[0] != expected:
            logger.warning("Diarization skip segment: length mismatch %s vs %s", seg.shape[0], expected)
            continue
        out.append([float(start_s), float(end_s), seg])

    if total_vad_speech_seconds([(x[0], x[1]) for x in out]) <= DIAR_MIN_EFFECTIVE_SECONDS:
        logger.warning("Diarization skipped: effective speech too short after slicing.")
        return None
    return out


def build_diar_input_full(wav_np: np.ndarray, sample_rate: int) -> list | None:
    """Single-list-item input: full utterance as ``[0.0, dur, float32_mono]``.

    Lets ModelScope perform internal segmentation instead of pre-slicing by VAD.
    """
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    if wav_np.dtype != np.int16:
        wav_np = wav_np.astype(np.int16, copy=False)
    n = wav_np.shape[0]
    dur = float(n) / float(sample_rate)
    if dur <= DIAR_MIN_EFFECTIVE_SECONDS:
        logger.warning(
            "Diarization skipped (full input): duration %.2fs <= %.2fs.",
            dur,
            DIAR_MIN_EFFECTIVE_SECONDS,
        )
        return None
    seg = wav_np.astype(np.float32) / 32768.0
    seg = np.clip(seg, -1.0, 1.0)
    return [[0.0, dur, seg]]


def refine_with_subsegment(
    wav_np: np.ndarray,
    sample_rate: int,
    spans: list[tuple[float, float, int]],
    *,
    win_s: float = 1.5,
    hop_s: float = 0.75,
) -> list[tuple[float, float, int]]:
    """Placeholder for embedding-based span refinement (T7+). Returns ``spans`` unchanged."""
    _ = (wav_np, sample_rate, win_s, hop_s)
    return list(spans)


def _coerce_span_row(row: Any) -> tuple[float, float, int] | None:
    if row is None:
        return None
    if isinstance(row, (list, tuple)) and len(row) >= 3:
        try:
            return float(row[0]), float(row[1]), int(row[2])
        except (TypeError, ValueError):
            return None
    if isinstance(row, dict):
        d = row
        t0 = d.get("start") or d.get("start_s") or d.get("stime") or d.get("begin")
        t1 = d.get("end") or d.get("end_s") or d.get("etime")
        spk = d.get("speaker") or d.get("spk") or d.get("speaker_id") or d.get("cluster")
        if t0 is None or t1 is None or spk is None:
            return None
        try:
            if isinstance(spk, str) and spk.isdigit():
                spk_i = int(spk)
            elif isinstance(spk, (int, float)):
                spk_i = int(spk)
            else:
                spk_i = abs(hash(str(spk))) % 10_000
            return float(t0), float(t1), spk_i
        except (TypeError, ValueError):
            return None
    return None


def _normalize_span_rows(spans: Any) -> list[tuple[float, float, int]] | None:
    if spans is None:
        return None
    if isinstance(spans, np.ndarray):
        spans = spans.tolist()
    if not isinstance(spans, (list, tuple)) or not spans:
        return None
    norm: list[tuple[float, float, int]] = []
    for row in spans:
        parsed = _coerce_span_row(row)
        if parsed is not None and parsed[1] > parsed[0]:
            norm.append(parsed)
    return norm or None


def _extract_spans_dict(raw: dict[str, Any]) -> Any:
    from modelscope.outputs import OutputKeys

    keys_try = (
        OutputKeys.TEXT,
        "text",
        "output",
        "labels",
        "speaker_labels",
        "sysanalyse",
        "value",
    )
    for k in keys_try:
        v = raw.get(k)
        if v is None:
            continue
        if isinstance(v, dict):
            inner = v.get("text") or v.get("labels") or v.get("output")
            if inner is not None:
                return inner
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return v
    return None


def run_diarization(
    model: str,
    segments_list: list,
    model_revision: str | None = None,
) -> list[tuple[float, float, int]] | None:
    """Run clustering diarization; returns [(start_s, end_s, speaker_id), ...]."""
    if not segments_list:
        return None
    try:
        pipe = _get_pipeline(model, model_revision)
        raw = pipe(segments_list)
        if not isinstance(raw, dict):
            logger.warning("Diarization unexpected non-dict output: %s", type(raw))
            return None
        spans = _extract_spans_dict(raw)
        norm = _normalize_span_rows(spans)
        if not norm:
            logger.warning("Diarization returned empty or unparsed output keys=%s", list(raw.keys()))
            return None
        return norm
    except Exception:
        logger.exception("Diarization failed")
        return None
