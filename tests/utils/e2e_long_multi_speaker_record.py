# Copyright 2026 Xiaohongshu.
"""Write the same artifact trio as ``examples/test_long_multi_speaker.py``:

- ``asr_transcribe_results.json`` — per-VAD-segment ``FireRedAsr2.transcribe`` list
- ``asr_system_result.json`` — full ``FireRedAsr2System.process`` dict
- ``E2E_TEST_REPORT.md`` — short Markdown summary

Enable by setting env ``FIREREDASR2S_E2E_RECORD_DIR`` to an output directory (or pass
``--e2e_record_dir`` to ``scripts/run_full_test_matrix.py``).

When ``FireRedAsr2System`` uses a denoiser, raw-wav segment transcribe does not match
the denoised ASR path; we skip writing ``asr_transcribe_results.json`` (empty list + note).

``save_e2e_merged_full_stack_json`` writes a **single** JSON that bundles metadata,
enabled-feature tags, optional per-VAD-segment raw ASR transcribe dumps, the full
``process`` dict (``pipeline``), and **stream replay** (``stream_session``): the same
wav pushed in chunks like ``examples/streaming_simulate_from_wav.py`` (Stream-VAD +
``process_pcm_segment``), for comparing offline vs online segmentation.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from fireredasr2s.firereddiar.audio import load_pcm_int16_mono, prepare_asr_stack_audio

# 与 ``fireredasr2system`` 中 VAD 空段回退上界一致，避免采集 transcribe 时拖超长音频。
_VAD_FALLBACK_MAX_DUR_S = 600.0


def e2e_record_root_from_env() -> Path | None:
    raw = (os.environ.get("FIREREDASR2S_E2E_RECORD_DIR") or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


MERGED_FULL_STACK_SCHEMA = "fireredasr2s.e2e_merged_full_stack.v2"


def e2e_stream_chunk_ms_from_env() -> int:
    raw = (os.environ.get("FIREREDASR2S_E2E_STREAM_CHUNK_MS") or "").strip()
    if not raw:
        return 200
    try:
        return max(10, min(int(raw), 2000))
    except ValueError:
        return 200


def collect_stream_session_segment_events(
    system: Any,
    wav_path: str,
    *,
    uttid_prefix: str,
    chunk_ms: int | None = None,
) -> dict[str, Any]:
    """Replay ``wav_path`` in small PCM chunks; collect ``segment_final`` events (stream API).

    Mirrors ``examples/streaming_simulate_from_wav.py`` using the already-loaded ``system``.
    """
    cm = int(chunk_ms) if chunk_ms is not None else e2e_stream_chunk_ms_from_env()
    out: dict[str, Any] = {
        "chunk_ms": cm,
        "uttid_prefix": uttid_prefix,
        "events": [],
    }
    cfg = getattr(system, "config", None)

    if getattr(system, "denoiser", None) is not None:
        out["skipped"] = True
        out["skip_reason"] = (
            "已跳过：启用了降噪，流式回放基于磁盘原始 wav，与 process 主线不一致。"
        )
        return out

    asr_type = (getattr(cfg, "asr_type", "") or "").strip().lower()
    if asr_type != "aed":
        out["skipped"] = True
        out["skip_reason"] = (
            f"已跳过：open_stream 仅支持 asr_type='aed'，当前为 {getattr(cfg, 'asr_type', None)!r}。"
        )
        return out

    try:
        session = system.open_stream(uttid_prefix=uttid_prefix)
    except Exception as e:
        out["skipped"] = True
        out["skip_reason"] = "open_stream 失败"
        out["error"] = str(e)
        return out

    pcm, sr = sf.read(str(wav_path), dtype="int16")
    pcm = np.asarray(pcm, dtype=np.int16)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1).astype(np.int16)
    sr_i = int(sr)
    chunk_samples = max(int(16 * cm), 160)
    events: list[dict[str, Any]] = []
    try:
        for i in range(0, len(pcm), chunk_samples):
            chunk = pcm[i : i + chunk_samples]
            events.extend(session.push_pcm_int16_mono(chunk, sample_rate=sr_i))
        events.extend(session.finalize())
    except Exception as e:
        out["skipped"] = True
        out["skip_reason"] = "流式回放失败"
        out["error"] = str(e)
        return out

    if cfg is not None and bool(getattr(cfg, "enable_diarization", False)):
        out["replay_note"] = (
            "离线 pipeline 含 diarization；流式 session 不应用在线 diar，段内说话人标签规则见 pipeline.speaker_label_note。"
        )
    out["events"] = events
    out["event_count"] = len(events)
    return out


def enabled_feature_tags_from_system(system: Any) -> list[str]:
    """Human-readable flags derived from ``FireRedAsr2System.config`` (for merged E2E JSON)."""
    c = getattr(system, "config", None)
    if c is None:
        return []
    tags: list[str] = []
    if bool(getattr(c, "enable_vad", True)):
        tags.append("vad")
    tags.append("asr")
    if bool(getattr(c, "enable_punc", False)):
        tags.append("punc")
    if bool(getattr(c, "enable_lid", False)):
        tags.append("lid")
    if bool(getattr(c, "enable_itn", False)):
        tags.append("itn")
    if bool(getattr(c, "enable_denoise", False)):
        backend = getattr(getattr(c, "denoise_config", None), "backend", "")
        tags.append(f"denoise:{backend or 'default'}")
    if bool(getattr(c, "enable_diarization", False)):
        tags.append(f"diarization:{getattr(c, 'diar_backend', '')}")
    if bool(getattr(c, "enable_speaker_id", False)):
        tags.append(f"speaker_id:{getattr(c, 'speaker_embedder', '')}")
    ac = getattr(c, "asr_config", None)
    if ac is not None:
        hw = getattr(ac, "hotwords", None) or []
        if hw and float(getattr(ac, "hotword_weight", 0.0) or 0.0) > 0.0:
            tags.append("hotwords")
        if bool(getattr(ac, "return_timestamp", False)):
            tags.append("word_timestamp")
    return tags


def save_e2e_merged_full_stack_json(
    *,
    output_root: Path,
    case_id: str,
    wav_path: str,
    uttid: str,
    system: Any,
    system_result: dict,
) -> Path:
    """Write ``e2e_merged_full_stack.json`` under ``output_root / <safe_case_id> /``."""
    safe_id = case_id.replace("/", "_").replace("\\", "_")
    out_dir = output_root / safe_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "e2e_merged_full_stack.json"

    transcribe_note = "与 `examples/test_long_multi_speaker.py` 一致"
    transcribe_list: list[dict] = []
    if getattr(system, "denoiser", None) is not None:
        transcribe_note = (
            "已跳过：启用了降噪，逐段 transcribe 基于原始 wav，与 process 主线不一致。"
        )
    else:
        transcribe_list = collect_segment_transcribe_results(system, wav_path, uttid)

    stream_prefix = f"{uttid}_stream"
    stream_session = collect_stream_session_segment_events(
        system, wav_path, uttid_prefix=stream_prefix
    )

    payload = {
        "schema": MERGED_FULL_STACK_SCHEMA,
        "meta": {
            "case_id": safe_id,
            "wav_path": wav_path,
            "uttid": uttid,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "enabled_features": enabled_feature_tags_from_system(system),
            "segment_transcribe_note": transcribe_note,
        },
        "segment_transcribe": transcribe_list,
        "pipeline": system_result,
        "stream_session": stream_session,
    }
    save_json(out_path, payload)
    return out_path


def collect_segment_transcribe_results(
    system: Any,
    wav_path: str,
    uttid_prefix: str,
) -> list[dict]:
    """Mirror ``transcribe_segments_like_test_api`` using the already-loaded ``system``."""
    if system.vad is None or system.asr is None:
        return []
    wav_np, sr = load_pcm_int16_mono(wav_path)
    wav_np, sr = prepare_asr_stack_audio(wav_np, sr)
    vad_result, _ = system.vad.detect((wav_np, sr))
    segments = list(vad_result["timestamps"])
    dur_s = float(len(wav_np)) / float(sr)
    if not segments and 0.2 < dur_s <= _VAD_FALLBACK_MAX_DUR_S:
        segments = [(0.0, dur_s)]
    all_results: list[dict] = []
    for start_s, end_s in segments:
        seg = wav_np[int(start_s * sr) : int(end_s * sr)]
        if seg.size == 0:
            continue
        seg_uttid = f"{uttid_prefix}_s{int(start_s * 1000)}_e{int(end_s * 1000)}"
        results = system.asr.transcribe([seg_uttid], [(sr, seg)])
        all_results.extend(results)
    return all_results


def write_e2e_test_report(
    path: Path,
    *,
    case_id: str,
    wav_path: str,
    uttid: str,
    transcribe_skipped_reason: str,
    system_result: dict,
    transcribe_count: int,
) -> None:
    text = system_result.get("text") or ""
    preview = text[:800] + ("…" if len(text) > 800 else "")
    sentences = system_result.get("sentences") or []
    diar = system_result.get("diarization_spans") or []
    lines = [
        "# FireRedASR2S — E2E 测试记录（与 `test_long_multi_speaker` 同构）",
        "",
        f"- **生成时间 (UTC)**：`{datetime.now(timezone.utc).isoformat()}`",
        f"- **用例目录名**：`{case_id}`",
        f"- **pytest 环境变量**：`FIREREDASR2S_E2E_RECORD_DIR`",
        f"- **输入 wav**：`{wav_path}`",
        f"- **uttid**：`{uttid}`",
        "",
        "## 输出文件",
        "",
        f"- **分段 transcribe JSON**：`asr_transcribe_results.json`（{transcribe_count} 条；{transcribe_skipped_reason}）",
        "- **系统 pipeline JSON**：`asr_system_result.json`",
        "",
        "## 系统结果摘要",
        "",
        f"- **合并句数**：{len(sentences)}",
        f"- **diarization_spans**：{len(diar)} 条",
        f"- **合并文本预览（前 800 字）**：",
        "",
        "```",
        preview,
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def save_long_multi_speaker_style_artifacts(
    *,
    output_root: Path | None,
    case_id: str,
    wav_path: str,
    uttid: str,
    system: Any,
    system_result: dict,
) -> None:
    """No-op if ``output_root`` is None."""
    if output_root is None:
        return
    safe_id = case_id.replace("/", "_").replace("\\", "_")
    out_dir = output_root / safe_id
    out_dir.mkdir(parents=True, exist_ok=True)

    transcribe_reason = "与 `examples/test_long_multi_speaker.py` 一致"
    transcribe_list: list[dict] = []
    if getattr(system, "denoiser", None) is not None:
        transcribe_reason = (
            "已跳过：当前 System 启用了降噪，`process` 使用增强后波形；"
            "逐段 transcribe 仍基于磁盘原始 wav，与主线不一致，故不写分段结果。"
        )
    else:
        prefix = uttid
        transcribe_list = collect_segment_transcribe_results(system, wav_path, prefix)

    save_json(out_dir / "asr_system_result.json", system_result)
    save_json(out_dir / "asr_transcribe_results.json", transcribe_list)
    write_e2e_test_report(
        out_dir / "E2E_TEST_REPORT.md",
        case_id=safe_id,
        wav_path=wav_path,
        uttid=uttid,
        transcribe_skipped_reason=transcribe_reason,
        system_result=system_result,
        transcribe_count=len(transcribe_list),
    )
