# Copyright 2026 Xiaohongshu.
"""Align ASR outputs with diarization spans (word-level speaker assignment)."""

from __future__ import annotations

import re
from typing import Any, Optional


def speaker_by_overlap_ms(
    start_ms: int,
    end_ms: int,
    diar_spans: list[tuple[float, float, int]],
) -> int:
    """Pick speaker id by maximum overlap in seconds (same policy as ``_speaker_by_overlap``)."""
    if not diar_spans:
        return 0
    s0, e0 = start_ms / 1000.0, end_ms / 1000.0
    center = (s0 + e0) / 2.0
    best_spk = int(diar_spans[0][2])
    best_ov = -1.0
    for t0, t1, spk in diar_spans:
        ov = max(0.0, min(e0, t1) - max(s0, t0))
        spk_i = int(spk)
        if ov > best_ov + 1e-9:
            best_ov = ov
            best_spk = spk_i
        elif abs(ov - best_ov) <= 1e-9 and ov > 0 and t0 <= center <= t1:
            best_spk = spk_i
    if best_ov > 0:
        return best_spk
    for t0, t1, spk in diar_spans:
        if t0 <= center <= t1:
            return int(spk)
    t0, t1, spk = min(
        diar_spans,
        key=lambda x: min(abs(center - x[0]), abs(center - x[1])),
    )
    return int(spk)


def _merge_token_text(parts: list[str]) -> str:
    if not parts:
        return ""
    out = []
    for i, p in enumerate(parts):
        if i == 0:
            out.append(p)
            continue
        prev = out[-1]
        if re.search(r"[a-zA-Z0-9]$", prev) and re.match(r"^[a-zA-Z0-9]", p):
            out.append(" ")
        out.append(p)
    return "".join(out)


def group_tokens_by_speaker(
    start_ms: int,
    timestamps: list,
    diar_spans: list[tuple[float, float, int]],
) -> list[tuple[int, int, int, str]]:
    """Return ``(w_start_ms, w_end_ms, spk_id, text)`` groups (consecutive same speaker)."""
    groups: list[tuple[int, int, int, str]] = []
    cur_spk: int | None = None
    buf_s = buf_e = 0
    buf_toks: list[str] = []
    for row in timestamps:
        if len(row) != 3:
            continue
        w, s, e = row[0], float(row[1]), float(row[2])
        w0 = int(start_ms + s * 1000)
        w1 = int(start_ms + e * 1000)
        if w1 < w0:
            w0, w1 = w1, w0
        spk = speaker_by_overlap_ms(w0, w1, diar_spans)
        tok = str(w).strip()
        if cur_spk is None:
            cur_spk = spk
            buf_s, buf_e = w0, w1
            buf_toks = [tok] if tok else []
        elif spk == cur_spk:
            buf_e = w1
            if tok:
                buf_toks.append(tok)
        else:
            groups.append((buf_s, buf_e, cur_spk, _merge_token_text(buf_toks)))
            cur_spk = spk
            buf_s, buf_e = w0, w1
            buf_toks = [tok] if tok else []
    if cur_spk is not None:
        groups.append((buf_s, buf_e, cur_spk, _merge_token_text(buf_toks)))
    return groups


def merge_short_speaker_groups(
    groups: list[tuple[int, int, int, str]],
    min_dur_ms: int,
) -> list[tuple[int, int, int, str]]:
    """Merge fragments shorter than ``min_dur_ms`` into the neighbour with the same speaker
    when possible; otherwise merge into the shorter adjacent interval.
    """
    if min_dur_ms <= 0 or len(groups) <= 1:
        return groups
    out = list(groups)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(out):
            s, e, spk, txt = out[i]
            if e - s >= min_dur_ms or not txt.strip():
                i += 1
                continue
            if i > 0 and out[i - 1][2] == spk:
                ps, pe, pspk, ptxt = out[i - 1]
                out[i - 1] = (ps, e, pspk, ptxt + txt)
                out.pop(i)
                changed = True
                continue
            if i + 1 < len(out) and out[i + 1][2] == spk:
                ns, ne, nspk, ntxt = out[i + 1]
                out[i] = (s, ne, spk, txt + ntxt)
                out.pop(i + 1)
                changed = True
                continue
            if i > 0:
                ps, pe, pspk, ptxt = out[i - 1]
                out[i - 1] = (ps, e, pspk, ptxt + txt)
                out.pop(i)
                changed = True
                continue
            if i + 1 < len(out):
                ns, ne, nspk, ntxt = out[i + 1]
                out[i] = (s, ne, spk, txt + ntxt)
                out.pop(i + 1)
                changed = True
                continue
            i += 1
    return out


def try_word_diar_sentences(
    *,
    asr_result: dict[str, Any],
    punc_result: dict[str, Any],
    lid_result: Optional[dict[str, Any]],
    diar_spans: list[tuple[float, float, int]],
    vad_segment_idx: int,
    min_speaker_dur_ms: int,
    enable_punc: bool,
    punc_model: Any,
) -> Optional[list[dict[str, Any]]]:
    """Build per-speaker sub-sentences from ASR token timestamps, or ``None`` to fall back."""
    ts = asr_result.get("timestamp")
    if not ts or not diar_spans:
        return None
    if enable_punc and "punc_sentences" in punc_result:
        if len(punc_result["punc_sentences"]) != 1:
            return None

    start_ms_s, end_ms_s = asr_result["uttid"].split("_")[-2:]
    start_ms, end_ms = int(start_ms_s[1:]), int(end_ms_s[1:])

    groups = group_tokens_by_speaker(start_ms, ts, diar_spans)
    groups = merge_short_speaker_groups(groups, min_speaker_dur_ms)
    out: list[dict[str, Any]] = []
    for gs, ge, spk, raw_text in groups:
        text = raw_text
        if enable_punc and punc_model is not None and text.strip():
            pr = punc_model.process([text], [asr_result["uttid"]])
            if pr:
                text = pr[0]["punc_text"]
        sent = {
            "start_ms": gs,
            "end_ms": ge,
            "text": text,
            "asr_confidence": asr_result.get("confidence", 0.0),
            "lang": None,
            "lang_confidence": 0,
            "vad_segment_idx": vad_segment_idx,
            "diar_speaker_id": spk,
            "spk_label": spk,
            "word_diar_spk": True,
        }
        if lid_result:
            sent["lang"] = lid_result.get("lang")
            sent["lang_confidence"] = lid_result.get("confidence", 0)
        out.append(sent)
    return out if out else None
