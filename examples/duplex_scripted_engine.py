# Copyright 2026 Xiaohongshu.
"""全双工脚本化回放：单线程按块 push PCM + 本地播放栈（无 sounddevice）。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from fireredasr2s.duplex import NlmsMonoAec


def _float_to_i16_mono(x: np.ndarray) -> np.ndarray:
    return (np.clip(x.astype(np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)


def _i16_to_float_mono(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def _chunk_rms_int16(mic_i16: np.ndarray) -> float:
    x = np.asarray(mic_i16, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x * x)))


def _sim_gate_barge_in_events(
    rt: DuplexSimRuntime,
    mic_i16: np.ndarray,
    evs: list[dict[str, Any]],
    *,
    process_barge_in: bool,
) -> list[dict[str, Any]]:
    """仿真侧抑制「客户麦克几乎无声却触发 VAD」的假抢话（AEC 残差易被当成起讲）。"""
    if not process_barge_in:
        return evs
    now = time.monotonic()
    rms = _chunk_rms_int16(mic_i16)
    with rt.lock:
        in_pb = rt.in_playback
        t0 = float(rt.playback_start_mono or 0.0)
    min_r = float(rt.barge_mic_rms16_min)
    grace = float(rt.barge_grace_s_after_playback)
    loud = rms >= float(rt.barge_loud_mic_rms16_bypass_grace)
    grace_ok = loud or ((now - t0) >= grace if t0 > 0.0 else True)
    energy_ok = rms >= min_r
    out: list[dict[str, Any]] = []
    for ev in evs:
        if ev.get("event") != "barge_in":
            out.append(ev)
            continue
        if not in_pb:
            out.append(ev)
            continue
        if energy_ok and grace_ok:
            out.append(ev)
        else:
            out.append(
                {
                    "event": "barge_in_suppressed",
                    "reason": "sim_no_customer_energy_or_grace",
                    "mic_rms_i16_approx": round(rms, 1),
                    "mic_rms_i16_min": min_r,
                    "loud_bypass_rms16": float(rt.barge_loud_mic_rms16_bypass_grace),
                    "grace_elapsed_s": round(now - t0, 3) if t0 > 0.0 else None,
                    "grace_min_s": grace,
                }
            )
    return out


def _pipeline_user_text(ev: dict[str, Any]) -> str:
    if ev.get("event") != "segment_final":
        return ""
    pipe = ev.get("pipeline") or {}
    t = (pipe.get("text") or "").strip()
    if t:
        return t
    parts: list[str] = []
    for s in pipe.get("sentences") or []:
        if isinstance(s, dict) and s.get("text"):
            parts.append(str(s["text"]).strip())
    return "".join(parts).strip()


@dataclass
class DuplexSimRuntime:
    lock: threading.Lock
    session: Any
    aec: NlmsMonoAec | None
    sr: int
    tts_i16: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    tts_pos: int = 0
    pending_tts: np.ndarray | None = None
    pending_tts_caption: str = ""
    in_playback: bool = False
    generation: int = 0
    playback_serial: int = 0
    current_playback_id: str | None = None
    next_tts_segment: str = "assistant"
    record_call_audio: bool = False
    pending_voice_note: str = ""
    #: ``begin_local_playback`` 时刻（``time.monotonic``），用于抢话保护窗
    playback_start_mono: float = 0.0
    #: 播发开始后若干秒内忽略 barge（贴近真人反应时间，并避开 TTS 起音诱发 VAD）
    barge_grace_s_after_playback: float = 0.48
    #: 原始客户麦克 RMS（int16）低于此值则视为「线上无人抢话」，抑制 barge_in
    barge_mic_rms16_min: float = 260.0
    #: RMS 高于此值视为客户明显在喊麦，**绕过** grace（避免真叠音被起音保护误杀）
    barge_loud_mic_rms16_bypass_grace: float = 1400.0


def _stop_playback_locked(rt: DuplexSimRuntime) -> None:
    rt.pending_tts = None
    rt.pending_tts_caption = ""
    rt.tts_i16 = np.zeros(0, dtype=np.int16)
    rt.tts_pos = 0
    if rt.in_playback:
        rt.session.end_local_playback()
        rt.in_playback = False
    rt.current_playback_id = None


def _try_start_pending_playback(
    rt: DuplexSimRuntime,
    *,
    rec_add: Callable[..., None] | None,
    verbose_print: Callable[[str], None] | None,
) -> None:
    if rt.tts_pos < len(rt.tts_i16):
        return
    if rt.pending_tts is None or len(rt.pending_tts) == 0:
        return
    rt.tts_i16 = rt.pending_tts
    rt.pending_tts = None
    rt.tts_pos = 0
    rt.playback_serial += 1
    pid = f"tts-{rt.playback_serial}"
    rt.session.begin_local_playback(
        playback_id=pid,
        anchor_wallclock_ms=0,
    )
    rt.in_playback = True
    rt.current_playback_id = pid
    rt.playback_start_mono = time.monotonic()
    cap = (rt.pending_tts_caption or "").strip()
    rt.pending_tts_caption = ""
    if rec_add is not None:
        seg = rt.next_tts_segment
        rt.next_tts_segment = "assistant"
        vn = (getattr(rt, "pending_voice_note", None) or "").strip()
        if hasattr(rt, "pending_voice_note"):
            rt.pending_voice_note = ""
        rec_add(
            event="tts_start",
            segment=seg,
            playback_id=pid,
            text=cap,
            samples=int(len(rt.tts_i16)),
            voice_note=vn,
        )
    if verbose_print:
        verbose_print(f"# begin_local_playback {pid} samples={len(rt.tts_i16)}")


def _finish_if_tts_done(
    rt: DuplexSimRuntime,
    *,
    rec_add: Callable[..., None] | None,
    verbose_print: Callable[[str], None] | None,
) -> None:
    if not rt.in_playback:
        return
    if rt.tts_pos < len(rt.tts_i16):
        return
    if len(rt.tts_i16) == 0:
        return
    done_pid = rt.current_playback_id
    rt.session.end_local_playback()
    rt.in_playback = False
    rt.tts_i16 = np.zeros(0, dtype=np.int16)
    rt.tts_pos = 0
    if rec_add is not None and done_pid:
        rec_add(event="tts_end", playback_id=done_pid)
    rt.current_playback_id = None
    if verbose_print:
        verbose_print("# end_local_playback()")


def feed_duplex_chunk(
    rt: DuplexSimRuntime,
    session: Any,
    mic_i16: np.ndarray,
    *,
    rec_add: Callable[..., None] | None = None,
    record_audio: tuple[list[np.ndarray], list[np.ndarray]] | None = None,
    verbose_print: Callable[[str], None] | None = None,
    process_barge_in: bool = True,
    collect_events: list[dict[str, Any]] | None = None,
) -> list[str]:
    """推送一块麦克 PCM（int16 mono），返回本块内触发的用户识别文本列表。

    若提供 ``collect_events``，会把 ``push_microphone_pcm`` 返回的原始事件追加进去（含 ``barge_in`` / ``segment_final`` 等）。"""
    n = int(len(mic_i16))
    ref = np.zeros(n, dtype=np.float32)
    with rt.lock:
        if rt.tts_pos < len(rt.tts_i16):
            take = min(n, len(rt.tts_i16) - rt.tts_pos)
            ref[:take] = _i16_to_float_mono(rt.tts_i16[rt.tts_pos : rt.tts_pos + take])
            rt.tts_pos += take

    mic = mic_i16.astype(np.float32) / 32768.0
    if rt.aec is not None:
        mic_u = rt.aec.process_block(mic, ref)
    else:
        mic_u = mic
    pcm16 = _float_to_i16_mono(mic_u)
    evs = session.push_microphone_pcm(pcm16, sample_rate=int(rt.sr))
    evs = _sim_gate_barge_in_events(
        rt, mic_i16, evs, process_barge_in=process_barge_in
    )
    if collect_events is not None:
        collect_events.extend(evs)

    out_utts: list[str] = []
    with rt.lock:
        for ev in evs:
            if ev.get("event") == "barge_in" and process_barge_in:
                if rec_add is not None:
                    rec_add(
                        event="barge_in",
                        playback_id=ev.get("playback_id"),
                    )
                rt.generation += 1
                _stop_playback_locked(rt)
                if verbose_print:
                    verbose_print("# barge_in：停止播放")
                continue
            if ev.get("event") == "barge_in_suppressed" and verbose_print:
                verbose_print(f"# {ev.get('event')}: {ev.get('reason')} {ev}")
            utt = _pipeline_user_text(ev)
            if utt:
                out_utts.append(utt)
        _finish_if_tts_done(rt, rec_add=rec_add, verbose_print=verbose_print)
        _try_start_pending_playback(rt, rec_add=rec_add, verbose_print=verbose_print)
        _finish_if_tts_done(rt, rec_add=rec_add, verbose_print=verbose_print)

    if record_audio is not None:
        mic_chunks, tts_chunks = record_audio
        # 落盘左声道用「脚本送入的原始麦克」，便于立体声听辨客户(L)/客服 TTS 参考(R)。
        # 仿真场景下 uplink 不含扬声器回声，AEC 后的 pcm16 在静音麦克+有 TTS 时会把参考泄漏到误差里，
        # 听起来像左右都是客服；流式 ASR 仍使用上面的 pcm16。
        mic_chunks.append(np.asarray(mic_i16, dtype=np.int16).copy())
        tts_chunks.append(_float_to_i16_mono(ref).copy())

    return out_utts


def feed_pcm_stream(
    rt: DuplexSimRuntime,
    session: Any,
    pcm: np.ndarray,
    chunk_samples: int,
    *,
    rec_add: Callable[..., None] | None,
    record_audio: tuple[list[np.ndarray], list[np.ndarray]] | None,
    verbose_print: Callable[[str], None] | None,
    on_segment_text: Callable[[str], None] | None = None,
) -> None:
    for i in range(0, len(pcm), chunk_samples):
        chunk = pcm[i : i + chunk_samples]
        if len(chunk) < chunk_samples:
            pad = np.zeros(chunk_samples - len(chunk), dtype=np.int16)
            chunk = np.concatenate([chunk, pad])
        for utt in feed_duplex_chunk(
            rt,
            session,
            chunk,
            rec_add=rec_add,
            record_audio=record_audio,
            verbose_print=verbose_print,
            collect_events=None,
        ):
            if on_segment_text:
                on_segment_text(utt)
