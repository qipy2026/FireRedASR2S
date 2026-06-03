#!/usr/bin/env python3
"""富用例：预置多轮中文对话 WAV（合成语音），脚本化推流全双工，紧凑「会话内时间轴」+ 通话记录/可选录音。

无需麦克风；不调用 LLM（助手轨为 scenario.json 中预生成 WAV）。

步骤::

  .venv\\Scripts\\python.exe scripts\\prepare_duplex_scenario_wavs.py
  .venv\\Scripts\\python.exe examples\\full_duplex_scripted_rich.py --device xpu --call-audio

场景清单: examples/duplex_scenario_rich/scenario.json
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
_EXAMPLES = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import stdio_utf8_windows  # noqa: E402

stdio_utf8_windows.apply_stdio_utf8()

os.environ.setdefault("PYTORCH_ENABLE_XPU_FALLBACK", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from fireredasr2s.fireredasr2 import FireRedAsr2Config  # noqa: E402
from fireredasr2s.fireredasr2system import FireRedAsr2System, FireRedAsr2SystemConfig  # noqa: E402
from fireredasr2s.firereddiar.audio import prepare_asr_stack_audio  # noqa: E402
from fireredasr2s.fireredlid import FireRedLidConfig  # noqa: E402
from fireredasr2s.fireredpunc import FireRedPuncConfig  # noqa: E402
from fireredasr2s.fireredvad import FireRedVadConfig  # noqa: E402
from fireredasr2s.repo_dotenv import default_asr_device, load_repo_dotenv  # noqa: E402

import duplex_scripted_engine as _dse  # noqa: E402

DuplexSimRuntime = _dse.DuplexSimRuntime
feed_duplex_chunk = _dse.feed_duplex_chunk


def _load_voice_helpers():
    p = _REPO / "examples" / "full_duplex_voice_llm_tts.py"
    spec = importlib.util.spec_from_file_location("_fdv_scripted", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.CallSessionRecord, mod._save_call_session_audio


def _read_wav_16k_mono(path: Path) -> tuple[np.ndarray, int]:
    pcm, sr = sf.read(str(path), dtype="int16")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1).astype(np.int16)
    out, sr2 = prepare_asr_stack_audio(pcm, int(sr))
    if int(sr2) != 16000:
        raise RuntimeError(f"expected 16 kHz: {path} got {sr2}")
    return out, 16000


def _silence_samples(sr: int, ms: float) -> np.ndarray:
    n = max(int(sr * ms / 1000.0), 0)
    return np.zeros(n, dtype=np.int16)


def main() -> None:
    load_repo_dotenv()
    CallSessionRecord, save_call_audio = _load_voice_helpers()

    p = argparse.ArgumentParser(description="脚本化富场景全双工（预置 WAV）")
    p.add_argument("--scenario", type=str, default="examples/duplex_scenario_rich/scenario.json")
    p.add_argument("--models_root", type=str, default="pretrained_models")
    p.add_argument(
        "--device",
        type=str,
        default=default_asr_device(),
        help="ASR 等设备；默认 xpu；仅 FIRERED_ASR_DEVICE 可覆盖（不读 ASR_DEVICE）",
    )
    p.add_argument("--aec", type=str, choices=("none", "nlms"), default="none")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-call-record", action="store_true")
    p.add_argument(
        "--call-record-dir",
        type=str,
        default="output/call_recordings",
        help="通话记录与录音目录（默认 output/call_recordings，与 output/logs 分离）",
    )
    p.add_argument("--no-call-record-jsonl", action="store_true")
    p.add_argument("--call-audio", action="store_true")
    args = p.parse_args()

    scen_path = Path(args.scenario)
    if not scen_path.is_absolute():
        scen_path = _REPO / scen_path
    if not scen_path.is_file():
        print(f"缺少场景文件: {scen_path}", file=sys.stderr)
        print("请先运行: .venv\\Scripts\\python.exe scripts\\prepare_duplex_scenario_wavs.py", file=sys.stderr)
        sys.exit(2)

    data = json.loads(scen_path.read_text(encoding="utf-8"))
    scen_dir = scen_path.parent
    chunk_ms = int(data.get("chunk_ms") or 40)
    sr = 16000
    chunk_samples = max(int(sr * chunk_ms / 1000.0), 160)
    user_pad_ms = float(data.get("user_utterance_pad_ms") or 180)
    greet_tail_extra = int(data.get("greeting_tail_extra_chunks") or 12)
    post_user_sil_ms = float(data.get("post_user_silence_ms") or 200)
    post_user_vad_chunks = int(data.get("post_user_vad_extra_chunks") or 18)
    assist_tail_extra = int(data.get("assistant_playback_extra_chunks") or 18)

    turns_for_hw: list[dict[str, Any]] = list(data.get("turns") or [])
    hotwords: list[str] = []
    _seen_hw: set[str] = set()
    for t in turns_for_hw:
        u = (t.get("user_text") or "").strip()
        if u and u not in _seen_hw:
            _seen_hw.add(u)
            hotwords.append(u)
    greet_txt = ((data.get("greeting") or {}).get("text") or "").strip()
    if greet_txt and greet_txt not in _seen_hw:
        hotwords.append(greet_txt)
        _seen_hw.add(greet_txt)
    for x in data.get("asr_hotwords_extra") or []:
        if isinstance(x, str):
            s = x.strip()
            if s and s not in _seen_hw:
                _seen_hw.add(s)
                hotwords.append(s)
    hw_w = float(data.get("asr_hotword_weight") or 2.0)
    hw_b = float(data.get("asr_hotword_complete_bonus") or 0.5)

    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    call_rec: CallSessionRecord | None = None
    session_stem: Path | None = None
    if not args.no_call_record:
        rec_path = Path(args.call_record_dir)
        if not rec_path.is_absolute():
            rec_path = _REPO / rec_path
        session_stem = rec_path / f"call_record_scripted_{datetime.datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
        call_rec = CallSessionRecord(session_stem, write_jsonl=not args.no_call_record_jsonl)
        print(f"# 通话记录: {session_stem}.json / _还原.txt", flush=True)
    elif args.call_audio:
        rec_path = Path(args.call_record_dir)
        if not rec_path.is_absolute():
            rec_path = _REPO / rec_path
        session_stem = rec_path / f"call_record_scripted_{datetime.datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
        print(f"# 仅通话录音: {session_stem}_mic_asr.wav", flush=True)

    scenario_clock = [0.0]

    def rec_add(**kw: Any) -> None:
        if call_rec is not None:
            call_rec.add(t_scenario_s=scenario_clock[0], **kw)

    def vprint(msg: str) -> None:
        if args.verbose:
            print(msg, flush=True)

    def advance_clock(n_samples: int) -> None:
        scenario_clock[0] += float(n_samples) / float(sr)

    audio_mic: list[np.ndarray] = []
    audio_tts: list[np.ndarray] = []
    record_bufs = (audio_mic, audio_tts) if args.call_audio else None

    root = Path(args.models_root)
    asr_cfg = FireRedAsr2Config(
        use_gpu=args.device != "cpu",
        device=args.device,
        return_timestamp=False,
        hotwords=hotwords,
        hotword_weight=hw_w,
        hotword_complete_bonus=hw_b,
    )
    print(
        f"# ASR 热词偏置: {len(hotwords)} 条 weight={hw_w} complete_bonus={hw_b}",
        flush=True,
    )
    cfg = FireRedAsr2SystemConfig(
        vad_model_dir=str(root / "FireRedVAD" / "VAD"),
        lid_model_dir=str(root / "FireRedLID"),
        asr_model_dir=str(root / "FireRedASR2-AED"),
        punc_model_dir=str(root / "FireRedPunc"),
        vad_config=FireRedVadConfig(use_gpu=False),
        lid_config=FireRedLidConfig(use_gpu=args.device != "cpu"),
        asr_config=asr_cfg,
        punc_config=FireRedPuncConfig(use_gpu=args.device != "cpu"),
        enable_vad=True,
        enable_lid=False,
        enable_punc=True,
        enable_diarization=False,
        stream_vad_use_gpu=False,
    )
    sys_m = FireRedAsr2System(cfg)
    session = sys_m.open_full_duplex_stream(uttid_prefix="scripted_rich", verbose_vad=args.verbose)

    from fireredasr2s.duplex import NlmsMonoAec  # noqa: E402

    aec = None
    if args.aec == "nlms":
        aec = NlmsMonoAec(filter_len=2048, mu=0.25, ref_delay_samples=0)

    rt = DuplexSimRuntime(
        lock=__import__("threading").Lock(),
        session=session,
        aec=aec,
        sr=sr,
        record_call_audio=bool(args.call_audio),
    )

    def push_chunk(mic_i16: np.ndarray) -> list[str]:
        utts = feed_duplex_chunk(
            rt,
            session,
            mic_i16,
            rec_add=rec_add if call_rec else None,
            record_audio=record_bufs,
            verbose_print=vprint if args.verbose else None,
        )
        advance_clock(len(mic_i16))
        return utts

    def push_pcm(pcm: np.ndarray) -> None:
        for i in range(0, len(pcm), chunk_samples):
            chunk = pcm[i : i + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.concatenate([chunk, np.zeros(chunk_samples - len(chunk), dtype=np.int16)])
            push_chunk(chunk)

    turns: list[dict[str, Any]] = list(data.get("turns") or [])
    turn_consumed = 0

    def apply_turn(asr_text: str) -> None:
        nonlocal turn_consumed
        if turn_consumed >= len(turns):
            return
        t = turns[turn_consumed]
        script_txt = str(t.get("user_text", ""))
        asr_hyp = (asr_text or "").strip()
        rec_add(
            event="user_asr",
            text=script_txt,
            asr_hypothesis=asr_hyp,
            turn_index=turn_consumed + 1,
        )
        rec_add(
            event="assistant_llm",
            text=str(t.get("assistant_text", "")),
            source="canned_wav",
            turn_index=turn_consumed + 1,
        )
        apath = scen_dir / str(t["assistant_wav"])
        apcm, _ = _read_wav_16k_mono(apath)
        with rt.lock:
            rt.pending_voice_note = "〔女声 Edge 晓晓〕"
            rt.pending_tts_caption = str(t.get("assistant_text", ""))
            rt.next_tts_segment = "assistant"
            rt.pending_tts = apcm
        turn_consumed += 1
        hyp_disp = asr_hyp if asr_hyp else "（无）"
        print(f"# 轮次 {turn_consumed} 用户(稿): {script_txt} 〔ASR:{hyp_disp}〕", flush=True)
        print(f"# 轮次 {turn_consumed} 助手(女声稿): {t.get('assistant_text', '')}", flush=True)

    g = data.get("greeting") or {}
    gwav, _ = _read_wav_16k_mono(scen_dir / str(g["wav"]))
    with rt.lock:
        rt.pending_voice_note = "〔女声 Edge 晓晓〕"
        rt.pending_tts_caption = str(g.get("text", ""))
        rt.next_tts_segment = "greeting"
        rt.pending_tts = gwav
    print("# 推送问候播放 + 静音上行 …", flush=True)
    greet_chunks = int(np.ceil(len(gwav) / chunk_samples)) + max(greet_tail_extra, 0)
    for _ in range(greet_chunks):
        push_chunk(np.zeros(chunk_samples, dtype=np.int16))

    pad_after_g = float(data.get("silence_after_greeting_ms") or 220)
    push_pcm(_silence_samples(sr, pad_after_g))

    between_ms = float(data.get("silence_between_turns_ms") or 280)

    for idx, t in enumerate(turns):
        upath = scen_dir / str(t["user_wav"])
        upcm, _ = _read_wav_16k_mono(upath)
        pad_pre = _silence_samples(sr, user_pad_ms)
        pad_post = _silence_samples(sr, user_pad_ms)
        upcm = np.concatenate([pad_pre, upcm, pad_post])
        print(f"# --- 推送用户轮 {idx + 1}/{len(turns)} wav={upath.name} pad={user_pad_ms}ms ---", flush=True)
        collected: list[str] = []
        for i in range(0, len(upcm), chunk_samples):
            chunk = upcm[i : i + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.concatenate([chunk, np.zeros(chunk_samples - len(chunk), dtype=np.int16)])
            collected.extend(push_chunk(chunk))
        push_pcm(_silence_samples(sr, post_user_sil_ms))
        for _ in range(max(post_user_vad_chunks, 0)):
            collected.extend(push_chunk(np.zeros(chunk_samples, dtype=np.int16)))
        asr_first = (collected[0] or "").strip() if collected else ""
        apply_turn(asr_first)
        assist_path = scen_dir / str(t["assistant_wav"])
        apcm, _ = _read_wav_16k_mono(assist_path)
        assist_chunks = int(np.ceil(len(apcm) / chunk_samples)) + max(assist_tail_extra, 0)
        for _ in range(assist_chunks):
            push_chunk(np.zeros(chunk_samples, dtype=np.int16))
        if idx + 1 < len(turns):
            push_pcm(_silence_samples(sr, between_ms))

    for ev in session.finalize():
        if args.verbose:
            print(__import__("json").dumps(ev, ensure_ascii=False, default=str)[:2000], flush=True)

    meta_end: dict[str, Any] = {
        "started_at": started_at,
        "ended_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "asr_device": args.device,
        "llm_model": "canned_wav",
        "llm_base_url": "n/a",
        "session_seconds": round(scenario_clock[0], 3),
        "transcript_clock": "scenario",
        "scenario_id": data.get("scenario_id", ""),
        "scenario_path": str(scen_path),
        "duplex_scenario_voices": (
            "音色（预生成）: 用户男声 zh-CN-YunxiNeural；问候与助手女声 zh-CN-XiaoxiaoNeural；"
            "Edge 默认语速 -5%、音高 +1Hz（见 scripts/prepare_duplex_scenario_wavs.py）。"
            "还原正文「用户（稿）」与 WAV 文案一致；〔ASR〕为识别参考。"
        ),
    }
    if args.call_audio and session_stem is not None and audio_mic:
        meta_end["call_audio"] = save_call_audio(session_stem, audio_mic, audio_tts, sr)
        print(f"# 通话录音: {Path(meta_end['call_audio']['mic_asr_wav']).name} 等", flush=True)

    if call_rec is not None:
        rec_add(event="session_end", detail="finalize_scripted")
        call_rec.finalize(meta_end)
        print(f"# 已写入: {call_rec.path_transcript.name}", flush=True)
    elif meta_end.get("call_audio") and session_stem is not None:
        stub = Path(str(session_stem) + "_meta.json")
        stub.write_text(json.dumps(meta_end, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
