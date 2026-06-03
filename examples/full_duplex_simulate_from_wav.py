#!/usr/bin/env python3
"""Simulate full-duplex: mic streaming + optional local-playback window (barge-in demo).

Uses ``FireRedAsr2System.open_full_duplex_stream``. During ``begin_local_playback``, user
speech onsets emit ``barge_in``; ``segment_final`` includes ``during_local_playback``.

This stack does not include AEC; use echo-cancelled mic or expect false barge-in if speaker
audio leaks into the uplink.

Example::

  .venv/Scripts/python.exe examples/full_duplex_simulate_from_wav.py \\
    --wav_path path/to/utterance_16k_mono.wav --chunk_ms 120 --during_playback

快捷打断自测（默认用富场景已合成的用户句 u01.wav，需先 prepare wavs）::

  .venv/Scripts/python.exe examples/full_duplex_simulate_from_wav.py --demo_barge_in --device xpu --chunk_ms 80

急躁客户（完整）：首轮助手长说明 a01 → 客户抢话 u02 → 停播 → 短静音 → 助手换稿重讲 a2 → 可选客户回应::

  .venv/Scripts/python.exe examples/full_duplex_simulate_from_wav.py --impatient_barge_in --device xpu --chunk_ms 80 \\
    --lead_in_silence_ms 2000

  仅「抢话」无第二轮（旧版三阶段）:: 加 --impatient-no-retry

  默认写录音到 output/call_recordings；立体声 R=两轮助手参考与时间轴对齐；不需要时用 --no-call-audio
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

_EXAMPLES = Path(__file__).resolve().parent
_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _EXAMPLES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import stdio_utf8_windows  # noqa: E402

stdio_utf8_windows.apply_stdio_utf8()

import numpy as np
import soundfile as sf

os.environ.setdefault("PYTORCH_ENABLE_XPU_FALLBACK", "1")

from fireredasr2s.fireredasr2 import FireRedAsr2Config  # noqa: E402
from fireredasr2s.fireredasr2system import FireRedAsr2System, FireRedAsr2SystemConfig  # noqa: E402
from fireredasr2s.fireredlid import FireRedLidConfig  # noqa: E402
from fireredasr2s.fireredpunc import FireRedPuncConfig  # noqa: E402
from fireredasr2s.fireredvad import FireRedVadConfig  # noqa: E402
from fireredasr2s.firereddiar.audio import prepare_asr_stack_audio  # noqa: E402
from fireredasr2s.repo_dotenv import default_asr_device, load_repo_dotenv  # noqa: E402

_SCENARIO_WAV = _REPO / "examples" / "duplex_scenario_rich" / "wavs"


def _load_mono_16k(path: Path) -> np.ndarray:
    raw, sr_ = sf.read(str(path), dtype="int16")
    if raw.ndim > 1:
        raw = raw.mean(axis=1).astype(np.int16)
    pcm_, _sr2 = prepare_asr_stack_audio(raw, int(sr_))
    return pcm_.astype(np.int16)


def _resolve_wav_path(p: str) -> Path:
    q = Path(p)
    return q if q.is_file() else (_REPO / p).resolve()


def _build_impatient_retry_stereo_ref(
    *,
    lead_n: int,
    user_pcm: np.ndarray,
    post_tail_n: int,
    gap_n: int,
    a1: np.ndarray,
    a2: np.ndarray,
    ack_n: int,
) -> np.ndarray:
    """右声道：首轮 a1 与客户并行 → 打断后静音 → 间隔 → 第二轮 a2；与 mic 时间轴等长。"""
    lu = int(len(user_pcm))
    listen_n = int(len(a2))
    total = lead_n + lu + post_tail_n + gap_n + listen_n + ack_n
    R = np.zeros(total, dtype=np.int16)
    s1 = lead_n + lu
    s_listen = s1 + post_tail_n + gap_n
    # 首轮：与 lead+用户时段对齐铺 a1
    for i in range(min(s1, total)):
        R[i] = a1[i] if i < len(a1) else 0
    # s1 .. s_listen : 已为零（打断后停播 + 间隔）
    for j in range(listen_n):
        idx = s_listen + j
        if idx < total:
            R[idx] = a2[j]
    return R


def _load_save_call_audio():
    p = _REPO / "examples" / "full_duplex_voice_llm_tts.py"
    spec = importlib.util.spec_from_file_location("_fdv_duplex_sim", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod._save_call_session_audio


def _push_pcm_in_chunks(
    session: Any,
    pcm: np.ndarray,
    *,
    sample_rate: int,
    chunk_samples: int,
    record_bufs: tuple[list[np.ndarray], list[np.ndarray]] | None = None,
    assistant_tts_ref: np.ndarray | None = None,
    tts_timeline_pos: list[int] | None = None,
) -> None:
    for i in range(0, len(pcm), chunk_samples):
        chunk = pcm[i : i + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.concatenate(
                [chunk, np.zeros(chunk_samples - len(chunk), dtype=np.int16)]
            )
        if record_bufs is not None:
            mic_chunks, tts_chunks = record_bufs
            mic_chunks.append(chunk.copy())
            if (
                assistant_tts_ref is not None
                and tts_timeline_pos is not None
                and len(tts_timeline_pos) == 1
            ):
                pos = int(tts_timeline_pos[0])
                ref = assistant_tts_ref
                tts_buf = np.zeros(chunk_samples, dtype=np.int16)
                if pos < len(ref):
                    take = min(chunk_samples, len(ref) - pos)
                    tts_buf[:take] = ref[pos : pos + take]
                tts_chunks.append(tts_buf)
                tts_timeline_pos[0] = pos + chunk_samples
            else:
                tts_chunks.append(np.zeros(chunk_samples, dtype=np.int16))
        evs = session.push_microphone_pcm(chunk, sample_rate=sample_rate)
        for ev in evs:
            print(json.dumps(ev, ensure_ascii=False, default=str)[:2000], flush=True)


def main() -> None:
    load_repo_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--wav_path", type=str, default=None, help="16 kHz 单声道 int16 WAV（与 --demo_barge_in 二选一即可）")
    p.add_argument("--chunk_ms", type=int, default=200)
    p.add_argument("--models_root", type=str, default="pretrained_models")
    p.add_argument(
        "--device",
        type=str,
        default=default_asr_device(),
        help="ASR 等设备；默认 xpu；仅 FIRERED_ASR_DEVICE 可覆盖（不读 ASR_DEVICE）",
    )
    p.add_argument("--enable_lid", type=int, default=0)
    p.add_argument("--enable_punc", type=int, default=1)
    p.add_argument(
        "--during_playback",
        action="store_true",
        help="Treat entire replay as local TTS playing (demo barge_in + during_local_playback).",
    )
    p.add_argument(
        "--demo_barge_in",
        action="store_true",
        help="快捷：模拟助手侧正在播时用户开说；隐含 --during_playback；默认 wav 为 duplex_scenario_rich/wavs/u01.wav。",
    )
    p.add_argument(
        "--impatient_barge_in",
        action="store_true",
        help=(
            "急躁客户完整用例：首轮助手说明 → 客户抢话 → 停播 → 间隔 → 助手换稿重讲 → 可选客户回应；"
            "隐含 --during_playback；默认客户 wav=u02。加 --impatient-no-retry 则仅抢话三阶段。"
        ),
    )
    p.add_argument(
        "--impatient-no-retry",
        action="store_true",
        help="与 --impatient_barge_in 同用：不打断后的第二轮讲解，仅 lead_in + 客户句 + 尾静音。",
    )
    p.add_argument(
        "--lead_in_silence_ms",
        type=float,
        default=1600.0,
        help="与 --impatient_barge_in 配合：播报窗口内、用户开说前的上行静音时长（毫秒）。",
    )
    p.add_argument(
        "--post_user_tail_ms",
        type=float,
        default=600.0,
        help="与 --impatient_barge_in 配合：用户句结束后追加静音，便于 VAD 收尾（毫秒）。",
    )
    p.add_argument(
        "--post-interrupt-gap-ms",
        type=float,
        default=450.0,
        help="（默认启用重讲时）首轮 end_local_playback 后、第二轮 begin 前的双静音间隔（毫秒）。",
    )
    p.add_argument(
        "--assistant-wav-retry",
        type=str,
        default=None,
        help="第二轮助手讲解参考 WAV（默认 duplex_scenario_rich/wavs/a02.wav）。",
    )
    p.add_argument(
        "--customer-ack-wav",
        type=str,
        default=None,
        help="重讲结束后可选客户一句（如 u03.wav）；不设则不再推客户语音。",
    )
    p.add_argument(
        "--verbose_vad",
        action="store_true",
        help="Also print vad_speech_start events from the inner stream session.",
    )
    p.add_argument(
        "--call-audio",
        action="store_true",
        help="将推入 session 的上行 PCM 落盘（三文件命名同 full_duplex）；impatient 时默认右声道铺助手参考 WAV。",
    )
    p.add_argument(
        "--no-call-audio",
        action="store_true",
        help="覆盖 --impatient_barge_in 的默认录音行为，不落盘。",
    )
    p.add_argument(
        "--call-record-dir",
        type=str,
        default="output/call_recordings",
        help="录音输出目录（默认 output/call_recordings，与 output/logs 分离以免清日志误删）。",
    )
    p.add_argument(
        "--assistant-wav",
        type=str,
        default=None,
        help=(
            "首轮助手参考 WAV（16k 单声道优先）。impatient 且铺立体声时未指定则默认 wavs/a01.wav；"
            "完整用例下第二轮见 --assistant-wav-retry。"
        ),
    )
    p.add_argument(
        "--no-assistant-in-stereo",
        action="store_true",
        help="录音时强制右声道为静音（不铺助手参考）。",
    )
    args = p.parse_args()
    if args.demo_barge_in and args.impatient_barge_in:
        p.error("--demo_barge_in 与 --impatient_barge_in 请勿同时使用")
    if args.demo_barge_in or args.impatient_barge_in:
        args.during_playback = True
    if not args.wav_path:
        if args.demo_barge_in:
            args.wav_path = str(_REPO / "examples" / "duplex_scenario_rich" / "wavs" / "u01.wav")
        elif args.impatient_barge_in:
            args.wav_path = str(_REPO / "examples" / "duplex_scenario_rich" / "wavs" / "u02.wav")
        else:
            p.error("请指定 --wav_path，或使用 --demo_barge_in / --impatient_barge_in")

    want_call_audio = (bool(args.call_audio) or bool(args.impatient_barge_in)) and (
        not bool(args.no_call_audio)
    )

    root = Path(args.models_root)
    wav_path = Path(args.wav_path)
    pcm, sr = sf.read(str(wav_path), dtype="int16")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1).astype(np.int16)
    if int(sr) != 16000:
        print("Warning: expected 16 kHz mono wav; resampling via System stack on first push.", flush=True)

    impatient_retry = bool(args.impatient_barge_in) and not bool(args.impatient_no_retry)
    a1_pcm: np.ndarray | None = None
    a2_pcm: np.ndarray | None = None
    a1_res: str | None = None
    a2_res: str | None = None
    ack_pcm: np.ndarray | None = None
    ack_res: str | None = None

    if impatient_retry:
        pa1 = _resolve_wav_path(
            (args.assistant_wav or "").strip() or str(_SCENARIO_WAV / "a01.wav")
        )
        pa2 = _resolve_wav_path(
            (args.assistant_wav_retry or "").strip() or str(_SCENARIO_WAV / "a02.wav")
        )
        if pa1.is_file():
            a1_pcm = _load_mono_16k(pa1)
            a1_res = str(pa1.resolve())
        else:
            print(f"# 警告：未找到首轮助手参考 {pa1}", flush=True)
        if pa2.is_file():
            a2_pcm = _load_mono_16k(pa2)
            a2_res = str(pa2.resolve())
        else:
            print(f"# 警告：未找到第二轮助手参考 {pa2}，将跳过重讲段。", flush=True)
        ack_arg = (args.customer_ack_wav or "").strip()
        if ack_arg:
            pak = _resolve_wav_path(ack_arg)
            if pak.is_file():
                ack_pcm = _load_mono_16k(pak)
                ack_res = str(pak.resolve())
            else:
                print(f"# 警告：未找到客户回应参考 {pak}，跳过该段。", flush=True)

    assistant_tts_ref: np.ndarray | None = None
    assistant_wav_resolved: str | None = None
    if impatient_retry and a1_res:
        assistant_wav_resolved = a1_res
    if want_call_audio and not args.no_assistant_in_stereo:
        if impatient_retry and a1_pcm is not None:
            a1b = a1_pcm
            a2b = a2_pcm if a2_pcm is not None else np.zeros(0, dtype=np.int16)
            lead_nn = max(int(int(sr) * float(args.lead_in_silence_ms) / 1000.0), 0)
            post_tail_nn = max(int(int(sr) * float(args.post_user_tail_ms) / 1000.0), 0)
            gap_nn = max(int(int(sr) * float(args.post_interrupt_gap_ms) / 1000.0), 0)
            ack_nn = int(len(ack_pcm)) if ack_pcm is not None else 0
            assistant_tts_ref = _build_impatient_retry_stereo_ref(
                lead_n=lead_nn,
                user_pcm=pcm,
                post_tail_n=post_tail_nn,
                gap_n=gap_nn,
                a1=a1b,
                a2=a2b,
                ack_n=ack_nn,
            )
        elif args.impatient_barge_in and not impatient_retry:
            ap = _resolve_wav_path(
                (args.assistant_wav or "").strip() or str(_SCENARIO_WAV / "a01.wav")
            )
            if ap.is_file():
                assistant_tts_ref = _load_mono_16k(ap)
                assistant_wav_resolved = str(ap.resolve())
            else:
                print(f"# 警告：未找到助手参考 {ap}，立体声右声道将为静音。", flush=True)
        elif args.impatient_barge_in:
            # impatient_retry 但缺 a01：已在上方警告，不铺参考、不落入通用 --assistant-wav
            pass
        else:
            aw = (args.assistant_wav or "").strip()
            if aw:
                ap = _resolve_wav_path(aw)
                if ap.is_file():
                    assistant_tts_ref = _load_mono_16k(ap)
                    assistant_wav_resolved = str(ap.resolve())
                else:
                    print(f"# 警告：未找到助手参考 {ap}。", flush=True)

    asr_cfg = FireRedAsr2Config(use_gpu=args.device != "cpu", device=args.device, return_timestamp=False)
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
        enable_lid=bool(args.enable_lid),
        enable_punc=bool(args.enable_punc),
        enable_diarization=False,
        stream_vad_use_gpu=False,
    )
    sys_m = FireRedAsr2System(cfg)
    session = sys_m.open_full_duplex_stream(uttid_prefix="duplex", verbose_vad=args.verbose_vad)

    chunk_samples = max(int(16 * args.chunk_ms), 160)
    session_stem: Path | None = None
    save_call_audio = None
    record_bufs: tuple[list[np.ndarray], list[np.ndarray]] | None = None
    if want_call_audio:
        save_call_audio = _load_save_call_audio()
        rec_dir = Path(args.call_record_dir)
        if not rec_dir.is_absolute():
            rec_dir = (_REPO / rec_dir).resolve()
        else:
            rec_dir = rec_dir.resolve()
        tag = (
            "impatient_retry"
            if impatient_retry
            else ("impatient" if args.impatient_barge_in else "wav_sim")
        )
        session_stem = rec_dir / (
            f"call_record_duplex_sim_{tag}_"
            f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
        )
        record_bufs = ([], [])

    tts_timeline_pos: list[int] | None = [0] if assistant_tts_ref is not None else None

    def _push(
        arr: np.ndarray,
    ) -> None:
        _push_pcm_in_chunks(
            session,
            arr,
            sample_rate=int(sr),
            chunk_samples=chunk_samples,
            record_bufs=record_bufs,
            assistant_tts_ref=assistant_tts_ref,
            tts_timeline_pos=tts_timeline_pos,
        )

    try:
        if impatient_retry:
            lead_n = max(int(int(sr) * float(args.lead_in_silence_ms) / 1000.0), 0)
            post_tail_n = max(int(int(sr) * float(args.post_user_tail_ms) / 1000.0), 0)
            gap_n = max(int(int(sr) * float(args.post_interrupt_gap_ms) / 1000.0), 0)
            a2_listen = a2_pcm if a2_pcm is not None else np.zeros(0, dtype=np.int16)
            n_listen = int(len(a2_listen))

            print(
                "# impatient_barge_in（完整用例）：首轮说明 → 客户打断 → 停播 → 静音间隔 → 助手换稿重讲"
                + (" → 客户回应" if ack_pcm is not None else "")
                + "。",
                flush=True,
            )
            if want_call_audio and assistant_tts_ref is not None:
                print(
                    "# 场景（录音）：L=客户上行；R=首轮 a01 + 停播留白 + 二轮 a02（与客户时间轴对齐）。",
                    flush=True,
                )
            elif want_call_audio:
                print("# 场景（录音）：未生成助手双轨参考（见上方警告）。", flush=True)

            print(
                f"# 阶段 A：推 {args.lead_in_silence_ms:g} ms 静音（对端首轮说明中，客户尚未开口）…",
                flush=True,
            )
            session.begin_local_playback(playback_id="assistant_round1_explain")
            _push(np.zeros(lead_n, dtype=np.int16))
            print("# 阶段 B：客户抢插 — 推用户句（应出现 barge_in + segment_final）…", flush=True)
            _push(pcm)
            print(
                f"# 阶段 C：用户句后 {args.post_user_tail_ms:g} ms 静音（首轮播报已打断，上行收尾）…",
                flush=True,
            )
            _push(np.zeros(post_tail_n, dtype=np.int16))
            session.end_local_playback()

            print(
                f"# 阶段 D：推 {args.post_interrupt_gap_ms:g} ms 双静音（对端准备换稿重讲）…",
                flush=True,
            )
            _push(np.zeros(gap_n, dtype=np.int16))

            if n_listen > 0:
                print("# 阶段 E：第二轮本地播报窗口 — 客户静听助手重讲（上行静音，长度对齐 a02）…", flush=True)
                session.begin_local_playback(playback_id="assistant_round2_retry")
                _push(a2_listen)
                session.end_local_playback()
            else:
                print("# 阶段 E：跳过（未加载第二轮助手 WAV）。", flush=True)

            if ack_pcm is not None:
                print("# 阶段 F：客户听完后的短回应 …", flush=True)
                _push(ack_pcm)

        elif args.impatient_barge_in:
            print(
                "# impatient_barge_in（简版 --impatient-no-retry）：仅抢话三阶段。",
                flush=True,
            )
            session.begin_local_playback(playback_id="impatient_assistant_tts")
            lead_n = max(int(int(sr) * float(args.lead_in_silence_ms) / 1000.0), 0)
            print(
                f"# 阶段 A：推 {args.lead_in_silence_ms:g} ms 静音（对端仍在说话，客户尚未开口）…",
                flush=True,
            )
            _push(np.zeros(lead_n, dtype=np.int16))
            print("# 阶段 B：客户抢插 — 推入用户句 WAV（此处应出现 barge_in + segment_final）…", flush=True)
            _push(pcm)
            if float(args.post_user_tail_ms) > 0:
                tail_n = max(int(int(sr) * float(args.post_user_tail_ms) / 1000.0), 0)
                print(
                    f"# 阶段 C：用户句后追加 {args.post_user_tail_ms:g} ms 静音，便于分段收尾…",
                    flush=True,
                )
                _push(np.zeros(tail_n, dtype=np.int16))
            session.end_local_playback()
            if want_call_audio and assistant_tts_ref is not None:
                print(
                    f"# 场景（录音）：左=客户上行，右=首轮助手参考「{Path(assistant_wav_resolved or '').name}」。",
                    flush=True,
                )
            elif want_call_audio:
                print(
                    "# 场景（录音）：未加载助手参考，右声道为静音。",
                    flush=True,
                )

        elif args.demo_barge_in:
            session.begin_local_playback(playback_id="demo_local_tts")
            print(
                "# demo_barge_in：已 begin_local_playback；推流中若出现 vad 起讲应打印 barge_in。",
                flush=True,
            )
            _push(pcm)
            session.end_local_playback()
        elif args.during_playback:
            session.begin_local_playback(playback_id="during_playback")
            _push(pcm)
            session.end_local_playback()
        else:
            _push(pcm)

        for ev in session.finalize():
            print(json.dumps(ev, ensure_ascii=False, default=str)[:2000], flush=True)

        if (
            want_call_audio
            and save_call_audio is not None
            and session_stem is not None
            and record_bufs is not None
        ):
            mic_c, tts_c = record_bufs
            meta = save_call_audio(session_stem, mic_c, tts_c, int(sr))
            if assistant_tts_ref is not None:
                note_extra = (
                    "duplex_simulate_from_wav：mic_asr=客户上行(入 ASR)；tts_reference=与 session "
                    "同一时间轴对齐的助手参考波形(模拟下行播报，非空气回采)；stereo L/R 同上。"
                    "编排仍仅 begin_local_playback，不向声卡推音。"
                    + (
                        " impatient_retry：右轨含首轮 a01、停播留白、二轮 a02（及 ack 段静音对齐）。"
                        if impatient_retry
                        else ""
                    )
                )
            else:
                note_extra = (
                    "duplex_simulate_from_wav：未加载助手参考时 tts 轨为静音。"
                    "mic_asr 与 push_microphone_pcm 块序列一致。"
                )
            meta = dict(meta)
            meta["note"] = f"{meta.get('note', '')} {note_extra}".strip()
            impatient = bool(args.impatient_barge_in)
            meta["duplex_sim_scenario"] = {
                "mode": "impatient_barge_in" if impatient else "wav_sim",
                "impatient_retry": bool(impatient_retry),
                "impatient_no_retry": bool(args.impatient_no_retry) if impatient else None,
                "customer_wav": str(wav_path.resolve()),
                "assistant_ref_wav": assistant_wav_resolved,
                "assistant_ref_wav_round2": a2_res if impatient_retry else None,
                "customer_ack_wav": ack_res,
                "post_interrupt_gap_ms": float(args.post_interrupt_gap_ms)
                if impatient_retry
                else None,
                "lead_in_silence_ms": float(args.lead_in_silence_ms) if impatient else None,
                "post_user_tail_ms": float(args.post_user_tail_ms) if impatient else None,
            }
            meta_path = Path(str(session_stem) + "_call_audio.json")
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"# 通话录音 mic_asr: {Path(meta['mic_asr_wav']).resolve()}", flush=True)
            print(f"# 通话录音 tts_ref: {Path(meta['tts_reference_wav']).resolve()}", flush=True)
            if meta.get("stereo_micL_ttsR_wav"):
                print(
                    f"# 通话录音 stereo: {Path(meta['stereo_micL_ttsR_wav']).resolve()}",
                    flush=True,
                )
            print(f"# 通话录音索引 JSON: {meta_path.resolve()}", flush=True)
    finally:
        while session.local_playback_active:
            session.end_local_playback()


if __name__ == "__main__":
    main()
