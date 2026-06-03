#!/usr/bin/env python3
"""语音输入 → 流式 ASR → 大模型（OpenAI 兼容 HTTP）→ TTS → 全双工外放。

同一块 ``sounddevice`` duplex 回调内：播放参考与麦克风流对齐，可选 NLMS 减轻回声。
编排层 **不** 在音频线程调用 LLM/TTS；由后台线程生成回复 PCM，经锁交换到回调播放。

依赖::

  pip install sounddevice pyttsx3
  # 或 edge-tts + 本机 ffmpeg

环境变量（推荐写入仓库根 ``.env``，启动时自动加载，不覆盖已在 shell 中设置的变量）::

  # Base URL 优先级：LLM_BASE_URL > OLLAMA_BASE_URL > OPENAI_BASE_URL
  LLM_BASE_URL=https://api.openai.com/v1
  LLM_API_KEY=sk-...
  # 模型优先级：LLM_MODEL_ID > LLM_MODEL > OLLAMA_MODEL
  LLM_MODEL_ID=gpt-4o-mini
  LLM_TIMEOUT=120

本机 Ollama（OpenAI 兼容 ``/v1``）::

  LLM_BASE_URL=http://localhost:11434/v1
  LLM_MODEL_ID=llama3.2
  # 未设置有效密钥时，对 11434 自动使用占位符 ollama

PowerShell 示例（Ollama）::

  $env:OLLAMA_BASE_URL = "http://localhost:11434/v1"
  $env:LLM_MODEL = "llama3.2"
  # 也可: $env:OLLAMA_MODEL = "llama3.2"
  .venv\\Scripts\\python.exe examples\\full_duplex_voice_llm_tts.py --device xpu \\
    --session-seconds 120 --greeting "你好，请直接说你的问题。"

示例（OpenAI）::

  .venv\\Scripts\\python.exe examples\\full_duplex_voice_llm_tts.py --device xpu \\
    --session-seconds 120 --greeting "你好，请直接说你的问题。"

仓库根 ``.env`` 与 HelloAgents 等统一：由 ``fireredasr2s.repo_dotenv.load_repo_dotenv()`` 加载。
可选 ``pip install python-dotenv`` 以完整支持 ``.env`` 语法；未安装时使用内置行解析。
ASR ``--device`` 默认 ``xpu``；仅当设置 ``FIRERED_ASR_DEVICE`` 时覆盖（不读通用 ``ASR_DEVICE``，避免 .env 误设为 cpu）。

急躁客户·真 LLM（首轮长说明 → 抢话 ``barge_in`` → 二轮短讲）::

  .venv\\Scripts\\python.exe examples\\full_duplex_voice_llm_tts.py --impatient-barge-in \\
    --device xpu --session-seconds 120 --call-audio

  **首轮 LLM** 在打开声卡 **之前** 用 ``--impatient-first-seed`` 作为 **user 文本** 调 API（不是麦克风）。
  **第二条及以后 LLM** 仅在 ASR 出 ``segment_final`` 后由 worker 调用；无人说话或会话过短则不会再请求 LLM。
  LLM/密钥读仓库根 ``.env``；勿依赖短 ``--greeting``（本模式会自动跳过）。

说明：生产环境外放请优先系统/WebRTC AEC 或耳机；见 docs/AEC_INTEGRATION_BOUNDARY.md。
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import queue
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
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

from fireredasr2s.repo_dotenv import default_asr_device, load_repo_dotenv, strip_env_quotes  # noqa: E402

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore[assignment]

from fireredasr2s.duplex import NlmsMonoAec  # noqa: E402
from fireredasr2s.fireredasr2 import FireRedAsr2Config  # noqa: E402
from fireredasr2s.fireredasr2system import FireRedAsr2System, FireRedAsr2SystemConfig  # noqa: E402
from fireredasr2s.firereddiar.audio import prepare_asr_stack_audio  # noqa: E402
from fireredasr2s.fireredlid import FireRedLidConfig  # noqa: E402
from fireredasr2s.fireredpunc import FireRedPuncConfig  # noqa: E402
from fireredasr2s.fireredvad import FireRedVadConfig  # noqa: E402


def _float_to_i16_mono(x: np.ndarray) -> np.ndarray:
    return (np.clip(x.astype(np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)


def _i16_to_float_mono(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


# 客服女声 / 客户男声（Edge 神经语音）。pyttsx3 无同名音色，按性别在 Windows SAPI 列表中匹配中文声线。
EDGE_VOICE_ASSISTANT_FEMALE_DEFAULT = "zh-CN-XiaoxiaoNeural"
EDGE_VOICE_CUSTOMER_MALE_DEFAULT = "zh-CN-YunxiNeural"


def _infer_pyttsx3_gender(edge_voice: str) -> str:
    """由 Edge 音名关键字推断 pyttsx3 侧应选男/女（未知则默认女声，偏助手场景）。"""
    ev = (edge_voice or "").lower()
    for t in ("yunxi", "yunyang", "kangkang", "yundeng", " male"):
        if t in ev:
            return "male"
    for t in ("xiaoxiao", "xiaoyi", "huihui", "yaoyao", "ling", " female"):
        if t in ev:
            return "female"
    return "female"


def _pyttsx3_apply_gender(engine: Any, gender: str) -> None:
    """在 pyttsx3 引擎上设置 voice id；仅 Windows SAPI 常见中文名做启发式匹配。"""
    g = (gender or "").strip().lower()
    if g not in ("male", "female"):
        return
    try:
        voices = engine.getProperty("voices")
    except Exception:
        return
    if not voices:
        return

    def blob(v: Any) -> str:
        return f"{getattr(v, 'id', '')} {getattr(v, 'name', '')}".lower()

    male_hints = ("kangkang", "yunxi", "yunyang", "yundeng", " male", "男声", "male")
    female_hints = ("xiaoxiao", "xiaoyi", "huihui", "yaoyao", " female", "女声", "female")

    def is_zh_cn(b: str) -> bool:
        return "chinese" in b or "中文" in b or "mandarin" in b or "简体" in b

    hints = male_hints if g == "male" else female_hints
    best: tuple[int, int, str] | None = None
    for vi, v in enumerate(voices):
        b = blob(v)
        if not is_zh_cn(b):
            continue
        for pri, hint in enumerate(hints):
            if hint in b:
                vid = str(getattr(v, "id", "") or "")
                if not vid:
                    continue
                cand = (pri, vi, vid)
                if best is None or cand < best:
                    best = cand
                break
    if best is not None:
        engine.setProperty("voice", best[2])
        return

    # 任意中文语音兜底（无法区分性别时仍比默认英文自然）
    for v in voices:
        b = blob(v)
        if is_zh_cn(b):
            vid = str(getattr(v, "id", "") or "")
            if vid:
                engine.setProperty("voice", vid)
                return


def synthesize_tts_16k_int16(
    text: str,
    engine: str,
    edge_voice: str,
    *,
    edge_rate: str = "+0%",
    edge_pitch: str = "+0Hz",
    pyttsx3_gender: str | None = None,
) -> np.ndarray:
    text = (text or "").strip()
    if not text:
        return np.zeros(0, dtype=np.int16)

    if engine == "pyttsx3":
        try:
            import pyttsx3  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError("pip install pyttsx3") from e
        import soundfile as sf

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            eng = pyttsx3.init()
            g = pyttsx3_gender if pyttsx3_gender in ("male", "female") else _infer_pyttsx3_gender(edge_voice)
            _pyttsx3_apply_gender(eng, g)
            eng.save_to_file(text, path)
            eng.runAndWait()
            pcm, sr = sf.read(path, dtype="float32")
            if pcm.ndim > 1:
                pcm = pcm.mean(axis=1)
            pcm_i16 = (pcm * 32767.0).clip(-32768, 32767).astype(np.int16)
            out, sr2 = prepare_asr_stack_audio(pcm_i16, int(sr))
            if int(sr2) != 16000:
                raise RuntimeError(f"expected 16 kHz after stack, got {sr2}")
            return out
        finally:
            Path(path).unlink(missing_ok=True)

    if engine == "edge":
        try:
            import asyncio

            import edge_tts  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError("pip install edge-tts") from e

        async def _save_mp3(out_mp3: str) -> None:
            com = edge_tts.Communicate(text, edge_voice, rate=edge_rate, pitch=edge_pitch)
            await com.save(out_mp3)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f_mp3:
            mp3_path = f_mp3.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_wav:
            wav_path = f_wav.name
        try:
            asyncio.run(_save_mp3(mp3_path))
            import subprocess

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    mp3_path,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-f",
                    "wav",
                    wav_path,
                ],
                check=True,
                capture_output=True,
            )
            import soundfile as sf

            pcm, sr = sf.read(wav_path, dtype="int16")
            if pcm.ndim > 1:
                pcm = pcm.mean(axis=1).astype(np.int16)
            out, sr2 = prepare_asr_stack_audio(pcm, int(sr))
            if int(sr2) != 16000:
                raise RuntimeError(f"expected 16 kHz after stack, got {sr2}")
            return out
        finally:
            Path(mp3_path).unlink(missing_ok=True)
            Path(wav_path).unlink(missing_ok=True)

    raise ValueError(f"unknown tts engine: {engine}")


def _default_llm_base_url() -> str:
    """Resolve OpenAI-compatible API base (trailing ``/v1`` optional upstream; we append ``/chat/completions``)."""
    for k in ("LLM_BASE_URL", "OLLAMA_BASE_URL", "OPENAI_BASE_URL"):
        v = (os.environ.get(k) or "").strip()
        if v:
            return strip_env_quotes(v).rstrip("/")
    return "https://api.openai.com/v1"


def _looks_like_local_ollama_openai_compat(base_url: str) -> bool:
    u = base_url.lower()
    return "127.0.0.1:11434" in u or "localhost:11434" in u or ":11434/" in u or u.endswith(":11434")


def _resolve_llm_api_key(base_url: str, explicit: str) -> str:
    """Ollama ``/v1`` 常不要求有效密钥；未配置时使用占位符以便保留 ``Authorization`` 头格式。"""
    raw = (explicit or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    key = strip_env_quotes(raw)
    if key.upper() in ("YOUR-API-KEY", "CHANGE_ME", "CHANGEME", "SK-YOUR-KEY-HERE"):
        key = ""
    if key:
        return key
    if _looks_like_local_ollama_openai_compat(base_url):
        return "ollama"
    return ""


def _default_llm_model() -> str:
    for k in ("LLM_MODEL_ID", "LLM_MODEL", "OLLAMA_MODEL"):
        v = strip_env_quotes((os.environ.get(k) or "").strip())
        if v:
            return v
    return "gpt-4o-mini"


def _default_llm_timeout_s() -> float:
    v = (os.environ.get("LLM_TIMEOUT") or "").strip()
    if not v:
        return 120.0
    try:
        return float(v)
    except ValueError:
        return 120.0


def _copy_messages_for_trace(
    messages: list[dict[str, str]],
    max_content_chars: int,
) -> list[dict[str, str]]:
    """深拷贝 messages；可选截断每条 content 以控制 meta 体积。"""
    out: list[dict[str, str]] = copy.deepcopy(messages)
    lim = int(max_content_chars)
    if lim > 0:
        for m in out:
            c = m.get("content")
            if isinstance(c, str) and len(c) > lim:
                m["content"] = c[:lim] + "…[已截断]"
    return out


def openai_compatible_chat(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout_s: float = 120.0,
    request_trace: list[dict[str, Any]] | None = None,
    trace_tag: str | None = None,
    trace_max_content_chars: int = 0,
    trace_extra: dict[str, Any] | None = None,
) -> str:
    """POST /chat/completions，返回 assistant 文本（非流式）。"""
    url = base_url.rstrip("/") + "/chat/completions"
    body_obj = {
        "model": model,
        "messages": messages,
        "temperature": 0.6,
    }
    if request_trace is not None:
        rec: dict[str, Any] = {
            "ts": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "tag": trace_tag,
            "method": "POST",
            "path": "/chat/completions",
            "base_url": base_url.rstrip("/"),
            "timeout_s": float(timeout_s),
            "note": "与实际上传 JSON 一致；不含 Authorization；temperature 固定 0.6。",
            "body": {
                "model": model,
                "temperature": 0.6,
                "messages": _copy_messages_for_trace(messages, trace_max_content_chars),
            },
        }
        if trace_extra:
            rec["extra"] = dict(trace_extra)
        request_trace.append(rec)
    body = json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e
    choices = raw.get("choices") or []
    if not choices:
        raise RuntimeError(f"unexpected response: {raw!r}")
    msg = choices[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    return content


class CallSessionRecord:
    """会话级通话时间线：写 JSON / JSONL，并生成可读「还原」文本。"""

    def __init__(self, stem: Path, *, write_jsonl: bool = True) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self.path_json = stem.with_suffix(".json")
        self.path_jsonl = stem.with_suffix(".jsonl") if write_jsonl else None
        self.path_transcript = Path(str(stem) + "_还原.txt")
        if self.path_jsonl is not None:
            self.path_jsonl.parent.mkdir(parents=True, exist_ok=True)
            self.path_jsonl.write_text("", encoding="utf-8")

    def add(self, **fields: Any) -> None:
        ev = {
            "ts_iso": datetime.datetime.now().isoformat(timespec="milliseconds"),
            **fields,
        }
        with self._lock:
            self._events.append(ev)
            if self.path_jsonl is not None:
                with self.path_jsonl.open("a", encoding="utf-8") as jf:
                    jf.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def event_counts(self) -> dict[str, int]:
        """当前已记录事件按 ``event`` 字段计数（不含尚未 ``add`` 的项）。"""
        with self._lock:
            m: dict[str, int] = {}
            for ev in self._events:
                k = str(ev.get("event") or "")
                m[k] = m.get(k, 0) + 1
            return m

    def finalize(self, meta: dict[str, Any]) -> None:
        doc = {"meta": meta, "events": self._events}
        self.path_json.parent.mkdir(parents=True, exist_ok=True)
        self.path_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = self._render_transcript(meta)
        self.path_transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _render_transcript(self, meta: dict[str, Any]) -> list[str]:
        title = "通话记录还原"
        out: list[str] = [
            title,
            "=" * max(len(title), 24),
            f"开始: {meta.get('started_at', '')}",
            f"设备: {meta.get('asr_device', '')}  |  LLM: {meta.get('llm_model', '')}  |  {meta.get('llm_base_url', '')}",
            f"会话时长(秒): {meta.get('session_seconds', '')}",
        ]
        if meta.get("transcript_clock") == "scenario":
            out.append("时间轴: 会话内相对秒 (+t)，由脚本化回放写入，非墙钟大段静音")
        if meta.get("scenario_id"):
            out.append(f"场景: {meta.get('scenario_id')}")
        if meta.get("duplex_scenario_voices"):
            out.append(str(meta.get("duplex_scenario_voices")))
        cf = meta.get("conversation_flags")
        if isinstance(cf, dict) and cf:
            out.append(
                f"话轮摘要: barge_in={cf.get('barge_in_events', 0)}  "
                f"user_asr={cf.get('user_asr_events', 0)}  "
                f"assistant_llm={cf.get('assistant_llm_events', 0)}"
            )
            if meta.get("impatient_barge_in") and int(cf.get("barge_in_events", 0) or 0) == 0:
                out.append(
                    "说明: 未出现打断与用户 ASR 成句，故无二轮对话；"
                    "录音 L=上行麦、R=TTS 数字参考，不是「多人分轨」预混。"
                )
        out.append("")
        use_sc = meta.get("transcript_clock") == "scenario"
        for ev in self._events:
            ts = ev.get("ts_iso", "")
            tshort = ts[11:23] if len(ts) >= 23 else ts
            tsc = ev.get("t_scenario_s")
            tick = f"+{float(tsc):.2f}s" if use_sc and tsc is not None else tshort
            turn = ev.get("turn_index")
            turn_prefix = f"回合{int(turn)} " if turn is not None else ""
            et = ev.get("event")
            if et == "user_asr":
                main = (ev.get("text") or "").strip()
                hyp = (ev.get("asr_hypothesis") or "").strip()
                retry_tag = "〔打断后重讲·ASR〕" if ev.get("impatient_retry_turn") else ""
                if hyp and hyp != main:
                    out.append(
                        f"[{tick}] {turn_prefix}用户{retry_tag}（稿，与录音一致）: {main} 〔ASR 识别:{hyp}〕"
                    )
                else:
                    out.append(
                        f"[{tick}] {turn_prefix}用户{retry_tag}（稿，与录音一致）: {main}"
                    )
            elif et == "impatient_round1_seed":
                out.append(f"[{tick}] 「首轮长说明」LLM 引导: {ev.get('text', '')}")
            elif et == "assistant_llm":
                src = ev.get("source", "llm")
                if src == "canned_wav":
                    tag = "（预置 TTS·女声稿）"
                elif src == "impatient_round1":
                    tag = "〔首轮长说明·LLM〕"
                else:
                    tag = ""
                out.append(f"[{tick}] {turn_prefix}助手{tag}: {ev.get('text', '')}")
            elif et == "barge_in":
                out.append(f"[{tick}] 【打断】用户起讲，停止播放 (playback_id={ev.get('playback_id', '')})")
            elif et == "tts_start":
                tx = (ev.get("text") or "").strip()
                preview = (tx[:120] + "…") if len(tx) > 120 else tx
                seg = ev.get("segment", "assistant")
                if seg == "greeting":
                    who = "系统(问候)"
                elif seg == "assistant_round1":
                    who = "助手播报·首轮长说明"
                else:
                    who = "助手播报"
                vnote = (ev.get("voice_note") or "").strip()
                vtag = f" {vnote}" if vnote else ""
                out.append(
                    f"[{tick}] ▶ {who}{vtag} {ev.get('playback_id', '')} "
                    f"({ev.get('samples', 0)} samples) {preview}"
                )
            elif et == "tts_end":
                out.append(f"[{tick}] ■ 播报结束 {ev.get('playback_id', '')}")
            elif et == "tts_abort":
                out.append(f"[{tick}] ✕ 播报中断 {ev.get('playback_id', '')} ({ev.get('reason', '')})")
            elif et == "llm_error":
                out.append(f"[{tick}] 【LLM 异常】 {ev.get('detail', '')}")
            elif et == "reply_discarded":
                out.append(f"[{tick}] 【已丢弃回复】打断导致本轮不播报")
            elif et == "session_end":
                out.append(f"[{tick}] 会话结束: {ev.get('detail', '')}")
        out.append("")
        out.append(f"完整时间线: {self.path_json.name}")
        if self.path_jsonl is not None:
            out.append(f"流式行: {self.path_jsonl.name}")
        ca = meta.get("call_audio") or {}
        if isinstance(ca, dict) and ca:
            out.append("")
            out.append("通话录音还原（WAV，与上表同一时间轴按块对齐）")
            for k in ("mic_asr_wav", "tts_reference_wav", "stereo_micL_ttsR_wav"):
                v = ca.get(k)
                if v:
                    out.append(f"  - {k}: {v}")
            note = ca.get("note")
            if note:
                out.append(f"  说明: {note}")
        return out


def _save_call_session_audio(
    stem: Path,
    mic_chunks: list[np.ndarray],
    tts_chunks: list[np.ndarray],
    sample_rate: int,
) -> dict[str, Any]:
    """将回调内按块采集的上/下行 PCM 落盘；块顺序与时间线一致。"""
    import soundfile as sf

    stem.parent.mkdir(parents=True, exist_ok=True)
    mic = np.concatenate(mic_chunks) if mic_chunks else np.zeros(0, dtype=np.int16)
    tts = np.concatenate(tts_chunks) if tts_chunks else np.zeros(0, dtype=np.int16)
    p_mic = Path(str(stem) + "_mic_asr.wav")
    p_tts = Path(str(stem) + "_tts_ref.wav")
    p_lr = Path(str(stem) + "_stereo_micL_ttsR.wav")
    sf.write(str(p_mic), mic, sample_rate, subtype="PCM_16")
    sf.write(str(p_tts), tts, sample_rate, subtype="PCM_16")
    n = max(len(mic), len(tts))
    if n > 0:
        mic_p = np.pad(mic.astype(np.int16), (0, n - len(mic)))
        tts_p = np.pad(tts.astype(np.int16), (0, n - len(tts)))
        stereo = np.stack([mic_p, tts_p], axis=1)
        sf.write(str(p_lr), stereo, sample_rate, subtype="PCM_16")
    return {
        "mic_asr_wav": str(p_mic.resolve()),
        "tts_reference_wav": str(p_tts.resolve()),
        "stereo_micL_ttsR_wav": str(p_lr.resolve()) if n > 0 else "",
        "sample_rate_hz": sample_rate,
        "mic_samples": int(len(mic)),
        "tts_ref_samples": int(len(tts)),
        "note": (
            "立体声 L 与单声道 mic：由 duplex_scripted_engine.feed_duplex_chunk 采集时为「脚本/仿真送入的上行麦克」原始 int16，"
            "R 为客服 TTS 数字参考；流式 ASR 实际吃的是 AEC 后波形，未在此重复落盘。"
            "其他入口若自行传入采集块，以各脚本为准。L/R 按样本对齐（不足侧补零）。"
        ),
    }


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
class _DuplexRuntime:
    lock: threading.Lock
    session: Any
    aec: NlmsMonoAec | None
    sr: int
    print_events: bool
    tts_i16: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    tts_pos: int = 0
    pending_tts: np.ndarray | None = None
    pending_tts_caption: str = ""
    in_playback: bool = False
    generation: int = 0
    playback_serial: int = 0
    current_playback_id: str | None = None
    call_rec: CallSessionRecord | None = None
    next_tts_segment: str = "assistant"
    record_call_audio: bool = False
    pending_voice_note: str = ""
    impatient_mode: bool = False
    impatient_retry_pending: bool = False


def _stop_playback_locked(rt: _DuplexRuntime) -> None:
    if rt.call_rec is not None and rt.in_playback and rt.current_playback_id:
        rt.call_rec.add(
            event="tts_abort",
            playback_id=rt.current_playback_id,
            reason="barge_in_or_reset",
        )
    rt.pending_tts = None
    rt.pending_tts_caption = ""
    rt.tts_i16 = np.zeros(0, dtype=np.int16)
    rt.tts_pos = 0
    if rt.in_playback:
        rt.session.end_local_playback()
        rt.in_playback = False
    rt.current_playback_id = None


def _try_start_pending_playback(rt: _DuplexRuntime) -> None:
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
        anchor_wallclock_ms=int(time.time() * 1000),
    )
    rt.in_playback = True
    rt.current_playback_id = pid
    cap = (rt.pending_tts_caption or "").strip()
    rt.pending_tts_caption = ""
    if rt.call_rec is not None:
        seg = rt.next_tts_segment
        rt.next_tts_segment = "assistant"
        vn = (rt.pending_voice_note or "").strip()
        rt.pending_voice_note = ""
        rt.call_rec.add(
            event="tts_start",
            segment=seg,
            playback_id=pid,
            text=cap,
            samples=int(len(rt.tts_i16)),
            voice_note=vn,
        )
    print(f"# begin_local_playback {pid} samples={len(rt.tts_i16)}", flush=True)


def _finish_if_tts_done(rt: _DuplexRuntime) -> None:
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
    if rt.call_rec is not None and done_pid:
        rt.call_rec.add(event="tts_end", playback_id=done_pid)
    rt.current_playback_id = None
    print("# end_local_playback()", flush=True)


def main() -> None:
    if sd is None:
        print("Install sounddevice: pip install sounddevice", file=sys.stderr)
        sys.exit(1)

    if load_repo_dotenv(override=False):
        print("# 已合并仓库根 .env（不覆盖 shell 中已设置的变量）", flush=True)

    p = argparse.ArgumentParser(
        description="Full-duplex: mic ASR → OpenAI-compatible LLM → TTS",
    )
    p.add_argument("--models_root", type=str, default="pretrained_models")
    p.add_argument(
        "--device",
        type=str,
        default=default_asr_device(),
        help="ASR 等设备；默认 xpu；仅环境变量 FIRERED_ASR_DEVICE 可覆盖（不读 ASR_DEVICE）",
    )
    p.add_argument("--session-seconds", type=float, default=180.0)
    p.add_argument("--greeting", type=str, default="你好，请讲话，说完停顿一下。")
    p.add_argument("--no-greeting", action="store_true")
    p.add_argument("--system-prompt", type=str, default="你是中文语音助手，回答简短清晰，适合语音播报。")
    p.add_argument(
        "--llm-base-url",
        type=str,
        default=_default_llm_base_url(),
        help="OpenAI 兼容 Base URL；环境变量 LLM_BASE_URL / OLLAMA_BASE_URL / OPENAI_BASE_URL（优先级递减）",
    )
    p.add_argument(
        "--llm-model",
        type=str,
        default=_default_llm_model(),
        help="模型名；.env：LLM_MODEL_ID / LLM_MODEL / OLLAMA_MODEL",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default="",
        help="默认读 LLM_API_KEY / OPENAI_API_KEY；本机 Ollama(11434) 可省略",
    )
    p.add_argument(
        "--llm-timeout",
        type=float,
        default=_default_llm_timeout_s(),
        help="秒；默认读环境变量 LLM_TIMEOUT",
    )
    p.add_argument("--max-turns", type=int, default=12, help="保留最近 N 轮 user+assistant（不含 system）")
    p.add_argument("--tts-engine", type=str, choices=("pyttsx3", "edge"), default="pyttsx3")
    p.add_argument("--edge-voice", type=str, default=EDGE_VOICE_ASSISTANT_FEMALE_DEFAULT)
    p.add_argument("--block-ms", type=int, default=20)
    p.add_argument("--aec", type=str, choices=("none", "nlms"), default="nlms")
    p.add_argument("--filter-len", type=int, default=2048)
    p.add_argument("--aec-mu", type=float, default=0.25)
    p.add_argument("--aec-delay-samples", type=int, default=0)
    p.add_argument("--enable-punc", type=int, default=1)
    p.add_argument("--verbose-vad", action="store_true")
    p.add_argument(
        "--no-call-record",
        action="store_true",
        help="不写入通话记录（默认写入 output/call_recordings，与日志目录分离）",
    )
    p.add_argument(
        "--call-record-dir",
        type=str,
        default="output/call_recordings",
        help="通话记录目录（相对仓库根）；与 --no-call-record 互斥",
    )
    p.add_argument(
        "--no-call-record-jsonl",
        action="store_true",
        help="不写入 .jsonl，仅会话结束时写 .json 与 _还原.txt",
    )
    p.add_argument(
        "--call-audio",
        action="store_true",
        help=(
            "另存通话 WAV：mic_asr（本机真麦时为回调所采；duplex 仿真见引擎注释）、"
            "tts_ref（下行数字参考）、stereo_micL_ttsR（L=上行麦克 R=TTS）；与 call_record 共用同一时间戳文件名前缀"
        ),
    )
    p.add_argument(
        "--impatient-barge-in",
        action="store_true",
        help=(
            "急躁客户真 LLM：启动后先播报首轮「长说明」（同步 LLM→TTS），请在播报中抢话；"
            "下一则 ASR 将带「打断重讲」提示再走 LLM。与短问候互斥（自动不播 greeting）。"
        ),
    )
    p.add_argument(
        "--impatient-first-seed",
        type=str,
        default=(
            "请扮演客服，用较长的一段话（约150到220字）向客户说明："
            "手机套餐变更的一般流程、费用结算常见注意点、以及何时生效。"
            "语气耐心、只陈述不要向客户提问。"
        ),
        help="首轮 LLM 所用的「伪用户」提示句，用于生成长说明（真麦克风尚未开口）。",
    )
    args = p.parse_args()

    base_url = str(args.llm_base_url).strip().rstrip("/")
    api_key = _resolve_llm_api_key(base_url, (args.api_key or "").strip())
    if not api_key:
        print(
            "请在 .env 或环境中设置 LLM_API_KEY / OPENAI_API_KEY，或传入 --api-key；"
            "本机 Ollama 请设置 LLM_BASE_URL=http://localhost:11434/v1（无需密钥）",
            file=sys.stderr,
        )
        sys.exit(2)
    if _looks_like_local_ollama_openai_compat(base_url) and api_key == "ollama":
        print("# LLM: Ollama OpenAI 兼容端点（占位密钥）", flush=True)

    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    call_rec: CallSessionRecord | None = None
    record_dir = "" if args.no_call_record else (args.call_record_dir or "").strip()
    session_stem: Path | None = None
    if record_dir:
        rec_path = Path(record_dir)
        if not rec_path.is_absolute():
            rec_path = _REPO / rec_path
        stem_tag = "call_record_voice_llm_impatient_" if args.impatient_barge_in else "call_record_"
        session_stem = rec_path / f"{stem_tag}{datetime.datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
        call_rec = CallSessionRecord(session_stem, write_jsonl=not args.no_call_record_jsonl)
        print(f"# 通话记录: {session_stem}.json / _还原.txt", flush=True)
    elif args.call_audio:
        rec_path = Path((args.call_record_dir or "output/call_recordings").strip())
        if not rec_path.is_absolute():
            rec_path = _REPO / rec_path
        stem_tag = "call_record_voice_llm_impatient_" if args.impatient_barge_in else "call_record_"
        session_stem = rec_path / f"{stem_tag}{datetime.datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
        print(f"# 仅通话录音: {session_stem}_mic_asr.wav 等", flush=True)

    audio_mic_chunks: list[np.ndarray] = []
    audio_tts_chunks: list[np.ndarray] = []

    root = Path(args.models_root)
    sr = 16000
    block = max(int(sr * args.block_ms / 1000.0), 80)

    asr_cfg = FireRedAsr2Config(
        use_gpu=args.device != "cpu",
        device=args.device,
        return_timestamp=False,
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
        enable_punc=bool(args.enable_punc),
        enable_diarization=False,
        stream_vad_use_gpu=False,
    )
    sys_m = FireRedAsr2System(cfg)
    session = sys_m.open_full_duplex_stream(
        uttid_prefix="voice_llm_duplex",
        verbose_vad=args.verbose_vad,
    )

    aec: NlmsMonoAec | None = None
    if args.aec == "nlms":
        aec = NlmsMonoAec(
            filter_len=args.filter_len,
            mu=args.aec_mu,
            ref_delay_samples=args.aec_delay_samples,
        )

    rt = _DuplexRuntime(
        lock=threading.Lock(),
        session=session,
        aec=aec,
        sr=sr,
        print_events=args.verbose_vad,
        call_rec=call_rec,
        record_call_audio=bool(args.call_audio),
    )

    utterance_q: queue.Queue[str] = queue.Queue()
    stop_ev = threading.Event()

    if (
        not args.no_greeting
        and (args.greeting or "").strip()
        and not args.impatient_barge_in
    ):
        greet = synthesize_tts_16k_int16(args.greeting, args.tts_engine, args.edge_voice)
        with rt.lock:
            rt.pending_tts_caption = (args.greeting or "").strip()
            rt.next_tts_segment = "greeting"
            rt.pending_tts = greet

    conv_lock = threading.Lock()
    messages: list[dict[str, str]] = [{"role": "system", "content": args.system_prompt}]

    if args.impatient_barge_in:
        seed = (args.impatient_first_seed or "").strip()
        if not seed:
            print("--impatient-first-seed 不能为空", file=sys.stderr)
            sys.exit(2)
        print("# impatient-barge-in：请求首轮长说明 LLM …", flush=True)
        seed_preview = seed if len(seed) <= 160 else seed[:157] + "…"
        print(
            f"# 首轮 LLM 请求：HTTP POST …/chat/completions  model={args.llm_model!r}\n"
            f"# 传入 2 条 message：system + user(**seed**，非麦克风 ASR)。user 预览：{seed_preview!r}",
            flush=True,
        )
        try:
            reply1 = openai_compatible_chat(
                [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": seed},
                ],
                api_key=api_key,
                base_url=base_url,
                model=args.llm_model,
                timeout_s=args.llm_timeout,
            )
        except Exception as e:
            print(f"首轮 LLM 失败：{e}", file=sys.stderr)
            sys.exit(3)
        reply1 = (reply1 or "").strip()
        if not reply1:
            print("首轮 LLM 返回空文本", file=sys.stderr)
            sys.exit(3)
        print(f"# 首轮 LLM 已返回 assistant 文本，长度 {len(reply1)} 字符（接着做 TTS）", flush=True)
        messages.append({"role": "user", "content": seed})
        messages.append({"role": "assistant", "content": reply1})
        try:
            pcm1 = synthesize_tts_16k_int16(reply1, args.tts_engine, args.edge_voice)
        except Exception as e:
            print(f"首轮 TTS 失败：{e}", file=sys.stderr)
            sys.exit(3)
        rt.impatient_mode = True
        with rt.lock:
            rt.pending_tts_caption = reply1
            rt.next_tts_segment = "assistant_round1"
            rt.pending_voice_note = "impatient·首轮长说明(LLM)"
            rt.pending_tts = pcm1
        dur_s = float(len(pcm1)) / float(sr) if len(pcm1) else 0.0
        print(
            f"# 首轮说明约 {dur_s:.1f}s，外放播报时请对麦克风抢话以触发 barge_in；"
            f"随后第一句话将走「打断重讲」LLM。\n",
            flush=True,
        )
        if float(args.session_seconds) < dur_s + 8.0:
            print(
                f"# 警告：--session-seconds={args.session_seconds:g} 短于首轮 TTS（≈{dur_s:.0f}s），"
                "会话会在播完前结束；且未出现 ASR 成句时 **不会再调** 第二条 LLM。\n",
                flush=True,
            )
        if call_rec is not None:
            call_rec.add(event="impatient_round1_seed", text=seed)
            call_rec.add(
                event="assistant_llm",
                text=reply1,
                source="impatient_round1",
                voice_note="首轮长说明",
            )

    def worker() -> None:
        while not stop_ev.is_set():
            try:
                user_text = utterance_q.get(timeout=0.3)
            except queue.Empty:
                continue
            user_text = user_text.strip()
            if not user_text:
                continue
            gen_before = rt.generation
            with rt.lock:
                do_impatient_retry = rt.impatient_retry_pending
                if do_impatient_retry:
                    rt.impatient_retry_pending = False
            user_for_llm = user_text
            if do_impatient_retry:
                user_for_llm = (
                    "【客户在你长说明过程中打断，原话为】"
                    + user_text
                    + "\n请先用一句话承认被打断，再用更短、更清晰的一段话重新说明核心信息，适合语音播报，总长度控制在约80字以内。"
                )
            with conv_lock:
                messages.append({"role": "user", "content": user_text})
                to_send = list(messages)
                if do_impatient_retry:
                    to_send[-1] = {"role": "user", "content": user_for_llm}
            print(f"\n# 用户(ASR): {user_text}\n", flush=True)
            if do_impatient_retry:
                print("# impatient：本轮 LLM 使用「打断重讲」提示生成二轮说明\n", flush=True)
            if call_rec is not None:
                call_rec.add(
                    event="user_asr",
                    text=user_text,
                    impatient_retry_turn=bool(do_impatient_retry),
                )
            try:
                reply = openai_compatible_chat(
                    to_send,
                    api_key=api_key,
                    base_url=base_url,
                    model=args.llm_model,
                    timeout_s=args.llm_timeout,
                )
            except Exception as e:
                reply = f"大模型请求失败：{e}"
                if call_rec is not None:
                    call_rec.add(event="llm_error", detail=str(e))
            if stop_ev.is_set():
                return
            if rt.generation != gen_before:
                print("# 已打断，丢弃本轮回复", flush=True)
                if call_rec is not None:
                    call_rec.add(event="reply_discarded")
                continue
            if not (reply or "").strip():
                reply = "我没有生成有效回复。"
            with conv_lock:
                messages.append({"role": "assistant", "content": reply})
                while len(messages) > 1 + 2 * args.max_turns:
                    if len(messages) > 1:
                        messages.pop(1)
                    if len(messages) > 1:
                        messages.pop(1)
            print(f"# 助手: {reply}\n", flush=True)
            if call_rec is not None:
                call_rec.add(event="assistant_llm", text=reply)
            if rt.generation != gen_before:
                continue
            try:
                pcm = synthesize_tts_16k_int16(reply, args.tts_engine, args.edge_voice)
            except Exception as e:
                pcm = synthesize_tts_16k_int16(
                    f"TTS 失败：{e}", args.tts_engine, args.edge_voice
                )
            if stop_ev.is_set():
                return
            if rt.generation != gen_before:
                continue
            with rt.lock:
                if rt.generation != gen_before:
                    continue
                rt.pending_tts_caption = (reply or "").strip()
                rt.next_tts_segment = "assistant"
                rt.pending_tts = pcm

    th = threading.Thread(target=worker, name="llm_tts_worker", daemon=True)
    th.start()

    def callback(indata, outdata, frames, _time, status) -> None:
        if status:
            print(f"# audio status: {status}", flush=True)
        n = int(frames)
        ref = np.zeros(n, dtype=np.float32)
        with rt.lock:
            if rt.tts_pos < len(rt.tts_i16):
                take = min(n, len(rt.tts_i16) - rt.tts_pos)
                ref[:take] = _i16_to_float_mono(rt.tts_i16[rt.tts_pos : rt.tts_pos + take])
                rt.tts_pos += take
        outdata[:, 0] = ref

        mic = indata[:, 0].astype(np.float32).copy()
        if rt.aec is not None:
            mic_u = rt.aec.process_block(mic, ref)
        else:
            mic_u = mic
        pcm16 = _float_to_i16_mono(mic_u)
        rec_m = pcm16.copy() if rt.record_call_audio else None
        rec_t = _float_to_i16_mono(ref).copy() if rt.record_call_audio else None
        evs = rt.session.push_microphone_pcm(pcm16, sample_rate=sr)

        to_queue: list[str] = []
        with rt.lock:
            for ev in evs:
                if ev.get("event") == "barge_in":
                    if rt.call_rec is not None:
                        rt.call_rec.add(
                            event="barge_in",
                            playback_id=ev.get("playback_id"),
                        )
                    rt.generation += 1
                    _stop_playback_locked(rt)
                    if rt.impatient_mode:
                        rt.impatient_retry_pending = True
                    print("# barge_in：停止播放", flush=True)
                    continue
                if rt.print_events:
                    print(json.dumps(ev, ensure_ascii=False, default=str)[:4000], flush=True)
                utt = _pipeline_user_text(ev)
                if utt:
                    to_queue.append(utt)
            _finish_if_tts_done(rt)
            _try_start_pending_playback(rt)
            _finish_if_tts_done(rt)
        if rec_m is not None and rec_t is not None:
            audio_mic_chunks.append(rec_m)
            audio_tts_chunks.append(rec_t)
        for utt in to_queue:
            utterance_q.put(utt)

    print(
        f"# 全双工开始 {sr} Hz block={block} session={args.session_seconds:.0f}s "
        f"llm={args.llm_model} base={base_url} aec={args.aec}"
        f"{' impatient_barge_in' if args.impatient_barge_in else ''}",
        flush=True,
    )

    try:
        with sd.Stream(
            samplerate=sr,
            blocksize=block,
            dtype="float32",
            channels=1,
            callback=callback,
        ):
            sd.sleep(int(max(args.session_seconds * 1000.0, block / sr * 1000.0)))
    finally:
        stop_ev.set()
        th.join(timeout=5.0)
        for ev in session.finalize():
            print(json.dumps(ev, ensure_ascii=False, default=str)[:4000], flush=True)
        meta_end: dict[str, Any] = {
            "started_at": started_at,
            "ended_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "asr_device": args.device,
            "llm_model": args.llm_model,
            "llm_base_url": base_url,
            "session_seconds": args.session_seconds,
            "greeting": (
                ""
                if args.impatient_barge_in
                else ((args.greeting or "").strip() if not args.no_greeting else "")
            ),
            "system_prompt": (args.system_prompt or "").strip(),
            "impatient_barge_in": bool(args.impatient_barge_in),
            "impatient_first_seed": (args.impatient_first_seed or "").strip()
            if args.impatient_barge_in
            else "",
        }
        if args.impatient_barge_in:
            meta_end["greeting_note"] = (
                "impatient 模式未播放短问候；首轮为 LLM 长说明 TTS，"
                "非本字段默认文案。"
            )
        if args.call_audio and session_stem is not None and audio_mic_chunks:
            meta_end["call_audio"] = _save_call_session_audio(
                session_stem, audio_mic_chunks, audio_tts_chunks, sr
            )
            print(
                f"# 通话录音已写入: {Path(meta_end['call_audio']['mic_asr_wav']).name} 等",
                flush=True,
            )
        elif args.call_audio and session_stem is not None:
            print("# 通话录音: 无采集块（会话过短或未进入回调）", flush=True)

        if call_rec is not None:
            pre_counts = call_rec.event_counts()
            meta_end["conversation_flags"] = {
                "barge_in_events": int(pre_counts.get("barge_in", 0)),
                "user_asr_events": int(pre_counts.get("user_asr", 0)),
                "assistant_llm_events": int(pre_counts.get("assistant_llm", 0)),
            }
            call_rec.add(event="session_end", detail="finalize")
            call_rec.finalize(meta_end)
            print(
                f"# 通话记录已写入: {call_rec.path_transcript.name}  |  {call_rec.path_json.name}",
                flush=True,
            )
        elif meta_end.get("call_audio") and session_stem is not None:
            stub = Path(str(session_stem) + "_meta.json")
            stub.write_text(json.dumps(meta_end, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"# 仅录音时的 meta: {stub.name}", flush=True)


if __name__ == "__main__":
    main()
