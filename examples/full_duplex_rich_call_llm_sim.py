#!/usr/bin/env python3
"""多轮真实感通话回放：精简开场与核身 → 客户确认 → 短过渡 → 讲解约 5s 后打断×3 → 告别。

- 助手：**每轮到播放前即时 LLM → 再即时 TTS**（非会话开始前一次性预生成全部轮次）。
- **barge 轮默认反应式**：助手话术按句 TTS + ``duplex_scripted_engine`` 推流；客户抢话触发 ``barge_in`` 时**停播**，再根据 ASR 片段调 LLM 短答（可说不清楚）。``--no-reactive-barge`` 回退旧版「整条音轨叠时间轴」。
- 加 ``--call-audio`` 时：旧版（``--no-reactive-barge``）在拼完时间轴后**先落盘**再推流；**默认反应式**在 duplex 推流**结束后**从采集缓冲写立体声 WAV（与真实通话一致，无法在 ASR 前预知最终 ref）。
- 客户：``customer_pool.json`` 片段可 **随机拼接** 进回放（天气/足球/闲聊+业务），仅供 ASR 侧验证；**仿真不从剧本向 LLM 注入客户全文**，助手行为由 ``system_prompt``、流式「客户·识别」、以及 **通话记忆智能体**（``rich_call_memory_context``：追踪块 + 可选 LLM 摘要）与各轮注册表编导约束共同决定。可用 ``--memory-agent-llm`` 开启摘要（多一次 LLM）。
- **Token 控制**：默认对发往 LLM 的 ``messages`` 做尾部条数限制与单条字符截断（见 ``--llm-api-max-*`` / ``--no-llm-context-compress``），减轻上下文过长导致的 500；抢话续接里 ASR 拼接也会截断。
- **排查**：默认在 ``meta.json`` 写入 ``llm_request_trace``（各次 ``/chat/completions`` 的 model、temperature、messages 快照，无 API Key）；可用 ``--no-llm-request-trace`` 关闭，``--llm-request-trace-max-chars`` 控制单条 content 长度。
- 号码规范：场景要求模型输出「幺零零八六」而非阿拉伯数字 10086，便于按数字读法播报。

运行::

  .venv\\Scripts\\python.exe scripts\\prepare_rich_call_scenario_wavs.py
  .venv\\Scripts\\python.exe examples\\full_duplex_rich_call_llm_sim.py --call-audio

一键端到端对话测试（准备 WAV + 仿真 + 校验 meta/录音）::

  .venv\\Scripts\\python.exe scripts\\run_rich_call_conversation_e2e.py

LLM 读仓库根 ``.env``（与 full_duplex_voice_llm_tts 相同变量）。"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EXAMPLES = Path(__file__).resolve().parent
_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _EXAMPLES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import stdio_utf8_windows  # noqa: E402

stdio_utf8_windows.apply_stdio_utf8()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

os.environ.setdefault("PYTORCH_ENABLE_XPU_FALLBACK", "1")

from fireredasr2s.fireredasr2 import FireRedAsr2Config  # noqa: E402
from fireredasr2s.fireredasr2system import FireRedAsr2System, FireRedAsr2SystemConfig  # noqa: E402
from fireredasr2s.firereddiar.audio import prepare_asr_stack_audio  # noqa: E402
from fireredasr2s.fireredlid import FireRedLidConfig  # noqa: E402
from fireredasr2s.fireredpunc import FireRedPuncConfig  # noqa: E402
from fireredasr2s.fireredvad import FireRedVadConfig  # noqa: E402
from fireredasr2s.duplex import NlmsMonoAec  # noqa: E402
from fireredasr2s.repo_dotenv import default_asr_device, load_repo_dotenv  # noqa: E402

import duplex_scripted_engine as _dse  # noqa: E402
from rich_call_memory_context import (  # noqa: E402
    RichCallMemoryAgent,
    compose_task_user_content,
    resolve_round_task_body,
)

DuplexSimRuntime = _dse.DuplexSimRuntime
feed_duplex_chunk = _dse.feed_duplex_chunk

_SCENARIO_DIR = _REPO / "examples" / "duplex_rich_call_scenario"


def _have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _resolve_tts_engine(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    try:
        import edge_tts  # noqa: F401

        if _have_ffmpeg():
            return "edge"
    except ImportError:
        pass
    return "pyttsx3"


def _round_needs_llm(rd: dict[str, Any]) -> bool:
    return str(rd.get("mode") or "").strip() != "customer_uplink_only"


def _normalize_10086_for_tts(text: str) -> str:
    """若模型仍输出阿拉伯数字，播报前改为数字读法常用写法。"""
    t = text.replace("１００８６", "幺零零八六")
    return t.replace("10086", "幺零零八六")


def _collapse_assistant_reply_formatting(text: str) -> str:
    """去掉模型偶发的换行分段（否则 meta 出现 \\n\\n，且按句切 TTS 会异常）。"""
    t = (text or "").strip()
    t = re.sub(r"\s*\n+\s*", " ", t)
    t = re.sub(r"[（(]\s*稍顿\s*[）)]", " ", t)
    t = re.sub(r" {2,}", " ", t)
    return t.strip()


def _synthesize_assistant_pcm(
    *,
    text: str,
    vm: Any,
    tts_engine: str,
    edge_voice: str,
    edge_rate: str,
    edge_pitch: str,
    round_label: str,
    sr: int,
) -> np.ndarray:
    print(f"# 轮次 {round_label!r} TTS …", flush=True)
    try:
        pcm = vm.synthesize_tts_16k_int16(
            text,
            tts_engine,
            edge_voice,
            edge_rate=str(edge_rate),
            edge_pitch=str(edge_pitch),
            pyttsx3_gender="female",
        )
    except Exception as e:
        if tts_engine == "edge":
            print(
                f"# 轮次 {round_label!r} Edge TTS 失败，回退 pyttsx3：{e}",
                file=sys.stderr,
                flush=True,
            )
            pcm = vm.synthesize_tts_16k_int16(
                text,
                "pyttsx3",
                edge_voice,
                edge_rate=str(edge_rate),
                edge_pitch=str(edge_pitch),
                pyttsx3_gender="female",
            )
        else:
            raise
    a = pcm.astype(np.int16)
    print(f"# 轮次 {round_label!r} TTS 时长 {len(a) / sr:.1f}s", flush=True)
    return a


def _format_agent_task_message(body: str, *, kind: str = "话务任务") -> str:
    """把 JSON 里的编导提示与「客户·识别」区分开，避免模型把静态说明当成客户原话。"""
    b = (body or "").strip()
    if not b:
        return b
    if b.startswith("【"):
        return b
    return f"【{kind}】{b}"


@dataclass(frozen=True)
class LlmApiSlimConfig:
    """控制发往 LLM 的 messages 体积，减轻上下文过长导致的网关 500 / token 超限。"""

    enabled: bool = True
    max_tail_messages: int = 12
    max_user_chars: int = 1400
    max_assistant_chars: int = 720
    max_system_chars: int = 3000


def _messages_for_openai_api(
    messages: list[dict[str, str]],
    cfg: LlmApiSlimConfig,
) -> list[dict[str, str]]:
    """保留 system + 最近若干条对话，并对单条 content 截断（不修改原列表）。"""
    out: list[dict[str, str]] = []
    sys_msgs = [m for m in messages if str(m.get("role")) == "system"]
    rest = [m for m in messages if str(m.get("role")) != "system"]
    for m in sys_msgs:
        c = str(m.get("content") or "")
        mx = int(cfg.max_system_chars)
        if len(c) > mx:
            c = c[: max(0, mx - 12)] + "…（system已截断）"
        out.append({"role": "system", "content": c})
    tail_n = max(1, int(cfg.max_tail_messages))
    tail = rest[-tail_n:] if len(rest) > tail_n else rest
    if len(rest) > len(tail):
        out.append(
            {
                "role": "user",
                "content": f"【系统】更早的 {len(rest) - len(tail)} 条对话已省略以控制长度。",
            }
        )
    mu, ma = int(cfg.max_user_chars), int(cfg.max_assistant_chars)
    for m in tail:
        role = str(m.get("role") or "user")
        c = str(m.get("content") or "")
        lim = ma if role == "assistant" else mu
        if len(c) > lim:
            c = c[: max(0, lim - 1)] + "…"
        out.append({"role": role, "content": c})
    return out


def _append_customer_asr_to_messages(
    messages: list[dict[str, str]],
    utts: list[str],
    *,
    memory: RichCallMemoryAgent | None = None,
) -> None:
    merged = " ".join(x.strip() for x in utts if x and str(x).strip())
    if merged:
        messages.append({"role": "user", "content": f"【客户·识别】{merged}"})
        if memory is not None:
            memory.add_customer_asr(merged)


def _llm_append_user(
    messages: list[dict[str, str]],
    user_text: str,
    *,
    vm: Any,
    api_key: str,
    base_url: str,
    llm_model: str,
    llm_timeout: float,
    round_label: str,
    slim_cfg: LlmApiSlimConfig | None,
    llm_request_trace: list[dict[str, Any]] | None = None,
    llm_trace_max_chars: int = 0,
) -> str:
    print(f"# 轮次 {round_label!r} 即时 LLM …", flush=True)
    messages.append({"role": "user", "content": _format_agent_task_message(user_text)})
    compress_on = slim_cfg is not None and slim_cfg.enabled
    payload = (
        _messages_for_openai_api(messages, slim_cfg)
        if compress_on
        else list(messages)
    )
    try:
        reply = vm.openai_compatible_chat(
            payload,
            api_key=api_key,
            base_url=base_url,
            model=str(llm_model),
            timeout_s=float(llm_timeout),
            request_trace=llm_request_trace,
            trace_tag=str(round_label),
            trace_max_content_chars=int(llm_trace_max_chars),
            trace_extra={
                "role": "main_assistant",
                "context_compress_applied": bool(compress_on),
            },
        )
    except Exception as e:
        print(f"LLM 失败 round={round_label}: {e}", file=sys.stderr)
        sys.exit(3)
    reply = (reply or "").strip()
    if not reply:
        print(f"LLM 空回复 round={round_label}", file=sys.stderr)
        sys.exit(3)
    return _collapse_assistant_reply_formatting(_normalize_10086_for_tts(reply))


def _assistant_llm_tts_jit(
    *,
    rd: dict[str, Any],
    round_index: int,
    messages: list[dict[str, str]],
    vm: Any,
    api_key: str,
    base_url: str,
    llm_model: str,
    llm_timeout: float,
    tts_engine: str,
    edge_voice: str,
    edge_rate: str,
    edge_pitch: str,
    sr: int,
    memory: RichCallMemoryAgent,
    memory_agent_llm: bool,
    no_memory_preamble: bool,
    memory_agent_timeout: float,
    slim_cfg: LlmApiSlimConfig | None,
    llm_request_trace: list[dict[str, Any]] | None = None,
    llm_trace_max_chars: int = 0,
) -> tuple[np.ndarray, str]:
    """该助手轮在拼进时间轴之前即时请求 LLM 并合成 TTS（单条整段，用于 listen / 旧版 barge）。"""
    nm = str(rd.get("name") or "?")
    if memory_agent_llm and not no_memory_preamble:
        print("# 记忆智能体：更新摘要 …", flush=True)
        memory.refresh_summary_with_llm(
            vm=vm,
            api_key=api_key,
            base_url=base_url,
            model=str(llm_model),
            timeout_s=float(memory_agent_timeout),
            request_trace=llm_request_trace,
            trace_max_content_chars=int(llm_trace_max_chars),
        )
    u = compose_task_user_content(
        rd,
        memory,
        use_memory_preamble=not no_memory_preamble,
    ).strip()
    if not u:
        print(
            f"rounds[{round_index}]（LLM 轮）缺少编导约束：请设置 llm_user 或为 name={nm!r} 配置注册表",
            file=sys.stderr,
        )
        sys.exit(2)
    reply = _llm_append_user(
        messages,
        u,
        vm=vm,
        api_key=api_key,
        base_url=base_url,
        llm_model=llm_model,
        llm_timeout=llm_timeout,
        round_label=nm,
        slim_cfg=slim_cfg,
        llm_request_trace=llm_request_trace,
        llm_trace_max_chars=llm_trace_max_chars,
    )
    messages.append({"role": "assistant", "content": reply})
    memory.add_assistant_spoken(reply)
    a = _synthesize_assistant_pcm(
        text=reply,
        vm=vm,
        tts_engine=tts_engine,
        edge_voice=edge_voice,
        edge_rate=edge_rate,
        edge_pitch=edge_pitch,
        round_label=nm,
        sr=sr,
    )
    return a, reply


def _split_assistant_utterances(text: str, *, max_chars: int = 52) -> list[str]:
    """按标点切句，过长句再硬切，便于抢话时停播。"""
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？；\n])\s*", text)
    chunks: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        while len(p) > max_chars:
            chunks.append(p[:max_chars])
            p = p[max_chars:]
        if p:
            chunks.append(p)
    return chunks if chunks else [text]


def _bump_stream_stats(stats: dict[str, int], ev: dict[str, Any]) -> None:
    et = ev.get("event")
    if et == "barge_in":
        stats["barge_in"] = stats.get("barge_in", 0) + 1
    if et == "barge_in_suppressed":
        stats["barge_in_suppressed"] = stats.get("barge_in_suppressed", 0) + 1
    if et == "segment_final":
        stats["segment_final"] = stats.get("segment_final", 0) + 1


def _print_stream_event(ev: dict[str, Any]) -> None:
    print(json.dumps(ev, ensure_ascii=False, default=str)[:2000], flush=True)


def _feed_duplex_pad_chunk(
    rt: DuplexSimRuntime,
    session: Any,
    chunk: np.ndarray,
    *,
    record_bufs: tuple[list[np.ndarray], list[np.ndarray]] | None,
    stats: dict[str, int],
) -> list[str]:
    evs: list[dict[str, Any]] = []
    utts = feed_duplex_chunk(
        rt,
        session,
        chunk,
        rec_add=None,
        record_audio=record_bufs,
        verbose_print=None,
        collect_events=evs,
    )
    for ev in evs:
        _print_stream_event(ev)
        _bump_stream_stats(stats, ev)
    return utts


def _feed_int16_pcm_duplex(
    rt: DuplexSimRuntime,
    session: Any,
    pcm: np.ndarray,
    *,
    chunk_samples: int,
    record_bufs: tuple[list[np.ndarray], list[np.ndarray]] | None,
    stats: dict[str, int],
) -> list[str]:
    acc: list[str] = []
    for i in range(0, len(pcm), chunk_samples):
        chunk = pcm[i : i + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.concatenate(
                [chunk, np.zeros(chunk_samples - len(chunk), dtype=np.int16)]
            )
        acc.extend(
            _feed_duplex_pad_chunk(rt, session, chunk, record_bufs=record_bufs, stats=stats)
        )
    return acc


def _silence_until_duplex_playback_done(
    rt: DuplexSimRuntime,
    session: Any,
    *,
    chunk_samples: int,
    sr: int,
    record_bufs: tuple[list[np.ndarray], list[np.ndarray]] | None,
    stats: dict[str, int],
    max_wall_s: float = 120.0,
) -> None:
    """客户时间轴结束后继续推静音块，直到本地 TTS 参考播完（避免短答被截断）。"""
    sil = np.zeros(chunk_samples, dtype=np.int16)
    max_n = max(int(float(max_wall_s) * float(sr) / float(chunk_samples)), 1)
    for _ in range(max_n):
        with rt.lock:
            pend = rt.pending_tts
            busy = rt.in_playback or (
                pend is not None and len(pend) > 0
            ) or (len(rt.tts_i16) > 0 and rt.tts_pos < len(rt.tts_i16))
        if not busy:
            return
        _feed_duplex_pad_chunk(rt, session, sil, record_bufs=record_bufs, stats=stats)


def _run_reactive_barge_round(
    *,
    rd: dict[str, Any],
    round_index: int,
    mic_full: np.ndarray,
    messages: list[dict[str, str]],
    vm: Any,
    api_key: str,
    base_url: str,
    llm_model: str,
    llm_timeout: float,
    tts_engine: str,
    edge_voice: str,
    edge_rate: str,
    edge_pitch: str,
    sr: int,
    chunk_samples: int,
    rt: DuplexSimRuntime,
    session: Any,
    record_bufs: tuple[list[np.ndarray], list[np.ndarray]] | None,
    stats: dict[str, int],
    assistant_turns_meta: list[dict[str, Any]],
    memory: RichCallMemoryAgent,
    memory_agent_llm: bool,
    no_memory_preamble: bool,
    memory_agent_timeout: float,
    slim_cfg: LlmApiSlimConfig | None,
    llm_request_trace: list[dict[str, Any]] | None = None,
    llm_trace_max_chars: int = 0,
) -> None:
    """按句播报助手，遇 barge_in 停播并以 ASR 片段触发第二轮短答。"""
    nm = str(rd.get("name") or f"round{round_index}")
    if memory_agent_llm and not no_memory_preamble:
        print("# 记忆智能体：更新摘要 …", flush=True)
        memory.refresh_summary_with_llm(
            vm=vm,
            api_key=api_key,
            base_url=base_url,
            model=str(llm_model),
            timeout_s=float(memory_agent_timeout),
            request_trace=llm_request_trace,
            trace_max_content_chars=int(llm_trace_max_chars),
        )
    u = compose_task_user_content(
        rd,
        memory,
        use_memory_preamble=not no_memory_preamble,
    ).strip()
    if not u:
        print(
            f"rounds[{round_index}]（LLM 轮）缺少编导约束：请设置 llm_user 或为 name={nm!r} 配置注册表",
            file=sys.stderr,
        )
        sys.exit(2)

    reply_main = _llm_append_user(
        messages,
        u,
        vm=vm,
        api_key=api_key,
        base_url=base_url,
        llm_model=llm_model,
        llm_timeout=llm_timeout,
        round_label=nm,
        slim_cfg=slim_cfg,
        llm_request_trace=llm_request_trace,
        llm_trace_max_chars=llm_trace_max_chars,
    )
    ut_chunks = _split_assistant_utterances(reply_main)
    sentence_pcms = [
        _synthesize_assistant_pcm(
            text=t,
            vm=vm,
            tts_engine=tts_engine,
            edge_voice=edge_voice,
            edge_rate=edge_rate,
            edge_pitch=edge_pitch,
            round_label=f"{nm}#{i + 1}",
            sr=sr,
        )
        for i, t in enumerate(ut_chunks)
    ]
    plan_q: deque[tuple[str, np.ndarray]] = deque(zip(ut_chunks, sentence_pcms))
    spoken_fragments: list[str] = []
    barge_happened = False
    reactive_llm_done = False
    pending_reactive_llm = False
    wait_chunks_after_barge = 0
    max_wait_chunks_after_barge = max(32, int(2.5 * float(sr) / float(max(chunk_samples, 1))))
    asr_collected: list[str] = []

    def try_schedule() -> None:
        with rt.lock:
            if rt.pending_tts is not None:
                return
            if rt.tts_pos < len(rt.tts_i16) and len(rt.tts_i16) > 0:
                return
            if not plan_q:
                return
            t, pcm = plan_q.popleft()
            rt.pending_tts = pcm
            rt.pending_tts_caption = t[:200]
            spoken_fragments.append(t)

    def fire_reactive_llm(asr_j: str) -> None:
        nonlocal reactive_llm_done, pending_reactive_llm
        part_a = "".join(spoken_fragments).strip()
        if part_a:
            messages.append({"role": "assistant", "content": part_a})
            memory.add_assistant_spoken(part_a)
        asr_hint = (asr_j or "").strip() or "（暂无明显字）"
        if len(asr_hint) > 420:
            asr_hint = asr_hint[:419] + "…"
        fu_body = (
            "客户在你播报时插话，双方声音叠在一起，流式识别结果可能漏字或不准；当前转写大致为："
            f"「{asr_hint}」。"
            "请像真人外呼客服一样自然应答，不要逐字复读。不要用「您先说」「您请讲」「您先说着我听着」「听着呢」「好的您说」等被动承接；应直接给要点："
            "套餐办理渠道与费用/生效规则；若涉及**话费/余额**可给**一句演示性模拟口径**（数字勿夸张）并点明**以中国移动 App 与系统为准**；"
            "若涉及**上网/信号**给**初排**一句；若**不满**可平静提示 App 投诉、继续幺零零八六升级、按规定申诉渠道。"
            "无关闲聊半句带过；总字数不超过78字，不要念稿，不要用舞台提示语如「稍顿」。"
        )
        pre_fb = (
            memory.render_preamble_for_task().strip() + "\n\n"
            if (not no_memory_preamble)
            else ""
        )
        fu = pre_fb + fu_body
        messages.append(
            {"role": "user", "content": _format_agent_task_message(fu, kind="话务任务 · 抢话续接")}
        )
        print(f"# 轮次 {nm!r} 抢话后接续 LLM …", flush=True)
        react_compress = slim_cfg is not None and slim_cfg.enabled
        react_payload = (
            _messages_for_openai_api(messages, slim_cfg)
            if react_compress
            else list(messages)
        )
        try:
            reply2 = vm.openai_compatible_chat(
                react_payload,
                api_key=api_key,
                base_url=base_url,
                model=str(llm_model),
                timeout_s=float(llm_timeout),
                request_trace=llm_request_trace,
                trace_tag=f"{nm}·抢话续接",
                trace_max_content_chars=int(llm_trace_max_chars),
                trace_extra={
                    "role": "reactive_barge",
                    "context_compress_applied": bool(react_compress),
                },
            )
        except Exception as e:
            print(f"LLM 失败 round={nm} reactive: {e}", file=sys.stderr)
            sys.exit(3)
        reply2 = _collapse_assistant_reply_formatting(
            _normalize_10086_for_tts((reply2 or "").strip())
        )
        if not reply2:
            reply2 = "刚才叠音我没听全，您打开移动App查一下账单明细，或打幺零零八六让专员帮您核。"
        messages.append({"role": "assistant", "content": reply2})
        memory.add_assistant_spoken(reply2)
        pcm2 = _synthesize_assistant_pcm(
            text=reply2,
            vm=vm,
            tts_engine=tts_engine,
            edge_voice=edge_voice,
            edge_rate=edge_rate,
            edge_pitch=edge_pitch,
            round_label=f"{nm}_reactive",
            sr=sr,
        )
        with rt.lock:
            rt.pending_tts = pcm2
            rt.pending_tts_caption = reply2[:200]
        assistant_turns_meta.append(
            {
                "name": nm,
                "text": reply_main,
                "reactive_after_barge": True,
                "reactive_reply": reply2,
                "asr_hint_joined": asr_j,
            }
        )
        reactive_llm_done = True
        pending_reactive_llm = False

    try_schedule()

    mic_pos = 0
    while mic_pos < len(mic_full):
        chunk = mic_full[mic_pos : mic_pos + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.concatenate(
                [chunk, np.zeros(chunk_samples - len(chunk), dtype=np.int16)]
            )
        mic_pos += chunk_samples
        gen_before = rt.generation
        utts = _feed_duplex_pad_chunk(
            rt, session, chunk, record_bufs=record_bufs, stats=stats
        )
        asr_collected.extend(utt for utt in utts if utt)
        if rt.generation != gen_before and not reactive_llm_done:
            barge_happened = True
            plan_q.clear()
            pending_reactive_llm = True
            wait_chunks_after_barge = 0

        if pending_reactive_llm and not reactive_llm_done:
            asr_j_now = " ".join(asr_collected).strip()
            if asr_j_now:
                fire_reactive_llm(asr_j_now)
            else:
                wait_chunks_after_barge += 1
                if wait_chunks_after_barge >= max_wait_chunks_after_barge:
                    fire_reactive_llm("")
        else:
            try_schedule()

    _silence_until_duplex_playback_done(
        rt,
        session,
        chunk_samples=chunk_samples,
        sr=sr,
        record_bufs=record_bufs,
        stats=stats,
    )

    if pending_reactive_llm and not reactive_llm_done:
        fire_reactive_llm(" ".join(asr_collected).strip())

    if not barge_happened:
        messages.append({"role": "assistant", "content": reply_main})
        assistant_turns_meta.append({"name": nm, "text": reply_main})
        memory.add_assistant_spoken(reply_main)

    cust_trace = " ".join(asr_collected).strip()
    if cust_trace:
        messages.append({"role": "user", "content": f"【客户·识别】{cust_trace}"})
        memory.add_customer_asr(cust_trace)


def _load_voice_llm():
    p = _REPO / "examples" / "full_duplex_voice_llm_tts.py"
    spec = importlib.util.spec_from_file_location("_fdv_rich_sim", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_mono_16k(path: Path) -> np.ndarray:
    raw, sr_ = sf.read(str(path), dtype="int16")
    if raw.ndim > 1:
        raw = raw.mean(axis=1).astype(np.int16)
    pcm_, _sr2 = prepare_asr_stack_audio(raw, int(sr_))
    return pcm_.astype(np.int16)


def _ms_to_samples(sr: int, ms: float) -> int:
    return max(int(float(sr) * float(ms) / 1000.0), 0)


def _snippet_wav_rel(snip_id: str) -> str:
    return f"wavs/cust_{snip_id}.wav"


def _pick_tangent_snippets(
    pool: dict[str, Any], rng: random.Random, min_n: int, max_n: int
) -> list[dict[str, Any]]:
    cats = [
        c
        for c in ("tangents_weather", "tangents_football", "tangents_misc")
        if pool.get(c)
    ]
    if not cats:
        return []
    if min_n > max_n:
        min_n, max_n = max_n, min_n
    n = rng.randint(min_n, max_n)
    picks: list[dict[str, Any]] = []
    used_cat: list[str] = []
    for _ in range(n):
        avail = [c for c in cats if c not in used_cat] or cats
        cat = rng.choice(avail)
        if len(avail) > 1:
            used_cat.append(cat)
        picks.append(rng.choice(pool[cat]))
    return picks


def _apply_random_customer_plan(
    scenario: dict[str, Any], scen_root: Path, seed: int
) -> dict[str, Any]:
    """按 customer_pool 随机拼接客户 uplink/打断音频（仅用于回放与 ASR 评测）。

    不向 LLM 注入 customer_pool 里的台词原文；流式 ASR 结果由仿真主程序写入对话中的「客户·识别」。"""
    rcfg = scenario.get("random_customer")
    if not isinstance(rcfg, dict) or not rcfg:
        return {}
    pool_path = scen_root / str(rcfg.get("pool") or "customer_pool.json")
    if not pool_path.is_file():
        print(f"找不到 customer_pool: {pool_path}", file=sys.stderr)
        sys.exit(2)
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    rng = random.Random(int(seed))
    min_t = int(rcfg.get("min_tangents", 1))
    max_t = int(rcfg.get("max_tangents", 2))
    if min_t > max_t:
        min_t, max_t = max_t, min_t
    if max_t <= 0:
        min_t, max_t = 0, 0
    gap_ms = float(rcfg.get("inter_snippet_gap_ms", 160))
    core_map: dict[str, str] = rcfg.get("barge_core_keys") or {}

    rounds = scenario.get("rounds") or []
    plan: dict[str, Any] = {
        "seed": int(seed),
        "identity": None,
        "package_followup": None,
        "hangup_end": None,
        "barges": {},
        "inter_snippet_gap_ms": gap_ms,
    }

    id_key = str(rcfg.get("identity_pool_key") or "identity_confirm").strip() or "identity_confirm"
    idents = pool.get(id_key) or []
    if not idents:
        print(f"customer_pool 缺少列表 {id_key}", file=sys.stderr)
        sys.exit(2)
    ident = rng.choice(idents)
    plan["identity"] = {"id": ident["id"], "text": ident["text"]}

    for rd in rounds:
        if str(rd.get("name")) == "customer_confirm_identity":
            rd["customer_concat_rels"] = [_snippet_wav_rel(str(ident["id"]))]
            rd.pop("customer_wav", None)
            rd["inter_snippet_gap_ms"] = gap_ms

    pkg_key = str(rcfg.get("package_followup_pool_key") or "").strip()
    if pkg_key:
        pkgs = pool.get(pkg_key) or []
        if not pkgs:
            print(f"customer_pool 缺少列表 {pkg_key}", file=sys.stderr)
            sys.exit(2)
        pkg = rng.choice(pkgs)
        plan["package_followup"] = {"id": pkg["id"], "text": pkg["text"]}
        for rd in rounds:
            if str(rd.get("name")) == "customer_confirm_package":
                rd["customer_concat_rels"] = [_snippet_wav_rel(str(pkg["id"]))]
                rd.pop("customer_wav", None)
                rd["inter_snippet_gap_ms"] = gap_ms

    has_hangup_uplink = any(str(rd.get("name")) == "customer_end_call" for rd in rounds)
    if has_hangup_uplink:
        hangups = pool.get("call_end_intent") or []
        if not hangups:
            print("customer_pool 缺少 call_end_intent（场景含 customer_end_call 轮）", file=sys.stderr)
            sys.exit(2)
        hang = rng.choice(hangups)
        plan["hangup_end"] = {"id": str(hang["id"]), "text": str(hang["text"])}
        for rd in rounds:
            if str(rd.get("name")) == "customer_end_call":
                rd["customer_concat_rels"] = [_snippet_wav_rel(str(hang["id"]))]
                rd.pop("customer_wav", None)
                rd["inter_snippet_gap_ms"] = gap_ms

    extra_map = rcfg.get("extra_customer_uplink_pools") or {}
    if isinstance(extra_map, dict):
        xu: dict[str, Any] = {}
        for rd in rounds:
            if str(rd.get("mode")) != "customer_uplink_only":
                continue
            nm = str(rd.get("name") or "")
            pk = str(extra_map.get(nm) or "").strip()
            if not pk:
                continue
            items = pool.get(pk) or []
            if not items:
                print(
                    f"customer_pool 缺少列表 {pk}（extra_customer_uplink_pools[{nm!r}]）",
                    file=sys.stderr,
                )
                sys.exit(2)
            it = rng.choice(items)
            xu[nm] = {"id": str(it["id"]), "text": str(it["text"])}
            rd["customer_concat_rels"] = [_snippet_wav_rel(str(it["id"]))]
            rd.pop("customer_wav", None)
            rd["inter_snippet_gap_ms"] = gap_ms
        if xu:
            plan["extra_uplinks"] = xu

    for rd in rounds:
        name = str(rd.get("name") or "")
        if str(rd.get("mode")) != "barge" or name not in core_map:
            continue
        pkey = str(core_map[name])
        core_list = pool.get(pkey) or []
        if not core_list:
            print(f"customer_pool 缺少列表 {pkey}", file=sys.stderr)
            sys.exit(2)
        core = rng.choice(core_list)
        tangents = (
            []
            if max_t <= 0
            else _pick_tangent_snippets(pool, rng, min_t, max_t)
        )
        pieces: list[dict[str, str]] = [
            {"id": str(t["id"]), "text": str(t["text"])} for t in tangents
        ]
        pieces.append({"id": str(core["id"]), "text": str(core["text"])})
        if len(pieces) > 1:
            rng.shuffle(pieces)
        rels = [_snippet_wav_rel(p["id"]) for p in pieces]
        rd["customer_concat_rels"] = rels
        rd.pop("customer_wav", None)
        rd["inter_snippet_gap_ms"] = gap_ms

        plan["barges"][name] = {
            "order": [p["id"] for p in pieces],
            "texts": [p["text"] for p in pieces],
        }

    return plan


def _resolve_customer_pcm(
    rd: dict[str, Any],
    scen_root: Path,
    repo_root: Path,
    *,
    default_inter_gap_ms: float,
    sr: int,
) -> np.ndarray:
    rels = rd.get("customer_concat_rels")
    gap_ms = float(rd.get("inter_snippet_gap_ms") or default_inter_gap_ms)
    gap_n = _ms_to_samples(sr, gap_ms)
    zgap = np.zeros(gap_n, dtype=np.int16)
    if isinstance(rels, list) and rels:
        acc: np.ndarray | None = None
        for rel in rels:
            r = str(rel).strip()
            cw = scen_root / r if r else Path("")
            if not cw.is_file():
                cw = repo_root / r
            if not cw.is_file():
                print(f"缺少客户 WAV: {r}（请运行 scripts/prepare_rich_call_scenario_wavs.py）", file=sys.stderr)
                sys.exit(2)
            part = _load_mono_16k(cw)
            if acc is None:
                acc = part
            else:
                acc = np.concatenate([acc, zgap, part])
        assert acc is not None
        return acc.astype(np.int16)
    rel = str(rd.get("customer_wav") or "").strip()
    cw = scen_root / rel if rel else Path("")
    if not cw.is_file():
        cw = repo_root / rel
    if not cw.is_file():
        print(f"缺少客户 WAV: {rel}（请运行 scripts/prepare_rich_call_scenario_wavs.py）", file=sys.stderr)
        sys.exit(2)
    return _load_mono_16k(cw)


def _push_pcm_in_chunks(
    session: Any,
    pcm: np.ndarray,
    *,
    sample_rate: int,
    chunk_samples: int,
    record_bufs: tuple[list[np.ndarray], list[np.ndarray]] | None,
    assistant_tts_ref: np.ndarray | None,
    tts_timeline_pos: list[int] | None,
    stats: dict[str, int],
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
            et = ev.get("event")
            if et == "barge_in":
                stats["barge_in"] = stats.get("barge_in", 0) + 1
            if et == "segment_final":
                stats["segment_final"] = stats.get("segment_final", 0) + 1
            print(json.dumps(ev, ensure_ascii=False, default=str)[:2000], flush=True)


def main() -> None:
    load_repo_dotenv()
    vm = _load_voice_llm()

    p = argparse.ArgumentParser(
        description="多轮 LLM 助手（每轮播放前即时 LLM+TTS）+ 预合成客户打断 全双工回放",
    )
    p.add_argument(
        "--scenario",
        type=str,
        default=str(_SCENARIO_DIR / "scenario.json"),
        help="场景 JSON 路径",
    )
    p.add_argument("--models_root", type=str, default="pretrained_models")
    p.add_argument("--device", type=str, default=default_asr_device())
    p.add_argument("--chunk_ms", type=int, default=80)
    p.add_argument("--enable_punc", type=int, default=1)
    p.add_argument("--verbose_vad", action="store_true")
    p.add_argument("--call-audio", action="store_true", help="写立体声通话 WAV")
    p.add_argument(
        "--call-record-dir",
        type=str,
        default="output/call_recordings",
    )
    p.add_argument(
        "--tts-engine",
        type=str,
        choices=("auto", "pyttsx3", "edge"),
        default="auto",
        help="auto：有 edge-tts+ffmpeg 则用 Edge（助手女声更自然）",
    )
    p.add_argument(
        "--edge-voice",
        type=str,
        default=vm.EDGE_VOICE_ASSISTANT_FEMALE_DEFAULT,
        help="助手 Edge 女声（默认晓晓）；客户男声见 prepare 脚本的云希常量",
    )
    p.add_argument(
        "--assistant-edge-rate",
        type=str,
        default="+8%",
        help="助手 edge-tts 语速（默认略快于标准，贴近生活通话；可改 +0%% 等）",
    )
    p.add_argument("--assistant-edge-pitch", type=str, default="+1Hz")
    p.add_argument("--llm-base-url", type=str, default=vm._default_llm_base_url())
    p.add_argument("--llm-model", type=str, default=vm._default_llm_model())
    p.add_argument("--api-key", type=str, default="")
    p.add_argument("--llm-timeout", type=float, default=vm._default_llm_timeout_s())
    p.add_argument(
        "--random-customer-seed",
        type=int,
        default=None,
        help="random_customer 场景下固定随机种子；默认每次随机",
    )
    p.add_argument(
        "--no-reactive-barge",
        action="store_true",
        help="关闭反应式抢话（停播+ASR 接续 LLM），恢复旧版整条助手音轨叠时间轴",
    )
    p.add_argument(
        "--aec",
        type=str,
        choices=("none", "nlms"),
        default="nlms",
        help="反应式推流时是否启用 NLMS 参考 AEC（默认 nlms）；仅非 --no-reactive-barge 时生效",
    )
    p.add_argument(
        "--barge-grace-ms",
        type=float,
        default=480.0,
        help="仿真反应式：每段本地 TTS 开始后多少毫秒内忽略 barge_in（起音+VAD 误触保护，贴近真人反应）",
    )
    p.add_argument(
        "--barge-mic-rms-min",
        type=float,
        default=260.0,
        help="仿真反应式：原始客户麦克块 RMS(int16) 低于此值视为未抢话，抑制 barge_in（避免 AEC 残差假抢话停播）",
    )
    p.add_argument(
        "--barge-loud-bypass-rms",
        type=float,
        default=1400.0,
        help="仿真反应式：麦克 RMS≥此值视为大声叠音，可绕过 grace 窗口仍触发停播",
    )
    p.add_argument(
        "--no-memory-preamble",
        action="store_true",
        help="不在主 LLM 的「话务任务」前拼接通话记忆追踪/摘要（仍可向对话写入「客户·识别」）",
    )
    p.add_argument(
        "--memory-agent-llm",
        action="store_true",
        help="每轮主 LLM 前额外调用「记忆智能体」压缩时间线（多一次 LLM，类似上下文工程中的 ContextBuilder）",
    )
    p.add_argument(
        "--memory-agent-timeout",
        type=float,
        default=18.0,
        help="记忆智能体 LLM 超时（秒）",
    )
    p.add_argument(
        "--no-llm-context-compress",
        action="store_true",
        help="关闭发往 LLM 的 messages 压缩（默认开启：尾部窗口+单条截断，减轻 token 过大导致 500）",
    )
    p.add_argument(
        "--no-llm-request-trace",
        action="store_true",
        help="不在 meta 中记录各次 /chat/completions 请求体（默认记录，便于排查）",
    )
    p.add_argument(
        "--llm-request-trace-max-chars",
        type=int,
        default=32000,
        help="写入 meta 时单条 message content 最大字符，超出追加「已截断」（0=不截断，注意 meta 体积）",
    )
    p.add_argument(
        "--llm-api-max-tail-messages",
        type=int,
        default=12,
        help="压缩后保留的最近非 system 消息条数（含当前 user）",
    )
    p.add_argument(
        "--llm-api-max-user-chars",
        type=int,
        default=1200,
        help="单条 user content 最大字符（超出截断）",
    )
    p.add_argument(
        "--llm-api-max-assistant-chars",
        type=int,
        default=600,
        help="单条 assistant content 最大字符（超出截断）",
    )
    p.add_argument(
        "--llm-api-max-system-chars",
        type=int,
        default=2800,
        help="system 提示最大字符（超出截断）",
    )
    args = p.parse_args()

    scen_path = Path(args.scenario)
    if not scen_path.is_file():
        scen_path = _REPO / args.scenario
    if not scen_path.is_file():
        print(f"找不到 scenario: {args.scenario}", file=sys.stderr)
        sys.exit(2)
    scenario = json.loads(scen_path.read_text(encoding="utf-8"))
    scen_root = scen_path.parent

    rcfg = scenario.get("random_customer")
    default_inter_gap_ms = 160.0
    random_plan: dict[str, Any] | None = None
    if isinstance(rcfg, dict) and rcfg:
        default_inter_gap_ms = float(rcfg.get("inter_snippet_gap_ms", 160))
        seed = args.random_customer_seed
        if seed is None:
            seed = random.randrange(1 << 31)
        random_plan = _apply_random_customer_plan(scenario, scen_root, int(seed))
        print(
            f"# 随机客户音轨 seed={seed}（meta.random_customer_plan 为离线对照；流式 ASR 写入 LLM 对话「客户·识别」）",
            flush=True,
        )

    base_url = str(args.llm_base_url).strip().rstrip("/")
    api_key = vm._resolve_llm_api_key(base_url, (args.api_key or "").strip())
    if not api_key:
        print("请配置 LLM：.env 中 LLM_BASE_URL + LLM_API_KEY（Ollama 11434 可省略密钥）", file=sys.stderr)
        sys.exit(2)

    rounds = scenario.get("rounds") or []
    if len(rounds) < 2:
        print("scenario.rounds 过短", file=sys.stderr)
        sys.exit(2)
    for ri, rd in enumerate(rounds):
        if _round_needs_llm(rd) and not resolve_round_task_body(rd).strip():
            print(
                f"rounds[{ri}] name={rd.get('name')!r} 缺少编导约束："
                f"请在 scenario 中写 llm_user，或在 rich_call_memory_context.ROUND_TASK_CONSTRAINTS 注册",
                file=sys.stderr,
            )
            sys.exit(2)

    tts_engine = _resolve_tts_engine(str(args.tts_engine))
    if tts_engine == "edge":
        print(
            f"# 助手 TTS：edge 女声 {args.edge_voice} rate={args.assistant_edge_rate}",
            flush=True,
        )
    else:
        print(
            "# 助手 TTS：pyttsx3（按女声择中文 SAPI；安装 edge-tts+ffmpeg 可改用 Edge 晓晓）",
            flush=True,
        )

    system_prompt = str(scenario.get("system_prompt") or "").strip()
    gap_ms = float(scenario.get("gap_ms") or 450)
    sr = 16000
    use_reactive = not bool(args.no_reactive_barge)

    if use_reactive:
        print(
            "# 助手侧：listen 仍为整段 JIT；barge 为反应式（按句 TTS、抢话停播 + ASR 接续短答）。",
            flush=True,
        )
    else:
        print(
            "# 助手侧：按剧情顺序 JIT LLM+TTS；barge 为旧版整条音轨叠时间轴（--no-reactive-barge）。",
            flush=True,
        )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    assistant_turns_meta: list[dict[str, Any]] = []
    llm_request_trace: list[dict[str, Any]] | None = (
        [] if not bool(args.no_llm_request_trace) else None
    )
    llm_trace_max = max(0, int(args.llm_request_trace_max_chars))
    call_memory = RichCallMemoryAgent()
    slim_cfg: LlmApiSlimConfig | None = (
        None
        if args.no_llm_context_compress
        else LlmApiSlimConfig(
            enabled=True,
            max_tail_messages=max(4, int(args.llm_api_max_tail_messages)),
            max_user_chars=max(400, int(args.llm_api_max_user_chars)),
            max_assistant_chars=max(200, int(args.llm_api_max_assistant_chars)),
            max_system_chars=max(800, int(args.llm_api_max_system_chars)),
        )
    )
    if slim_cfg is not None:
        print(
            f"# LLM 上下文压缩：tail={slim_cfg.max_tail_messages} 条，"
            f"user≤{slim_cfg.max_user_chars}字 assistant≤{slim_cfg.max_assistant_chars}字",
            flush=True,
        )
    if not args.no_memory_preamble:
        print(
            "# 通话记忆：话务任务前将拼接「追踪」块；"
            + (
                "已启用记忆智能体 LLM 摘要。"
                if args.memory_agent_llm
                else "未启用 --memory-agent-llm（仅结构化追踪）。"
            ),
            flush=True,
        )

    gap_n = _ms_to_samples(sr, gap_ms)
    zgap = np.zeros(gap_n, dtype=np.int16)

    scheduled: list[tuple[Any, ...]] = []
    blocks: list[tuple[np.ndarray, np.ndarray, bool, str]] = []
    first = True
    for ri, rd in enumerate(rounds):
        mode = str(rd.get("mode") or "").strip()
        if not first:
            if use_reactive:
                scheduled.append(("gap", zgap.copy()))
            else:
                blocks.append((zgap.copy(), zgap.copy(), False, "gap"))
        first = False
        if mode == "customer_uplink_only":
            c = _resolve_customer_pcm(
                rd,
                scen_root,
                _REPO,
                default_inter_gap_ms=default_inter_gap_ms,
                sr=sr,
            )
            tail = _ms_to_samples(sr, float(rd.get("post_user_tail_ms") or 450))
            mic = np.concatenate([c, np.zeros(tail, dtype=np.int16)])
            if use_reactive:
                scheduled.append(("uplink", mic, str(rd.get("name"))))
            else:
                ref = np.zeros(len(mic), dtype=np.int16)
                blocks.append((mic, ref, False, f"uplink:{rd.get('name')}"))
        elif mode == "listen":
            if not _round_needs_llm(rd):
                print(f"rounds[{ri}] mode=listen 但缺少 LLM 配置", file=sys.stderr)
                sys.exit(2)
            if use_reactive:
                scheduled.append(("listen", rd, ri))
            else:
                a, reply = _assistant_llm_tts_jit(
                    rd=rd,
                    round_index=ri,
                    messages=messages,
                    vm=vm,
                    api_key=api_key,
                    base_url=base_url,
                    llm_model=str(args.llm_model),
                    llm_timeout=float(args.llm_timeout),
                    tts_engine=tts_engine,
                    edge_voice=str(args.edge_voice),
                    edge_rate=str(args.assistant_edge_rate),
                    edge_pitch=str(args.assistant_edge_pitch),
                    sr=sr,
                    memory=call_memory,
                    memory_agent_llm=bool(args.memory_agent_llm),
                    no_memory_preamble=bool(args.no_memory_preamble),
                    memory_agent_timeout=float(args.memory_agent_timeout),
                    slim_cfg=slim_cfg,
                    llm_request_trace=llm_request_trace,
                    llm_trace_max_chars=llm_trace_max,
                )
                assistant_turns_meta.append({"name": str(rd.get("name")), "text": reply})
                mic = np.zeros(len(a), dtype=np.int16)
                ref = a
                blocks.append((mic, ref, True, f"listen:{rd.get('name')}"))
        elif mode == "barge":
            if not _round_needs_llm(rd):
                print(f"rounds[{ri}] mode=barge 但缺少 LLM 配置", file=sys.stderr)
                sys.exit(2)
            if use_reactive:
                scheduled.append(("barge_rx", rd, ri))
            else:
                a, reply = _assistant_llm_tts_jit(
                    rd=rd,
                    round_index=ri,
                    messages=messages,
                    vm=vm,
                    api_key=api_key,
                    base_url=base_url,
                    llm_model=str(args.llm_model),
                    llm_timeout=float(args.llm_timeout),
                    tts_engine=tts_engine,
                    edge_voice=str(args.edge_voice),
                    edge_rate=str(args.assistant_edge_rate),
                    edge_pitch=str(args.assistant_edge_pitch),
                    sr=sr,
                    memory=call_memory,
                    memory_agent_llm=bool(args.memory_agent_llm),
                    no_memory_preamble=bool(args.no_memory_preamble),
                    memory_agent_timeout=float(args.memory_agent_timeout),
                    slim_cfg=slim_cfg,
                    llm_request_trace=llm_request_trace,
                    llm_trace_max_chars=llm_trace_max,
                )
                assistant_turns_meta.append({"name": str(rd.get("name")), "text": reply})
                lead = _ms_to_samples(sr, float(rd.get("lead_silence_ms") or 5000))
                tail = _ms_to_samples(sr, float(rd.get("post_user_tail_ms") or 550))
                c = _resolve_customer_pcm(
                    rd,
                    scen_root,
                    _REPO,
                    default_inter_gap_ms=default_inter_gap_ms,
                    sr=sr,
                )
                mic = np.concatenate(
                    [
                        np.zeros(lead, dtype=np.int16),
                        c,
                        np.zeros(tail, dtype=np.int16),
                    ]
                )
                n = len(mic)
                ref = np.zeros(n, dtype=np.int16)
                cover = lead + len(c)
                for i in range(min(cover, n)):
                    if i < len(a):
                        ref[i] = a[i]
                blocks.append((mic, ref, True, f"barge:{rd.get('name')}"))
        else:
            print(f"未知 mode: {mode}", file=sys.stderr)
            sys.exit(2)

    full_mic = np.zeros(0, dtype=np.int16)
    full_ref = np.zeros(0, dtype=np.int16)
    if not use_reactive:
        full_mic = np.concatenate([b[0] for b in blocks]) if blocks else np.zeros(0, dtype=np.int16)
        full_ref = np.concatenate([b[1] for b in blocks]) if blocks else np.zeros(0, dtype=np.int16)
        assert len(full_mic) == len(full_ref)

    rec_dir = Path(args.call_record_dir)
    if not rec_dir.is_absolute():
        rec_dir = _REPO / rec_dir
    stem = rec_dir / (
        f"call_record_rich_call_llm_sim_{datetime.datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    )
    call_audio_saved: dict[str, Any] | None = None
    if args.call_audio:
        rec_dir.mkdir(parents=True, exist_ok=True)
        if use_reactive:
            print(
                "# 反应式：立体声通话 WAV 将在 ASR 推流结束后由 duplex 缓冲写出（无法在推流前预知最终 TTS）。",
                flush=True,
            )
        else:
            call_audio_saved = vm._save_call_session_audio(stem, [full_mic], [full_ref], sr)
            print(
                "# 步骤1 完成：对话时间轴（mic+tts）已落盘，可先打开立体声录音收听。",
                flush=True,
            )
            print(
                f"# 立体声 WAV: {call_audio_saved.get('stereo_micL_ttsR_wav', '')}",
                flush=True,
            )

    root = Path(args.models_root)
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
        uttid_prefix="rich_call_sim",
        verbose_vad=args.verbose_vad,
    )
    chunk_samples = max(int(16 * args.chunk_ms), 160)
    stats: dict[str, int] = {}

    if use_reactive:
        print(
            f"# 步骤2：加载 ASR 完毕，反应式 duplex 推流开始 device={args.device} "
            f"scheduled={len(scheduled)} chunk_ms={args.chunk_ms} aec={args.aec} "
            f"barge_grace_ms={args.barge_grace_ms:.0f} barge_mic_rms_min={args.barge_mic_rms_min:.0f} "
            f"barge_loud_bypass_rms={args.barge_loud_bypass_rms:.0f}",
            flush=True,
        )
    else:
        print(
            f"# 步骤2：加载 ASR 完毕，推流开始 device={args.device} "
            f"samples={len(full_mic)} ref对齐 块={len(blocks)}",
            flush=True,
        )
    try:
        if use_reactive:
            aec: NlmsMonoAec | None = None
            if str(args.aec) == "nlms":
                aec = NlmsMonoAec(filter_len=2048, mu=0.25, ref_delay_samples=0)
            elif str(args.aec) != "none":
                print(f"未知 --aec: {args.aec}", file=sys.stderr)
                sys.exit(2)
            rt = DuplexSimRuntime(
                lock=threading.Lock(),
                session=session,
                aec=aec,
                sr=sr,
                record_call_audio=bool(args.call_audio),
                barge_grace_s_after_playback=max(float(args.barge_grace_ms), 0.0) / 1000.0,
                barge_mic_rms16_min=max(float(args.barge_mic_rms_min), 0.0),
                barge_loud_mic_rms16_bypass_grace=max(float(args.barge_loud_bypass_rms), 0.0),
            )
            record_bufs: tuple[list[np.ndarray], list[np.ndarray]] | None = (
                ([], []) if args.call_audio else None
            )
            for step in scheduled:
                sk = str(step[0])
                if sk == "gap":
                    _feed_int16_pcm_duplex(
                        rt,
                        session,
                        step[1],
                        chunk_samples=chunk_samples,
                        record_bufs=record_bufs,
                        stats=stats,
                    )
                elif sk == "uplink":
                    up_utts = _feed_int16_pcm_duplex(
                        rt,
                        session,
                        step[1],
                        chunk_samples=chunk_samples,
                        record_bufs=record_bufs,
                        stats=stats,
                    )
                    _append_customer_asr_to_messages(
                        messages, up_utts, memory=call_memory
                    )
                elif sk == "listen":
                    rd_l, ri_l = step[1], step[2]
                    a, reply = _assistant_llm_tts_jit(
                        rd=rd_l,
                        round_index=int(ri_l),
                        messages=messages,
                        vm=vm,
                        api_key=api_key,
                        base_url=base_url,
                        llm_model=str(args.llm_model),
                        llm_timeout=float(args.llm_timeout),
                        tts_engine=tts_engine,
                        edge_voice=str(args.edge_voice),
                        edge_rate=str(args.assistant_edge_rate),
                        edge_pitch=str(args.assistant_edge_pitch),
                        sr=sr,
                        memory=call_memory,
                        memory_agent_llm=bool(args.memory_agent_llm),
                        no_memory_preamble=bool(args.no_memory_preamble),
                        memory_agent_timeout=float(args.memory_agent_timeout),
                        slim_cfg=slim_cfg,
                        llm_request_trace=llm_request_trace,
                        llm_trace_max_chars=llm_trace_max,
                    )
                    assistant_turns_meta.append(
                        {"name": str(rd_l.get("name")), "text": reply}
                    )
                    with rt.lock:
                        rt.pending_tts = a
                        rt.pending_tts_caption = reply[:200]
                    _ = _feed_int16_pcm_duplex(
                        rt,
                        session,
                        np.zeros(len(a), dtype=np.int16),
                        chunk_samples=chunk_samples,
                        record_bufs=record_bufs,
                        stats=stats,
                    )
                    _silence_until_duplex_playback_done(
                        rt,
                        session,
                        chunk_samples=chunk_samples,
                        sr=sr,
                        record_bufs=record_bufs,
                        stats=stats,
                    )
                elif sk == "barge_rx":
                    rd_b, ri_b = step[1], step[2]
                    lead = _ms_to_samples(sr, float(rd_b.get("lead_silence_ms") or 5000))
                    tail = _ms_to_samples(sr, float(rd_b.get("post_user_tail_ms") or 550))
                    c_b = _resolve_customer_pcm(
                        rd_b,
                        scen_root,
                        _REPO,
                        default_inter_gap_ms=default_inter_gap_ms,
                        sr=sr,
                    )
                    mic_full = np.concatenate(
                        [
                            np.zeros(lead, dtype=np.int16),
                            c_b,
                            np.zeros(tail, dtype=np.int16),
                        ]
                    )
                    _run_reactive_barge_round(
                        rd=rd_b,
                        round_index=int(ri_b),
                        mic_full=mic_full,
                        messages=messages,
                        vm=vm,
                        api_key=api_key,
                        base_url=base_url,
                        llm_model=str(args.llm_model),
                        llm_timeout=float(args.llm_timeout),
                        tts_engine=tts_engine,
                        edge_voice=str(args.edge_voice),
                        edge_rate=str(args.assistant_edge_rate),
                        edge_pitch=str(args.assistant_edge_pitch),
                        sr=sr,
                        chunk_samples=chunk_samples,
                        rt=rt,
                        session=session,
                        record_bufs=record_bufs,
                        stats=stats,
                        assistant_turns_meta=assistant_turns_meta,
                        memory=call_memory,
                        memory_agent_llm=bool(args.memory_agent_llm),
                        no_memory_preamble=bool(args.no_memory_preamble),
                        memory_agent_timeout=float(args.memory_agent_timeout),
                        slim_cfg=slim_cfg,
                        llm_request_trace=llm_request_trace,
                        llm_trace_max_chars=llm_trace_max,
                    )
                else:
                    print(f"内部错误：未知 scheduled 步骤 {sk!r}", file=sys.stderr)
                    sys.exit(2)

            if args.call_audio and record_bufs is not None:
                mic_parts, tts_parts = record_bufs
                if mic_parts:
                    call_audio_saved = vm._save_call_session_audio(
                        stem, mic_parts, tts_parts, sr
                    )
                    print(
                        "# 反应式录音已落盘（立体声 L=客户上行原始块，R=客服 TTS 参考；ASR 仍走 AEC 后流）。",
                        flush=True,
                    )
                    print(
                        f"# 立体声 WAV: {call_audio_saved.get('stereo_micL_ttsR_wav', '')}",
                        flush=True,
                    )
        else:
            pos_holder = [0]
            bi = 0
            for mic_b, _ref_b, playback, tag in blocks:
                if playback:
                    session.begin_local_playback(playback_id=f"rich-{bi}-{tag}")
                    bi += 1
                _push_pcm_in_chunks(
                    session,
                    mic_b,
                    sample_rate=sr,
                    chunk_samples=chunk_samples,
                    record_bufs=None,
                    assistant_tts_ref=full_ref,
                    tts_timeline_pos=pos_holder,
                    stats=stats,
                )
                if playback:
                    session.end_local_playback()
        for ev in session.finalize():
            print(json.dumps(ev, ensure_ascii=False, default=str)[:2000], flush=True)
    finally:
        while session.local_playback_active:
            session.end_local_playback()

    meta: dict[str, Any] = {
        "scenario_id": scenario.get("scenario_id"),
        "llm_model": args.llm_model,
        "llm_base_url": base_url,
        "asr_device": args.device,
        "assistant_tts_engine": tts_engine,
        "assistant_edge_voice": args.edge_voice if tts_engine == "edge" else None,
        "assistant_llm_tts_timing": "just_in_time_before_each_playback",
        "reactive_barge_duplex": use_reactive,
        "duplex_aec": str(args.aec) if use_reactive else None,
        "barge_sim_gate": (
            {
                "grace_ms": float(args.barge_grace_ms),
                "mic_rms16_min": float(args.barge_mic_rms_min),
                "loud_bypass_rms16": float(args.barge_loud_bypass_rms),
                "note": "仅仿真 duplex：需原始麦克能量；小声起讲需过 grace；大声叠音可绕过 grace",
            }
            if use_reactive
            else None
        ),
        "assistant_turns": assistant_turns_meta,
        "stream_stats": stats,
        "call_memory_agent": {
            "preamble_enabled": not bool(args.no_memory_preamble),
            "memory_agent_llm": bool(args.memory_agent_llm),
            "memory_agent_timeout_s": float(args.memory_agent_timeout),
            "task_registry": "examples/rich_call_memory_context.py:ROUND_TASK_CONSTRAINTS",
            "note": "话务任务正文由轮次 name 查注册表，或与 scenario.llm_user 覆盖；动态前言为「通话记忆·追踪」或 --memory-agent-llm 时的「智能体摘要」",
        },
        "llm_context_compress": (
            None
            if slim_cfg is None
            else {
                "enabled": slim_cfg.enabled,
                "max_tail_messages": slim_cfg.max_tail_messages,
                "max_user_chars": slim_cfg.max_user_chars,
                "max_assistant_chars": slim_cfg.max_assistant_chars,
                "max_system_chars": slim_cfg.max_system_chars,
            }
        ),
    }
    if llm_request_trace is not None:
        meta["llm_request_trace"] = llm_request_trace
        meta["llm_request_trace_note"] = (
            "按时间顺序记录每次 POST /chat/completions 的请求体字段（model、temperature、messages）；"
            "不含 API Key 与响应正文。"
            + (
                f" 单条 message content 截断上限={llm_trace_max} 字符。"
                if llm_trace_max > 0
                else " 单条 content 未截断（--llm-request-trace-max-chars 为 0）。"
            )
        )
    if random_plan is not None:
        meta["random_customer_plan"] = random_plan
        meta["random_customer_plan_note"] = (
            "customer_pool 拼接原文仅作离线对照与评测；LLM 侧客户内容以流式 ASR 写入的「客户·识别」为准，不注入 pool 全文。"
        )
    if args.call_audio and call_audio_saved is not None:
        meta["call_audio"] = call_audio_saved
        if use_reactive:
            meta["call_audio_timing"] = (
                "reactive_duplex_record_buffers_concat_after_finalize"
            )
        else:
            meta["call_audio_timing"] = (
                "stereo_and_mono_wavs_written_after_llm_tts_before_asr_stream"
            )

    meta_path = Path(str(stem) + "_meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"# 摘要已写: {meta_path.resolve()}", flush=True)

    b_in = int(stats.get("barge_in", 0))
    bs_in = int(stats.get("barge_in_suppressed", 0))
    sf_in = int(stats.get("segment_final", 0))
    min_bi, min_sf = 3, 3
    mv_exit = scenario.get("matrix_validation")
    if isinstance(mv_exit, dict):
        if "min_barge_in" in mv_exit:
            min_bi = int(mv_exit["min_barge_in"])
        if "min_segment_final" in mv_exit:
            min_sf = int(mv_exit["min_segment_final"])
    print(
        f"# RICH_CALL_SUMMARY barge_in={b_in} barge_in_suppressed={bs_in} segment_final={sf_in} "
        f"(期望 barge_in>={min_bi} 且 segment_final>={min_sf}；"
        f"顺畅场景可在 scenario 中设 matrix_validation)",
        flush=True,
    )
    if b_in < min_bi or sf_in < min_sf:
        print(
            "# 警告：打断/分段少于预期，请检查 lead_silence、客户 WAV 与 VAD，"
            "或在 scenario.json 中配置 matrix_validation。",
            flush=True,
        )
        sys.exit(4)
    print("# 验收通过。", flush=True)


if __name__ == "__main__":
    main()
