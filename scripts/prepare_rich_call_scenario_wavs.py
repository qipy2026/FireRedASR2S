#!/usr/bin/env python3
"""为 examples/duplex_rich_call_scenario 生成客户侧 WAV（16k 单声道）。

从 ``customer_pool.json`` 读取全部话术片段，生成 ``wavs/cust_<id>.wav``。

**同一批次只使用一种 TTS 引擎**（整批 Edge 云希男声 **或** 整批 pyttsx3 男声），避免混引擎导致「像换一个人」。
pyttsx3 时在 Windows SAPI 中按**男声**匹配中文音色（与客服女声脚本逻辑对称）。
``auto``：先试 Edge（含重试），不行则**全部**改用 pyttsx3。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

SCEN = _REPO / "examples" / "duplex_rich_call_scenario"
POOL_PATH = SCEN / "customer_pool.json"

EDGE_RATE = "-12%"
EDGE_PITCH = "+2Hz"


def _load_voice_tts_mod():
    p = _REPO / "examples" / "full_duplex_voice_llm_tts.py"
    spec = importlib.util.spec_from_file_location("_fdv_prich", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _iter_pool_lines(pool: dict) -> list[tuple[str, str]]:
    """返回 (wav 文件名, 文本)。wav 名为 cust_<id>.wav"""
    out: list[tuple[str, str]] = []
    for _key, v in pool.items():
        if not isinstance(v, list) or not v:
            continue
        if not isinstance(v[0], dict):
            continue
        if "id" not in v[0] or "text" not in v[0]:
            continue
        for item in v:
            sid = str(item["id"]).strip()
            text = str(item["text"]).strip()
            out.append((f"cust_{sid}.wav", text))
    return out


def _synthesize_with_retries(
    synthesize,
    text: str,
    engine: str,
    edge_voice_id: str,
    *,
    retries: int,
) -> object:
    last: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            return synthesize(
                text,
                engine,
                edge_voice_id,
                edge_rate=EDGE_RATE,
                edge_pitch=EDGE_PITCH,
                pyttsx3_gender="male",
            )
        except Exception as e:
            last = e
            if attempt + 1 < max(1, retries):
                time.sleep(0.6 * (attempt + 1))
    assert last is not None
    raise last


def main() -> None:
    try:
        from fireredasr2s.win_console_utf8 import ensure_stdio_utf8

        ensure_stdio_utf8()
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("edge", "pyttsx3", "auto"), default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--edge-retries",
        type=int,
        default=4,
        help="单条 Edge 合成失败时的重试次数（整批统一引擎时每条共用）",
    )
    args = ap.parse_args()

    if not POOL_PATH.is_file():
        print(f"# 缺少话术池: {POOL_PATH}", file=sys.stderr)
        sys.exit(2)

    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    lines = _iter_pool_lines(pool)
    if not lines:
        print("# customer_pool.json 未解析到任何片段", file=sys.stderr)
        sys.exit(2)

    out_dir = SCEN / "wavs"
    out_dir.mkdir(parents=True, exist_ok=True)
    voicemod = _load_voice_tts_mod()
    synthesize = voicemod.synthesize_tts_16k_int16
    voice_male = getattr(voicemod, "EDGE_VOICE_CUSTOMER_MALE_DEFAULT", "zh-CN-YunxiNeural")

    if args.engine == "pyttsx3":
        chosen = "pyttsx3"
    elif args.engine == "edge":
        chosen = "edge"
        try:
            import edge_tts  # noqa: F401
        except ImportError as e:
            print("# 未安装 edge-tts", e, file=sys.stderr)
            sys.exit(2)
        if not _have_ffmpeg():
            print("# Edge 需要 ffmpeg", file=sys.stderr)
            sys.exit(2)
    else:
        chosen = "edge"
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            chosen = "pyttsx3"
        if chosen == "edge" and not _have_ffmpeg():
            chosen = "pyttsx3"
        if chosen == "edge":
            try:
                _synthesize_with_retries(
                    synthesize,
                    "嗯",
                    "edge",
                    voice_male,
                    retries=args.edge_retries,
                )
                print("# auto：本批次客户 WAV 统一使用 Edge 云希", flush=True)
            except Exception as e:
                print(
                    f"# auto：Edge 不可用（{e}），本批次**全部**改用 pyttsx3 以统一声线",
                    flush=True,
                )
                chosen = "pyttsx3"
        else:
            print("# auto：未满足 edge+ffmpeg，本批次客户 WAV 统一 pyttsx3", flush=True)

    print(f"# 客户 TTS 引擎（整批一致）: {chosen}", flush=True)

    for name, text in lines:
        path = out_dir / name
        if path.is_file() and not args.force:
            print(f"# 跳过已存在 {path.name}", flush=True)
            continue
        try:
            pcm = _synthesize_with_retries(
                synthesize,
                text,
                chosen,
                voice_male,
                retries=args.edge_retries if chosen == "edge" else 2,
            )
        except Exception as e:
            print(f"# 合成失败 {path.name}: {e}", file=sys.stderr)
            sys.exit(1)
        sf.write(str(path), pcm, 16000, subtype="PCM_16")
        print(f"# 写入 {path.relative_to(_REPO)} ({len(pcm) / 16000:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
