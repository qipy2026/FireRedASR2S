#!/usr/bin/env python3
"""根据 examples/duplex_scenario_rich/scenario.json 预生成 16k 单声道 WAV。

默认使用 Edge TTS：问候 + 助手 = 女声晓晓；用户句 = 男声云希；语速略降以更自然。
需: pip install edge-tts 且 ffmpeg 在 PATH。失败时回退 pyttsx3（按角色在 SAPI 中匹配中文明细男/女）。

加 --force 可覆盖已存在文件。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# 略慢、略抑扬（可按需改）
EDGE_RATE = "-5%"
EDGE_PITCH = "+1Hz"


def _load_voice_tts_mod():
    p = _REPO / "examples" / "full_duplex_voice_llm_tts.py"
    spec = importlib.util.spec_from_file_location("_fdv_prepare", p)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--engine",
        choices=("edge", "pyttsx3", "auto"),
        default="auto",
        help="auto：有 edge-tts+ffmpeg 用 edge，否则 pyttsx3",
    )
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 wav")
    args = ap.parse_args()

    scen_dir = _REPO / "examples" / "duplex_scenario_rich"
    manifest = scen_dir / "scenario.json"
    if not manifest.is_file():
        print(f"missing {manifest}", file=sys.stderr)
        sys.exit(2)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    wav_root = scen_dir
    voicemod = _load_voice_tts_mod()
    synthesize = voicemod.synthesize_tts_16k_int16
    VOICE_FEMALE = getattr(voicemod, "EDGE_VOICE_ASSISTANT_FEMALE_DEFAULT", "zh-CN-XiaoxiaoNeural")
    VOICE_MALE = getattr(voicemod, "EDGE_VOICE_CUSTOMER_MALE_DEFAULT", "zh-CN-YunxiNeural")

    if args.engine == "pyttsx3":
        use_edge = False
    elif args.engine == "edge":
        use_edge = True
    else:
        use_edge = False
        try:
            import edge_tts  # noqa: F401

            use_edge = _have_ffmpeg()
        except ImportError:
            use_edge = False
    if args.engine == "edge" and not _have_ffmpeg():
        print("ffmpeg 不在 PATH，无法使用 edge 管线", file=sys.stderr)
        sys.exit(4)

    jobs: list[tuple[Path, str, str, str]] = []
    g = data.get("greeting") or {}
    jobs.append((wav_root / g["wav"], g["text"], VOICE_FEMALE, "greeting"))
    for t in data.get("turns") or []:
        jobs.append((wav_root / t["user_wav"], t["user_text"], VOICE_MALE, "user"))
        jobs.append((wav_root / t["assistant_wav"], t["assistant_text"], VOICE_FEMALE, "assistant"))

    for path, text, voice, role in jobs:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and not args.force:
            print(f"skip exists: {path.relative_to(_REPO)}")
            continue
        text = (text or "").strip()
        if not text:
            print(f"empty text for {path}", file=sys.stderr)
            sys.exit(3)
        pcm: object = None
        eng_lbl = "pyttsx3"
        if use_edge:
            try:
                pcm = synthesize(
                    text,
                    "edge",
                    voice,
                    edge_rate=EDGE_RATE,
                    edge_pitch=EDGE_PITCH,
                    pyttsx3_gender=("female" if role in ("greeting", "assistant") else "male"),
                )
                eng_lbl = "edge"
            except Exception as e:
                print(f"edge 失败 ({path.name}): {e}", file=sys.stderr)
                if args.engine == "edge":
                    sys.exit(5)
                pcm = None
        if pcm is None:
            print(f"fallback pyttsx3: {path.relative_to(_REPO)} ({role})", flush=True)
            gender = "female" if role in ("greeting", "assistant") else "male"
            pcm = synthesize(
                text,
                "pyttsx3",
                voice,
                edge_rate=EDGE_RATE,
                edge_pitch=EDGE_PITCH,
                pyttsx3_gender=gender,
            )
            eng_lbl = "pyttsx3"
        arr = pcm
        if len(arr) == 0:
            print(f"empty tts: {path}", file=sys.stderr)
            sys.exit(3)
        sf.write(str(path), arr, 16000, subtype="PCM_16")
        print(
            f"wrote {path.relative_to(_REPO)} ({len(arr) / 16000:.2f}s) "
            f"[{eng_lbl} {role} {voice}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
