#!/usr/bin/env python3
"""接近真实外呼场景的端到端对话测试（编排器）。

流程：准备客户 WAV → 运行 ``full_duplex_rich_call_llm_sim``（反应式 duplex + LLM + TTS + ASR）
→ 校验 meta 与录音文件。

依赖：仓库根 ``.env`` 中 LLM 可用；``pretrained_models`` 下 ASR 栈齐全。

用法::

  .venv\\Scripts\\python.exe scripts\\run_rich_call_conversation_e2e.py
  .venv\\Scripts\\python.exe scripts\\run_rich_call_conversation_e2e.py --device cpu --skip-prepare

成功退出码 0；校验失败 2；子进程非零按其码退出。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_VENV_PY = _REPO / ".venv" / "Scripts" / "python.exe"
if not _VENV_PY.is_file():
    _VENV_PY = _REPO / ".venv" / "bin" / "python"


def _child_env() -> dict[str, str]:
    """子进程 stdout 经管道时须显式 UTF-8，否则 Windows 下常用 cp936 写入，父进程按 utf-8 解会乱码。"""
    e = os.environ.copy()
    e["PYTHONIOENCODING"] = "utf-8"
    e["PYTHONUTF8"] = "1"
    return e


def _venv_python_args(script: str) -> list[str]:
    """``-X utf8`` 与 env 双保险，保证管道捕获时 print/logging 为 UTF-8。"""
    return [str(_VENV_PY), "-X", "utf8", str(script)]


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_child_env(),
    )


def _parse_meta_path(sim_stdout: str) -> Path:
    m = re.search(r"# 摘要已写:\s*(.+\.json)\s*$", sim_stdout, re.MULTILINE)
    if not m:
        raise RuntimeError("仿真 stdout 中未找到「摘要已写」行，无法定位 meta.json")
    return Path(m.group(1).strip())


def _validate_meta(meta_path: Path) -> list[str]:
    errs: list[str] = []
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if not data.get("reactive_barge_duplex"):
        errs.append("期望 reactive_barge_duplex=true（反应式全双工）")
    stats = data.get("stream_stats") or {}
    bi = int(stats.get("barge_in", 0))
    sf = int(stats.get("segment_final", 0))
    if bi < 3:
        errs.append(f"stream_stats.barge_in={bi} 期望 >=3")
    if sf < 3:
        errs.append(f"stream_stats.segment_final={sf} 期望 >=3")
    turns = data.get("assistant_turns") or []
    reactive_n = sum(1 for t in turns if isinstance(t, dict) and t.get("reactive_after_barge"))
    if reactive_n < 3:
        errs.append(
            f"含 reactive_after_barge 的助手轮次={reactive_n}，期望至少 3（三轮 barge）"
        )
    ca = data.get("call_audio") or {}
    for key in ("stereo_micL_ttsR_wav", "mic_asr_wav", "tts_reference_wav"):
        p = ca.get(key)
        if not p or not Path(str(p)).is_file():
            errs.append(f"call_audio.{key} 缺失或不是文件: {p!r}")
        elif Path(str(p)).stat().st_size < 256:
            errs.append(f"call_audio.{key} 过小: {p}")
    plan = data.get("random_customer_plan")
    if not isinstance(plan, dict) or not plan.get("barges"):
        errs.append("期望 meta 含 random_customer_plan.barges（随机客户拼接对照）")
    return errs


def main() -> int:
    p = argparse.ArgumentParser(description="富场景外呼对话端到端测试（编排）")
    p.add_argument("--skip-prepare", action="store_true", help="跳过客户 WAV 准备（已生成过时使用）")
    p.add_argument("--device", type=str, default="", help="ASR 设备，如 cpu / xpu；默认由仓库环境决定")
    p.add_argument(
        "--random-customer-seed",
        type=int,
        default=424242,
        help="固定随机客户拼接，便于复现（默认 424242）",
    )
    p.add_argument(
        "--call-record-dir",
        type=str,
        default="output/call_recordings",
        help="通话录音与 meta 输出目录（相对仓库根）",
    )
    args = p.parse_args()

    if not _VENV_PY.is_file():
        print(f"找不到 venv Python: {_VENV_PY}", file=sys.stderr)
        return 2

    if not args.skip_prepare:
        r = _run(
            _venv_python_args(str(_REPO / "scripts" / "prepare_rich_call_scenario_wavs.py")),
            cwd=_REPO,
        )
        print(r.stdout, end="" if r.stdout.endswith("\n") else r.stdout + "\n", flush=True)
        if r.returncode != 0:
            return r.returncode

    sim_cmd = _venv_python_args(str(_REPO / "examples" / "full_duplex_rich_call_llm_sim.py")) + [
        "--call-audio",
        "--call-record-dir",
        str(args.call_record_dir),
        "--random-customer-seed",
        str(int(args.random_customer_seed)),
    ]
    if str(args.device or "").strip():
        sim_cmd.extend(["--device", str(args.device).strip()])

    r2 = subprocess.run(
        sim_cmd,
        cwd=str(_REPO),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_child_env(),
    )
    print(r2.stdout, end="" if (r2.stdout or "").endswith("\n") else (r2.stdout or "") + "\n", flush=True)
    if r2.returncode != 0:
        return r2.returncode

    try:
        meta_path = _parse_meta_path(r2.stdout or "")
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    errs = _validate_meta(meta_path)
    if errs:
        print("# E2E 校验失败：", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 2

    print("# E2E 对话测试通过（反应式抢话 + 录音 + meta 校验）", flush=True)
    print(f"# meta: {meta_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        from fireredasr2s.win_console_utf8 import ensure_stdio_utf8

        ensure_stdio_utf8()
    except Exception:
        pass
    raise SystemExit(main())
