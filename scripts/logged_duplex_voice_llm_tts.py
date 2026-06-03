#!/usr/bin/env python3
"""将 examples/full_duplex_voice_llm_tts.py 的 stdout/stderr 以 UTF-8 写入日志文件，并同时打印到终端。"""
from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--device", type=str, default="xpu")
    p.add_argument("--session-seconds", type=float, default=120.0)
    p.add_argument("extra", nargs="*", help="附加参数传给 full_duplex_voice_llm_tts.py")
    args = p.parse_args()

    repo = args.repo.resolve()
    log_path = args.log.resolve()
    py = repo / ".venv" / "Scripts" / "python.exe"
    if sys.platform != "win32":
        py = repo / ".venv" / "bin" / "python"
    script = repo / "examples" / "full_duplex_voice_llm_tts.py"
    if not py.is_file():
        print(f"missing venv python: {py}", file=sys.stderr)
        sys.exit(2)
    if not script.is_file():
        print(f"missing example: {script}", file=sys.stderr)
        sys.exit(2)

    dev = args.device
    if dev.casefold() == "xpu":
        dev = "xpu"

    cmd: list[str] = [
        str(py),
        str(script),
        "--device",
        dev,
        "--session-seconds",
        str(args.session_seconds),
        *args.extra,
    ]

    env = {
        **os.environ,
        "PYTORCH_ENABLE_XPU_FALLBACK": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as f:
        f.write("=== started " + datetime.datetime.now().isoformat() + " ===\n")
        f.write("=== cmd: " + repr(cmd) + " ===\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            f.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        rc = proc.wait()
        f.write("=== finished rc=%s ===\n" % (rc,))
    sys.exit(rc)


if __name__ == "__main__":
    main()
