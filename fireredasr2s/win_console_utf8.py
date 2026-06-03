"""Windows 控制台与 stdio 尽量使用 UTF-8（在 ``logging.basicConfig`` 之前调用）。

避免中文日志、JSON 在 cp936 控制台下变成乱码；可多次调用（幂等）。
"""

from __future__ import annotations

import os
import sys


def ensure_stdio_utf8() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if sys.platform != "win32":
        return
    try:
        import ctypes

        _cp_utf8 = 65001
        ctypes.windll.kernel32.SetConsoleOutputCP(_cp_utf8)  # type: ignore[attr-defined]
        ctypes.windll.kernel32.SetConsoleCP(_cp_utf8)  # type: ignore[attr-defined]
    except Exception:
        pass
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
