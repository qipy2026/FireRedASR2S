"""Load repository-root ``.env`` into ``os.environ`` for examples and CLI.

Default **does not override** variables already set in the shell. Supports
``python-dotenv`` when installed; otherwise a small line parser (no multiline values).
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def strip_env_quotes(val: str) -> str:
    v = val.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1].strip()
    return v


def _parse_dotenv_lines(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        if not key:
            continue
        out[key] = strip_env_quotes(v)
    return out


def load_repo_dotenv(override: bool = False) -> bool:
    """Merge ``<repo>/.env`` into ``os.environ``. 若存在并已读取则返回 True。"""
    path = repo_root() / ".env"
    if not path.is_file():
        return False
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=override)
        return True
    except ImportError:
        pass
    try:
        parsed = _parse_dotenv_lines(path.read_text(encoding="utf-8"))
    except OSError:
        return False
    for key, val in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = val
    return True


def default_asr_device(fallback: str = "xpu") -> str:
    """``--device`` 默认值：仅读 ``FIRERED_ASR_DEVICE``，否则 *fallback*（默认 xpu）。

    故意 **不读** 通用 ``ASR_DEVICE``，以免合并其它项目 ``.env`` 时把默认设成 ``cpu``。
    需要 CPU 推理时请显式传 ``--device cpu``。
    """
    v = strip_env_quotes((os.environ.get("FIRERED_ASR_DEVICE") or "").strip())
    return v if v else fallback
