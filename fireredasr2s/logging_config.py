# Copyright 2026 Xiaohongshu.
"""根 logger 默认写到 stdout。

PowerShell 在 ``command 2>&1`` 时会把子进程 stderr 包装成 ``NativeCommandError``；
标准库的 ``logging.basicConfig()`` 默认使用 stderr，导致 INFO 也像报错。统一改到 stdout
可避免该误报（仍可用环境变量将真正的错误信息写到 stderr，例如库内部的 print）。"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
