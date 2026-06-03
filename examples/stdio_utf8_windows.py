"""在导入会写 logging/print 的模块之前调用 ``apply_stdio_utf8()``。

实现位于 ``fireredasr2s.win_console_utf8``（与 ``fireredasr2system`` 共用），保证库侧日志与示例输出编码一致。
"""

from __future__ import annotations

from fireredasr2s.win_console_utf8 import ensure_stdio_utf8


def apply_stdio_utf8() -> None:
    ensure_stdio_utf8()
