"""富场景外呼对话端到端测试（可选，默认跳过）。

需 LLM + 全量 ASR 模型，耗时数分钟。仅当设置环境变量 ``RUN_RICH_CALL_CONVERSATION_E2E=1`` 时执行。

按功能的轻量 LLM E2E（Mock + 可选真实 HTTP + ASR→LLM）见 ``tests/test_llm_e2e_by_feature.py``。

  RUN_RICH_CALL_CONVERSATION_E2E=1 .venv\\Scripts\\python.exe -m pytest tests/test_rich_call_conversation_e2e.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_VENV_PY = _REPO / ".venv" / "Scripts" / "python.exe"
if not _VENV_PY.is_file():
    _VENV_PY = _REPO / ".venv" / "bin" / "python"


@pytest.mark.skipif(
    os.environ.get("RUN_RICH_CALL_CONVERSATION_E2E", "").strip() not in ("1", "true", "yes"),
    reason="设置 RUN_RICH_CALL_CONVERSATION_E2E=1 以运行（需 LLM 与完整模型）",
)
def test_rich_call_conversation_e2e_orchestrator() -> None:
    assert _VENV_PY.is_file(), f"需要项目 venv: {_VENV_PY}"
    script = _REPO / "scripts" / "run_rich_call_conversation_e2e.py"
    assert script.is_file()
    r = subprocess.run(
        [str(_VENV_PY), "-X", "utf8", str(script), "--skip-prepare"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        },
    )
    if r.returncode != 0:
        sys.stderr.write(r.stdout or "")
        sys.stderr.write(r.stderr or "")
    assert r.returncode == 0, f"E2E 退出码 {r.returncode}"
