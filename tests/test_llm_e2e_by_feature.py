"""LLM 相关功能的端到端 / 集成测试（合成或真实录音 + 调用 LLM 路径）。

默认用 **Mock** 模拟 HTTP，无需密钥；CI 可全量收集。

- **真实 LLM（外网）**：设置 ``FIREREDASR2S_LLM_E2E_NET=1``，并配置 ``.env`` 或环境中的
  ``LLM_BASE_URL`` / ``LLM_API_KEY``（或 OpenAI/Ollama 等价变量），再跑带 ``net`` 标记的用例。

- **ASR → LLM**：依赖 ``asr_system_xpu``（与 ``test_e2e_by_feature`` 相同模型前置），对
  ``tests/fixtures/clean_zh_short.wav`` 做 ``process``，将识别文本送入富场景 ``_llm_append_user``
  （VM 仍为 Mock，避免子进程）。

示例::

  .venv\\\\Scripts\\\\python.exe -m pytest tests/test_llm_e2e_by_feature.py -m e2e -v
  $env:FIREREDASR2S_LLM_E2E_NET=\"1\"; pytest tests/test_llm_e2e_by_feature.py -m \"e2e and net\" -v
"""

from __future__ import annotations

import io
import json
import os
import sys
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).resolve().parent.parent
_EXAMPLES = _REPO / "examples"
for _p in (_REPO, _EXAMPLES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import full_duplex_rich_call_llm_sim as _rich_sim  # noqa: E402
import full_duplex_voice_llm_tts as _vllm  # noqa: E402
from rich_call_memory_context import (  # noqa: E402
    RichCallMemoryAgent,
    compose_task_user_content,
    resolve_round_task_body,
)


E2E_WAV_CLEAN = "clean_zh_short.wav"


def _require_clean_fixture(fixtures_dir: Path) -> Path:
    p = fixtures_dir / E2E_WAV_CLEAN
    if not p.is_file():
        pytest.skip(
            f"缺少 {E2E_WAV_CLEAN}，请执行: python scripts/generate_test_fixtures.py"
        )
    return p


class _FakeHTTPResponse(io.BytesIO):
    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        pass


@pytest.mark.e2e
def test_e2e_llm_openai_compatible_chat_parses_assistant_content() -> None:
    """功能：OpenAI 兼容 ``/chat/completions`` 响应解析为 assistant 文本。"""
    payload = {
        "choices": [{"message": {"content": "  移动幺零零八六客服为您服务。  "}}]
    }
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _fake_urlopen(req, timeout=None):  # noqa: ANN001
        return _FakeHTTPResponse(raw)

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        out = _vllm.openai_compatible_chat(
            [{"role": "user", "content": "你好"}],
            api_key="sk-test",
            base_url="https://example.invalid/v1",
            model="dummy",
            timeout_s=5.0,
        )
    assert "幺零零八六" in out


@pytest.mark.e2e
def test_e2e_llm_openai_compatible_chat_http_error_message() -> None:
    """功能：HTTP 错误时抛出带正文的 ``RuntimeError``（便于排障）。"""

    class _Err(urllib.error.HTTPError):
        def __init__(self) -> None:
            super().__init__("http://x", 401, "Unauthorized", hdrs=None, fp=None)

        def read(self):  # noqa: ANN201
            return b'{"error":"bad key"}'

    def _fake_urlopen(*a, **k):  # noqa: ANN002
        raise _Err()

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        with pytest.raises(RuntimeError) as ei:
            _vllm.openai_compatible_chat(
                [{"role": "user", "content": "x"}],
                api_key="bad",
                base_url="https://example.invalid/v1",
                model="m",
                timeout_s=2.0,
            )
    assert "401" in str(ei.value)


@pytest.mark.e2e
def test_e2e_llm_api_slim_truncates_tail_and_content() -> None:
    """功能：发往 LLM 的 messages 尾部条数与单条字符上限（减轻网关/超长）。"""
    cfg = _rich_sim.LlmApiSlimConfig(
        enabled=True,
        max_tail_messages=2,
        max_user_chars=12,
        max_assistant_chars=10,
        max_system_chars=40,
    )
    long = "abcdefghijklmnopqrstuvwxyz"
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "S" * 80},
        {"role": "user", "content": "drop1"},
        {"role": "assistant", "content": "drop2"},
        {"role": "user", "content": long},
        {"role": "assistant", "content": long},
    ]
    slim = _rich_sim._messages_for_openai_api(messages, cfg)
    assert any("省略" in str(m.get("content", "")) for m in slim)
    sys_msg = next(m for m in slim if m.get("role") == "system")
    assert "截断" in sys_msg["content"] or len(sys_msg["content"]) <= 45
    for m in slim:
        if m.get("role") == "user" and "省略" not in str(m.get("content")):
            assert len(str(m.get("content"))) <= 13


@pytest.mark.e2e
def test_e2e_rich_call_memory_agent_llm_summary_mock_vm() -> None:
    """功能：记忆智能体调用 LLM 压缩时间线并写入 ``agent_summary``。"""
    mem = RichCallMemoryAgent()
    mem.add_customer_asr("我想查一下本月套餐")
    mem.add_assistant_spoken("好的，帮您看一下")

    class _Vm:
        def openai_compatible_chat(self, messages, **kwargs):  # noqa: ANN001
            assert any("记忆智能体" in str(m.get("content", "")) for m in messages)
            return "摘要：客户要查套餐；客服已应答。"

    mem.refresh_summary_with_llm(
        vm=_Vm(),
        api_key="k",
        base_url="http://127.0.0.1:11434/v1",
        model="m",
        timeout_s=5.0,
    )
    assert "摘要" in mem.agent_summary
    pre = mem.render_preamble_for_task()
    assert "智能体摘要" in pre


@pytest.mark.e2e
def test_e2e_compose_task_user_content_with_round_registry() -> None:
    """功能：按轮次 ``name`` 解析编导约束并与通话记忆前言拼装。"""
    mem = RichCallMemoryAgent()
    mem.add_customer_asr("喂")
    rd = {"name": "opening_verify"}
    body = resolve_round_task_body(rd)
    assert "张先生" in body or "核身" in body or "开场" in body
    u = compose_task_user_content(rd, mem, use_memory_preamble=True)
    assert "【本轮编导约束】" in u
    assert body[:20] in u or "开场" in u


@pytest.mark.e2e
def test_e2e_append_customer_asr_to_messages_updates_memory() -> None:
    """功能：流式 ASR 片段合并为「客户·识别」并写入对话 messages 与记忆。"""
    messages: list[dict[str, str]] = [{"role": "system", "content": "你是客服"}]
    mem = RichCallMemoryAgent()
    _rich_sim._append_customer_asr_to_messages(
        messages, ["  第一句  ", "", "第二句"], memory=mem
    )
    assert messages[-1]["role"] == "user"
    assert "【客户·识别】" in messages[-1]["content"]
    assert "第一句" in messages[-1]["content"] and "第二句" in messages[-1]["content"]
    assert mem.customer_batches and "第一句" in mem.customer_batches[-1]


@pytest.mark.e2e
@pytest.mark.xpu
@pytest.mark.slow
def test_e2e_asr_wav_then_rich_round_llm_mock(
    asr_system_xpu, fixtures_dir: Path, metric
) -> None:
    """功能：合成/真实录音经 ASR 后，富场景主轮 ``_llm_append_user`` 调用 LLM（Mock）。"""
    wav = _require_clean_fixture(fixtures_dir)
    r = asr_system_xpu.process(str(wav), uttid="e2e_llm_asr")
    customer_line = (r.get("text") or "").strip() or "（无识别文本）"
    metric("e2e_llm_asr_text_len", len(customer_line))

    mem = RichCallMemoryAgent()
    mem.add_customer_asr(customer_line)
    rd = {"name": "bridge_to_biz"}
    user_text = compose_task_user_content(rd, mem, use_memory_preamble=True)
    assert user_text

    messages: list[dict[str, str]] = [
        {"role": "system", "content": "你是中国移动外呼客服，回答简短。"},
    ]
    vm = MagicMock()
    vm.openai_compatible_chat.return_value = "谢谢接听。请问您是要咨询套餐变更吗？幺零零八六也可以办理。"

    reply = _rich_sim._llm_append_user(
        messages,
        user_text,
        vm=vm,
        api_key="mock",
        base_url="http://127.0.0.1:1/v1",
        llm_model="mock",
        llm_timeout=30.0,
        round_label="e2e_bridge",
        slim_cfg=_rich_sim.LlmApiSlimConfig(enabled=True),
    )
    assert reply
    assert "幺零零八六" in reply or len(reply) >= 4
    vm.openai_compatible_chat.assert_called_once()
    kw = vm.openai_compatible_chat.call_args.kwargs
    assert kw.get("model") == "mock"
    payload = vm.openai_compatible_chat.call_args.args[0]
    assert isinstance(payload, list) and len(payload) >= 2


@pytest.mark.e2e
@pytest.mark.net
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FIREREDASR2S_LLM_E2E_NET", "").strip().lower()
    not in ("1", "true", "yes"),
    reason="设置 FIREREDASR2S_LLM_E2E_NET=1 以启用真实 LLM HTTP（需网络与密钥）",
)
def test_e2e_llm_openai_compatible_chat_live_minimal() -> None:
    """功能：真实 OpenAI 兼容端点最短对话（需 .env 或环境变量中的 URL/KEY）。"""
    try:
        from fireredasr2s.repo_dotenv import load_repo_dotenv

        load_repo_dotenv(override=False)
    except Exception:
        pass

    base = _vllm._default_llm_base_url()
    key = _vllm._resolve_llm_api_key(base, "")
    if not key:
        pytest.skip("未配置 LLM_API_KEY / OPENAI_API_KEY，且非本机 Ollama 占位场景")

    model = _vllm._default_llm_model()
    out = _vllm.openai_compatible_chat(
        [{"role": "user", "content": "只回复一个字：好"}],
        api_key=key,
        base_url=base,
        model=model,
        timeout_s=min(90.0, float(_vllm._default_llm_timeout_s())),
    )
    assert len((out or "").strip()) >= 1
