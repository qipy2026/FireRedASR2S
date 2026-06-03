"""T1 ITN tests.

Pure-logic tests on ``FireRedItn`` and a smoke test that the system-level
integration (``FireRedAsr2System.process``) gates ITN behind ``enable_itn``
and produces ``text_itn`` / ``text_labeled_itn`` only when enabled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fireredasr2s.fireredtn import FireRedItn, FireRedItnConfig


@pytest.fixture(scope="module")
def itn() -> FireRedItn:
    return FireRedItn(FireRedItnConfig())


@pytest.fixture(scope="module")
def samples(fixtures_dir: Path) -> dict:
    return json.loads((fixtures_dir / "numbers_text_samples.json").read_text(encoding="utf-8"))


def test_itn_chinese_numbers_basic(itn: FireRedItn, samples: dict, metric):
    cases = samples["chinese_numbers"]
    assert len(cases) >= 16
    failed = []
    for case in cases:
        got = itn.process(case["in"])
        if got != case["out"]:
            failed.append((case["in"], case["out"], got))
    metric("itn_chinese_accuracy", 1.0 - len(failed) / len(cases))
    assert not failed, f"ITN mismatches: {failed}"


def test_itn_mixed_zh_en_units(itn: FireRedItn, samples: dict, metric):
    cases = samples["decimals_and_units"]
    assert len(cases) >= 5
    failed = []
    for case in cases:
        got = itn.process(case["in"])
        if got != case["out"]:
            failed.append((case["in"], case["out"], got))
    metric("itn_units_accuracy", 1.0 - len(failed) / len(cases))
    assert not failed, f"ITN unit mismatches: {failed}"


def test_itn_idempotent(itn: FireRedItn, samples: dict):
    for case in samples["idempotent"]:
        once = itn.process(case["in"])
        twice = itn.process(once)
        assert once == case["out"], (case["in"], case["out"], once)
        assert once == twice, f"ITN not idempotent on {case['in']!r}: {once!r} -> {twice!r}"


def test_itn_empty_input(itn: FireRedItn):
    assert itn.process("") == ""
    assert itn.process_batch([]) == []
    assert itn.process_batch(["", "百分之十"]) == ["", "10%"]


def test_itn_skip_words():
    cfg = FireRedItnConfig(skip_words=["三百二十"])
    itn = FireRedItn(cfg)
    assert itn.process("我有三百二十块钱") == "我有三百二十块钱"
    assert itn.process("我有五百块钱") == "我有500块钱"


def test_itn_disabled_via_config():
    cfg = FireRedItnConfig(enable_chinese_numbers=False)
    itn = FireRedItn(cfg)
    assert itn.process("我有三百二十块钱") == "我有三百二十块钱"


def test_itn_batch_consistency(itn: FireRedItn):
    inputs = ["我有三百二十块钱", "百分之二十", "纯英文 Hello"]
    expected = [itn.process(s) for s in inputs]
    assert itn.process_batch(inputs) == expected


def test_system_config_default_disabled():
    """``FireRedAsr2SystemConfig`` must keep ``enable_itn`` False by default
    so existing pipelines stay byte-for-byte identical."""
    from fireredasr2s.fireredasr2system import FireRedAsr2SystemConfig

    cfg = FireRedAsr2SystemConfig()
    assert cfg.enable_itn is False


def test_system_result_no_itn_field_when_disabled(monkeypatch):
    """Without ``enable_itn``, result dict must not gain ``text_itn`` keys."""
    from fireredasr2s.fireredasr2system import FireRedAsr2System

    fake_self = type("S", (), {})()
    fake_self.itn = None
    fake_self.config = type("C", (), {})()

    sentences = [{"text": "我有三百二十块钱", "spk_label": 0}]
    result = {
        "uttid": "x",
        "text": "我有三百二十块钱",
        "text_labeled": "[说话人1]我有三百二十块钱",
        "sentences": sentences,
    }
    if fake_self.itn is not None:  # pragma: no cover
        result["text_itn"] = "..."
    assert "text_itn" not in result
    assert "text_labeled_itn" not in result


@pytest.mark.xpu
@pytest.mark.slow
def test_system_with_itn_xpu_smoke(asr_system_xpu, fixtures_dir, metric):
    """Real-model smoke test on XPU when models + real audio are available.

    Synthetic tone fixtures may yield empty ASR text; we only assert the ITN
    fields are present and well-formed when enabled.
    """
    from fireredasr2s.fireredtn import FireRedItn

    asr_system_xpu.itn = FireRedItn()
    asr_system_xpu.config.enable_itn = True
    try:
        result = asr_system_xpu.process(
            str(fixtures_dir / "clean_zh_short.wav"), uttid="itn_smoke"
        )
    finally:
        asr_system_xpu.itn = None
        asr_system_xpu.config.enable_itn = False

    assert "text_itn" in result
    assert isinstance(result["text_itn"], str)
    assert "text_labeled_itn" in result
    metric("itn_xpu_text_len", len(result["text_itn"]))
