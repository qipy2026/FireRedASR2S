"""Self-tests for the T9 report writer.

These avoid invoking real pytest/torch — they synthesize a junit XML, an
``env.json`` snapshot, and assert ``scripts._report_writer`` produces a
well-formed ``test_report.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts._report_writer import parse_junit, render_markdown, write_report  # noqa: E402


JUNIT_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" tests="3" failures="1" skipped="1">
  <testcase classname="tests.test_itn" name="test_pass_with_metric"
            file="tests/test_itn.py" time="0.10">
    <properties>
      <property name="metric_itn_chinese_accuracy" value="1.0"/>
    </properties>
  </testcase>
  <testcase classname="tests.test_denoise" name="test_failing"
            file="tests/test_denoise.py" time="0.20">
    <failure message="WER did not drop">AssertionError: ...</failure>
  </testcase>
  <testcase classname="tests.test_diar_align" name="test_skipped"
            file="tests/test_diar_align.py" time="0.00">
    <skipped message="modelscope not installed"/>
  </testcase>
</testsuite>
"""


def test_parse_junit_extracts_status_and_metrics(tmp_path: Path):
    p = tmp_path / "junit.xml"
    p.write_text(JUNIT_TEMPLATE, encoding="utf-8")
    cases, _ = parse_junit(p)
    assert len(cases) == 3
    by_name = {c.name: c for c in cases}
    assert by_name["test_pass_with_metric"].status == "PASS"
    assert by_name["test_pass_with_metric"].metrics["itn_chinese_accuracy"] == "1.0"
    assert by_name["test_failing"].status == "FAIL"
    assert by_name["test_skipped"].status == "SKIP"
    tasks = {c.task for c in cases}
    assert {"T1", "T2", "T6"}.issubset(tasks)


def test_render_markdown_contains_all_sections(tmp_path: Path):
    p = tmp_path / "junit.xml"
    p.write_text(JUNIT_TEMPLATE, encoding="utf-8")
    cases, attrs = parse_junit(p)
    md = render_markdown(
        cases=cases,
        suite_attrs=attrs,
        env={"platform": "win32", "python": "3.12.10", "torch": "2.11.0+xpu", "has_xpu": True},
        html_link="test_report.html",
    )
    assert "FireRedASR2S 迭代验收测试报告" in md
    assert "PASS" in md and "FAIL" in md and "SKIP" in md
    assert "T1 ITN" in md and "T2 独立降噪前端" in md and "T6 声纹分离输入与对齐" in md
    assert "metric_itn_chinese_accuracy" not in md
    assert "itn_chinese_accuracy=1.0" in md
    assert "modelscope not installed" in md
    assert "复现命令" in md


def test_write_report_creates_md(tmp_path: Path):
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT_TEMPLATE, encoding="utf-8")
    env = tmp_path / "env.json"
    env.write_text(json.dumps({"torch": "2.11.0+xpu", "has_xpu": True}), encoding="utf-8")
    out = tmp_path / "test_report.md"
    summary = write_report(junit, env, out, html_link="test_report.html")
    assert out.exists()
    md = out.read_text(encoding="utf-8")
    assert "复现命令" in md
    assert summary["total"] == 3
    assert summary["pass"] == 1
    assert summary["fail"] == 1
    assert summary["skip"] == 1


def test_record_metric_pipeline(metric, request: pytest.FixtureRequest):
    """End-to-end: ``metric`` fixture appends user_properties; junit then carries
    them so the report writer can surface them."""
    metric("smoke_e2e", 0.999)
    props = dict(request.node.user_properties)
    assert props.get("metric_smoke_e2e") == 0.999
