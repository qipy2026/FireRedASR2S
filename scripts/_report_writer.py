"""Markdown summarizer for the FireRedASR2S iteration test matrix (T9).

Reads a pytest junit XML + an environment dump and renders a human-readable
``test_report.md`` grouped by task (T0/T1/T2/...). The pytest run is expected
to attach numeric metrics via ``record_metric(request, "metric_name", value)``
defined in ``tests/conftest.py``; those land in junit as
``<property name="metric_xxx" value="..."/>`` and we surface them per task.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


TASK_LABELS: dict[str, str] = {
    "T0": "T0 测试基础设施",
    "T1": "T1 ITN（逆文本正则化）",
    "T2": "T2 独立降噪前端",
    "T3": "T3 量化与加速兜底",
    "T4": "T4 自定义热词偏置",
    "T5": "T5 加速运行时骨架",
    "T6": "T6 声纹分离输入与对齐",
    "T7": "T7 声纹分离多 backend",
    "T8": "T8 声纹注册",
    "E2E": "E2E 录音端到端（按功能点）",
}

TASK_BY_FILE: dict[str, str] = {
    "test_infra_smoke": "T0",
    "test_report_meta": "T0",
    "test_xpu_device": "T0",
    "test_itn": "T1",
    "test_denoise": "T2",
    "test_compute_dtype": "T3",
    "test_hotword": "T4",
    "test_runtime": "T5",
    "test_diar_align": "T6",
    "test_diar_backends": "T7",
    "test_speaker_enroll": "T8",
    "test_e2e_by_feature": "E2E",
    "test_llm_e2e_by_feature": "E2E",
}

UNKNOWN_TASK_LABEL = "T? 其他/未归类"


@dataclass
class Case:
    file: str
    classname: str
    name: str
    status: str
    time_s: float
    message: str = ""
    metrics: dict[str, str] = field(default_factory=dict)

    @property
    def task(self) -> str:
        haystack = f"{self.file} {self.classname}".lower()
        for key, task in TASK_BY_FILE.items():
            if key in haystack:
                return task
        return "T?"

    @property
    def display(self) -> str:
        if self.file:
            return f"{Path(self.file).stem}::{self.name}"
        if self.classname:
            return f"{self.classname.rsplit('.', 1)[-1]}::{self.name}"
        return self.name


def _classify_status(case_el: ET.Element) -> tuple[str, str]:
    if case_el.find("failure") is not None:
        return "FAIL", (case_el.find("failure").get("message") or "")  # type: ignore[union-attr]
    if case_el.find("error") is not None:
        return "ERROR", (case_el.find("error").get("message") or "")  # type: ignore[union-attr]
    if case_el.find("skipped") is not None:
        return "SKIP", (case_el.find("skipped").get("message") or "")  # type: ignore[union-attr]
    return "PASS", ""


def parse_junit(junit_path: Path) -> tuple[list[Case], dict[str, str]]:
    if not junit_path.exists():
        return [], {}
    tree = ET.parse(junit_path)
    root = tree.getroot()
    suite = root if root.tag == "testsuite" else (root.find("testsuite") or root)

    cases: list[Case] = []
    for c in suite.iter("testcase"):
        status, message = _classify_status(c)
        metrics: dict[str, str] = {}
        props_el = c.find("properties")
        if props_el is not None:
            for p in props_el.findall("property"):
                name = p.get("name", "")
                if name.startswith("metric_"):
                    metrics[name[len("metric_") :]] = p.get("value", "")
        cases.append(
            Case(
                file=c.get("file", ""),
                classname=c.get("classname", ""),
                name=c.get("name", ""),
                status=status,
                time_s=float(c.get("time", "0") or 0.0),
                message=message,
                metrics=metrics,
            )
        )

    suite_attrs = dict(suite.attrib)
    return cases, suite_attrs


def _group_by_task(cases: Iterable[Case]) -> dict[str, list[Case]]:
    groups: dict[str, list[Case]] = {k: [] for k in TASK_LABELS}
    for c in cases:
        groups.setdefault(c.task, []).append(c)
    return {k: v for k, v in groups.items() if v}


def _label(task: str) -> str:
    return TASK_LABELS.get(task, UNKNOWN_TASK_LABEL)


def _md_status(s: str) -> str:
    return {"PASS": "PASS", "FAIL": "FAIL", "ERROR": "ERROR", "SKIP": "SKIP"}.get(s, s)


def _summarize(cases: list[Case]) -> tuple[int, int, int, int, float]:
    p = sum(1 for c in cases if c.status == "PASS")
    f = sum(1 for c in cases if c.status in ("FAIL", "ERROR"))
    s = sum(1 for c in cases if c.status == "SKIP")
    t = sum(c.time_s for c in cases)
    return p, f, s, len(cases), t


def render_markdown(
    cases: list[Case],
    suite_attrs: dict[str, str],
    env: dict,
    html_link: str | None,
    started_at: datetime | None = None,
) -> str:
    started_at = started_at or datetime.now()
    p, f, s, total, t = _summarize(cases)
    lines: list[str] = []
    lines.append("# FireRedASR2S 迭代验收测试报告")
    lines.append("")
    lines.append(
        f"- 生成时间: `{started_at.strftime('%Y-%m-%d %H:%M:%S')}`  "
        f"  总用时: `{t:.2f}s`  "
        f"  PASS: **{p}**  FAIL/ERROR: **{f}**  SKIP: **{s}**  TOTAL: **{total}**"
    )
    if html_link:
        lines.append(f"- 详细 HTML 报告: [{html_link}]({html_link})")
    lines.append("")

    lines.append("## 环境")
    lines.append("")
    lines.append("| 字段 | 值 |")
    lines.append("|---|---|")
    for k in (
        "platform",
        "python",
        "torch",
        "ipex",
        "has_xpu",
        "xpu_device",
        "has_cuda",
        "cuda_device",
        "has_modelscope",
        "has_pyannote",
        "has_speakerlab",
    ):
        if k in env and env[k] is not None:
            lines.append(f"| `{k}` | `{env[k]}` |")
    lines.append("")

    lines.append("## 总览")
    lines.append("")
    lines.append("| 任务 | 名称 | PASS | FAIL/ERROR | SKIP | 用时 |")
    lines.append("|---|---|---|---|---|---|")
    grouped = _group_by_task(cases)
    ordered_tasks = list(TASK_LABELS) + [t for t in grouped if t not in TASK_LABELS]
    for task in ordered_tasks:
        if task not in grouped:
            continue
        tp, tf, ts, _, tt = _summarize(grouped[task])
        lines.append(f"| {task} | {_label(task)} | {tp} | {tf} | {ts} | {tt:.2f}s |")
    lines.append("")

    for task in ordered_tasks:
        if task not in grouped:
            continue
        lines.append(f"## {_label(task)}")
        lines.append("")
        lines.append("| 测试 | 结果 | 用时 | 关键指标 |")
        lines.append("|---|---|---|---|")
        for c in grouped[task]:
            metrics_text = (
                ", ".join(f"`{k}={v}`" for k, v in c.metrics.items()) if c.metrics else ""
            )
            lines.append(
                f"| `{c.display}` | **{_md_status(c.status)}** | {c.time_s:.2f}s | {metrics_text} |"
            )
        lines.append("")

    failures = [c for c in cases if c.status in ("FAIL", "ERROR", "SKIP")]
    if failures:
        lines.append("## 失败与跳过原因汇总")
        lines.append("")
        for c in failures:
            msg = (c.message or "").strip().splitlines()[0] if c.message else ""
            msg = re.sub(r"\s+", " ", msg)[:240]
            lines.append(f"- **{c.status}** `{c.display}` — {msg}")
        lines.append("")

    lines.append("## 附录：复现命令")
    lines.append("")
    lines.append("```powershell")
    lines.append('$env:PYTORCH_ENABLE_XPU_FALLBACK = "1"')
    lines.append("python scripts/run_full_test_matrix.py --device xpu --report_dir reports/")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_report(
    junit_path: Path,
    env_path: Path,
    out_md: Path,
    html_link: str | None = None,
) -> dict:
    cases, suite_attrs = parse_junit(junit_path)
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}
    md = render_markdown(cases, suite_attrs, env, html_link)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    p, f, s, total, _ = _summarize(cases)
    return {"pass": p, "fail": f, "skip": s, "total": total, "out": str(out_md)}
