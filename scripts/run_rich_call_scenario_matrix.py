#!/usr/bin/env python3
"""富通话多场景矩阵测试：依次跑若干 scenario JSON，校验 meta/录音，并生成 Markdown 报告。

用于外呼/呼入等不同「真实案例」编导下的系统化回归（依赖 LLM + 全量 ASR，耗时较长）。

用法::

  .venv\\Scripts\\python.exe scripts\\run_rich_call_scenario_matrix.py --skip-prepare
  .venv\\Scripts\\python.exe scripts\\run_rich_call_scenario_matrix.py --skip-prepare --random-customer-seed 424242
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_VENV_PY = _REPO / ".venv" / "Scripts" / "python.exe"
if not _VENV_PY.is_file():
    _VENV_PY = _REPO / ".venv" / "bin" / "python"

_SCEN_DIR = _REPO / "examples" / "duplex_rich_call_scenario"

# （标签，scenario 文件名，案例说明）
MATRIX_CASES: list[tuple[str, str, str]] = [
    (
        "1",
        "scenario_matrix_01_outbound_smooth.json",
        "顺畅结案·外呼：套餐变更说明与收线（无抢话、无跑题拼接）。",
    ),
    (
        "2",
        "scenario_matrix_02_inbound_billing_smooth.json",
        "顺畅结案·呼入：话费/账单引导自助查询后收线。",
    ),
    (
        "3",
        "scenario_matrix_03_inbound_network_smooth.json",
        "顺畅结案·呼入：网络/信号首问指引后收线。",
    ),
    (
        "4",
        "scenario_matrix_04_inbound_balance_smooth.json",
        "顺畅结案·呼入：余额/流量查询引导后收线。",
    ),
    (
        "5",
        "scenario_matrix_05_inbound_billing_one_barge.json",
        "抢话一条·呼入账单：仅首轮说明时礼貌打断一次，后续正常 listen 收束。",
    ),
]


def _child_env() -> dict[str, str]:
    e = os.environ.copy()
    e["PYTHONIOENCODING"] = "utf-8"
    e["PYTHONUTF8"] = "1"
    e["PYTHONUNBUFFERED"] = "1"
    return e


def _venv_python_args(script: str) -> list[str]:
    return [str(_VENV_PY), "-X", "utf8", str(script)]


def _parse_meta_path(sim_stdout: str) -> Path:
    m = re.search(r"# 摘要已写:\s*(.+\.json)\s*$", sim_stdout, re.MULTILINE)
    if not m:
        raise RuntimeError("仿真 stdout 中未找到「摘要已写」行")
    return Path(m.group(1).strip())


_DEFAULT_MATRIX_VALIDATION: dict[str, object] = {
    "min_barge_in": 3,
    "min_segment_final": 3,
    "min_reactive_after_barge": 3,
    "require_barge_plan": True,
}


def _validate_meta(meta_path: Path, scenario_path: Path | None = None) -> list[str]:
    errs: list[str] = []
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    mv = dict(_DEFAULT_MATRIX_VALIDATION)
    if scenario_path is not None and scenario_path.is_file():
        try:
            scen = json.loads(scenario_path.read_text(encoding="utf-8"))
            u = scen.get("matrix_validation")
            if isinstance(u, dict):
                for k, v in u.items():
                    if k in mv:
                        mv[k] = v
        except (OSError, json.JSONDecodeError):
            pass
    if not data.get("reactive_barge_duplex"):
        errs.append("期望 reactive_barge_duplex=true")
    stats = data.get("stream_stats") or {}
    bi = int(stats.get("barge_in", 0))
    sf = int(stats.get("segment_final", 0))
    min_bi = int(mv.get("min_barge_in", 0) or 0)
    min_sf = int(mv.get("min_segment_final", 0) or 0)
    if bi < min_bi:
        errs.append(f"stream_stats.barge_in={bi} 期望 >={min_bi}")
    if sf < min_sf:
        errs.append(f"stream_stats.segment_final={sf} 期望 >={min_sf}")
    turns = data.get("assistant_turns") or []
    reactive_n = sum(1 for t in turns if isinstance(t, dict) and t.get("reactive_after_barge"))
    min_rx = int(mv.get("min_reactive_after_barge", 0) or 0)
    if reactive_n < min_rx:
        errs.append(f"reactive_after_barge 轮次={reactive_n} 期望 >={min_rx}")
    ca = data.get("call_audio") or {}
    for key in ("stereo_micL_ttsR_wav",):
        p = ca.get(key)
        if not p or not Path(str(p)).is_file():
            errs.append(f"call_audio.{key} 缺失: {p!r}")
        elif Path(str(p)).stat().st_size < 256:
            errs.append(f"call_audio.{key} 过小")
    plan = data.get("random_customer_plan")
    req_barge_plan = bool(mv.get("require_barge_plan", True))
    if req_barge_plan:
        if not isinstance(plan, dict) or not plan.get("barges"):
            errs.append("期望 meta.random_customer_plan.barges")
    return errs


def _preview_turns(turns: list[dict], *, limit: int = 8) -> list[str]:
    out: list[str] = []
    for i, t in enumerate(turns[:limit]):
        if not isinstance(t, dict):
            continue
        nm = str(t.get("name", ""))
        tx = (t.get("text") or "").replace("\n", " ").strip()
        if len(tx) > 120:
            tx = tx[:119] + "…"
        extra = ""
        if t.get("reactive_reply"):
            r = str(t.get("reactive_reply", "")).replace("\n", " ").strip()
            if len(r) > 80:
                r = r[:79] + "…"
            extra = f" | 抢话续接: {r}"
        out.append(f"{i + 1}. **{nm}**: {tx}{extra}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="富通话多场景矩阵测试与报告")
    ap.add_argument("--skip-prepare", action="store_true", help="跳过客户 WAV 准备")
    ap.add_argument("--device", type=str, default="", help="ASR 设备，如 cpu / xpu")
    ap.add_argument("--random-customer-seed", type=int, default=424242, help="各案例统一种子，便于复现")
    ap.add_argument(
        "--call-record-dir",
        type=str,
        default="output/call_recordings",
        help="录音与 meta 输出目录（相对仓库根）",
    )
    ap.add_argument(
        "--report",
        type=str,
        default="",
        help="Markdown 报告路径（默认 output/reports/rich_call_matrix_<UTC时间>.md）",
    )
    ap.add_argument(
        "--tts-engine",
        type=str,
        default="edge",
        choices=("edge", "pyttsx3", "auto"),
        help="传给仿真的助手 TTS；矩阵多案例时 Edge 偶发无音频可改 pyttsx3",
    )
    args = ap.parse_args()

    if not _VENV_PY.is_file():
        print(f"找不到 venv: {_VENV_PY}", file=sys.stderr)
        return 2

    report_path = Path(args.report) if str(args.report).strip() else None
    if report_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = _REPO / "output" / "reports" / f"rich_call_matrix_{ts}.md"
    if not report_path.is_absolute():
        report_path = _REPO / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    started_wall = datetime.now(timezone.utc)

    if not args.skip_prepare:
        r0 = subprocess.run(
            _venv_python_args(str(_REPO / "scripts" / "prepare_rich_call_scenario_wavs.py")),
            cwd=str(_REPO),
            env=_child_env(),
        )
        if r0.returncode != 0:
            return r0.returncode

    sim_py = _REPO / "examples" / "full_duplex_rich_call_llm_sim.py"
    seed = int(args.random_customer_seed)
    rec_rel = str(args.call_record_dir)

    rows: list[dict] = []
    md_lines: list[str] = [
        "# 富通话多场景矩阵测试报告",
        "",
        f"- **生成时间（UTC）**: {started_wall.isoformat()}",
        f"- **仓库**: `{_REPO}`",
        f"- **统一 random_customer_seed**: `{seed}`",
        f"- **说明**: 案例 1～4 为**顺畅结案**（无抢话轮、客户句为业务内短答）；案例 5 仅 **1 次**礼貌抢话（无天气/球赛跑题拼接）。验证 **LLM 编导 + 反应式 duplex + ASR + 上下文压缩** 稳定性。",
        "",
        "## 案例定义",
        "",
    ]
    for tag, fn, desc in MATRIX_CASES:
        md_lines.append(f"- **案例 {tag}**: `{fn}` — {desc}")
    md_lines.extend(["", "## 分项结果", ""])

    all_ok = True
    for tag, fn, desc in MATRIX_CASES:
        scen = _SCEN_DIR / fn
        if not scen.is_file():
            print(f"# 跳过：找不到场景 {scen}", flush=True)
            rows.append({"tag": tag, "ok": False, "error": f"missing {scen}"})
            md_lines.append(f"### 案例 {tag} — **失败**（场景文件缺失）\n")
            all_ok = False
            continue

        cmd = _venv_python_args(str(sim_py)) + [
            "--call-audio",
            "--call-record-dir",
            rec_rel,
            "--random-customer-seed",
            str(seed),
            "--tts-engine",
            str(args.tts_engine),
            "--scenario",
            str(scen),
        ]
        if str(args.device or "").strip():
            cmd.extend(["--device", str(args.device).strip()])

        print("+ " + " ".join(cmd), flush=True)
        t0 = datetime.now(timezone.utc)
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_child_env(),
        )
        t1 = datetime.now(timezone.utc)
        dur_s = (t1 - t0).total_seconds()
        out = proc.stdout or ""

        row: dict = {
            "tag": tag,
            "scenario_file": fn,
            "exit_code": proc.returncode,
            "duration_s": round(dur_s, 1),
        }
        md_lines.append(f"### 案例 {tag}: {desc}\n")
        md_lines.append(f"- **场景文件**: `{fn}`")
        md_lines.append(f"- **子进程退出码**: `{proc.returncode}`")
        md_lines.append(f"- **耗时（约）**: {dur_s:.1f}s")

        if proc.returncode != 0:
            all_ok = False
            row["ok"] = False
            row["error"] = "sim non-zero exit"
            md_lines.append("- **结果**: **失败**（仿真退出非零）")
            md_lines.append("\n<details><summary>stdout 尾部</summary>\n\n```text")
            md_lines.append("\n".join(out.splitlines()[-40:]))
            md_lines.append("```\n</details>\n")
            rows.append(row)
            continue

        try:
            meta_path = _parse_meta_path(out)
        except RuntimeError as e:
            all_ok = False
            row["ok"] = False
            row["error"] = str(e)
            md_lines.append(f"- **结果**: **失败**（{e}）")
            rows.append(row)
            continue

        errs = _validate_meta(meta_path, scen)
        if errs:
            all_ok = False
            row["ok"] = False
            row["validation_errors"] = errs
            md_lines.append("- **结果**: **失败**（meta 校验）")
            for e in errs:
                md_lines.append(f"  - {e}")
        else:
            row["ok"] = True
            row["meta_path"] = str(meta_path.resolve())
            md_lines.append("- **结果**: **通过**")
            md_lines.append(f"- **meta**: `{meta_path.resolve()}`")
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            st = data.get("stream_stats") or {}
            md_lines.append(
                f"- **stream_stats**: barge_in={st.get('barge_in')} "
                f"barge_in_suppressed={st.get('barge_in_suppressed')} "
                f"segment_final={st.get('segment_final')}"
            )
            slim = data.get("llm_context_compress")
            if slim:
                md_lines.append(f"- **llm_context_compress**: `{slim}`")
            turns = data.get("assistant_turns") or []
            md_lines.append(f"- **助手轮次数**: {len(turns)}")
            md_lines.append("\n**助手话术预览**（前若干轮）:\n")
            for line in _preview_turns(turns if isinstance(turns, list) else []):
                md_lines.append(f"- {line}")

        md_lines.append("")
        rows.append(row)

    ended_wall = datetime.now(timezone.utc)
    md_lines.extend(
        [
            "## 汇总表",
            "",
            "| 案例 | 场景文件 | 退出码 | 耗时(s) | 校验 |",
            "|------|----------|--------|---------|------|",
        ]
    )
    for r in rows:
        ok = r.get("ok")
        chk = "通过" if ok else "失败"
        md_lines.append(
            f"| {r.get('tag')} | `{r.get('scenario_file', '')}` | "
            f"{r.get('exit_code', '')} | {r.get('duration_s', '')} | {chk} |"
        )
    md_lines.extend(
        [
            "",
            "## 结论与使用说明",
            "",
            "- **案例 1～4** 无 `mode=barge` 轮次，矩阵校验要求 `barge_in≥0`；**案例 5** 仅 `biz_long` 为抢话轮，校验要求至少一次 `barge_in` 与 `reactive_after_barge`。",
            "- 客户句来自 `customer_pool` 的业务向短答；案例 5 抢话音频为单条礼貌追问（`barge_polite_billing`），不再混入天气/足球跑题。",
            f"- **总墙钟（UTC）**: {started_wall.isoformat()} → {ended_wall.isoformat()}",
            "",
        ]
    )

    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"# 矩阵报告已写: {report_path.resolve()}", flush=True)
    return 0 if all_ok else 3


if __name__ == "__main__":
    try:
        from fireredasr2s.win_console_utf8 import ensure_stdio_utf8

        ensure_stdio_utf8()
    except Exception:
        pass
    raise SystemExit(main())
