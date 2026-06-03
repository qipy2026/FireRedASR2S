#!/usr/bin/env python3
"""Run the FireRedASR2S iteration test matrix and emit a Markdown + HTML report.

Usage (PowerShell, Intel XPU):

    $env:PYTORCH_ENABLE_XPU_FALLBACK = "1"
    .venv/Scripts/python.exe scripts/run_full_test_matrix.py \
        --device xpu --report_dir reports/

Steps:
  1. Snapshot env (torch/IPEX/XPU/CUDA flags) → ``reports/env.json``
  2. Generate synthetic audio fixtures if missing.
  3. Run ``pytest tests/`` with junit XML + pytest-html.
  4. Render Markdown summary via ``scripts._report_writer.write_report``.

Optional: pass ``--e2e_record_dir DIR`` to set ``FIREREDASR2S_E2E_RECORD_DIR`` so
``tests/test_e2e_by_feature.py`` writes per-case JSON/Markdown (same layout as
``output/long_multi_speaker``). See ``docs/FEATURE_E2E_TESTS.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Make ``scripts._report_writer`` importable when invoked as a script.
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_THIS.parent))

from scripts._report_writer import write_report  # noqa: E402


def _collect_env(device_pref: str) -> dict:
    info: dict = {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "device_pref": device_pref,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["has_cuda"] = bool(torch.cuda.is_available())
        if info["has_cuda"]:
            try:
                info["cuda_device"] = torch.cuda.get_device_name(0)
            except Exception:
                info["cuda_device"] = "<unknown>"
    except Exception:
        info["torch"] = "<missing>"
        info["has_cuda"] = False
    try:
        from fireredasr2s.torch_device import xpu_runtime_available

        info["has_xpu"] = bool(xpu_runtime_available())
        if info["has_xpu"]:
            try:
                import torch

                info["xpu_device"] = torch.xpu.get_device_name(0)
            except Exception:
                info["xpu_device"] = "<unknown>"
    except Exception:
        info["has_xpu"] = False
    try:
        import intel_extension_for_pytorch as ipex

        info["ipex"] = ipex.__version__
    except Exception:
        info["ipex"] = None
    for mod in ("modelscope", "pyannote.audio", "speakerlab", "deepfilternet", "noisereduce"):
        info[f"has_{mod.replace('.', '_')}"] = _module_available(mod)
    return info


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _ensure_fixtures() -> None:
    fix = _REPO / "tests" / "fixtures"
    if fix.exists() and any(fix.glob("*.wav")):
        return
    print("[matrix] generating synthetic fixtures...", flush=True)
    subprocess.check_call([sys.executable, str(_REPO / "scripts" / "generate_test_fixtures.py")])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="xpu", choices=["xpu", "cuda", "cpu"])
    parser.add_argument("--report_dir", default="reports")
    parser.add_argument(
        "--include",
        default="",
        help="comma-list of extra markers to include (e.g. slow,modelscope,pyannote)",
    )
    parser.add_argument(
        "--tests",
        default="tests",
        help="tests path passed to pytest",
    )
    parser.add_argument(
        "--e2e_record_dir",
        default="",
        help=(
            "If set, sets FIREREDASR2S_E2E_RECORD_DIR so E2E tests write "
            "asr_system_result.json / asr_transcribe_results.json / E2E_TEST_REPORT.md "
            "(same layout as output/long_multi_speaker) under each pytest case subfolder."
        ),
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_XPU_FALLBACK", "1")
    if (args.e2e_record_dir or "").strip():
        rec = (_REPO / args.e2e_record_dir).resolve()
        rec.mkdir(parents=True, exist_ok=True)
        os.environ["FIREREDASR2S_E2E_RECORD_DIR"] = str(rec)
        print(f"[matrix] E2E record dir -> {rec}", flush=True)

    report_dir = (_REPO / args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    env_path = report_dir / "env.json"
    junit_path = report_dir / "junit.xml"
    html_path = report_dir / "test_report.html"
    md_path = report_dir / "test_report.md"

    env = _collect_env(args.device)
    env_path.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[matrix] env -> {env_path}")
    print(json.dumps(env, ensure_ascii=False, indent=2))

    _ensure_fixtures()

    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        args.tests,
        "-ra",
        "--maxfail=0",
        f"--junitxml={junit_path}",
        f"--html={html_path}",
        "--self-contained-html",
    ]
    print("[matrix] running:", " ".join(pytest_cmd), flush=True)
    rc = subprocess.call(pytest_cmd, cwd=str(_REPO))
    print(f"[matrix] pytest exit code = {rc}")

    summary = write_report(
        junit_path=junit_path,
        env_path=env_path,
        out_md=md_path,
        html_link=html_path.name,
    )
    print(f"[matrix] report -> {md_path} ({summary})")
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())
