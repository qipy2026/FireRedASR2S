"""Shared pytest fixtures and hooks for FireRedASR2S tests.

Highlights:

- Sets ``PYTORCH_ENABLE_XPU_FALLBACK=1`` very early so XPU-only kernels can fall
  back to CPU during tests (e.g. ``torchaudio.functional.forced_align``).
- Auto-skips ``xpu`` / ``cuda`` / ``modelscope`` / ``pyannote`` marked tests when
  the underlying runtime/package is missing, so a single environment can run
  the full matrix without false failures.
- Provides ``record_metric(item, name, value)`` plumbing: tests call
  ``record_metric(request, "wer", 0.05)`` and the value is captured into the
  pytest junit XML as ``<property name="metric_wer" value="0.05"/>``. The T9
  report writer parses these properties.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Set XPU fallback BEFORE torch is imported by anything else.
os.environ.setdefault("PYTORCH_ENABLE_XPU_FALLBACK", "1")

import pytest

# Make repo root importable as ``fireredasr2s`` & ``tests.utils``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Runtime probes
# ---------------------------------------------------------------------------


def _xpu_available() -> bool:
    try:
        from fireredasr2s.torch_device import xpu_runtime_available

        return bool(xpu_runtime_available())
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


_HAS_XPU = _xpu_available()
_HAS_CUDA = _cuda_available()
_HAS_MODELSCOPE = _module_available("modelscope")
_HAS_PYANNOTE = _module_available("pyannote.audio")
_HAS_SPEAKERLAB = _module_available("speakerlab")
_HAS_JIWER = _module_available("jiwer")
_HAS_PYANNOTE_METRICS = _module_available("pyannote.metrics")


# ---------------------------------------------------------------------------
# Auto-skip by markers
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_specs = [
        ("xpu", _HAS_XPU, "Intel XPU not available"),
        ("cuda", _HAS_CUDA, "CUDA not available"),
        ("modelscope", _HAS_MODELSCOPE, "modelscope not installed"),
        ("pyannote", _HAS_PYANNOTE, "pyannote.audio not installed"),
        ("speakerlab", _HAS_SPEAKERLAB, "speakerlab not installed"),
    ]
    for item in items:
        for marker, available, reason in skip_specs:
            if marker in item.keywords and not available:
                item.add_marker(pytest.mark.skip(reason=reason))


# ---------------------------------------------------------------------------
# Generic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return _REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def has_xpu() -> bool:
    return _HAS_XPU


@pytest.fixture(scope="session")
def has_cuda() -> bool:
    return _HAS_CUDA


@pytest.fixture(scope="session")
def runtime_info() -> dict[str, Any]:
    """Snapshot of the runtime; saved by the T9 report writer to ``reports/env.json``."""
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": sys.platform,
        "has_xpu": _HAS_XPU,
        "has_cuda": _HAS_CUDA,
        "has_modelscope": _HAS_MODELSCOPE,
        "has_pyannote": _HAS_PYANNOTE,
        "has_speakerlab": _HAS_SPEAKERLAB,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if _HAS_XPU and hasattr(torch, "xpu"):
            try:
                info["xpu_device"] = torch.xpu.get_device_name(0)
            except Exception:
                info["xpu_device"] = "<unknown>"
        if _HAS_CUDA:
            try:
                info["cuda_device"] = torch.cuda.get_device_name(0)
            except Exception:
                info["cuda_device"] = "<unknown>"
    except Exception:
        info["torch"] = "<missing>"
    try:
        import intel_extension_for_pytorch as ipex

        info["ipex"] = ipex.__version__
    except Exception:
        info["ipex"] = None
    return info


# ---------------------------------------------------------------------------
# Metric recording for the T9 report
# ---------------------------------------------------------------------------


def record_metric(request: pytest.FixtureRequest, name: str, value: Any) -> None:
    """Attach a numeric metric to the current test; surfaces in junit XML."""
    rp = getattr(request.node, "user_properties", None)
    payload = ("metric_" + name, value)
    if rp is None:
        request.node.user_properties = [payload]
    else:
        rp.append(payload)


@pytest.fixture
def metric(request: pytest.FixtureRequest):
    def _record(name: str, value: Any) -> None:
        record_metric(request, name, value)

    return _record


# ---------------------------------------------------------------------------
# Optional ASR system fixture (real model, only used by `slow` xpu tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def asr_system_xpu(has_xpu):
    """Build a minimal FireRedAsr2System on XPU if model dirs are present.

    Skips at session scope when XPU or model assets are unavailable. Path is
    overridable via ``FIREREDASR2S_MODELS_DIR`` env var.
    """
    if not has_xpu:
        pytest.skip("Intel XPU not available")

    models_root = Path(os.environ.get("FIREREDASR2S_MODELS_DIR", "pretrained_models"))
    expected = {
        "vad_model_dir": models_root / "FireRedVAD" / "VAD",
        "lid_model_dir": models_root / "FireRedLID",
        "asr_model_dir": models_root / "FireRedASR2-AED",
        "punc_model_dir": models_root / "FireRedPunc",
    }
    missing = [str(p) for p in expected.values() if not p.exists()]
    if missing:
        pytest.skip(f"Missing model dirs: {missing}")

    from fireredasr2s.fireredasr2 import FireRedAsr2Config
    from fireredasr2s.fireredasr2system import FireRedAsr2System, FireRedAsr2SystemConfig

    asr_cfg = FireRedAsr2Config(use_gpu=True, device="xpu", return_timestamp=False)
    cfg = FireRedAsr2SystemConfig(
        vad_model_dir=str(expected["vad_model_dir"]),
        lid_model_dir=str(expected["lid_model_dir"]),
        asr_model_dir=str(expected["asr_model_dir"]),
        punc_model_dir=str(expected["punc_model_dir"]),
        asr_config=asr_cfg,
        enable_lid=False,
        enable_punc=True,
        enable_diarization=False,
    )
    return FireRedAsr2System(cfg)
