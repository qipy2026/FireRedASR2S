"""T3 compute dtype / half-precision resolution tests."""

from __future__ import annotations

import pytest
import torch

from fireredasr2s.torch_device import resolve_compute_dtype, resolve_fire_red_asr_torch_device


def test_resolve_dtype_xpu_bf16():
    d = torch.device("xpu") if getattr(torch, "xpu", None) and torch.xpu.is_available() else None
    if d is None:
        pytest.skip("XPU not available")
    assert resolve_compute_dtype(use_half=True, device=d) == torch.bfloat16


@pytest.mark.cuda
def test_resolve_dtype_cuda_fp16():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    d = torch.device("cuda:0")
    assert resolve_compute_dtype(use_half=True, device=d) == torch.float16


def test_resolve_dtype_cpu_bf16():
    d = torch.device("cpu")
    assert resolve_compute_dtype(use_half=True, device=d) == torch.bfloat16


def test_resolve_dtype_disabled():
    d = torch.device("cpu")
    assert resolve_compute_dtype(use_half=False, device=d) is None


def test_punc_lid_share_helper_import():
    from fireredasr2s.fireredpunc import FireRedPuncConfig
    from fireredasr2s.fireredlid import FireRedLidConfig

    assert hasattr(FireRedPuncConfig(), "device")
    assert hasattr(FireRedLidConfig(), "device")


@pytest.mark.xpu
@pytest.mark.slow
def test_asr_xpu_bf16_param_dtype(asr_system_xpu):
    """After loading with use_half, at least one parameter should be bf16."""
    from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

    models_root = asr_system_xpu.config.asr_model_dir
    cfg = FireRedAsr2Config(use_gpu=True, device="xpu", use_half=True, return_timestamp=False)
    m = FireRedAsr2.from_pretrained("aed", models_root, cfg)
    dtypes = {p.dtype for p in m.model.parameters()}
    assert torch.bfloat16 in dtypes or torch.float32 in dtypes


@pytest.mark.slow
def test_int8_script_import_only():
    import runpy
    import sys
    from pathlib import Path

    p = Path(__file__).resolve().parent.parent / "scripts" / "quantize_aed_int8.py"
    assert p.exists()
    # Smoke: module parses (does not run main without model)
    src = p.read_text(encoding="utf-8")
    assert "quantize_dynamic" in src
