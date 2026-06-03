# Copyright 2026 Xiaohongshu.
"""PyTorch device resolution for ASR (CUDA / Intel XPU / CPU).

Supports two Intel stacks (same idea as hello-agents ``00_quick_test.py``):

1. **Unified PyTorch XPU** (e.g. ``torch 2.11.0+xpu`` from Intel wheels): ``import torch``
   then ``torch.xpu.is_available()`` — **no** separate ``intel_extension_for_pytorch`` package.
2. **Legacy IPEX**: CPU PyTorch + ``intel_extension_for_pytorch``; XPU may only work **after**
   IPEX is imported — we try IPEX if the native check is false.

For ``use_gpu=True`` without an explicit device string: CUDA if available, else XPU, else CPU.
"""

from __future__ import annotations

import torch


def try_import_ipex() -> bool:
    """Import Intel Extension for PyTorch if present. Returns False on any failure."""
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        return True
    except Exception:
        return False


def xpu_runtime_available() -> bool:
    """True if ``torch.xpu.is_available()`` (after optional IPEX import for legacy stacks)."""
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and bool(xpu.is_available()):
        return True
    if try_import_ipex():
        xpu = getattr(torch, "xpu", None)
        return xpu is not None and bool(xpu.is_available())
    return False


def resolve_fire_red_asr_torch_device(*, device_str: str, use_gpu: bool) -> torch.device:
    """
    Resolve ``torch.device`` for ``FireRedAsr2``.

    - Non-empty ``device_str`` (e.g. ``\"xpu\"``, ``\"cuda:0\"``): use as-is; for
      ``xpu``, :func:`xpu_runtime_available` must be true.
    - Empty ``device_str`` and ``use_gpu`` False: CPU.
    - Empty ``device_str`` and ``use_gpu`` True: CUDA if available, else Intel XPU
      if :func:`xpu_runtime_available`, else CPU.
    """
    raw = (device_str or "").strip()
    if raw:
        dev = torch.device(raw)
        if dev.type == "xpu":
            if not xpu_runtime_available():
                raise RuntimeError(
                    "device=xpu was requested but torch.xpu is not available. "
                    "Install Intel PyTorch with XPU (e.g. torch 2.11+xpu wheels; see "
                    "scripts/install_intel_xpu_pytorch.ps1). Legacy setups need IPEX + a "
                    "matching PyTorch build. See README «Setup (Intel GPU, XPU / IPEX)»."
                )
        return dev
    if not use_gpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if xpu_runtime_available():
        return torch.device("xpu")
    return torch.device("cpu")


def resolve_compute_dtype(*, use_half: bool, device: torch.device) -> torch.dtype | None:
    """Pick a low-precision dtype for inference when ``use_half`` is True.

    - CUDA: ``float16`` (legacy ``model.half()`` behaviour).
    - Intel XPU: ``bfloat16`` (preferred on current Intel stacks).
    - CPU: ``bfloat16`` when PyTorch supports it (torch>=2.1); else ``None``.
    """
    if not use_half:
        return None
    if device.type == "cuda":
        return torch.float16
    if device.type == "xpu":
        return torch.bfloat16
    if device.type == "cpu":
        return torch.bfloat16
    return None
