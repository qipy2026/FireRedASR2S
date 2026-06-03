# Copyright 2026 Xiaohongshu.
"""Intel XPU tests for ``fireredasr2s.torch_device`` (unified torch+xpu or legacy IPEX).

Runs everywhere: resolver invariants and explicit-xpu error when no runtime.
Runs only when :func:`xpu_runtime_available`: tensor op + auto device (no CUDA).
"""

from __future__ import annotations

import os
import sys
import unittest

# Repo root (parent of ``tests/``)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from fireredasr2s.torch_device import (  # noqa: E402
    resolve_fire_red_asr_torch_device,
    xpu_runtime_available,
)


class TestTorchDeviceResolver(unittest.TestCase):
    def test_use_gpu_false_returns_cpu(self) -> None:
        d = resolve_fire_red_asr_torch_device(device_str="", use_gpu=False)
        self.assertEqual(d.type, "cpu")

    def test_empty_device_use_gpu_true_cuda_wins(self) -> None:
        d = resolve_fire_red_asr_torch_device(device_str="", use_gpu=True)
        if torch.cuda.is_available():
            self.assertEqual(d.type, "cuda")
        elif xpu_runtime_available():
            self.assertEqual(d.type, "xpu")
        else:
            self.assertEqual(d.type, "cpu")

    def test_explicit_xpu_raises_when_runtime_missing(self) -> None:
        if xpu_runtime_available():
            self.skipTest("XPU runtime present; negative case not applicable")
        with self.assertRaises(RuntimeError) as ctx:
            resolve_fire_red_asr_torch_device(device_str="xpu", use_gpu=True)
        self.assertIn("xpu", str(ctx.exception).lower())


@unittest.skipUnless(
    xpu_runtime_available(),
    "torch.xpu not available (install Intel torch +xpu; skip hardware checks)",
)
class TestXpuRuntime(unittest.TestCase):
    def test_torch_xpu_available(self) -> None:
        self.assertTrue(torch.xpu.is_available())

    def test_xpu_tensor_add(self) -> None:
        t = torch.ones(2, 3, device="xpu")
        u = t + 1.0
        self.assertEqual(u.device.type, "xpu")
        self.assertTrue(torch.allclose(u.cpu(), torch.full((2, 3), 2.0)))

    def test_resolve_explicit_xpu(self) -> None:
        d = resolve_fire_red_asr_torch_device(device_str="xpu", use_gpu=False)
        self.assertEqual(d.type, "xpu")

    def test_resolve_auto_use_gpu_prefers_xpu_without_cuda(self) -> None:
        if torch.cuda.is_available():
            self.skipTest("Auto order is CUDA > XPU; CUDA is present")
        d = resolve_fire_red_asr_torch_device(device_str="", use_gpu=True)
        self.assertEqual(d.type, "xpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
