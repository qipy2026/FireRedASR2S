"""Sanity checks for AED dynamic INT8 loading rules (no full transcribe)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

_REPO = Path(__file__).resolve().parent.parent
_AED_DIR = _REPO / "pretrained_models" / "FireRedASR2-AED"
_AED_CKPT = _AED_DIR / "model.pth.tar"


def test_aed_int8_rejects_use_half():
    if not _AED_CKPT.is_file():
        pytest.skip("pretrained FireRedASR2-AED not present")
    cfg = FireRedAsr2Config(
        use_gpu=False,
        use_half=True,
        return_timestamp=False,
        aed_dynamic_int8_pt="any.pt",
    )
    with pytest.raises(ValueError, match="use_half"):
        FireRedAsr2.from_pretrained("aed", str(_AED_DIR), cfg)


@pytest.mark.cuda
def test_aed_int8_rejects_non_cpu_when_cuda_available():
    if not _AED_CKPT.is_file():
        pytest.skip("pretrained FireRedASR2-AED not present")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = FireRedAsr2Config(
        use_gpu=True,
        device="cuda",
        use_half=False,
        return_timestamp=False,
        aed_dynamic_int8_pt="any.pt",
    )
    with pytest.raises(ValueError, match="CPU"):
        FireRedAsr2.from_pretrained("aed", str(_AED_DIR), cfg)
