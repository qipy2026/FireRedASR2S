#!/usr/bin/env python3
"""Experimental: dynamic INT8 quantization for FireRedASR-AED (CPU only).

Uses ``torch.ao.quantization.quantize_dynamic`` on ``nn.Linear`` modules.
This is a research / deployment helper — accuracy is not guaranteed.

After generating ``--out_path``, run inference (CPU) with the main CLI::

    .venv\\Scripts\\python.exe -m fireredasr2s.fireredasr2s_cli ^
        --asr_type aed --asr_use_gpu 0 --asr_use_half 0 ^
        --aed_dynamic_int8_pt path\\to\\aed_dynamic_int8_cpu.pt ^
        --wav_path your.wav --outdir out

See also: ``docs/WINDOWS_INFERENCE_SPEED.md``.

Usage (quantize step):
    .venv/Scripts/python.exe scripts/quantize_aed_int8.py \\
        --model_dir pretrained_models/FireRedASR2-AED \\
        --out_path aed_dynamic_int8_cpu.pt
"""

from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="aed_dynamic_int8_cpu.pt")
    args = parser.parse_args()

    if torch.cuda.is_available() or getattr(torch, "xpu", None) and torch.xpu.is_available():
        print("Warning: run this script on CPU-only PyTorch for predictable dynamic quant.")

    from fireredasr2s.fireredasr2.asr import load_fireredasr_aed_model

    mp = os.path.join(args.model_dir, "model.pth.tar")
    model = load_fireredasr_aed_model(mp)
    model.eval()
    model.cpu()

    qmodel = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save({"quantized": True, "state": qmodel.state_dict()}, args.out_path)
    print(f"[ok] wrote {args.out_path}")
    print(
        "Next: CPU inference with -m fireredasr2s.fireredasr2s_cli "
        "--asr_use_gpu 0 --asr_use_half 0 "
        f"--aed_dynamic_int8_pt {args.out_path}"
    )


if __name__ == "__main__":
    main()
