#!/usr/bin/env bash
# Example: FireRedASR2-LLM with vLLM backend (CUDA only).
# This repo's Python `VllmLlmRuntime.transcribe` is still a placeholder; use this
# script as a starting point for a custom vLLM deployment or upstream recipes.
set -euo pipefail

: "${WAV_PATH:?Set WAV_PATH to a 16 kHz mono wav}"
: "${OUTDIR:=output_vllm}"

echo "Stub: wire your vLLM server / CLI here. WAV_PATH=${WAV_PATH} OUTDIR=${OUTDIR}"
echo "For in-tree PyTorch LLM decoding on XPU, use:"
echo "  python -m fireredasr2s.fireredasr2s_cli --asr_type llm --asr_runtime torch --asr_device xpu --wav_path \"${WAV_PATH}\" --outdir \"${OUTDIR}\""
