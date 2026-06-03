#!/usr/bin/env python3
"""Simulate microphone streaming by pushing a 16 kHz wav in small PCM chunks.

Requires real models (same as ``fireredasr2s_cli``). Example::

  .venv/Scripts/python.exe examples/streaming_simulate_from_wav.py \\
    --wav_path tests/fixtures/clean_zh_short.wav --chunk_ms 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("PYTORCH_ENABLE_XPU_FALLBACK", "1")

from fireredasr2s.fireredasr2 import FireRedAsr2Config  # noqa: E402
from fireredasr2s.fireredasr2system import FireRedAsr2System, FireRedAsr2SystemConfig  # noqa: E402
from fireredasr2s.fireredlid import FireRedLidConfig  # noqa: E402
from fireredasr2s.fireredpunc import FireRedPuncConfig  # noqa: E402
from fireredasr2s.fireredvad import FireRedVadConfig  # noqa: E402
from fireredasr2s.repo_dotenv import default_asr_device, load_repo_dotenv  # noqa: E402


def main() -> None:
    load_repo_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--wav_path", type=str, required=True)
    p.add_argument("--chunk_ms", type=int, default=200)
    p.add_argument("--models_root", type=str, default="pretrained_models")
    p.add_argument(
        "--device",
        type=str,
        default=default_asr_device(),
        help="ASR 等设备；默认 xpu；仅 FIRERED_ASR_DEVICE 可覆盖（不读 ASR_DEVICE）",
    )
    p.add_argument("--enable_lid", type=int, default=0)
    p.add_argument("--enable_punc", type=int, default=1)
    args = p.parse_args()

    root = Path(args.models_root)
    wav_path = Path(args.wav_path)
    pcm, sr = sf.read(str(wav_path), dtype="int16")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1).astype(np.int16)
    if int(sr) != 16000:
        print("Warning: expected 16 kHz mono wav; resampling via System stack on first push.", flush=True)

    asr_cfg = FireRedAsr2Config(use_gpu=args.device != "cpu", device=args.device, return_timestamp=False)
    cfg = FireRedAsr2SystemConfig(
        vad_model_dir=str(root / "FireRedVAD" / "VAD"),
        lid_model_dir=str(root / "FireRedLID"),
        asr_model_dir=str(root / "FireRedASR2-AED"),
        punc_model_dir=str(root / "FireRedPunc"),
        vad_config=FireRedVadConfig(use_gpu=False),
        lid_config=FireRedLidConfig(use_gpu=args.device != "cpu"),
        asr_config=asr_cfg,
        punc_config=FireRedPuncConfig(use_gpu=args.device != "cpu"),
        enable_vad=True,
        enable_lid=bool(args.enable_lid),
        enable_punc=bool(args.enable_punc),
        enable_diarization=False,
        stream_vad_use_gpu=False,
    )
    sys_m = FireRedAsr2System(cfg)
    session = sys_m.open_stream(uttid_prefix="sim")

    chunk_samples = max(int(16 * args.chunk_ms), 160)
    for i in range(0, len(pcm), chunk_samples):
        chunk = pcm[i : i + chunk_samples]
        evs = session.push_pcm_int16_mono(chunk, sample_rate=int(sr))
        for ev in evs:
            print(json.dumps(ev, ensure_ascii=False, default=str)[:2000], flush=True)

    for ev in session.finalize():
        print(json.dumps(ev, ensure_ascii=False, default=str)[:2000], flush=True)


if __name__ == "__main__":
    main()
