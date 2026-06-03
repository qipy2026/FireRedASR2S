#!/usr/bin/env python3
"""Capture microphone PCM and stream to FireRedASR2S (16 kHz int16 chunks).

Requires optional dependency::

  pip install sounddevice

Uses the same model layout as other examples (``pretrained_models``).

Example::

  .venv\\Scripts\\python.exe examples\\mic_stream_to_asr.py --device xpu --seconds 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("PYTORCH_ENABLE_XPU_FALLBACK", "1")

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore[assignment]

from fireredasr2s.fireredasr2 import FireRedAsr2Config  # noqa: E402
from fireredasr2s.fireredasr2system import FireRedAsr2System, FireRedAsr2SystemConfig  # noqa: E402
from fireredasr2s.fireredlid import FireRedLidConfig  # noqa: E402
from fireredasr2s.fireredpunc import FireRedPuncConfig  # noqa: E402
from fireredasr2s.fireredvad import FireRedVadConfig  # noqa: E402
from fireredasr2s.repo_dotenv import default_asr_device, load_repo_dotenv  # noqa: E402


def main() -> None:
    load_repo_dotenv()
    if sd is None:
        print("Install sounddevice: pip install sounddevice", file=sys.stderr)
        sys.exit(1)

    p = argparse.ArgumentParser()
    p.add_argument("--models_root", type=str, default="pretrained_models")
    p.add_argument(
        "--device",
        type=str,
        default=default_asr_device(),
        help="ASR 等设备；默认 xpu；仅 FIRERED_ASR_DEVICE 可覆盖（不读 ASR_DEVICE）",
    )
    p.add_argument("--seconds", type=float, default=10.0, help="capture duration")
    p.add_argument("--chunk_ms", type=int, default=200)
    p.add_argument("--input_sr", type=int, default=0, help="0 = use device default sample rate")
    p.add_argument("--enable_lid", type=int, default=0)
    p.add_argument("--enable_punc", type=int, default=1)
    p.add_argument("--telemetry", action="store_true", help="log stream session latency lines")
    args = p.parse_args()

    root = Path(args.models_root)
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
    session = sys_m.open_stream(uttid_prefix="mic", telemetry=args.telemetry)

    dev_sr = int(args.input_sr) if args.input_sr else int(sd.query_devices(kind="input")["default_samplerate"])
    chunk_samples = max(int(dev_sr * args.chunk_ms / 1000.0), 1)
    total_chunks = max(1, int(np.ceil(args.seconds * 1000.0 / args.chunk_ms)))

    print(f"# capture default_input sr={dev_sr} chunk_samples={chunk_samples} chunks={total_chunks}", flush=True)

    for _ in range(total_chunks):
        block, _overflow = sd.rec(
            chunk_samples,
            samplerate=dev_sr,
            channels=1,
            dtype="float32",
            blocking=True,
        )
        if _overflow:
            print("# warning: input overflow", flush=True)
        mono = block.reshape(-1)
        pcm = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
        evs = session.push_pcm_int16_mono(pcm, sample_rate=dev_sr)
        for ev in evs:
            print(json.dumps(ev, ensure_ascii=False, default=str)[:4000], flush=True)

    for ev in session.finalize():
        print(json.dumps(ev, ensure_ascii=False, default=str)[:4000], flush=True)


if __name__ == "__main__":
    main()
