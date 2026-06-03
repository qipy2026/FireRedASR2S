#!/usr/bin/env python3
"""Full-duplex demo: TTS playback + microphone → ASR with optional software NLMS AEC.

Uses one duplex ``sounddevice.Stream`` so playback samples align with the AEC reference.
**外放仍可能残留回声**；生产环境优先 **耳机** 或 **系统/WebRTC AEC**。NLMS 仅为轻量缓解。

Dependencies::

  pip install sounddevice pyttsx3
  # 可选：edge-tts（需 ffmpeg 将 mp3 转为 wav） pip install edge-tts

Example::

  .venv\\Scripts\\python.exe examples\\full_duplex_mic_tts_demo.py --device xpu \\
    --tts-text "您好，请问有什么可以帮您？" --listen-after-tts 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import stdio_utf8_windows  # noqa: E402

stdio_utf8_windows.apply_stdio_utf8()

os.environ.setdefault("PYTORCH_ENABLE_XPU_FALLBACK", "1")

import numpy as np  # noqa: E402

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
from fireredasr2s.duplex import NlmsMonoAec  # noqa: E402

from full_duplex_voice_llm_tts import (  # noqa: E402
    EDGE_VOICE_ASSISTANT_FEMALE_DEFAULT,
    synthesize_tts_16k_int16,
)


def _float_to_i16_mono(x: np.ndarray) -> np.ndarray:
    return (np.clip(x.astype(np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)


def _i16_to_float_mono(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


@dataclass
class _DuplexState:
    tts_i16: np.ndarray
    pos: int = 0
    ended_playback: bool = False
    aec: NlmsMonoAec | None = None
    session: object = None  # FireRedFullDuplexStreamSession
    print_events: bool = True
    t0_perf: float = field(default_factory=time.perf_counter)


def main() -> None:
    load_repo_dotenv()
    if sd is None:
        print("Install sounddevice: pip install sounddevice", file=sys.stderr)
        sys.exit(1)

    p = argparse.ArgumentParser(description="Full-duplex TTS + mic ASR demo (16 kHz)")
    p.add_argument("--models_root", type=str, default="pretrained_models")
    p.add_argument(
        "--device",
        type=str,
        default=default_asr_device(),
        help="ASR 等设备；默认 xpu；仅 FIRERED_ASR_DEVICE 可覆盖（不读 ASR_DEVICE）",
    )
    p.add_argument("--tts-text", type=str, default="您好，请问有什么可以帮您？")
    p.add_argument("--tts-engine", type=str, choices=("pyttsx3", "edge"), default="pyttsx3")
    p.add_argument("--edge-voice", type=str, default=EDGE_VOICE_ASSISTANT_FEMALE_DEFAULT)
    p.add_argument("--listen-after-tts", type=float, default=10.0, help="keep streaming after TTS ends")
    p.add_argument("--block-ms", type=int, default=20, help="duplex block size (20 ms @ 16 kHz = 320 samples)")
    p.add_argument("--aec", type=str, choices=("none", "nlms"), default="nlms")
    p.add_argument("--filter-len", type=int, default=2048)
    p.add_argument("--aec-mu", type=float, default=0.25)
    p.add_argument("--aec-delay-samples", type=int, default=0, help="extra ref delay (tune for device latency)")
    p.add_argument("--enable-punc", type=int, default=1)
    p.add_argument("--verbose-vad", action="store_true")
    args = p.parse_args()

    root = Path(args.models_root)
    sr = 16000
    block = max(int(sr * args.block_ms / 1000.0), 80)

    print("# synthesizing TTS...", flush=True)
    tts_i16 = synthesize_tts_16k_int16(
        args.tts_text,
        args.tts_engine,
        args.edge_voice,
        pyttsx3_gender="female",
    )
    print(f"# TTS samples={len(tts_i16)} dur_s={len(tts_i16) / sr:.2f}", flush=True)

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
        enable_lid=False,
        enable_punc=bool(args.enable_punc),
        enable_diarization=False,
        stream_vad_use_gpu=False,
    )
    sys_m = FireRedAsr2System(cfg)
    session = sys_m.open_full_duplex_stream(
        uttid_prefix="duplex_demo",
        verbose_vad=args.verbose_vad,
    )

    aec: NlmsMonoAec | None = None
    if args.aec == "nlms":
        aec = NlmsMonoAec(
            filter_len=args.filter_len,
            mu=args.aec_mu,
            ref_delay_samples=args.aec_delay_samples,
        )

    state = _DuplexState(tts_i16=tts_i16, aec=aec, session=session)

    anchor_ms = int(time.time() * 1000)
    if len(tts_i16) > 0:
        session.begin_local_playback(playback_id="tts-1", anchor_wallclock_ms=anchor_ms)
    else:
        print("# empty TTS text; no begin_local_playback", flush=True)

    def callback(indata, outdata, frames, _time, status) -> None:
        if status:
            print(f"# audio status: {status}", flush=True)
        n = int(frames)
        ref = np.zeros(n, dtype=np.float32)
        if state.pos < len(state.tts_i16):
            take = min(n, len(state.tts_i16) - state.pos)
            ref[:take] = _i16_to_float_mono(state.tts_i16[state.pos : state.pos + take])
            state.pos += take
        outdata[:, 0] = ref

        mic = indata[:, 0].astype(np.float32).copy()
        if state.aec is not None:
            mic_u = state.aec.process_block(mic, ref)
        else:
            mic_u = mic
        pcm16 = _float_to_i16_mono(mic_u)
        evs = state.session.push_microphone_pcm(pcm16, sample_rate=sr)
        if state.print_events and evs:
            for ev in evs:
                print(json.dumps(ev, ensure_ascii=False, default=str)[:4000], flush=True)

        if (
            not state.ended_playback
            and len(state.tts_i16) > 0
            and state.pos >= len(state.tts_i16)
        ):
            state.session.end_local_playback()
            state.ended_playback = True
            print("# TTS playback finished; end_local_playback()", flush=True)

    total_s = len(tts_i16) / sr + float(args.listen_after_tts)
    print(f"# duplex {sr} Hz block={block} total_stream_s≈{total_s:.1f} aec={args.aec}", flush=True)

    try:
        with sd.Stream(
            samplerate=sr,
            blocksize=block,
            dtype="float32",
            channels=1,
            callback=callback,
        ):
            sd.sleep(int(max(total_s * 1000.0, block / sr * 1000.0)))
    finally:
        for ev in state.session.finalize():
            print(json.dumps(ev, ensure_ascii=False, default=str)[:4000], flush=True)


if __name__ == "__main__":
    main()
