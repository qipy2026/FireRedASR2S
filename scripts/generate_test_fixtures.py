#!/usr/bin/env python3
"""Generate synthetic audio fixtures for the iteration plan tests.

Outputs (16kHz int16 mono PCM) under ``tests/fixtures/``:

- ``clean_zh_short.wav``      : 3s sine-tone "speech proxy" (no real speech)
- ``mixed_zh_en_short.wav``   : 3s dual-tone proxy
- ``noisy_short.wav``         : ``clean_zh_short.wav`` + AWGN at SNR 5dB
- ``dialog_2spk_30s.wav``     : 30s alternating two-tone (proxy for two speakers)
- ``dialog_2spk_30s.rttm``    : matching RTTM with ground-truth turns
- ``enroll_spkA_{1..3}.wav``  : 2s tone-A samples (proxy for speaker A enroll)
- ``enroll_spkB_{1..3}.wav``  : 2s tone-B samples (proxy for speaker B enroll)
- ``e2e_vad_speech_proxy.wav``: ~5s band-limited noise bursts + silence gaps
  (non-stationary, speech-like energy envelope) so **FireRedVAD** yields at least
  one segment and E2E ``asr_transcribe_results.json`` is non-empty.

These are synthetic stand-ins so unit tests run without external assets. Real
ASR/diarization smoke tests are gated by ``slow``/``xpu`` markers and require
``FIREREDASR2S_MODELS_DIR`` plus real audio elsewhere.

Usage:
    .venv/Scripts/python.exe scripts/generate_test_fixtures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
INT16_MAX = 32767
ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
FIX.mkdir(parents=True, exist_ok=True)


def _tone(freq: float, dur_s: float, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    sig = amp * np.sin(2 * math.pi * freq * t)
    return sig.astype(np.float32)


def _save(path: Path, sig: np.ndarray, sr: int = SR) -> None:
    sig = np.clip(sig, -1.0, 1.0)
    pcm = (sig * INT16_MAX).astype(np.int16)
    sf.write(str(path), pcm, sr, subtype="PCM_16")


def _e2e_vad_speech_proxy(rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Energy-modulated noise bursts; steady sine is often rejected by VAD."""
    parts: list[np.ndarray] = []
    for _ in range(6):
        n_burst = int(0.5 * sr)
        n_gap = int(0.14 * sr)
        burst = rng.standard_normal(n_burst).astype(np.float32)
        burst *= 0.22
        win = np.hamming(n_burst).astype(np.float32)
        burst *= win
        # light band emphasis ~300–3kHz (speech-ish) via two-pole emphasis
        alpha = 0.97
        for i in range(1, burst.shape[0]):
            burst[i] = burst[i] - alpha * burst[i - 1]
        burst = np.clip(burst, -1.0, 1.0) * 0.65
        parts.append(burst)
        parts.append(np.zeros(n_gap, dtype=np.float32))
    y = np.concatenate(parts)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)
    return y


def _add_awgn(sig: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p_signal = float(np.mean(sig.astype(np.float64) ** 2))
    if p_signal <= 0:
        return sig
    p_noise = p_signal / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, math.sqrt(p_noise), size=sig.shape).astype(np.float32)
    return sig + noise


def main() -> None:
    rng = np.random.default_rng(0)

    clean = _tone(220.0, 3.0)
    _save(FIX / "clean_zh_short.wav", clean)

    mixed = 0.5 * (_tone(220.0, 3.0) + _tone(440.0, 3.0))
    _save(FIX / "mixed_zh_en_short.wav", mixed)

    noisy = _add_awgn(clean, snr_db=5.0, rng=rng)
    _save(FIX / "noisy_short.wav", noisy)

    e2e_proxy = _e2e_vad_speech_proxy(np.random.default_rng(42))
    _save(FIX / "e2e_vad_speech_proxy.wav", e2e_proxy)

    chunks = []
    turns = []
    spk_a_freq, spk_b_freq = 220.0, 330.0
    for i, spk in enumerate(["A", "B", "A", "B", "A"]):
        f = spk_a_freq if spk == "A" else spk_b_freq
        chunks.append(_tone(f, 6.0, amp=0.25))
        turns.append({"start_s": 6.0 * i, "end_s": 6.0 * (i + 1), "spk": f"spk{spk}"})
    dialog = np.concatenate(chunks)
    _save(FIX / "dialog_2spk_30s.wav", dialog)
    with open(FIX / "dialog_2spk_30s.rttm", "w", encoding="utf-8") as f:
        for t in turns:
            dur = t["end_s"] - t["start_s"]
            f.write(
                "SPEAKER dialog_2spk_30s 1 "
                f"{t['start_s']:.3f} {dur:.3f} <NA> <NA> {t['spk']} <NA> <NA>\n"
            )

    for i in range(1, 4):
        _save(FIX / f"enroll_spkA_{i}.wav", _tone(spk_a_freq, 2.0, amp=0.25))
        _save(FIX / f"enroll_spkB_{i}.wav", _tone(spk_b_freq, 2.0, amp=0.25))

    manifest = {
        "sr": SR,
        "files": sorted(p.name for p in FIX.glob("*.wav")),
        "rttm": sorted(p.name for p in FIX.glob("*.rttm")),
        "json": sorted(p.name for p in FIX.glob("*.json")),
    }
    with open(FIX / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[ok] generated {len(manifest['files'])} wavs into {FIX}")


if __name__ == "__main__":
    main()
