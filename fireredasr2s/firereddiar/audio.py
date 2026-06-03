# Copyright 2026 Xiaohongshu.

"""Mono + resampling helpers for 16 kHz CAM++/FireRed stacks."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ASR_STACK_SAMPLE_RATE = 16000


def to_mono_int16(wav: np.ndarray) -> np.ndarray:
    """Collapse multi-channel int16/float wav to mono int16 (mean across channels)."""
    if wav.ndim == 1:
        return wav.astype(np.int16, copy=False)
    if wav.ndim == 2:
        x = wav.astype(np.float64).mean(axis=1)
        return np.clip(np.round(x), -32768, 32767).astype(np.int16)
    raise ValueError(f"Expected 1D or 2D audio, got shape {wav.shape}")


def resample_int16_pcm(wav_int16: np.ndarray, src_sr: int, dst_sr: int = ASR_STACK_SAMPLE_RATE) -> np.ndarray:
    """Resample mono int16 PCM using torchaudio (high-quality band-limited sinc)."""
    if src_sr <= 0 or dst_sr <= 0:
        raise ValueError(f"Invalid sample rates: src={src_sr} dst={dst_sr}")
    if wav_int16.ndim != 1:
        raise ValueError("resample_int16_pcm expects mono 1-D array")
    if src_sr == dst_sr:
        return wav_int16
    import torch
    import torchaudio.functional as AF

    x = torch.from_numpy(wav_int16.astype(np.float32) / 32768.0).view(1, -1)
    y = AF.resample(x, orig_freq=int(src_sr), new_freq=int(dst_sr))
    y_np = y.squeeze(0).detach().cpu().numpy().astype(np.float64)
    y_i16 = np.clip(np.round(y_np * 32767.0), -32768, 32767).astype(np.int16)
    return y_i16


def prepare_asr_stack_audio(
    wav: np.ndarray,
    sample_rate: int,
    target_sr: int = ASR_STACK_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Mono + resample to ``target_sr`` (default 16 kHz) for VAD/ASR/diar/SV."""
    mono = to_mono_int16(wav)
    sr = int(sample_rate)
    if sr != int(target_sr):
        logger.info("Resampling audio %s Hz -> %s Hz (mono) for FireRed stack", sr, int(target_sr))
        mono = resample_int16_pcm(mono, sr, int(target_sr))
        sr = int(target_sr)
    return mono, sr


def _load_int16_via_torchaudio(path: str) -> tuple[np.ndarray, int]:
    import torch
    import torchaudio

    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    x = wav.clamp(-1.0, 1.0).cpu().numpy().astype(np.float64) * 32767.0
    pcm = np.clip(np.round(x), -32768, 32767).astype(np.int16)
    return pcm, int(sr)


def load_pcm_int16_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Load mono int16 PCM via soundfile; on failure (e.g. MP3) fall back to torchaudio."""
    import soundfile as sf

    p = str(Path(path).resolve())
    try:
        try:
            data, sr = sf.read(p, dtype="int16", always_2d=False)
        except TypeError:
            data, sr = sf.read(p, dtype="int16")
    except Exception as exc:
        logger.info("soundfile could not read %s (%s); trying torchaudio", p, exc)
        try:
            data, sr = _load_int16_via_torchaudio(p)
        except Exception as exc2:
            raise RuntimeError(
                f"Could not decode audio {p!r}: soundfile: {exc!r}; torchaudio: {exc2!r}. "
                "For MP3/M4A install a torchaudio codec backend (often ffmpeg in PATH)."
            ) from exc2
    if data.ndim > 1:
        data = to_mono_int16(data)
    return data.astype(np.int16, copy=False), int(sr)
