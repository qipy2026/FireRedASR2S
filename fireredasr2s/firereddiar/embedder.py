# Copyright 2026 Xiaohongshu.
"""Speaker embedding backends: deterministic hash, spectral stats, ModelScope CAM++ SV."""

from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Optional, Protocol

import numpy as np


class SpeakerEmbedder(Protocol):
    def embed_wav(self, wav_int16: np.ndarray, sample_rate: int) -> np.ndarray: ...


class ContentHashEmbedder:
    """Deterministic unit vector from raw PCM bytes (same clip => same embedding)."""

    dim: int = 128

    def embed_wav(self, wav_int16: np.ndarray, sample_rate: int) -> np.ndarray:
        _ = sample_rate
        h = hashlib.sha256(np.ascontiguousarray(wav_int16).tobytes()).digest()
        seed = int.from_bytes(h[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim)
        n = float(np.linalg.norm(v)) + 1e-8
        return (v / n).astype(np.float64)


class SpectralStatsEmbedder:
    """Log-magnitude spectrum embedding (L2-normalized). Separates tonal / spectral patterns."""

    dim: int = 64

    def embed_wav(self, wav_int16: np.ndarray, sample_rate: int) -> np.ndarray:
        _ = sample_rate
        x = wav_int16.astype(np.float64)
        peak = float(np.max(np.abs(x))) + 1.0
        x = x / peak
        n = min(8192, max(512, x.shape[0]))
        if x.shape[0] < n:
            x = np.pad(x, (0, n - x.shape[0]))
        seg = x[-n:]
        w = np.hanning(n).astype(np.float64)
        spec = np.fft.rfft(seg * w)
        mag = np.log(np.abs(spec[1 : 1 + self.dim]) + 1e-8).astype(np.float64)
        if mag.shape[0] < self.dim:
            mag = np.pad(mag, (0, self.dim - mag.shape[0]))
        v = mag / (float(np.linalg.norm(mag)) + 1e-8)
        return v.astype(np.float64)


class ModelScopeCampplusEmbedder:
    """CAM++ speaker verification embedding via ModelScope (16 kHz mono)."""

    def __init__(
        self,
        model_id: str = "damo/speech_campplus_sv_zh-cn_16k-common",
        model_revision: Optional[str] = None,
    ):
        self.model_id = model_id
        self.model_revision = model_revision
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            try:
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks
            except ImportError as e:
                raise ImportError(
                    "modelscope (and transitive deps) required for modelscope_campplus_sv. "
                    "Install: pip install 'fireredasr2s[modelscope]'"
                ) from e
            kw: dict = {"task": Tasks.speaker_verification, "model": self.model_id}
            if self.model_revision:
                kw["model_revision"] = self.model_revision
            self._pipe = pipeline(**kw)
        return self._pipe

    def embed_wav(self, wav_int16: np.ndarray, sample_rate: int) -> np.ndarray:
        from fireredasr2s.firereddiar.audio import prepare_asr_stack_audio

        mono16, sr16 = prepare_asr_stack_audio(wav_int16, int(sample_rate))
        import soundfile as sf

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, mono16, sr16, subtype="PCM_16")
            pipe = self._pipeline()
            raw = pipe([path], output_emb=True)
            if not isinstance(raw, dict):
                raise RuntimeError(f"unexpected SV output type: {type(raw)}")
            embs = raw.get("embs")
            if embs is None:
                embs = raw.get("embedding")
            if embs is None and isinstance(raw.get("output"), (list, tuple)):
                embs = raw["output"]
            if embs is None:
                raise RuntimeError(f"SV output has no embs; keys={list(raw.keys())}")
            first = embs[0] if isinstance(embs, (list, tuple)) else embs
            emb = np.asarray(first, dtype=np.float64).ravel()
            emb = emb / (float(np.linalg.norm(emb)) + 1e-8)
            return emb
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


def get_speaker_embedder(
    name: str,
    *,
    model_id: str = "",
    model_revision: Optional[str] = None,
) -> SpeakerEmbedder:
    n = (name or "content_hash").strip().lower()
    if n in ("content_hash", "dummy", "hash"):
        return ContentHashEmbedder()
    if n in ("spectral_stats", "spectral", "tone_spectral"):
        return SpectralStatsEmbedder()
    if n in ("modelscope_campplus_sv", "campplus_sv", "modelscope_sv"):
        mid = (model_id or "").strip() or "damo/speech_campplus_sv_zh-cn_16k-common"
        rev = model_revision
        if rev == "":
            rev = None
        return ModelScopeCampplusEmbedder(model_id=mid, model_revision=rev)
    raise ValueError(f"Unknown speaker embedder: {name!r}")
