# Copyright 2026 Xiaohongshu.
"""Single-pass speech enhancement frontend.

Backends:

- ``noisereduce`` (default; pure-Python, MIT, ~no model download): spectral
  gating; works on arbitrary sample rates.
- ``df`` (DeepFilterNet, optional): higher quality DNN denoiser; needs
  ``pip install deepfilternet`` and downloads weights on first use.

I/O contract: ``process(wav, sr) -> (wav, sr)`` with ``wav`` shape ``(N,)``,
dtype ``int16``. Mono only; multi-channel input is mean-pooled. The output
sample rate equals the input (no resampling here).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FireRedDenoiserConfig:
    backend: str = "noisereduce"
    """One of ``noisereduce`` / ``df``."""

    stationary: bool = False
    """Only used by ``noisereduce``: assume stationary noise (less aggressive)."""

    prop_decrease: float = 0.85
    """Only used by ``noisereduce``: 0..1, controls denoise strength."""

    df_atten_lim_db: float = 30.0
    """Only used by ``df``: attenuation limit in dB."""


class FireRedDenoiser:
    """Stateful only on the loaded model; ``process`` is reentrant per call."""

    SUPPORTED_BACKENDS = ("noisereduce", "df")

    def __init__(self, config: FireRedDenoiserConfig | None = None) -> None:
        self.config = config or FireRedDenoiserConfig()
        if self.config.backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported denoise backend: {self.config.backend!r}; "
                f"choose from {self.SUPPORTED_BACKENDS}"
            )
        self._df_state = None  # lazy DeepFilterNet state
        self._check_backend_importable()

    def _check_backend_importable(self) -> None:
        if self.config.backend == "noisereduce":
            try:
                import noisereduce  # noqa: F401
            except Exception as e:  # pragma: no cover - import guard
                raise ImportError(
                    "Backend 'noisereduce' requires `pip install noisereduce`."
                ) from e
        elif self.config.backend == "df":
            try:
                from df.enhance import init_df  # noqa: F401
            except Exception as e:  # pragma: no cover - import guard
                raise ImportError(
                    "Backend 'df' requires `pip install deepfilternet`."
                ) from e

    def process(self, wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        if wav.size == 0:
            return wav.astype(np.int16, copy=False), sr
        mono = self._to_mono(wav)
        if self.config.backend == "noisereduce":
            out = self._run_noisereduce(mono, sr)
        else:
            out = self._run_df(mono, sr)
        return self._to_int16(out), sr

    @staticmethod
    def _to_mono(wav: np.ndarray) -> np.ndarray:
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav

    @staticmethod
    def _to_int16(wav: np.ndarray) -> np.ndarray:
        if wav.dtype == np.int16:
            return wav
        if np.issubdtype(wav.dtype, np.floating):
            wav = np.clip(wav, -1.0, 1.0)
            return (wav * 32767.0).astype(np.int16)
        return wav.astype(np.int16, copy=False)

    @staticmethod
    def _to_float32(wav: np.ndarray) -> np.ndarray:
        if np.issubdtype(wav.dtype, np.floating):
            return wav.astype(np.float32, copy=False)
        if wav.dtype == np.int16:
            return wav.astype(np.float32) / 32768.0
        return wav.astype(np.float32, copy=False)

    def _run_noisereduce(self, wav: np.ndarray, sr: int) -> np.ndarray:
        import noisereduce as nr

        f32 = self._to_float32(wav)
        out = nr.reduce_noise(
            y=f32,
            sr=sr,
            stationary=self.config.stationary,
            prop_decrease=float(self.config.prop_decrease),
        )
        return np.asarray(out, dtype=np.float32)

    def _run_df(self, wav: np.ndarray, sr: int) -> np.ndarray:
        from df.enhance import enhance, init_df

        if self._df_state is None:
            model, df_state, _ = init_df()
            self._df_state = (model, df_state)
        model, df_state = self._df_state

        import torch

        f32 = self._to_float32(wav)
        target_sr = df_state.sr()
        if sr != target_sr:
            logger.warning(
                "DeepFilterNet expects sr=%d, got %d; running at original sr (no resample).",
                target_sr,
                sr,
            )
        tensor = torch.from_numpy(f32).unsqueeze(0)
        out = enhance(model, df_state, tensor, atten_lim_db=self.config.df_atten_lim_db)
        return out.squeeze(0).cpu().numpy().astype(np.float32)
