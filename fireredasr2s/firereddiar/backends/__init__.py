# Copyright 2026 Xiaohongshu.
"""Pluggable diarization backends (ModelScope default, optional pyannote)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def run_diarization_backend(
    backend: str,
    wav_np: np.ndarray,
    sample_rate: int,
    vad_segments: list[tuple[float, float]],
    *,
    model_id: str,
    model_revision: str | None,
    hf_token: str = "",
    diar_input_mode: str = "full",
    wav_path: str = "",
    spectral_f0_hz: float = 220.0,
    spectral_f1_hz: float = 330.0,
) -> list[tuple[float, float, int]] | None:
    name = (backend or "modelscope_campplus").strip().lower()
    if name in ("rttm_sidecar", "rttm", "sidecar_rttm"):
        return _run_rttm_sidecar(wav_path)
    if name in ("spectral_tone_pair", "spectral_tone", "tone_pair"):
        from fireredasr2s.firereddiar.backends.spectral_tone import (
            run_spectral_tone_pair_diar,
        )

        return run_spectral_tone_pair_diar(
            wav_np,
            sample_rate,
            f0_hz=float(spectral_f0_hz),
            f1_hz=float(spectral_f1_hz),
        )
    if name in ("modelscope", "modelscope_campplus", "campplus"):
        from fireredasr2s.firereddiar.diar import (
            build_diar_input_from_vad,
            build_diar_input_full,
            run_diarization,
        )

        if (diar_input_mode or "full").strip().lower() == "full":
            diar_in = build_diar_input_full(wav_np, sample_rate)
        else:
            diar_in = build_diar_input_from_vad(wav_np, sample_rate, vad_segments)
        if diar_in is None:
            return None
        return run_diarization(model_id, diar_in, model_revision=model_revision)
    if name == "pyannote":
        return _run_pyannote(wav_np, sample_rate, hf_token)
    if name in ("speakerlab", "3dspeaker"):
        raise NotImplementedError(
            "speakerlab backend is not wired in this build; use modelscope_campplus, "
            "spectral_tone_pair, rttm_sidecar, or pyannote."
        )
    raise ValueError(f"Unknown diar backend: {backend!r}")


def _run_rttm_sidecar(wav_path: str) -> list[tuple[float, float, int]] | None:
    """Load ``<stem>.rttm`` next to the wav (test / offline ground-truth diarization)."""
    p = (wav_path or "").strip()
    if not p:
        return None
    rttm = Path(p).with_suffix(".rttm")
    if not rttm.is_file():
        return None
    from fireredasr2s.firereddiar.rttm_io import (
        read_rttm_label_spans,
        rttm_spans_to_speaker_ids,
    )

    labeled = read_rttm_label_spans(str(rttm))
    if not labeled:
        return None
    return rttm_spans_to_speaker_ids(labeled)


def _run_pyannote(
    wav_np: np.ndarray,
    sample_rate: int,
    hf_token: str,
) -> list[tuple[float, float, int]] | None:
    try:
        import soundfile as sf
        from pyannote.audio import Pipeline
    except Exception:
        return None
    if not hf_token:
        return None
    try:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    except Exception:
        return None
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, wav_np, sample_rate, subtype="PCM_16")
        diar = pipe(path)
        out: list[tuple[float, float, int]] = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            lab = str(speaker)
            suffix = lab.split("_")[-1]
            try:
                spk_int = int(suffix)
            except ValueError:
                spk_int = abs(hash(lab)) % 10_000
            out.append((float(turn.start), float(turn.end), spk_int))
        return out or None
    except Exception:
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
