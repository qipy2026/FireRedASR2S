# Copyright 2026 Xiaohongshu.

"""Production-oriented presets: ModelScope CAM++ diarization + CAM++ SV (natural speech)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fireredasr2s.fireredasr2system import FireRedAsr2SystemConfig

# Hub IDs align with ModelScope DAMO CAM++ family (Chinese-centric).
CAMPLUS_DIAR_HUB_ID = "damo/speech_campplus_speaker-diarization_common"
CAMPLUS_SV_HUB_ID = "damo/speech_campplus_sv_zh-cn_16k-common"

# Pin SV revision for reproducible deployments; set to None to follow hub default.
CAMPLUS_SV_DEFAULT_REVISION = "v1.0.0"
# Diar revision varies by hub; None uses ModelScope default for the model card.
CAMPLUS_DIAR_DEFAULT_REVISION: str | None = None

# Cosine similarity gate for CAM++ SV (typical raw scores ~0.3–0.6 on CN-Celeb-like data).
CAMPLUS_SV_DEFAULT_MATCH_THRESHOLD = 0.35


def with_natural_speech_speaker_stack(
    base: FireRedAsr2SystemConfig,
    *,
    diar_model_id: str = CAMPLUS_DIAR_HUB_ID,
    diar_model_revision: str | None = CAMPLUS_DIAR_DEFAULT_REVISION,
    sv_model_id: str = CAMPLUS_SV_HUB_ID,
    sv_model_revision: str | None = CAMPLUS_SV_DEFAULT_REVISION,
    sv_match_threshold: float = CAMPLUS_SV_DEFAULT_MATCH_THRESHOLD,
    enable_diarization: bool | None = None,
    enable_speaker_id: bool | None = None,
) -> FireRedAsr2SystemConfig:
    """Return a copy of ``base`` with CAM++ diar + CAM++ SV defaults for **natural speech**.

    - Diarization: ``modelscope_campplus`` segmentation + clustering (not identity).
    - Voiceprint: ``modelscope_campplus_sv`` embeddings + cosine ``sv_match_threshold``.

    Enrollment: ``FireRedAsr2System.register_speaker(name, wav_path)`` or CLI
    ``--register_speaker name=wav`` (1:N match via ``SpeakerRegistry.best_match``).

    You still need ``pip install modelscope`` (or ``pip install -e .[modelscope]``) and first-run
    model downloads. For pyannote instead, set ``diar_backend='pyannote'`` and ``diar_hf_token``.
    """
    kw: dict = {
        "diar_backend": "modelscope_campplus",
        "diar_model_id": diar_model_id,
        "diar_model_revision": diar_model_revision,
        "speaker_embedder": "modelscope_campplus_sv",
        "speaker_embedder_model_id": sv_model_id,
        "speaker_embedder_model_revision": sv_model_revision,
        "speaker_match_threshold": float(sv_match_threshold),
    }
    if enable_diarization is not None:
        kw["enable_diarization"] = bool(enable_diarization)
    if enable_speaker_id is not None:
        kw["enable_speaker_id"] = bool(enable_speaker_id)
    return replace(base, **kw)
