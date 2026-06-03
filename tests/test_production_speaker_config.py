"""CAM++ production preset (config only; no ModelScope inference)."""

from __future__ import annotations

from fireredasr2s.fireredasr2system import FireRedAsr2SystemConfig
from fireredasr2s.firereddiar.production import (
    CAMPLUS_DIAR_HUB_ID,
    CAMPLUS_SV_HUB_ID,
    with_natural_speech_speaker_stack,
)


def test_natural_speech_stack_sets_campplus_models():
    base = FireRedAsr2SystemConfig(enable_diarization=False, enable_speaker_id=False)
    p = with_natural_speech_speaker_stack(base)
    assert p.diar_backend == "modelscope_campplus"
    assert p.diar_model_id == CAMPLUS_DIAR_HUB_ID
    assert p.speaker_embedder == "modelscope_campplus_sv"
    assert p.speaker_embedder_model_id == CAMPLUS_SV_HUB_ID
    assert p.enable_diarization is False
    assert p.enable_speaker_id is False


def test_natural_speech_stack_can_enable_flags():
    base = FireRedAsr2SystemConfig()
    p = with_natural_speech_speaker_stack(
        base, enable_diarization=True, enable_speaker_id=True
    )
    assert p.enable_diarization is True
    assert p.enable_speaker_id is True
