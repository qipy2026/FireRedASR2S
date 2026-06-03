"""T6 diarization input + word-level alignment tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from fireredasr2s.firereddiar.align import (
    group_tokens_by_speaker,
    merge_short_speaker_groups,
    speaker_by_overlap_ms,
    try_word_diar_sentences,
)
from fireredasr2s.firereddiar.diar import build_diar_input_full, build_diar_input_from_vad
import numpy as np


def test_speaker_by_overlap_ms_basic():
    spans = [(0.0, 1.0, 7), (1.0, 2.0, 8)]
    assert speaker_by_overlap_ms(100, 900, spans) == 7
    assert speaker_by_overlap_ms(1100, 1900, spans) == 8


def test_word_level_grouping():
    diar = [(0.0, 0.5, 0), (0.5, 2.0, 1)]
    start_ms = 0
    ts = [["a", 0.0, 0.2], ["b", 0.2, 0.4], ["c", 0.6, 0.9], ["d", 1.0, 1.5]]
    g = group_tokens_by_speaker(start_ms, ts, diar)
    assert len(g) >= 2
    assert any(spk == 0 for *_, spk, _ in g) and any(spk == 1 for *_, spk, _ in g)


def test_min_speaker_dur_merge():
    groups = [(0, 50, 0, "a"), (50, 60, 0, "b"), (60, 2000, 0, "c")]
    merged = merge_short_speaker_groups(groups, min_dur_ms=100)
    assert len(merged) < len(groups)


def test_try_word_diar_requires_single_punc_sentence():
    diar = [(0.0, 10.0, 0)]
    asr = {
        "uttid": "u_s0_e10000",
        "confidence": 0.9,
        "timestamp": [["x", 0.0, 0.5], ["y", 0.5, 1.0]],
    }
    punc_multi = {"uttid": "u_s0_e10000", "punc_sentences": [{"punc_text": "a", "start_s": 0, "end_s": 0.4}, {"punc_text": "b", "start_s": 0.4, "end_s": 1.0}]}
    assert try_word_diar_sentences(
        asr_result=asr,
        punc_result=punc_multi,
        lid_result=None,
        diar_spans=diar,
        vad_segment_idx=0,
        min_speaker_dur_ms=0,
        enable_punc=True,
        punc_model=None,
    ) is None


def test_build_diar_input_full_short_skips():
    x = np.zeros(16000, dtype=np.int16)
    assert build_diar_input_full(x, 16000) is None


def test_build_diar_input_full_long():
    x = np.zeros(16000 * 10, dtype=np.int16)
    out = build_diar_input_full(x, 16000)
    assert out is not None and len(out) == 1
    assert out[0][0] == 0.0 and abs(out[0][1] - 10.0) < 0.05


@pytest.mark.modelscope
@pytest.mark.slow
def test_full_pipeline_der_drop():
    pytest.skip("Requires ModelScope diarization + stable offline reference run")


def test_eval_diarization_script(tmp_path, fixtures_dir):
    ref = fixtures_dir / "dialog_2spk_30s.rttm"
    hyp = tmp_path / "hyp.rttm"
    hyp.write_text(ref.read_text(encoding="utf-8"), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(Path("scripts/eval_diarization.py")), str(ref), str(hyp)],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
    )
    if r.returncode == 2:
        pytest.skip("pyannote.metrics not installed")
    assert r.returncode == 0
    assert "DER=" in r.stdout
