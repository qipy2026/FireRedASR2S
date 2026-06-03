"""T8: speaker registry + deterministic embedder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from fireredasr2s.firereddiar.embedder import (
    ContentHashEmbedder,
    SpectralStatsEmbedder,
    get_speaker_embedder,
)
from fireredasr2s.firereddiar.enroll import SpeakerRegistry


def test_content_hash_embedder_stable():
    emb = ContentHashEmbedder()
    x = np.random.randint(-1000, 1000, size=4000, dtype=np.int16)
    a = emb.embed_wav(x, 16000)
    b = emb.embed_wav(x, 16000)
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-6


def test_registry_match_same_audio(tmp_path: Path):
    wav_path = tmp_path / "u.wav"
    x = np.random.randint(-2000, 2000, size=8000, dtype=np.int16)
    sf.write(wav_path, x, 16000, subtype="PCM_16")

    emb = get_speaker_embedder("content_hash")
    reg = SpeakerRegistry("")
    x16, sr = sf.read(wav_path, dtype="int16")
    reg.register_vector("alice", emb.embed_wav(x16, sr))
    e2 = emb.embed_wav(x16, sr)
    name, sim = reg.best_match(e2, threshold=0.999)
    assert name == "alice"
    assert sim >= 0.999


def test_registry_persist_roundtrip(tmp_path: Path):
    p = tmp_path / "spk.json"
    reg = SpeakerRegistry(str(p))
    v = np.ones(8, dtype=np.float64)
    reg.register_vector("bob", v)
    reg.save()
    reg2 = SpeakerRegistry(str(p))
    assert reg2.count() == 1
    name, sim = reg2.best_match(v, threshold=0.99)
    assert name == "bob"
    assert sim > 0.99


def test_spectral_stats_embedder_separates_fixture_tones(fixtures_dir: Path):
    emb = SpectralStatsEmbedder()
    a1, _ = sf.read(str(fixtures_dir / "enroll_spkA_1.wav"), dtype="int16")
    a2, _ = sf.read(str(fixtures_dir / "enroll_spkA_2.wav"), dtype="int16")
    b1, _ = sf.read(str(fixtures_dir / "enroll_spkB_1.wav"), dtype="int16")
    ea1 = emb.embed_wav(a1, 16000)
    ea2 = emb.embed_wav(a2, 16000)
    eb1 = emb.embed_wav(b1, 16000)
    sim_aa = float(np.dot(ea1, ea2))
    sim_ab = float(np.dot(ea1, eb1))
    assert sim_aa > sim_ab + 0.05


def test_get_speaker_embedder_accepts_spectral_stats():
    emb = get_speaker_embedder("spectral_stats")
    assert isinstance(emb, SpectralStatsEmbedder)


def test_registry_json_format(tmp_path: Path):
    p = tmp_path / "spk.json"
    reg = SpeakerRegistry(str(p))
    reg.register_vector("c", np.arange(4, dtype=np.float64))
    reg.save()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "c" in data
    assert len(data["c"]) == 4
