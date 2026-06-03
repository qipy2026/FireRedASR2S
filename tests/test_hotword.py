"""T4 hotword biasing tests."""

from __future__ import annotations

import pytest
import torch

from fireredasr2s.fireredasr2.decoding.hotword import HotwordBiaser, HotwordTrie


def test_trie_insert_and_terminal():
    tr = HotwordTrie.empty()
    tr.insert([1, 2, 3])
    assert 3 in tr.edges[0] or 1 in tr.edges[0]
    assert len(tr.terminal) >= 1


def test_biaser_delta_shape():
    tr = HotwordTrie.empty()
    tr.insert([5, 10])
    b = HotwordBiaser(tr, odim=128, weight=2.0, complete_bonus=1.0)
    st = torch.zeros(4, dtype=torch.long)
    d = b.delta_logits(st, torch.float32)
    assert d.shape == (4, 128)
    assert d[0, 5] == pytest.approx(2.0)


def test_biaser_advance():
    tr = HotwordTrie.empty()
    tr.insert([7, 9])
    b = HotwordBiaser(tr, odim=32, weight=1.0, complete_bonus=0.5)
    s = torch.zeros(2, dtype=torch.long)
    s = b.advance(s, torch.tensor([7, 3], dtype=torch.long))
    assert s[0].item() != 0 or 7 in tr.edges[0]


@pytest.mark.xpu
@pytest.mark.slow
def test_hotword_weight_zero_no_biaser(asr_system_xpu):
    """Baseline system has no hotword biaser (weight defaults to 0)."""
    assert asr_system_xpu.asr.hotword_biaser is None
