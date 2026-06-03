# Copyright 2026 Xiaohongshu.
"""Hotword biasing for AED beam search (shallow fusion on decoder logits)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from fireredasr2s.fireredasr2.asr import FireRedAsr2Config


@dataclass
class HotwordTrie:
    """Sparse trie: ``edges[s][tok] = next_state``."""

    edges: list[dict[int, int]]
    terminal: set[int]

    @classmethod
    def empty(cls) -> HotwordTrie:
        return cls(edges=[{}], terminal=set())

    def insert(self, token_ids: list[int]) -> None:
        s = 0
        for tid in token_ids:
            if tid not in self.edges[s]:
                self.edges[s][tid] = len(self.edges)
                self.edges.append({})
            s = self.edges[s][tid]
        self.terminal.add(s)


class HotwordBiaser:
    """Per-beam trie state + additive log-bonuses for next-token distribution."""

    def __init__(
        self,
        trie: HotwordTrie,
        odim: int,
        weight: float,
        complete_bonus: float,
    ) -> None:
        self.trie = trie
        self.odim = odim
        self.weight = float(weight)
        self.complete_bonus = float(complete_bonus)

    def delta_logits(self, states: torch.Tensor, ref_dtype: torch.dtype) -> torch.Tensor:
        """``states`` (N*B,) int64 trie node ids → (N*B, odim) additive bias."""
        device = states.device
        NB = states.shape[0]
        out = torch.zeros(NB, self.odim, device=device, dtype=ref_dtype)
        if self.weight == 0.0 and self.complete_bonus == 0.0:
            return out
        edges = self.trie.edges
        term = self.trie.terminal
        for i in range(NB):
            s = int(states[i].item())
            for tok, ns in edges[s].items():
                bonus = self.weight
                if ns in term:
                    bonus += self.complete_bonus
                if 0 <= tok < self.odim:
                    out[i, tok] = bonus
        return out

    def advance(self, states: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Update trie state after appending ``tokens`` (N*B,) int64."""
        edges = self.trie.edges
        new = states.clone()
        for i in range(states.shape[0]):
            s = int(states[i].item())
            tid = int(tokens[i].item())
            ns = edges[s].get(tid)
            if ns is not None:
                new[i] = ns
            else:
                ns0 = edges[0].get(tid)
                new[i] = ns0 if ns0 is not None else 0
        return new


def build_hotword_biaser(
    tokenizer: Any,
    config: "FireRedAsr2Config",
    odim: int,
) -> Optional[HotwordBiaser]:
    if not getattr(config, "hotwords", None) or float(getattr(config, "hotword_weight", 0.0)) <= 0.0:
        return None
    trie = HotwordTrie.empty()
    for phrase in config.hotwords:
        if not (phrase or "").strip():
            continue
        _, ids = tokenizer.tokenize(str(phrase).strip(), replace_punc=True)
        if ids:
            trie.insert(ids)
    return HotwordBiaser(
        trie,
        odim=odim,
        weight=float(config.hotword_weight),
        complete_bonus=float(getattr(config, "hotword_complete_bonus", 0.0)),
    )
