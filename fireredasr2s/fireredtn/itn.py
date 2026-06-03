# Copyright 2026 Xiaohongshu.
"""Inverse Text Normalization for FireRedASR2S.

Pipeline for CN/CN-EN-mixed text:

1. ``_preprocess`` rewrites spoken-form variants that ``cn2an`` mishandles
   (notably "两" → "二" before a numeric measure word).
2. ``cn2an.transform(text, "cn2an")`` does the heavy lifting (CN numbers,
   "百分之X" → "X%", "X点Y" → "X.Y", etc).
3. ``_postprocess`` rewrites unit phrasings that ``cn2an`` leaves intact
   ("X度Y" → "X.Y度", "X摄氏度" → "X℃").
4. ``skip_words`` lets callers freeze terms that look numeric but should not
   be transformed (e.g. proper nouns).

The class is idempotent for already-normalized text and returns the input
unchanged when ``cn2an`` is unavailable (graceful degrade).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


_TWO_NEXT_TOKENS = "万千百十亿个本只张条匹头道辆次倍点位天年月日岁元块毛分号"
_RE_LIANG = re.compile(rf"两(?=[{_TWO_NEXT_TOKENS}])")
_RE_DEGREE_DECIMAL = re.compile(r"(\d+)度(\d+)(?!\d)")
_RE_CELSIUS = re.compile(r"(\d+(?:\.\d+)?)摄氏度")


@dataclass
class FireRedItnConfig:
    enable_chinese_numbers: bool = True
    enable_unit_rules: bool = True
    skip_words: list[str] = field(default_factory=list)
    """Words frozen via placeholder substitution before ITN runs."""


class FireRedItn:
    """Pipeline-style ITN; thread-safe (no mutable state on the instance)."""

    def __init__(self, config: FireRedItnConfig | None = None) -> None:
        self.config = config or FireRedItnConfig()
        try:
            import cn2an  # noqa: F401
            self._has_cn2an = True
        except Exception as e:
            logger.warning("cn2an not available, ITN will be a no-op: %s", e)
            self._has_cn2an = False

    def process(self, text: str) -> str:
        if not text:
            return text
        if not self._has_cn2an or not self.config.enable_chinese_numbers:
            return text

        placeholders, frozen = self._freeze_skip_words(text)
        try:
            converted = self._run_cn2an(frozen)
        except Exception as e:
            logger.debug("cn2an.transform failed on %r: %s", frozen, e)
            converted = frozen
        if self.config.enable_unit_rules:
            converted = self._postprocess_units(converted)
        return self._unfreeze(converted, placeholders)

    def process_batch(self, texts: list[str]) -> list[str]:
        return [self.process(t) for t in texts]

    @staticmethod
    def _preprocess(text: str) -> str:
        return _RE_LIANG.sub("二", text)

    def _run_cn2an(self, text: str) -> str:
        import cn2an

        text = self._preprocess(text)
        return cn2an.transform(text, "cn2an")

    @staticmethod
    def _postprocess_units(text: str) -> str:
        text = _RE_DEGREE_DECIMAL.sub(r"\1.\2度", text)
        text = _RE_CELSIUS.sub(r"\1℃", text)
        return text

    def _freeze_skip_words(self, text: str) -> tuple[dict[str, str], str]:
        if not self.config.skip_words:
            return {}, text
        mapping: dict[str, str] = {}
        out = text
        for i, word in enumerate(self.config.skip_words):
            if not word or word not in out:
                continue
            ph = f"\x00ITN{i:04d}\x00"
            mapping[ph] = word
            out = out.replace(word, ph)
        return mapping, out

    @staticmethod
    def _unfreeze(text: str, placeholders: dict[str, str]) -> str:
        for ph, word in placeholders.items():
            text = text.replace(ph, word)
        return text
