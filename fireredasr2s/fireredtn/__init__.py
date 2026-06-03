"""Inverse Text Normalization (ITN) module for FireRedASR2S.

Public API:

- ``FireRedItn``: thin wrapper around ``cn2an.transform`` with curated
  preprocessing/postprocessing rules so common spoken numerics in CN/EN/mixed
  text get rewritten to standard forms (digits, ``%``, ``℃`` ...).
- ``FireRedItnConfig``: feature flags + an optional skip-list.

Example:
    >>> from fireredasr2s.fireredtn import FireRedItn, FireRedItnConfig
    >>> itn = FireRedItn(FireRedItnConfig())
    >>> itn.process("我有三百二十块钱")
    '我有320块钱'
"""

from .itn import FireRedItn, FireRedItnConfig

__all__ = ["FireRedItn", "FireRedItnConfig"]
