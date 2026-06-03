# Copyright 2026 Xiaohongshu.
"""LLM decoding runtime abstraction (torch / vLLM / TensorRT-LLM)."""

from __future__ import annotations

from typing import Any, Protocol


class AsrLlmRuntime(Protocol):
    def transcribe(
        self,
        model: Any,
        feats: Any,
        lengths: Any,
        input_ids: Any,
        attention_mask: Any,
        beam_size: int,
        decode_max_len: int,
        decode_min_len: int,
        repetition_penalty: float,
        llm_length_penalty: float,
        temperature: float,
    ) -> Any:
        ...
