# Copyright 2026 Xiaohongshu.
"""Default PyTorch LLM decoding (same path as pre-refactor ``FireRedAsrLlm.transcribe``)."""

from __future__ import annotations

from typing import Any


class TorchLlmRuntime:
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
        return model.transcribe(
            feats,
            lengths,
            input_ids,
            attention_mask,
            beam_size,
            decode_max_len,
            decode_min_len,
            repetition_penalty,
            llm_length_penalty,
            temperature,
        )
