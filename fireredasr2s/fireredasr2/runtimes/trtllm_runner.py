# Copyright 2026 Xiaohongshu.
"""TensorRT-LLM runtime placeholder."""

from __future__ import annotations

from typing import Any


class TrtLlmRuntime:
    def transcribe(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "TensorRT-LLM runtime is not implemented in this repository."
        )
