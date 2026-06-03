# Copyright 2026 Xiaohongshu.
"""vLLM-backed runtime (CUDA). Not used on Intel XPU."""

from __future__ import annotations

from typing import Any


class VllmLlmRuntime:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        try:
            import torch
        except Exception:
            torch = None  # type: ignore
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("vLLM runtime requires CUDA; use runtime='torch' on XPU/CPU.")

    def transcribe(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "vLLM runner is a placeholder; use runtime='torch' or implement prompt_embeds path."
        )
