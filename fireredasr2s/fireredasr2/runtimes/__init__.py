# Copyright 2026 Xiaohongshu.

from __future__ import annotations

from typing import Any

from .torch_runner import TorchLlmRuntime
from .trtllm_runner import TrtLlmRuntime
from .vllm_runner import VllmLlmRuntime


def get_llm_runtime(name: str) -> Any:
    n = (name or "torch").strip().lower()
    if n == "torch":
        return TorchLlmRuntime()
    if n == "vllm":
        return VllmLlmRuntime()
    if n in ("trtllm", "tensorrt_llm", "trt-llm"):
        return TrtLlmRuntime()
    raise ValueError(f"Unknown LLM runtime: {name!r}")
