"""T5: LLM runtime factory (torch / vLLM / TensorRT-LLM placeholder)."""

from __future__ import annotations

import pytest

from fireredasr2s.fireredasr2.runtimes import TrtLlmRuntime, VllmLlmRuntime, get_llm_runtime
from fireredasr2s.fireredasr2.runtimes.torch_runner import TorchLlmRuntime


def test_get_llm_runtime_torch():
    r = get_llm_runtime("torch")
    assert isinstance(r, TorchLlmRuntime)


def test_get_llm_runtime_unknown():
    with pytest.raises(ValueError):
        get_llm_runtime("unknown_runtime_xyz")


def test_trtllm_transcribe_not_implemented():
    r = TrtLlmRuntime()
    with pytest.raises(NotImplementedError):
        r.transcribe([])


@pytest.mark.cuda
def test_vllm_runtime_init_when_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    r = VllmLlmRuntime()
    with pytest.raises(NotImplementedError):
        r.transcribe([])


def test_vllm_runtime_raises_without_cuda():
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present; use test_vllm_runtime_init_when_cuda")
    with pytest.raises(RuntimeError, match="vLLM"):
        VllmLlmRuntime()
