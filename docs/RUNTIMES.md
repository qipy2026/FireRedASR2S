# FireRedASR2-LLM 推理运行时（骨架说明）

`FireRedAsr2Config.runtime` / CLI `--asr_runtime` 选择 **LLM 分支** 使用的解码后端；**AED 路径忽略该字段**。

| 取值 | 类 | 行为 |
|------|-----|------|
| `torch`（默认） | `TorchLlmRuntime` | 调用 `FireRedAsrLlm.transcribe` 的 PyTorch 路径；适用于 **CPU / CUDA / Intel XPU**（与 `asr_device` 一致）。 |
| `vllm` | `VllmLlmRuntime` | **仅当 `torch.cuda.is_available()`** 时允许构造；`transcribe` 仍为占位（`NotImplementedError`）。在 XPU/CPU 环境请保持 `torch`。 |
| `trtllm` | `TrtLlmRuntime` | 占位；`transcribe` 抛出 `NotImplementedError`。 |

工厂方法：`fireredasr2s.fireredasr2.runtimes.get_llm_runtime(name)`。

## Intel XPU

请在 XPU 机器上使用：

```text
--asr_type llm --asr_runtime torch --asr_device xpu
```

并建议设置 `PYTORCH_ENABLE_XPU_FALLBACK=1`（与主 README / pytest 一致），以便部分算子在无 XPU 内核时回退 CPU。

## vLLM（CUDA 参考）

本仓库内 **未实现** 完整 vLLM 转写逻辑；上游 vLLM 对 FireRedASR2-LLM 的集成见官方 PR / 文档。可参考脚本示例：`scripts/run_vllm_llm_asr.sh`（需自行补齐环境与模型路径）。

**Windows 本机**：AED 半精度（CUDA/XPU）与 CPU 动态 INT8 量化流程见 [`WINDOWS_INFERENCE_SPEED.md`](WINDOWS_INFERENCE_SPEED.md)。
