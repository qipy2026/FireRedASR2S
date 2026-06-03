# 流式「部分结果」与稳定段结果（AED）

## 当前行为（AED 主路径）

`FireRedAsr2StreamSession` 在 Stream-VAD **闭合一段** 后，对该段调用 `process_pcm_segment`，事件为 **`segment_final`**。  
即：**稳定结果以「段」为单位**，**不提供** 段内逐帧或逐字的 **不稳定中间假设**（partial hypotheses）。

若产品需要「边说边出字」的 **UI 级 partial**，通常需要：

1. **上游接受段末首字延迟**：仅用 `segment_final` 更新 UI；或  
2. **更换/叠加解码形态**：例如探索 **LLM ASR** 流式解码（见 [`RUNTIMES.md`](./RUNTIMES.md) 与 `FireRedAsr2Config.runtime`），其实现与延迟特征与本仓库 AED 路径不同，需单独评估。

## 与「毫秒级响应」的关系

「毫秒级」若指 **用户停嘴到可见文本**，包含 **VAD 挂尾、整段 ASR、标点/LID 等**；仅优化 ASR 模型不足以承诺端到端 SLA，需在系统集成层 **压测**（可用 `open_stream(..., telemetry=True)` 观察段级推理耗时）。

## VAD 边界事件

`emit_vad_boundaries=True` 时可提前收到 **`vad_speech_start`**，用于 **打断 TTS** 等，但 **不等价于** 已有可展示转写文本。
