# 用户麦克风 → FireRedASR2S（16 kHz PCM）集成说明

本仓库 **不提供** 麦克风驱动与 OS 音频栈；以下说明如何把 **用户语音（麦克风）** 以约定格式送入 `FireRedAsr2System.open_stream()` / `open_full_duplex_stream()`。

## 约定

| 项 | 要求 |
|----|------|
| 采样率 | 送入栈前统一到 **16 kHz**（`prepare_asr_stack_audio` 可处理常见输入采样率） |
| 格式 | **`int16` 单声道** 块（`numpy.ndarray` shape `(N,)`) |
| 分块 | 建议 **20 ms～200 ms** 每块（与 `examples/streaming_simulate_from_wav.py` 的 `--chunk_ms` 一致） |
| 全双工 | 扬声器播放 TTS 同时开麦时，优先在 **采集层或 WebRTC/OS** 做 **AEC**，或 **播放时静音上行**；可选软件 NLMS 见 [AEC_INTEGRATION_BOUNDARY.md](./AEC_INTEGRATION_BOUNDARY.md) 与 [`examples/full_duplex_mic_tts_demo.py`](../examples/full_duplex_mic_tts_demo.py) |

## 实施阶段（摘要）

1. **P0 需求**：目标平台（桌面/Web/移动）、设备原生采样率、权限与路由。
2. **P1 采集 PoC**：`sounddevice` / PyAudio / `getUserMedia` 连续读入 → 队列。
3. **P2 重采样**：48 k / 44.1 k → 16 k，出口 `int16` mono。
4. **P3 对接 ASR**：循环调用 `push_pcm_int16_mono` / `push_microphone_pcm`，会话结束 `finalize()`。
5. **P4 全双工**：与 TTS 对齐调用 `begin_local_playback` / `end_local_playback`（可带 `playback_id` / 时间锚点，见 [STREAMING_FULL_DUPLEX_CONTRACT.md](./STREAMING_FULL_DUPLEX_CONTRACT.md)）。
6. **P5 SLO**：首段延迟、切段延迟、采集欠载次数；可与流式 **telemetry** 日志合并分析。

## 示例脚本

- 实时麦克风（需安装 `sounddevice`）：[`examples/mic_stream_to_asr.py`](../examples/mic_stream_to_asr.py)
- 全双工：TTS 外放 + 麦克风流式 ASR（可选 NLMS）：[`examples/full_duplex_mic_tts_demo.py`](../examples/full_duplex_mic_tts_demo.py)（依赖见 `pyproject.toml` 的 `[duplex]`）
- WAV 分块模拟采集节奏：[`examples/streaming_simulate_from_wav.py`](../examples/streaming_simulate_from_wav.py)

## 可选：长会话 PCM 上限

`FireRedAsr2StreamSession` 支持 `max_pcm_duration_s`：超过上限且无未闭合 VAD 段时，从时间线头部丢弃旧样本并重置 Stream-VAD 状态（**会丢失早期音频上的起讲状态**）。详见 `fireredasr2s/stream_session.py` 文档字符串。

## 与端到端「硅基员工」编排的关系

对话状态机、LLM、工单、业务指标看板等 **不在 FireRedASR2S 主仓**；本仓交付 **ASR 与流式事件**。集成方可单独维护编排服务，并按本文档接入麦克风流。
