# 流式 / 全双工集成契约（FireRedASR2S）

面向 **用户麦克风 → ASR** 与 **TTS 播放** 同会话编排的约定，便于应用层与 TTS 引擎对齐。

## 流式事件（`FireRedAsr2StreamSession`）

| `event` | 含义 | 关键字段 |
|---------|------|----------|
| `vad_speech_start` | Stream-VAD 判定起讲（需 `open_stream(..., emit_vad_boundaries=True)`） | `start_ms`, `sample_index` |
| `segment_final` | 一段语音结束并完成 `process_pcm_segment` | `segment_index`, `start_ms`, `end_ms`, `pipeline` |

`pipeline` 结构与离线分段一致（`text`, `sentences`, …）。

## 全双工扩展（`FireRedFullDuplexStreamSession`）

在 **本地播放窗口** 内（`begin_local_playback` … `end_local_playback`）：

| `event` | 含义 | 附加字段 |
|---------|------|----------|
| `barge_in` | 播放中检测到用户起讲 | `playback_id`, `anchor_wallclock_ms`（若调用时传入） |
| `segment_final` | 同左 | `during_local_playback=true`；可选 `playback_id`, `anchor_wallclock_ms` |

### 与 TTS 对齐

1. TTS **开始播放**（或音频设备真正开始出声）时调用：

   ```python
   session.begin_local_playback(
       playback_id=tts_utterance_id,
       anchor_wallclock_ms=int(time.time() * 1000),
   )
   ```

2. TTS **结束**时调用：`session.end_local_playback()`（与 `begin` **成对**，支持嵌套则按栈逆序结束）。

3. **麦克风 PCM** 与 TTS 并行持续送入 `push_microphone_pcm`（不要在应用线程里阻塞 ASR）。

## 参数建议

| 参数 | 建议 | 说明 |
|------|------|------|
| `chunk_ms` | 20～200 | 与 `examples/streaming_simulate_from_wav.py` 一致；过小增加调用开销，过大增加起讲延迟感。 |
| `telemetry` | 联调 / 压测 | `open_stream(..., telemetry=True)` 打每段 `process_pcm_segment` 耗时（日志）。 |
| `max_pcm_duration_s` | 长会话 | 限制会话内 PCM 时间线长度；裁剪时会重置 Stream-VAD（见 `stream_session` 文档）。 |

## 无 AEC 时的语义

本仓 **不实现回声消除**。**扬声器外放** 时麦克风流可能含 TTS 泄漏，`barge_in` 与 ASR 文本可能不可靠；须在方案中明确 **AEC 责任方**（见 [AEC_INTEGRATION_BOUNDARY.md](./AEC_INTEGRATION_BOUNDARY.md)）。契约层 **不保证** 无 AEC 下的打断准确率。

## 代码入口

- `fireredasr2s/stream_session.py`
- `fireredasr2s/full_duplex_stream.py`
- `fireredasr2s/fireredasr2system.py`：`open_stream`, `open_full_duplex_stream`
- 示例：`examples/streaming_simulate_from_wav.py`, `examples/full_duplex_simulate_from_wav.py`, `examples/mic_stream_to_asr.py`
