# 回声消除（AEC）集成边界

FireRedASR2S **不把生产级 AEC 作为内置前提**：全双工场景下，若 **扬声器播放 TTS** 与 **麦克风上行** 同时工作，未在 **采集/OS/WebRTC** 侧做回声处理时常见现象包括：

- `barge_in` **误报**（把回声当作用户起讲）或 **漏报**；
- ASR 文本混入 **TTS 播放内容**（回声串入识别）。

## 责任边界建议

| 层级 | 职责 |
|------|------|
| **客户端 / 浏览器** | 使用 **WebRTC APM**、OS 自带全双工处理、或 **耳机拾音** 避免外放。 |
| **采集服务** | 在送入本仓库前对麦克风流做 AEC，或 **播放期间暂停上行**（产品可接受时）。 |
| **FireRedASR2S** | 假设输入为 **已按产品约定处理过的 16 kHz 单声道 PCM**；提供 `barge_in` / `during_local_playback` 等 **编排信号**。另附 **可选** 软件级 **NLMS 自适应滤波**（`fireredasr2s.duplex.NlmsMonoAec`）：仅作外放场景的 **轻量缓解**，**不能替代** WebRTC/OS AEC；滤波长度、`ref_delay_samples` 需按设备时延调参。 |

## 可选：NLMS（演示 / PoC）

- 实现：`fireredasr2s/duplex/nlms_aec.py`；单测：`tests/test_nlms_aec.py`。
- 示例：[`examples/full_duplex_mic_tts_demo.py`](../examples/full_duplex_mic_tts_demo.py)（同一块 duplex 回调里 **播放参考 `ref` 与麦克风流对齐**，再 `push_microphone_pcm`）。
- **仍推荐**：集成方在客户端使用系统或 WebRTC AEC，或 **耳机拾音**；NLMS 在强混响、非线性失真、立体声路由错误时效果有限。

## 联调建议

1. **基线**：耳机 + 关闭外放，验证 ASR 与 `barge_in` 行为。  
2. **外放**：在目标设备上开启/关闭系统或 WebRTC AEC，对比误报率。  
3. **文档化**：在交付说明中写明「AEC 由哪一组件提供」及失败回退（例如静音上行）。

## 相关文档

- [STREAMING_FULL_DUPLEX_CONTRACT.md](./STREAMING_FULL_DUPLEX_CONTRACT.md)
- [MICROPHONE_ASR_INTEGRATION.md](./MICROPHONE_ASR_INTEGRATION.md)
- 全双工 + TTS 实机演示说明见示例脚本顶部 docstring（`full_duplex_mic_tts_demo.py`）；可选依赖：`pip install -e ".[duplex]"`。
