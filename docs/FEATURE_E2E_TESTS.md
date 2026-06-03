# 按功能点的录音 E2E 测试说明

本页描述 **`tests/test_e2e_by_feature.py`**：先检查**合成录音**是否存在，再对 **`FireRedAsr2System.process`** 做端到端验证。录音由 `scripts/generate_test_fixtures.py` 生成（正弦/噪声代理，非真人语音，但走真实 **WAV → VAD → ASR → …** 链路）。

### ASR 基线用例的输入音频优先级

`test_e2e_asr_vad_punc_pipeline` 按顺序选用（实现见 `_asr_e2e_input_wav`）：

1. 环境变量 **`FIREREDASR2S_E2E_ASR_WAV`**（任意 `soundfile` 或 **torchaudio** 可读格式，含 **MP3**；系统内会转单声道 16 kHz）
2. 仓库 **`assets/metting_0507.mp3`**（会议录音占位名；将文件放到 `assets/` 后无需再设环境变量即可跑主线 E2E）
3. **`assets/long_multi_speaker_65s.wav`**（若已用 `examples/test_long_multi_speaker.py` 生成）
4. **`tests/fixtures/e2e_vad_speech_proxy.wav`**（合成代理；VAD 可能无段，依赖全文件 ASR 回退）

**MP3 解码**：需本机 **ffmpeg** 等在 PATH 中（torchaudio 后端），否则请将音频转为 16 kHz WAV。

## 前置步骤

```powershell
# 1. 生成 wav（仓库内 tests/fixtures/）
python scripts/generate_test_fixtures.py

# 2. 仅跑 E2E（可选）
pytest tests/test_e2e_by_feature.py -m e2e -v

# 3. 含真实模型 + XPU 的慢速用例（需 pretrained_models 或 FIREREDASR2S_MODELS_DIR）
$env:PYTORCH_ENABLE_XPU_FALLBACK = "1"
pytest tests/test_e2e_by_feature.py -m "e2e and xpu" -v
```

## 功能点 ↔ 测试用例 ↔ 录音 ↔ 断言摘要

| 功能点 | pytest 用例 | 使用的录音文件 | 主要断言 / 说明 |
|--------|-------------|----------------|-----------------|
| **Fixture 可读** | `test_e2e_fixture_wavs_exist_and_readable` | `clean_zh_short.wav`, `noisy_short.wav`, `dialog_2spk_30s.wav` | 文件存在；16 kHz；可 `soundfile.read` |
| **ASR 主线** | `test_e2e_asr_vad_punc_pipeline` | 见下文 **ASR 输入优先级** | `uttid`, `text`, `sentences`, `vad_segments_ms`, `speaker_label_note` 等结构完整；VAD 无段时对 ≤600s 音频 **自动全文件 ASR**（与 `asr_transcribe_results` 采集逻辑一致） |
| **ITN** | `test_e2e_itn_fields_on_wav` | `clean_zh_short.wav` | `enable_itn` 时存在 `text_itn` / `text_labeled_itn` |
| **降噪** | `test_e2e_denoise_branch_on_wav` | `noisy_short.wav` | `denoise_backend == noisereduce` |
| **热词（AED）** | `test_e2e_hotword_config_on_wav` | `clean_zh_short.wav` | 配置 `hotwords` 后整段 `process` 无异常；**不强制**转写含热词（合成音常无字） |
| **热词 + 嘈杂代理** | `test_e2e_hotword_on_noisy_proxy_wav` | 临时 `noisy_proxy_hotword.wav`（对 clean 叠加轻噪） | 与热词用例相同栈；验证麦克风嘈杂代理下仍可跑通 |
| **LID** | `test_e2e_lid_on_wav` | `clean_zh_short.wav` | 每句含 `lang` 字段 |
| **说话人分离（双频）** | `test_e2e_diarization_on_dialog_wav` | `dialog_2spk_30s.wav` | `diar_backend=spectral_tone_pair`（220/330 Hz）：≥4 段、**2** 类 `speaker_id`；有句时 `diar_speaker_id` / `spk_label` 对齐 |
| **说话人分离（RTTM）** | `test_e2e_diarization_rttm_sidecar_on_dialog_wav` | `dialog_2spk_30s.wav` + `dialog_2spk_30s.rttm` | `rttm_sidecar`：**5** 段真值 |
| **说话人分离（ModelScope）** | `test_e2e_diarization_modelscope_on_dialog_wav` | `dialog_2spk_30s.wav` | `@pytest.mark.modelscope`；无 span 时 **skip**（合成音常见） |
| **声纹注册/匹配** | `test_e2e_speaker_enroll_on_wav` | `enroll_spkA_*.wav` / `enroll_spkB_*.wav` | `speaker_embedder=spectral_stats`；`utterance_enrolled_speaker` / `utterance_enrolled_similarity`（整段匹配）；有句时仍有 `enrolled_*` |
| **词级时间戳** | `test_e2e_return_timestamp_path` | `clean_zh_short.wav` | `return_timestamp=True`；有句时 `start_ms`/`end_ms`；`words` 计数写入 metric |

### 流式 ASR（独立文件 `tests/test_stream_session.py`）

| 层级 | 用例 | 依赖 | 断言 / 说明 |
|------|------|------|-------------|
| 无模型 | `test_stream_session_symbol_export` | 无 | 包导出与模块类一致 |
| 无模型 | `test_open_stream_rejects_non_aed` | 无 | 非 AED 的 dummy `config` 调用 `open_stream` 抛 `ValueError` |
| 无模型 | `test_stream_session_mock_vad_emits_segment_final` | `unittest.mock` 替换 `FireRedStreamVad.from_pretrained` | 固定长度 PCM + 模拟第 123 帧 `is_speech_end` → 恰好 **1** 条 `segment_final`，且调用伪 `process_pcm_segment` |
| 无模型 | `test_stream_session_reset_clears_state` | Mock Stream-VAD | `reset()` 会调用底层 `stream_vad.reset` |
| 无模型 | `test_stream_session_max_pcm_trims_timeline_when_no_open_segment` | Mock 全静音 VAD | `max_pcm_duration_s` 下 PCM 时间线被裁剪 |
| 无模型 | `test_full_duplex_playback_context_on_barge_in_mock` | Mock VAD | `begin_local_playback(playback_id, anchor_wallclock_ms=...)` 写入 `barge_in` / `segment_final` |
| E2E（可选） | `test_e2e_stream_session_replay_wav` | `@pytest.mark.e2e xpu slow`；输入优先级：`FIREREDASR2S_E2E_STREAM_WAV` → `assets/metting_0507_seg03.wav` → `e2e_vad_speech_proxy.wav`；与 `test_e2e_by_feature` 相同模型前置 | 分块 `push_pcm_int16_mono` + `finalize`；若 Stream-VAD 产生闭合段则校验 `event`/`pipeline` 结构；**无闭合段时 skip** |

**麦克风代理（无 ASR 模型）**：`tests/test_mic_scenario_fixtures.py` 对 `clean_zh_short.wav` 叠加噪声后校验 `prepare_asr_stack_audio` 输出 16 kHz 单声道。

```powershell
pytest tests/test_stream_session.py -v
pytest tests/test_stream_session.py -m "e2e and xpu" -v
pytest tests/test_mic_scenario_fixtures.py -v
```

## Marker 约定

| Marker | 含义 |
|--------|------|
| `e2e` | 本文件均为 E2E；可用 `-m e2e` 筛选 |
| `xpu` | 需要 Intel XPU + 本地模型目录 |
| `slow` | 加载模型与推理，耗时较长 |
| `modelscope` | `test_e2e_diarization_modelscope_on_dialog_wav` 等；**主线 Diar/声纹 E2E 默认不依赖 modelscope**（`spectral_*` + 可选 `pip install fireredasr2s[modelscope]`） |

## 与单元测试的区别

- **单元测试**（如 `test_itn.py` 文本用例、`test_hotword.py` Trie）：不依赖完整 `process(wav)`。
- **E2E**：**必须先有 wav 文件**，再走 System；更接近集成验收。

## 与 `output/long_multi_speaker` 同构的测试记录（JSON + Markdown）

在跑 E2E 用例时，若指定输出目录，会在其下为 **每个 pytest 用例** 建子目录，并写入与 `examples/test_long_multi_speaker.py` **相同三件套**：

| 文件 | 说明 |
|------|------|
| `asr_system_result.json` | 本次 `FireRedAsr2System.process` 的完整结果 |
| `asr_transcribe_results.json` | 对**磁盘上同一 wav** 做 VAD 切段后，逐段 `FireRedAsr2.transcribe` 的列表（与示例脚本一致） |
| `E2E_TEST_REPORT.md` | 简短 Markdown：wav 路径、uttid、句数、diar 条数、文本预览 |

**启用方式（二选一）**

```powershell
# 方式 A：环境变量（路径可绝对或相对，会 resolve）
$env:FIREREDASR2S_E2E_RECORD_DIR = "E:\work\aicc\FireRedASR2S\output\e2e_runs"
pytest tests/test_e2e_by_feature.py -m "e2e and xpu" -v

# 方式 B：测试矩阵脚本参数（相对仓库根目录）
python scripts/run_full_test_matrix.py --report_dir reports/ --e2e_record_dir output/e2e_runs
```

**目录示例**

```text
output/e2e_runs/
  test_e2e_asr_vad_punc_pipeline/
    asr_system_result.json
    asr_transcribe_results.json
    E2E_TEST_REPORT.md
  test_e2e_itn_fields_on_wav/
    ...
```

**降噪用例**：`process` 内部使用增强后的波形做 ASR，与「磁盘原始 wav + VAD + transcribe」不一致，因此 **`asr_transcribe_results.json` 写为空列表**，原因写在 `E2E_TEST_REPORT.md` 与工具模块注释中。

实现代码：`tests/utils/e2e_long_multi_speaker_record.py`。

## 报告输出（pytest 汇总）

与全量矩阵相同，运行：

```powershell
python scripts/run_full_test_matrix.py --report_dir reports/
```

合并后的 Markdown/HTML 中会出现任务 **「E2E 录音端到端（按功能点）」** 小节（见 `scripts/_report_writer.py` 中 `TASK_LABELS["E2E"]`）。

## 换用真人录音（可选）

可将自有 wav（建议 **16 kHz 单声道**）路径通过 **临时改测试** 或 **环境变量** 扩展（当前版本固定使用 `tests/fixtures/` 下文件名以保证 CI 可复现）。若需官方支持 `FIREREDASR2S_E2E_WAV` 等变量，可在后续迭代中加。
