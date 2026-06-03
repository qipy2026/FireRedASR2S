# FireRedASR2S 迭代验收测试报告

- 生成时间: `2026-05-08 13:48:49`    总用时: `107.23s`    PASS: **70**  FAIL/ERROR: **0**  SKIP: **7**  TOTAL: **77**
- 详细 HTML 报告: [test_report.html](test_report.html)

## 环境

| 字段 | 值 |
|---|---|
| `platform` | `win32` |
| `python` | `3.12.10` |
| `torch` | `2.11.0+xpu` |
| `has_xpu` | `True` |
| `xpu_device` | `Intel(R) Arc(TM) Graphics` |
| `has_cuda` | `False` |
| `has_modelscope` | `True` |
| `has_speakerlab` | `False` |

## 总览

| 任务 | 名称 | PASS | FAIL/ERROR | SKIP | 用时 |
|---|---|---|---|---|---|
| T0 | T0 测试基础设施 | 19 | 0 | 1 | 0.09s |
| T1 | T1 ITN（逆文本正则化） | 10 | 0 | 0 | 0.04s |
| T2 | T2 独立降噪前端 | 3 | 0 | 2 | 13.25s |
| T3 | T3 量化与加速兜底 | 6 | 0 | 1 | 23.30s |
| T4 | T4 自定义热词偏置 | 4 | 0 | 0 | 0.02s |
| T5 | T5 加速运行时骨架 | 4 | 0 | 1 | 0.00s |
| T6 | T6 声纹分离输入与对齐 | 6 | 0 | 2 | 0.38s |
| T7 | T7 声纹分离多 backend | 5 | 0 | 0 | 0.00s |
| T8 | T8 声纹注册 | 4 | 0 | 0 | 0.02s |
| E2E | E2E 录音端到端（按功能点） | 9 | 0 | 0 | 70.14s |

## T0 测试基础设施

| 测试 | 结果 | 用时 | 关键指标 |
|---|---|---|---|
| `test_infra_smoke::test_fixtures_present` | **PASS** | 0.00s |  |
| `test_infra_smoke::test_clean_audio_loads` | **PASS** | 0.00s |  |
| `test_infra_smoke::test_wer_basic_metrics` | **PASS** | 0.00s |  |
| `test_infra_smoke::test_estimate_snr_reasonable` | **PASS** | 0.00s |  |
| `test_infra_smoke::test_rttm_roundtrip` | **PASS** | 0.00s |  |
| `test_infra_smoke::test_dialog_rttm_aligns_with_audio` | **PASS** | 0.00s |  |
| `test_infra_smoke::test_numbers_samples_schema` | **PASS** | 0.00s |  |
| `test_infra_smoke::test_record_metric_works` | **PASS** | 0.00s | `smoke_dummy=0.123` |
| `test_infra_smoke::test_xpu_runtime_is_real` | **PASS** | 0.00s |  |
| `test_report_meta::test_parse_junit_extracts_status_and_metrics` | **PASS** | 0.01s |  |
| `test_report_meta::test_render_markdown_contains_all_sections` | **PASS** | 0.00s |  |
| `test_report_meta::test_write_report_creates_md` | **PASS** | 0.01s |  |
| `test_report_meta::test_record_metric_pipeline` | **PASS** | 0.00s | `smoke_e2e=0.999` |
| `TestTorchDeviceResolver::test_empty_device_use_gpu_true_cuda_wins` | **PASS** | 0.00s |  |
| `TestTorchDeviceResolver::test_explicit_xpu_raises_when_runtime_missing` | **SKIP** | 0.00s |  |
| `TestTorchDeviceResolver::test_use_gpu_false_returns_cpu` | **PASS** | 0.00s |  |
| `TestXpuRuntime::test_resolve_auto_use_gpu_prefers_xpu_without_cuda` | **PASS** | 0.00s |  |
| `TestXpuRuntime::test_resolve_explicit_xpu` | **PASS** | 0.00s |  |
| `TestXpuRuntime::test_torch_xpu_available` | **PASS** | 0.00s |  |
| `TestXpuRuntime::test_xpu_tensor_add` | **PASS** | 0.05s |  |

## T1 ITN（逆文本正则化）

| 测试 | 结果 | 用时 | 关键指标 |
|---|---|---|---|
| `test_itn::test_itn_chinese_numbers_basic` | **PASS** | 0.01s | `itn_chinese_accuracy=1.0` |
| `test_itn::test_itn_mixed_zh_en_units` | **PASS** | 0.00s | `itn_units_accuracy=1.0` |
| `test_itn::test_itn_idempotent` | **PASS** | 0.00s |  |
| `test_itn::test_itn_empty_input` | **PASS** | 0.00s |  |
| `test_itn::test_itn_skip_words` | **PASS** | 0.00s |  |
| `test_itn::test_itn_disabled_via_config` | **PASS** | 0.00s |  |
| `test_itn::test_itn_batch_consistency` | **PASS** | 0.00s |  |
| `test_itn::test_system_config_default_disabled` | **PASS** | 0.00s |  |
| `test_itn::test_system_result_no_itn_field_when_disabled` | **PASS** | 0.00s |  |
| `test_itn::test_system_with_itn_xpu_smoke` | **PASS** | 0.02s | `itn_xpu_text_len=0` |

## T2 独立降噪前端

| 测试 | 结果 | 用时 | 关键指标 |
|---|---|---|---|
| `test_denoise::test_denoiser_dtype_shape_cpu` | **PASS** | 0.62s |  |
| `test_denoise::test_denoiser_noisereduce_snr_gain` | **PASS** | 0.04s | `denoise_snr_before_db=1.976`, `denoise_snr_after_db=2.108`, `denoise_snr_gain_db=0.132` |
| `test_denoise::test_denoiser_df_backend_optional` | **SKIP** | 0.00s |  |
| `test_denoise::test_system_denoise_off_no_denoiser_attr` | **PASS** | 0.00s |  |
| `test_denoise::test_system_denoise_xpu_wer_drop` | **SKIP** | 12.59s | `denoise_asr_text_len_off=0`, `denoise_asr_text_len_on=0` |

## T3 量化与加速兜底

| 测试 | 结果 | 用时 | 关键指标 |
|---|---|---|---|
| `test_compute_dtype::test_resolve_dtype_xpu_bf16` | **PASS** | 0.00s |  |
| `test_compute_dtype::test_resolve_dtype_cuda_fp16` | **SKIP** | 0.00s |  |
| `test_compute_dtype::test_resolve_dtype_cpu_bf16` | **PASS** | 0.00s |  |
| `test_compute_dtype::test_resolve_dtype_disabled` | **PASS** | 0.00s |  |
| `test_compute_dtype::test_punc_lid_share_helper_import` | **PASS** | 0.00s |  |
| `test_compute_dtype::test_asr_xpu_bf16_param_dtype` | **PASS** | 23.30s |  |
| `test_compute_dtype::test_int8_script_import_only` | **PASS** | 0.00s |  |

## T4 自定义热词偏置

| 测试 | 结果 | 用时 | 关键指标 |
|---|---|---|---|
| `test_hotword::test_trie_insert_and_terminal` | **PASS** | 0.00s |  |
| `test_hotword::test_biaser_delta_shape` | **PASS** | 0.02s |  |
| `test_hotword::test_biaser_advance` | **PASS** | 0.00s |  |
| `test_hotword::test_hotword_weight_zero_no_biaser` | **PASS** | 0.00s |  |

## T5 加速运行时骨架

| 测试 | 结果 | 用时 | 关键指标 |
|---|---|---|---|
| `test_runtime::test_get_llm_runtime_torch` | **PASS** | 0.00s |  |
| `test_runtime::test_get_llm_runtime_unknown` | **PASS** | 0.00s |  |
| `test_runtime::test_trtllm_transcribe_not_implemented` | **PASS** | 0.00s |  |
| `test_runtime::test_vllm_runtime_init_when_cuda` | **SKIP** | 0.00s |  |
| `test_runtime::test_vllm_runtime_raises_without_cuda` | **PASS** | 0.00s |  |

## T6 声纹分离输入与对齐

| 测试 | 结果 | 用时 | 关键指标 |
|---|---|---|---|
| `test_diar_align::test_speaker_by_overlap_ms_basic` | **PASS** | 0.00s |  |
| `test_diar_align::test_word_level_grouping` | **PASS** | 0.00s |  |
| `test_diar_align::test_min_speaker_dur_merge` | **PASS** | 0.00s |  |
| `test_diar_align::test_try_word_diar_requires_single_punc_sentence` | **PASS** | 0.00s |  |
| `test_diar_align::test_build_diar_input_full_short_skips` | **PASS** | 0.00s |  |
| `test_diar_align::test_build_diar_input_full_long` | **PASS** | 0.00s |  |
| `test_diar_align::test_full_pipeline_der_drop` | **SKIP** | 0.00s |  |
| `test_diar_align::test_eval_diarization_script` | **SKIP** | 0.37s |  |

## T7 声纹分离多 backend

| 测试 | 结果 | 用时 | 关键指标 |
|---|---|---|---|
| `test_diar_backends::test_unknown_backend_raises` | **PASS** | 0.00s |  |
| `test_diar_backends::test_speakerlab_raises_not_implemented` | **PASS** | 0.00s |  |
| `test_diar_backends::test_modelscope_backend_short_audio_returns_none` | **PASS** | 0.00s |  |
| `test_diar_backends::test_pyannote_without_token_returns_none` | **PASS** | 0.00s |  |
| `test_diar_backends::test_diar_span_roundtrip` | **PASS** | 0.00s |  |

## T8 声纹注册

| 测试 | 结果 | 用时 | 关键指标 |
|---|---|---|---|
| `test_speaker_enroll::test_content_hash_embedder_stable` | **PASS** | 0.00s |  |
| `test_speaker_enroll::test_registry_match_same_audio` | **PASS** | 0.01s |  |
| `test_speaker_enroll::test_registry_persist_roundtrip` | **PASS** | 0.00s |  |
| `test_speaker_enroll::test_registry_json_format` | **PASS** | 0.00s |  |

## E2E 录音端到端（按功能点）

| 测试 | 结果 | 用时 | 关键指标 |
|---|---|---|---|
| `test_e2e_by_feature::test_e2e_fixture_wavs_exist_and_readable` | **PASS** | 0.00s |  |
| `test_e2e_by_feature::test_e2e_asr_vad_punc_pipeline` | **PASS** | 0.02s | `e2e_asr_sentence_count=0`, `e2e_asr_text_len=0` |
| `test_e2e_by_feature::test_e2e_itn_fields_on_wav` | **PASS** | 8.81s | `e2e_itn_text_len=0` |
| `test_e2e_by_feature::test_e2e_denoise_branch_on_wav` | **PASS** | 9.37s | `e2e_denoise_text_len=0` |
| `test_e2e_by_feature::test_e2e_hotword_config_on_wav` | **PASS** | 8.48s | `e2e_hotword_text_len=0` |
| `test_e2e_by_feature::test_e2e_lid_on_wav` | **PASS** | 15.61s | `e2e_lid_sentence_count=0` |
| `test_e2e_by_feature::test_e2e_diarization_on_dialog_wav` | **PASS** | 10.83s | `e2e_diar_span_count=0` |
| `test_e2e_by_feature::test_e2e_speaker_enroll_on_wav` | **PASS** | 8.67s | `e2e_enroll_sentence_count=0` |
| `test_e2e_by_feature::test_e2e_return_timestamp_path` | **PASS** | 8.35s | `e2e_ts_word_count=0` |

## 失败与跳过原因汇总

- **SKIP** `test_compute_dtype::test_resolve_dtype_cuda_fp16` — CUDA not available
- **SKIP** `test_denoise::test_denoiser_df_backend_optional` — deepfilternet not installed
- **SKIP** `test_denoise::test_system_denoise_xpu_wer_drop` — synthetic audio produced empty ASR; WER comparison N/A
- **SKIP** `test_diar_align::test_full_pipeline_der_drop` — Requires ModelScope diarization + stable offline reference run
- **SKIP** `test_diar_align::test_eval_diarization_script` — pyannote.metrics not installed
- **SKIP** `test_runtime::test_vllm_runtime_init_when_cuda` — CUDA not available
- **SKIP** `TestTorchDeviceResolver::test_explicit_xpu_raises_when_runtime_missing` — XPU runtime present; negative case not applicable

## 附录：复现命令

```powershell
$env:PYTORCH_ENABLE_XPU_FALLBACK = "1"
python scripts/run_full_test_matrix.py --device xpu --report_dir reports/
```
