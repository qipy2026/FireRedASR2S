# Windows 本机：推理加速与量化（FireRedASR2S）

本文说明在 **Windows** 上可用的加速手段；**vLLM / TensorRT-LLM** 仅 FireRedASR2-**LLM** 分支相关，本仓库内转写仍为占位，完整链路见上游与 [`RUNTIMES.md`](RUNTIMES.md)。

---

## 1. 半精度 dtype（推荐：有 GPU / Intel XPU 时）

由 `fireredasr2s/torch_device.py` 的 `resolve_compute_dtype` 决定：

| 设备 | `use_half=1` 时计算 dtype |
|------|---------------------------|
| CUDA | `float16` |
| Intel XPU | `bfloat16` |
| CPU | `bfloat16`（若 PyTorch 支持） |

**命令行示例（Intel XPU + 半精度）**

```powershell
$env:PYTORCH_ENABLE_XPU_FALLBACK = "1"
.\.venv\Scripts\python.exe -m fireredasr2s.fireredasr2s_cli `
  --asr_type aed --asr_device xpu --asr_use_gpu 1 --asr_use_half 1 `
  --wav_path "path\to\16k_mono.wav" --outdir output_speed
```

**NVIDIA CUDA**

```powershell
.\.venv\Scripts\python.exe -m fireredasr2s.fireredasr2s_cli `
  --asr_type aed --asr_device cuda:0 --asr_use_gpu 1 --asr_use_half 1 `
  --wav_path "path\to\16k_mono.wav" --outdir output_speed
```

VAD / LID / Punc 仍由各自 `*_use_gpu` 控制；与 ASR 设备对齐即可。

---

## 2. 动态 INT8（CPU，实验性）

脚本：`scripts/quantize_aed_int8.py`  
使用 PyTorch `quantize_dynamic` 对 **AED 中的 `nn.Linear`** 做 INT8，**不保证精度**。

### 2.1 生成 INT8 权重（建议在「纯 CPU」PyTorch 上跑量化脚本）

```powershell
cd E:\work\aicc\FireRedASR2S
.\.venv\Scripts\python.exe scripts\quantize_aed_int8.py `
  --model_dir pretrained_models\FireRedASR2-AED `
  --out_path output\aed_dynamic_int8_cpu.pt
```

### 2.2 用 INT8 权重推理（**必须 CPU**，且关闭半精度）

新增 CLI：`--aed_dynamic_int8_pt`

```powershell
.\.venv\Scripts\python.exe -m fireredasr2s.fireredasr2s_cli `
  --asr_type aed --asr_use_gpu 0 --asr_use_half 0 `
  --aed_dynamic_int8_pt output\aed_dynamic_int8_cpu.pt `
  --wav_path "path\to\16k_mono.wav" --outdir output_int8
```

说明：

- INT8 路径与 **CUDA/XPU 半精度互斥**：GPU 加速请用上一节 `--asr_use_half 1`，不要用 `--aed_dynamic_int8_pt`。
- 配置约束在 `FireRedAsr2.from_pretrained` 中校验：`use_half` 必须为 False，设备必须为 CPU。

---

## 3. FireRedASR2-LLM：vLLM / TensorRT-LLM

本仓库 **未实现** `VllmLlmRuntime` / `TrtLlmRuntime` 的完整 `transcribe`；在 Windows 上若要 vLLM/TRT，需按上游文档自行部署（通常为 **Linux + CUDA**）。

- 说明：[`RUNTIMES.md`](RUNTIMES.md)  
- 占位脚本：`scripts/run_vllm_llm_asr.sh`（Bash；本机可用 WSL 或自行改写）

在 Windows + **XPU** 上跑 LLM 推理，请使用 **PyTorch 路径**：

```powershell
$env:PYTORCH_ENABLE_XPU_FALLBACK = "1"
.\.venv\Scripts\python.exe -m fireredasr2s.fireredasr2s_cli `
  --asr_type llm --asr_runtime torch --asr_device xpu `
  --wav_path "path\to\16k_mono.wav" --outdir output_llm
```

---

## 4. 一键脚本（可选）

- `scripts/windows/quantize_aed_int8_cpu.ps1`：量化并提示后续 CLI。  
- `scripts/windows/run_cli_aed_half_xpu.ps1`：XPU + 半精度示例（按本机路径改 `wav`）。

---

## 5. 功能矩阵对照

量化 / dtype 在 [`ASR_FEATURE_MATRIX.md`](ASR_FEATURE_MATRIX.md) 中归类为「部分」：INT8 为实验脚本 + 可选加载；TRT/vLLM 见上游。
