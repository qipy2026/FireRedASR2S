<div align="center">

# FireRedASR2S

**工业级一体化语音识别系统**（ASR / VAD / LID / 标点）

[[论文]](https://arxiv.org/pdf/2603.10420) · [[HuggingFace 模型集]](https://huggingface.co/collections/FireRedTeam/fireredasr2s) · [[ModelScope]](https://www.modelscope.cn/collections/xukaituo/FireRedASR2S) · [[Demo]](https://huggingface.co/spaces/FireRedTeam/FireRedASR2S)

</div>

> **本仓库在官方 FireRedASR2S 推理栈之上，重点做了 Windows / Intel XPU 落地、推理加速与工程化扩展。**  
> 已在 **Windows + Intel Arc + torch 2.11.0+xpu** 上完成 **77 项验收测试（70 PASS）**，报告见 [reports/test_report.md](reports/test_report.md)。

---

## ⭐ 本仓库优化亮点（相对上游）

| 方向 | 做了什么 | 入口 |
|------|----------|------|
| **Intel XPU 全栈适配** | 统一设备解析 `CUDA → XPU → CPU`；支持 `torch 2.11+xpu` 与 Legacy IPEX；LLM 分支可走 PyTorch + XPU | [`fireredasr2s/torch_device.py`](fireredasr2s/torch_device.py)、[`scripts/install_intel_xpu_pytorch.ps1`](scripts/install_intel_xpu_pytorch.ps1) |
| **半精度推理加速** | CUDA 用 **fp16**，Intel XPU / CPU 用 **bf16**；CLI `--asr_use_half 1` | [`docs/WINDOWS_INFERENCE_SPEED.md`](docs/WINDOWS_INFERENCE_SPEED.md)、[`scripts/windows/run_cli_aed_half_xpu.ps1`](scripts/windows/run_cli_aed_half_xpu.ps1) |
| **CPU 动态 INT8（实验）** | AED 线性层 INT8 量化 + 推理加载 `--aed_dynamic_int8_pt` | [`scripts/quantize_aed_int8.py`](scripts/quantize_aed_int8.py)、[`scripts/windows/quantize_aed_int8_cpu.ps1`](scripts/windows/quantize_aed_int8_cpu.ps1) |
| **运行时抽象** | LLM 解码后端骨架：`torch`（XPU/CPU/CUDA）/ `vllm` / `trtllm` | [`docs/RUNTIMES.md`](docs/RUNTIMES.md)、[`fireredasr2s/fireredasr2/runtimes/`](fireredasr2s/fireredasr2/runtimes/) |
| **业务向能力扩展** | ITN、热词偏置、降噪前端、多 backend 说话人分离、声纹注册 | [`docs/ASR_FEATURE_MATRIX.md`](docs/ASR_FEATURE_MATRIX.md) |
| **流式 / 全双工编排** | `open_stream()`、`open_full_duplex_stream()` + 麦克风 / TTS 示例 | [`docs/STREAMING_FULL_DUPLEX_CONTRACT.md`](docs/STREAMING_FULL_DUPLEX_CONTRACT.md)、[`examples/`](examples/) |
| **可重复验收体系** | 分任务 pytest 矩阵 + Markdown/HTML 报告自动生成 | [`docs/TEST_REPORT_GUIDE.md`](docs/TEST_REPORT_GUIDE.md)、`scripts/run_full_test_matrix.py` |

**推荐加速路径（Intel XPU · Windows）**

```powershell
$env:PYTORCH_ENABLE_XPU_FALLBACK = "1"
$env:PATH = "$PWD\fireredasr2s\;$env:PATH"
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
.\.venv\Scripts\python.exe .\fireredasr2s\fireredasr2s_cli.py `
  --wav_paths "assets/hello_zh.wav" --outdir output `
  --asr_device xpu --asr_use_half 1 --vad_use_gpu 0 --enable_punc 0
```

或一键脚本：[`scripts/windows/run_cli_aed_half_xpu.ps1`](scripts/windows/run_cli_aed_half_xpu.ps1)

| 设备 | 加速手段 | CLI 要点 |
|------|----------|----------|
| **Intel XPU** | bf16 半精度 | `--asr_device xpu --asr_use_half 1` |
| **NVIDIA CUDA** | fp16 半精度 | `--asr_device cuda:0 --asr_use_half 1` |
| **纯 CPU** | 动态 INT8（实验） | `--asr_use_gpu 0 --aed_dynamic_int8_pt output\aed_dynamic_int8_cpu.pt` |

> **已知限制（优化进行中）**：VAD 仍硬编码 `.cuda()`，XPU 环境请 `--vad_use_gpu 0`；标点模块 XPU dtype 兼容问题，暂 `--enable_punc 0`。详见 [§6 FAQ](#6-能力边界与-faq)。

---

## 1. 产品是什么

FireRedASR2S 是一套**开箱即用的语音转写管线**：输入音频，输出带时间戳的文本，并可按需叠加 VAD 切段、语种识别、标点、热词、说话人分离等能力。

在普通话、方言、英语及代码切换等场景达到 SOTA（见 [论文](https://arxiv.org/pdf/2603.10420)）。**本 fork 的差异化**在于：把官方模型能力落到 **Windows 本机 + Intel 显卡**，并补齐加速、扩展模块与自动化验收，而不只是 Linux/CUDA 实验环境。

| 角色 | 典型诉求 | 本仓库如何满足 |
|------|----------|----------------|
| **算法 / 研发** | 对比 SOTA、调参、二次开发 | 模块化 `fireredasr2s`、`examples_infer/`、pytest 矩阵 |
| **集成工程师** | 把 ASR 接到业务系统 | `FireRedAsr2System.process()`、JSON 结构化输出 |
| **运维 / 部署** | **XPU / CUDA / CPU 选型与加速** | 见上文 **优化亮点** + [WINDOWS_INFERENCE_SPEED](docs/WINDOWS_INFERENCE_SPEED.md) |
| **测试 / 验收** | 可重复评测与报告 | [TEST_REPORT_GUIDE](docs/TEST_REPORT_GUIDE.md)、[reports/test_report.md](reports/test_report.md) |

---

## 2. 核心能力一览

| 能力 | 默认 | 本仓库增强 |
|------|:----:|------------|
| **ASR 转写** | ✅ | AED / LLM；**XPU bf16 / CUDA fp16 / CPU INT8** |
| **VAD / LID / 标点** | ✅ | LID 已适配 XPU；VAD / 标点见已知限制 |
| **字/词级时间戳** | ✅ | TextGrid / SRT；XPU 建议 `PYTORCH_ENABLE_XPU_FALLBACK=1` |
| **ITN 逆文本正则** | 可选 | `--enable_itn`（T1 验收 10/10 PASS） |
| **热词偏置（AED）** | 可选 | `--hotwords`（T4 验收 4/4 PASS） |
| **降噪前端** | 可选 | `--enable_denoise`（T2） |
| **说话人分离 / 声纹** | 可选 | 多 backend + 注册（T6–T8） |
| **流式 / 全双工** | 可选 | `open_stream()` / `open_full_duplex_stream()` |

**管线示意**（离线主路径）：

```mermaid
flowchart LR
  A[音频 WAV] --> B[VAD 切段]
  B --> C[ASR 转写]
  C --> D[LID 语种]
  D --> E[标点 / ITN]
  E --> F[Text / SRT / TextGrid]
```

---

## 3. 典型场景与任务流

| 场景 | 目标 | 入口 |
|------|------|------|
| **XPU 加速验收** | Arc 显卡上跑通 bf16 | → [§4.3 推荐路径](#43-第一次运行推荐intel-xpu--半精度) |
| **CPU 轻量化** | 无 GPU 时 INT8 实验 | [WINDOWS_INFERENCE_SPEED §2](docs/WINDOWS_INFERENCE_SPEED.md) |
| **快速冒烟** | 5 分钟看到转写 | → [§4 五分钟上手](#4-五分钟上手) |
| **单文件 / 批量** | 会议录音、话单 | CLI `--wav_paths` / `--wav_dir` |
| **嵌入业务** | 服务内调用 | Python API（§5.2） |
| **实时 / 全双工** | 边说边出字、TTS 打断 | [`examples/mic_stream_to_asr.py`](examples/mic_stream_to_asr.py) |
| **跑全量验收** | 回归测试 | `scripts/run_full_test_matrix.py` |

---

## 4. 五分钟上手

### 4.1 实施清单（首次必做）

| 步骤 | 动作 | 完成标准 |
|:----:|------|----------|
| ① | Python 3.10+，创建 `.venv` | `python --version` 正常 |
| ② | `pip install -r requirements.txt` | 无报错 |
| ③ | **Intel XPU**：运行 [`scripts/install_intel_xpu_pytorch.ps1`](scripts/install_intel_xpu_pytorch.ps1) | `torch.xpu.is_available()` 为 True |
| ④ | 下载模型到 `pretrained_models/` | 存在 AED / VAD / LID / Punc |
| ⑤ | 设置 `PATH` / `PYTHONPATH` | 能 import `fireredasr2s` |
| ⑥ | 跑通 CLI（推荐 XPU 半精度命令） | `output/` 出现 `.srt` |

**模型下载**：

```bash
pip install -U modelscope
modelscope download --model xukaituo/FireRedASR2-AED --local_dir ./pretrained_models/FireRedASR2-AED
# FireRedVAD / FireRedLID / FireRedPunc 同理
```

**音频**：推荐 **16 kHz、16-bit、单声道 WAV**。

### 4.2 环境安装

```powershell
# Windows · 仓库根目录
.\.venv\Scripts\pip.exe install -r requirements.txt
.\scripts\install_intel_xpu_pytorch.ps1   # Intel Arc / XPU 必做
$env:PATH = "$PWD\fireredasr2s\;$env:PATH"
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
$env:PYTORCH_ENABLE_XPU_FALLBACK = "1"
```

```bash
# Linux / macOS
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PATH="$PWD/fireredasr2s:$PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### 4.3 第一次运行（推荐：Intel XPU + 半精度）

```powershell
$env:PYTORCH_ENABLE_XPU_FALLBACK = "1"
$env:PATH = "$PWD\fireredasr2s\;$env:PATH"
$env:PYTHONPATH = "$PWD;$env:PYTHONPATH"
.\.venv\Scripts\python.exe .\fireredasr2s\fireredasr2s_cli.py `
  --wav_paths "assets/hello_zh.wav" --outdir output `
  --asr_device xpu --asr_use_half 1 --vad_use_gpu 0 --enable_punc 0
```

**预期**：`FINAL` 含「你好世界」，`output/asr_srt/`、`output/asr_tg/` 有产物。

### 4.4 第一次运行（通用 · CUDA / CPU）

```powershell
# Windows · CUDA 半精度示例
.\.venv\Scripts\python.exe .\fireredasr2s\fireredasr2s_cli.py `
  --wav_paths "assets/hello_zh.wav" --outdir output `
  --asr_device cuda:0 --asr_use_half 1
```

```bash
# Linux / macOS
fireredasr2s-cli --wav_paths "assets/hello_zh.wav" --outdir output
```

**纯 CPU**：`--asr_use_gpu 0 --vad_use_gpu 0 --lid_use_gpu 0 --punc_use_gpu 0`；或 INT8 见 [WINDOWS_INFERENCE_SPEED](docs/WINDOWS_INFERENCE_SPEED.md)。

---

## 5. 日常使用

### 5.1 优化相关 CLI

| 需求 | 参数 |
|------|------|
| XPU 半精度 | `--asr_device xpu --asr_use_half 1` |
| CUDA 半精度 | `--asr_device cuda:0 --asr_use_half 1` |
| CPU INT8 AED | `--asr_use_gpu 0 --aed_dynamic_int8_pt <path.pt>` |
| 热词 | `--hotwords "挂失,止付"` |
| ITN | `--enable_itn 1` |
| 降噪 | `--enable_denoise 1` |
| 说话人分离 | `--enable_diarization 1` |

完整参数：`python fireredasr2s/fireredasr2s_cli.py --help`。

### 5.2 Python API

```python
from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig

cfg = FireRedAsr2SystemConfig()
sys = FireRedAsr2System(cfg)
print(sys.process("assets/hello_zh.wav"))
```

### 5.3 验收与回归

```powershell
$env:PYTORCH_ENABLE_XPU_FALLBACK = "1"
.\.venv\Scripts\python.exe scripts\run_full_test_matrix.py
# 报告：reports/test_report.md / test_report.html
```

---

## 6. 能力边界与 FAQ

| 问题 | 说明 |
|------|------|
| 本 fork 相对上游多了什么？ | **XPU 适配、半精度/INT8、ITN/热词/降噪/分离、流式全双工、pytest 验收矩阵**（见文首表格） |
| Intel XPU 标点失败？ | 暂 `--enable_punc 0`；ASR/LID 半精度加速不受影响 |
| VAD 在 XPU 上报 CUDA 错？ | `--vad_use_gpu 0`（VAD 待统一 `torch_device`） |
| 单段时长上限？ | AED ≤60 s；LLM ≤40 s |
| 支持什么音频？ | 16 kHz mono WAV 最佳；`process()` 可重采样 |

---

## 7. 文档索引

| 文档 | 内容 |
|------|------|
| [docs/WINDOWS_INFERENCE_SPEED.md](docs/WINDOWS_INFERENCE_SPEED.md) | **半精度 / INT8 加速（重点）** |
| [docs/RUNTIMES.md](docs/RUNTIMES.md) | LLM 运行时与 XPU |
| [docs/ASR_FEATURE_MATRIX.md](docs/ASR_FEATURE_MATRIX.md) | 功能矩阵与代码入口 |
| [docs/TEST_REPORT_GUIDE.md](docs/TEST_REPORT_GUIDE.md) | 验收报告生成 |
| [docs/FEATURE_E2E_TESTS.md](docs/FEATURE_E2E_TESTS.md) | 录音 E2E |
| [reports/test_report.md](reports/test_report.md) | **最近一次全量验收结果** |

---

## 8. 更新日志（节选）

- **本 fork** Intel XPU 设备层、bf16 半精度、AED INT8 脚本、全量 pytest 矩阵（见 `reports/`）
- [2026.03.12] 上游技术报告 [arXiv:2603.10420](https://arxiv.org/abs/2603.10420)
- [2026.03.05] 上游 vLLM 支持 FireRedASR2-LLM
- [2026.02.13] 上游 FireRedASR2-AED TensorRT-LLM 加速

---

## 9. 致谢与引用

致谢：[Qwen](https://huggingface.co/Qwen)、[WenetSpeech-Yue](https://github.com/ASLP-lab/WenetSpeech-Yue)、[WenetSpeech-Chuan](https://github.com/ASLP-lab/WenetSpeech-Chuan) 等。

```bibtex
@article{xu2026fireredasr2s,
  title={FireRedASR2S: A State-of-the-Art Industrial-Grade All-in-One Automatic Speech Recognition System},
  author={Xu, Kaituo and Jia, Yan and Huang, Kai and Chen, Junjie and Li, Wenpeng and Liu, Kun and Xie, Feng-Long and Tang, Xu and Hu, Yao},
  journal={arXiv preprint arXiv:2603.10420},
  year={2026}
}
```
