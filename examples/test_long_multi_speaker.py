# Generate >60s two-speaker Chinese dialogue (Edge TTS), then run FireRedAsr2System.
# Also saves raw ASR transcribe() results per VAD segment (same shape as test_api_asr.py).
# Run from repo root: PYTHONPATH=. python examples/test_long_multi_speaker.py
# Optional ModelScope diarization: ... python examples/test_long_multi_speaker.py --diarization
# 默认脚本层：ASR 先试 Intel XPU（IPEX），无则 CUDA 全模块；库内 FireRedAsr2：use_gpu=1 且 device 空时为 CUDA→XPU→CPU。
# 强制只用 CPU：python examples/test_long_multi_speaker.py --use_gpu 0
# 显式 Intel：python examples/test_long_multi_speaker.py --xpu
# 显式 NVIDIA：python examples/test_long_multi_speaker.py --use_gpu 1
# 若已安装 Intel torch+xpu（或旧版 IPEX），未传参数时也会自动把 ASR 放到 xpu；要强制 CPU：--no_auto_xpu 或 $env:FIRERED_DISABLE_AUTO_XPU=1
# 环境变量：$env:FIRERED_ASR_DEVICE="xpu"（与 --asr_device 二选一，CLI 优先）
# GPU 测试报告: 默认写入 output/long_multi_speaker/GPU_TEST_REPORT.md；跳过: --no-report；自定义路径: --report PATH
import asyncio
import json
import os
import sys
import traceback
from datetime import datetime, timezone

try:
    import numpy as np
    import soundfile as sf
except ModuleNotFoundError as _e:
    _venv_py = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".venv",
        "Scripts",
        "python.exe",
    )
    print(
        "缺少依赖（例如 soundfile / numpy）。请在本仓库已配置好的虚拟环境中运行：\n"
        f"  {_venv_py} examples\\test_long_multi_speaker.py\n"
        "并先在仓库根目录设置: $env:PYTHONPATH=\"$PWD\"\n"
        "\n"
        "若坚持用当前解释器，请在仓库根目录执行：\n"
        "  pip install soundfile numpy edge-tts\n"
        "并确保已按 README 安装 torch、transformers 等与 requirements.txt 一致的包。\n"
        "注意：Python 3.14 可能与官方 PyTorch 版本不兼容，建议用 Python 3.10–3.12 + 项目 .venv。\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from _e

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _try_intel_xpu_as_default_asr_device() -> str:
    """When ``torch.xpu`` is available (unified +xpu wheel or legacy IPEX), return 'xpu' for ASR."""
    try:
        from fireredasr2s.torch_device import xpu_runtime_available

        return "xpu" if xpu_runtime_available() else ""
    except Exception:
        return ""


def _print_pytorch_gpu_diagnostics() -> None:
    """When ASR falls back to CPU, print why XPU/CUDA were not selected."""
    lines = ["[PyTorch / GPU 诊断 — 用于解释为何仍为 CPU]"]
    try:
        import torch

        from fireredasr2s.torch_device import try_import_ipex

        lines.append(f"  torch.__version__ = {getattr(torch, '__version__', '?')}")
        if try_import_ipex():
            lines.append("  intel_extension_for_pytorch: 已导入")
            xpu = getattr(torch, "xpu", None)
            if xpu is None:
                lines.append(
                    "  torch.xpu: 不存在（当前 PyTorch 多半不是 Intel XPU 构建，仅有 CUDA/CPU 版时常见）"
                )
            else:
                lines.append(f"  torch.xpu.is_available() = {xpu.is_available()}")
        else:
            lines.append(
                "  intel_extension_for_pytorch: 无法导入（未安装、版本与 torch 不匹配，或 WinError 126 等 DLL 依赖缺失）"
            )
        lines.append(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                lines.append(f"  CUDA device[0] = {torch.cuda.get_device_name(0)}")
            except Exception as e:
                lines.append(f"  CUDA device name: (error) {e}")
    except Exception as e:
        lines.append(f"  导入 torch 失败: {e}")
    print("\n".join(lines), file=sys.stderr)


def _torch_env_lines() -> list[str]:
    lines: list[str] = []
    try:
        import torch

        from fireredasr2s.torch_device import try_import_ipex

        lines.append(f"- `torch.__version__`: `{torch.__version__}`")
        lines.append(f"- `torch.cuda.is_available()`: `{torch.cuda.is_available()}`")
        if torch.cuda.is_available():
            try:
                lines.append(f"- `cuda:0` name: `{torch.cuda.get_device_name(0)}`")
            except Exception as e:
                lines.append(f"- CUDA device name: (error) `{e}`")
        if try_import_ipex():
            lines.append("- `intel_extension_for_pytorch`: imported")
            xpu = getattr(torch, "xpu", None)
            if xpu is None:
                lines.append("- `torch.xpu`: *missing*")
            else:
                lines.append(f"- `torch.xpu.is_available()`: `{xpu.is_available()}`")
        else:
            lines.append(
                "- `intel_extension_for_pytorch`: not loaded (missing, version mismatch, or DLL error — see README IPEX troubleshooting)"
            )
    except Exception as e:
        lines.append(f"- (torch import failed) `{e}`")
    return lines


def _infer_asr_backend_label(use_gpu: bool, asr_device: str) -> str:
    """Label ASR compute device (matches ``resolve_fire_red_asr_torch_device``)."""
    try:
        from fireredasr2s.torch_device import resolve_fire_red_asr_torch_device

        dev = resolve_fire_red_asr_torch_device(
            device_str=(asr_device or "").strip(),
            use_gpu=use_gpu,
        )
    except Exception as exc:
        return f"ASR device error ({type(exc).__name__})"
    if dev.type == "xpu":
        return "Intel XPU (ASR)"
    if dev.type == "cuda":
        ds = (asr_device or "").strip()
        return f"CUDA ASR ({ds})" if ds else "CUDA ASR (auto)"
    return "CPU (ASR)"


def write_gpu_test_report(
    path: str,
    *,
    argv: list[str],
    use_gpu: bool,
    asr_device: str,
    enable_diarization: bool,
    wav_path: str,
    wav_duration_s: float,
    wav_sr: int,
    asr_segment_count: int,
    system_result: dict,
) -> None:
    """Write a Markdown summary for GPU/CPU test runs."""
    text = system_result.get("text") or ""
    preview = text[:800] + ("…" if len(text) > 800 else "")
    sentences = system_result.get("sentences") or []
    diar = system_result.get("diarization_spans") or []

    lines = [
        "# FireRedASR2S — `test_long_multi_speaker.py` 运行报告",
        "",
        f"- **生成时间 (UTC)**：`{datetime.now(timezone.utc).isoformat()}`",
        f"- **命令行**：`{' '.join(argv)}`",
        f"- **配置**：`use_gpu={use_gpu}`，`asr_device={repr(asr_device)}`，`enable_diarization={enable_diarization}`",
        f"- **推断 ASR 后端**：{_infer_asr_backend_label(use_gpu, asr_device)}",
        "",
        "## 环境 (PyTorch)",
        "",
        *_torch_env_lines(),
        "",
        "## 输入音频",
        "",
        f"- **路径**：`{wav_path}`",
        f"- **时长**：{wav_duration_s:.2f} s",
        f"- **采样率**：{wav_sr} Hz",
        "",
        "## 输出文件",
        "",
        f"- **分段 transcribe JSON**：`{OUT_ASR_TRANSCRIBE_JSON}`（共 {asr_segment_count} 条 VAD 段结果）",
        f"- **系统 pipeline JSON**：`{OUT_SYSTEM_JSON}`",
        "",
        "## 系统结果摘要",
        "",
        f"- **合并句数**：{len(sentences)}",
        f"- **diarization_spans**：{len(diar)} 条（无 diarization 时为 0）",
        f"- **合并文本预览（前 800 字）**：",
        "",
        "```",
        preview,
        "```",
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote report -> {path}")


def write_failed_gpu_test_report(
    path: str,
    *,
    argv: list[str],
    use_gpu: bool,
    asr_device: str,
    exc: BaseException,
) -> None:
    """Write Markdown when the run aborts (e.g. CUDA requested but torch is CPU-only)."""
    tb = traceback.format_exc()
    try:
        env_lines = _torch_env_lines()
    except Exception as e2:
        env_lines = [f"- (environment section failed) `{e2}`"]
    lines = [
        "# FireRedASR2S — `test_long_multi_speaker.py` 运行报告（失败）",
        "",
        f"- **生成时间 (UTC)**：`{datetime.now(timezone.utc).isoformat()}`",
        f"- **命令行**：`{' '.join(argv)}`",
        f"- **配置**：`use_gpu={use_gpu}`，`asr_device={repr(asr_device)}`",
        f"- **推断 ASR 后端（计划）**：{_infer_asr_backend_label(use_gpu, asr_device)}",
        "",
        "## 环境 (PyTorch)",
        "",
        *env_lines,
        "",
        "## 错误",
        "",
        f"```\n{type(exc).__name__}: {exc}\n```",
        "",
        "## Traceback",
        "",
        "```",
        tb.rstrip(),
        "```",
        "",
        "## 说明",
        "",
        "- 若使用 `--use_gpu 1` 但 `torch.cuda.is_available()` 为 false，或出现 **Torch not compiled with CUDA**，请安装 **CUDA 版 PyTorch**（见仓库 `requirements.txt`），或改用 **`--xpu` / 自动 XPU**（Intel）。",
        "- 若仅需生成成功报告，请在能跑通推理的环境中加上 `--report` 再执行。",
        "",
    ]
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Wrote failure report -> {path}", file=sys.stderr)
    except Exception as e2:
        print(f"Could not write failure report to {path}: {e2}", file=sys.stderr)


TARGET_SR = 16000
SILENCE_S = 0.35
OUT_WAV = os.path.join(ROOT, "assets", "long_multi_speaker_65s.wav")
OUT_DIR = os.path.join(ROOT, "output", "long_multi_speaker")
OUT_ASR_TRANSCRIBE_JSON = os.path.join(OUT_DIR, "asr_transcribe_results.json")
OUT_SYSTEM_JSON = os.path.join(OUT_DIR, "asr_system_result.json")

# Alternating two voices (multi-speaker)
VOICE_F = "zh-CN-XiaoxiaoNeural"
VOICE_M = "zh-CN-YunyangNeural"

DIALOGUE = [
    (VOICE_F, "各位同事下午好，我们开会讨论本季度产品发布计划。"),
    (VOICE_M, "收到，我这边研发排期已经整理好了，大概需要五个迭代。"),
    (VOICE_F, "五个迭代是指十周吗？市场活动希望能在六月底前上线。"),
    (VOICE_M, "如果测试资源充足，六月底可以，但要把接口联调提前两周。"),
    (VOICE_F, "运营同事也在，你们对活动页面有什么硬性要求？"),
    (VOICE_M, "我们需要支持多语言切换，以及夜间模式的默认开启。"),
    (VOICE_F, "多语言可以接翻译服务，夜间模式交给前端主题配置就行。"),
    (VOICE_M, "另外埋点方案要和数据团队对齐，避免重复上报。"),
    (VOICE_F, "好的，数据组下周三之前给出字段规范，大家按规范接入。"),
    (VOICE_M, "客户端还要兼容上一版本接口，否则老用户会闪退。"),
    (VOICE_F, "兼容性我们做灰度发布，先开放百分之五用户观察一天。"),
    (VOICE_M, "风险点在于支付通道，如果限额调整需要财务提前通知。"),
    (VOICE_F, "财务那边我来协调，今天下班前确认限额变更窗口。"),
    (VOICE_M, "客服话术也要更新，特别是退款流程有三处文案改动。"),
    (VOICE_F, "客服培训安排在发布前两天，材料由产品组统一发放。"),
    (VOICE_M, "测试环境的数据量不够，建议导入一百万条模拟订单。"),
    (VOICE_F, "一百万条我来申请，预计两天内同步到测试库。"),
    (VOICE_M, "性能压测目标是一千并发，响应时间控制在两百毫秒内。"),
    (VOICE_F, "压测报告需要包含错误率和百分之九十五分位延迟。"),
    (VOICE_M, "如果压测不过，我们优先降级非核心功能，保证主链路。"),
    (VOICE_F, "没问题，发布当天安排双人值班，重大问题立刻回滚。"),
    (VOICE_M, "我补充一下，监控告警阈值要比平时收紧百分之二十。"),
    (VOICE_F, "同意，那今天先到这里，大家按分工推进，散会。"),
]


def resample_to_16k(mono: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == TARGET_SR:
        return mono.astype(np.float32, copy=False)
    t_old = np.arange(len(mono)) / float(src_sr)
    t_new = np.arange(0, len(mono) / float(src_sr), 1.0 / TARGET_SR)
    return np.interp(t_new, t_old, mono.astype(np.float32)).astype(np.float32)


async def synth_to_float32(voice: str, text: str) -> tuple[np.ndarray, int]:
    import edge_tts
    import io

    buf = io.BytesIO()
    com = edge_tts.Communicate(text, voice)
    async for chunk in com.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    buf.seek(0)
    data, sr = sf.read(buf, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, int(sr)


async def build_wav() -> str:
    chunks = []
    for voice, text in DIALOGUE:
        audio, sr = await synth_to_float32(voice, text)
        audio = resample_to_16k(audio, sr)
        chunks.append(audio)
        chunks.append(np.zeros(int(SILENCE_S * TARGET_SR), dtype=np.float32))
    y = np.concatenate(chunks)
    y = (y * 0.95 * 32767.0).clip(-32768, 32767).astype(np.int16)
    os.makedirs(os.path.dirname(OUT_WAV), exist_ok=True)
    sf.write(OUT_WAV, y, TARGET_SR, subtype="PCM_16")
    return OUT_WAV


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, ensure_ascii=False, indent=2)


def transcribe_segments_like_test_api(
    wav_path: str,
    uttid_prefix: str = "long_meeting",
    *,
    use_gpu: bool = False,
    asr_device: str = "",
):
    """VAD 切段后对每段调用 model.transcribe(batch_uttid, batch_wav_path)，合并为 results 列表并返回。"""
    from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
    from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig

    wav_np, sr = sf.read(wav_path, dtype="int16")
    assert sr == TARGET_SR, f"expected {TARGET_SR} Hz, got {sr}"

    vad_config = FireRedVadConfig(
        use_gpu=use_gpu,
        smooth_window_size=5,
        speech_threshold=0.4,
        min_speech_frame=20,
        max_speech_frame=2000,
        min_silence_frame=20,
        merge_silence_frame=0,
        extend_speech_frame=0,
        chunk_max_frame=30000,
    )
    vad = FireRedVad.from_pretrained("pretrained_models/FireRedVAD/VAD", vad_config)
    vad_result, _ = vad.detect(wav_path)
    segments = vad_result["timestamps"]

    asr_config = FireRedAsr2Config(
        use_gpu=use_gpu,
        use_half=False,
        beam_size=3,
        nbest=1,
        decode_max_len=0,
        softmax_smoothing=1.25,
        aed_length_penalty=0.6,
        eos_penalty=1.0,
        return_timestamp=True,
        device=(asr_device or "").strip(),
    )
    model = FireRedAsr2.from_pretrained(
        "aed", "pretrained_models/FireRedASR2-AED", asr_config
    )

    all_results = []
    for start_s, end_s in segments:
        seg = wav_np[int(start_s * sr) : int(end_s * sr)]
        if seg.size == 0:
            continue
        seg_uttid = f"{uttid_prefix}_s{int(start_s * 1000)}_e{int(end_s * 1000)}"
        batch_uttid = [seg_uttid]
        batch_wav = [(sr, seg)]
        results = model.transcribe(batch_uttid, batch_wav)
        all_results.extend(results)

    return all_results


def run_asr_system(
    wav_path: str,
    *,
    enable_diarization: bool = False,
    use_gpu: bool = False,
    asr_device: str = "",
):
    from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig
    from fireredasr2s.fireredasr2 import FireRedAsr2Config
    from fireredasr2s.fireredlid import FireRedLidConfig
    from fireredasr2s.fireredpunc import FireRedPuncConfig
    from fireredasr2s.fireredvad import FireRedVadConfig

    vad_config = FireRedVadConfig(
        use_gpu=use_gpu,
        smooth_window_size=5,
        speech_threshold=0.4,
        min_speech_frame=20,
        max_speech_frame=2000,
        min_silence_frame=20,
        merge_silence_frame=0,
        extend_speech_frame=0,
        chunk_max_frame=30000,
    )
    lid_config = FireRedLidConfig(use_gpu=use_gpu, use_half=False)
    asr_config = FireRedAsr2Config(
        use_gpu=use_gpu,
        use_half=False,
        beam_size=3,
        nbest=1,
        decode_max_len=0,
        softmax_smoothing=1.25,
        aed_length_penalty=0.6,
        eos_penalty=1.0,
        return_timestamp=True,
        device=(asr_device or "").strip(),
    )
    punc_config = FireRedPuncConfig(use_gpu=use_gpu)

    cfg = FireRedAsr2SystemConfig(
        "pretrained_models/FireRedVAD/VAD",
        "pretrained_models/FireRedLID",
        "aed",
        "pretrained_models/FireRedASR2-AED",
        "pretrained_models/FireRedPunc",
        vad_config,
        lid_config,
        asr_config,
        punc_config,
        enable_vad=1,
        enable_lid=1,
        enable_punc=1,
        enable_diarization=enable_diarization,
    )
    system = FireRedAsr2System(cfg)
    result = system.process(wav_path, uttid="long_meeting")
    return result


async def main(
    enable_diarization: bool = False,
    *,
    use_gpu: bool = False,
    asr_device: str = "",
    report_path: str | None = None,
):
    path = OUT_WAV
    if os.path.isfile(path):
        print(f"Using existing wav: {path}")
    else:
        print("Synthesizing long multi-speaker wav (Edge TTS)...")
        path = await build_wav()
    info = sf.info(path)
    dur = info.duration
    print(f"Wrote {path}, duration={dur:.2f}s, sr={info.samplerate}")
    if dur < 60:
        print("WARNING: duration under 60s, add more dialogue lines.")

    from fireredasr2s.torch_device import resolve_fire_red_asr_torch_device

    dev_note = (asr_device or "").strip()
    try:
        asr_resolved = resolve_fire_red_asr_torch_device(
            device_str=dev_note,
            use_gpu=use_gpu,
        )
    except Exception as e:
        print(f"Inference: ASR device resolution failed: {e}", file=sys.stderr)
        asr_resolved = None

    if asr_resolved is not None:
        if asr_resolved.type == "xpu":
            print(
                "Inference: ASR on Intel XPU (unified torch+xpu or IPEX; see README). "
                f"VAD/LID/Punc use_gpu={use_gpu} (推荐 0 / CPU)."
            )
        elif asr_resolved.type == "cuda":
            print(
                f"Inference: ASR on CUDA ({asr_resolved!s}). "
                f"VAD/LID/Punc use_gpu={use_gpu}."
            )
        else:
            if dev_note:
                print(
                    f"Inference: ASR on CPU; explicit device string was {dev_note!r} but resolved CPU "
                    f"(check torch / IPEX)."
                )
            elif use_gpu:
                print(
                    "Inference: ASR on CPU (use_gpu=True 但本机无可用 CUDA/XPU；"
                    "空 device 时顺序为 CUDA → XPU → CPU)。"
                )
            else:
                print(
                    "Inference: ASR on CPU (use_gpu=False)。"
                    "需要 GPU 时不要传 --use_gpu 0，并安装 CUDA 版 torch 或 IPEX+XPU。"
                )
    elif dev_note:
        print(
            f"Inference: VAD/LID/ASR/Punc use_gpu={use_gpu}; "
            f"ASR device={dev_note!r} (resolution failed — see stderr above)."
        )
    else:
        print(
            "Inference: ASR device unknown (resolution failed). "
            "空 --asr_device 且 use_gpu=1 时库内顺序: CUDA → XPU → CPU。"
        )

    print("Running FireRedAsr2.transcribe per VAD segment (save JSON)...")
    asr_results = transcribe_segments_like_test_api(
        path,
        uttid_prefix="long_meeting",
        use_gpu=use_gpu,
        asr_device=asr_device,
    )
    save_json(OUT_ASR_TRANSCRIBE_JSON, asr_results)
    print(f"Saved {len(asr_results)} ASR segments -> {OUT_ASR_TRANSCRIBE_JSON}")
    if asr_results:
        print("First transcribe item:", asr_results[0])

    print(
        "Running FireRedAsr2System (VAD segments -> ASR per segment)"
        + (" + ModelScope diarization" if enable_diarization else "")
        + "..."
    )
    result = run_asr_system(
        path,
        enable_diarization=enable_diarization,
        use_gpu=use_gpu,
        asr_device=asr_device,
    )
    save_json(OUT_SYSTEM_JSON, result)
    print(f"Saved system output -> {OUT_SYSTEM_JSON}")

    print("--- merged text ---")
    print(result.get("text", ""))
    print("--- num sentences ---", len(result.get("sentences", [])))
    for i, s in enumerate(result.get("sentences", [])[:8]):
        print(f"  [{i}] {s}")
    if len(result.get("sentences", [])) > 8:
        print(f"  ... and {len(result['sentences']) - 8} more")

    if report_path:
        write_gpu_test_report(
            report_path,
            argv=list(sys.argv),
            use_gpu=use_gpu,
            asr_device=asr_device,
            enable_diarization=enable_diarization,
            wav_path=path,
            wav_duration_s=float(dur),
            wav_sr=int(info.samplerate),
            asr_segment_count=len(asr_results),
            system_result=result,
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Long multi-speaker demo + FireRedAsr2System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "默认: 自动 XPU（Intel torch+xpu / IPEX）→ 否则 CUDA → 否则 CPU；与显式 --xpu 时 ASR 上 XPU 一致（无需再写 --xpu）。\n"
            "默认写入 GPU 测试报告（见 --report）；不需要报告时加 --no-report。\n"
            "强制 CPU:  python examples/test_long_multi_speaker.py --use_gpu 0\n"
            "仅 Intel:  python examples/test_long_multi_speaker.py --xpu\n"
            "仅 CUDA:   python examples/test_long_multi_speaker.py --use_gpu 1"
        ),
    )
    ap.add_argument(
        "--diarization",
        action="store_true",
        help="Enable ModelScope speaker diarization (requires: pip install modelscope; large first-time download).",
    )
    ap.add_argument(
        "--use_gpu",
        type=int,
        default=None,
        choices=[0, 1],
        metavar="0|1",
        help="Omit: auto (XPU then CUDA). 1: all modules on CUDA. 0: CPU-only for VAD/LID/ASR/Punc.",
    )
    ap.add_argument(
        "--asr_device",
        type=str,
        default="",
        help='Optional ASR-only torch device, e.g. "cuda:0" or "xpu" (non-empty overrides ASR placement; Intel: unified torch+xpu or IPEX).',
    )
    ap.add_argument(
        "--xpu",
        action="store_true",
        help="Intel GPU: force use_gpu=0 for VAD/LID/Punc and run ASR on device xpu (intel_extension_for_pytorch; see README).",
    )
    ap.add_argument(
        "--no_auto_xpu",
        action="store_true",
        help="When Intel XPU is available, do not auto-select it (keep ASR on CPU unless --xpu / --asr_device).",
    )
    _report_default = os.path.join(OUT_DIR, "GPU_TEST_REPORT.md")
    ap.add_argument(
        "--no-report",
        action="store_true",
        help=f"Do not write the Markdown GPU test report (default is to write {_report_default}).",
    )
    ap.add_argument(
        "--report",
        nargs="?",
        const=_report_default,
        default=None,
        metavar="PATH",
        help=(
            "Override report path (default without this flag is still the standard report file). "
            f"If the flag is given without PATH, writes to {_report_default}."
        ),
    )
    ns = ap.parse_args()
    explicit_use_gpu = ns.use_gpu is not None

    asr_dev = (ns.asr_device or "").strip()
    if not asr_dev:
        asr_dev = (os.environ.get("FIRERED_ASR_DEVICE") or "").strip()

    use_gpu = False

    if ns.xpu:
        use_gpu = False
        if not asr_dev:
            asr_dev = "xpu"
    elif explicit_use_gpu:
        use_gpu = bool(ns.use_gpu)
    else:
        # Default: prefer GPU — XPU (ASR) first, else CUDA (all modules).
        if (
            not asr_dev
            and not ns.no_auto_xpu
            and os.environ.get("FIRERED_DISABLE_AUTO_XPU", "").lower() not in ("1", "true", "yes")
        ):
            asr_dev = _try_intel_xpu_as_default_asr_device()
            if asr_dev == "xpu":
                print(
                    "Auto-selected ASR device=xpu (Intel). "
                    "--no_auto_xpu or FIRERED_DISABLE_AUTO_XPU=1 skips this step."
                )
        if not asr_dev:
            try:
                import torch

                if torch.cuda.is_available():
                    use_gpu = True
                    print(
                        "Auto-selected CUDA for VAD/LID/ASR/Punc (NVIDIA). "
                        "--use_gpu 0 forces CPU-only."
                    )
            except Exception:
                pass

    if not use_gpu and not asr_dev and not explicit_use_gpu:
        _print_pytorch_gpu_diagnostics()
        print(
            "\n"
            "======================================================================\n"
            "  结论: 脚本已尝试「默认走 GPU」，但当前环境无可用 XPU/CUDA，故 ASR 仍为 CPU。\n"
            "  这不是脚本报错，而是本机 PyTorch 与驱动组合未暴露 GPU 给 torch。\n"
            "\n"
            "  Intel Arc: 不要用 requirements.txt 里的 +cu118 轮子。在仓库根目录执行:\n"
            "    .\\scripts\\install_intel_xpu_pytorch.ps1\n"
            "    pip install -r requirements-asr-no-torch.txt\n"
            "  若 pip 失败，按 README「XPU / IPEX」里 Intel 官方安装页逐条安装。\n"
            "  NVIDIA: 安装带 CUDA 的 PyTorch 且驱动正常时，无参运行会自动选 CUDA。\n"
            "\n"
            "  强制 CPU:  python examples/test_long_multi_speaker.py --use_gpu 0\n"
            "======================================================================\n",
            file=sys.stderr,
        )

    if ns.no_report:
        report_path = None
    elif ns.report is not None:
        report_path = ns.report
    else:
        report_path = _report_default

    try:
        asyncio.run(
            main(
                enable_diarization=ns.diarization,
                use_gpu=use_gpu,
                asr_device=asr_dev,
                report_path=report_path,
            )
        )
    except BaseException as e:
        if report_path:
            write_failed_gpu_test_report(
                report_path,
                argv=list(sys.argv),
                use_gpu=use_gpu,
                asr_device=asr_dev,
                exc=e,
            )
        raise
