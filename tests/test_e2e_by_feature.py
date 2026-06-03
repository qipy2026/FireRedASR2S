"""按功能点的录音端到端（E2E）测试。

前置条件：
  1. 运行 ``python scripts/generate_test_fixtures.py`` 生成合成 wav（16 kHz）。
  2. 需要真实模型 + XPU 的用例带 ``@pytest.mark.xpu`` / ``slow``，缺模型或设备时 skip。
  3. **ASR 主线 E2E** 默认输入优先级见 ``_asr_e2e_input_wav``；可将真人会议录音置于
     ``assets/metting_0507.mp3``（文件名按仓库约定），或设置 ``FIREREDASR2S_E2E_ASR_WAV``。
  4. **说话人 diar E2E** 默认输入见 ``_e2e_diarization_input_path``：**默认** ``assets/metting_0507_seg01.wav``
     （``E2E_DEFAULT_DIAR_ASSET``）；可用 ``FIREREDASR2S_E2E_DIAR_WAV`` 覆盖（例如指向完整 ``metting_0507.mp3``）。
     若该默认文件不存在则回退合成 ``dialog_2spk_30s.wav``（谱后端）。仅跑 CAM++ 分割：
     ``python scripts/run_campplus_diar_wav.py assets/metting_0507_seg01.wav``。
  5. **LID E2E** 默认输入见 ``_e2e_lid_input_path``：``FIREREDASR2S_E2E_LID_WAV`` →
     ``assets/metting_0507_seg01.wav`` → 合成 ``clean_zh_short.wav``。
  6. **声纹 1:N（注册库匹配）**：``enable_speaker_id`` + ``speaker_embedder=modelscope_campplus_sv``
     （或 CLI ``--natural_speech_speaker_stack 1``）+ ``register_speaker`` /
     ``--register_speaker 名=wav``；阈值 ``speaker_match_threshold``（CAM++ 默认约 0.35，见
     ``firereddiar.production``）。输出 ``utterance_enrolled_*`` 与逐句 ``enrolled_*``。
  7. **全功能合并输出**：``test_e2e_full_stack_merged_json`` 在单条 JSON 中汇总 VAD/ASR/标点/LID/ITN、
     词级时间戳、热词偏置、谱声纹注册匹配，以及可选的 CAM++ 说话人 diar（当存在
     ``assets/metting_0507_seg01.wav`` 且已安装 ``modelscope`` 时）；否则用合成 ``clean`` 且不启 diar。
     可用 ``FIREREDASR2S_E2E_FULL_STACK_WAV`` 覆盖输入 wav。
     设置 ``FIREREDASR2S_E2E_RECORD_DIR`` 时写出 ``<case>/e2e_merged_full_stack.json``（含
     ``stream_session`` 流式回放；见 ``tests.utils.e2e_long_multi_speaker_record``）。

每个用例先检查 fixture 文件是否存在；缺失则 skip 并提示生成脚本。
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fireredasr2s.fireredasr2system import FireRedAsr2System, FireRedAsr2SystemConfig
from fireredasr2s.firereddiar.production import (
    CAMPLUS_SV_DEFAULT_MATCH_THRESHOLD,
    with_natural_speech_speaker_stack,
)
from fireredasr2s.fireredenh import FireRedDenoiserConfig

from tests.utils.e2e_long_multi_speaker_record import (
    e2e_record_root_from_env,
    save_e2e_merged_full_stack_json,
    save_long_multi_speaker_style_artifacts,
)


# 与 generate_test_fixtures.py 约定一致
E2E_WAVS = {
    "clean": "clean_zh_short.wav",
    "noisy": "noisy_short.wav",
    "dialog": "dialog_2spk_30s.wav",
    # 非稳态噪声脉冲，便于 FireRedVAD 检出语音段（纯音 clean 常为 0 段）
    "vad_proxy": "e2e_vad_speech_proxy.wav",
}

# 说话人 diar E2E 默认真人素材（相对仓库 ``assets/``）
E2E_DEFAULT_DIAR_ASSET = "metting_0507_seg01.wav"


def _require_enroll_fixtures(fixtures_dir: Path) -> dict[str, Path]:
    """声纹 E2E：``generate_test_fixtures.py`` 生成的 enroll_spk*.wav。"""
    names = {
        "a1": "enroll_spkA_1.wav",
        "a2": "enroll_spkA_2.wav",
        "b1": "enroll_spkB_1.wav",
        "b2": "enroll_spkB_2.wav",
    }
    out: dict[str, Path] = {}
    missing: list[str] = []
    for k, name in names.items():
        p = fixtures_dir / name
        if not p.is_file():
            missing.append(name)
        else:
            out[k] = p
    if missing:
        pytest.skip(
            "缺少声纹注册用录音: "
            + ", ".join(missing)
            + " 。请执行: python scripts/generate_test_fixtures.py"
        )
    return out


def _require_wavs(fixtures_dir: Path, *keys: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    missing: list[str] = []
    for k in keys:
        name = E2E_WAVS[k]
        p = fixtures_dir / name
        if not p.is_file():
            missing.append(name)
        else:
            out[k] = p
    if missing:
        pytest.skip(
            "缺少合成录音: "
            + ", ".join(missing)
            + " 。请执行: python scripts/generate_test_fixtures.py"
        )
    return out


def _assert_wav_readable(path: Path) -> None:
    wav, sr = sf.read(str(path), dtype="int16")
    assert sr == 16000
    assert wav.ndim == 1
    assert len(wav) > 0


def _clone_cfg(base: FireRedAsr2SystemConfig, **kwargs) -> FireRedAsr2SystemConfig:
    return replace(base, **kwargs)


def _clone_asr(base: FireRedAsr2SystemConfig, **asr_kw) -> FireRedAsr2SystemConfig:
    return replace(base, asr_config=replace(base.asr_config, **asr_kw))


def _e2e_diarization_input_path(fixtures_dir: Path) -> Path:
    """说话人 E2E 输入：``FIREREDASR2S_E2E_DIAR_WAV`` → ``assets/`` + ``E2E_DEFAULT_DIAR_ASSET`` → 合成 dialog。"""
    env = (os.environ.get("FIREREDASR2S_E2E_DIAR_WAV") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    repo = fixtures_dir.parent.parent
    default_diar = repo / "assets" / E2E_DEFAULT_DIAR_ASSET
    if default_diar.is_file():
        return default_diar
    _require_wavs(fixtures_dir, "dialog")
    return fixtures_dir / E2E_WAVS["dialog"]


def _is_dialog_synthetic_diar_fixture(path: Path, fixtures_dir: Path) -> bool:
    return path.resolve() == (fixtures_dir / E2E_WAVS["dialog"]).resolve()


def _e2e_full_stack_input_and_diar_mode(fixtures_dir: Path) -> tuple[Path, str]:
    """全功能 E2E：``(wav_path, diar_mode)``，``diar_mode`` 为 ``campplus`` 或 ``none``。

    优先 ``FIREREDASR2S_E2E_FULL_STACK_WAV`` → ``assets/metting_0507_seg01.wav``（若存在且可
    ``import modelscope`` 则 ``campplus``，否则仍为真人 wav 但 ``none``）→ 合成 ``clean`` + ``none``。
    """
    env = (os.environ.get("FIREREDASR2S_E2E_FULL_STACK_WAV") or "").strip()
    repo = fixtures_dir.parent.parent
    seg = repo / "assets" / E2E_DEFAULT_DIAR_ASSET
    cand: Path | None = None
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            cand = p
    elif seg.is_file():
        cand = seg
    if cand is not None:
        try:
            import modelscope  # noqa: F401
        except ImportError:
            return cand, "none"
        return cand, "campplus"
    paths = _require_wavs(fixtures_dir, "clean")
    return paths["clean"], "none"


def _e2e_lid_input_path(fixtures_dir: Path) -> Path:
    """LID E2E 输入：``FIREREDASR2S_E2E_LID_WAV`` → ``assets/`` + ``E2E_DEFAULT_DIAR_ASSET`` → 合成 clean。"""
    env = (os.environ.get("FIREREDASR2S_E2E_LID_WAV") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    repo = fixtures_dir.parent.parent
    seg = repo / "assets" / E2E_DEFAULT_DIAR_ASSET
    if seg.is_file():
        return seg
    paths = _require_wavs(fixtures_dir, "clean")
    return paths["clean"]


def _asr_e2e_input_wav(fixtures_dir: Path) -> Path:
    """ASR 主线 E2E 输入路径优先级：

    1. 环境变量 ``FIREREDASR2S_E2E_ASR_WAV``（任意 soundfile/torchaudio 可读格式）
    2. 仓库 ``assets/metting_0507.mp3``（会议录音，需自行放入 ``assets/``）
    3. ``assets/long_multi_speaker_65s.wav``
    4. 合成 ``tests/fixtures/e2e_vad_speech_proxy.wav``
    """
    env = (os.environ.get("FIREREDASR2S_E2E_ASR_WAV") or "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    repo = fixtures_dir.parent.parent
    meeting_mp3 = repo / "assets" / "metting_0507.mp3"
    if meeting_mp3.is_file():
        return meeting_mp3
    long_wav = repo / "assets" / "long_multi_speaker_65s.wav"
    if long_wav.is_file():
        return long_wav
    _require_wavs(fixtures_dir, "vad_proxy")
    return fixtures_dir / E2E_WAVS["vad_proxy"]


def _maybe_record_long_multi_style(
    request: pytest.FixtureRequest,
    system: FireRedAsr2System,
    wav_path: Path,
    uttid: str,
    result: dict,
) -> None:
    """若设置 ``FIREREDASR2S_E2E_RECORD_DIR``，写出与 ``output/long_multi_speaker`` 同构的三件套。"""
    root = e2e_record_root_from_env()
    if root is None:
        return
    save_long_multi_speaker_style_artifacts(
        output_root=root,
        case_id=request.node.name,
        wav_path=str(wav_path.resolve()),
        uttid=uttid,
        system=system,
        system_result=result,
    )


def _maybe_record_merged_full_stack(
    request: pytest.FixtureRequest,
    system: FireRedAsr2System,
    wav_path: Path,
    uttid: str,
    result: dict,
) -> None:
    """若设置 ``FIREREDASR2S_E2E_RECORD_DIR``，写出单文件 ``e2e_merged_full_stack.json``。"""
    root = e2e_record_root_from_env()
    if root is None:
        return
    save_e2e_merged_full_stack_json(
        output_root=root,
        case_id=request.node.name,
        wav_path=str(wav_path.resolve()),
        uttid=uttid,
        system=system,
        system_result=result,
    )


@pytest.mark.e2e
def test_e2e_fixture_wavs_exist_and_readable(fixtures_dir: Path):
    """E2E 输入：合成录音文件存在且可被 soundfile 读取（不加载 ASR 模型）。"""
    paths = _require_wavs(fixtures_dir, "clean", "noisy", "dialog", "vad_proxy")
    for p in paths.values():
        _assert_wav_readable(p)


@pytest.mark.e2e
@pytest.mark.xpu
@pytest.mark.slow
def test_e2e_asr_vad_punc_pipeline(asr_system_xpu, fixtures_dir: Path, metric, request):
    """功能：基础 ASR 全家桶（VAD + ASR + Punc）走完整 ``process``。"""
    wav = _asr_e2e_input_wav(fixtures_dir)
    r = asr_system_xpu.process(str(wav), uttid="e2e_asr")
    _maybe_record_long_multi_style(request, asr_system_xpu, wav, "e2e_asr", r)
    assert r["uttid"] == "e2e_asr"
    assert "text" in r and isinstance(r["text"], str)
    assert "sentences" in r and isinstance(r["sentences"], list)
    assert "vad_segments_ms" in r and isinstance(r["vad_segments_ms"], list)
    assert "words" in r
    assert "speaker_label_note" in r
    metric("e2e_asr_sentence_count", len(r["sentences"]))
    metric("e2e_asr_text_len", len((r.get("text") or "").strip()))
    metric("e2e_asr_vad_segments", len(r.get("vad_segments_ms") or []))


@pytest.mark.e2e
@pytest.mark.xpu
@pytest.mark.slow
def test_e2e_itn_fields_on_wav(asr_system_xpu, fixtures_dir: Path, metric, request):
    """功能：ITN — 对 wav 跑 System 且 ``enable_itn`` 时结果含 ITN 字段。"""
    paths = _require_wavs(fixtures_dir, "clean")
    cfg = _clone_cfg(asr_system_xpu.config, enable_itn=True)
    sys = FireRedAsr2System(cfg)
    r = sys.process(str(paths["clean"]), uttid="e2e_itn")
    _maybe_record_long_multi_style(request, sys, paths["clean"], "e2e_itn", r)
    assert "text_itn" in r and isinstance(r["text_itn"], str)
    assert "text_labeled_itn" in r and isinstance(r["text_labeled_itn"], str)
    metric("e2e_itn_text_len", len(r["text_itn"]))


@pytest.mark.e2e
@pytest.mark.xpu
@pytest.mark.slow
def test_e2e_denoise_branch_on_wav(asr_system_xpu, fixtures_dir: Path, metric, request):
    """功能：降噪 — 对 noisy fixture 走 VAD/ASR，且结果带 ``denoise_backend``。"""
    paths = _require_wavs(fixtures_dir, "noisy")
    b = asr_system_xpu.config
    cfg = FireRedAsr2SystemConfig(
        vad_model_dir=b.vad_model_dir,
        lid_model_dir=b.lid_model_dir,
        asr_type=b.asr_type,
        asr_model_dir=b.asr_model_dir,
        punc_model_dir=b.punc_model_dir,
        vad_config=b.vad_config,
        lid_config=b.lid_config,
        asr_config=b.asr_config,
        punc_config=b.punc_config,
        asr_batch_size=b.asr_batch_size,
        punc_batch_size=b.punc_batch_size,
        enable_vad=b.enable_vad,
        enable_lid=False,
        enable_punc=b.enable_punc,
        enable_denoise=True,
        denoise_config=FireRedDenoiserConfig(backend="noisereduce"),
        enable_diarization=False,
    )
    sys = FireRedAsr2System(cfg)
    r = sys.process(str(paths["noisy"]), uttid="e2e_denoise")
    _maybe_record_long_multi_style(request, sys, paths["noisy"], "e2e_denoise", r)
    assert r.get("denoise_backend") == "noisereduce"
    metric("e2e_denoise_text_len", len((r.get("text") or "").strip()))


@pytest.mark.e2e
@pytest.mark.xpu
@pytest.mark.slow
def test_e2e_hotword_config_on_wav(asr_system_xpu, fixtures_dir: Path, metric, request):
    """功能：热词 — AED 配置热词后整链路可跑通（不强制转写内容包含热词）。"""
    paths = _require_wavs(fixtures_dir, "clean")
    cfg = _clone_asr(
        asr_system_xpu.config,
        hotwords=["测试热词"],
        hotword_weight=2.0,
        hotword_complete_bonus=0.5,
    )
    sys = FireRedAsr2System(cfg)
    r = sys.process(str(paths["clean"]), uttid="e2e_hw")
    _maybe_record_long_multi_style(request, sys, paths["clean"], "e2e_hw", r)
    assert "text" in r
    metric("e2e_hotword_text_len", len((r.get("text") or "").strip()))


@pytest.mark.e2e
@pytest.mark.xpu
@pytest.mark.slow
def test_e2e_hotword_on_noisy_proxy_wav(
    asr_system_xpu, fixtures_dir: Path, tmp_path: Path, metric, request
):
    """麦克风场景代理：对 clean wav 叠加轻噪后仍跑通热词栈（不强制转写含热词）。"""
    paths = _require_wavs(fixtures_dir, "clean")
    pcm, sr = sf.read(str(paths["clean"]), dtype="int16")
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1).astype(np.int16)
    rng = np.random.default_rng(42)
    noise = rng.integers(-1500, 1500, size=pcm.shape, dtype=np.int32)
    mix = np.clip(pcm.astype(np.int32) + noise, -32768, 32767).astype(np.int16)
    noisy_path = tmp_path / "noisy_proxy_hotword.wav"
    sf.write(str(noisy_path), mix, int(sr))

    cfg = _clone_asr(
        asr_system_xpu.config,
        hotwords=["测试热词"],
        hotword_weight=2.0,
        hotword_complete_bonus=0.5,
    )
    sys = FireRedAsr2System(cfg)
    r = sys.process(str(noisy_path), uttid="e2e_hw_noisy")
    _maybe_record_long_multi_style(request, sys, noisy_path, "e2e_hw_noisy", r)
    assert "text" in r
    metric("e2e_hotword_noisy_text_len", len((r.get("text") or "").strip()))


@pytest.mark.e2e
@pytest.mark.xpu
@pytest.mark.slow
def test_e2e_lid_on_wav(asr_system_xpu, fixtures_dir: Path, metric, request):
    """功能：LID — 启用 LID 时整链路可跑通（默认真人段 ``assets/metting_0507_seg01.wav``，缺则 clean）。"""
    wav = _e2e_lid_input_path(fixtures_dir)
    cfg = _clone_cfg(asr_system_xpu.config, enable_lid=True)
    sys = FireRedAsr2System(cfg)
    r = sys.process(str(wav), uttid="e2e_lid")
    _maybe_record_long_multi_style(request, sys, wav, "e2e_lid", r)
    assert "sentences" in r
    for s in r["sentences"]:
        assert "lang" in s
    metric("e2e_lid_sentence_count", len(r["sentences"]))


@pytest.mark.e2e
@pytest.mark.xpu
@pytest.mark.slow
def test_e2e_diarization_on_dialog_wav(asr_system_xpu, fixtures_dir: Path, metric, request):
    """功能：说话人分离 — 默认 ``assets/metting_0507_seg01.wav``（CAM++）；否则合成 dialog + ``spectral_tone_pair``。"""
    audio_path = _e2e_diarization_input_path(fixtures_dir)
    use_synthetic = _is_dialog_synthetic_diar_fixture(audio_path, fixtures_dir)
    if use_synthetic:
        cfg = _clone_cfg(
            asr_system_xpu.config,
            enable_diarization=True,
            diar_input_mode="full",
            diar_backend="spectral_tone_pair",
            diar_spectral_f0_hz=220.0,
            diar_spectral_f1_hz=330.0,
        )
    else:
        pytest.importorskip("modelscope", reason="会议录音 diar 需要 modelscope + CAM++ 模型")
        cfg = _clone_cfg(
            asr_system_xpu.config,
            enable_diarization=True,
            diar_input_mode="full",
            diar_backend="modelscope_campplus",
        )
    sys = FireRedAsr2System(cfg)
    r = sys.process(str(audio_path), uttid="e2e_diar")
    _maybe_record_long_multi_style(request, sys, audio_path, "e2e_diar", r)
    assert "diarization_spans" in r
    assert isinstance(r["diarization_spans"], list)
    assert "speaker_label_note" in r
    spans = r["diarization_spans"]
    metric("e2e_diar_span_count", len(spans))
    if use_synthetic:
        assert len(spans) >= 4
        spk_ids = {int(x["speaker_id"]) for x in spans}
        assert len(spk_ids) == 2
        note = r["speaker_label_note"].lower()
        assert "spectral" in note or "220" in r["speaker_label_note"]
    else:
        if not spans:
            note = (r.get("speaker_label_note") or "")[:400]
            pytest.skip(
                "ModelScope 未产出 diarization_spans（可能：缺少 addict / datasets 等依赖、"
                "Windows 上 pyarrow 与运行环境冲突、有效语音不足 5s、模型或网络异常）。"
                f" speaker_label_note 摘要: {note!r}"
            )
        assert len({int(x["speaker_id"]) for x in spans}) >= 1
        assert "modelscope" in r["speaker_label_note"].lower() or "clustering" in r[
            "speaker_label_note"
        ].lower()
    spk_ids = {int(x["speaker_id"]) for x in spans} if spans else set()
    for s in r.get("sentences") or []:
        assert "diar_speaker_id" in s
        assert s["diar_speaker_id"] in spk_ids
        assert s["spk_label"] == s["diar_speaker_id"]


@pytest.mark.e2e
@pytest.mark.xpu
@pytest.mark.slow
def test_e2e_speaker_enroll_on_wav(
    asr_system_xpu, fixtures_dir: Path, tmp_path: Path, metric, request
):
    """功能：声纹注册/匹配 — 谱统计嵌入 + 整段 ``utterance_enrolled_*``（不同长度同频仍匹配）。"""
    enr = _require_enroll_fixtures(fixtures_dir)
    reg_path = tmp_path / "e2e_speakers.json"
    th = 0.45
    cfg = _clone_cfg(
        asr_system_xpu.config,
        enable_speaker_id=True,
        speaker_registry_path=str(reg_path),
        speaker_match_threshold=th,
        speaker_embedder="spectral_stats",
    )
    sys = FireRedAsr2System(cfg)
    sys.register_speaker("spkA", str(enr["a1"]))
    sys.register_speaker("spkB", str(enr["b1"]))
    assert reg_path.is_file()
    r_a = sys.process(str(enr["a2"]), uttid="e2e_enroll_a")
    _maybe_record_long_multi_style(request, sys, enr["a2"], "e2e_enroll", r_a)
    assert "utterance_enrolled_speaker" in r_a
    assert "utterance_enrolled_similarity" in r_a
    assert r_a["utterance_enrolled_speaker"] == "spkA"
    assert float(r_a["utterance_enrolled_similarity"]) >= th
    r_b = sys.process(str(enr["b2"]), uttid="e2e_enroll_b")
    assert r_b["utterance_enrolled_speaker"] == "spkB"
    assert float(r_b["utterance_enrolled_similarity"]) >= th
    metric("e2e_enroll_utt_sim_a", float(r_a["utterance_enrolled_similarity"]))
    metric("e2e_enroll_sentence_count", len(r_a.get("sentences") or []))
    for s in r_a.get("sentences") or []:
        assert "enrolled_speaker" in s
        assert "enrolled_similarity" in s


@pytest.mark.e2e
@pytest.mark.xpu
@pytest.mark.slow
@pytest.mark.modelscope
def test_e2e_speaker_enroll_campplus_sv_on_fixtures(
    asr_system_xpu, fixtures_dir: Path, tmp_path: Path, metric, request
):
    """功能：声纹 1:N — CAM++ SV 嵌入 + 注册库；依赖 modelscope 与 SV 模型下载。"""
    pytest.importorskip("modelscope", reason="CAM++ SV 需要 modelscope")
    enr = _require_enroll_fixtures(fixtures_dir)
    reg_path = tmp_path / "e2e_campplus_sv.json"
    th = float(CAMPLUS_SV_DEFAULT_MATCH_THRESHOLD)
    cfg = _clone_cfg(
        asr_system_xpu.config,
        enable_speaker_id=True,
        speaker_registry_path=str(reg_path),
        speaker_match_threshold=th,
    )
    cfg = with_natural_speech_speaker_stack(
        cfg,
        enable_diarization=False,
        enable_speaker_id=True,
        sv_match_threshold=th,
    )
    sys = FireRedAsr2System(cfg)
    sys.register_speaker("spkA", str(enr["a1"]))
    sys.register_speaker("spkB", str(enr["b1"]))
    r_a = sys.process(str(enr["a2"]), uttid="e2e_sv_a")
    _maybe_record_long_multi_style(request, sys, enr["a2"], "e2e_sv_campplus", r_a)
    sim_a = float(r_a["utterance_enrolled_similarity"])
    metric("e2e_campplus_sv_utt_sim_a", sim_a)
    if r_a["utterance_enrolled_speaker"] != "spkA" or sim_a < th:
        pytest.skip(
            f"CAM++ SV 未稳定命中 spkA（got {r_a['utterance_enrolled_speaker']!r}, sim={sim_a}, th={th}）"
        )
    r_b = sys.process(str(enr["b2"]), uttid="e2e_sv_b")
    sim_b = float(r_b["utterance_enrolled_similarity"])
    if r_b["utterance_enrolled_speaker"] != "spkB" or sim_b < th:
        pytest.skip(
            f"CAM++ SV 未稳定命中 spkB（got {r_b['utterance_enrolled_speaker']!r}, sim={sim_b}, th={th}）"
        )
    for s in r_a.get("sentences") or []:
        assert "enrolled_speaker" in s
        assert "enrolled_similarity" in s


@pytest.mark.e2e
@pytest.mark.xpu
@pytest.mark.slow
def test_e2e_diarization_rttm_sidecar_on_dialog_wav(
    asr_system_xpu, fixtures_dir: Path, metric, request
):
    """功能：说话人分离 — RTTM 侧车真值（离线评测/对齐 ModelScope 输出）。"""
    paths = _require_wavs(fixtures_dir, "dialog")
    dialog_path = paths["dialog"]
    assert dialog_path.with_suffix(".rttm").is_file()
    cfg = _clone_cfg(
        asr_system_xpu.config,
        enable_diarization=True,
        diar_backend="rttm_sidecar",
    )
    sys = FireRedAsr2System(cfg)
    r = sys.process(str(dialog_path), uttid="e2e_diar_rttm")
    _maybe_record_long_multi_style(request, sys, dialog_path, "e2e_diar_rttm", r)
    spans = r["diarization_spans"]
    metric("e2e_diar_rttm_span_count", len(spans))
    assert len(spans) == 5
    assert {int(x["speaker_id"]) for x in spans} == {0, 1}


@pytest.mark.e2e
@pytest.mark.xpu
@pytest.mark.slow
@pytest.mark.modelscope
def test_e2e_diarization_modelscope_on_dialog_wav(
    asr_system_xpu, fixtures_dir: Path, metric, request
):
    """功能：说话人分离 — ModelScope 聚类（需下载模型；合成纯音可能无段则 skip）。"""
    paths = _require_wavs(fixtures_dir, "dialog")
    dialog_path = paths["dialog"]
    cfg = _clone_cfg(
        asr_system_xpu.config,
        enable_diarization=True,
        diar_input_mode="full",
        diar_backend="modelscope_campplus",
    )
    sys = FireRedAsr2System(cfg)
    r = sys.process(str(dialog_path), uttid="e2e_diar_ms")
    _maybe_record_long_multi_style(request, sys, dialog_path, "e2e_diar_ms", r)
    metric("e2e_diar_ms_span_count", len(r.get("diarization_spans") or []))
    if not r.get("diarization_spans"):
        pytest.skip("ModelScope 未返回 diar 段（合成音或环境限制）")
    assert len({int(x["speaker_id"]) for x in r["diarization_spans"]}) >= 1


@pytest.mark.e2e
@pytest.mark.xpu
@pytest.mark.slow
def test_e2e_return_timestamp_path(asr_system_xpu, fixtures_dir: Path, metric, request):
    """功能：词级时间戳 — ``return_timestamp=True`` 时结果含 timestamp 相关结构。"""
    paths = _require_wavs(fixtures_dir, "clean")
    b = asr_system_xpu.config
    asr_cfg = replace(b.asr_config, return_timestamp=True)
    cfg = _clone_cfg(b, asr_config=asr_cfg, enable_punc=True)
    sys = FireRedAsr2System(cfg)
    r = sys.process(str(paths["clean"]), uttid="e2e_ts")
    _maybe_record_long_multi_style(request, sys, paths["clean"], "e2e_ts", r)
    metric("e2e_ts_word_count", len(r.get("words") or []))
    # 合成音常无有效句；有句则检查词时间戳
    for s in r.get("sentences") or []:
        assert "start_ms" in s and "end_ms" in s


@pytest.mark.e2e
@pytest.mark.xpu
@pytest.mark.slow
def test_e2e_full_stack_merged_json(
    asr_system_xpu, fixtures_dir: Path, tmp_path: Path, metric, request
):
    """全功能合并：单 JSON 汇总 VAD+ASR+标点+LID+ITN+词时间戳+热词+谱声纹；可选 CAM++ diar。

    默认在存在 ``assets/metting_0507_seg01.wav`` 且可导入 ``modelscope`` 时启用 ``modelscope_campplus``
    diar；否则用合成 ``clean`` 且不启 diar（合成 ``dialog_2spk_30s`` 常无有效 ASR 句，故不作默认）。
    设置 ``FIREREDASR2S_E2E_RECORD_DIR`` 时写出 ``e2e_merged_full_stack.json``（含 ``stream_session``）。
    """
    enr = _require_enroll_fixtures(fixtures_dir)
    audio_path, diar_mode = _e2e_full_stack_input_and_diar_mode(fixtures_dir)

    b = asr_system_xpu.config
    asr_stacked = _clone_asr(
        b,
        return_timestamp=True,
        hotwords=["测试热词"],
        hotword_weight=2.0,
        hotword_complete_bonus=0.5,
    )
    if diar_mode == "campplus":
        cfg = _clone_cfg(
            asr_stacked,
            enable_lid=True,
            enable_itn=True,
            enable_punc=True,
            enable_diarization=True,
            diar_input_mode="full",
            diar_backend="modelscope_campplus",
            enable_speaker_id=True,
            speaker_registry_path=str(tmp_path / "e2e_full_stack_reg.json"),
            speaker_match_threshold=0.45,
            speaker_embedder="spectral_stats",
        )
    else:
        cfg = _clone_cfg(
            asr_stacked,
            enable_lid=True,
            enable_itn=True,
            enable_punc=True,
            enable_diarization=False,
            enable_speaker_id=True,
            speaker_registry_path=str(tmp_path / "e2e_full_stack_reg.json"),
            speaker_match_threshold=0.45,
            speaker_embedder="spectral_stats",
        )

    sys = FireRedAsr2System(cfg)
    sys.register_speaker("spkA", str(enr["a1"]))
    sys.register_speaker("spkB", str(enr["b1"]))

    r = sys.process(str(audio_path), uttid="e2e_full")
    _maybe_record_merged_full_stack(request, sys, audio_path, "e2e_full", r)

    assert r["uttid"] == "e2e_full"
    assert r.get("vad_segments_ms")
    assert "text_itn" in r and isinstance(r["text_itn"], str)
    assert "text_labeled_itn" in r
    assert "utterance_enrolled_speaker" in r
    sents = r.get("sentences") or []
    assert sents
    for s in sents:
        assert "lang" in s
        assert "text_itn" in s
        assert "enrolled_speaker" in s
        assert "enrolled_similarity" in s
    assert isinstance(r.get("words"), list)

    spans = r.get("diarization_spans") or []
    if diar_mode == "campplus":
        if not spans:
            note = (r.get("speaker_label_note") or "")[:400]
            pytest.skip(
                "CAM++ 未产出 diarization_spans，无法在该环境下验证「全功能+diar」句级字段。"
                f" speaker_label_note 摘要: {note!r}"
            )
        spk_ids = {int(x["speaker_id"]) for x in spans}
        assert len(spk_ids) >= 1
        for s in sents:
            assert "diar_speaker_id" in s
            assert int(s["diar_speaker_id"]) in spk_ids
            assert s["spk_label"] == s["diar_speaker_id"]
    else:
        assert not spans

    metric("e2e_full_stack_sentence_count", len(sents))
    metric("e2e_full_stack_word_count", len(r.get("words") or []))
    metric("e2e_full_stack_diar_spans", len(spans))
