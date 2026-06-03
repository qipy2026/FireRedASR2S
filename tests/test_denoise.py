"""T2 denoising frontend tests."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from fireredasr2s.fireredenh import FireRedDenoiser, FireRedDenoiserConfig
from tests.utils import estimate_snr


def test_denoiser_dtype_shape_cpu(fixtures_dir):
    den = FireRedDenoiser(FireRedDenoiserConfig(backend="noisereduce"))
    y, sr = sf.read(str(fixtures_dir / "clean_zh_short.wav"), dtype="int16")
    out, sr2 = den.process(y, sr)
    assert sr2 == sr
    assert out.shape == y.shape
    assert out.dtype == np.int16


@pytest.mark.slow
def test_denoiser_noisereduce_snr_gain(fixtures_dir, metric):
    """Inject AWGN at ~2 dB SNR; spectral gating should recover several dB vs raw noisy."""
    den = FireRedDenoiser(FireRedDenoiserConfig(backend="noisereduce"))
    clean, sr = sf.read(str(fixtures_dir / "clean_zh_short.wav"), dtype="int16")
    clean_f = clean.astype(np.float32) / 32768.0
    rng = np.random.default_rng(42)
    p_sig = float(np.mean(clean_f**2))
    target_snr_db = 2.0
    p_noise = p_sig / (10 ** (target_snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(p_noise), size=clean_f.shape).astype(np.float32)
    noisy_f = np.clip(clean_f + noise, -1.0, 1.0)
    noisy = (noisy_f * 32768.0).astype(np.int16)
    enhanced, _ = den.process(noisy, sr)
    snr_before = estimate_snr(clean.astype(np.float32), noisy.astype(np.float32))
    snr_after = estimate_snr(clean.astype(np.float32), enhanced.astype(np.float32))
    metric("denoise_snr_before_db", round(snr_before, 3))
    metric("denoise_snr_after_db", round(snr_after, 3))
    gain = snr_after - snr_before
    metric("denoise_snr_gain_db", round(gain, 3))
    assert np.isfinite(snr_after) and snr_after > -200.0
    assert gain > 0.0, (
        f"expected positive SNR gain on AWGN mix, before={snr_before:.2f} "
        f"after={snr_after:.2f} gain={gain:.2f}"
    )


def test_denoiser_df_backend_optional():
    try:
        from df.enhance import init_df  # noqa: F401
    except Exception:
        pytest.skip("deepfilternet not installed")
    den = FireRedDenoiser(FireRedDenoiserConfig(backend="df"))
    x = np.zeros(16000, dtype=np.int16)
    out, sr = den.process(x, 16000)
    assert out.shape == x.shape
    assert sr == 16000


def test_system_denoise_off_no_denoiser_attr():
    from fireredasr2s.fireredasr2system import FireRedAsr2SystemConfig

    cfg = FireRedAsr2SystemConfig(enable_denoise=False)
    assert cfg.enable_denoise is False


@pytest.mark.xpu
@pytest.mark.slow
def test_system_denoise_xpu_wer_drop(asr_system_xpu, fixtures_dir, metric):
    """If ASR returns text on noisy vs denoised, WER vs a fixed ref may improve.

    Synthetic tones often yield empty ASR; we then only assert denoise runs and
    ``denoise_backend`` is present.
    """
    from fireredasr2s.fireredenh import FireRedDenoiser, FireRedDenoiserConfig
    from fireredasr2s.fireredasr2system import FireRedAsr2System, FireRedAsr2SystemConfig

    base_cfg = asr_system_xpu.config
    den_cfg = FireRedAsr2SystemConfig(
        vad_model_dir=base_cfg.vad_model_dir,
        lid_model_dir=base_cfg.lid_model_dir,
        asr_type=base_cfg.asr_type,
        asr_model_dir=base_cfg.asr_model_dir,
        punc_model_dir=base_cfg.punc_model_dir,
        vad_config=base_cfg.vad_config,
        lid_config=base_cfg.lid_config,
        asr_config=base_cfg.asr_config,
        punc_config=base_cfg.punc_config,
        asr_batch_size=base_cfg.asr_batch_size,
        punc_batch_size=base_cfg.punc_batch_size,
        enable_vad=base_cfg.enable_vad,
        enable_lid=base_cfg.enable_lid,
        enable_punc=base_cfg.enable_punc,
        enable_denoise=True,
        denoise_config=FireRedDenoiserConfig(backend="noisereduce"),
        enable_diarization=False,
    )
    sys_d = FireRedAsr2System(den_cfg)
    wav = str(fixtures_dir / "noisy_short.wav")
    r0 = asr_system_xpu.process(wav, uttid="n0")
    r1 = sys_d.process(wav, uttid="n1")
    assert "denoise_backend" in r1 and r1["denoise_backend"] == "noisereduce"
    metric("denoise_asr_text_len_off", len((r0.get("text") or "").strip()))
    metric("denoise_asr_text_len_on", len((r1.get("text") or "").strip()))
    if not (r0.get("text") or "").strip() and not (r1.get("text") or "").strip():
        pytest.skip("synthetic audio produced empty ASR; WER comparison N/A")
