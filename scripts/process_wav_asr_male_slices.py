#!/usr/bin/env python3
"""Offline ASR (FireRedASR2S + XPU) and export male-speaker WAV slices.

Usage:
  .venv\\Scripts\\python.exe scripts/process_wav_asr_male_slices.py <wav_path> [--outdir DIR]

Writes next to the wav (or --outdir):
  <stem>.txt, <stem>.json, <stem>_male_slices/<stem>_male_NNN.wav, transcript sidecar.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# Repo root on sys.path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("PYTORCH_ENABLE_XPU_FALLBACK", "1")

from fireredasr2s.fireredasr2 import FireRedAsr2Config
from fireredasr2s.fireredasr2system import FireRedAsr2System, FireRedAsr2SystemConfig
from fireredasr2s.fireredlid import FireRedLidConfig
from fireredasr2s.fireredpunc import FireRedPuncConfig
from fireredasr2s.fireredvad import FireRedVadConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("process_wav_asr_male_slices")

# Plausible male fundamental range (Hz); outside → ignore cluster for gender vote.
_MALE_F0_MIN_HZ = 90.0
_MALE_F0_MAX_HZ = 210.0
_MIN_SPK_DUR_S = 2.0


def _median_f0_hz(wav_f32: np.ndarray, sr: int) -> float | None:
    if wav_f32.size < int(0.08 * sr):
        return None
    f0, voiced_flag, _ = librosa.pyin(
        wav_f32,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    vals = f0[voiced_flag & np.isfinite(f0)]
    if vals.size < 3:
        return None
    return float(np.median(vals))


def _speaker_stats(
    wav_np: np.ndarray,
    sr: int,
    spans: list[tuple[float, float, int]],
) -> dict[int, dict[str, float]]:
    x = wav_np.astype(np.float32)
    if np.max(np.abs(x)) > 1.5:
        x = x / 32768.0
    out: dict[int, dict[str, float]] = {}
    for t0, t1, spk in spans:
        spk = int(spk)
        dur = max(0.0, t1 - t0)
        s0 = int(max(0, t0) * sr)
        s1 = int(min(len(x), t1) * sr)
        if s1 - s0 < int(0.1 * sr):
            continue
        f0 = _median_f0_hz(x[s0:s1], sr)
        if spk not in out:
            out[spk] = {"dur_s": 0.0, "f0_vals": []}
        out[spk]["dur_s"] += dur
        if f0 is not None:
            out[spk]["f0_vals"].append(f0)
    stats: dict[int, dict[str, float]] = {}
    for spk, raw in out.items():
        vals = raw["f0_vals"]
        stats[spk] = {
            "dur_s": float(raw["dur_s"]),
            "median_f0_hz": float(np.median(vals)) if vals else float("nan"),
        }
    return stats


def _pick_male_speaker_id(stats: dict[int, dict[str, float]]) -> int | None:
    candidates: list[tuple[int, float, float]] = []
    for spk, st in stats.items():
        dur = st["dur_s"]
        f0 = st["median_f0_hz"]
        if dur < _MIN_SPK_DUR_S or not np.isfinite(f0):
            continue
        if f0 < _MALE_F0_MIN_HZ or f0 > _MALE_F0_MAX_HZ:
            continue
        candidates.append((spk, f0, dur))
    if not candidates:
        # Relax F0 band but keep duration gate
        for spk, st in stats.items():
            dur, f0 = st["dur_s"], st["median_f0_hz"]
            if dur >= _MIN_SPK_DUR_S and np.isfinite(f0) and f0 >= _MALE_F0_MIN_HZ:
                candidates.append((spk, f0, dur))
    if not candidates:
        return None
    # Prefer plausible male pitch; tie-break by more speech time
    candidates.sort(key=lambda x: (x[1], -x[2]))
    return int(candidates[0][0])


def _merge_spans(
    spans: list[tuple[int, int]],
    gap_ms: int = 120,
) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap_ms:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _export_slices(
    wav_path: Path,
    wav_np: np.ndarray,
    sr: int,
    spans_ms: list[tuple[int, int]],
    out_dir: Path,
    stem: str,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict] = []
    for i, (start_ms, end_ms) in enumerate(spans_ms, start=1):
        s0 = int(start_ms / 1000.0 * sr)
        s1 = int(end_ms / 1000.0 * sr)
        seg = wav_np[s0:s1]
        if seg.size == 0:
            continue
        name = f"{stem}_male_{i:03d}.wav"
        seg_path = out_dir / name
        sf.write(str(seg_path), seg, sr)
        saved.append(
            {
                "file": name,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "dur_s": round((end_ms - start_ms) / 1000.0, 3),
            }
        )
    return saved


def _build_system(repo: Path, device: str) -> FireRedAsr2System:
    asr_config = FireRedAsr2Config(
        use_gpu=1,
        use_half=0,
        return_timestamp=True,
        device=device,
    )
    cfg = FireRedAsr2SystemConfig(
        asr_model_dir=str(repo / "pretrained_models" / "FireRedASR2-AED"),
        vad_model_dir=str(repo / "pretrained_models" / "FireRedVAD" / "VAD"),
        lid_model_dir=str(repo / "pretrained_models" / "FireRedLID"),
        punc_model_dir=str(repo / "pretrained_models" / "FireRedPunc"),
        asr_config=asr_config,
        vad_config=FireRedVadConfig(use_gpu=False),
        lid_config=FireRedLidConfig(use_gpu=False),
        punc_config=FireRedPuncConfig(use_gpu=False),
        enable_vad=True,
        enable_lid=True,
        enable_punc=True,
        enable_diarization=True,
        diar_align_level="word",
        diar_backend="modelscope_campplus",
    )
    return FireRedAsr2System(cfg)


def process_wav(wav_path: Path, outdir: Path | None, device: str) -> dict:
    wav_path = wav_path.resolve()
    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)
    stem = wav_path.stem
    out_base = (outdir or wav_path.parent).resolve()
    out_base.mkdir(parents=True, exist_ok=True)

    repo = _REPO
    logger.info("Loading FireRedASR2S on device=%s", device)
    system = _build_system(repo, device)

    logger.info("ASR + diarization: %s", wav_path)
    result = system.process(str(wav_path), stem)

    # Full transcript
    lines = []
    for sent in result.get("sentences", []):
        t = (sent.get("text") or "").strip()
        if not t:
            continue
        spk = sent.get("diar_speaker_id")
        prefix = f"[spk{spk}] " if spk is not None else ""
        lines.append(f"{prefix}{t}")
    full_text = "\n".join(lines)

    txt_path = out_base / f"{stem}.txt"
    json_path = out_base / f"{stem}.json"
    txt_path.write_text(full_text + ("\n" if full_text else ""), encoding="utf-8")
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote %s and %s", txt_path.name, json_path.name)

    # Male slices from diarization + pitch
    wav_np, sr = sf.read(str(wav_path), dtype="int16")
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1).astype(np.int16)

    diar_spans_raw = result.get("diarization_spans") or []
    diar_spans: list[tuple[float, float, int]] = [
        (s["start_ms"] / 1000.0, s["end_ms"] / 1000.0, int(s["speaker_id"]))
        for s in diar_spans_raw
    ]

    male_spk: int | None = None
    spk_stats: dict[int, dict[str, float]] = {}
    if diar_spans:
        spk_stats = _speaker_stats(wav_np, int(sr), diar_spans)
        male_spk = _pick_male_speaker_id(spk_stats)
        logger.info("Speaker stats: %s → male_spk=%s", spk_stats, male_spk)

    male_span_ms: list[tuple[int, int]] = []
    if male_spk is not None and diar_spans:
        for t0, t1, spk in diar_spans:
            if int(spk) == int(male_spk):
                male_span_ms.append((int(round(t0 * 1000)), int(round(t1 * 1000))))
    else:
        # Fallback: sentences tagged male by per-sentence F0
        logger.warning("Diar/pitch cluster failed; using per-sentence F0 fallback")
        x = wav_np.astype(np.float32) / 32768.0
        for sent in result.get("sentences", []):
            s0 = int(sent["start_ms"])
            s1 = int(sent["end_ms"])
            a0 = int(s0 / 1000.0 * sr)
            a1 = int(s1 / 1000.0 * sr)
            med = _median_f0_hz(x[a0:a1], int(sr))
            if (
                med is not None
                and _MALE_F0_MIN_HZ <= med <= _MALE_F0_MAX_HZ
            ):
                male_span_ms.append((s0, s1))

    male_span_ms = _merge_spans(male_span_ms)
    slice_dir = out_base / f"{stem}_male_slices"
    saved = _export_slices(wav_path, wav_np, int(sr), male_span_ms, slice_dir, stem)

    male_lines = []
    for sent in result.get("sentences", []):
        if male_spk is not None:
            if sent.get("diar_speaker_id") != male_spk:
                continue
        else:
            s0, s1 = int(sent["start_ms"]), int(sent["end_ms"])
            a0 = int(s0 / 1000.0 * sr)
            a1 = int(s1 / 1000.0 * sr)
            med = _median_f0_hz(wav_np.astype(np.float32)[a0:a1] / 32768.0, int(sr))
            if med is None or med < _MALE_F0_MIN_HZ or med > _MALE_F0_MAX_HZ:
                continue
        t = (sent.get("text") or "").strip()
        if t:
            male_lines.append(t)

    male_txt = out_base / f"{stem}_male.txt"
    male_txt.write_text("\n".join(male_lines) + ("\n" if male_lines else ""), encoding="utf-8")

    summary = {
        "wav": str(wav_path),
        "device": device,
        "transcript_txt": str(txt_path),
        "transcript_json": str(json_path),
        "male_txt": str(male_txt),
        "male_speaker_id": male_spk,
        "speaker_stats": spk_stats,
        "male_slice_dir": str(slice_dir),
        "male_slices": saved,
        "dur_s": result.get("dur_s"),
    }
    summary_path = out_base / f"{stem}_male_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Male slices: %d files in %s", len(saved), slice_dir)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wav_path", type=Path)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--device", type=str, default="xpu", help="torch device for ASR (default: xpu)")
    args = p.parse_args()

    import torch

    if args.device == "xpu" and not (getattr(torch, "xpu", None) and torch.xpu.is_available()):
        logger.warning("XPU not available, falling back to cpu")
        args.device = "cpu"

    summary = process_wav(args.wav_path, args.outdir, args.device)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
