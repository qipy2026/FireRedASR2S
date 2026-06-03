#!/usr/bin/env python3
"""Run ModelScope CAM++ speaker diarization on one audio file (no ASR / VAD stack).

Loads mono 16 kHz via ``fireredasr2s.firereddiar.audio``, calls
``run_diarization_backend(..., diar_backend=modelscope_campplus, diar_input_mode=full)``,
prints JSON list of ``{start_ms, end_ms, speaker_id}``.

Requires: ``pip install 'fireredasr2s[modelscope]'`` (pulls clustering deps: ``scikit-learn``,
``hdbscan``, ``umap-learn``, etc.) and first-run model download.

Example:
    .venv/Scripts/python.exe scripts/run_campplus_diar_wav.py assets/metting_0507_seg01.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        type=Path,
        help="WAV/MP3 path (decoded like the ASR stack: mono, 16 kHz)",
    )
    parser.add_argument(
        "--model",
        default="damo/speech_campplus_speaker-diarization_common",
        help="ModelScope diarization model id or local dir",
    )
    parser.add_argument(
        "--revision",
        default="",
        help="Optional model_revision",
    )
    args = parser.parse_args()
    ap = args.audio.expanduser().resolve()
    if not ap.is_file():
        print(f"not a file: {ap}", file=sys.stderr)
        return 1

    from fireredasr2s.firereddiar.audio import load_pcm_int16_mono, prepare_asr_stack_audio
    from fireredasr2s.firereddiar.backends import run_diarization_backend

    wav_np, sr = load_pcm_int16_mono(ap)
    wav_np, sr = prepare_asr_stack_audio(wav_np, sr)
    dur = float(len(wav_np)) / float(sr)
    vad_dummy = [(0.0, dur)]

    try:
        spans = run_diarization_backend(
            "modelscope_campplus",
            wav_np,
            sr,
            vad_dummy,
            model_id=args.model,
            model_revision=args.revision or None,
            diar_input_mode="full",
            wav_path=str(ap),
        )
    except ImportError as e:
        print(f"ImportError (install fireredasr2s[modelscope]): {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"diarization failed: {e}", file=sys.stderr)
        return 3

    if not spans:
        print(
            json.dumps(
                {
                    "audio": str(ap),
                    "dur_s": round(dur, 6),
                    "n_spans": 0,
                    "spans": [],
                    "ok": False,
                    "note": "backend returned no spans (deps, model, or audio too short / unparsed output)",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    out = [
        {
            "start_ms": int(round(t0 * 1000)),
            "end_ms": int(round(t1 * 1000)),
            "speaker_id": int(spk),
        }
        for t0, t1, spk in spans
    ]
    spk_ids = sorted({x["speaker_id"] for x in out})
    print(
        json.dumps(
            {
                "audio": str(ap),
                "dur_s": round(dur, 6),
                "ok": True,
                "n_spans": len(out),
                "n_speaker_clusters": len(spk_ids),
                "speaker_ids": spk_ids,
                "spans": out,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
