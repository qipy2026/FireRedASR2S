#!/usr/bin/env python3

# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Kai Huang, Yan Jia, Junjie Chen, Wenpeng Li)

import argparse
import glob
import json
import logging
import os

import soundfile as sf
from textgrid import IntervalTier, TextGrid

from fireredasr2s.fireredasr2 import FireRedAsr2Config
from fireredasr2s.fireredasr2system import (FireRedAsr2System,
                                            FireRedAsr2SystemConfig)
from fireredasr2s.fireredenh import FireRedDenoiserConfig
from fireredasr2s.fireredlid import FireRedLidConfig
from fireredasr2s.logging_config import configure_logging
from fireredasr2s.fireredpunc import FireRedPuncConfig
from fireredasr2s.fireredvad import FireRedVadConfig

configure_logging()
logger = logging.getLogger("fireredasr2s.asr_system")


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
input_g = parser.add_argument_group("Input Options")
input_g.add_argument("--wav_path", type=str)
input_g.add_argument("--wav_paths", type=str, nargs="*")
input_g.add_argument("--wav_dir", type=str)
input_g.add_argument("--wav_scp", type=str)
input_g.add_argument("--sort_wav_by_dur", type=int, default=0)

output_g = parser.add_argument_group("Output Options")
output_g.add_argument("--outdir", type=str, default="output")
output_g.add_argument("--write_textgrid", type=int, default=1)
output_g.add_argument("--write_srt", type=int, default=1)
output_g.add_argument("--save_segment", type=int, default=0)

module_g = parser.add_argument_group("Module Switches")
module_g.add_argument('--enable_vad', type=int, default=1, choices=[0, 1])
module_g.add_argument('--enable_lid', type=int, default=1, choices=[0, 1])
module_g.add_argument('--enable_punc', type=int, default=1, choices=[0, 1])
module_g.add_argument(
    '--enable_itn',
    type=int,
    default=0,
    choices=[0, 1],
    help='Inverse Text Normalization (CN spoken numbers/units → digits/%/℃ via cn2an)',
)
module_g.add_argument(
    '--enable_denoise',
    type=int,
    default=0,
    choices=[0, 1],
    help='Speech enhancement before VAD (optional deps: noisereduce / deepfilternet)',
)
module_g.add_argument(
    '--denoise_backend',
    type=str,
    default='noisereduce',
    choices=['noisereduce', 'df'],
    help='Denoiser backend: noisereduce (default) or df (DeepFilterNet)',
)
module_g.add_argument(
    '--enable_diarization',
    type=int,
    default=0,
    choices=[0, 1],
    help='ModelScope segmentation_clustering speaker diarization (requires modelscope + downloads)',
)
module_g.add_argument(
    '--diar_model_id',
    type=str,
    default='damo/speech_campplus_speaker-diarization_common',
    help='ModelScope model id or local dir for speaker_diarization task',
)
module_g.add_argument(
    '--diar_model_revision',
    type=str,
    default='',
    help='Optional model_revision for ModelScope pipeline',
)
module_g.add_argument(
    '--diar_input_mode',
    type=str,
    default='full',
    choices=['full', 'vad_segments'],
    help='Diarization audio packaging: full file vs VAD slices',
)
module_g.add_argument(
    '--diar_align_level',
    type=str,
    default='segment',
    choices=['segment', 'word'],
    help='Map speakers to sentences by VAD segment overlap or ASR token timestamps',
)
module_g.add_argument(
    '--diar_refine_subsegment',
    type=int,
    default=0,
    choices=[0, 1],
    help='Placeholder span refinement hook (currently no-op)',
)
module_g.add_argument(
    '--diar_min_speaker_dur_ms',
    type=int,
    default=400,
    help='Merge word-diar fragments shorter than this (ms)',
)
module_g.add_argument(
    '--diar_backend',
    type=str,
    default='modelscope_campplus',
    help=(
        'Diarization: modelscope_campplus | pyannote | rttm_sidecar | '
        'spectral_tone_pair (dual-tone / proxy dialog) | speakerlab (raises)'
    ),
)
module_g.add_argument(
    '--diar_hf_token',
    type=str,
    default='',
    help='HuggingFace token for pyannote backend (optional)',
)
module_g.add_argument(
    '--diar_spectral_f0_hz',
    type=float,
    default=220.0,
    help='spectral_tone_pair backend: first reference tone (Hz)',
)
module_g.add_argument(
    '--diar_spectral_f1_hz',
    type=float,
    default=330.0,
    help='spectral_tone_pair backend: second reference tone (Hz)',
)
module_g.add_argument(
    '--diar_text_labeled_mode',
    type=str,
    default='transcript_order',
    choices=['transcript_order', 'cluster_id'],
    help=(
        'Prefix in text_labeled: transcript_order = 说话人1..K by first sentence occurrence; '
        'cluster_id = 聚类 id + 1 (aligns with diarization_spans, may skip numbers)'
    ),
)
module_g.add_argument(
    '--enable_speaker_id',
    type=int,
    default=0,
    choices=[0, 1],
    help='Optional enrollment: match sentence audio to registered speaker embeddings',
)
module_g.add_argument(
    '--speaker_registry_path',
    type=str,
    default='',
    help='JSON path to persist speaker embeddings (optional)',
)
module_g.add_argument(
    '--speaker_match_threshold',
    type=float,
    default=0.999,
    help='Cosine similarity threshold for enrollment match',
)
module_g.add_argument(
    '--speaker_embedder',
    type=str,
    default='content_hash',
    help=(
        'Speaker embedding: content_hash | spectral_stats | modelscope_campplus_sv '
        '(requires pip install modelscope + model download)'
    ),
)
module_g.add_argument(
    '--speaker_embedder_model_id',
    type=str,
    default='',
    help='modelscope_campplus_sv: ModelScope hub id (default built-in CAM++ zh 16k)',
)
module_g.add_argument(
    '--speaker_embedder_model_revision',
    type=str,
    default='',
    help='Optional ModelScope model_revision for SV embedder',
)
module_g.add_argument(
    '--register_speaker',
    action='append',
    default=[],
    metavar='NAME=WAV',
    help=(
        'Before transcribing: enroll display name with audio (repeatable). '
        'Implies --enable_speaker_id 1. Example: --register_speaker 张三=./enroll_zhang.wav'
    ),
)
module_g.add_argument(
    '--natural_speech_speaker_stack',
    type=int,
    default=0,
    choices=[0, 1],
    help=(
        'Set CAM++ hub IDs + production SV cosine threshold (~0.35); use with '
        '--enable_diarization 1 and/or --enable_speaker_id 1 (requires modelscope). '
        '--register_speaker implies --enable_speaker_id 1.'
    ),
)
module_g.add_argument(
    '--production_sv_threshold',
    type=float,
    default=0.35,
    help='With --natural_speech_speaker_stack 1: enrollment cosine threshold for CAM++ SV',
)

asr_g = parser.add_argument_group("ASR Options")
asr_g.add_argument('--asr_type', type=str, default="aed", choices=["aed", "llm"])
asr_g.add_argument('--asr_model_dir', type=str, default="pretrained_models/FireRedASR2-AED")
asr_g.add_argument('--asr_use_gpu', type=int, default=1)
asr_g.add_argument(
    '--asr_device',
    type=str,
    default='',
    help='ASR/LLM torch device, e.g. xpu or cuda:0; empty + asr_use_gpu=1 → cuda else xpu (Intel +xpu / IPEX) else cpu',
)
asr_g.add_argument(
    '--asr_runtime',
    type=str,
    default='torch',
    choices=['torch', 'vllm', 'trtllm'],
    help='LLM decoding runtime (AED ignores this)',
)
asr_g.add_argument('--asr_use_half', type=int, default=0)
asr_g.add_argument(
    '--aed_dynamic_int8_pt',
    type=str,
    default='',
    help=(
        'AED only: load CPU dynamic INT8 Linear weights from this file '
        '(see scripts/quantize_aed_int8.py). Requires --asr_use_gpu 0 and --asr_use_half 0.'
    ),
)
asr_g.add_argument(
    '--hotwords',
    type=str,
    default='',
    help='Comma-separated hotword phrases for AED beam biasing',
)
asr_g.add_argument('--hotword_weight', type=float, default=0.0)
asr_g.add_argument('--hotword_complete_bonus', type=float, default=0.0)
asr_g.add_argument("--asr_batch_size", type=int, default=1)
# FireRedASR-AED
asr_g.add_argument("--beam_size", type=int, default=3)
asr_g.add_argument("--decode_max_len", type=int, default=0)
asr_g.add_argument("--nbest", type=int, default=1)
asr_g.add_argument("--softmax_smoothing", type=float, default=1.25)
asr_g.add_argument("--aed_length_penalty", type=float, default=0.6)
asr_g.add_argument("--eos_penalty", type=float, default=1.0)
asr_g.add_argument("--return_timestamp", type=int, default=1)
# FireRedASR-AED External LM
asr_g.add_argument("--elm_dir", type=str, default="")
asr_g.add_argument("--elm_weight", type=float, default=0.0)

vad_g = parser.add_argument_group("VAD Options")
vad_g.add_argument('--vad_model_dir', type=str, default="pretrained_models/FireRedVAD/VAD")
vad_g.add_argument('--vad_use_gpu', type=int, default=1)
# Non-streaming VAD
vad_g.add_argument("--vad_chunk_max_frame", type=int, default=30000)
vad_g.add_argument("--smooth_window_size", type=int, default=5)
vad_g.add_argument("--speech_threshold", type=float, default=0.2)
vad_g.add_argument("--min_speech_frame", type=int, default=20)
vad_g.add_argument("--max_speech_frame", type=int, default=1000)
vad_g.add_argument("--min_silence_frame", type=int, default=10)
vad_g.add_argument("--merge_silence_frame", type=int, default=50)
vad_g.add_argument("--extend_speech_frame", type=int, default=10)

lid_g = parser.add_argument_group("LID Options")
lid_g.add_argument('--lid_model_dir', type=str, default="pretrained_models/FireRedLID")
lid_g.add_argument('--lid_use_gpu', type=int, default=1)

punc_g = parser.add_argument_group("Punc Options")
punc_g.add_argument('--punc_model_dir', type=str, default="pretrained_models/FireRedPunc")
punc_g.add_argument('--punc_use_gpu', type=int, default=1)
punc_g.add_argument("--punc_batch_size", type=int, default=1)
punc_g.add_argument('--punc_with_timestamp', type=int, default=1)
punc_g.add_argument('--punc_sentence_max_length', type=int, default=-1)


def main(args):
    reg_specs = [x.strip() for x in (args.register_speaker or []) if x and str(x).strip()]
    if reg_specs and not int(args.enable_speaker_id):
        logger.info("--register_speaker set: enabling speaker_id (enable_speaker_id=1)")
        args.enable_speaker_id = 1

    wavs = get_wav_info(args)
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
    fout = open(args.outdir + "/result.jsonl", "w") if args.outdir else None

    # Build Models
    # VAD
    vad_config = FireRedVadConfig(
        args.vad_use_gpu,
        args.smooth_window_size,
        args.speech_threshold,
        args.min_speech_frame,
        args.max_speech_frame,
        args.min_silence_frame,
        args.merge_silence_frame,
        args.extend_speech_frame,
        args.vad_chunk_max_frame
    )
    # LID
    lid_config = FireRedLidConfig(args.lid_use_gpu)
    # ASR
    hotwords = [x.strip() for x in args.hotwords.split(",") if x.strip()]
    asr_config = FireRedAsr2Config(
        args.asr_use_gpu,
        args.asr_use_half,
        args.beam_size,
        args.nbest,
        args.decode_max_len,
        args.softmax_smoothing,
        args.aed_length_penalty,
        args.eos_penalty,
        args.return_timestamp,
        0, 1.0, 0.0, 1.0,
        args.elm_dir,
        args.elm_weight,
        device=(args.asr_device or "").strip(),
        hotwords=hotwords,
        hotword_weight=args.hotword_weight,
        hotword_complete_bonus=args.hotword_complete_bonus,
        runtime=args.asr_runtime,
        aed_dynamic_int8_pt=(args.aed_dynamic_int8_pt or "").strip(),
    )
    # Punc
    punc_config = FireRedPuncConfig(
        args.punc_use_gpu,
        args.punc_sentence_max_length
    )
    denoise_config = FireRedDenoiserConfig(backend=args.denoise_backend)

    asr_system_config = FireRedAsr2SystemConfig(
        vad_model_dir=args.vad_model_dir,
        lid_model_dir=args.lid_model_dir,
        asr_type=args.asr_type,
        asr_model_dir=args.asr_model_dir,
        punc_model_dir=args.punc_model_dir,
        vad_config=vad_config,
        lid_config=lid_config,
        asr_config=asr_config,
        punc_config=punc_config,
        asr_batch_size=args.asr_batch_size,
        punc_batch_size=args.punc_batch_size,
        enable_vad=bool(args.enable_vad),
        enable_lid=bool(args.enable_lid),
        enable_punc=bool(args.enable_punc),
        enable_itn=bool(args.enable_itn),
        enable_denoise=bool(args.enable_denoise),
        denoise_config=denoise_config,
        enable_diarization=bool(args.enable_diarization),
        diar_model_id=args.diar_model_id,
        diar_model_revision=(args.diar_model_revision or None),
        diar_input_mode=args.diar_input_mode,
        diar_align_level=args.diar_align_level,
        diar_refine_subsegment=bool(args.diar_refine_subsegment),
        diar_min_speaker_dur_ms=int(args.diar_min_speaker_dur_ms),
        diar_backend=args.diar_backend,
        diar_hf_token=(args.diar_hf_token or "").strip(),
        diar_spectral_f0_hz=float(args.diar_spectral_f0_hz),
        diar_spectral_f1_hz=float(args.diar_spectral_f1_hz),
        diar_text_labeled_mode=str(args.diar_text_labeled_mode),
        enable_speaker_id=bool(args.enable_speaker_id),
        speaker_registry_path=(args.speaker_registry_path or "").strip(),
        speaker_match_threshold=float(args.speaker_match_threshold),
        speaker_embedder=args.speaker_embedder,
        speaker_embedder_model_id=(args.speaker_embedder_model_id or "").strip(),
        speaker_embedder_model_revision=(
            (args.speaker_embedder_model_revision or "").strip() or None
        ),
    )
    if bool(args.natural_speech_speaker_stack):
        from fireredasr2s.firereddiar.production import with_natural_speech_speaker_stack

        asr_system_config = with_natural_speech_speaker_stack(
            asr_system_config,
            sv_match_threshold=float(args.production_sv_threshold),
        )
    asr_system = FireRedAsr2System(asr_system_config)

    for spec in reg_specs:
        if "=" not in spec:
            raise ValueError(
                f"--register_speaker expects NAME=path, got {spec!r} "
                "(use e.g. --register_speaker alice=./alice_enroll.wav)"
            )
        name, wpath = spec.split("=", 1)
        name, wpath = name.strip(), wpath.strip()
        if not name or not wpath:
            raise ValueError(f"invalid --register_speaker entry: {spec!r}")
        if not os.path.isfile(wpath):
            raise FileNotFoundError(f"enrollment wav not found for {name!r}: {wpath}")
        logger.info("Register speaker %r from %s", name, wpath)
        asr_system.register_speaker(name, wpath)

    for i, (uttid, wav_path) in enumerate(wavs):
        logger.info("")

        result = asr_system.process(wav_path, uttid)

        logger.info(f"FINAL: {result}")

        if fout:
            fout.write(f"{json.dumps(result, ensure_ascii=False)}\n")
            fout.flush()
        name = os.path.basename(wav_path).replace(".wav", "")
        if args.write_textgrid:
            tg_dir = os.path.join(args.outdir, "asr_tg")
            write_textgrid(tg_dir, name, result["dur_s"], result["sentences"], result["words"])
        if args.write_srt:
            srt_dir = os.path.join(args.outdir, "asr_srt")
            write_srt(srt_dir, name, result["sentences"])
        if args.save_segment:
            save_segment_dir = os.path.join(args.outdir, "vad_segment")
            split_and_save_segment(wav_path, result["vad_segments_ms"], save_segment_dir)

    if fout:
        fout.close()
    logger.info("All Done")


def get_wav_info(args):
    """
    Returns:
        wavs: list of (uttid, wav_path)
    """
    def base(p): return os.path.basename(p).replace(".wav", "")
    if args.wav_path:
        wavs = [(base(args.wav_path), args.wav_path)]
    elif args.wav_paths and len(args.wav_paths) >= 1:
        wavs = [(base(p), p) for p in sorted(args.wav_paths)]
    elif args.wav_scp:
        wavs = [line.strip().split() for line in open(args.wav_scp)]
    elif args.wav_dir:
        wavs = glob.glob(f"{args.wav_dir}/**/*.wav", recursive=True)
        wavs = [(base(p), p) for p in sorted(wavs)]
    else:
        raise ValueError("Please provide valid wav info")
    logger.info(f"#wavs={len(wavs)}")
    return wavs


def write_textgrid(tg_dir, name, wav_dur, sentences, words=None):
    os.makedirs(tg_dir, exist_ok=True)
    textgrid_file = os.path.join(tg_dir, name + ".TextGrid")
    logger.info(f"Write {textgrid_file}")
    textgrid = TextGrid(maxTime=wav_dur)

    tier = IntervalTier(name="sentence", maxTime=wav_dur)
    for sentence in sentences:
        start_s = sentence["start_ms"] / 1000.0
        end_s = sentence["end_ms"] / 1000.0
        text = sentence["text"]
        confi = sentence["asr_confidence"]
        if start_s == end_s:
            logger.info(f"(sent) Write TG, skip start=end {start_s} {text}")
            continue
        start_s = max(start_s, 0)
        end_s = min(end_s, wav_dur)
        tier.add(minTime=start_s, maxTime=end_s, mark=f"{text}\n{confi}")
    textgrid.append(tier)

    if words:
        tier = IntervalTier(name="token", maxTime=wav_dur)
        for word in words:
            start_s = word["start_ms"] / 1000.0
            end_s = word["end_ms"] / 1000.0
            text = word["text"]
            if start_s == end_s:
                logger.info(f"(word) Write TG, skip start=end {start_s} {text}")
                continue
            start_s = max(start_s, 0)
            end_s = min(end_s, wav_dur)
            tier.add(minTime=start_s, maxTime=end_s, mark=text)
        textgrid.append(tier)
    textgrid.write(textgrid_file)


def write_srt(srt_dir, name, sentences):
    def _ms2srt_time(ms):
        h = ms // 1000 // 3600
        m = (ms // 1000 % 3600) // 60
        s = (ms // 1000 % 3600) % 60
        ms = (ms % 1000)
        r = f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
        return r
    os.makedirs(srt_dir, exist_ok=True)
    srt_file = os.path.join(srt_dir, name + ".srt")
    logger.info(f"Write {srt_file}")

    i = 0
    with open(srt_file, "w") as fout:
        for sentence in sentences:
            start_ms = sentence["start_ms"]
            end_ms = sentence["end_ms"]
            text = sentence["text"]
            if text.strip() == "":
                continue

            i += 1
            fout.write(f"{i}\n")
            s = _ms2srt_time(start_ms)
            e = _ms2srt_time(end_ms)
            fout.write(f"{s} --> {e}\n")
            fout.write(f"{text}\n")
            if i != len(sentences):
                fout.write("\n")


def split_and_save_segment(wav_path, timestamps_ms, save_segment_dir):
    logger.info("Split & save segment")
    os.makedirs(save_segment_dir, exist_ok=True)
    wav_np, sample_rate = sf.read(wav_path, dtype="int16")
    for i, (start_ms, end_ms) in enumerate(timestamps_ms):
        uttid = wav_path.split("/")[-1].replace(".wav", "")
        seg_id = f"{uttid}_{i}_{start_ms}_{end_ms}"
        seg_path = f"{save_segment_dir}/{seg_id}.wav"
        start = int(start_ms / 1000 * sample_rate)
        end = int(end_ms / 1000 * sample_rate)
        sf.write(seg_path, wav_np[start:end], samplerate=sample_rate)


def cli_main():
    args = parser.parse_args()
    logger.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
