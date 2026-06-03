# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Kai Huang, Yan Jia, Junjie Chen, Wenpeng Li)

from __future__ import annotations

import os
import sys

from fireredasr2s.win_console_utf8 import ensure_stdio_utf8

ensure_stdio_utf8()

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import soundfile as sf

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
from fireredasr2s.fireredenh import FireRedDenoiser, FireRedDenoiserConfig
from fireredasr2s.fireredlid import FireRedLid, FireRedLidConfig
from fireredasr2s.fireredpunc import FireRedPunc, FireRedPuncConfig
from fireredasr2s.fireredtn import FireRedItn, FireRedItnConfig
from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig

from fireredasr2s.firereddiar.align import try_word_diar_sentences
from fireredasr2s.firereddiar.audio import load_pcm_int16_mono, prepare_asr_stack_audio
from fireredasr2s.logging_config import configure_logging

configure_logging()
logger = logging.getLogger("fireredasr2s.asr_system")

# VAD 未检出语音时，对「较短」整段音频回退为全文件 ASR（避免纯静音长文件误跑数小时）。
_VAD_EMPTY_FALLBACK_MIN_DUR_S = 0.2
_VAD_EMPTY_FALLBACK_MAX_DUR_S = 600.0


def _vad_seg_key_to_index(timestamps_s):
    """Map (start_ms, end_ms) as in VAD uttid suffix _s###_e### -> segment index."""
    m = {}
    for idx, (s, e) in enumerate(timestamps_s):
        m[(int(s * 1000), int(e * 1000))] = idx
    return m


def _vad_segment_idx_from_uttid(uttid, seg_key_to_idx):
    m = re.search(r"_s(\d+)_e(\d+)$", uttid)
    if not m:
        return -1
    key = (int(m.group(1)), int(m.group(2)))
    return seg_key_to_idx.get(key, -1)


def _speaker_by_overlap(
    start_ms: int,
    end_ms: int,
    diar_spans: list[tuple[float, float, int]],
) -> int:
    """Pick speaker id by maximum overlap in seconds; tie-break / no-overlap uses interval center."""
    if not diar_spans:
        return 0
    s0, e0 = start_ms / 1000.0, end_ms / 1000.0
    center = (s0 + e0) / 2.0
    best_spk = int(diar_spans[0][2])
    best_ov = -1.0
    for t0, t1, spk in diar_spans:
        ov = max(0.0, min(e0, t1) - max(s0, t0))
        spk_i = int(spk)
        if ov > best_ov + 1e-9:
            best_ov = ov
            best_spk = spk_i
        elif abs(ov - best_ov) <= 1e-9 and ov > 0 and t0 <= center <= t1:
            best_spk = spk_i
    if best_ov > 0:
        return best_spk
    for t0, t1, spk in diar_spans:
        if t0 <= center <= t1:
            return int(spk)
    t0, t1, spk = min(
        diar_spans,
        key=lambda x: min(abs(center - x[0]), abs(center - x[1])),
    )
    return int(spk)


def _transcript_order_speaker_display_map(sentences: list[dict]) -> dict[int, int]:
    """Map raw ``diar_speaker_id`` / ``spk_label`` → 1-based 「说话人N」序号（按句序首次出现）。"""
    order: list[int] = []
    for s in sentences:
        raw = s.get("diar_speaker_id")
        if raw is None:
            raw = s.get("spk_label")
        if raw is None:
            continue
        rid = int(raw)
        if rid not in order:
            order.append(rid)
    return {rid: i + 1 for i, rid in enumerate(order)}


@dataclass
class FireRedAsr2SystemConfig:
    vad_model_dir: str = "pretrained_models/FireRedVAD/VAD"
    lid_model_dir: str = "pretrained_models/FireRedLID"
    asr_type: str = "aed"
    asr_model_dir: str = "pretrained_models/FireRedASR2-AED"
    punc_model_dir: str = "pretrained_models/FireRedPunc"
    vad_config: FireRedVadConfig = field(default_factory=FireRedVadConfig)
    lid_config: FireRedLidConfig = field(default_factory=FireRedLidConfig)
    asr_config: FireRedAsr2Config = field(default_factory=FireRedAsr2Config)
    punc_config: FireRedPuncConfig = field(default_factory=FireRedPuncConfig)
    itn_config: FireRedItnConfig = field(default_factory=FireRedItnConfig)
    denoise_config: FireRedDenoiserConfig = field(default_factory=FireRedDenoiserConfig)
    asr_batch_size: int = 1
    punc_batch_size: int = 1
    enable_vad: bool = True
    enable_lid: bool = True
    enable_punc: bool = True
    enable_itn: bool = False
    enable_denoise: bool = False
    enable_diarization: bool = False
    diar_model_id: str = "damo/speech_campplus_speaker-diarization_common"
    diar_model_revision: Optional[str] = None
    diar_input_mode: str = "full"
    diar_align_level: str = "segment"
    diar_refine_subsegment: bool = False
    diar_min_speaker_dur_ms: int = 400
    diar_backend: str = "modelscope_campplus"
    diar_hf_token: str = ""
    diar_spectral_f0_hz: float = 220.0
    diar_spectral_f1_hz: float = 330.0
    # ``text_labeled`` / ``text_labeled_itn`` 前缀：transcript_order=按句序首次出现的说话人编 1..K；
    # cluster_id=聚类 id+1（与 diarization_spans.speaker_id 一致，可能有缺号）。
    diar_text_labeled_mode: str = "transcript_order"
    enable_speaker_id: bool = False
    speaker_registry_path: str = ""
    speaker_match_threshold: float = 0.999
    speaker_embedder: str = "content_hash"
    speaker_embedder_model_id: str = ""
    speaker_embedder_model_revision: Optional[str] = None
    # Streaming ASR (``open_stream``): optional separate Stream-VAD checkpoint; empty → ``vad_model_dir``.
    stream_vad_model_dir: str = ""
    # Stream VAD uses CUDA only in this build; keep False on Intel XPU hosts.
    stream_vad_use_gpu: bool = False


class FireRedAsr2System:
    def __init__(self, config):
        c = config
        dev = (getattr(c.asr_config, "device", None) or "").strip()
        if dev:
            if c.enable_lid and not (getattr(c.lid_config, "device", None) or "").strip():
                c.lid_config.device = dev
            if c.enable_punc and not (getattr(c.punc_config, "device", None) or "").strip():
                c.punc_config.device = dev
        self.vad = FireRedVad.from_pretrained(c.vad_model_dir, c.vad_config) if c.enable_vad else None
        self.lid = FireRedLid.from_pretrained(c.lid_model_dir, c.lid_config) if c.enable_lid else None
        self.asr = FireRedAsr2.from_pretrained(c.asr_type, c.asr_model_dir, c.asr_config)
        self.punc = FireRedPunc.from_pretrained(c.punc_model_dir, c.punc_config) if c.enable_punc else None
        self.itn = FireRedItn(c.itn_config) if c.enable_itn else None
        self.denoiser = FireRedDenoiser(c.denoise_config) if c.enable_denoise else None
        self.config = config
        self.embedder = None
        self.speaker_registry = None
        if getattr(c, "enable_speaker_id", False):
            from fireredasr2s.firereddiar.embedder import get_speaker_embedder
            from fireredasr2s.firereddiar.enroll import SpeakerRegistry

            self.speaker_registry = SpeakerRegistry(
                getattr(c, "speaker_registry_path", "") or ""
            )
            self.embedder = get_speaker_embedder(
                getattr(c, "speaker_embedder", "content_hash"),
                model_id=(getattr(c, "speaker_embedder_model_id", "") or "").strip(),
                model_revision=getattr(
                    c, "speaker_embedder_model_revision", None
                ),
            )

    def register_speaker(self, name: str, wav_path: str) -> None:
        if self.embedder is None or self.speaker_registry is None:
            raise RuntimeError("enable_speaker_id must be True to register speakers")
        wav_np, sr = load_pcm_int16_mono(wav_path)
        wav_np, sr = prepare_asr_stack_audio(wav_np, sr)
        emb = self.embedder.embed_wav(wav_np, sr)
        self.speaker_registry.register_vector(name, emb)
        self.speaker_registry.save()

    def process(self, wav_path, uttid="tmpid"):
        wav_np, sample_rate = load_pcm_int16_mono(wav_path)

        if self.denoiser is not None:
            wav_np, sample_rate = self.denoiser.process(wav_np, sample_rate)

        wav_np, sample_rate = prepare_asr_stack_audio(wav_np, sample_rate)
        dur = wav_np.shape[0] / sample_rate

        # 1. VAD (in-memory 16 kHz tuple matches ASR/diar; supports MP3 etc. without temp wav)
        if self.config.enable_vad:
            vad_result, prob = self.vad.detect((wav_np, sample_rate))
            vad_segments = vad_result["timestamps"]
            logger.info(f"VAD: {vad_result}")
            # Offline batch: if VAD misses (e.g. non-speech synthetic), still run ASR on full clip.
            if (
                not vad_segments
                and _VAD_EMPTY_FALLBACK_MIN_DUR_S < dur <= _VAD_EMPTY_FALLBACK_MAX_DUR_S
            ):
                logger.warning(
                    "VAD returned no speech segments; falling back to full-file ASR [0, %.2fs].",
                    dur,
                )
                vad_segments = [(0.0, float(dur))]
                vad_result = {**vad_result, "timestamps": vad_segments}
        else:
            vad_segments = [(0, dur)]
            vad_result = {"timestamps": vad_segments}

        diar_spans: list[tuple[float, float, int]] | None = None
        if self.config.enable_diarization:
            try:
                from fireredasr2s.firereddiar.backends import run_diarization_backend
                from fireredasr2s.firereddiar.diar import refine_with_subsegment

                diar_spans = run_diarization_backend(
                    self.config.diar_backend,
                    wav_np,
                    sample_rate,
                    vad_segments,
                    model_id=self.config.diar_model_id,
                    model_revision=self.config.diar_model_revision,
                    hf_token=getattr(self.config, "diar_hf_token", "") or "",
                    diar_input_mode=self.config.diar_input_mode,
                    wav_path=wav_path,
                    spectral_f0_hz=float(
                        getattr(self.config, "diar_spectral_f0_hz", 220.0)
                    ),
                    spectral_f1_hz=float(
                        getattr(self.config, "diar_spectral_f1_hz", 330.0)
                    ),
                )
                if diar_spans and self.config.diar_refine_subsegment:
                    diar_spans = refine_with_subsegment(
                        wav_np, sample_rate, diar_spans
                    )
            except ImportError as e:
                logger.warning("Diarization disabled (import): %s", e)
            except Exception:
                logger.exception("Diarization setup failed")
                diar_spans = None

        diarization_spans: list[dict] = []
        if diar_spans:
            for t0, t1, spk in diar_spans:
                diarization_spans.append(
                    {
                        "start_ms": int(round(t0 * 1000)),
                        "end_ms": int(round(t1 * 1000)),
                        "speaker_id": int(spk),
                    }
                )

        # 2. VAD output to ASR input
        asr_results = []
        lid_results = []
        if int(sample_rate) != 16000:
            raise RuntimeError(
                f"internal error: expected 16 kHz after prepare_asr_stack_audio, got {sample_rate}"
            )
        batch_asr_uttid = []
        batch_asr_wav = []
        for j, (start_s, end_s) in enumerate(vad_segments):
            wav_segment = wav_np[int(start_s*sample_rate):int(end_s*sample_rate)]
            vad_uttid = f"{uttid}_s{int(start_s*1000)}_e{int(end_s*1000)}"
            batch_asr_uttid.append(vad_uttid)
            batch_asr_wav.append((sample_rate, wav_segment))
            if len(batch_asr_uttid) < self.config.asr_batch_size and j != len(vad_segments) - 1:
                continue

            # 3. ASR
            batch_asr_results = self.asr.transcribe(batch_asr_uttid, batch_asr_wav)
            logger.info(f"ASR: {batch_asr_results}")

            if self.config.enable_lid:
                batch_lid_results = self.lid.process(batch_asr_uttid, batch_asr_wav)
                logger.info(f"LID: {batch_lid_results}")
            else:
                # Note: The original batch size is used here to ensure alignment with the initial number of ASR results
                batch_lid_results = [None] * len(batch_asr_results)

            # Synchronously traverse and filter to ensure that asr_results and lid_results always maintain a one-to-one correspondence
            for a_res, l_res in zip(batch_asr_results, batch_lid_results):
                text = a_res.get("text", "").strip()
                # Filter out <blank>, <sil> and completely empty strings ""
                if not text or re.search(r"(<blank>)|(<sil>)", text):
                    continue
                asr_results.append(a_res)
                lid_results.append(l_res)

            batch_asr_uttid = []
            batch_asr_wav = []

        # 4. ASR output to Postprocess input
        if self.config.enable_punc:
            punc_results = []
            batch_asr_text = []
            batch_asr_uttid = []
            batch_asr_timestamp = []
            for j, asr_result in enumerate(asr_results):
                batch_asr_text.append(asr_result["text"])
                batch_asr_uttid.append(asr_result["uttid"])
                if self.config.asr_config.return_timestamp:
                    batch_asr_timestamp.append(asr_result.get("timestamp", []))
                elif "timestamp" in asr_result:
                    batch_asr_timestamp.append(asr_result["timestamp"])
                if len(batch_asr_text) < self.config.punc_batch_size and j != len(asr_results) - 1:
                    continue

                # 5. Punc
                if self.config.asr_config.return_timestamp:
                    batch_punc_results = self.punc.process_with_timestamp(batch_asr_timestamp, batch_asr_uttid)
                else:
                    batch_punc_results = self.punc.process(batch_asr_text, batch_asr_uttid)
                logger.info(f"Punc: {batch_punc_results}")

                punc_results.extend(batch_punc_results)
                batch_asr_text = []
                batch_asr_uttid = []
                batch_asr_timestamp = []
        else:
            punc_results = asr_results

        # 6. Put all together & Format
        seg_key_to_idx = _vad_seg_key_to_index(vad_result["timestamps"])
        sentences = []
        words = []
        for asr_result, punc_result, lid_result in zip(asr_results, punc_results, lid_results):
            assert asr_result["uttid"] == punc_result["uttid"], f"fix code: {asr_result} | {punc_result}"
            start_ms, end_ms = asr_result["uttid"].split("_")[-2:]
            assert start_ms.startswith("s") and end_ms.startswith("e")
            start_ms, end_ms = int(start_ms[1:]), int(end_ms[1:])
            vad_segment_idx = _vad_segment_idx_from_uttid(asr_result["uttid"], seg_key_to_idx)
            if (
                diar_spans
                and self.config.diar_align_level == "word"
                and self.config.asr_config.return_timestamp
            ):
                sub = try_word_diar_sentences(
                    asr_result=asr_result,
                    punc_result=punc_result,
                    lid_result=lid_result,
                    diar_spans=diar_spans,
                    vad_segment_idx=vad_segment_idx,
                    min_speaker_dur_ms=self.config.diar_min_speaker_dur_ms,
                    enable_punc=self.config.enable_punc,
                    punc_model=self.punc,
                )
                if sub is not None:
                    sentences.extend(sub)
                    if "timestamp" in asr_result:
                        for w, s, e in asr_result["timestamp"]:
                            word = {
                                "start_ms": int(s * 1000 + start_ms),
                                "end_ms": int(e * 1000 + start_ms),
                                "text": w,
                            }
                            words.append(word)
                    continue

            if self.config.asr_config.return_timestamp:
                sub_sentences = []
                if self.config.enable_punc:
                    for i, punc_sent in enumerate(punc_result["punc_sentences"]):
                        start = start_ms + int(punc_sent["start_s"]*1000)
                        end = start_ms + int(punc_sent["end_s"]*1000)
                        if i == 0:
                            start = start_ms
                        if i == len(punc_result["punc_sentences"]) - 1:
                            end = end_ms
                        sub_sentence = {
                            "start_ms": start,
                            "end_ms": end,
                            "text": punc_sent["punc_text"],
                            "asr_confidence": asr_result["confidence"],
                            "lang": None,
                            "lang_confidence": 0,
                            "vad_segment_idx": vad_segment_idx,
                        }
                        if lid_result:
                            sub_sentence["lang"] = lid_result["lang"]
                            sub_sentence["lang_confidence"] = lid_result["confidence"]
                        sub_sentences.append(sub_sentence)
                else:
                    sub_sentences = [{
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "text": asr_result["text"],
                        "asr_confidence": asr_result["confidence"],
                        "lang": None,
                        "lang_confidence": 0,
                        "vad_segment_idx": vad_segment_idx,
                    }]
                sentences.extend(sub_sentences)
            else:
                text = punc_result["punc_text"] if self.config.enable_punc else asr_result["text"]
                sentence = {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                    "asr_confidence": asr_result["confidence"],
                    "lang": None,
                    "lang_confidence": 0,
                    "vad_segment_idx": vad_segment_idx,
                }
                if lid_result:
                    sentence["lang"] = lid_result["lang"]
                    sentence["lang_confidence"] = lid_result["confidence"]
                sentences.append(sentence)
            
            if "timestamp" in asr_result:
                for w, s, e in asr_result["timestamp"]:
                    word = {"start_ms": int(s*1000+start_ms), "end_ms":int(e*1000+start_ms), "text": w}
                    words.append(word)

        vad_segments_ms = [(int(s*1000), int(e*1000)) for s, e in vad_result["timestamps"]]

        prev_vad_idx = None
        spk_turn = -1
        for s in sentences:
            vidx = s["vad_segment_idx"]
            if vidx != prev_vad_idx:
                spk_turn += 1
                prev_vad_idx = vidx
            s["spk_turn"] = spk_turn

        if (
            self.embedder is not None
            and self.speaker_registry is not None
            and self.speaker_registry.count() > 0
        ):
            th = float(getattr(self.config, "speaker_match_threshold", 0.999))
            for s in sentences:
                s0 = int(max(0, s["start_ms"]) / 1000.0 * sample_rate)
                s1 = int(max(0, s["end_ms"]) / 1000.0 * sample_rate)
                if s1 <= s0:
                    s1 = min(len(wav_np), s0 + max(1, int(0.05 * sample_rate)))
                s1 = min(len(wav_np), max(s0 + 1, s1))
                seg = wav_np[s0:s1]
                if seg.size == 0:
                    s["enrolled_speaker"] = None
                    s["enrolled_similarity"] = 0.0
                    continue
                emb = self.embedder.embed_wav(seg, sample_rate)
                name, sim = self.speaker_registry.best_match(emb, th)
                s["enrolled_speaker"] = name
                s["enrolled_similarity"] = float(sim)

        diar_speaker_id_stats: dict | None = None
        if diar_spans:
            for s in sentences:
                if s.get("word_diar_spk"):
                    continue
                spk = _speaker_by_overlap(s["start_ms"], s["end_ms"], diar_spans)
                s["diar_speaker_id"] = spk
                s["spk_label"] = spk
            b = (self.config.diar_backend or "modelscope_campplus").strip()
            if b.lower() in ("rttm_sidecar", "rttm", "sidecar_rttm"):
                speaker_label_note = (
                    f"diarization backend={b!r}: spans loaded from sidecar .rttm next to the wav "
                    "(reference labels for tests/offline eval, not neural diarization). "
                    "说话人N = spk_label + 1."
                )
            elif b.lower() in ("spectral_tone_pair", "spectral_tone", "tone_pair"):
                f0 = float(getattr(self.config, "diar_spectral_f0_hz", 220.0))
                f1 = float(getattr(self.config, "diar_spectral_f1_hz", 330.0))
                speaker_label_note = (
                    f"diarization backend={b!r}: dual-tone energy ({f0:.1f} Hz vs {f1:.1f} Hz) — "
                    "for alternating single-frequency / proxy dialog audio; use modelscope_campplus "
                    "for natural speech. 说话人N = spk_label + 1."
                )
            else:
                speaker_label_note = (
                    f"diarization backend={b!r}: diar_speaker_id / spk_label are clustering ids "
                    "(not ground-truth identity). 说话人N = spk_label + 1."
                )
            all_c = sorted({int(sp[2]) for sp in diar_spans})
            used_c = sorted(
                {
                    int(s["diar_speaker_id"])
                    for s in sentences
                    if s.get("diar_speaker_id") is not None
                }
            )
            only_spans = sorted(set(all_c) - set(used_c))
            diar_speaker_id_stats = {
                "n_clusters_in_diar_spans": len(all_c),
                "cluster_ids_in_diar_spans": all_c,
                "cluster_ids_in_sentences": used_c,
                "cluster_ids_in_diar_only": only_spans,
            }
            if only_spans:
                speaker_label_note += (
                    " （说明）diar 时间轴上共 "
                    f"{len(all_c)} 个聚类（id {all_c}）；下列 id 未与任何有效 ASR 句重叠 "
                    f"（常见于首段短声/气口未形成独立句）：{only_spans}。"
                    "完整时段请对照 diarization_spans。"
                )
            _m = (getattr(self.config, "diar_text_labeled_mode", "") or "transcript_order").strip().lower()
            if _m in ("transcript_order", "first_seen", "utterance_order"):
                speaker_label_note += (
                    " text_labeled 中「说话人N」按转写句序 **首次出现** 编号为 1..K（便于阅读）；"
                    "与聚类 id 对应关系见各句 diar_speaker_id。"
                )
            elif only_spans:
                speaker_label_note += (
                    " text_labeled 中「说话人N」= 聚类 id + 1，故可能缺某些编号。"
                )
        else:
            for s in sentences:
                s["spk_label"] = s["spk_turn"] % 2
            if self.config.enable_diarization:
                speaker_label_note = (
                    "Diarization was requested but unavailable or skipped (e.g. speech < 5s, "
                    "missing modelscope, or pipeline error). spk_label uses VAD turn heuristic "
                    "(alternating 1/2 via 说话人N = spk_label + 1)."
                )
            else:
                speaker_label_note = (
                    "spk_label / 说话人N are heuristics from VAD segment order (alternating 1/2), "
                    "not speaker diarization. Set enable_diarization with "
                    "modelscope_campplus, spectral_tone_pair, or rttm_sidecar."
                )

        utterance_enrolled_speaker = None
        utterance_enrolled_similarity = 0.0
        if (
            self.config.enable_speaker_id
            and self.embedder is not None
            and self.speaker_registry is not None
            and self.speaker_registry.count() > 0
        ):
            th_u = float(getattr(self.config, "speaker_match_threshold", 0.999))
            utt_emb = self.embedder.embed_wav(wav_np, sample_rate)
            utterance_enrolled_speaker, utterance_enrolled_similarity = (
                self.speaker_registry.best_match(utt_emb, th_u)
            )
            utterance_enrolled_similarity = float(utterance_enrolled_similarity)

        text = "".join(s["text"] for s in sentences)
        # Add space after English punctuation when followed by a letter
        text = re.sub(r'([.,!?])\s*([a-zA-Z])', r'\1 \2', text)
        _tl_mode = (getattr(self.config, "diar_text_labeled_mode", "") or "transcript_order").strip().lower()
        _disp_map = (
            _transcript_order_speaker_display_map(sentences)
            if _tl_mode in ("transcript_order", "first_seen", "utterance_order")
            else None
        )

        def _labeled_num(s: dict) -> int:
            if _disp_map is None:
                return int(s["spk_label"]) + 1
            raw = s.get("diar_speaker_id")
            if raw is None:
                raw = int(s["spk_label"])
            else:
                raw = int(raw)
            return int(_disp_map.get(raw, raw + 1))

        text_labeled = "".join(f"[说话人{_labeled_num(s)}]{s['text']}" for s in sentences)
        text_labeled = re.sub(r'([.,!?])\s*([a-zA-Z])', r'\1 \2', text_labeled)

        result = {
            "uttid": uttid,
            "text": text,
            "text_labeled": text_labeled,
            "sentences": sentences,
            "vad_segments_ms": vad_segments_ms,
            "diarization_spans": diarization_spans,
            "dur_s": dur,
            "words": words,
            "wav_path": wav_path,
            "speaker_label_note": speaker_label_note,
        }
        if diar_speaker_id_stats is not None:
            result["diar_speaker_id_stats"] = diar_speaker_id_stats
        if self.config.enable_speaker_id:
            result["utterance_enrolled_speaker"] = utterance_enrolled_speaker
            result["utterance_enrolled_similarity"] = utterance_enrolled_similarity
        if self.denoiser is not None:
            result["denoise_backend"] = self.config.denoise_config.backend

        if self.itn is not None:
            for s in sentences:
                s["text_itn"] = self.itn.process(s["text"])
            text_itn = "".join(s["text_itn"] for s in sentences)
            text_itn = re.sub(r'([.,!?])\s*([a-zA-Z])', r'\1 \2', text_itn)
            text_labeled_itn = "".join(
                f"[说话人{_labeled_num(s)}]{s['text_itn']}" for s in sentences
            )
            text_labeled_itn = re.sub(r'([.,!?])\s*([a-zA-Z])', r'\1 \2', text_labeled_itn)
            result["text_itn"] = text_itn
            result["text_labeled_itn"] = text_labeled_itn

        return result

    def process_pcm_segment(
        self,
        wav_np: np.ndarray,
        sample_rate: int,
        seg_uttid: str,
        segment_start_ms: int,
        segment_end_ms: int,
    ) -> dict[str, Any]:
        """ASR → LID → Punc → optional ITN for one in-memory segment (mono). Resamples to 16 kHz.

        Used by streaming sessions after online VAD endpoints. No diarization / RTTM.
        """
        if self.config.asr_type != "aed":
            raise ValueError("process_pcm_segment supports asr_type='aed' only.")
        wav_np = np.asarray(wav_np)
        if wav_np.ndim != 1:
            raise ValueError("process_pcm_segment expects mono 1-D audio")
        if self.denoiser is not None:
            wav_np, sample_rate = self.denoiser.process(wav_np, int(sample_rate))
        wav_np, sample_rate = prepare_asr_stack_audio(wav_np, int(sample_rate))
        if int(sample_rate) != 16000:
            raise RuntimeError(f"internal error: expected 16 kHz after prepare_asr_stack_audio, got {sample_rate}")
        dur = float(wav_np.shape[0]) / float(sample_rate)
        if dur <= 0.0:
            return {
                "uttid": seg_uttid,
                "text": "",
                "text_labeled": "",
                "sentences": [],
                "words": [],
                "vad_segments_ms": [(segment_start_ms, segment_end_ms)],
                "diarization_spans": [],
                "dur_s": 0.0,
                "speaker_label_note": "streaming pcm segment (empty)",
                "segment_start_ms": segment_start_ms,
                "segment_end_ms": segment_end_ms,
            }

        vad_result = {"timestamps": [(0.0, dur)]}
        seg_key_to_idx = _vad_seg_key_to_index(vad_result["timestamps"])
        batch_asr_results = self.asr.transcribe([seg_uttid], [(sample_rate, wav_np)])
        if self.config.enable_lid:
            batch_lid_results = self.lid.process([seg_uttid], [(sample_rate, wav_np)])
        else:
            batch_lid_results = [None] * len(batch_asr_results)

        asr_results: list[dict] = []
        lid_results: list[Any] = []
        for a_res, l_res in zip(batch_asr_results, batch_lid_results):
            text = a_res.get("text", "").strip()
            if not text or re.search(r"(<blank>)|(<sil>)", text):
                continue
            asr_results.append(a_res)
            lid_results.append(l_res)

        if self.config.enable_punc and asr_results:
            punc_results: list[dict] = []
            batch_asr_text: list[str] = []
            batch_asr_uttid: list[str] = []
            batch_asr_timestamp: list[Any] = []
            for j, asr_result in enumerate(asr_results):
                batch_asr_text.append(asr_result["text"])
                batch_asr_uttid.append(asr_result["uttid"])
                if self.config.asr_config.return_timestamp:
                    batch_asr_timestamp.append(asr_result.get("timestamp", []))
                elif "timestamp" in asr_result:
                    batch_asr_timestamp.append(asr_result["timestamp"])
                if len(batch_asr_text) < self.config.punc_batch_size and j != len(asr_results) - 1:
                    continue
                if self.config.asr_config.return_timestamp:
                    batch_punc_results = self.punc.process_with_timestamp(
                        batch_asr_timestamp, batch_asr_uttid
                    )
                else:
                    batch_punc_results = self.punc.process(batch_asr_text, batch_asr_uttid)
                punc_results.extend(batch_punc_results)
                batch_asr_text = []
                batch_asr_uttid = []
                batch_asr_timestamp = []
        elif asr_results:
            punc_results = asr_results
        else:
            punc_results = []

        diar_spans = None
        sentences: list[dict] = []
        words: list[dict] = []
        for asr_result, punc_result, lid_result in zip(asr_results, punc_results, lid_results):
            assert asr_result["uttid"] == punc_result["uttid"]
            start_ms, end_ms = asr_result["uttid"].split("_")[-2:]
            assert start_ms.startswith("s") and end_ms.startswith("e")
            start_ms_i, end_ms_i = int(start_ms[1:]), int(end_ms[1:])
            vad_segment_idx = _vad_segment_idx_from_uttid(asr_result["uttid"], seg_key_to_idx)
            if (
                diar_spans
                and self.config.diar_align_level == "word"
                and self.config.asr_config.return_timestamp
            ):
                sub = try_word_diar_sentences(
                    asr_result=asr_result,
                    punc_result=punc_result,
                    lid_result=lid_result,
                    diar_spans=diar_spans,
                    vad_segment_idx=vad_segment_idx,
                    min_speaker_dur_ms=self.config.diar_min_speaker_dur_ms,
                    enable_punc=self.config.enable_punc,
                    punc_model=self.punc,
                )
                if sub is not None:
                    sentences.extend(sub)
                    if "timestamp" in asr_result:
                        for w, s, e in asr_result["timestamp"]:
                            words.append(
                                {
                                    "start_ms": int(s * 1000 + start_ms_i),
                                    "end_ms": int(e * 1000 + start_ms_i),
                                    "text": w,
                                }
                            )
                    continue

            if self.config.asr_config.return_timestamp:
                sub_sentences: list[dict] = []
                if self.config.enable_punc:
                    for i, punc_sent in enumerate(punc_result["punc_sentences"]):
                        start = start_ms_i + int(punc_sent["start_s"] * 1000)
                        end = start_ms_i + int(punc_sent["end_s"] * 1000)
                        if i == 0:
                            start = start_ms_i
                        if i == len(punc_result["punc_sentences"]) - 1:
                            end = end_ms_i
                        sub_sentence = {
                            "start_ms": start,
                            "end_ms": end,
                            "text": punc_sent["punc_text"],
                            "asr_confidence": asr_result["confidence"],
                            "lang": None,
                            "lang_confidence": 0,
                            "vad_segment_idx": vad_segment_idx,
                        }
                        if lid_result:
                            sub_sentence["lang"] = lid_result["lang"]
                            sub_sentence["lang_confidence"] = lid_result["confidence"]
                        sub_sentences.append(sub_sentence)
                else:
                    sub_sentences = [
                        {
                            "start_ms": start_ms_i,
                            "end_ms": end_ms_i,
                            "text": asr_result["text"],
                            "asr_confidence": asr_result["confidence"],
                            "lang": None,
                            "lang_confidence": 0,
                            "vad_segment_idx": vad_segment_idx,
                        }
                    ]
                sentences.extend(sub_sentences)
            else:
                text = punc_result["punc_text"] if self.config.enable_punc else asr_result["text"]
                sentence = {
                    "start_ms": start_ms_i,
                    "end_ms": end_ms_i,
                    "text": text,
                    "asr_confidence": asr_result["confidence"],
                    "lang": None,
                    "lang_confidence": 0,
                    "vad_segment_idx": vad_segment_idx,
                }
                if lid_result:
                    sentence["lang"] = lid_result["lang"]
                    sentence["lang_confidence"] = lid_result["confidence"]
                sentences.append(sentence)

            if "timestamp" in asr_result:
                for w, s, e in asr_result["timestamp"]:
                    words.append(
                        {
                            "start_ms": int(s * 1000 + start_ms_i),
                            "end_ms": int(e * 1000 + start_ms_i),
                            "text": w,
                        }
                    )

        prev_vad_idx = None
        spk_turn = -1
        for s in sentences:
            vidx = s["vad_segment_idx"]
            if vidx != prev_vad_idx:
                spk_turn += 1
                prev_vad_idx = vidx
            s["spk_turn"] = spk_turn

        if (
            self.embedder is not None
            and self.speaker_registry is not None
            and self.speaker_registry.count() > 0
        ):
            th = float(getattr(self.config, "speaker_match_threshold", 0.999))
            for s in sentences:
                s0 = int(max(0, s["start_ms"]) / 1000.0 * sample_rate)
                s1 = int(max(0, s["end_ms"]) / 1000.0 * sample_rate)
                if s1 <= s0:
                    s1 = min(len(wav_np), s0 + max(1, int(0.05 * sample_rate)))
                s1 = min(len(wav_np), max(s0 + 1, s1))
                seg = wav_np[s0:s1]
                if seg.size == 0:
                    s["enrolled_speaker"] = None
                    s["enrolled_similarity"] = 0.0
                    continue
                emb = self.embedder.embed_wav(seg, sample_rate)
                name, sim = self.speaker_registry.best_match(emb, th)
                s["enrolled_speaker"] = name
                s["enrolled_similarity"] = float(sim)

        for s in sentences:
            s["spk_label"] = s["spk_turn"] % 2
        speaker_label_note = (
            "Streaming pcm segment: spk_label alternates by VAD-local turn (no diarization)."
        )

        utterance_enrolled_speaker = None
        utterance_enrolled_similarity = 0.0
        if (
            self.config.enable_speaker_id
            and self.embedder is not None
            and self.speaker_registry is not None
            and self.speaker_registry.count() > 0
        ):
            th_u = float(getattr(self.config, "speaker_match_threshold", 0.999))
            utt_emb = self.embedder.embed_wav(wav_np, sample_rate)
            utterance_enrolled_speaker, utterance_enrolled_similarity = (
                self.speaker_registry.best_match(utt_emb, th_u)
            )
            utterance_enrolled_similarity = float(utterance_enrolled_similarity)

        text = "".join(s["text"] for s in sentences)
        text = re.sub(r"([.,!?])\s*([a-zA-Z])", r"\1 \2", text)
        _tl_mode = (getattr(self.config, "diar_text_labeled_mode", "") or "transcript_order").strip().lower()
        _disp_map = (
            _transcript_order_speaker_display_map(sentences)
            if _tl_mode in ("transcript_order", "first_seen", "utterance_order")
            else None
        )

        def _labeled_num(s: dict) -> int:
            if _disp_map is None:
                return int(s["spk_label"]) + 1
            raw = s.get("diar_speaker_id")
            if raw is None:
                raw = int(s["spk_label"])
            else:
                raw = int(raw)
            return int(_disp_map.get(raw, raw + 1))

        text_labeled = "".join(f"[说话人{_labeled_num(s)}]{s['text']}" for s in sentences)
        text_labeled = re.sub(r"([.,!?])\s*([a-zA-Z])", r"\1 \2", text_labeled)

        result: dict[str, Any] = {
            "uttid": seg_uttid,
            "text": text,
            "text_labeled": text_labeled,
            "sentences": sentences,
            "vad_segments_ms": [(segment_start_ms, segment_end_ms)],
            "diarization_spans": [],
            "dur_s": dur,
            "words": words,
            "wav_path": "<stream_pcm>",
            "speaker_label_note": speaker_label_note,
            "segment_start_ms": segment_start_ms,
            "segment_end_ms": segment_end_ms,
        }
        if self.config.enable_speaker_id:
            result["utterance_enrolled_speaker"] = utterance_enrolled_speaker
            result["utterance_enrolled_similarity"] = utterance_enrolled_similarity
        if self.denoiser is not None:
            result["denoise_backend"] = self.config.denoise_config.backend

        if self.itn is not None:
            for s in sentences:
                s["text_itn"] = self.itn.process(s["text"])
            text_itn = "".join(s["text_itn"] for s in sentences)
            text_itn = re.sub(r"([.,!?])\s*([a-zA-Z])", r"\1 \2", text_itn)
            text_labeled_itn = "".join(
                f"[说话人{_labeled_num(s)}]{s['text_itn']}" for s in sentences
            )
            text_labeled_itn = re.sub(r"([.,!?])\s*([a-zA-Z])", r"\1 \2", text_labeled_itn)
            result["text_itn"] = text_itn
            result["text_labeled_itn"] = text_labeled_itn

        return result

    def open_stream(
        self,
        uttid_prefix: str = "live",
        *,
        emit_vad_boundaries: bool = False,
        max_pcm_duration_s: float | None = None,
        telemetry: bool = False,
    ):
        """Create a streaming ASR session (online Stream-VAD + ``process_pcm_segment`` per utterance).

        Currently **AED only**. Diarization is not applied; enable_speaker_id on segments is supported.

        If ``emit_vad_boundaries`` is True, each ``push_pcm_int16_mono`` / ``finalize`` may emit
        ``vad_speech_start`` events (before ``segment_final``) for full-duplex / barge-in hooks.

        ``max_pcm_duration_s`` caps in-session PCM history (see ``FireRedAsr2StreamSession``).
        ``telemetry`` enables INFO logs with per-segment ``process_pcm_segment`` wall time (ms).
        """
        if self.config.asr_type != "aed":
            raise ValueError("open_stream supports asr_type='aed' only.")
        from fireredasr2s.stream_session import FireRedAsr2StreamSession

        return FireRedAsr2StreamSession(
            self,
            uttid_prefix=uttid_prefix,
            emit_vad_boundaries=emit_vad_boundaries,
            max_pcm_duration_s=max_pcm_duration_s,
            telemetry=telemetry,
        )

    def open_full_duplex_stream(
        self,
        uttid_prefix: str = "live",
        *,
        verbose_vad: bool = False,
        max_pcm_duration_s: float | None = None,
        telemetry: bool = False,
    ):
        """Full-duplex **orchestration** on top of streaming ASR (AED only).

        While ``begin_local_playback()`` is active (simulating TTS / local prompt playback),
        a detected user speech onset yields a ``barge_in`` event so the app can cancel playback.

        This repo does **not** perform acoustic echo cancellation; feed echo-cancelled mic PCM or
        mute the uplink during playback if you cannot tolerate false ``barge_in`` / ASR echo.
        """
        if self.config.asr_type != "aed":
            raise ValueError("open_full_duplex_stream supports asr_type='aed' only.")
        from fireredasr2s.full_duplex_stream import FireRedFullDuplexStreamSession

        return FireRedFullDuplexStreamSession(
            self,
            uttid_prefix=uttid_prefix,
            verbose_vad=verbose_vad,
            max_pcm_duration_s=max_pcm_duration_s,
            telemetry=telemetry,
        )
