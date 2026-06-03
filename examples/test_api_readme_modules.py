# Run from repository root: PYTHONPATH=. python examples/test_api_readme_modules.py
# Covers README: VAD, Stream VAD, mVAD, LID, Punc, ASR System (after #### VAD).
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def ok(name: str):
    print(f"\n[PASS] {name}")


def fail(name: str, e: BaseException):
    print(f"\n[FAIL] {name}: {e}")
    traceback.print_exc()


def main():
    wav_zh = "assets/hello_zh.wav"
    wav_en = "assets/hello_en.wav"
    wav_evt = "assets/event.wav"

    # ---- VAD ----
    try:
        from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig

        vad_config = FireRedVadConfig(
            use_gpu=False,
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
        result, _probs = vad.detect(wav_zh)
        print("VAD:", result)
        assert "timestamps" in result
        ok("VAD")
    except Exception as e:
        fail("VAD", e)

    # ---- Stream VAD ----
    try:
        from fireredasr2s.fireredvad import FireRedStreamVad, FireRedStreamVadConfig

        vad_config = FireRedStreamVadConfig(
            use_gpu=False,
            smooth_window_size=5,
            speech_threshold=0.4,
            pad_start_frame=5,
            min_speech_frame=8,
            max_speech_frame=2000,
            min_silence_frame=20,
            chunk_max_frame=30000,
        )
        stream_vad = FireRedStreamVad.from_pretrained(
            "pretrained_models/FireRedVAD/Stream-VAD", vad_config
        )
        _frame_results, result = stream_vad.detect_full(wav_zh)
        print("Stream VAD:", result)
        assert "timestamps" in result
        ok("Stream VAD")
    except Exception as e:
        fail("Stream VAD", e)

    # ---- mVAD (AED) ----
    try:
        from fireredasr2s.fireredvad import FireRedAed, FireRedAedConfig

        aed_config = FireRedAedConfig(
            use_gpu=False,
            smooth_window_size=5,
            speech_threshold=0.4,
            singing_threshold=0.5,
            music_threshold=0.5,
            min_event_frame=20,
            max_event_frame=2000,
            min_silence_frame=20,
            merge_silence_frame=0,
            extend_speech_frame=0,
            chunk_max_frame=30000,
        )
        aed = FireRedAed.from_pretrained("pretrained_models/FireRedVAD/AED", aed_config)
        result, _probs = aed.detect(wav_evt)
        print("mVAD:", result)
        assert "event2timestamps" in result
        ok("mVAD (AED)")
    except Exception as e:
        fail("mVAD (AED)", e)

    # ---- LID ----
    try:
        from fireredasr2s.fireredlid import FireRedLid, FireRedLidConfig

        batch_uttid = ["hello_zh", "hello_en"]
        batch_wav_path = [wav_zh, wav_en]
        config = FireRedLidConfig(use_gpu=False, use_half=False)
        model = FireRedLid.from_pretrained("pretrained_models/FireRedLID", config)
        results = model.process(batch_uttid, batch_wav_path)
        print("LID:", results)
        assert len(results) == 2 and results[0].get("lang")
        ok("LID")
    except Exception as e:
        fail("LID", e)

    # ---- Punc ----
    try:
        from fireredasr2s.fireredpunc.punc import FireRedPunc, FireRedPuncConfig

        config = FireRedPuncConfig(use_gpu=False)
        model = FireRedPunc.from_pretrained("pretrained_models/FireRedPunc", config)
        batch_text = ["你好世界", "Hello world"]
        results = model.process(batch_text)
        print("Punc:", results)
        assert len(results) == 2 and "punc_text" in results[0]
        ok("Punc")
    except Exception as e:
        fail("Punc", e)

    # ---- ASR System ----
    try:
        from fireredasr2s.fireredasr2 import FireRedAsr2Config
        from fireredasr2s.fireredlid import FireRedLidConfig
        from fireredasr2s.fireredpunc import FireRedPuncConfig
        from fireredasr2s.fireredvad import FireRedVadConfig
        from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig

        vad_config = FireRedVadConfig(
            use_gpu=False,
            smooth_window_size=5,
            speech_threshold=0.4,
            min_speech_frame=20,
            max_speech_frame=2000,
            min_silence_frame=20,
            merge_silence_frame=0,
            extend_speech_frame=0,
            chunk_max_frame=30000,
        )
        lid_config = FireRedLidConfig(use_gpu=False, use_half=False)
        asr_config = FireRedAsr2Config(
            use_gpu=False,
            use_half=False,
            beam_size=3,
            nbest=1,
            decode_max_len=0,
            softmax_smoothing=1.25,
            aed_length_penalty=0.6,
            eos_penalty=1.0,
            return_timestamp=True,
        )
        punc_config = FireRedPuncConfig(use_gpu=False)

        asr_system_config = FireRedAsr2SystemConfig(
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
        )
        asr_system = FireRedAsr2System(asr_system_config)

        for wav_path, uttid in zip([wav_zh, wav_en], ["hello_zh", "hello_en"]):
            result = asr_system.process(wav_path, uttid)
            print("ASR System:", uttid, result.get("text"), "...")
        ok("ASR System")
    except Exception as e:
        fail("ASR System", e)


if __name__ == "__main__":
    main()
