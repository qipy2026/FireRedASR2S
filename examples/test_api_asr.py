# Run from repository root: PYTHONPATH=. python examples/test_api_asr.py
# Mirrors README "Usage of Each Module" -> Python API Usage -> ASR (FireRedASR2-AED).
from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

batch_uttid = ["hello_zh"]
batch_wav_path = ["assets/hello_zh.wav"]

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
model = FireRedAsr2.from_pretrained("aed", "pretrained_models/FireRedASR2-AED", asr_config)
results = model.transcribe(batch_uttid, batch_wav_path)
print(results)
