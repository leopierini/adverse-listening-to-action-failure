import base64
import io

import numpy as np
import soundfile as sf

import pipeline


def test_flac_to_wav_b64_roundtrip(tiny_flac):
    b64 = pipeline.flac_to_wav_b64(tiny_flac)
    raw = base64.b64decode(b64)
    audio, sr = sf.read(io.BytesIO(raw))
    assert sr == 16000
    assert len(audio) == 8000  # 0.5 s at 16 kHz


def test_build_omni_messages_structure():
    msgs = pipeline.build_omni_messages("BASE64DATA", "SYS PROMPT", fmt="wav")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "SYS PROMPT"
    assert msgs[1]["role"] == "user"
    part = msgs[1]["content"][0]
    assert part["type"] == "input_audio"
    assert part["input_audio"]["data"] == "BASE64DATA"
    assert part["input_audio"]["format"] == "wav"
