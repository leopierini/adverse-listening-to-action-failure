import numpy as np

import corrupt_audio as ca


def test_degrade_babble_hits_target_snr():
    sr = 16000
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    babble = rng.standard_normal(sr * 4).astype(np.float32)  # 4 s "babble"
    out = ca.degrade_babble(audio, sr, 10.0, rng, babble)
    added = out - audio
    sig_p = float(np.mean(audio ** 2))
    noise_p = float(np.mean(added ** 2))
    achieved_snr = 10 * np.log10(sig_p / noise_p)
    assert abs(achieved_snr - 10.0) < 0.5  # mixing math is correct


def test_degrade_babble_wraps_when_track_short():
    sr = 16000
    rng = np.random.default_rng(1)
    audio = np.ones(sr * 2, dtype=np.float32) * 0.3   # 2 s
    babble = rng.standard_normal(sr // 2).astype(np.float32)  # only 0.5 s
    out = ca.degrade_babble(audio, sr, 20.0, rng, babble)
    assert out.shape == audio.shape  # wrapped/tiled, no crash


def test_degrade_babble_is_deterministic_given_seed():
    sr = 16000
    audio = np.ones(sr, dtype=np.float32) * 0.2
    babble = np.random.default_rng(0).standard_normal(sr * 3).astype(np.float32)
    a = ca.degrade_babble(audio, sr, 10.0, np.random.default_rng(7), babble)
    b = ca.degrade_babble(audio, sr, 10.0, np.random.default_rng(7), babble)
    assert np.allclose(a, b)


def test_babble_in_degradation_list():
    assert "babble" in ca.DEGRADATIONS
