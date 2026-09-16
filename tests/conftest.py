"""Shared pytest fixtures. Puts scripts/ on sys.path so tests can import the
pipeline modules directly, and provides a tiny on-disk FLAC + a SLURP-shaped
gold dict for the audio/annotation tests."""
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture
def tiny_flac(tmp_path):
    """0.5 s of 16 kHz mono tone written as FLAC; returns the path."""
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    p = tmp_path / "tone.flac"
    sf.write(p, audio, sr, subtype="PCM_16", format="FLAC")
    return str(p)


@pytest.fixture
def sample_gold():
    """A SLURP-shaped gold entry: tokens + entities with span indices."""
    return {
        "slurp_id": 999,
        "sentence": "set an alarm for sarah at seven am",
        "tokens": [
            {"surface": "set"}, {"surface": "an"}, {"surface": "alarm"},
            {"surface": "for"}, {"surface": "sarah"}, {"surface": "at"},
            {"surface": "seven"}, {"surface": "am"},
        ],
        "entities": [
            {"type": "person", "span": [4]},
            {"type": "time", "span": [6, 7]},
        ],
    }
