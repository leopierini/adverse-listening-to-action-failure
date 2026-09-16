"""Apply WildASR-style audio degradations to SLURP audio.

For each input (audio_file, degradation, snr_db) tuple, produces one output file
at: <output-dir>/<degradation>/snr<snr>/<original_filename>

Five degradations:
    noise      — additive Gaussian noise at target SNR
    reverb     — convolution with a synthetic room impulse response (T60 mapped from SNR)
    farfield   — low-pass filter + amplitude attenuation
    codec      — encode → low-bitrate Opus → decode (lossy compression)
    clipping   — hard-clip waveform to ±threshold

Three SNR levels: 20 dB (mild), 10 dB (moderate), 0 dB (severe).

All operations are deterministic given --seed (audio output is byte-stable on re-run).

Usage:
    python corrupt_audio.py \\
        --dataset SLURP/dataset/slurp/devel.jsonl \\
        --audio-root SLURP/audio/slurp_real \\
        --scenarios calendar,alarm,lists \\
        --output-dir SLURP/audio/corrupted \\
        --degradations noise,reverb,farfield,codec,clipping \\
        --snr-levels 20,10,0 \\
        --seed 42
"""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
import pyroomacoustics as pra
from tqdm import tqdm

# ─── Constants ──────────────────────────────────────────────────────────────

DEFAULT_AUDIO_REAL = "dataset/audio"   # SLURP/ removed 2026-06-02 — frozen 416 source
DEFAULT_AUDIO_SYNTH = "dataset/audio"
TARGET_SAMPLE_RATE = 16000  # SLURP audio is 16 kHz; corrupted outputs match.

# SNR-to-parameter mappings, per degradation.
# 20 dB = mild, 10 dB = moderate, 0 dB = severe.
REVERB_T60_BY_SNR = {20: 0.3, 10: 0.6, 0: 1.2}                      # seconds
FARFIELD_CUTOFF_BY_SNR = {20: 6000, 10: 4000, 0: 2500}              # Hz
FARFIELD_ATTEN_BY_SNR = {20: 0.7, 10: 0.4, 0: 0.15}                 # linear gain
CODEC_BITRATE_BY_SNR = {20: 32, 10: 16, 0: 8}                       # kbps Opus
CLIPPING_THRESHOLD_BY_SNR = {20: 0.5, 10: 0.25, 0: 0.1}             # fraction of peak

DEGRADATIONS = ["noise", "reverb", "farfield", "codec", "clipping", "babble"]


# ─── Helpers ────────────────────────────────────────────────────────────────

def resolve_audio_path(audio_file: str, audio_root: str) -> Optional[str]:
    primary = os.path.join(audio_root, audio_file)
    if os.path.exists(primary):
        return primary
    if audio_root == DEFAULT_AUDIO_REAL:
        synth = os.path.join(DEFAULT_AUDIO_SYNTH, audio_file)
        if os.path.exists(synth):
            return synth
    return None


def pick_available_recording(sample: dict, audio_root: str):
    """Return (filename, path) for the sample's HEADSET recording, or None if none.

    Mic policy (locked 2026-06-02): use the headset (close-talk) recording ONLY.
    Samples with no headset file on disk are dropped (→ None), giving a fully
    mic-homogeneous clean baseline (n=421). Rationale: the controlled degradation
    axis already has a *synthetic* far-field condition, so admitting *real* far-field
    into the clean baseline would confound it (see tracker §6/§14)."""
    for fname in (r.get("file") for r in sample.get("recordings", []) if r.get("file")):
        if "-headset" in fname:
            p = resolve_audio_path(fname, audio_root)
            if p is not None:
                return fname, p
    return None


def stable_seed(base_seed: int, *parts) -> int:
    """Derive a per-(file, degradation, snr) seed deterministically.
    Uses SHA-256 of the concatenated parts so the seed is reproducible across runs."""
    h = hashlib.sha256()
    h.update(str(base_seed).encode())
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode())
    return int.from_bytes(h.digest()[:4], "big")


def peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    p = float(np.max(np.abs(audio))) + 1e-12
    return (audio / p) * target_peak


# ─── Degradations ───────────────────────────────────────────────────────────

def degrade_noise(audio: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white Gaussian noise scaled to the target SNR."""
    noise = rng.standard_normal(audio.shape).astype(np.float32)
    sig_power = float(np.mean(audio ** 2)) + 1e-12
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10.0))
    scaling = np.sqrt(target_noise_power / noise_power)
    return (audio + noise * scaling).astype(np.float32)


def degrade_reverb(audio: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Convolve with a synthetic shoebox-room impulse response.
    SNR levels map to T60 (reverb decay time): 20→0.3 s, 10→0.6 s, 0→1.2 s."""
    t60 = REVERB_T60_BY_SNR.get(int(snr_db), 0.6)
    room_dim = [8.0, 6.0, 3.0]  # meters
    e_absorption, max_order = pra.inverse_sabine(t60, room_dim)
    room = pra.ShoeBox(
        room_dim, fs=sr,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )
    # Source and mic with a small seeded jitter so re-runs are stable.
    src_pos = [
        4.0 + 0.5 * rng.uniform(-1, 1),
        3.0 + 0.5 * rng.uniform(-1, 1),
        1.5,
    ]
    mic_pos = np.array([[5.5], [4.0], [1.5]])
    room.add_source(src_pos, signal=audio.astype(np.float32))
    room.add_microphone_array(pra.MicrophoneArray(mic_pos, fs=sr))
    room.simulate()
    out = room.mic_array.signals[0]
    # Truncate/pad to the original length and renormalize to original peak.
    target_len = len(audio)
    if len(out) > target_len:
        out = out[:target_len]
    elif len(out) < target_len:
        out = np.pad(out, (0, target_len - len(out)))
    original_peak = float(np.max(np.abs(audio))) + 1e-12
    out = (out / (np.max(np.abs(out)) + 1e-12)) * original_peak
    return out.astype(np.float32)


def degrade_farfield(audio: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Low-pass filter + attenuation, mimicking distance from microphone."""
    cutoff = FARFIELD_CUTOFF_BY_SNR.get(int(snr_db), 4000)
    atten = FARFIELD_ATTEN_BY_SNR.get(int(snr_db), 0.4)
    nyq = sr / 2.0
    b, a = butter(4, cutoff / nyq, btype="low")
    out = lfilter(b, a, audio) * atten
    return out.astype(np.float32)


def degrade_codec(audio: np.ndarray, sr: int, snr_db: float,
                  rng: np.random.Generator, tmp_dir: Path) -> np.ndarray:
    """Encode to low-bitrate Opus and decode back. Models telephony-style codec loss."""
    bitrate = CODEC_BITRATE_BY_SNR.get(int(snr_db), 16)
    in_path = tmp_dir / "in.wav"
    enc_path = tmp_dir / "enc.opus"
    sf.write(in_path, audio, sr)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(in_path),
         "-c:a", "libopus", "-b:a", f"{bitrate}k",
         "-application", "voip",
         str(enc_path)],
        check=True,
    )
    out, _ = librosa.load(str(enc_path), sr=sr, mono=True)
    target_len = len(audio)
    if len(out) > target_len:
        out = out[:target_len]
    elif len(out) < target_len:
        out = np.pad(out, (0, target_len - len(out)))
    return out.astype(np.float32)


def degrade_clipping(audio: np.ndarray, sr: int, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Hard-clip waveform to ±(fraction × peak). Models mic overload."""
    frac = CLIPPING_THRESHOLD_BY_SNR.get(int(snr_db), 0.3)
    peak = float(np.max(np.abs(audio))) + 1e-12
    thresh = frac * peak
    return np.clip(audio, -thresh, thresh).astype(np.float32)


# NOISEX-92 babble track path (NOT committed to the repo — fetch via
# scripts/fetch_noisex_babble.py). 100 talkers, canteen recording.
DEFAULT_BABBLE_PATH = "dataset/noise/babble.wav"


def load_babble_track(path: str, target_sr: int) -> np.ndarray:
    """Load + resample the NOISEX-92 babble track once (19.98 kHz → target_sr)."""
    track, _ = librosa.load(path, sr=target_sr, mono=True)
    return track.astype(np.float32)


def degrade_babble(audio: np.ndarray, sr: int, snr_db: float,
                   rng: np.random.Generator, babble_track: np.ndarray) -> np.ndarray:
    """Additive multi-talker (babble) noise at the target SNR. A seeded random
    offset into the track gives each utterance a different babble segment;
    the track is tiled if shorter than the utterance. Same SNR math as degrade_noise."""
    n = len(audio)
    track = babble_track
    if len(track) < n:
        reps = int(np.ceil(n / max(len(track), 1)))
        track = np.tile(track, reps)
    max_off = len(track) - n
    off = int(rng.integers(0, max_off + 1)) if max_off > 0 else 0
    noise = track[off:off + n].astype(np.float32)
    sig_power = float(np.mean(audio ** 2)) + 1e-12
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10.0))
    scaling = np.sqrt(target_noise_power / noise_power)
    return (audio + noise * scaling).astype(np.float32)


DEGRADE_FUNCS = {
    "noise": degrade_noise,
    "reverb": degrade_reverb,
    "farfield": degrade_farfield,
    "codec": degrade_codec,
    "clipping": degrade_clipping,
}


# ─── Driver ─────────────────────────────────────────────────────────────────

def load_target_samples(dataset_path: str, scenarios) -> list:
    samples = []
    with open(dataset_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("scenario") in scenarios:
                samples.append(d)
    return samples


def corrupt_one(
    audio_path: str,
    out_path: Path,
    degradation: str,
    snr_db: int,
    seed: int,
    tmp_dir: Path,
    babble_track=None,
) -> bool:
    """Apply one degradation to one file. Returns True if written, False if skipped (exists)."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return False
    audio, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)
    audio = audio.astype(np.float32)
    audio = peak_normalize(audio, target_peak=0.9)  # uniform input loudness
    rng = np.random.default_rng(seed)
    fn = DEGRADE_FUNCS.get(degradation)
    if degradation == "codec":
        out = degrade_codec(audio, sr, snr_db, rng, tmp_dir)
    elif degradation == "babble":
        out = degrade_babble(audio, sr, snr_db, rng, babble_track)
    else:
        out = fn(audio, sr, snr_db, rng)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, out, sr, subtype="PCM_16", format="FLAC")
    return True


def main():
    p = argparse.ArgumentParser(
        description="Apply WildASR-style degradations to SLURP audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="dataset/gold_devel_416.jsonl")
    p.add_argument("--audio-root", default=DEFAULT_AUDIO_REAL)
    p.add_argument("--scenarios", default="calendar,alarm,lists")
    p.add_argument("--output-dir", default="corrupted")
    p.add_argument("--degradations", default=",".join(DEGRADATIONS))
    p.add_argument("--snr-levels", default="20,10,0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--babble-track", default=DEFAULT_BABBLE_PATH,
                   help="NOISEX-92 babble WAV (fetch via fetch_noisex_babble.py).")
    p.add_argument("--limit", type=int, default=0, help="Cap total files (debugging). 0 = no cap.")
    args = p.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    degradations = [d.strip() for d in args.degradations.split(",") if d.strip()]
    snr_levels = [int(x) for x in args.snr_levels.split(",") if x.strip()]

    babble_track = None
    if "babble" in degradations:
        if not os.path.exists(args.babble_track):
            raise SystemExit(
                f"❌ babble track not found at {args.babble_track}. "
                f"Run: ./venv/bin/python scripts/fetch_noisex_babble.py")
        babble_track = load_babble_track(args.babble_track, TARGET_SAMPLE_RATE)
        print(f"🔊 Loaded babble track ({len(babble_track)} samples @ {TARGET_SAMPLE_RATE} Hz)")

    print(f"📋 Loading SLURP samples from {args.dataset} (scenarios: {scenarios})")
    samples = load_target_samples(args.dataset, scenarios)
    print(f"   {len(samples)} samples in scope")

    # Resolve one audio file per slurp_id (matches batch_evaluate logic).
    chosen = []
    skipped_no_audio = 0
    for s in samples:
        picked = pick_available_recording(s, args.audio_root)
        if picked is None:
            skipped_no_audio += 1
            continue
        chosen.append((s["slurp_id"], picked[0], picked[1]))
    print(f"   {len(chosen)} samples have at least one recording on disk ({skipped_no_audio} unusable)")

    total = len(chosen) * len(degradations) * len(snr_levels)
    if args.limit > 0:
        total = min(total, args.limit)
    print(f"🎯 Plan: {len(chosen)} files × {len(degradations)} degradations × {len(snr_levels)} SNRs = {total} outputs")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="corrupt_audio_"))
    written = 0
    skipped = 0
    failed = []

    count = 0
    limit_reached = False
    pbar = tqdm(total=total, desc="corrupting")
    for slurp_id, fname, path in chosen:
        if limit_reached:
            break
        for deg in degradations:
            if limit_reached:
                break
            for snr in snr_levels:
                if args.limit > 0 and count >= args.limit:
                    limit_reached = True
                    break
                out_path = output_root / deg / f"snr{snr}" / fname
                seed = stable_seed(args.seed, fname, deg, snr)
                try:
                    wrote = corrupt_one(path, out_path, deg, snr, seed, tmp_root,
                                        babble_track=babble_track)
                    if wrote:
                        written += 1
                    else:
                        skipped += 1
                except Exception as e:
                    failed.append((fname, deg, snr, repr(e)))
                count += 1
                pbar.update(1)
    pbar.close()

    print(f"\n✅ Corruption done. Written: {written}, already-present: {skipped}, failed: {len(failed)}")
    if failed:
        print(f"   First 5 failures:")
        for f in failed[:5]:
            print(f"   {f}")

    # Clean tmp
    for f in tmp_root.glob("*"):
        try: f.unlink()
        except: pass
    try: tmp_root.rmdir()
    except: pass


if __name__ == "__main__":
    main()
