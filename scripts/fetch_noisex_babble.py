"""Download the NOISEX-92 'babble' noise track to dataset/noise/babble.wav.

The raw track is NOT committed to the repo (license: verify SPIB/TNO
redistribution terms before any Zenodo deposit — tracker §14/§16). This helper
fetches it on demand so the corrupted bank is reproducible.

NOISEX-92 babble = 100 people in a canteen (Krishnamurthy & Hansen 2009),
originally 19.98 kHz; corrupt_audio.py resamples it to 16 kHz.

Usage:
    ./venv/bin/python scripts/fetch_noisex_babble.py [--url <mirror>]
"""
import argparse
import urllib.request
from pathlib import Path

# UFSC's SPIB mirror of the NOISEX-92 signals — VERIFIED 2026-07-13:
# HTTP 200, Content-Length 9,399,852 (audio/x-wav, ~235 s @ 19.98 kHz 16-bit mono).
DEFAULT_URL = "http://spib.linse.ufsc.br/data/noise/babble.wav"
# Fallback mirror (same file, byte-identical size, verified same day):
# https://raw.githubusercontent.com/speechdnn/Noises/master/NoiseX-92/babble.wav


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--out", default="dataset/noise/babble.wav")
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"✅ Already present: {out}")
        return
    print(f"⬇️  Downloading babble track from {args.url}")
    try:
        urllib.request.urlretrieve(args.url, out)
    except Exception as e:
        raise SystemExit(
            f"❌ Download failed ({e!r}). Fetch the NOISEX-92 babble track "
            f"manually and place it at {out}.")
    print(f"✅ Saved {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
