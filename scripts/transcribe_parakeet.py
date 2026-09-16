"""ASR-only batch transcription with Parakeet TDT 0.6B (MLX) on Apple Silicon.

Mirrors the ASR half of batch_evaluate.py but writes JSONL rows with LLM fields
left null — those get filled in by route_transcripts.py in a later step.

Why split ASR and LLM? Parakeet-MLX needs Python ≥ 3.10 (venv312) and the
router client lives in venv (3.9). Decoupling keeps both running.

Usage (in venv312):
    python transcribe_parakeet.py \\
        --asr-model mlx-community/parakeet-tdt-0.6b-v3 \\
        --audio-root dataset/audio \\
        --degradation clean \\
        --out results/parakeet_clean_transcripts.jsonl
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

# Ensure HF cache points at the project models folder (scripts/ is one level below root).
os.environ.setdefault(
    "HF_HOME",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
)

import parakeet_mlx
from tqdm import tqdm

VALID_INTENTS = {
    "calendar_set", "calendar_query", "calendar_remove",
    "alarm_set", "alarm_query", "alarm_remove",
    "lists_createoradd", "lists_query", "lists_remove",
    "unknown",
}
TARGET_SCENARIOS = ["calendar", "alarm", "lists"]
# Source is now the frozen dataset (SLURP/ was removed 2026-06-02). Labels: use
# dataset/manifest.jsonl `gold_intent_final`; gold_devel_416.jsonl = raw SLURP lines.
DEFAULT_DATASET = "dataset/gold_devel_416.jsonl"
DEFAULT_MANIFEST = "dataset/manifest.jsonl"
DEFAULT_AUDIO_REAL = "dataset/audio"
DEFAULT_AUDIO_SYNTH = "dataset/audio"


def normalize_gold_intent(intent: str, scenario: str) -> str:
    if intent in VALID_INTENTS:
        return intent
    candidate = f"{scenario}_{intent}"
    return candidate if candidate in VALID_INTENTS else intent


def load_gold_final_map(manifest_path: str) -> dict:
    """slurp_id → gold_intent_final (the manually-corrected evaluation label).

    The raw `intent` in gold_devel_416.jsonl still carries uncorrected SLURP labels;
    grading downstream (route_transcripts.py) compares against the gold_intent we
    stamp here, so it must be the corrected one (tracker §0a)."""
    label_map = {}
    p = Path(manifest_path)
    if not p.exists():
        return label_map
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("slurp_id")
            gif = d.get("gold_intent_final")
            if sid is not None and gif:
                label_map[sid] = gif
    return label_map


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
    """Use the headset (close-talk) recording ONLY; samples with no headset file on
    disk are dropped (→ None) for a mic-homogeneous baseline (n=421).
    Mic policy locked 2026-06-02 (see tracker §6/§14)."""
    for fname in (r.get("file") for r in sample.get("recordings", []) if r.get("file")):
        if "-headset" in fname:
            p = resolve_audio_path(fname, audio_root)
            if p is not None:
                return fname, p
    return None


def load_samples(dataset_path: str, scenarios, num_samples: int, seed: int,
                 manifest_path: str = DEFAULT_MANIFEST):
    """Evaluation label = gold_intent_final from the manifest (manually corrected);
    fall back to the normalized raw SLURP intent only if a slurp_id is missing there."""
    gold_final = load_gold_final_map(manifest_path)
    n_overridden = 0
    samples = []
    with open(dataset_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("scenario") in scenarios:
                raw_norm = normalize_gold_intent(d.get("intent", ""), d["scenario"])
                final = gold_final.get(d.get("slurp_id"))
                if final:
                    final_norm = normalize_gold_intent(final, d["scenario"])
                    if final_norm != raw_norm:
                        n_overridden += 1
                    d["intent"] = final_norm
                else:
                    d["intent"] = raw_norm
                samples.append(d)
    if gold_final:
        print(f"🏷️  Gold label: gold_intent_final from {manifest_path} "
              f"({len(gold_final)} ids; {n_overridden} differ from raw SLURP intent)")
    else:
        print(f"⚠️  Manifest {manifest_path} not found — falling back to raw SLURP intent")
    if num_samples and num_samples > 0 and num_samples < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, num_samples)
    return samples


def load_processed(out_path: str) -> set:
    """Rows already transcribed. Rows without a transcription (transient ASR
    failure in an older run) do not count — they are retried on resume."""
    processed = set()
    p = Path(out_path)
    if not p.exists():
        return processed
    with p.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("transcription") is None:
                continue
            processed.add((row.get("slurp_id"), row.get("asr_model"),
                           row.get("degradation"), row.get("snr_db")))
    return processed


def main():
    p = argparse.ArgumentParser(description="ASR-only transcription with Parakeet MLX.")
    p.add_argument("--asr-model", default="mlx-community/parakeet-tdt-0.6b-v3")
    p.add_argument("--asr-language", default="en",
                   help="Logged for provenance only — parakeet-mlx transcribe() has no "
                        "language argument (v3 auto-detects; spot-check English output).")
    p.add_argument("--num-samples", type=int, default=0)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST,
                   help="Manifest with gold_intent_final — the corrected evaluation label.")
    p.add_argument("--audio-root", default=DEFAULT_AUDIO_REAL)
    p.add_argument("--degradation", default="clean")
    p.add_argument("--snr-db", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    samples = load_samples(args.dataset, TARGET_SCENARIOS, args.num_samples, args.seed,
                           manifest_path=args.manifest)
    print(f"📋 Loaded {len(samples)} samples")
    print(f"🎙️ ASR model: {args.asr_model}")
    print(f"📁 Audio root: {args.audio_root}")
    print(f"🔧 Degradation: {args.degradation}, SNR: {args.snr_db}")
    print(f"💾 Output: {args.out}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed = load_processed(args.out)
    if processed:
        print(f"⏩ Resume: {len(processed)} rows already in output, will skip")

    print(f"⏳ Loading Parakeet model …")
    model = parakeet_mlx.from_pretrained(args.asr_model)
    print(f"✅ Loaded: {type(model).__name__}")

    out_f = out_path.open("a", encoding="utf-8")
    n_written = 0
    n_skip_resume = 0
    n_skip_no_audio = 0
    n_failed = 0
    # Failed rows go to a sidecar, not the main output → retried on resume.
    fail_path = out_path.parent / (out_path.stem + ".failures.jsonl")
    fail_f = None
    t0 = time.time()

    for sample in tqdm(samples, desc=args.asr_model.split("/")[-1]):
        slurp_id = sample["slurp_id"]
        gold_intent = sample["intent"]
        gold_sentence = sample.get("sentence", "")

        key = (slurp_id, args.asr_model, args.degradation, args.snr_db)
        if key in processed:
            n_skip_resume += 1
            continue

        picked = pick_available_recording(sample, args.audio_root)
        if picked is None:
            n_skip_no_audio += 1
            continue
        audio_file, audio_path = picked

        row = {
            "slurp_id": slurp_id,
            "audio_file": audio_file,
            "scenario": sample["scenario"],
            "gold_sentence": gold_sentence,
            "gold_intent": gold_intent,
            "gold_parameters": {},
            "pathway": "cascade",
            "model_family": None,   # set at routing time (route_transcripts)
            "asr_model": args.asr_model,
            "degradation": args.degradation,
            "snr_db": args.snr_db,
            "rng_seed": args.seed,
            "transcription": None,
            "wer": None,
            "cer": None,
            "slot_error_rate": None,
            "error_categories": [],
            "phonetic_distance": None,   # Phase 4 (annotate_errors)
            "full_hallucination": None,  # Phase 4
            "llm_model": None,
            "llm_prompt_variant": None,
            "predicted_intent_raw": None,
            "predicted_intent": None,
            "predicted_parameters": {},
            "tsa": None,
            "pf": None,
            "ees": None,
            "tsh_category": None,
            "tch_category": None,
            "notes": "",
        }

        try:
            result = model.transcribe(audio_path)
            text = result.text if hasattr(result, "text") else str(result)
            row["transcription"] = text.strip()
        except Exception as e:
            row["notes"] = f"asr_exception: {e!r}"
            if fail_f is None:
                fail_f = fail_path.open("a", encoding="utf-8")
            fail_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            fail_f.flush()
            n_failed += 1
            continue

        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        n_written += 1

    out_f.close()
    if fail_f is not None:
        fail_f.close()
        print(f"⚠️ {n_failed} row(s) failed → {fail_path} (not in the main output; re-run to retry)")
    dt = time.time() - t0
    rate = n_written / dt if dt > 0 else 0
    print(f"\n✅ Done. Wrote {n_written}, skipped resume {n_skip_resume}, no-audio {n_skip_no_audio}")
    print(f"   Elapsed: {dt:.1f}s  ({rate:.1f} files/sec)")


if __name__ == "__main__":
    main()
