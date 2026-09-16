"""Batch evaluation runner for the ASR → LLM pipeline.

Reads SLURP utterances, runs them through one or more ASR models and one LLM
router, and writes per-utterance results to a JSONL file matching the schema
in docs/design/SCHEMA.md §10.

Supports:
  - Reproducible random sampling (--seed)
  - Multiple ASR models in a single invocation (--asr-models)
  - Custom audio root for corrupted-audio runs (--audio-root, --degradation, --snr-db)
  - Resume-from-row (skips utterances already written to the output file)
  - JSON output one row per line for downstream analysis scripts

Usage examples:

  # Small clean baseline (n=50, Whisper-Turbo, local Qwen-9B via mlx_lm.server):
  python scripts/batch_evaluate.py \\
      --num-samples 50 \\
      --asr-models mlx-community/whisper-large-v3-turbo \\
      --llm-model mlx-community/Qwen3.5-9B-MLX-8bit \\
      --out results/baseline_clean_n50.jsonl

  # Full clean sweep (all 416 samples, both Whisper variants):
  python scripts/batch_evaluate.py \\
      --num-samples 0 \\
      --asr-models mlx-community/whisper-large-v3-turbo,mlx-community/whisper-base-mlx \\
      --llm-model mlx-community/Qwen3.5-9B-MLX-8bit \\
      --out results/clean_baseline_full.jsonl

  # Phase 4 corrupted-audio sweep (one degradation × one SNR):
  python scripts/batch_evaluate.py \\
      --num-samples 0 \\
      --asr-models mlx-community/whisper-large-v3-turbo \\
      --audio-root corrupted/noise/snr10 \\
      --degradation noise --snr-db 10 \\
      --out results/1a_noise_snr10.jsonl
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from pipeline import transcribe_audio, get_intent_from_llm, model_family_of

# ─── Constants ──────────────────────────────────────────────────────────────

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

# Alias map for common LLM deviations from the canonical intent strings.
_PRED_ALIASES = {
    "set_alarm": "alarm_set",
    "query_alarm": "alarm_query",
    "remove_alarm": "alarm_remove",
    "cancel_alarm": "alarm_remove",
    "delete_alarm": "alarm_remove",
    "set_calendar": "calendar_set",
    "schedule_meeting": "calendar_set",
    "schedule_event": "calendar_set",
    "query_calendar": "calendar_query",
    "check_calendar": "calendar_query",
    "check_schedule": "calendar_query",
    "remove_calendar": "calendar_remove",
    "cancel_meeting": "calendar_remove",
    "cancel_event": "calendar_remove",
    "delete_event": "calendar_remove",
    "clear_schedule": "calendar_remove",
    "clear_calendar": "calendar_remove",
    "add_to_list": "lists_createoradd",
    "create_list": "lists_createoradd",
    "add_item": "lists_createoradd",
    "list_add": "lists_createoradd",
    "check_list": "lists_query",
    "query_list": "lists_query",
    "view_list": "lists_query",
    "remove_from_list": "lists_remove",
    "delete_from_list": "lists_remove",
    "list_remove": "lists_remove",
}

# ─── Intent normalization ───────────────────────────────────────────────────

def normalize_gold_intent(intent: str, scenario: str) -> str:
    """SLURP sometimes stores bare intents like 'query' instead of 'calendar_query'.
    Prepend the scenario when needed."""
    if intent in VALID_INTENTS:
        return intent
    candidate = f"{scenario}_{intent}"
    return candidate if candidate in VALID_INTENTS else intent


def normalize_predicted_intent(predicted) -> str:
    """Map common LLM deviations to canonical intent strings."""
    if not predicted:
        return predicted
    if not isinstance(predicted, str):
        # LLMs occasionally emit a non-string (number, list, nested dict).
        predicted = str(predicted)
    if predicted in VALID_INTENTS:
        return predicted
    key = predicted.strip().lower().replace(" ", "_")
    return _PRED_ALIASES.get(key, key)


# ─── Sample loading ─────────────────────────────────────────────────────────

def load_gold_final_map(manifest_path: str) -> dict:
    """slurp_id → gold_intent_final (the manually-corrected evaluation label).

    The raw `intent` in gold_devel_416.jsonl still carries uncorrected SLURP labels
    (e.g. "when is my meeting" mislabelled calendar_set); grading must use
    gold_intent_final from the manifest (tracker §0a)."""
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


def load_samples(dataset_path: str, scenarios, num_samples: int, seed: int,
                 manifest_path: str = DEFAULT_MANIFEST):
    """Load SLURP devel rows, filter to target scenarios, optionally sample N.

    Evaluation label = gold_intent_final from the manifest (manually corrected).
    Falls back to the normalized raw SLURP intent only if a slurp_id is missing
    from the manifest."""
    gold_final = load_gold_final_map(manifest_path)
    n_overridden = 0
    samples = []
    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            if data.get("scenario") in scenarios:
                raw_norm = normalize_gold_intent(data.get("intent", ""), data["scenario"])
                final = gold_final.get(data.get("slurp_id"))
                if final:
                    final_norm = normalize_gold_intent(final, data["scenario"])
                    if final_norm != raw_norm:
                        n_overridden += 1
                    data["intent"] = final_norm
                else:
                    data["intent"] = raw_norm
                samples.append(data)

    if gold_final:
        print(f"🏷️  Gold label: gold_intent_final from {manifest_path} "
              f"({len(gold_final)} ids; {n_overridden} differ from raw SLURP intent)")
    else:
        print(f"⚠️  Manifest {manifest_path} not found — falling back to raw SLURP intent")

    if num_samples and num_samples > 0 and num_samples < len(samples):
        rng = random.Random(seed)
        samples = rng.sample(samples, num_samples)
    return samples


def resolve_audio_path(audio_file: str, audio_root: str) -> Optional[str]:
    """Return the on-disk path for an audio file, or None if not found.
    When audio_root is the default real directory, also tries the synth dir as fallback."""
    primary = os.path.join(audio_root, audio_file)
    if os.path.exists(primary):
        return primary
    if audio_root == DEFAULT_AUDIO_REAL:
        synth = os.path.join(DEFAULT_AUDIO_SYNTH, audio_file)
        if os.path.exists(synth):
            return synth
    return None


def pick_available_recording(sample: dict, audio_root: str) -> Optional[tuple]:
    """SLURP samples have multiple recordings (same sentence, different mics).

    Mic policy (locked 2026-06-02): use the headset (close-talk) recording ONLY.
    Returns (audio_file, path), or None if no headset file is on disk (the sample is
    then dropped). This yields a fully mic-homogeneous clean baseline (n=421); keeping
    *real* far-field out of "clean" avoids confounding the *synthetic* far-field
    degradation (tracker §6/§14)."""
    for fname in (r.get("file") for r in sample.get("recordings", []) if r.get("file")):
        if "-headset" in fname:
            p = resolve_audio_path(fname, audio_root)
            if p is not None:
                return fname, p
    return None


# ─── Resume-from-row ────────────────────────────────────────────────────────

def load_processed(out_path: str) -> set:
    """Read existing JSONL output, return set of identifying tuples already done.
    Key is (slurp_id, asr_model, degradation, snr_db, llm_model).

    Rows without a usable result (tsa is null — transient ASR/LLM failure in an
    older run) do NOT count as processed, so they are retried on resume instead
    of being lost forever."""
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
            if row.get("tsa") is None:
                continue
            key = (
                row.get("slurp_id"),
                row.get("asr_model"),
                row.get("degradation"),
                row.get("snr_db"),
                row.get("llm_model"),
            )
            processed.add(key)
    return processed


# ─── Main evaluation loop ───────────────────────────────────────────────────

def evaluate(args):
    samples = load_samples(args.dataset, TARGET_SCENARIOS, args.num_samples, args.seed,
                           manifest_path=args.manifest)
    print(f"📋 Loaded {len(samples)} samples from {args.dataset}")

    asr_models = [m.strip() for m in args.asr_models.split(",") if m.strip()]
    print(f"🎙️ ASR models: {asr_models}")
    print(f"🧠 LLM model: {args.llm_model}")
    print(f"📁 Audio root: {args.audio_root}")
    print(f"🔧 Degradation: {args.degradation}, SNR: {args.snr_db}")
    print(f"💾 Output: {args.out}")

    # Ensure output directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip rows already in the output file
    processed = load_processed(args.out)
    if processed:
        print(f"⏩ Resume: {len(processed)} rows already processed, will skip duplicates")

    # Open in append mode — never overwrite
    out_f = out_path.open("a", encoding="utf-8")

    # Failed rows (transient exceptions) go to a sidecar file — NOT the main
    # output — so resume retries them while the audit trail is preserved.
    fail_path = out_path.parent / (out_path.stem + ".failures.jsonl")
    fail_f = None
    n_failures = 0

    def write_failure(row):
        nonlocal fail_f, n_failures
        if fail_f is None:
            fail_f = fail_path.open("a", encoding="utf-8")
        fail_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        fail_f.flush()
        n_failures += 1

    # Per-(asr,llm) summary counters
    summary = {asr: {"total": 0, "tsa_correct": 0} for asr in asr_models}

    for asr_model in asr_models:
        asr_short = asr_model.split("/")[-1]
        print(f"\n{'=' * 60}\n🚀 ASR: {asr_short}\n{'=' * 60}")

        for sample in tqdm(samples, desc=asr_short):
            slurp_id = sample["slurp_id"]
            gold_intent = sample["intent"]
            gold_sentence = sample.get("sentence", "")

            picked = pick_available_recording(sample, args.audio_root)
            if picked is None:
                print(f"  ⚠️ No usable recording for slurp_id={slurp_id} "
                      f"(tried {len(sample.get('recordings', []))} files)")
                continue
            audio_file, audio_path = picked

            key = (
                slurp_id, asr_model,
                args.degradation, args.snr_db, args.llm_model,
            )
            if key in processed:
                continue

            row = {
                "slurp_id": slurp_id,
                "audio_file": audio_file,
                "scenario": sample["scenario"],
                "gold_sentence": gold_sentence,
                "gold_intent": gold_intent,
                "gold_parameters": {},  # filled in by a later phase
                "pathway": "cascade",
                "model_family": model_family_of(args.llm_model),
                "asr_model": asr_model,
                "degradation": args.degradation,
                "snr_db": args.snr_db,
                "rng_seed": args.seed,
                "transcription": None,
                "wer": None,             # Phase 4
                "cer": None,             # Phase 4
                "slot_error_rate": None, # Phase 4
                "error_categories": [],  # Phase 4
                "phonetic_distance": None,   # Phase 4 (annotate_errors)
                "full_hallucination": None,  # Phase 4
                "llm_model": args.llm_model,
                "llm_prompt_variant": args.llm_prompt_variant,
                "predicted_intent_raw": None,
                "predicted_intent": None,
                "predicted_parameters": {},
                "tsa": None,
                "pf": None,             # Phase 2
                "ees": None,            # Phase 2
                "tsh_category": None,   # Phase 5
                "tch_category": None,   # Phase 5
                "notes": "",
            }

            try:
                transcription = transcribe_audio(
                    audio_path,
                    model_path=asr_model,
                    language=args.asr_language,
                )
                row["transcription"] = transcription

                if not transcription or not transcription.strip():
                    # Convention pinned in Step 4c and implemented in
                    # route_transcripts.route_one: an empty transcription is a
                    # real end-to-end failure, stamped deterministically with NO
                    # model call. This driver used to pass "" straight to the
                    # router and score whatever came back — asking a model about
                    # nothing and recording its guess. All 18 affected rows
                    # happened to get `unknown` from Qwen-9B, so nothing
                    # collected is corrupt; that was luck, not design, and a
                    # different router would have fabricated successes.
                    # (audit 2026-08-14)
                    row["predicted_intent_raw"] = "NO_TRANSCRIPTION"
                    row["predicted_intent"] = "NO_TRANSCRIPTION"
                    row["predicted_parameters"] = {}
                    row["notes"] = (row.get("notes") or "") + " | no_transcription"
                    llm_response = None
                else:
                    llm_response = get_intent_from_llm(
                        transcription, model=args.llm_model, base_url=args.llm_base_url
                    )
                if llm_response is None and not (transcription or "").strip():
                    # Empty transcription: stamped above, no model was asked.
                    # The `and not transcription` matters: _parse_json_output
                    # ALSO returns None for a literal `null` from the model, and
                    # a bare `is None` here silently wrote a row with no
                    # prediction at all — which load_processed then marks done
                    # forever, since it only skips on `tsa is None`.
                    # (regression introduced and caught 2026-08-14)
                    pass
                elif isinstance(llm_response, dict) and llm_response.get("error") == "llm_request_failed":
                    # Infrastructure failure (server down / timeout) — NOT a model
                    # prediction. Must not be scored (it would conflate with the
                    # model's legitimate "unknown"). Raise → failure sidecar → retried.
                    raise RuntimeError(f"LLM request failed: {llm_response.get('detail')}")
                elif isinstance(llm_response, dict) and "error" not in llm_response:
                    # "MISSING_INTENT" (not "UNKNOWN"): a JSON object without an
                    # intent key must not normalize into the valid "unknown" label.
                    raw_pred = llm_response.get("intent", "MISSING_INTENT")
                    norm_pred = normalize_predicted_intent(raw_pred)
                    row["predicted_intent_raw"] = raw_pred
                    row["predicted_intent"] = norm_pred
                    row["predicted_parameters"] = llm_response.get("parameters", {}) or {}
                else:
                    # Model emitted unparseable output — deterministic at temp 0,
                    # so record it as a scored failure rather than retrying.
                    row["predicted_intent_raw"] = "JSON_PARSE_ERROR"
                    row["predicted_intent"] = "JSON_PARSE_ERROR"
                    row["notes"] = "LLM did not return a parseable dict"

                row["tsa"] = int(row["predicted_intent"] == gold_intent)

                summary[asr_model]["total"] += 1
                summary[asr_model]["tsa_correct"] += row["tsa"]
                if row["tsa"] == 0:
                    print(
                        f"   ❌ slurp_id={slurp_id} | gold={gold_intent} "
                        f"| raw={row['predicted_intent_raw']} | norm={row['predicted_intent']}"
                    )

            except Exception as e:
                row["notes"] = f"exception: {e!r}"
                print(f"  ❌ Error processing {audio_file}: {e} (logged to {fail_path.name}, will retry on resume)")
                write_failure(row)
                continue

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

    out_f.close()
    if fail_f is not None:
        fail_f.close()
        print(f"\n⚠️ {n_failures} row(s) failed with exceptions → {fail_path} "
              f"(not in the main output; re-run to retry them)")

    # Final summary
    print(f"\n{'=' * 60}\n📊 Summary\n{'=' * 60}")
    for asr_model, s in summary.items():
        if s["total"] == 0:
            print(f"  {asr_model}: 0 processed (resume from earlier run?)")
            continue
        acc = 100.0 * s["tsa_correct"] / s["total"]
        print(f"  {asr_model}: TSA = {acc:.2f}% ({s['tsa_correct']}/{s['total']})")
    print(f"\n📄 Detailed results: {args.out}")


# ─── CLI ────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="Batch-evaluate the ASR→LLM pipeline on SLURP samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--num-samples", type=int, default=20,
        help="Number of utterances to evaluate. 0 or negative = all matching samples.",
    )
    p.add_argument(
        "--asr-models", type=str,
        default="mlx-community/whisper-large-v3-turbo,mlx-community/whisper-base-mlx",
        help="Comma-separated HF model IDs for ASR.",
    )
    p.add_argument(
        "--llm-model", type=str,
        default=os.environ.get("LLM_MODEL", "mlx-community/Qwen3.5-9B-MLX-8bit"),
        help="Model name passed to the OpenAI-compatible router endpoint.",
    )
    p.add_argument(
        "--llm-base-url", type=str,
        default=os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"),
        help="OpenAI-compatible endpoint: mlx_lm.server locally, vLLM on GCP.",
    )
    p.add_argument(
        "--llm-prompt-variant", type=str, default="primary_9intent_9shot",
        help="Label for the prompt variant in use (logged into the JSONL).",
    )
    p.add_argument(
        "--asr-language", type=str, default="en",
        help="Forced ASR language code (passes through to mlx_whisper).",
    )
    p.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET,
        help="SLURP devel.jsonl path (sentence/recordings/entities + raw intent).",
    )
    p.add_argument(
        "--manifest", type=str, default=DEFAULT_MANIFEST,
        help="Manifest with gold_intent_final — the corrected evaluation label.",
    )
    p.add_argument(
        "--audio-root", type=str, default=DEFAULT_AUDIO_REAL,
        help="Directory containing the audio files referenced by the dataset.",
    )
    p.add_argument(
        "--degradation", type=str, default="clean",
        help="Degradation label, written into each output row (e.g. clean/noise/reverb/...).",
    )
    p.add_argument(
        "--snr-db", type=int, default=None,
        help="SNR in dB for the corrupted run, or omit for clean audio.",
    )
    p.add_argument(
        "--out", type=str, default="results/baseline_n20_patched.jsonl",
        help="Output JSONL path (appended to; supports resume).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for sample selection.",
    )
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    evaluate(args)
