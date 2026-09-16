"""Phase 9 — prompt-sensitivity ablation.

Tests whether the H1-rejection finding is robust to prompt choice by re-routing
existing transcriptions through three prompt variants of the LLM router:

  primary      — current 9-intent prompt with 9 few-shot examples (production)
  minimal      — zero-shot, just the ontology + JSON format instruction
  expanded     — 12-shot, adds 3 examples covering edge cases

Re-uses transcriptions already in results/clean_baseline_full.jsonl and
results/1a_noise_snr10.jsonl — no new ASR work. Only the LLM is re-invoked,
through the same OpenAI-compatible backend as the main sweep
(pipeline.get_intent_from_llm → mlx_lm.server locally / vLLM on GCP).

Output: a TSV table comparing TSA across (prompt_variant × condition).

Usage:
    python prompt_ablation.py \\
        --n-samples 100 \\
        --llm-model mlx-community/Qwen3.5-9B-MLX-8bit \\
        --out results/analysis/prompt_ablation.jsonl
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from tqdm import tqdm

from pipeline import get_intent_from_llm, SYSTEM_PROMPT
from batch_evaluate import normalize_predicted_intent, VALID_INTENTS

# ─── The three prompt variants ──────────────────────────────────────────────

# Single source of truth: the production prompt lives in pipeline.py.
PROMPT_PRIMARY = SYSTEM_PROMPT


PROMPT_MINIMAL = """You are an Intent Router. Classify each transcription into EXACTLY ONE of these intent labels:

calendar_set, calendar_query, calendar_remove,
alarm_set, alarm_query, alarm_remove,
lists_createoradd, lists_query, lists_remove,
unknown

Respond ONLY with valid JSON in this format:
{"intent": "<label>", "parameters": {...}}

If unsure, use "unknown"."""


PROMPT_EXPANDED = PROMPT_PRIMARY + """

Example 10:
User: "Tell me what's on my agenda for next Monday"
{"intent": "calendar_query", "parameters": {"date": "next Monday"}}

Example 11:
User: "Push back my dentist appointment to Thursday"
{"intent": "calendar_set", "parameters": {"event_name": "dentist", "date": "Thursday"}}

Example 12:
User: "Show me the items on my todo list"
{"intent": "lists_query", "parameters": {"list_name": "todo"}}"""


PROMPTS = {
    "primary": PROMPT_PRIMARY,
    "minimal": PROMPT_MINIMAL,
    "expanded": PROMPT_EXPANDED,
}


# ─── Driver ─────────────────────────────────────────────────────────────────

def load_subset(jsonl_path: str, asr_filter: str, n: int, seed: int):
    """Load up to n Whisper-Turbo rows with non-null transcription."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if asr_filter not in (r.get("asr_model") or ""):
                continue
            if not r.get("transcription"):
                continue
            rows.append(r)
    rng = random.Random(seed)
    if n and n < len(rows):
        rows = rng.sample(rows, n)
    return rows


# The three conditions the ablation runs over: clean, one moderate additive
# noise cell, one moderate reverb cell. ⚠️ These were the Study-1a filenames
# (`clean_baseline_full.jsonl`, `1a_noise_snr10.jsonl`, `1a_reverb_snr10.jsonl`)
# until 2026-08-28, by which time the cascade files had been renamed and none
# of the three existed. The script skipped all three, printed an empty summary
# followed by its "Interpretation:" paragraph, and exited 0 — six seconds of
# apparent success having read nothing. `check_inputs` is why that cannot
# happen again.
CONDITIONS = [
    ("clean", "results/cascade_whisper_qwen9b_clean.jsonl"),
    ("noise_snr10", "results/cascade_whisper_qwen9b_noise_snr10.jsonl"),
    ("reverb_snr10", "results/cascade_whisper_qwen9b_reverb_snr10.jsonl"),
]


def check_inputs(conditions):
    """Refuse the run unless every declared input exists.

    An ablation over zero conditions is not a null result, it is a broken
    command — and this script's own summary table renders identically either
    way. Fail here, loudly, with the paths that are missing.
    """
    missing = [(name, path) for name, path in conditions
               if not Path(path).exists()]
    if missing:
        raise SystemExit(
            "prompt_ablation: %d of %d declared inputs do not exist:\n%s\n"
            "These are hardcoded paths; if the result files were renamed, "
            "update CONDITIONS. Refusing rather than summarising nothing."
            % (len(missing), len(conditions),
               "\n".join("  %-14s %s" % (n, p) for n, p in missing)))
    return conditions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm-model",
                   default=os.environ.get("LLM_MODEL", "mlx-community/Qwen3.5-9B-MLX-8bit"))
    p.add_argument("--llm-base-url",
                   default=os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"),
                   help="OpenAI-compatible endpoint: mlx_lm.server locally, vLLM on GCP.")
    p.add_argument("--asr-filter", default="whisper-large-v3-turbo")
    p.add_argument("--out", default="results/analysis/prompt_ablation.jsonl")
    args = p.parse_args()

    conditions = check_inputs(CONDITIONS)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_path.open("w", encoding="utf-8")

    # cell_key = (condition, prompt_variant) → {n, ok, slot_position breakdowns}
    summary = {}

    for cond_name, jsonl_path in conditions:
        if not Path(jsonl_path).exists():
            print(f"⏭️  Skipping {cond_name}: {jsonl_path} not found")
            continue
        rows = load_subset(jsonl_path, args.asr_filter, args.n_samples, args.seed)
        print(f"\n═══ Condition: {cond_name} ({len(rows)} samples)")
        for variant_name, prompt in PROMPTS.items():
            tsa_ok = 0
            n_scored = 0
            n_llm_errors = 0
            t0 = time.time()
            for r in tqdm(rows, desc=f"  {variant_name}"):
                pred = get_intent_from_llm(
                    r["transcription"], model=args.llm_model,
                    base_url=args.llm_base_url, system_prompt=prompt,
                )
                out_row = {
                    "condition": cond_name,
                    "prompt_variant": variant_name,
                    "slurp_id": r["slurp_id"],
                    "gold_intent": r["gold_intent"],
                    "gold_sentence": r.get("gold_sentence"),
                    "transcription": r["transcription"],
                    "asr_model": r["asr_model"],
                    "wer": r.get("wer"),
                }
                if not isinstance(pred, dict) or "error" in pred:
                    # Request/parse failure — excluded from the TSA denominator
                    # so infra errors don't masquerade as model behaviour.
                    n_llm_errors += 1
                    out_row.update({
                        "predicted_intent_raw": "LLM_ERROR",
                        "predicted_intent": "LLM_ERROR",
                        "predicted_parameters": {},
                        "tsa": None,
                        "llm_error": (pred.get("error") if isinstance(pred, dict)
                                      else "non_object_json"),
                    })
                else:
                    pred_intent_raw = pred.get("intent", "MISSING_INTENT")
                    pred_intent = normalize_predicted_intent(pred_intent_raw)
                    tsa = int(pred_intent == r.get("gold_intent"))
                    tsa_ok += tsa
                    n_scored += 1
                    out_row.update({
                        "predicted_intent_raw": pred_intent_raw,
                        "predicted_intent": pred_intent,
                        "predicted_parameters": pred.get("parameters", {}) or {},
                        "tsa": tsa,
                    })
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                out_f.flush()
            dt = time.time() - t0
            acc = 100.0 * tsa_ok / n_scored if n_scored else 0.0
            summary[(cond_name, variant_name)] = {
                "n": n_scored, "tsa_ok": tsa_ok, "tsa_pct": acc,
                "n_llm_errors": n_llm_errors, "elapsed": dt,
            }
            err_note = f"  ({n_llm_errors} LLM errors excluded)" if n_llm_errors else ""
            print(f"     TSA = {acc:.1f}%  ({tsa_ok}/{n_scored}){err_note}  [{dt:.0f}s]")

    out_f.close()

    # Final comparison table
    print(f"\n{'═'*70}")
    print(f"PROMPT ABLATION SUMMARY (Whisper-Turbo, n={args.n_samples}/condition)")
    print(f"{'═'*70}")
    print(f"{'condition':18s} | " + " | ".join(f"{v:>10s}" for v in PROMPTS))
    print("─" * 70)
    for cond_name, _ in conditions:
        if not any(k[0] == cond_name for k in summary):
            continue
        cells = []
        for v in PROMPTS:
            s = summary.get((cond_name, v))
            cells.append(f"{s['tsa_pct']:>9.1f}%" if s else "       —")
        print(f"{cond_name:18s} | " + " | ".join(cells))
    print()
    print("Interpretation: if the three variants give TSA within 2–3 pts of each other,")
    print("the H1 contrast is robust to prompt engineering — passes the most")
    print("common reviewer critique on prompt-tuned LLM evaluations.")


if __name__ == "__main__":
    main()
