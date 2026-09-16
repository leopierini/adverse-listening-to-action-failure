"""Route an existing transcription JSONL through the LLM intent classifier.

Reads a JSONL (typically the output of transcribe_parakeet.py), runs each
transcription through the LLM router (via pipeline.get_intent_from_llm,
OpenAI-compatible backend: mlx_lm.server locally or vLLM on GCP), and writes
a new JSONL with predicted_intent / predicted_parameters / tsa populated.

This decouples ASR from LLM so we can run Parakeet (venv312, MLX) and the
router client (venv) in separate processes.

Usage (in venv):
    python route_transcripts.py \\
        --in  results/parakeet_clean_transcripts.jsonl \\
        --out results/parakeet_clean_routed.jsonl \\
        --llm-model mlx-community/Qwen3.5-9B-MLX-8bit
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import pipeline
from pipeline import (
    get_intent_from_llm, model_family_of, thinking_request,
    verify_model_served, ModelNotServedError,
)
from batch_evaluate import normalize_predicted_intent

DEFAULT_PROMPT_VARIANT = "primary_9intent_9shot"

# Every field that belongs to a PREVIOUS router's run and must never survive
# onto a row stamped with a new llm_model (audit 2026-08-12, finding C1/C2).
# This matters because the Whisper transcripts only exist inside already-routed
# files, so the GCP input rows arrive carrying Qwen-9B's answers and scores.
# `wer`/`cer`/`slot_error_rate`/`error_categories`/`phonetic_distance`/
# `full_hallucination` are deliberately NOT cleared — they are ASR-stage facts,
# identical for every router, and reusing them saves recomputing them.
STALE_PREDICTION_FIELDS = (
    "predicted_intent", "predicted_intent_raw", "tsa",
    # `notes` holds ROUTER-stage state ("LLM did not return a parseable dict",
    # " | no_transcription"). It was omitted from this list by oversight, not by
    # decision: a 9B parse failure would be attributed to the 27B on the row the
    # 27B wrote. One such row exists today in the whisper cascade files, and
    # Step 6 feeds exactly those files. (audit 2026-08-14)
    "notes",
    "pf", "pf_n_gold", "pf_n_matched", "ees", "ees_strict",
    "tsh_category", "tch_category",
)


def clear_prediction_fields(row):
    """Wipe the previous router's prediction and scores from an input row."""
    for field in STALE_PREDICTION_FIELDS:
        row[field] = None
    row["predicted_parameters"] = {}
    return row


def load_processed(out_path: str) -> set:
    """Rows already routed. A row counts as done only if it carries a usable
    prediction — rows that failed with a transient LLM exception live in the
    sidecar, not here, so they are retried on resume rather than lost.

    Empty transcriptions now carry predicted_intent="NO_TRANSCRIPTION" and so are
    caught by the first clause; the notes check is kept only for rows written by
    the pre-2026-08-12 code, which left predicted_intent null for those."""
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
            done = (row.get("predicted_intent") is not None
                    or "no_transcription" in (row.get("notes") or ""))
            if not done:
                continue
            processed.add((row.get("slurp_id"), row.get("asr_model"),
                           row.get("degradation"), row.get("snr_db"),
                           row.get("llm_model")))
    return processed


def route_one(row, llm_model, base_url, prompt_variant, served_model=None):
    """Route one transcript row. Returns ('ok'|'fail', row)."""
    # Clear BEFORE routing, on every path: whatever happens below, this row must
    # not leave carrying the previous router's prediction (finding C1).
    clear_prediction_fields(row)
    row["llm_model"] = llm_model
    row["llm_model_served"] = served_model
    row["llm_prompt_variant"] = prompt_variant
    row["llm_thinking_request"] = thinking_request(llm_model)
    row["pathway"] = "cascade"
    row["model_family"] = model_family_of(llm_model)
    transcription = row.get("transcription")
    if not transcription or not transcription.strip():
        # Convention (decided 2026-08-12): an empty transcription is a real
        # end-to-end failure, not a missing datapoint — the ASR produced nothing,
        # so the user's action did not happen. compute_metrics.py already scores
        # it WER=1.0, so scoring it here is the convention the codebase follows.
        # Stamped deterministically: no request is sent (an empty user turn would
        # cost a GPU call to ask the model about nothing), and the denominators
        # stay identical across the three routers, which paired H2a requires.
        row["predicted_intent_raw"] = "NO_TRANSCRIPTION"
        row["predicted_intent"] = "NO_TRANSCRIPTION"
        row["predicted_parameters"] = {}
        row["tsa"] = 0
        row["notes"] = (row.get("notes") or "") + " | no_transcription"
        return "ok", row
    try:
        r = get_intent_from_llm(transcription, model=llm_model, base_url=base_url)
        if isinstance(r, dict) and r.get("error") == "llm_request_failed":
            # Infrastructure failure — not a prediction; must not be scored
            # (would conflate with the model's legitimate "unknown").
            raise RuntimeError(f"LLM request failed: {r.get('detail')}")
        if isinstance(r, dict) and "error" not in r:
            # "MISSING_INTENT" (not "UNKNOWN"): an object without an intent
            # key must not normalize into the valid "unknown" label.
            raw_pred = r.get("intent", "MISSING_INTENT")
            row["predicted_intent_raw"] = raw_pred
            row["predicted_intent"] = normalize_predicted_intent(raw_pred)
            row["predicted_parameters"] = r.get("parameters", {}) or {}
        else:
            # Unparseable model output — deterministic at temp 0; score it.
            row["predicted_intent_raw"] = "JSON_PARSE_ERROR"
            row["predicted_intent"] = "JSON_PARSE_ERROR"
            row["notes"] = (row.get("notes") or "") + " | llm_output_not_json"
        row["tsa"] = int(row["predicted_intent"] == row.get("gold_intent"))
        return "ok", row
    except Exception as e:
        row["notes"] = (row.get("notes") or "") + f" | llm_exception: {e!r}"
        return "fail", row


def _verify_or_exit(llm_model, base_url, skip):
    """Confirm the endpoint really serves `llm_model`, or abort the run."""
    if skip:
        print("⚠️ --skip-model-check: NOT verifying the endpoint. "
              "llm_model_served will be null — you are trusting --llm-model.")
        return None
    try:
        served = verify_model_served(llm_model, base_url)
    except ModelNotServedError as e:
        print(f"\n❌ {e}\n")
        sys.exit(2)
    print(f"✅ Endpoint {base_url} serves {served}")
    return served


def main():
    p = argparse.ArgumentParser(description="Route a transcripts JSONL through the LLM.")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--llm-model",
                   default=os.environ.get("LLM_MODEL", "mlx-community/Qwen3.5-9B-MLX-8bit"))
    p.add_argument("--llm-base-url",
                   default=os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"),
                   help="OpenAI-compatible endpoint: mlx_lm.server locally, vLLM on GCP.")
    p.add_argument("--llm-prompt-variant", default=DEFAULT_PROMPT_VARIANT)
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent routing requests. 1 locally; 16-32 on GCP vLLM.")
    p.add_argument("--skip-model-check", action="store_true",
                   help="Do NOT verify the endpoint serves --llm-model. Only for "
                        "backends without /v1/models; llm_model_served stays null.")
    p.add_argument("--dump-raw-responses", type=int, default=0, metavar="N",
                   help="Save the first N raw responses per model to "
                        "<out>.rawdump.jsonl for pilot inspection (thinking-mode, "
                        "JSON compliance). 0 = off.")
    args = p.parse_args()

    served_model = _verify_or_exit(args.llm_model, args.llm_base_url,
                                   args.skip_model_check)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed = load_processed(args.out)
    if processed:
        print(f"⏩ Resume: {len(processed)} rows already routed, will skip")

    in_rows = []
    with open(args.in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            in_rows.append(json.loads(line))
    print(f"📋 {len(in_rows)} transcripts to route")

    if args.dump_raw_responses:
        dump_path = out_path.parent / (out_path.stem + ".rawdump.jsonl")
        pipeline.enable_raw_dump(str(dump_path), args.dump_raw_responses)
        print(f"🔍 Saving the first {args.dump_raw_responses} raw responses "
              f"per model → {dump_path}")

    out_f = out_path.open("a", encoding="utf-8")
    summary = {"total": 0, "tsa_correct": 0}

    # Failed rows go to a sidecar, not the main output → retried on resume.
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

    worklist = [row for row in in_rows
                if (row.get("slurp_id"), row.get("asr_model"),
                    row.get("degradation"), row.get("snr_db"), args.llm_model) not in processed]

    def handle(status, row):
        if status == "fail":
            write_failure(row)
            return
        summary["total"] += 1
        summary["tsa_correct"] += row["tsa"]
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()

    interrupted = False
    try:
        if args.workers <= 1:
            for row in tqdm(worklist, desc="routing"):
                handle(*route_one(row, args.llm_model, args.llm_base_url,
                                  args.llm_prompt_variant, served_model))
        else:
            # cancel_futures on shutdown (audit 2026-08-12, finding H3): without
            # it, ThreadPoolExecutor.__exit__ drains every queued future, so one
            # Ctrl-C left the A100 saturated for the rest of the sweep with no
            # progress bar and no way out but kill -9.
            ex = ThreadPoolExecutor(max_workers=args.workers)
            futs = []
            try:
                for row in worklist:
                    futs.append(ex.submit(route_one, row, args.llm_model,
                                          args.llm_base_url, args.llm_prompt_variant,
                                          served_model))
                for fut in tqdm(as_completed(futs), total=len(futs), desc="routing"):
                    handle(*fut.result())
            finally:
                ex.shutdown(wait=True, cancel_futures=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\n⛔ Interrupted — queued requests cancelled. Everything already "
              "written is complete; re-run the same command to resume.")

    out_f.close()
    if fail_f is not None:
        fail_f.close()
        print(f"⚠️ {n_failures} row(s) failed with exceptions → {fail_path} "
              f"(not in the main output; re-run to retry them)")
    if summary["total"]:
        acc = 100.0 * summary["tsa_correct"] / summary["total"]
        print(f"\n📊 TSA = {acc:.2f}% ({summary['tsa_correct']}/{summary['total']})")
    print(f"📄 Output: {args.out}")
    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()
