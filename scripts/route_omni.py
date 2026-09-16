"""Run the degraded-audio bank through an omni (end-to-end) model — audio in,
action JSON out, NO ASR stage. Writes §10 rows with pathway="omni".

The omni model is served on the GCP A100 via vLLM (OpenAI-compatible, audio
endpoint). One model at a time: Qwen3-Omni-30B-A3B (AWQ-4bit) or Gemma 4 12B.

Usage (in venv):
    python scripts/route_omni.py \\
        --audio-root corrupted \\
        --degradations clean,noise,reverb,farfield,codec,clipping,babble \\
        --snr-levels 20,10,0 \\
        --out results/omni_qwen.jsonl \\
        --llm-model Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --llm-base-url http://<vm-ip>:8000/v1 \\
        --workers 16
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
    get_intent_from_omni, model_family_of, thinking_request,
    verify_model_served, ModelNotServedError,
)
from batch_evaluate import (
    normalize_predicted_intent, load_samples, pick_available_recording,
    DEFAULT_MANIFEST,
)

DEFAULT_PROMPT_VARIANT = "primary_9intent_9shot"


def gold_parameters_from_entities(sample):
    """SLURP entities → {slot_type: value}, duplicate types suffixed _2, _3
    (mirrors compute_metrics.py so PF scoring is identical across pathways)."""
    gp = {}
    tokens = sample.get("tokens", [])
    for ent in sample.get("entities", []):
        etype = ent.get("type")
        val = " ".join(
            tokens[i].get("surface", "")
            for i in ent.get("span", [])
            if 0 <= i < len(tokens)
        )
        key_name, k = etype, 2
        while key_name in gp:
            key_name = f"{etype}_{k}"
            k += 1
        gp[key_name] = val
    return gp


def build_omni_row(sample, audio_file, degradation, snr_db, seed, llm_model,
                   prompt_variant=DEFAULT_PROMPT_VARIANT, served_model=None):
    """One §10 row, omni flavour: transcript-derived fields null, pathway=omni."""
    return {
        "slurp_id": sample["slurp_id"],
        "audio_file": audio_file,
        "scenario": sample.get("scenario"),
        "gold_sentence": sample.get("sentence", ""),
        "gold_intent": sample["intent"],
        "gold_parameters": gold_parameters_from_entities(sample),
        "pathway": "omni",
        "model_family": model_family_of(llm_model),
        "asr_model": None,
        "degradation": degradation,
        "snr_db": snr_db,
        "rng_seed": seed,
        "transcription": None,
        "wer": None,
        "cer": None,
        "slot_error_rate": None,
        "error_categories": [],
        "phonetic_distance": None,
        "full_hallucination": None,
        "llm_model": llm_model,
        "llm_model_served": served_model,
        "llm_prompt_variant": prompt_variant,
        "llm_thinking_request": thinking_request(llm_model),
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


def apply_prediction(row, omni_response):
    """Stamp an omni JSON response onto the row (mirrors route_transcripts.py)."""
    if isinstance(omni_response, dict) and omni_response.get("error") == "llm_request_failed":
        raise RuntimeError(f"Omni request failed: {omni_response.get('detail')}")
    if isinstance(omni_response, dict) and "error" not in omni_response:
        raw_pred = omni_response.get("intent", "MISSING_INTENT")
        row["predicted_intent_raw"] = raw_pred
        row["predicted_intent"] = normalize_predicted_intent(raw_pred)
        row["predicted_parameters"] = omni_response.get("parameters", {}) or {}
    else:
        row["predicted_intent_raw"] = "JSON_PARSE_ERROR"
        row["predicted_intent"] = "JSON_PARSE_ERROR"
        row["notes"] = (row.get("notes") or "") + " | omni_output_not_json"
    row["tsa"] = int(row["predicted_intent"] == row.get("gold_intent"))
    return row


def resume_key(row):
    return (row.get("slurp_id"), row.get("pathway"), row.get("asr_model"),
            row.get("degradation"), row.get("snr_db"), row.get("llm_model"))


def load_processed(out_path):
    """A row counts as done ONLY if it carries a prediction.

    Contract changed 2026-08-12 (audit finding C5): missing audio used to count as
    done AND was written to the main output, so one run with a wrong --audio-root
    or an incomplete rsync of the corrupted bank wrote up to 7,488 rows with
    tsa=null, exited 0, and then permanently skipped them on every corrected
    re-run. Missing audio now goes to the failure sidecar instead, so it never
    reaches this file and is always retried."""
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
            if row.get("predicted_intent") is not None:
                processed.add(resume_key(row))
    return processed


def audio_path_for(audio_root, degradation, snr_db, audio_file):
    """Locate the corrupted clip. Clean audio lives at the dataset root; degraded
    clips at <root>/<degradation>/snr<snr>/<file> (matches corrupt_audio.py)."""
    if degradation == "clean":
        return os.path.join("dataset", "audio", audio_file)
    return os.path.join(audio_root, degradation, f"snr{snr_db}", audio_file)


def build_worklist(samples, audio_root, degradations, snr_levels, seed, llm_model,
                   served_model=None):
    """Yield (row, audio_path) for every (sample × degradation × snr) cell."""
    for sample in samples:
        picked = pick_available_recording(sample, os.path.join("dataset", "audio"))
        if picked is None:
            continue
        audio_file, _ = picked
        for deg in degradations:
            snrs = [None] if deg == "clean" else snr_levels
            for snr in snrs:
                row = build_omni_row(sample, audio_file, deg, snr, seed, llm_model,
                                     served_model=served_model)
                yield row, audio_path_for(audio_root, deg, snr, audio_file)


def process_one(row, apath, llm_model, base_url):
    """Route one clip. Returns ("ok", row) or ("fail", row) or ("skip", row)."""
    if not os.path.exists(apath):
        row["notes"] = (row.get("notes") or "") + f" | missing_audio:{apath}"
        return "skip", row
    try:
        resp = get_intent_from_omni(apath, model=llm_model, base_url=base_url)
        apply_prediction(row, resp)
        return "ok", row
    except Exception as e:
        row["notes"] = (row.get("notes") or "") + f" | omni_exception: {e!r}"
        return "fail", row


def run_worklist(worklist, llm_model, base_url, workers, on_ok, on_fail, on_skip):
    """Dispatch process_one over a thread pool; invoke callbacks from THIS thread
    (as futures complete) so file writes stay single-threaded and resume-safe.
    workers=1 → effectively sequential (used locally).
    Returns True if the run was interrupted with Ctrl-C."""
    def dispatch(status, row):
        (on_ok if status == "ok" else on_skip if status == "skip" else on_fail)(row)

    if workers <= 1:
        for row, apath in tqdm(worklist, desc="omni"):
            dispatch(*process_one(row, apath, llm_model, base_url))
        return False
    # cancel_futures on shutdown (audit 2026-08-12, finding H3): without it,
    # ThreadPoolExecutor.__exit__ drains every queued future, so one Ctrl-C left
    # the A100 saturated for the rest of the sweep with no way out but kill -9.
    ex = ThreadPoolExecutor(max_workers=workers)
    futs = []
    try:
        for row, apath in worklist:
            futs.append(ex.submit(process_one, row, apath, llm_model, base_url))
        for fut in tqdm(as_completed(futs), total=len(futs), desc="omni"):
            dispatch(*fut.result())
    except KeyboardInterrupt:
        print("\n⛔ Interrupted — queued requests cancelled. Everything already "
              "written is complete; re-run the same command to resume.")
        return True
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    return False


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


def _abort_if_audio_bank_missing(worklist, audio_root, max_frac):
    """Refuse to start when most of the audio bank is absent (finding C5).

    Checked BEFORE any request, so a wrong --audio-root costs zero GPU time
    instead of a full sweep of unusable rows."""
    if not worklist:
        return
    missing = [apath for _row, apath in worklist if not os.path.exists(apath)]
    if len(missing) <= max_frac * len(worklist):
        return
    pct = 100.0 * len(missing) / len(worklist)
    print(f"\n❌ {len(missing)}/{len(worklist)} clips ({pct:.1f}%) are not on disk — "
          f"refusing to start.\n"
          f"  First missing: {missing[0]}\n"
          f"  Likely causes: --audio-root is wrong (got {audio_root!r}; the bank\n"
          f"  lives in 'corrupted/<deg>/snr<snr>/'), the corrupted bank was never\n"
          f"  generated or rsynced, or this was not run from the repo root.\n"
          f"  Running anyway would spend the session writing rows with no audio.\n")
    sys.exit(3)


def main():
    p = argparse.ArgumentParser(description="Route the degraded audio bank through an omni model.")
    p.add_argument("--audio-root", default="corrupted")
    p.add_argument("--dataset", default="dataset/gold_devel_416.jsonl")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--scenarios", default="calendar,alarm,lists")
    p.add_argument("--degradations",
                   default="clean,noise,reverb,farfield,codec,clipping,babble")
    p.add_argument("--snr-levels", default="20,10,0")
    p.add_argument("--out", required=True)
    p.add_argument("--num-samples", type=int, default=0,
                   help="Utterances to run. 0 = all 416. Use 100 for the Phase-7 "
                        "gating pilot (tracker §11 step 5).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm-model",
                   default=os.environ.get("LLM_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct"))
    p.add_argument("--llm-base-url",
                   default=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--skip-model-check", action="store_true",
                   help="Do NOT verify the endpoint serves --llm-model. Only for "
                        "backends without /v1/models; llm_model_served stays null.")
    p.add_argument("--dump-raw-responses", type=int, default=0, metavar="N",
                   help="Save the first N raw responses per model to "
                        "<out>.rawdump.jsonl for pilot inspection (thinking-mode, "
                        "JSON compliance). 0 = off.")
    p.add_argument("--max-missing-audio-frac", type=float, default=0.02,
                   help="Abort before any request if more than this fraction of the "
                        "worklist has no audio on disk.")
    args = p.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    degradations = [d.strip() for d in args.degradations.split(",") if d.strip()]
    snr_levels = [int(x) for x in args.snr_levels.split(",") if x.strip()]

    served_model = _verify_or_exit(args.llm_model, args.llm_base_url,
                                   args.skip_model_check)

    samples = load_samples(args.dataset, scenarios, args.num_samples, args.seed,
                           manifest_path=args.manifest)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed = load_processed(args.out)
    if processed:
        print(f"⏩ Resume: {len(processed)} omni rows already done, will skip")

    worklist = [(r, a) for (r, a) in
                build_worklist(samples, args.audio_root, degradations, snr_levels,
                               args.seed, args.llm_model, served_model)
                if resume_key(r) not in processed]
    print(f"📋 {len(worklist)} omni cells to route with {args.llm_model}")

    _abort_if_audio_bank_missing(worklist, args.audio_root,
                                 args.max_missing_audio_frac)

    if args.dump_raw_responses:
        dump_path = out_path.parent / (out_path.stem + ".rawdump.jsonl")
        pipeline.enable_raw_dump(str(dump_path), args.dump_raw_responses)
        print(f"🔍 Saving the first {args.dump_raw_responses} raw responses "
              f"per model → {dump_path}")

    out_f = out_path.open("a", encoding="utf-8")
    fail_path = out_path.parent / (out_path.stem + ".failures.jsonl")
    fail_f = None
    n_ok = n_fail = n_skip = 0

    def write_ok(row):
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()

    def write_fail(row):
        nonlocal fail_f
        if fail_f is None:
            fail_f = fail_path.open("a", encoding="utf-8")
        fail_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        fail_f.flush()

    counts = {"ok": 0, "skip": 0, "fail": 0}

    def _ok(row):
        write_ok(row); counts["ok"] += 1

    def _skip(row):
        # Missing audio → SIDECAR, never the main output (finding C5). Writing it
        # to the main output marked the cell permanently done, so a corrected
        # re-run skipped it forever.
        write_fail(row); counts["skip"] += 1

    def _fail(row):
        write_fail(row); counts["fail"] += 1

    interrupted = run_worklist(worklist, args.llm_model, args.llm_base_url,
                               args.workers, on_ok=_ok, on_fail=_fail, on_skip=_skip)
    n_ok, n_skip, n_fail = counts["ok"], counts["skip"], counts["fail"]

    out_f.close()
    if fail_f is not None:
        fail_f.close()
    print(f"\n✅ omni done. ok={n_ok} skip(missing audio)={n_skip} failed={n_fail}")
    if n_skip or n_fail:
        print(f"⚠️ {n_skip + n_fail} row(s) → {fail_path} "
              f"(not in the main output; re-run to retry them)")
    print(f"📄 Output: {args.out}")
    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()
