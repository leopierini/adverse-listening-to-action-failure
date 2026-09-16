"""For each utterance, show the side-by-side clean → degraded behavior.

Reads:
    --clean   JSONL with clean-audio rows (typically results/clean_baseline_annotated.jsonl)
    --noisy   JSONL with degraded-audio rows (typically results/analysis/1a_*_annotated.jsonl)

For each (slurp_id, asr_model) pair present in both files, prints:
    - clean transcription, predicted intent, outcome
    - degraded transcription, predicted intent, outcome
Classification is delegated to absorption_metrics.classify_pair, so the
ABSORBED / PROPAGATED counts here and gain_paired in analyze_absorption.py
can never drift apart. The outcome column defaults to `ees` (tracker §8);
pass --outcome tsa to reproduce the pre-2026-08-12 behaviour.

    PROPAGATED   clean ok, degraded failed, WER>0  (the ASR error killed it)
    ABSORBED     clean ok, degraded ok, WER>0
    RECOVERED    clean failed, degraded ok
    SAME_FAIL    failed in both
    SAME_OK      ok in both, WER=0 (no ASR error)
    NEW_ERROR    clean ok, degraded failed, but WER=0 — LLM nondeterminism (shouldn't happen at temp=0)

Usage:
    python compare_clean_vs_degraded.py \\
        --clean results/clean_baseline_annotated.jsonl \\
        --noisy results/analysis/1a_noise_snr10_annotated.jsonl \\
        --asr-model whisper-large-v3-turbo \\
        --limit 30
"""

import argparse
import json
from collections import Counter

from absorption_metrics import classify_pair


def load_indexed(path, asr_filter=None):
    """Index rows by (slurp_id, asr_model, llm_model). Warns when the file holds more than
    one row per key (e.g. several degradations/SNRs mixed in one JSONL) — in
    that case the FIRST row wins and the comparison is ambiguous: split the
    file or pass a single-condition file instead."""
    out = {}
    n_dups = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            asr = (r.get("asr_model") or "").split("/")[-1]
            if asr_filter and asr_filter not in asr:
                continue
            # `asr_model` belongs in the key. Without it, a substring filter
            # such as --asr-model whisper collides whisper-base with
            # whisper-large-v3-turbo, and "first row wins" below silently
            # discards half the rows — pairing one ASR's clean row with the
            # other's degraded row. absorption_metrics.item_key has always
            # included it; this file drifted. (audit 2026-08-14)
            key = (r.get("slurp_id"), r.get("asr_model"), r.get("llm_model"))
            if key in out:
                n_dups += 1
                continue
            out[key] = r
    if n_dups:
        print(f"⚠️ {path}: {n_dups} duplicate (slurp_id, asr_model, llm_model) rows ignored "
              f"(file mixes conditions? results may be ambiguous)")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", required=True)
    p.add_argument("--noisy", required=True)
    p.add_argument("--asr-model", default="whisper-large-v3-turbo",
                   help="Substring to match the asr_model field on (e.g. whisper-large-v3-turbo, parakeet, whisper-base).")
    p.add_argument("--show", default="PROPAGATED",
                   help="Category to print (PROPAGATED / ABSORBED / RECOVERED / ALL). Default PROPAGATED.")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--outcome", default="ees", choices=["ees", "tsa", "ees_strict"],
                   help="Success column to classify on. Default ees (tracker §8); "
                        "tsa reproduces the pre-2026-08-12 behaviour.")
    args = p.parse_args()

    clean_idx = load_indexed(args.clean, args.asr_model)
    deg_idx = load_indexed(args.noisy, args.asr_model)
    overlap = sorted(set(clean_idx) & set(deg_idx))
    print(f"clean rows: {len(clean_idx)}, degraded rows: {len(deg_idx)}, overlap: {len(overlap)}")

    counts = Counter()
    examples = {"PROPAGATED": [], "ABSORBED": [], "RECOVERED": [], "SAME_FAIL": [],
                "SAME_OK": [], "NEW_ERROR": []}
    for sid in overlap:
        c = clean_idx[sid]
        d = deg_idx[sid]
        cat = classify_pair(c, d, args.outcome)
        counts[cat] += 1
        if cat in examples:
            examples[cat].append((sid, c, d))

    n_total = sum(counts.values())
    print(f"\nCategory counts (n={n_total}):")
    for k in ["PROPAGATED", "ABSORBED", "RECOVERED", "SAME_OK", "SAME_FAIL", "NEW_ERROR", "OTHER"]:
        v = counts.get(k, 0)
        pct = 100.0 * v / n_total if n_total else 0
        print(f"  {k:12s} {v:4d}  {pct:5.1f}%")

    if n_total:
        n_absorbed = counts.get("ABSORBED", 0)
        n_paired = n_absorbed + counts.get("PROPAGATED", 0)
        if n_paired:
            print(f"\n  Retention on the paired subset "
                  f"(clean {args.outcome}=1 AND degraded WER>0, n={n_paired}):")
            print(f"    P(stay correct) = {100.0 * n_absorbed / n_paired:.1f}%")
        else:
            print("\n  No paired items — nothing to report.")

    if args.show in examples:
        cats = [args.show]
    elif args.show == "ALL":
        cats = ["PROPAGATED", "ABSORBED", "RECOVERED"]
    else:
        cats = []
    for cat in cats:
        print(f"\n────── examples: {cat} ──────")
        for key, c, d in examples[cat][:args.limit]:
            wer_str = f"{d.get('wer'):.2f}" if d.get("wer") is not None else "—"
            print(f"slurp_id={key[0]}  scenario={c.get('scenario')}  gold_intent={c.get('gold_intent')}")
            print(f"   clean : '{(c.get('transcription') or '').strip()}' → {c.get('predicted_intent')}  tsa={c.get('tsa')}")
            print(f"   noisy : '{(d.get('transcription') or '').strip()}' → {d.get('predicted_intent')}  tsa={d.get('tsa')}  wer={wer_str}")
            print(f"   gold  : '{c.get('gold_sentence')}'")
            print()


if __name__ == "__main__":
    main()
