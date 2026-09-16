"""Quick summary of a results JSONL: per-(asr, degradation, snr, llm) TSA + per-intent breakdown.

Usage:
    python summarize_results.py results/clean_baseline_full.jsonl
    python summarize_results.py results/1a_noise_snr10.jsonl
    python summarize_results.py results/*.jsonl  # concatenated summary
"""

import argparse
import json
import sys
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--per-intent", action="store_true", help="Show per-intent breakdown.")
    p.add_argument("--errors-only", action="store_true",
                   help="Show only the failing rows (where tsa=0).")
    args = p.parse_args()

    # cell_key = (asr_model_short, degradation, snr_db, llm_model)
    cells = defaultdict(lambda: {"total": 0, "tsa": 0, "by_intent": defaultdict(lambda: {"n": 0, "ok": 0})})
    errors = []

    for path in args.files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                asr = (row.get("asr_model") or "?").split("/")[-1]
                deg = row.get("degradation") or "clean"
                snr = row.get("snr_db")
                llm = row.get("llm_model") or "?"
                key = (asr, deg, snr, llm)
                tsa = row.get("tsa")
                if tsa is None:
                    continue
                cells[key]["total"] += 1
                cells[key]["tsa"] += tsa
                intent = row.get("gold_intent", "?")
                cells[key]["by_intent"][intent]["n"] += 1
                cells[key]["by_intent"][intent]["ok"] += tsa
                if tsa == 0:
                    errors.append({
                        "slurp_id": row.get("slurp_id"),
                        "scenario": row.get("scenario"),
                        "gold_intent": row.get("gold_intent"),
                        "predicted_intent": row.get("predicted_intent"),
                        "transcription": row.get("transcription"),
                        "gold_sentence": row.get("gold_sentence"),
                        "cell": f"{asr} {deg} snr={snr} llm={llm}",
                    })

    if args.errors_only:
        for e in errors:
            print(f"[{e['cell']}] slurp_id={e['slurp_id']} {e['scenario']}: gold={e['gold_intent']} pred={e['predicted_intent']}")
            print(f"   gold: {e['gold_sentence']}")
            print(f"   hyp:  {e['transcription']}")
        return

    print("=" * 80)
    print(f"{'ASR':35s} {'DEG':10s} {'SNR':>5s} {'LLM':15s} {'N':>5s} {'TSA':>7s}")
    print("=" * 80)
    for (asr, deg, snr, llm), s in sorted(cells.items()):
        acc = 100.0 * s["tsa"] / s["total"] if s["total"] else 0.0
        snr_str = "—" if snr is None else f"{snr}"
        print(f"{asr:35s} {deg:10s} {snr_str:>5s} {llm:15s} {s['total']:>5d} {acc:>6.2f}%")
    print("=" * 80)

    if args.per_intent:
        print("\nPer-intent breakdown:")
        for (asr, deg, snr, llm), s in sorted(cells.items()):
            snr_str = "—" if snr is None else f"{snr}"
            print(f"\n  {asr} | {deg} | snr={snr_str} | {llm}")
            for intent, b in sorted(s["by_intent"].items()):
                acc = 100.0 * b["ok"] / b["n"] if b["n"] else 0
                print(f"    {intent:25s} {b['n']:>4d}  {acc:>6.2f}%")


if __name__ == "__main__":
    main()
