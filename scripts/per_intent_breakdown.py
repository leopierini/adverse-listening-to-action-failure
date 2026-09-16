"""Per-intent TSA breakdown across (asr_model × degradation) cells.

Reveals which intent labels are most/least robust to ASR error propagation —
useful for the Discussion section ("alarm_set is more recoverable than
calendar_remove because…").

Usage:
    python per_intent_breakdown.py results/analysis/*_annotated.jsonl
"""

import argparse
import json
import glob
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    args = p.parse_args()

    buckets = defaultdict(lambda: {"n": 0, "tsa_ok": 0})
    for path in args.files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                asr = (r.get("asr_model") or "?").split("/")[-1]
                deg = r.get("degradation") or "clean"
                intent = r.get("gold_intent") or "?"
                tsa = r.get("tsa")
                if tsa is None:
                    continue
                buckets[(asr, deg, intent)]["n"] += 1
                buckets[(asr, deg, intent)]["tsa_ok"] += tsa

    asrs = sorted(set(k[0] for k in buckets))
    degradations = ["clean", "noise", "reverb", "farfield", "codec", "clipping"]
    intents = sorted(set(k[2] for k in buckets))

    for asr in asrs:
        # Only print if there's data for this ASR
        if not any(k[0] == asr for k in buckets):
            continue
        print(f"\n═══ {asr}: TSA % per intent × degradation ═══")
        header = f"{'intent':28s} " + " ".join(f"{d:>10s}" for d in degradations)
        print(header)
        print("─" * len(header))
        for intent in intents:
            cols = []
            for deg in degradations:
                b = buckets.get((asr, deg, intent))
                if b and b["n"] >= 10:
                    cols.append(f"{100*b['tsa_ok']/b['n']:>8.1f}% ")
                elif b and b["n"] > 0:
                    cols.append(f"  (n={b['n']:>2d})   ")
                else:
                    cols.append("       —   ")
            print(f"{intent:28s} " + " ".join(cols))


if __name__ == "__main__":
    main()
