"""Categorize failures into the Peng (2026) TSH/TCH taxonomy (tracker §2, §11 Phase 5).

TSH (tool-selection hallucination), when tsa == 0:
    missing        — no usable intent (None / "" / MISSING_INTENT / JSON_PARSE_ERROR
                     / NO_TRANSCRIPTION)
    unknown        — the model chose the "unknown" escape hatch (tracked separately)
    hallucinated   — a label outside the 9-intent ontology
    wrong          — a valid ontology label, but not the gold one
TCH (tool-calling hallucination), when tsa == 1 and pf < threshold:
    schema_mismatch — right intent, no parameters at all (gold has some)
    fabricated      — parameters present but NONE soft-match any gold value
    semantic_drift  — partial recovery (some values match, PF below threshold)

Works identically for cascade and omni rows (uses only intent/parameter fields).

Usage:
    python scripts/categorize_failures.py --in <pf_scored>.jsonl --out <categorized>.jsonl
"""

import argparse

import io_guards
import json
from pathlib import Path

from batch_evaluate import VALID_INTENTS
from compute_pf import soft_match

# NO_TRANSCRIPTION (added 2026-08-12 with the empty-transcription convention):
# the ASR produced nothing, so the router was never asked. That is a *missing*
# intent, not a hallucinated one — without this it would fall through to
# "hallucinated" (a label outside the ontology) and inflate the router's
# hallucination count in exactly the cells where empty transcriptions cluster.
NON_PREDICTIONS = {None, "", "MISSING_INTENT", "JSON_PARSE_ERROR", "NO_TRANSCRIPTION"}


def tsh_category(row):
    pred = row.get("predicted_intent")
    if pred in NON_PREDICTIONS:
        return "missing"
    if pred == "unknown":
        return "unknown"
    if pred not in VALID_INTENTS:
        return "hallucinated"
    return "wrong"


def tch_category(row):
    gold = row.get("gold_parameters") or {}
    pred = row.get("predicted_parameters") or {}
    if not pred and gold:
        return "schema_mismatch"
    any_match = any(
        soft_match(gv, pv)
        for gv in gold.values() if gv
        for pv in pred.values() if pv
    )
    return "semantic_drift" if any_match else "fabricated"


def main():
    p = argparse.ArgumentParser(description="Stamp TSH/TCH failure categories (Peng taxonomy).")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pf-threshold", type=float, default=0.5,
                   help="Must match the compute_pf.py threshold used upstream.")
    args = p.parse_args()
    # A scoring script truncates its output before reading its input;
    # --in X --out X therefore empties X and exits 0 (scripts/io_guards.py).
    io_guards.refuse_in_place_or_exit(args.in_path, args.out)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tallies = {}
    n_rows = 0
    with open(args.in_path) as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            row["tsh_category"] = None
            row["tch_category"] = None
            tsa, pf = row.get("tsa"), row.get("pf")
            if tsa == 0:
                row["tsh_category"] = tsh_category(row)
                tallies[f"TSH:{row['tsh_category']}"] = tallies.get(f"TSH:{row['tsh_category']}", 0) + 1
            elif tsa == 1 and pf is not None and pf < args.pf_threshold:
                row["tch_category"] = tch_category(row)
                tallies[f"TCH:{row['tch_category']}"] = tallies.get(f"TCH:{row['tch_category']}", 0) + 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {n_rows} rows → {args.out}")
    for k, v in sorted(tallies.items(), key=lambda kv: -kv[1]):
        print(f"   {k:25s}: {v}")


if __name__ == "__main__":
    main()
