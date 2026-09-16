"""Compute Parameter Fidelity (PF) and End-to-End Success (EES) per row.

PF is computed with **value-based soft matching** because the LLM router and the
SLURP gold annotation use different slot vocabularies (e.g., the LLM puts a date
under `date` while SLURP might call it `timeofday`, or values are partial like
"Bola" → "Bola birthday party"). Strict key-equality matching would massively
underestimate PF.

The rule:
  For each gold entity (slot_type, gold_value):
    matched = ANY (pred_key, pred_value) such that:
      lowercase gold_value substring-overlaps lowercase pred_value
      (either direction; minimum overlap 3 chars for proper nouns)

PF = matched_gold_entities / total_gold_entities  (recall)
EES = (tsa == 1) AND (PF >= --pf-threshold, default 0.5)

A second, stricter `ees_strict` column requires PF == 1.0.

Usage:
    python compute_pf.py \\
        --in  results/analysis/clean_baseline_full_annotated.jsonl \\
        --out results/analysis/clean_baseline_full_pf.jsonl
"""

import argparse

import io_guards
import json
import re
from pathlib import Path

_NUMBER_WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
}


def normalize_value(s: str) -> str:
    """Lowercase, strip, normalize number-words → digits, collapse whitespace."""
    if s is None:
        return ""
    s = str(s).lower().strip()
    # Replace number words with digits at word boundaries
    for w, d in _NUMBER_WORD_TO_DIGIT.items():
        s = re.sub(rf"\b{w}\b", d, s)
    # Strip filler punctuation
    s = re.sub(r"[.,!?;:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Function words that must not count as "shared content token" in soft_match —
# otherwise gold "the office party" vs pred "the morning" matches on "the".
_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "at",
    "my", "me", "is", "it", "this", "that", "with", "from", "all", "any",
}


def soft_match(gold_value, pred_value) -> bool:
    """True if either string is contained in the other (post-normalization, len ≥ 3)."""
    gv = normalize_value(gold_value)
    pv = normalize_value(pred_value)
    if not gv or not pv:
        return False
    # Exact match
    if gv == pv:
        return True
    # Substring (need at least 3 characters to count, to avoid spurious matches)
    if len(gv) >= 3 and gv in pv:
        return True
    if len(pv) >= 3 and pv in gv:
        return True
    # Token overlap (for multi-word slots)
    gset = set(gv.split())
    pset = set(pv.split())
    if gset and pset and (gset & pset):
        # At least one shared content token of ≥ 3 chars (stopwords excluded)
        if any(len(t) >= 3 and t not in _STOPWORDS for t in (gset & pset)):
            return True
    return False


def compute_row_pf(gold_params, pred_params):
    """Returns (pf, n_gold, n_matched). pf is None if there are no gold entities."""
    if not gold_params:
        return None, 0, 0
    pred_vals = list((pred_params or {}).values())
    matched = 0
    for gold_key, gold_val in gold_params.items():
        if not gold_val:
            continue
        if any(soft_match(gold_val, pv) for pv in pred_vals if pv):
            matched += 1
    n_gold = sum(1 for v in gold_params.values() if v)
    pf = matched / n_gold if n_gold else None
    return pf, n_gold, matched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pf-threshold", type=float, default=0.5,
                   help="PF threshold for the lenient EES.")
    args = p.parse_args()
    # A scoring script truncates its output before reading its input;
    # --in X --out X therefore empties X and exits 0 (scripts/io_guards.py).
    io_guards.refuse_in_place_or_exit(args.in_path, args.out)

    in_path = Path(args.in_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_pf_computed = 0
    n_ees_ok = 0
    n_ees_strict_ok = 0

    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_rows += 1
            gold = r.get("gold_parameters") or {}
            pred = r.get("predicted_parameters") or {}
            pf, n_gold, n_matched = compute_row_pf(gold, pred)
            r["pf"] = pf
            r["pf_n_gold"] = n_gold
            r["pf_n_matched"] = n_matched
            tsa = r.get("tsa")
            if pf is not None:
                n_pf_computed += 1
            if tsa is None:
                # Unrouted/failed row — EES must stay missing, not become a
                # spurious failure (int(None == 1) would silently yield 0).
                r["ees"] = None
                r["ees_strict"] = None
            elif pf is not None:
                r["ees"] = int(tsa == 1 and pf >= args.pf_threshold)
                r["ees_strict"] = int(tsa == 1 and pf >= 1.0)
            else:
                # No gold entities → EES = TSA (we can't fail on params we don't have)
                r["ees"] = tsa
                r["ees_strict"] = tsa
            if r["ees"] == 1:
                n_ees_ok += 1
            if r["ees_strict"] == 1:
                n_ees_strict_ok += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {n_rows} rows → {args.out}")
    print(f"   PF computed (had gold entities): {n_pf_computed}")
    print(f"   EES (lenient, pf>={args.pf_threshold}): {n_ees_ok} ({100*n_ees_ok/n_rows:.1f}%)")
    print(f"   EES strict (pf=1.0):              {n_ees_strict_ok} ({100*n_ees_strict_ok/n_rows:.1f}%)")


if __name__ == "__main__":
    main()
