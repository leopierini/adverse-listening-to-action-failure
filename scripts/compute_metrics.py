"""Compute WER / CER / slot-error rate for a results JSONL file.

Reads a JSONL produced by batch_evaluate.py, joins each row with the SLURP gold
annotation (sentence + entities) to compute:
    wer              — word error rate, full sentence
    cer              — character error rate, full sentence
    slot_error_rate  — WER restricted to the gold slot tokens
                       (date / time / person / event_name / list_name / item_name / ...)

Writes a new JSONL with the metric columns populated, preserving every other field.

Usage:
    python compute_metrics.py \\
        --in  results/clean_baseline_full.jsonl \\
        --out results/clean_baseline_full_metrics.jsonl
"""

import argparse

import io_guards
import json
import re
import string
from pathlib import Path

from typing import Optional

import jiwer

DEFAULT_DATASET = "dataset/gold_devel_416.jsonl"  # SLURP/ removed 2026-06-02

# Normalize transcriptions and references identically before scoring.
# (Lowercase, strip leading/trailing whitespace, drop punctuation, collapse internal whitespace.)
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def load_gold_index(dataset_path: str) -> dict:
    """slurp_id → {sentence, tokens, entities}."""
    idx = {}
    with open(dataset_path) as f:
        for line in f:
            d = json.loads(line)
            sid = d.get("slurp_id")
            if sid is None:
                continue
            idx[sid] = {
                "sentence": d.get("sentence", ""),
                "tokens": d.get("tokens", []),
                "entities": d.get("entities", []),
            }
    return idx


def slot_tokens_from_gold(gold: dict) -> tuple:
    """Return (slot_words_lowercase_list, slot_types_set)."""
    tokens = gold.get("tokens", [])
    entities = gold.get("entities", [])
    slot_words = []
    slot_types = set()
    for ent in entities:
        slot_types.add(ent.get("type"))
        for idx in ent.get("span", []):
            if 0 <= idx < len(tokens):
                surface = tokens[idx].get("surface", "")
                if surface:
                    slot_words.append(surface.lower())
    return slot_words, slot_types


def compute_slot_error_rate(slot_words: list, hyp_text_norm: str) -> Optional[float]:
    """How many of the slot words appear (as whole tokens) in the normalized hypothesis?
    Returns the fraction MISSING — i.e. higher = more slot errors. Returns None if no slots."""
    if not slot_words:
        return None
    hyp_tokens = set(hyp_text_norm.split())
    missing = sum(1 for w in slot_words if w.lower() not in hyp_tokens)
    return missing / len(slot_words)


def main():
    p = argparse.ArgumentParser(description="Compute WER / CER / slot-ER for a results JSONL.")
    p.add_argument("--in", dest="in_path", required=True, help="Input JSONL from batch_evaluate.py")
    p.add_argument("--out", required=True, help="Output JSONL with metric columns filled.")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="SLURP devel.jsonl path for gold annotations.")
    args = p.parse_args()
    # A scoring script truncates its output before reading its input;
    # --in X --out X therefore empties X and exits 0 (scripts/io_guards.py).
    io_guards.refuse_in_place_or_exit(args.in_path, args.out)

    print(f"📋 Loading SLURP gold index from {args.dataset}")
    gold = load_gold_index(args.dataset)
    print(f"   {len(gold)} gold entries")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_with_metrics = 0
    n_no_transcription = 0
    n_no_gold = 0
    n_empty_hyp = 0

    with open(args.in_path) as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1

            if row.get("pathway") == "omni":
                # No transcript to score. Still ensure gold_parameters is filled
                # (PF needs it) so the omni rows are gradable downstream.
                if not row.get("gold_parameters"):
                    g = gold.get(row.get("slurp_id"))
                    if g is not None:
                        gp = {}
                        for ent in g.get("entities", []):
                            etype = ent.get("type")
                            val = " ".join(
                                g["tokens"][i].get("surface", "")
                                for i in ent.get("span", [])
                                if 0 <= i < len(g["tokens"]))
                            key_name, k = etype, 2
                            while key_name in gp:
                                key_name = f"{etype}_{k}"; k += 1
                            gp[key_name] = val
                        row["gold_parameters"] = gp
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            sid = row.get("slurp_id")
            hyp = row.get("transcription")
            g = gold.get(sid)

            if hyp is None:
                n_no_transcription += 1
            elif g is None:
                n_no_gold += 1
            else:
                ref_norm = normalize_text(g["sentence"])
                hyp_norm = normalize_text(hyp)
                if ref_norm:
                    slot_words, _slot_types = slot_tokens_from_gold(g)
                    if not hyp_norm:
                        # Empty transcription = total deletion. Must score WER=1.0,
                        # not None — leaving it None silently drops these rows from
                        # every downstream analysis and biases WER/absorption.
                        n_empty_hyp += 1
                        row["wer"] = 1.0
                        row["cer"] = 1.0
                        row["slot_error_rate"] = 1.0 if slot_words else None
                    else:
                        try:
                            row["wer"] = float(jiwer.wer(ref_norm, hyp_norm))
                        except Exception:
                            row["wer"] = None
                        try:
                            row["cer"] = float(jiwer.cer(ref_norm, hyp_norm))
                        except Exception:
                            row["cer"] = None
                        # Slot-error rate using gold entities
                        row["slot_error_rate"] = compute_slot_error_rate(slot_words, hyp_norm)
                    # Also store gold_parameters as the entity tokens (mostly for downstream PF computation)
                    if not row.get("gold_parameters"):
                        gp = {}
                        for ent in g.get("entities", []):
                            etype = ent.get("type")
                            val = " ".join(
                                g["tokens"][i].get("surface", "")
                                for i in ent.get("span", [])
                                if 0 <= i < len(g["tokens"])
                            )
                            # Duplicate slot types must not overwrite each other
                            # (9/416 rows carry >1 entity of the same type).
                            key_name = etype
                            k = 2
                            while key_name in gp:
                                key_name = f"{etype}_{k}"
                                k += 1
                            gp[key_name] = val
                        row["gold_parameters"] = gp
                    n_with_metrics += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅ Wrote {n_rows} rows → {args.out}")
    print(f"   With metrics filled:  {n_with_metrics}")
    print(f"   Empty transcriptions (scored WER=1.0): {n_empty_hyp}")
    print(f"   No transcription:     {n_no_transcription}")
    print(f"   No gold (slurp_id missing in dataset): {n_no_gold}")


if __name__ == "__main__":
    main()
