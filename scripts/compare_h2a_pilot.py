#!/usr/bin/env python
"""Paired 9B-vs-27B comparison for the Step-5 H2a pilot.

WHY THIS EXISTS
---------------
The session-1 runbook used to say "compare to the same rows on 9B" after taking
`head -n 100` of a transcripts file. Those are NOT the same rows: with
`--workers > 1`, `route_transcripts.py` writes in *completion* order
(`as_completed`, :244), so a routed file holds the same 416 utterances in a
different order than its input. Measured 2026-08-14: the first 100 rows of
`parakeet_clean.jsonl` and `parakeet_clean_qwen9b.jsonl` overlap on only 98
slurp_ids, and they diverge from index 0.

Comparing by position therefore puts two utterances against a *different*
utterance. On a 100-row sample whose job is to decide whether to kill an entire
hypothesis, that is avoidable noise. This script joins on the row key instead.

It also fixes the metric. The design (tracker §8) specifies **EES**
(End-to-End Success: correct intent AND correct parameters), not TSA
(Task Success Accuracy: intent only) — using `tsa` where `ees` belongs is a
mistake this project has already made once. EES exists only in `_scored.jsonl`
files, so both inputs must have been through
compute_metrics -> annotate_errors -> compute_pf first.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from scipy import stats

# The row identity, per tracker §10. Deliberately excludes llm_model: that is
# the thing being varied.
KEY_FIELDS = ("slurp_id", "asr_model", "degradation", "snr_db")


def row_key(row):
    return tuple(row.get(f) for f in KEY_FIELDS)


def load_scored(path, outcome):
    rows = {}
    models = Counter()
    missing_outcome = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get(outcome) is None:
                missing_outcome += 1
            rows[row_key(row)] = row
            models[row.get("llm_model")] += 1
    return rows, models, missing_outcome


def main():
    ap = argparse.ArgumentParser(
        description="Paired EES comparison between two routers on identical rows.")
    ap.add_argument("--baseline", required=True,
                    help="scored JSONL for the incumbent router (Qwen-9B)")
    ap.add_argument("--candidate", required=True,
                    help="scored JSONL for the new router (Qwen-27B)")
    ap.add_argument("--outcome", default="ees", choices=["ees", "ees_strict", "tsa"],
                    help="design specifies ees (tracker §8); tsa is diagnostic only")
    args = ap.parse_args()

    base, base_models, base_missing = load_scored(args.baseline, args.outcome)
    cand, cand_models, cand_missing = load_scored(args.candidate, args.outcome)

    print("baseline : {}".format(args.baseline))
    print("           {} rows, models={}".format(len(base), dict(base_models)))
    print("candidate: {}".format(args.candidate))
    print("           {} rows, models={}".format(len(cand), dict(cand_models)))

    if set(base_models) & set(cand_models):
        print("\n  WARNING: both files report the same llm_model. If you forgot "
              "--llm-base-url the candidate may have routed through the "
              "baseline server. Check llm_model_served on the candidate rows.")

    shared = sorted(set(base) & set(cand))
    only_cand = len(cand) - len(shared)
    print("\npaired rows: {}   (candidate rows with no baseline match: {})"
          .format(len(shared), only_cand))
    if not shared:
        raise SystemExit("no rows in common — check that both files cover the "
                         "same ASR/degradation/SNR cell")

    # Rows where either side has a null outcome cannot enter a paired test.
    usable = [k for k in shared
              if base[k].get(args.outcome) is not None
              and cand[k].get(args.outcome) is not None]
    dropped = len(shared) - len(usable)
    if dropped:
        print("  dropped {} pair(s) with a null {} on one side".format(dropped, args.outcome))
        print("  (if this is non-zero, the empty-transcription convention may "
              "still differ between the two files — see "
              "scripts/normalize_no_transcription.py)")

    b_hits = sum(base[k][args.outcome] for k in usable)
    c_hits = sum(cand[k][args.outcome] for k in usable)
    n = len(usable)

    print("\n--- {} on {} paired rows ---".format(args.outcome.upper(), n))
    print("  baseline : {:3d}/{} = {:.1%}".format(b_hits, n, b_hits / n))
    print("  candidate: {:3d}/{} = {:.1%}".format(c_hits, n, c_hits / n))
    print("  gap      : {:+.1%}".format((c_hits - b_hits) / n))

    # McNemar: only the discordant pairs carry information about the difference.
    cand_only = sum(1 for k in usable
                    if cand[k][args.outcome] == 1 and base[k][args.outcome] == 0)
    base_only = sum(1 for k in usable
                    if cand[k][args.outcome] == 0 and base[k][args.outcome] == 1)
    disc = cand_only + base_only
    print("\n  discordant pairs: {}".format(disc))
    print("    candidate right, baseline wrong: {}".format(cand_only))
    print("    baseline right, candidate wrong: {}".format(base_only))
    if disc:
        p = stats.binomtest(cand_only, disc, 0.5).pvalue
        print("    exact McNemar p = {:.4f}".format(p))
    else:
        p = 1.0
        print("    exact McNemar p = 1.0 (identical on every row)")

    print("\n--- the stop condition (runbook §4) ---")
    if c_hits > b_hits and p < 0.05:
        print("  27B shows a gap over 9B. H2a is alive — proceed to Step 6.")
    elif c_hits > b_hits:
        print("  27B is ahead but the gap is not significant at n={}.".format(n))
        print("  Underpowered, not dead: a 100-row pilot cannot resolve a small gap.")
        print("  Judgement call — discuss with Testolin before committing to the sweep.")
    else:
        print("  27B does NOT beat 9B. STOP and discuss with Testolin.")
        print("  'Scale does not help' is a valid finding and is much cheaper to")
        print("  learn now than after the full sweep.")


if __name__ == "__main__":
    main()
