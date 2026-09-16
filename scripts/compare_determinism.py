#!/usr/bin/env python
"""Compare two identical passes of the same model and say how far they drifted.

Why this is a script and not a snippet
--------------------------------------
The determinism probe is run once per served model and its numbers go into
Methods.  Until 2026-08-30 it was an ad-hoc snippet pasted from the runbook,
which is how the Gemma-31B probe came to be reported as "0/50 raw" from a pair
that was never scored, and how the Gemma-12B probe nearly reported 2/950 when
the answer including slot parameters is 27/950.

The comparison has three levels, and they disagree by an order of magnitude:

  1. `predicted_intent_raw` alone     — what a careless probe sees
  2. intent **or** `predicted_parameters` — the raw generation, honestly
  3. any SCORED metric moving          — the only level that can change a result

Level 3 is the runbook's tripwire.  Level 2 is the number Methods quotes as the
model's raw drift rate.  Reporting 1 as if it were 2 understates the drift.

Usage
-----
    ./venv/bin/python scripts/compare_determinism.py --a pass_a.jsonl --b pass_b.jsonl

Both `.jsonl` and `.jsonl.gz` are accepted.
"""
from __future__ import print_function

import argparse
import gzip
import json
import sys
from collections import OrderedDict

# The metrics whose movement stops a sweep. `ees_strict` is included even where
# a file does not carry it: absent on both sides it can never differ, and naming
# it here means a file that DOES carry it is not silently ignored.
SCORED_FIELDS = ("tsa", "pf", "ees", "ees_strict")

# One row is identified by the cell it belongs to plus the utterance. `asr_model`
# is null on omni rows and populated on cascade rows; including it keeps the two
# pathways comparable with one key.
KEY_FIELDS = ("slurp_id", "asr_model", "degradation", "snr_db")


def _open(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def load(path):
    rows = OrderedDict()
    with _open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[tuple(row.get(f) for f in KEY_FIELDS)] = row
    return rows


def _canonical(value):
    """Compare dicts by content, not by key order or float formatting."""
    return json.dumps(value, sort_keys=True, default=str)


def compare(a_rows, b_rows):
    shared = [k for k in a_rows if k in b_rows]
    intent_only = []
    intent_or_params = []
    scored_moved = []

    for key in shared:
        a, b = a_rows[key], b_rows[key]
        intent_differs = a.get("predicted_intent_raw") != b.get("predicted_intent_raw")
        params_differ = (_canonical(a.get("predicted_parameters"))
                         != _canonical(b.get("predicted_parameters")))
        if intent_differs:
            intent_only.append(key)
        if intent_differs or params_differ:
            intent_or_params.append(key)
        if any(a.get(f) != b.get(f) for f in SCORED_FIELDS):
            scored_moved.append(key)

    return {
        "compared": len(shared),
        "only_in_a": len([k for k in a_rows if k not in b_rows]),
        "only_in_b": len([k for k in b_rows if k not in a_rows]),
        "intent_only": intent_only,
        "intent_or_params": intent_or_params,
        "scored_moved": scored_moved,
    }


def _mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def outcome_drift(a_rows, b_rows, outcome="ees"):
    """Mean outcome on each side, and the per-cell picture behind it.

    A rate of differing rows says how often the model wavered; it does not say
    whether the wavering moves a result.  A cell mean does.
    """
    shared = [k for k in a_rows if k in b_rows]
    mean_a = _mean([a_rows[k].get(outcome) for k in shared])
    mean_b = _mean([b_rows[k].get(outcome) for k in shared])

    cells = OrderedDict()
    for key in shared:
        cell = (a_rows[key].get("degradation"), a_rows[key].get("snr_db"))
        cells.setdefault(cell, [[], []])
        cells[cell][0].append(a_rows[key].get(outcome))
        cells[cell][1].append(b_rows[key].get(outcome))

    per_cell = []
    for cell, (va, vb) in cells.items():
        ma, mb = _mean(va), _mean(vb)
        if ma is None or mb is None:
            continue
        per_cell.append((cell, ma, mb, abs(mb - ma), len(va)))

    per_cell.sort(key=lambda item: -item[3])
    return {
        "outcome": outcome,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "cells": per_cell,
        "mean_abs_cell_delta": _mean([item[3] for item in per_cell]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--a", required=True, help="first pass (.jsonl or .jsonl.gz)")
    parser.add_argument("--b", required=True, help="second pass")
    parser.add_argument("--outcome", default="ees")
    parser.add_argument("--top-cells", type=int, default=5)
    args = parser.parse_args(argv)

    a_rows, b_rows = load(args.a), load(args.b)
    result = compare(a_rows, b_rows)
    drift = outcome_drift(a_rows, b_rows, args.outcome)

    n = result["compared"]
    if n == 0:
        print("🛑 no row is present in both passes — the key fields do not line up")
        return 2

    def rate(rows):
        return "%d / %d (%.2f%%)" % (len(rows), n, 100.0 * len(rows) / n)

    print("rows compared                          : %d" % n)
    if result["only_in_a"] or result["only_in_b"]:
        print("⚠️ rows present in only one pass        : a=%d  b=%d"
              % (result["only_in_a"], result["only_in_b"]))
    print("differ on predicted_intent_raw ALONE   : %s" % rate(result["intent_only"]))
    print("differ on intent OR parameters         : %s   <- the raw drift rate" % rate(result["intent_or_params"]))
    print("move a SCORED metric %-17s: %s   <- the tripwire"
          % ("(%s)" % ",".join(SCORED_FIELDS), rate(result["scored_moved"])))
    print("")
    if drift["mean_a"] is not None:
        print("mean %s: %.4f -> %.4f  = %+.2f pp"
              % (drift["outcome"], drift["mean_a"], drift["mean_b"],
                 100.0 * (drift["mean_b"] - drift["mean_a"])))
    if drift["mean_abs_cell_delta"] is not None:
        print("mean |delta| per cell (%d cells)        : %.2f pp"
              % (len(drift["cells"]), 100.0 * drift["mean_abs_cell_delta"]))
    for cell, ma, mb, delta, size in drift["cells"][:args.top_cells]:
        print("   %-10s snr=%-5s n=%-4d  %.4f -> %.4f  (%.2f pp)"
              % (cell[0], cell[1], size, ma, mb, 100.0 * delta))
    print("")
    if result["scored_moved"]:
        print("TRIPWIRE: FIRED — a scored metric moved on %d of %d rows."
              % (len(result["scored_moved"]), n))
        print("  The runbook's prescription is to stop and re-run at --workers 1.")
        print("  §14 has already overridden that once, deliberately. Whether this")
        print("  magnitude is inside that override is a judgement, not a number.")
    else:
        print("TRIPWIRE: CLEAR — no scored metric moved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
