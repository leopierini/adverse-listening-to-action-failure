#!/usr/bin/env python3
"""Study 1b descriptive breakdown: (error_subtype × slot_position) counts.

§11 step 5 and EXECUTION_TODO Step 7 ask for this table by name, with cells of
fewer than 30 flagged exploratory. §1 (`docs/design/DESIGN.md`) explains why it
is descriptive and not confirmatory: the deterministic four-way `error_subtype`
contrast was replaced as H3's predictor by the continuous `phonetic_distance`
on 2026-06-04, and the labels were kept for this breakdown only.

TWO THINGS THIS FILE REFUSES TO DO
----------------------------------
1. **Put `delete` and `insert` in a `None` subtype row.** `annotate_errors.py`
   assigns a subtype to substitutions only; the gaps carry `None` by
   construction, not by accident. They are counted, separately, under their own
   chunk type.
2. **Pool two routers.** These counts describe the ASR stage. Every router
   re-routes the SAME transcripts, so a table built from two routers' files
   counts each transcription error twice and doubles every cell without
   changing a single proportion — the same failure `result_sets.py` exists to
   prevent one level up.

Run (from the repo root):

    ./venv/bin/python scripts/tabulate_error_subtypes.py \
        --out results/analysis/error_subtype_by_slot.txt
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import result_sets                                    # noqa: E402

EXPLORATORY_THRESHOLD = 30
POSITIONS = ("critical", "function")


class MixedRouters(Exception):
    """Rows from more than one `llm_model`. See the module docstring."""


def _router_guard(rows):
    routers = sorted({r.get("llm_model") for r in rows if r.get("llm_model")})
    if len(routers) > 1:
        raise MixedRouters(
            "these rows carry %d routers (%s). The counts describe the ASR "
            "stage; every router re-routes the same transcripts, so pooling "
            "them multiplies every cell by the number of routers. Tabulate one "
            "router's files." % (len(routers), ", ".join(routers)))


def files_for_router(paths, router):
    """The subset of `paths` belonging to one router arm.

    `result_sets.sweep_files()` stopped meaning "one router" on 2026-08-27,
    when the Qwen-27B arm took it from 38 files to 76. The tag is matched with
    a boundary on both sides so that `qwen9b` cannot match `qwen27b` — and, in
    the other direction, so a later `qwen9b_v2` would not be swallowed by it.
    """
    pat = re.compile(r"(?:^|[^a-z0-9])%s(?:[^a-z0-9]|$)" % re.escape(router))
    kept = [p for p in paths if pat.search(Path(p).name)]
    if not kept:
        raise SystemExit(
            "no files for router %r among %d candidates. The known arms are in "
            "scripts/result_sets.py (ROUTER_TAGS = %r)."
            % (router, len(paths), result_sets.ROUTER_TAGS))
    return kept


def tabulate(rows):
    """Counts by (error_subtype, slot_position) and by (chunk type, position)."""
    rows = list(rows)
    _router_guard(rows)
    subs, gaps = Counter(), Counter()
    for r in rows:
        for c in (r.get("error_categories") or []):
            pos = c.get("slot_position")
            if c.get("type") == "substitute":
                subs[(c.get("error_subtype"), pos)] += 1
            else:
                gaps[(c.get("type"), pos)] += 1
    return {"substitutions": dict(subs), "gaps": dict(gaps),
            "n_rows": len(rows),
            "n_utterances": len({r.get("slurp_id") for r in rows})}


def tabulate_by(rows, field):
    """One table per level of `field` (typically `degradation`)."""
    grouped = {}
    for r in rows:
        grouped.setdefault(r.get(field), []).append(r)
    return {k: tabulate(v) for k, v in sorted(grouped.items(),
                                              key=lambda kv: str(kv[0]))}


def exploratory_cells(table):
    """Occupied subtype cells below the threshold the design names.

    An absent cell is not thin, it is absent — reporting `n = 0` as
    "exploratory" would invite a reading of a category that never occurred.
    """
    return sorted(((sub, pos, n)
                   for (sub, pos), n in table["substitutions"].items() if n < EXPLORATORY_THRESHOLD),
                  key=lambda c: (-c[2], str(c[0])))


def exploratory_rollup(split):
    """Thin cells across a `tabulate_by` split, each carrying its level.

    Pooled over all conditions every cell can clear 30 while several cells of
    every single condition do not — which is the case that matters, because the
    Study 1b breakdown is read per condition.
    """
    out = []
    for level, table in split.items():
        for subtype, pos, n in exploratory_cells(table):
            out.append((level, subtype, pos, n))
    return sorted(out, key=lambda c: (str(c[0]), -c[3], str(c[1])))


def format_table(table, title=None):
    subs = table["substitutions"]
    subtypes = sorted({s for s, _ in subs}, key=str)
    positions = [p for p in POSITIONS
                 if any(pos == p for _, pos in subs)] or list(POSITIONS)
    out = []
    if title:
        out.append(title)
    out.append("%-22s %10s %10s %10s" % ("error_subtype", *positions, "total")
               if len(positions) == 2 else
               "%-22s %10s %10s" % ("error_subtype", positions[0], "total"))
    out.append("-" * (22 + 11 * (len(positions) + 1)))
    for s in subtypes:
        cells = [subs.get((s, p), 0) for p in positions]
        marks = ["%d%s" % (n, "*" if 0 < n < EXPLORATORY_THRESHOLD else "")
                 for n in cells]
        out.append("%-22s %s %10d"
                   % (s, " ".join("%10s" % m for m in marks), sum(cells)))
    total = sum(subs.values())
    out.append("%-22s %s %10d"
               % ("ALL SUBSTITUTIONS",
                  " ".join("%10d" % sum(subs.get((s, p), 0) for s in subtypes)
                           for p in positions), total))
    gaps = table["gaps"]
    if gaps:
        out.append("")
        out.append("Chunks with no subtype (annotate_errors.py assigns one to "
                   "substitutions only):")
        for kind in sorted({k for k, _ in gaps}):
            cells = [gaps.get((kind, p), 0) for p in positions]
            out.append("%-22s %s %10d"
                       % (kind, " ".join("%10d" % n for n in cells), sum(cells)))
    out.append("")
    out.append("* = n < %d, report as exploratory (§11 step 5)."
               % EXPLORATORY_THRESHOLD)
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*",
                   help="scored JSONL files; defaults to the cascade sweep.")
    p.add_argument("--by-degradation", action="store_true")
    p.add_argument("--router", default="qwen9b",
                   help="Which router arm to tabulate when no files are given. "
                        "These are ASR-stage counts, so the arm changes nothing "
                        "about them — but pooling two arms would double every "
                        "cell, so one has to be named. Default: qwen9b.")
    p.add_argument("--out", default="results/analysis/error_subtype_by_slot.txt")
    args = p.parse_args()

    files = args.files or files_for_router(
        [str(f) for f in result_sets.sweep_files("results")], args.router)
    rows = []
    for f in files:
        with open(f) as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())

    table = tabulate(rows)
    header = ("Error subtype × slot position — Study 1b descriptive breakdown\n"
              + "=" * 70 + "\n\n"
              + "%d files, %d rows, %d utterances. One router: these are ASR-stage\n"
                "counts, and every router re-routes the same transcripts.\n"
              % (len(files), table["n_rows"], table["n_utterances"]))
    parts = [header, format_table(table, "ALL CONDITIONS")]

    thin = exploratory_cells(table)
    parts.append("\nExploratory cells (n < %d): %s"
                 % (EXPLORATORY_THRESHOLD,
                    ", ".join("%s/%s n=%d" % c for c in thin) if thin
                    else "none — every occupied cell is at or above the threshold."))

    if args.by_degradation:
        split = tabulate_by(rows, "degradation")
        for deg, t in split.items():
            parts.append("\n" + format_table(t, "DEGRADATION = %s" % deg))
        roll = exploratory_rollup(split)
        parts.append("\nExploratory cells once split by degradation (n < %d):"
                     % EXPLORATORY_THRESHOLD)
        parts.extend(["  %-10s %-20s %-9s n=%d" % c for c in roll] if roll
                     else ["  none."])

    text = "\n".join(parts) + "\n"
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print("📄 Saved to %s" % args.out)


if __name__ == "__main__":
    main()
