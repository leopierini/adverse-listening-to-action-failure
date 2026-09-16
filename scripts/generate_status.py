#!/usr/bin/env python
"""Generate STATUS.md — where the project is, counted rather than remembered.

WHY THIS EXISTS
---------------
Every stale claim found on 2026-08-18 was a countable fact that a human had to
remember to update: "Step 5 is next" five days after Step 5 finished, "102
tests" when there were 141, "57 cells" when a bare glob said 59. None of them
needed judgement. They needed arithmetic, and arithmetic should not be typed.

So STATUS.md has two halves:

  ABOVE the marker  generated from the data and the checklist. Never edited by
                    hand; regenerating is the only way to change it, which
                    means it cannot be stale.
  BELOW the marker  a short hand-written section for what cannot be counted:
                    what is blocking, what was decided, why. Regeneration never
                    touches it, and refuses outright if the markers are damaged.

    ./venv/bin/python scripts/generate_status.py

Run it after any batch job, any analysis run, and before any session where the
numbers matter. `.claude/hooks/check-docs.sh` reports when it is out of date.
"""

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

BEGIN = "<!-- GENERATED — do not edit below this line by hand; run scripts/generate_status.py -->"
END = "<!-- END GENERATED — everything below is hand-written and is never overwritten -->"

TEMPLATE_TAIL = """
## What is blocking, and why

*Hand-written. Regenerating STATUS.md never touches anything below the marker.
Keep it under ~15 lines: what cannot be counted, and only that.*

_(nothing recorded yet)_

## The last decision that changed course

_(nothing recorded yet)_
"""


class StatusFileDamaged(Exception):
    """STATUS.md exists but its markers are gone, so the hand-written half
    cannot be located. Overwriting would destroy the only part of the file a
    script cannot reproduce, so refuse."""


def write_status(path, generated_block):
    """Replace the generated half of `path`, preserving the hand-written half."""
    path = Path(path)
    header = "# Project status\n\n"
    if not path.exists():
        path.write_text(header + BEGIN + "\n" + generated_block + END + "\n"
                        + TEMPLATE_TAIL)
        return

    text = path.read_text()
    if BEGIN not in text or END not in text:
        raise StatusFileDamaged(
            "%s has no generated-block markers. The hand-written half cannot be "
            "located, and overwriting it would destroy the only content this "
            "script cannot reproduce. Restore the markers (%r and %r) by hand, "
            "or delete the file if its hand-written half is expendable."
            % (path, BEGIN, END))

    before = text[:text.index(BEGIN)]
    after = text[text.index(END) + len(END):]
    path.write_text(before + BEGIN + "\n" + generated_block + END + after)


def collect_facts(repo=REPO):
    """Everything about the project's state that can be counted."""
    import result_sets
    import verify_docs

    repo = Path(repo)
    results_dir = repo / "results"

    def rows_in(paths):
        total = 0
        for p in paths:
            with open(p) as f:
                total += sum(1 for line in f if line.strip())
        return total

    sweep = result_sets.sweep_files(results_dir) if results_dir.exists() else []
    pilots = result_sets.pilot_files(results_dir) if results_dir.exists() else []

    cells, asrs, routers, pathways = set(), set(), set(), set()
    for path in sweep:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                cells.add((r.get("asr_model"), r.get("llm_model"),
                           r.get("degradation"), r.get("snr_db")))
                asrs.add(r.get("asr_model"))
                routers.add(r.get("llm_model"))
                pathways.add(r.get("pathway") or "cascade")

    todo = repo / "EXECUTION_TODO.md"
    next_step = (verify_docs.first_unchecked_step(todo.read_text())
                 if todo.exists() else None)

    facts = {
        "date": date.today().isoformat(),
        "sweep_files": len(sweep),
        "sweep_rows": rows_in(sweep),
        "pilot_files": len(pilots),
        "pilot_rows": rows_in(pilots),
        "cells": len(cells),
        "asrs": len({a for a in asrs if a}),
        "routers": len({r for r in routers if r}),
        "pathways": sorted(pathways),
        "next_step": next_step,
        "tests": verify_docs.actual_test_count(repo),
    }

    fits = repo / "results/analysis/fits/diagnostics.json"
    if fits.exists():
        facts["fits"] = json.loads(fits.read_text())
    return facts


# The design's fixed totals. Sourced from docs/design/DESIGN.md §4 and
# machine-checked there by scripts/verify_tracker_claims.py.
TOTAL_CELLS = 209
TOTAL_ROWS_CASCADE = 23712


def _short_reason(error, limit=70):
    """First sentence of a failure, never cut mid-word.

    The generated table is the first thing a reader is told to trust; a message
    ending "carries no va" undermines that for no reason.
    """
    text = (error or "").split("—")[0].strip().rstrip(".")
    if len(text) <= limit:
        return "not fitted: %s" % text
    cut = text[:limit].rsplit(" ", 1)[0]
    return "not fitted: %s…" % cut


def render(facts):
    pct = 100.0 * facts["cells"] / TOTAL_CELLS if TOTAL_CELLS else 0
    lines = [
        "",
        "*Counted from the files on %s by `scripts/generate_status.py`. "
        "Nothing here is typed by hand.*" % facts["date"],
        "",
        "| | |",
        "|---|---|",
        "| **Cells complete** | **%d of %d** (%.0f%%) |"
        % (facts["cells"], TOTAL_CELLS, pct),
        "| Sweep rows collected | %s in %d files |"
        % ("{:,}".format(facts["sweep_rows"]), facts["sweep_files"]),
        "| Distinct ASR models | %d |" % facts["asrs"],
        "| Distinct routers | %d |" % facts["routers"],
        "| Pathways present | %s |" % ", ".join(facts["pathways"]),
        "| Pilot rows (NOT part of any count above) | %s in %d files |"
        % ("{:,}".format(facts["pilot_rows"]), facts["pilot_files"]),
        "| Test suite | %s tests |" % (facts["tests"] if facts["tests"] else "?"),
        "| **Next step in `EXECUTION_TODO.md`** | **Step %s** |"
        % (facts["next_step"] if facts["next_step"] else "— none open"),
        "",
    ]

    if "fits" in facts:
        lines += ["**Confirmatory models, last run of `fit_mixed_effects.py`:**", "",
                  "| model | outcome | rows | state |", "|---|---|---|---|"]
        for name, entry in facts["fits"].items():
            diag = entry.get("diagnostics") or {}
            state = _short_reason(entry["error"]) if entry.get("error") else "fitted"
            lines.append("| %s | %s | %s | %s |"
                         % (name.upper(), entry.get("outcome", "?"),
                            "{:,}".format(diag.get("rows_out", 0)), state))
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    facts = collect_facts()
    path = REPO / "STATUS.md"
    write_status(path, render(facts))
    print("STATUS.md regenerated: %d cells, %s sweep rows, next = Step %s"
          % (facts["cells"], "{:,}".format(facts["sweep_rows"]),
             facts["next_step"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
