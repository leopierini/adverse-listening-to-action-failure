"""Guards for the metric chain's file handling.

WHY THIS EXISTS
---------------
`compute_metrics.py`, `annotate_errors.py`, `compute_pf.py` and
`categorize_failures.py` all stream input to output:

    with open(args.in_path) as fin, out_path.open("w") as fout:

Opening the output with "w" truncates it immediately, before the input handle
has read a byte. Pointed at the same file, the script therefore reads an empty
file, writes nothing, prints "✅ Wrote 0 rows" and exits 0.

On 2026-08-19 the session-2 runbook's final step did exactly that:

    categorize_failures.py --in "${b}_scored.jsonl" --out "${b}_scored.jsonl"

Reproduced on a 5-row copy of a real scored file: 5 rows and 5,028 bytes in,
"✅ Wrote 0 rows", exit 0, 0 bytes out. On the last step of a session that has
just spent hours of A100 time, against a file `CLAUDE.md` says must never be
modified, with a green checkmark on the way past.

The runbook line was wrong and is fixed. This is the other half: the scripts
must refuse, so that no future session can step on it — including one that
reaches for `--in X --out X` because it looks like the obvious way to annotate
in place.
"""

import os
import sys
from pathlib import Path


class InPlaceWriteRefused(Exception):
    """--out names the same file as --in."""


def _same_file(a, b):
    """True when two paths name one file, following what actually exists.

    `results/x.jsonl` and `results/./x.jsonl` are the same file; string
    comparison says otherwise. `os.path.samefile` answers correctly but raises
    when the output does not exist yet, which is the normal case.
    """
    a, b = Path(a), Path(b)
    if a.exists() and b.exists():
        try:
            return os.path.samefile(str(a), str(b))
        except OSError:
            pass
    return a.resolve() == b.resolve()


def refuse_in_place(in_path, out_path):
    """Raise unless `out_path` is a different file from `in_path`.

    Called before either handle is opened. Raising rather than warning is
    deliberate: the failure it prevents is silent and total, and a warning on a
    long console scroll is a warning nobody reads.
    """
    if _same_file(in_path, out_path):
        raise InPlaceWriteRefused(
            "refusing to write in place: --out is the same file as --in (%s). "
            "These scripts truncate the output before reading the input, so "
            "this would empty the file and exit 0. Write to a new path, then "
            "replace the original yourself if that is really what you want."
            % in_path)


def refuse_in_place_or_exit(in_path, out_path):
    """`refuse_in_place`, reported as a clean CLI error instead of a traceback."""
    try:
        refuse_in_place(in_path, out_path)
    except InPlaceWriteRefused as exc:
        sys.stderr.write("ERROR: %s\n" % exc)
        raise SystemExit(2)
