#!/usr/bin/env python
"""Bring pre-2026-08-12 routed files into line with the empty-transcription
convention decided in Step 4c.

WHY THIS EXISTS
---------------
`route_transcripts.py` used to *refuse* an empty transcription: it returned
"noroute" and left `tsa = None`, so the row was dropped from every denominator.
Step 4c (2026-08-12) replaced that with the opposite convention — an empty
transcription is a real end-to-end failure, so the row is stamped
`predicted_intent = "NO_TRANSCRIPTION"`, `tsa = 0`, and no model call is made.

The 77 affected rows in the local Qwen-9B Parakeet files were written under the
OLD convention and were never re-stamped. Everything GCP produces uses the NEW
one. Left alone, those utterances would be *excluded* on the 9B side and counted
as *failures* on the 27B/Gemma side of the paired H2a comparison — and 27 of the
77 sit in babble @ 0 dB, the hardest cells the thesis argues about. The bias runs
against the larger model, i.e. against H2a, which is the direction that would
produce a false "scale does not help" verdict.

The Whisper cascade files were ALSO re-stamped on 2026-08-14: their 18 empty
rows carried `tsa = 0` but `predicted_intent="unknown"` rather than
"NO_TRANSCRIPTION", which categorize_failures.py maps to a different TSH
category — the same physical event split by ASR model.

WHAT IT CHANGES
---------------
Exactly the fields `route_transcripts.route_one` sets on its empty-transcription
branch, and nothing else. No transcript, no WER/CER, no ASR-stage field is
touched. The operation is deterministic and idempotent: running it twice is the
same as running it once.

Originals are never edited in place — `--archive-dir` COPIES the superseded file
aside first, so the collected data stays recoverable (repo rule: results files
are days of compute).
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

# Mirrors route_transcripts.route_one's empty-transcription branch (:106-111).
# Keep the two in sync: this is the same convention, applied after the fact.
NO_TRANSCRIPTION = "NO_TRANSCRIPTION"
NOTE_MARKER = "no_transcription"


def is_empty_transcription(row) -> bool:
    return not (row.get("transcription") or "").strip()


def already_normalized(row) -> bool:
    """True if the row already carries the Step-4c stamp."""
    return (row.get("predicted_intent") == NO_TRANSCRIPTION
            and row.get("tsa") == 0)


def normalize_row(row):
    """Apply the Step-4c convention. Returns (row, changed: bool)."""
    if not is_empty_transcription(row) or already_normalized(row):
        return row, False
    row["predicted_intent_raw"] = NO_TRANSCRIPTION
    row["predicted_intent"] = NO_TRANSCRIPTION
    row["predicted_parameters"] = {}
    row["tsa"] = 0
    notes = row.get("notes") or ""
    if NOTE_MARKER not in notes:
        row["notes"] = notes + " | " + NOTE_MARKER
    return row, True


def normalize_file(in_path: str, out_path: str) -> dict:
    """Rewrite one routed JSONL. Returns a per-file report."""
    rows, changed, empty = [], 0, 0
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if is_empty_transcription(row):
                empty += 1
            row, did = normalize_row(row)
            changed += int(did)
            rows.append(row)
    tmp = Path(str(out_path) + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            # ensure_ascii=False: 525 rows across the results files carry
            # non-ASCII transcriptions. Without it the rewrite escaped them
            # to \\uXXXX — semantically identical, but it byte-rewrote data
            # this script promises not to touch. (audit 2026-08-15)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    return {"file": in_path, "rows": len(rows), "empty": empty, "changed": changed}


def main():
    ap = argparse.ArgumentParser(
        description="Apply the Step-4c empty-transcription convention to "
                    "routed JSONL files written before 2026-08-12.")
    ap.add_argument("inputs", nargs="+", help="routed JSONL files")
    ap.add_argument("--archive-dir", default=None,
                    help="Move each original here before rewriting it in place. "
                         "Required unless --dry-run or --suffix is given.")
    ap.add_argument("--suffix", default=None,
                    help="Write to <stem><suffix>.jsonl instead of in place. "
                         "Leaves the original untouched.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change; write nothing.")
    args = ap.parse_args()

    if not args.dry_run and not args.archive_dir and not args.suffix:
        ap.error("refusing to rewrite in place without --archive-dir "
                 "(or use --suffix / --dry-run)")

    archive: Optional[Path] = None
    if args.archive_dir and not args.dry_run:
        archive = Path(args.archive_dir)
        archive.mkdir(parents=True, exist_ok=True)

    total_rows = total_empty = total_changed = 0
    touched = []
    for in_path in sorted(args.inputs):
        p = Path(in_path)
        if args.dry_run:
            rows = empty = changed = 0
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    rows += 1
                    if is_empty_transcription(row):
                        empty += 1
                        if not already_normalized(row):
                            changed += 1
            rep = {"file": in_path, "rows": rows, "empty": empty, "changed": changed}
        else:
            if args.suffix:
                out_path = str(p.with_name(p.stem + args.suffix + p.suffix))
            else:
                shutil.copy2(str(p), str(archive / p.name))
                out_path = in_path
            rep = normalize_file(in_path, out_path)

        total_rows += rep["rows"]
        total_empty += rep["empty"]
        total_changed += rep["changed"]
        if rep["changed"]:
            touched.append(rep)
            print("  {changed:4d} changed / {empty:4d} empty / {rows:4d} rows  "
                  "{file}".format(**rep))

    print("\n{} file(s) scanned, {} touched".format(len(args.inputs), len(touched)))
    print("rows: {}  empty transcriptions: {}  re-stamped: {}".format(
        total_rows, total_empty, total_changed))
    if args.dry_run:
        print("DRY RUN — nothing written.")
    elif archive is not None:
        print("originals archived to: {}".format(archive))


if __name__ == "__main__":
    main()
