"""The four scoring scripts must refuse `--in X --out X`.

WHY: on 2026-08-19 the session-2 runbook's last step read

    categorize_failures.py --in "${b}_scored.jsonl" --out "${b}_scored.jsonl"

Every one of these scripts opens the output with "w" while the input handle is
still unread (`categorize_failures.py:72` and its three siblings), so the file
is truncated to zero before a single line is parsed. Reproduced on a 5-row copy
of a real scored file: 5 rows in, "✅ Wrote 0 rows", exit 0, 0 bytes left.

Exit 0 and a green checkmark, on the last step of a session that has just spent
hours of A100 time, against a file CLAUDE.md says must never be modified. The
runbook line is fixed, but the landmine is in the scripts, and any future
session can step on it.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = ("compute_metrics.py", "annotate_errors.py", "compute_pf.py",
           "categorize_failures.py")


@pytest.mark.parametrize("script", SCRIPTS)
def test_the_script_refuses_to_write_over_its_own_input(script, tmp_path):
    target = tmp_path / "rows.jsonl"
    target.write_text('{"slurp_id": "x", "predicted_intent": "alarm_set"}\n')

    proc = subprocess.run(
        [str(REPO / "venv/bin/python"), str(REPO / "scripts" / script),
         "--in", str(target), "--out", str(target)],
        capture_output=True, text=True, cwd=str(REPO),
    )

    assert proc.returncode != 0, "%s accepted an in-place write" % script
    assert target.read_text(), "%s truncated its own input" % script
    assert "in place" in (proc.stdout + proc.stderr).lower()


@pytest.mark.parametrize("script", SCRIPTS)
def test_a_path_that_only_looks_different_is_still_the_same_file(script, tmp_path):
    """`results/x.jsonl` and `results/./x.jsonl` are one file."""
    target = tmp_path / "rows.jsonl"
    target.write_text('{"slurp_id": "x", "predicted_intent": "alarm_set"}\n')
    disguised = tmp_path / "." / "rows.jsonl"

    proc = subprocess.run(
        [str(REPO / "venv/bin/python"), str(REPO / "scripts" / script),
         "--in", str(target), "--out", str(disguised)],
        capture_output=True, text=True, cwd=str(REPO),
    )

    assert proc.returncode != 0, "%s accepted a disguised in-place write" % script
    assert target.read_text(), "%s truncated its own input" % script
