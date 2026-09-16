"""Unit tests for scripts/prompt_ablation.py.

Written 2026-08-28, after the script ran to completion in 6 seconds having
read nothing. Its three input paths were the Study-1a filenames
(`results/clean_baseline_full.jsonl`, `results/1a_noise_snr10.jsonl`,
`results/1a_reverb_snr10.jsonl`), which stopped existing when the cascade
files were renamed. It printed "Skipping" three times, then an empty summary
table, then a paragraph beginning "Interpretation: if the three variants give
TSA within 2-3 pts of each other..." — and exited 0. Appearance of success,
nothing done. That is the same failure mode the runbook's `&&`-chained scoring
loop exists to prevent.
"""
import pytest

import prompt_ablation as pa


def test_every_declared_input_exists_on_disk():
    """The conditions are hardcoded; if a rename breaks them the script must
    not be the last thing to find out."""
    from pathlib import Path
    missing = [(name, path) for name, path in pa.CONDITIONS
               if not Path(path).exists()]
    assert missing == [], "declared inputs that do not exist: %s" % missing


def test_the_run_refuses_instead_of_summarising_nothing():
    """An ablation over zero conditions is not a result."""
    with pytest.raises(SystemExit):
        pa.check_inputs([("nope", "results/does_not_exist.jsonl")])


def test_check_inputs_passes_when_every_file_is_there(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text("{}\n")
    assert pa.check_inputs([("x", str(f))]) == [("x", str(f))]
