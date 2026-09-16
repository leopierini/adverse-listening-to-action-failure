"""Unit tests for scripts/generate_status.py.

STATUS.md is the file that says where the project IS. Everything countable in
it is generated from the data, so it cannot go stale; below the marker sits a
short hand-written section for judgement, which no regeneration may touch.

That split is the whole point: on 2026-08-18 CLAUDE.md said "Step 5 is next"
five days after Step 5 finished, because a human had to remember to edit it.
"""
import pytest

import generate_status as gs


def test_the_handwritten_tail_survives_regeneration(tmp_path):
    status = tmp_path / "STATUS.md"
    status.write_text(
        "# Project status\n\n"
        + gs.BEGIN + "\nstale numbers from last week\n" + gs.END + "\n\n"
        "## Judgement\n\nBlocked on the H2a decision with Testolin.\n"
    )

    gs.write_status(status, "fresh numbers\n")

    text = status.read_text()
    assert "Blocked on the H2a decision with Testolin." in text
    assert "fresh numbers" in text
    assert "stale numbers from last week" not in text


def test_a_missing_file_is_created_with_both_halves(tmp_path):
    status = tmp_path / "STATUS.md"

    gs.write_status(status, "the numbers\n")

    text = status.read_text()
    assert gs.BEGIN in text and gs.END in text
    assert "the numbers" in text
    # The hand-written half must exist as a prompt, or nobody will add it.
    assert text.index(gs.END) < text.index("##", text.index(gs.END))


def test_a_file_whose_markers_were_damaged_is_never_overwritten(tmp_path):
    """The hand-written half is the only part a human cannot regenerate.
    Losing it to a broken marker would be unrecoverable, so refuse instead."""
    status = tmp_path / "STATUS.md"
    status.write_text("# Project status\n\nsomeone deleted the markers\n")

    with pytest.raises(gs.StatusFileDamaged):
        gs.write_status(status, "the numbers\n")

    assert "someone deleted the markers" in status.read_text()


def test_counts_come_from_the_sweep_and_pilots_are_reported_separately(tmp_path):
    """A bare glob over results/ mixes the 200-row H2a pilot into the sweep
    counts. STATUS.md is where that error would be most visible and most
    believed."""
    results = tmp_path / "results"
    results.mkdir()
    (results / "cascade_whisper_qwen9b_clean_scored.jsonl").write_text(
        '{"slurp_id": 1, "asr_model": "a", "llm_model": "x", '
        '"degradation": "clean", "snr_db": null}\n')
    (results / "h2a_pilot_27b_scored.jsonl").write_text(
        '{"slurp_id": 1, "asr_model": "a", "llm_model": "big", '
        '"degradation": "clean", "snr_db": null}\n')

    facts = gs.collect_facts(repo=tmp_path)

    assert facts["sweep_rows"] == 1
    assert facts["pilot_rows"] == 1
    assert facts["routers"] == 1
