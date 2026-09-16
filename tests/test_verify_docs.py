"""Unit tests for scripts/verify_docs.py — the cross-document claim checker.

WHY: verify_tracker_claims.py covers THESIS_TRACKER.md only. CLAUDE.md is the
file injected into every session, and on 2026-08-18 it still said "Step 5 is
next" and "no A100 capacity since 2026-08-12" — both false since 2026-08-17.
Every session therefore started from a stale premise and repeated it.

Each test here pins one class of staleness that has actually occurred.
"""
from pathlib import Path

import pytest

import verify_docs as vd

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- dead file references ---------------------------------------------------

def test_a_reference_to_a_missing_file_is_reported(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "alive.py").write_text("")
    text = "Run `scripts/alive.py`, then `scripts/renamed_away.py` for the rest."

    dead = vd.dead_paths(text, tmp_path)

    assert dead == ["scripts/renamed_away.py"]


def test_globs_and_urls_are_not_treated_as_file_references(tmp_path):
    text = ("Read `results/*_scored.jsonl` and see https://example.org/a/b.md "
            "plus `docs/superpowers/specs/`.")
    (tmp_path / "docs" / "superpowers" / "specs").mkdir(parents=True)

    assert vd.dead_paths(text, tmp_path) == []


# --- the test count, quoted in two documents --------------------------------

def test_the_stated_test_count_is_extracted():
    """The 'how to run the suite' line, as it appears in CLAUDE.md and §0a."""
    text = "./venv/bin/python -m pytest tests/ -q     # 141 tests, all green"
    assert vd.stated_test_counts(text) == [141]


def test_several_stated_counts_are_all_returned():
    text = ("./venv/bin/python -m pytest tests/ -q   # 102 tests (counted 2026-08-17)\n"
            "./venv/bin/python -m pytest tests/ -q   # 141 tests, all green")
    assert vd.stated_test_counts(text) == [102, 141]


def test_a_past_run_recorded_as_an_outcome_is_not_a_live_claim():
    """History has to stay writable: a milestone saying '34 passed' on a date
    records what was true then. Flagging those would train everyone to ignore
    the checker, which is worse than not having one."""
    text = ("- **2026-07-19 — code implemented.** Test suite: `tests/`, 34 passing.\n"
            "**Done when:** `./venv/bin/python -m pytest tests/ -q` -> 34 passed.")
    assert vd.stated_test_counts(text) == []


# --- CLAUDE.md's next action vs the checklist -------------------------------

def test_next_action_step_is_read_from_claude_md():
    text = "## Next action\n\n**Step 6 — the full remote sweep — is held.**\n"
    assert vd.next_action_step(text) == 6


def test_first_unchecked_step_is_read_from_the_checklist():
    text = (
        "## STEP 4 — done  · ✅ DONE\n- [x] a\n- [x] b\n"
        "## STEP 5 — pilots\n- [x] boot\n- [x] shut down\n"
        "## STEP 6 — full sweep\n- [ ] serve the 27B\n"
        "## STEP 7 — analysis\n- [ ] concatenate\n"
    )
    assert vd.first_unchecked_step(text) == 6


def test_a_step_whose_boxes_are_all_ticked_is_not_the_next_one():
    text = ("## STEP 5 — pilots\n- [x] one\n"
            "## STEP 6 — sweep\n- [ ] two\n")
    assert vd.first_unchecked_step(text) == 6


# --- claims the hard rules forbid -------------------------------------------

def test_quoting_the_deprecated_gain_column_is_flagged():
    """CLAUDE.md forbids quoting `gain_legacy` and absorption_qwen9b.csv's
    `absorption_gain` anywhere — a rule nothing enforced until now."""
    text = "The mean gain_legacy across cells was -0.359, which shows..."
    assert vd.forbidden_claims(text)
    assert "gain_legacy" in vd.forbidden_claims(text)[0]


def test_naming_the_deprecated_column_to_forbid_it_is_allowed():
    """The rule itself has to be writable, or the checker forbids its own
    documentation."""
    text = ("- **Never quote `absorption_gain` from "
            "`results/analysis/absorption_qwen9b.csv`,** nor the `gain_legacy` "
            "column anywhere.")
    assert vd.forbidden_claims(text) == []


def test_the_falsified_bracket_is_flagged():
    """'the truth is bracketed by -0.106 to +0.093' was falsified 2026-08-14
    by the project's own calibrated null (+0.303). It must not come back."""
    text = "The absorption gain is bracketed between -0.106 and +0.093."
    assert vd.forbidden_claims(text)


def test_the_word_desk_verified_is_flagged():
    text = "The model specs are desk-verified 2026-07-13."
    assert any("desk-verified" in f for f in vd.forbidden_claims(text))


# --- model ids that look right and are wrong --------------------------------

def test_a_trap_model_id_in_a_document_is_flagged():
    """The failure that cost this project the most: `google/gemma-4-31b` is
    the BASE checkpoint. It sat in the tracker for a month labelled verified."""
    text = "Serve `google/gemma-4-31b` on the A100 and route the transcripts."

    hits = vd.trap_model_ids(text)

    assert hits and "google/gemma-4-31b" in hits[0]
    assert "BASE checkpoint" in hits[0]


def test_the_correct_suffixed_id_is_not_flagged_as_its_own_trap():
    """`google/gemma-4-31B-it-qat-w4a16-ct` contains `google/gemma-4-31B-it`
    as a prefix. A substring match would condemn the right answer."""
    text = "Serve `google/gemma-4-31B-it-qat-w4a16-ct` with --enforce-eager."
    assert vd.trap_model_ids(text) == []


def test_naming_a_trap_in_order_to_warn_about_it_is_allowed():
    text = ("⚠️ `google/gemma-4-31b` and `google/gemma-4-12b` are **base** "
            "checkpoints — no chat template, useless as routers.")
    assert vd.trap_model_ids(text) == []


# --- the retention pairing that was wrong in three documents ----------------

def test_the_retention_floor_paired_with_the_wrong_degradation_is_flagged():
    """Until 2026-08-14 three documents read '18.3% at whisper-base reverb
    0 dB'. 18.3% is babble; reverb is 12.0% and six points worse."""
    text = "Retention bottoms out at 18.3% (Whisper Base, reverb 0 dB)."
    assert vd.retention_pairing_errors(text)


def test_the_correct_retention_pairings_pass():
    text = ("Lowest: **12.0%** (Whisper Base, reverb 0 dB), then babble 0 dB "
            "**18.3%** and noise 0 dB **18.6%**.")
    assert vd.retention_pairing_errors(text) == []


# --- false positives the first real run produced -----------------------------

def test_a_huggingface_repo_id_is_not_a_file_path(tmp_path):
    """`Qwen/Qwen3.5-27B-GPTQ-Int4` is a model id, not something on disk.
    Reporting it as a dead path buries the real findings."""
    (tmp_path / "scripts").mkdir()
    text = ("Serve `Qwen/Qwen3.5-27B-GPTQ-Int4` and "
            "`cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit`.")
    assert vd.dead_paths(text, tmp_path) == []


def test_dois_cidr_blocks_and_template_placeholders_are_not_paths(tmp_path):
    (tmp_path / "corrupted").mkdir()
    text = ("DOI `10.4324/9781315772110-15`, firewall `0.0.0.0/0`, "
            "the bank at `corrupted/{degradation}/snr{snr}/`, "
            "an elided `02_Education/…`, and `vllm/envs.py:49` in someone "
            "else's tree.")
    assert vd.dead_paths(text, tmp_path) == []


def test_a_path_under_a_real_repo_directory_is_still_checked(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "alive.py").write_text("")
    text = "`scripts/alive.py` and `scripts/gone.py`"
    assert vd.dead_paths(text, tmp_path) == ["scripts/gone.py"]


def test_defining_gain_legacy_as_audit_only_is_not_a_forbidden_claim():
    """The metric definitions have to be writable. What is forbidden is
    REPORTING the column as a result, not naming it in its own definition."""
    text = ("- **`gain_legacy` — audit trail only.** `absorption_rate − "
            "(1 − mean(min(WER, 1)))` over **all** rows: the original, "
            "defective formula. **Do not report as a result.**")
    assert vd.forbidden_claims(text) == []


def test_a_retention_value_far_from_a_degradation_name_is_not_paired_with_it():
    """A sentence that lists every degradation is not a claim that 99.0%
    belongs to one of them. Pairing needs proximity, not co-occurrence anywhere
    on the line. Fixture taken verbatim from a real document that produced this
    false positive on the checker's first run."""
    text = ("The number to lead with is **retention**: of the actions that "
            "worked on clean audio, how many survive the noise — **99.0%** at "
            "its best, **12.0%** at its worst. What is safe to claim is the "
            "**ordering**: absorption survives codec, far-field and clipping, "
            "and is destroyed by reverb, noise and babble.")
    assert vd.retention_pairing_errors(text) == []


def test_a_hyphenated_degradation_next_to_its_value_still_matches():
    """'far-field' must read as 'farfield', or a correct pairing looks like a
    missing one. Fixture verbatim from a real document."""
    text = ("> - **Retention**, which needs no null model: **99.0%** "
            "(Whisper-Turbo, far-field 20 dB) down to **12.0%** "
            "(Whisper Base, reverb 0 dB).")
    assert vd.retention_pairing_errors(text) == []


def test_a_retention_value_directly_beside_the_wrong_degradation_is_flagged():
    text = "the floor is **18.3%** at reverb 0 dB"
    assert vd.retention_pairing_errors(text)


def test_a_forbidden_label_inside_quotation_marks_is_being_quoted_not_asserted():
    """CLAUDE.md line 19 and THESIS_TRACKER.md lines 675/693 verbatim. Quoting
    a label in order to say it was false is the opposite of claiming it, and
    the decision log cannot record the failure without writing the words."""
    for text in (
        'The specs were labelled "desk-verified 2026-07-13"; nothing had been.',
        'This box previously carried the label "desk-verified 2026-07-13".',
        'The serving path was labelled "desk-verified" for a month.',
    ):
        assert vd.forbidden_claims(text) == [], text


def test_asserting_desk_verification_as_a_decision_is_still_flagged():
    text = ("| **Omni serving path?** | **Desk-verified 2026-07-13**: "
            "Qwen3-Omni via community AWQ 4-bit | Both arms feasible |")
    assert any("desk-verified" in f for f in vd.forbidden_claims(text))


def test_naming_the_column_is_allowed_but_quoting_a_value_from_it_is_not():
    """The rule forbids REPORTING the deprecated comparator, not referring to
    it. A name in a function signature carries no result; a name beside a
    number does."""
    naming = ("Create `scripts/absorption_metrics.py` — pure functions: "
              "`cell_stats` (absorption, ceiling, `gain_legacy`, `gain_ceiling`).")
    quoting = "Mean `gain_legacy` across the cells was -0.359."

    assert vd.forbidden_claims(naming) == []
    assert vd.forbidden_claims(quoting)


def test_writing_the_audit_csv_is_not_quoting_it():
    """EXECUTION_TODO.md line 149 verbatim: a recorded command whose --out is
    the audit CSV. Producing the file is not reporting a number from it."""
    text = ("- [x] First look: `./venv/bin/python scripts/analyze_absorption.py "
            "results/*_scored.jsonl --out results/analysis/absorption_qwen9b.csv`")
    assert vd.forbidden_claims(text) == []


def test_a_correction_notice_quoting_the_old_wrong_pairing_is_allowed():
    """CLAUDE.md verbatim. The record of the 2026-08-14 fix has to quote the
    error it fixed, or the fix is unreadable."""
    text = ('⚠️ The pairing "18.3% at reverb 0 dB" was wrong in three documents '
            'until 2026-08-14: 18.3% is Whisper Base at **babble** 0 dB.')
    assert vd.retention_pairing_errors(text) == []


def test_a_step_marked_done_is_skipped_even_if_it_holds_a_deferred_item():
    """A completed step can still carry something waiting on someone else.
    Reading that as 'the next piece of work' is how the checklist and CLAUDE.md
    drift apart — which is what happened on 2026-08-18."""
    text = (
        "## STEP 4b — Fix the absorption metric  · Mac · ✅ DONE 2026-08-12\n"
        "- [x] rewrite it\n"
        "- [ ] restate the finding to the advisor in September\n"
        "## STEP 6 — full remote sweep  · GCP\n"
        "- [ ] serve the 27B\n"
    )
    assert vd.first_unchecked_step(text) == 6


def test_a_step_not_marked_done_with_an_open_box_is_still_the_next_one():
    text = ("## STEP 5 — pilots  · ✅ DONE 2026-08-17\n- [x] boot\n"
            "## STEP 6 — sweep  · GCP · a few sessions\n- [ ] serve\n")
    assert vd.first_unchecked_step(text) == 6


def test_the_test_count_is_none_when_the_interpreter_is_missing(tmp_path):
    """A checker that raises where it cannot check is a checker that takes the
    whole session down with it. Report 'unknown', never crash."""
    assert vd.actual_test_count(tmp_path) is None


# --- the section addressing scheme ------------------------------------------

def test_the_map_is_read_from_the_tracker_table(tmp_path):
    (tmp_path / "THESIS_TRACKER.md").write_text(
        "| § | what it is | file |\n|---|---|---|\n"
        "| §8 | metrics | `docs/design/METRICS.md` |\n"
        "| ~~§0a~~ | *dissolved* | — |\n")

    assert vd.section_map(tmp_path) == {"8": "docs/design/METRICS.md"}


def test_a_section_the_map_sends_to_a_file_that_lacks_it_is_flagged(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "METRICS.md").write_text("# Metrics\n\n## 9. Power\n")
    (tmp_path / "THESIS_TRACKER.md").write_text(
        "| § | what | file |\n|---|---|---|\n| §8 | metrics | `docs/METRICS.md` |\n")

    problems = vd.section_map_problems(tmp_path)

    assert problems and "§8" in problems[0]


def test_a_section_owned_by_two_mapped_documents_is_flagged(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "A.md").write_text("## 8. Metrics\n")
    (tmp_path / "docs" / "B.md").write_text("## 8. Metrics again\n")
    (tmp_path / "THESIS_TRACKER.md").write_text(
        "| § | what | file |\n|---|---|---|\n"
        "| §8 | metrics | `docs/A.md` |\n| §9 | power | `docs/B.md` |\n")

    problems = vd.section_map_problems(tmp_path)

    assert any("more than one" in p for p in problems)


def test_another_documents_own_numbering_is_not_a_thesis_section(tmp_path):
    """A session record and a runbook have their own `## 1.`, `## 2.` headings.
    Reading those as claims to own §1 and §2 produced 11 false alarms on the
    first run and buried the one real finding."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "A.md").write_text("## 8. Metrics\n")
    (tmp_path / "docs" / "RUNBOOK.md").write_text("## 1. Boot the VM\n## 8. Shut down\n")
    (tmp_path / "THESIS_TRACKER.md").write_text(
        "| § | what | file |\n|---|---|---|\n| §8 | metrics | `docs/A.md` |\n")

    assert vd.section_map_problems(tmp_path) == []


def test_a_dissolved_section_still_resolves_to_an_explanation(tmp_path):
    """§0a is cited in older documents and commits. The map must keep saying
    what became of it, or every one of those citations dangles forever."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "A.md").write_text("## 8. Metrics\n")
    (tmp_path / "THESIS_TRACKER.md").write_text(
        "| § | what | file |\n|---|---|---|\n| §8 | metrics | `docs/A.md` |\n"
        "| ~~§0a~~ | *dissolved* — state is now STATUS.md | — |\n")
    (tmp_path / "old.md").write_text("As §0a explained, the blocker was capacity.")

    assert vd.section_map_problems(tmp_path) == []


def test_a_cited_section_that_the_map_does_not_know_is_flagged(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "A.md").write_text("## 8. Metrics\n")
    (tmp_path / "THESIS_TRACKER.md").write_text(
        "| § | what | file |\n|---|---|---|\n| §8 | metrics | `docs/A.md` |\n")
    (tmp_path / "notes.md").write_text("As explained in §18, the design holds.")

    problems = vd.section_map_problems(tmp_path)

    assert any("§18" in p for p in problems)


# --- STATUS.md must have been regenerated after the data changed -------------

def test_status_numbers_are_read_back_out_of_the_generated_table():
    text = ("| **Cells complete** | **57 of 209** (27%) |\n"
            "| Sweep rows collected | 23,712 in 38 files |\n"
            "| Test suite | 181 tests |\n"
            "| **Next step in `EXECUTION_TODO.md`** | **Step 6** |\n")

    assert vd.status_numbers(text) == {
        "cells": 57, "sweep_rows": 23712, "tests": 181, "next_step": 6}


def test_a_status_file_that_disagrees_with_the_data_is_reported():
    stated = {"cells": 57, "sweep_rows": 23712, "tests": 181, "next_step": 6}
    actual = {"cells": 59, "sweep_rows": 23912, "tests": 181, "next_step": 6}

    drift = vd.status_drift(stated, actual)

    assert len(drift) == 2
    assert any("cells" in d and "57" in d and "59" in d for d in drift)


def test_a_status_file_in_step_with_the_data_reports_nothing():
    facts = {"cells": 57, "sweep_rows": 23712, "tests": 181, "next_step": 6}
    assert vd.status_drift(facts, dict(facts)) == []


def test_a_file_paired_with_a_section_it_does_not_own_is_flagged(tmp_path):
    """`THESIS_TRACKER.md §8` reads as authoritative and sends the reader to a
    file that has not held §8 since the 2026-08-18 split. The plain §N check
    misses it, because §8 does resolve — just not there."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "METRICS.md").write_text("## 8. Metrics\n")
    (tmp_path / "THESIS_TRACKER.md").write_text(
        "| § | what | file |\n|---|---|---|\n| §8 | metrics | `docs/METRICS.md` |\n")
    (tmp_path / "OTHER.md").write_text(
        "The formulas are in `THESIS_TRACKER.md` §8, do not deviate.")

    problems = vd.stale_section_pointers(tmp_path)

    assert len(problems) == 1
    assert "THESIS_TRACKER.md" in problems[0] and "docs/METRICS.md" in problems[0]


def test_a_file_paired_with_a_section_it_does_own_is_fine(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "METRICS.md").write_text("## 8. Metrics\n")
    (tmp_path / "THESIS_TRACKER.md").write_text(
        "| § | what | file |\n|---|---|---|\n| §8 | metrics | `docs/METRICS.md` |\n")
    (tmp_path / "OTHER.md").write_text("See `docs/METRICS.md` §8 for the formulas.")

    assert vd.stale_section_pointers(tmp_path) == []


def test_a_nickname_for_the_map_is_not_a_stale_pointer(tmp_path):
    """'tracker §13' means 'look §13 up in the map', which is correct and is
    how most of the repo is written. Flagging it would be pure noise."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "FINDINGS.md").write_text("## 13. Findings\n")
    (tmp_path / "THESIS_TRACKER.md").write_text(
        "| § | what | file |\n|---|---|---|\n| §13 | findings | `docs/FINDINGS.md` |\n")
    (tmp_path / "OTHER.md").write_text("Current findings: tracker §13.")

    assert vd.stale_section_pointers(tmp_path) == []


def test_the_checker_runs_as_a_command(tmp_path):
    """A smoke test with teeth: functions appended after the `if __name__`
    block are invisible to import but crash the CLI with NameError. That
    happened twice while this file was being written, and only running it
    caught it."""
    import subprocess
    import sys as _sys

    proc = subprocess.run(
        [_sys.executable, "scripts/verify_docs.py"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )

    assert "Traceback" not in proc.stderr, proc.stderr[-800:]
    assert proc.returncode in (0, 1)
    assert "section addressing" in proc.stdout


def test_an_archived_spec_describing_the_old_layout_is_not_a_stale_pointer(tmp_path):
    """`docs/superpowers/specs/` and `plans/` are dated snapshots. A spec whose
    job is to record that §1–§10 USED TO live in THESIS_TRACKER.md is correct
    precisely because it names the old pairing; flagging it would make the
    project unable to document its own history."""
    (tmp_path / "docs" / "superpowers" / "specs").mkdir(parents=True)
    (tmp_path / "docs" / "design").mkdir(parents=True)
    (tmp_path / "docs" / "design" / "DESIGN.md").write_text("## 1. The thesis\n")
    (tmp_path / "THESIS_TRACKER.md").write_text(
        "| § | what | file |\n|---|---|---|\n| §1 | overview | `docs/design/DESIGN.md` |\n")
    (tmp_path / "docs" / "superpowers" / "specs" / "old.md").write_text(
        "| design | almost never | `THESIS_TRACKER.md` §1-§10 |")

    assert vd.stale_section_pointers(tmp_path) == []


def test_two_adjacent_but_separate_references_are_not_a_pairing(tmp_path):
    """"...the log is `literature/search-log.md`. See §17 for the anchors."
    is two references that happen to sit near each other, not a claim that the
    log holds §17. Pairing means adjacency, not proximity."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "READING.md").write_text("## 17. Reading list\n")
    (tmp_path / "THESIS_TRACKER.md").write_text(
        "| § | what | file |\n|---|---|---|\n| §17 | reading | `docs/READING.md` |\n")
    (tmp_path / "log.md").write_text(
        "Search log with every query: `literature/search-log.md`. "
        "See §17 for the anchors.")

    assert vd.stale_section_pointers(tmp_path) == []


# --- sync duplicates, which silently inflate every count ---------------------
#
# WHY: on 2026-08-19 the session-start check reported "STATUS.md says 201 tests,
# the files say 222" and told the operator to regenerate STATUS.md — which would
# have written the WRONG number. STATUS.md was right. Three byte-identical
# copies named `... 2.ext` (the macOS/cloud-sync duplicate form) had appeared,
# and `tests/test_model_frames 2.py` carried 21 tests that pytest collected a
# second time: 201 + 21 = 222. The counters trust the filesystem, so a stray
# copy is not a cosmetic mess — it is a false count with a plausible cause
# attached to the wrong file.

def test_a_sync_duplicate_of_a_test_file_is_reported(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_frames.py").write_text("def test_one():\n    pass\n")
    (tmp_path / "tests" / "test_frames 2.py").write_text("def test_one():\n    pass\n")

    found = vd.sync_duplicate_files(tmp_path)

    assert len(found) == 1
    assert "tests/test_frames 2.py" in found[0]
    assert "tests/test_frames.py" in found[0]


def test_a_byte_identical_duplicate_is_named_as_safe_to_delete(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "fit.json").write_text('{"a": 1}')
    (tmp_path / "results" / "fit 2.json").write_text('{"a": 1}')

    found = vd.sync_duplicate_files(tmp_path)

    assert len(found) == 1
    assert "identical" in found[0]


def test_a_duplicate_that_diverged_from_its_twin_is_flagged_differently(tmp_path):
    """A differing copy may hold an edit that never landed. Never say 'delete'."""
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "fit.json").write_text('{"a": 1}')
    (tmp_path / "results" / "fit 2.json").write_text('{"a": 2}')

    found = vd.sync_duplicate_files(tmp_path)

    assert len(found) == 1
    assert "DIVERGED" in found[0]
    assert "identical" not in found[0]


def test_a_numbered_name_with_no_twin_is_not_a_duplicate(tmp_path):
    """`gcp-session 2.md` is a legitimate name when no `gcp-session.md` exists."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "gcp-session 2.md").write_text("session two")

    assert vd.sync_duplicate_files(tmp_path) == []


def test_the_virtualenvs_and_git_are_not_scanned(tmp_path):
    """venv holds thousands of vendored files; scanning it is noise, not signal."""
    (tmp_path / "venv" / "lib").mkdir(parents=True)
    (tmp_path / "venv" / "lib" / "mod.py").write_text("x = 1")
    (tmp_path / "venv" / "lib" / "mod 2.py").write_text("x = 1")

    assert vd.sync_duplicate_files(tmp_path) == []


def test_the_real_repo_has_no_sync_duplicates(tmp_path):
    """The live check: a duplicate anywhere here means the counts are lying."""
    assert vd.sync_duplicate_files(REPO_ROOT) == []


def test_the_checker_reports_a_sync_duplicate_when_it_runs(tmp_path, monkeypatch):
    """Finding duplicates is useless if main() never asks. The count checks run
    downstream of this one, so the operator must read the cause before the
    symptom."""
    import io
    import contextlib

    monkeypatch.setattr(vd, "sync_duplicate_files",
                        lambda repo=None: ["`tests/x 2.py` is a sync copy of "
                                           "`tests/x.py`, byte-identical"])
    monkeypatch.setattr(vd, "_failures", [])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = vd.main([])

    assert code == 1
    assert "tests/x 2.py" in out.getvalue()


# --- two live documents disagreeing about the same step ---------------------
#
# WHY: on 2026-08-19 `STATUS.md` said "Step 6 is unblocked as of 2026-08-18"
# while `EXECUTION_TODO.md` — the file whose entire job is "what do I do next"
# — still said "Step 6 must not start until this is discussed with Testolin".
# Both were hand-written prose, so every existing check passed. An agent
# reading the checklist stops; an agent reading STATUS.md proceeds.
#
# Scope is the LIVE documents only. `docs/log/` is append-only history and must
# keep saying what was true when it was written.

def test_one_document_blocking_a_step_another_declares_unblocked_is_reported():
    texts = {
        "STATUS.md": "✅ Step 6 is unblocked as of 2026-08-18.",
        "EXECUTION_TODO.md": "🛑 Step 6 must not start until Testolin agrees.",
    }

    conflicts = vd.step_blocking_conflicts(texts)

    assert len(conflicts) == 1
    assert "Step 6" in conflicts[0]
    assert "STATUS.md" in conflicts[0]
    assert "EXECUTION_TODO.md" in conflicts[0]


def test_documents_agreeing_that_a_step_is_blocked_are_not_a_conflict():
    texts = {
        "STATUS.md": "Step 7 must not start: the data is not in.",
        "EXECUTION_TODO.md": "Step 7 is blocked until Step 6 lands.",
    }

    assert vd.step_blocking_conflicts(texts) == []


def test_a_block_on_one_step_and_a_clearance_on_another_is_not_a_conflict():
    texts = {
        "STATUS.md": "Step 6 is unblocked. Step 8 must not start yet.",
    }

    assert vd.step_blocking_conflicts(texts) == []


def test_the_append_only_logs_are_not_read_for_blocking_claims():
    """A milestone recording that Step 6 was blocked in August stays true."""
    texts = {
        "STATUS.md": "Step 6 is unblocked as of 2026-08-18.",
        "docs/log/MILESTONES.md": "2026-08-17: Step 6 must not start yet.",
    }

    assert vd.step_blocking_conflicts(texts) == []


def test_the_live_documents_do_not_contradict_each_other_today():
    """The live check: this is the contradiction that shipped on 2026-08-19."""
    texts = {name: (REPO_ROOT / name).read_text()
             for name in vd.DOCUMENTS if (REPO_ROOT / name).exists()}

    assert vd.step_blocking_conflicts(texts) == []


# --- a document nobody classified is a document nobody checks ----------------
#
# WHY: a cold-start audit on 2026-08-19 found `docs/gcp-session-1-runbook.md`
# telling the operator to STOP unless pytest reports 102 green — 111 tests
# stale, and the exact class of rot `stated_test_counts` was built to catch.
# It passed because DOCUMENTS is a hand-maintained tuple and the runbook was
# never added to it. The same gap hid a runbook that no file referenced.
#
# Coverage is therefore no longer a memory exercise: a new .md under the
# project's own directories must be classified as checked or excluded, the way
# result_sets.py refuses an unclassified *_scored.jsonl.

def test_a_new_project_document_in_neither_list_is_reported(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "NOTES.md").write_text("# notes")

    uncovered = vd.uncovered_documents(tmp_path, documents=(), excluded=())

    assert uncovered == ["docs/NOTES.md"]


def test_a_checked_document_is_not_reported(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "NOTES.md").write_text("# notes")

    assert vd.uncovered_documents(
        tmp_path, documents=("docs/NOTES.md",), excluded=()) == []


def test_an_explicitly_excluded_document_is_not_reported(tmp_path):
    """Excluded is a decision on the record, not an oversight."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "NOTES.md").write_text("# notes")

    assert vd.uncovered_documents(
        tmp_path, documents=(), excluded=("docs/NOTES.md",)) == []


def test_an_excluded_directory_covers_the_files_under_it(tmp_path):
    (tmp_path / "docs" / "superpowers" / "specs").mkdir(parents=True)
    (tmp_path / "docs" / "superpowers" / "specs" / "old.md").write_text("# spec")

    assert vd.uncovered_documents(
        tmp_path, documents=(), excluded=("docs/superpowers/",)) == []


def test_vendored_and_tooling_trees_are_never_scanned(tmp_path):
    """Model cards and venv README files are not this project's documents."""
    for tree in ("venv", "models", ".git", ".pytest_cache"):
        (tmp_path / tree).mkdir()
        (tmp_path / tree / "README.md").write_text("# vendored")

    assert vd.uncovered_documents(tmp_path, documents=(), excluded=()) == []


def test_every_document_in_the_real_repo_is_classified():
    """The live check: adding a .md must force a decision about checking it."""
    assert vd.uncovered_documents(REPO_ROOT) == []


def test_a_struck_through_instruction_is_not_a_live_block():
    """`THESIS_TRACKER.md:38` — a superseded claim stays, with its correction
    beside it. Strikethrough IS the project's mark for "this no longer holds",
    so reading it as a live instruction would punish the convention the repo
    documents and force deletion of the audit trail instead."""
    texts = {
        "STATUS.md": "Step 6 is unblocked as of 2026-08-18.",
        "docs/gcp-session-1-runbook.md":
            "~~Do not start Step 6's sweep before talking to Testolin.~~ "
            "→ DECIDED 2026-08-18: it runs.",
    }

    assert vd.step_blocking_conflicts(texts) == []


def test_a_block_outside_the_strikethrough_still_counts():
    """Striking one clause must not silence the rest of the sentence."""
    texts = {
        "STATUS.md": "Step 6 is unblocked as of 2026-08-18.",
        "EXECUTION_TODO.md": "~~old note~~ Step 6 must not start yet.",
    }

    assert len(vd.step_blocking_conflicts(texts)) == 1


# --- a model id nobody classified ------------------------------------------
#
# WHY: this is the class RULE ZERO was written for, and it came back. On
# 2026-08-19 a cold-start audit found `docs/log/DECISIONS.md` presenting
# `google/gemma-4-31b-it` as the CORRECTED id, while this project's own
# `verify_model_specs.py` lists that same id under "Ids that look right and are
# WRONG" (BF16 62.5 GB, does not fit the A100). The row advertising itself as
# the correction was the stale one, sitting in the decision log.
#
# `trap_model_ids` only catches ids already known to be wrong. This catches the
# other direction: an id nobody has classified at all.

def test_a_model_id_in_no_list_is_reported():
    texts = {"docs/design/STACK.md": "We serve `Qwen/Qwen4.5-99B-Imaginary`."}

    unclassified = vd.unclassified_model_ids(texts)

    assert len(unclassified) == 1
    assert "Qwen/Qwen4.5-99B-Imaginary" in unclassified[0]
    assert "docs/design/STACK.md" in unclassified[0]


def test_the_four_served_models_are_classified():
    """Straight from verify_model_specs.MODELS — the design's own list."""
    texts = {"d.md": "`Qwen/Qwen3.5-27B-GPTQ-Int4` and "
                     "`google/gemma-4-31B-it-qat-w4a16-ct`"}

    assert vd.unclassified_model_ids(texts) == []


def test_a_documented_trap_is_classified_not_unclassified():
    """A trap id is SUPPOSED to appear — that is how the trap gets documented.
    `trap_model_ids` judges whether it is presented as correct."""
    texts = {"d.md": "Not `google/gemma-4-31b`, which is the BASE checkpoint."}

    assert vd.unclassified_model_ids(texts) == []


def test_a_struck_through_model_id_is_not_reported():
    """A superseded id kept for the audit trail, per THESIS_TRACKER.md:38."""
    texts = {"d.md": "~~`Qwen/Qwen4.5-99B-Imaginary`~~ superseded 2026-08-19."}

    assert vd.unclassified_model_ids(texts) == []


def test_every_model_id_in_the_real_documents_is_classified():
    """The live check: naming a model must be a decision, not a keystroke."""
    from pathlib import Path as _P
    texts = {n: (REPO_ROOT / n).read_text()
             for n in vd.DOCUMENTS if (REPO_ROOT / n).exists()}

    assert vd.unclassified_model_ids(texts) == []
