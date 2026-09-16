"""Unit tests for scripts/tabulate_error_subtypes.py.

The Study 1b descriptive breakdown (§11 step 5, EXECUTION_TODO Step 7):
"Tabulate (error_subtype × slot_position) counts; n < 30 → flag exploratory."

Two facts about the data shape this file:

* `error_subtype` exists only on `substitute` chunks. `delete` and `insert`
  carry `None` by construction, so a table that puts them in a `None` row
  invents a subtype that the annotation never claimed.
* The counts are a property of the **ASR stage**, not of the router. Every
  router re-routes the same transcripts, so pooling two routers' files would
  count each transcription error once per router. The tabulation refuses that.

conftest.py puts scripts/ on sys.path, so the module imports by bare name.
"""
import pytest

import tabulate_error_subtypes as tab


def row(llm="qwen9b", degradation="babble", cats=()):
    return {"llm_model": llm, "degradation": degradation,
            "slurp_id": 1, "error_categories": list(cats)}


def sub(subtype, position):
    return {"type": "substitute", "error_subtype": subtype,
            "slot_position": position, "ref": "a", "hyp": "b"}


def gap(kind, position):
    return {"type": kind, "error_subtype": None, "slot_position": position,
            "ref": "a", "hyp": ""}


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------

def test_counts_substitutions_by_subtype_and_slot_position():
    rows = [row(cats=[sub("number_error", "critical"),
                      sub("number_error", "critical"),
                      sub("semantic_drift", "function")])]
    t = tab.tabulate(rows)
    assert t["substitutions"][("number_error", "critical")] == 2
    assert t["substitutions"][("semantic_drift", "function")] == 1


def test_deletions_and_insertions_are_reported_apart_from_the_subtypes():
    """They have no subtype, so they must not become a `None` subtype row."""
    rows = [row(cats=[gap("delete", "critical"), gap("insert", "function")])]
    t = tab.tabulate(rows)
    assert t["gaps"][("delete", "critical")] == 1
    assert t["gaps"][("insert", "function")] == 1
    assert not any(k[0] is None for k in t["substitutions"])


def test_rows_without_any_error_contribute_nothing():
    t = tab.tabulate([row(cats=[]), row(cats=[])])
    assert t["substitutions"] == {}
    assert t["gaps"] == {}
    assert t["n_rows"] == 2


def test_by_degradation_keeps_the_conditions_apart():
    rows = [row(degradation="babble", cats=[sub("number_error", "critical")]),
            row(degradation="codec", cats=[sub("number_error", "critical")])]
    split = tab.tabulate_by(rows, "degradation")
    assert split["babble"]["substitutions"][("number_error", "critical")] == 1
    assert split["codec"]["substitutions"][("number_error", "critical")] == 1


# --------------------------------------------------------------------------
# The refusal that keeps the counts meaning what they say
# --------------------------------------------------------------------------

def test_refuses_files_that_mix_two_routers():
    """Each router re-routes the SAME transcripts: pooling them would count
    every transcription error once per router and double the table."""
    rows = [row(llm="qwen9b", cats=[sub("number_error", "critical")]),
            row(llm="qwen27b", cats=[sub("number_error", "critical")])]
    with pytest.raises(tab.MixedRouters) as exc:
        tab.tabulate(rows)
    assert "qwen9b" in str(exc.value) and "qwen27b" in str(exc.value)


def test_one_router_is_fine_however_many_files_it_came_from():
    rows = [row(llm="qwen9b", cats=[sub("number_error", "critical")])] * 3
    assert tab.tabulate(rows)["substitutions"][("number_error", "critical")] == 3


# --------------------------------------------------------------------------
# The n < 30 exploratory flag, which the design asks for by name
# --------------------------------------------------------------------------

def test_a_thin_cell_is_flagged_exploratory():
    rows = [row(cats=[sub("phonetic_similar", "critical")] * 29)]
    assert tab.exploratory_cells(tab.tabulate(rows)) == [
        ("phonetic_similar", "critical", 29)]


def test_a_cell_at_thirty_is_not_flagged():
    rows = [row(cats=[sub("phonetic_similar", "critical")] * 30)]
    assert tab.exploratory_cells(tab.tabulate(rows)) == []


def test_the_rendered_table_marks_the_thin_cells_and_names_the_threshold():
    rows = [row(cats=[sub("phonetic_similar", "critical")] * 12
                     + [sub("number_error", "critical")] * 100)]
    text = tab.format_table(tab.tabulate(rows))
    assert "n < 30" in text
    assert "phonetic_similar" in text and "number_error" in text


# --------------------------------------------------------------------------
# Pooled, every cell clears 30; split by condition, some do not. The design
# asks for the flag, so the split has to roll its thin cells up where they can
# be read without scanning seven tables for asterisks.
# --------------------------------------------------------------------------

def test_the_rollup_names_the_level_the_thin_cell_belongs_to():
    rows = ([row(degradation="babble", cats=[sub("phonetic_similar", "critical")] * 8)]
            + [row(degradation="noise", cats=[sub("phonetic_similar", "critical")] * 40)])
    roll = tab.exploratory_rollup(tab.tabulate_by(rows, "degradation"))
    assert roll == [("babble", "phonetic_similar", "critical", 8)]


def test_the_rollup_is_empty_when_every_cell_clears_the_threshold():
    rows = [row(degradation="babble", cats=[sub("number_error", "critical")] * 30)]
    assert tab.exploratory_rollup(tab.tabulate_by(rows, "degradation")) == []


# --------------------------------------------------------------------------
# Two router arms exist as of 2026-08-27, so "the sweep" is no longer one
# router's files and the default has to name the arm it means.
# --------------------------------------------------------------------------

def test_files_for_router_keeps_only_the_named_arm():
    paths = ["results/cascade_whisper_qwen9b_clean_scored.jsonl",
             "results/cascade_whisper_qwen27b_clean_scored.jsonl",
             "results/parakeet_clean_qwen9b_scored.jsonl",
             "results/parakeet_clean_qwen27b_scored.jsonl"]
    assert tab.files_for_router(paths, "qwen9b") == [
        "results/cascade_whisper_qwen9b_clean_scored.jsonl",
        "results/parakeet_clean_qwen9b_scored.jsonl"]


def test_files_for_router_does_not_let_qwen9b_match_qwen27b():
    with pytest.raises(SystemExit):
        tab.files_for_router(
            ["results/cascade_whisper_qwen27b_clean_scored.jsonl"], "qwen9b")


def test_files_for_router_refuses_an_arm_that_is_not_there():
    with pytest.raises(SystemExit):
        tab.files_for_router(["results/cascade_whisper_qwen9b_clean_scored.jsonl"],
                             "gemma31b")
