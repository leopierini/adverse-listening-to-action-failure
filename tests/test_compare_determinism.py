"""The determinism comparison, pinned against the figures Methods will quote.

Why this file exists
--------------------
The Gemma-12B determinism figures (2/950 raw-intent, 27/950 intent+params,
11/950 scored, ees +0.32 pp) are quoted in `STATUS.md`, `EXECUTION_TODO.md`,
§13 and `docs/gcp-session-3/ESITO.md`, and they are an input to a decision
Leonardo and the advisor still have to take.  Until 2026-08-30 they came from a
snippet that no longer existed, over data that could not reproduce them: the
archived passes were **unscored**, so `ees`, `pf`, `wer` and `cer` were null on
all 950 rows and the scored figure recomputed as 1/950, not 11/950.

Nothing was wrong with the numbers — the scoring chain had simply been run on
copies that were not kept.  Both archives now exist, and this file asserts that
the published figures come back out of them.
"""
import gzip
import json
import os
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION3 = os.path.join(REPO, "docs", "gcp-session-3")
RAW_A = os.path.join(SESSION3, "det_gemma12b_a.jsonl.gz")
RAW_B = os.path.join(SESSION3, "det_gemma12b_b.jsonl.gz")
SCORED_A = os.path.join(SESSION3, "det_gemma12b_a_scored.jsonl.gz")
SCORED_B = os.path.join(SESSION3, "det_gemma12b_b_scored.jsonl.gz")

import sys
sys.path.insert(0, os.path.join(REPO, "scripts"))
import compare_determinism as cd  # noqa: E402


def _rows(a, b):
    return cd.load(a), cd.load(b)


def test_the_published_gemma12b_figures_come_back_out_of_the_archive():
    a, b = _rows(SCORED_A, SCORED_B)
    result = cd.compare(a, b)
    assert result["compared"] == 950
    assert result["only_in_a"] == 0 and result["only_in_b"] == 0
    assert len(result["intent_only"]) == 2, "§13 says 2/950 on intent alone"
    assert len(result["intent_or_params"]) == 27, "§13 says 27/950 with parameters"
    assert len(result["scored_moved"]) == 11, "§13 says 11/950 move a scored metric"
    assert round(100.0 * len(result["scored_moved"]) / 950.0, 2) == 1.16


def test_the_published_ees_drift_comes_back_out_of_the_archive():
    a, b = _rows(SCORED_A, SCORED_B)
    drift = cd.outcome_drift(a, b, "ees")
    assert round(drift["mean_a"], 4) == 0.4242
    assert round(drift["mean_b"], 4) == 0.4274
    assert round(100.0 * (drift["mean_b"] - drift["mean_a"]), 2) == 0.32
    assert len(drift["cells"]) == 19
    assert round(100.0 * drift["mean_abs_cell_delta"], 2) == 0.32
    worst = drift["cells"][0]
    assert round(100.0 * worst[3], 2) == 2.00
    # ESITO.md named `reverb @ 20 dB` as the worst cell. Three cells tie at
    # 2.00 pp, so "the maximum" is not one cell — pinned so the prose says so.
    tied = [c for c in drift["cells"] if round(100.0 * c[3], 2) == 2.00]
    assert len(tied) == 3
    assert ("reverb", 20) in [c[0] for c in tied]


def test_the_unscored_archive_cannot_answer_the_tripwire_question():
    """The trap this file exists to mark.

    Running the comparison on the RAW passes looks like it worked — it prints
    a rate, and the rate is small.  It is 1/950 rather than 11/950 because the
    only scored field those rows carry is `tsa`.  A reader who reached for the
    raw archive would under-report the drift by a factor of eleven.
    """
    a, b = _rows(RAW_A, RAW_B)
    result = cd.compare(a, b)
    assert len(result["intent_only"]) == 2, "raw drift IS readable without scoring"
    assert len(result["intent_or_params"]) == 27
    assert len(result["scored_moved"]) == 1, (
        "the raw archive answers the scored question WRONG, and must not be "
        "used for it")
    with gzip.open(RAW_A, "rt") as handle:
        first = json.loads(handle.readline())
    assert first["ees"] is None
    assert cd.outcome_drift(a, b, "ees")["mean_a"] is None


def _write(tmp_path, name, rows):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _row(**kw):
    base = {"slurp_id": 1, "asr_model": None, "degradation": "clean",
            "snr_db": None, "predicted_intent_raw": "alarm_set",
            "predicted_parameters": {"time": "8am"},
            "tsa": 1, "pf": 1.0, "ees": 1.0, "ees_strict": 1.0}
    base.update(kw)
    return base


def test_a_difference_in_parameters_alone_is_invisible_at_level_one(tmp_path):
    a = _write(tmp_path, "a.jsonl", [_row()])
    b = _write(tmp_path, "b.jsonl",
               [_row(predicted_parameters={"time": "8 am"})])
    result = cd.compare(cd.load(a), cd.load(b))
    assert len(result["intent_only"]) == 0
    assert len(result["intent_or_params"]) == 1
    assert len(result["scored_moved"]) == 0


def test_parameter_dicts_compare_by_content_not_by_key_order(tmp_path):
    a = _write(tmp_path, "a.jsonl",
               [_row(predicted_parameters={"time": "8am", "day": "mon"})])
    b = _write(tmp_path, "b.jsonl",
               [_row(predicted_parameters={"day": "mon", "time": "8am"})])
    result = cd.compare(cd.load(a), cd.load(b))
    assert len(result["intent_or_params"]) == 0, (
        "key order is not drift, and counting it as drift would inflate every "
        "probe this project runs")


@pytest.mark.parametrize("field", ["tsa", "pf", "ees", "ees_strict"])
def test_each_scored_field_can_fire_the_tripwire_on_its_own(tmp_path, field):
    a = _write(tmp_path, "a.jsonl", [_row()])
    b = _write(tmp_path, "b.jsonl", [_row(**{field: 0.0})])
    result = cd.compare(cd.load(a), cd.load(b))
    assert len(result["scored_moved"]) == 1, (
        "%s moved and the tripwire stayed silent" % field)


def test_rows_present_in_only_one_pass_are_reported_not_dropped(tmp_path):
    a = _write(tmp_path, "a.jsonl", [_row(slurp_id=1), _row(slurp_id=2)])
    b = _write(tmp_path, "b.jsonl", [_row(slurp_id=1)])
    result = cd.compare(cd.load(a), cd.load(b))
    assert result["compared"] == 1
    assert result["only_in_a"] == 1


def test_the_cli_runs_and_names_the_tripwire():
    out = subprocess.check_output(
        [os.path.join(REPO, "venv", "bin", "python"),
         os.path.join(REPO, "scripts", "compare_determinism.py"),
         "--a", SCORED_A, "--b", SCORED_B]).decode()
    assert "11 / 950 (1.16%)" in out
    assert "TRIPWIRE: FIRED" in out
    assert "+0.32 pp" in out


# --- The Qwen3-Omni probe on vLLM 0.28.0, 2026-08-30 -------------------
SESSION4 = os.path.join(REPO, "docs", "gcp-session-4")
OMNI_A = os.path.join(SESSION4, "det_omni_v028_a_scored.jsonl.gz")
OMNI_B = os.path.join(SESSION4, "det_omni_v028_b_scored.jsonl.gz")
V028_SWEEP = os.path.join(REPO, "results", "v028",
                          "omni_qwen_sweep_scored.jsonl")
V0271_SWEEP = os.path.join(REPO, "results", "omni_qwen_sweep_scored.jsonl")


def test_the_omni_v028_determinism_figures_come_back_out_of_the_archive():
    a, b = _rows(OMNI_A, OMNI_B)
    result = cd.compare(a, b)
    assert result["compared"] == 950
    assert len(result["intent_only"]) == 6
    assert len(result["intent_or_params"]) == 53, "ESITO says 53/950 (5.58%)"
    assert len(result["scored_moved"]) == 24, "ESITO says 24/950 (2.53%)"
    drift = cd.outcome_drift(a, b, "ees")
    assert round(drift["mean_a"], 4) == 0.5905
    assert round(drift["mean_b"], 4) == 0.5895
    assert round(100.0 * (drift["mean_b"] - drift["mean_a"]), 2) == -0.11


def test_the_measured_version_split_is_what_esito_reports():
    """The number that replaces the borrowed 0.48 pp bound.

    This is the whole point of the 2026-08-30 GPU session: the residual
    serving-version split stops being *declarable* and becomes measured,
    on the same model over the same 7,904 rows.
    """
    a, b = _rows(V0271_SWEEP, V028_SWEEP)
    result = cd.compare(a, b)
    assert result["compared"] == 7904
    assert result["only_in_a"] == 0 and result["only_in_b"] == 0
    assert len(result["scored_moved"]) == 372
    assert round(100.0 * 372 / 7904.0, 2) == 4.71
    drift = cd.outcome_drift(a, b, "ees")
    assert round(100.0 * (drift["mean_b"] - drift["mean_a"]), 2) == 0.40
    # §14 accepted a 0.42 pp `--workers` drift. The version split lands
    # just under it — the comparison Methods makes, pinned here so it
    # cannot be quoted after either number has moved.
    assert 100.0 * (drift["mean_b"] - drift["mean_a"]) < 0.42


def test_half_the_version_split_is_ordinary_run_to_run_noise():
    """On IDENTICAL rows, so no difference of sample can explain it."""
    probe_a, probe_b = _rows(OMNI_A, OMNI_B)
    v0271, v028 = _rows(V0271_SWEEP, V028_SWEEP)
    keys = [k for k in probe_a if k in probe_b and k in v0271 and k in v028]
    assert len(keys) == 950

    def restrict(rows):
        return dict((k, rows[k]) for k in keys)

    within = cd.compare(restrict(probe_a), restrict(probe_b))
    between = cd.compare(restrict(v0271), restrict(v028))
    assert len(within["scored_moved"]) == 24     # 2.53%
    assert len(between["scored_moved"]) == 46    # 4.84%
    assert len(between["scored_moved"]) < 2.5 * len(within["scored_moved"]), (
        "the version split would have to be far larger than run-to-run "
        "nondeterminism for ESITO's reading to be wrong")


def test_the_v028_sweep_is_complete_and_carries_the_right_model_id():
    n = 0
    cells = set()
    with open(V028_SWEEP) as handle:
        for line in handle:
            row = json.loads(line)
            n += 1
            assert row["llm_model_served"] == \
                "cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"
            assert row.get("predicted_intent") is not None
            assert row["pathway"] == "omni"
            cells.add((row["degradation"], row["snr_db"]))
    assert n == 7904
    assert len(cells) == 19


def test_the_v028_sweep_stays_invisible_to_the_counts():
    """It is collected, not promoted.

    Which omni arm is confirmatory is a §14 decision for Leonardo and the
    advisor.  `result_sets.py` globs `results/*_scored.jsonl`
    NON-recursively, so a file in `results/v028/` cannot silently enter a
    count — and this test fails the moment someone moves it up a level
    without taking that decision.
    """
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import result_sets

    sweep = result_sets.sweep_files(os.path.join(REPO, "results"))
    names = [os.path.basename(str(p)) for p in sweep]
    assert "omni_qwen_sweep_scored.jsonl" in names
    assert len(sweep) == 116, (
        "the sweep is 116 files; adding v028 to it is a §14 decision, not "
        "a side effect")
    for path in sweep:
        assert "v028" not in str(path)
