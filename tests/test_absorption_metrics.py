"""Unit tests for scripts/absorption_metrics.py.

conftest.py puts scripts/ on sys.path, so the module imports by bare name.
"""
import json
import os

import numpy as np
import pytest

import absorption_metrics as am


def row(slurp_id=1, wer=0.0, ees=1, tsa=1, degradation="noise", snr_db=10,
        asr="whisper-large-v3-turbo", llm="qwen9b"):
    """One results row, shaped like the real *_scored.jsonl schema."""
    return {
        "slurp_id": slurp_id,
        "wer": wer,
        "ees": ees,
        "tsa": tsa,
        "ees_strict": ees,
        "degradation": degradation,
        "snr_db": snr_db,
        "asr_model": asr,
        "llm_model": llm,
    }


def test_cell_key_strips_model_path_and_defaults_clean():
    r = row(degradation=None, asr="mlx-community/whisper-large-v3-turbo")
    assert am.cell_key(r) == ("whisper-large-v3-turbo", "clean", 10, "qwen9b")


def test_cap_wer_clamps_repetition_collapse():
    assert am.cap_wer(40.0) == 1.0
    assert am.cap_wer(0.25) == 0.25


def test_gain_ceiling_is_invariant_to_the_number_of_perfect_rows():
    """The defect being fixed: gain must not move just because more
    perfectly-transcribed rows were added to the cell. Ten errored rows,
    half of them successful, against a ceiling of 1.0 in both fixtures."""
    errored = [row(slurp_id=i, wer=0.5, ees=1 if i < 5 else 0) for i in range(10)]
    few_perfect = [row(slurp_id=100 + i, wer=0.0, ees=1) for i in range(10)]
    many_perfect = [row(slurp_id=100 + i, wer=0.0, ees=1) for i in range(90)]

    a = am.cell_stats(errored + few_perfect)
    b = am.cell_stats(errored + many_perfect)

    assert a["ceiling"] == 1.0 and b["ceiling"] == 1.0
    assert a["absorption"] == 0.5 and b["absorption"] == 0.5
    assert a["gain_ceiling"] == pytest.approx(0.0)
    assert b["gain_ceiling"] == pytest.approx(0.0)
    # ...while the legacy comparator moves a lot, which is the bug.
    assert a["gain_legacy"] == pytest.approx(-0.25)
    assert b["gain_legacy"] == pytest.approx(-0.45)


def test_ceiling_of_one_half_halves_the_baseline():
    errored = [row(slurp_id=i, wer=0.5, ees=1 if i < 5 else 0) for i in range(10)]
    perfect = [row(slurp_id=100 + i, wer=0.0, ees=1 if i < 5 else 0) for i in range(10)]
    s = am.cell_stats(errored + perfect)
    assert s["ceiling"] == pytest.approx(0.5)
    assert s["naive_ceiling"] == pytest.approx(0.25)   # 0.5 * (1 - 0.5)
    assert s["gain_ceiling"] == pytest.approx(0.25)    # 0.5 - 0.25


def test_ceiling_is_none_when_no_perfect_rows_exist():
    """whisper-base babble/noise/reverb @ 0 dB. Must be None, never a default."""
    s = am.cell_stats([row(slurp_id=i, wer=0.8, ees=0) for i in range(10)])
    assert s["n_ceiling"] == 0
    assert s["ceiling"] is None
    assert s["naive_ceiling"] is None
    assert s["gain_ceiling"] is None
    assert s["gain_legacy"] is not None


def test_collapse_rows_are_capped_not_dropped():
    rows = [row(slurp_id=1, wer=40.0, ees=0), row(slurp_id=2, wer=0.5, ees=0)]
    s = am.cell_stats(rows)
    assert s["n_collapse"] == 1
    assert s["mean_wer_err"] == pytest.approx(0.75)   # (1.0 + 0.5) / 2


def test_rows_with_null_outcome_leave_the_outcome_denominator():
    rows = [row(slurp_id=1, wer=0.5, ees=1), row(slurp_id=2, wer=0.5, ees=None)]
    s = am.cell_stats(rows)
    assert s["n"] == 2
    assert s["n_scored"] == 1
    assert s["absorption"] == 1.0


def test_outcome_argument_selects_the_column():
    rows = [row(slurp_id=1, wer=0.5, ees=0, tsa=1)]
    assert am.cell_stats(rows, outcome="ees")["absorption"] == 0.0
    assert am.cell_stats(rows, outcome="tsa")["absorption"] == 1.0


def test_empty_cell_returns_nones_not_exceptions():
    s = am.cell_stats([])
    assert s["n"] == 0
    assert s["absorption"] is None
    assert s["gain_legacy"] is None
    assert s["gain_ceiling"] is None


def clean_row(slurp_id=1, ees=1, wer=0.0, asr="whisper-large-v3-turbo", llm="qwen9b"):
    return row(slurp_id=slurp_id, wer=wer, ees=ees, tsa=ees,
               degradation="clean", snr_db=None, asr=asr, llm=llm)


def test_paired_subset_inclusion_rules():
    """Item 1 qualifies. Item 2's anchor failed on clean. Item 3 has a perfect
    degraded transcript, so there is no ASR error to absorb. Item 4 has no
    clean anchor at all."""
    clean_index = am.build_clean_index([
        clean_row(slurp_id=1, ees=1),
        clean_row(slurp_id=2, ees=0),
        clean_row(slurp_id=3, ees=1),
    ])
    degraded = [
        row(slurp_id=1, wer=0.4, ees=1),
        row(slurp_id=2, wer=0.4, ees=1),
        row(slurp_id=3, wer=0.0, ees=1),
        row(slurp_id=4, wer=0.4, ees=1),
    ]
    kept, dropped = am.paired_rows(degraded, clean_index)
    assert [r["slurp_id"] for r in kept] == [1]
    assert dropped == 1


def test_anchor_need_not_have_a_perfect_transcript():
    """The anchor is end-to-end competence on clean audio, not WER = 0 —
    Whisper Base reaches WER = 0 on only 126 of 416 clean rows."""
    clean_index = am.build_clean_index([clean_row(slurp_id=1, ees=1, wer=0.3)])
    kept, _ = am.paired_rows([row(slurp_id=1, wer=0.4, ees=1)], clean_index)
    assert len(kept) == 1


def test_gain_paired_uses_the_paired_subset_wer():
    """Two qualifying items, one survives; mean WER on the subset is 0.5,
    so the honest comparator is 0.5 and the gain is 0.0."""
    clean_index = am.build_clean_index([
        clean_row(slurp_id=1, ees=1), clean_row(slurp_id=2, ees=1),
    ])
    degraded = [row(slurp_id=1, wer=0.5, ees=1), row(slurp_id=2, wer=0.5, ees=0)]
    s = am.paired_stats(degraded, clean_index)
    assert s["n_paired"] == 2
    assert s["retention"] == pytest.approx(0.5)
    assert s["mean_wer_paired"] == pytest.approx(0.5)
    assert s["naive_paired"] == pytest.approx(0.5)
    assert s["gain_paired"] == pytest.approx(0.0)


def test_clean_cell_yields_empty_paired_columns_not_zeros():
    """A clean cell is the reference, not a result."""
    clean_index = am.build_clean_index([clean_row(slurp_id=1, ees=1)])
    s = am.paired_stats([clean_row(slurp_id=1, ees=1)], clean_index)
    assert s["n_paired"] == 0
    assert s["retention"] is None
    assert s["gain_paired"] is None


def test_underpowered_flag_trips_below_thirty_not_at_thirty():
    def build(n):
        clean_index = am.build_clean_index(
            [clean_row(slurp_id=i, ees=1) for i in range(n)])
        degraded = [row(slurp_id=i, wer=0.4, ees=1) for i in range(n)]
        return am.paired_stats(degraded, clean_index)

    assert build(29)["underpowered"] is True
    assert build(30)["underpowered"] is False


def test_clean_index_keys_on_asr_and_llm_too():
    """Two ASRs share a slurp_id; each must find its own anchor."""
    idx = am.build_clean_index([
        clean_row(slurp_id=1, ees=1, asr="whisper-large-v3-turbo"),
        clean_row(slurp_id=1, ees=0, asr="parakeet-tdt-0.6b-v3"),
    ])
    turbo, _ = am.paired_rows(
        [row(slurp_id=1, wer=0.4, ees=1, asr="whisper-large-v3-turbo")], idx)
    parakeet, _ = am.paired_rows(
        [row(slurp_id=1, wer=0.4, ees=1, asr="parakeet-tdt-0.6b-v3")], idx)
    assert len(turbo) == 1
    assert len(parakeet) == 0


@pytest.mark.parametrize("clean_ees,deg_ees,deg_wer,expected", [
    (1, 0, 0.4, "PROPAGATED"),
    (1, 1, 0.4, "ABSORBED"),
    (0, 1, 0.4, "RECOVERED"),
    (0, 0, 0.4, "SAME_FAIL"),
    (1, 1, 0.0, "SAME_OK"),
    (1, 0, 0.0, "NEW_ERROR"),
])
def test_classify_pair(clean_ees, deg_ees, deg_wer, expected):
    c = clean_row(slurp_id=1, ees=clean_ees)
    d = row(slurp_id=1, wer=deg_wer, ees=deg_ees)
    assert am.classify_pair(c, d) == expected


def test_classify_pair_absorbed_matches_the_paired_subset():
    """ABSORBED + PROPAGATED must be exactly the paired subset, so the two
    scripts can never report different absorption numbers."""
    clean_index = am.build_clean_index([
        clean_row(slurp_id=i, ees=1 if i < 3 else 0) for i in range(5)])
    degraded = [row(slurp_id=i, wer=0.4, ees=1 if i % 2 == 0 else 0)
                for i in range(5)]
    kept, _ = am.paired_rows(degraded, clean_index)
    cats = [am.classify_pair(clean_index[am.item_key(d)], d)
            for d in degraded if am.item_key(d) in clean_index]
    n_absorbed_or_propagated = sum(
        1 for c in cats if c in ("ABSORBED", "PROPAGATED"))
    assert n_absorbed_or_propagated == len(kept)


LEGACY_FILE = "results/cascade_whisper_qwen9b_clean_scored.jsonl"


@pytest.mark.skipif(not os.path.exists(LEGACY_FILE),
                    reason="results/ not present in this checkout")
def test_gain_legacy_reproduces_the_shipped_august_csv():
    """absorption_qwen9b.csv:43 — whisper-large-v3-turbo, clean, TSA outcome.
    Pinning it proves the correction comes from the new comparators and not
    from an unrelated change to how rows are loaded or filtered."""
    rows = []
    with open(LEGACY_FILE) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("asr_model") or "").split("/")[-1] == "whisper-large-v3-turbo":
                rows.append(r)

    s = am.cell_stats(rows, outcome="tsa")
    assert s["n"] == 416
    assert s["n_err"] == 225
    assert s["absorption"] == pytest.approx(0.6533333333333333)
    assert s["naive_legacy"] == pytest.approx(0.8256760488265571)
    assert s["gain_legacy"] == pytest.approx(-0.17234271549322377)


def test_resample_by_slurp_id_keeps_an_items_rows_together():
    rows = [row(slurp_id=i, wer=0.4, ees=1) for i in range(3)]
    rows += [row(slurp_id=i, wer=0.4, ees=0, asr="parakeet-tdt-0.6b-v3")
             for i in range(3)]
    rng = np.random.default_rng(42)
    sample = am.resample_by_slurp_id(rows, rng)
    assert len(sample) == len(rows)
    counts = {}
    for r in sample:
        counts.setdefault(r["slurp_id"], []).append(r["asr_model"])
    # every drawn id contributes both of its rows, every time it is drawn
    for sid, asrs in counts.items():
        assert len(asrs) % 2 == 0
        assert asrs.count("whisper-large-v3-turbo") == asrs.count("parakeet-tdt-0.6b-v3")


def test_resample_is_deterministic_for_a_seed():
    rows = [row(slurp_id=i, wer=0.4, ees=1) for i in range(20)]
    a = am.resample_by_slurp_id(rows, np.random.default_rng(7))
    b = am.resample_by_slurp_id(rows, np.random.default_rng(7))
    assert [r["slurp_id"] for r in a] == [r["slurp_id"] for r in b]
