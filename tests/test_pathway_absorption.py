"""Unit tests for §8 metric 5 — paired cascade-vs-omni absorption.

§8 (docs/design/METRICS.md, metric 5) defines it as: on the clips where the
REFERENCE CASCADE had WER > 0, `mean(EES_omni) - mean(EES_cascade)`. Positive
means the omni pathway recovers actions the cascade lost to transcription
error; negative means the text stage was protective.

conftest.py puts scripts/ on sys.path, so the module imports by bare name.
"""
import absorption_metrics as am


def cascade_row(slurp_id=1, wer=0.0, ees=1, degradation="noise", snr_db=10,
                asr="whisper-large-v3-turbo", llm="Qwen3.5-27B-GPTQ-Int4"):
    """A cascade row: it has a transcript, so it has a WER."""
    return {
        "slurp_id": slurp_id, "wer": wer, "ees": ees, "tsa": ees,
        "ees_strict": ees, "degradation": degradation, "snr_db": snr_db,
        "asr_model": asr, "llm_model": llm, "pathway": "cascade",
    }


def omni_row(slurp_id=1, ees=1, degradation="noise", snr_db=10,
             llm="cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"):
    """An omni row: NO transcript stage, so wer and asr_model are null.

    That is why §8 selects the subset on the cascade's WER — the omni side has
    no WER to select on. Measured on the real data 2026-08-28: every row of
    results/omni_qwen_sweep_scored.jsonl carries wer = None.
    """
    return {
        "slurp_id": slurp_id, "wer": None, "ees": ees, "tsa": ees,
        "ees_strict": ees, "degradation": degradation, "snr_db": snr_db,
        "asr_model": None, "llm_model": llm, "pathway": "omni",
    }


def test_subset_is_selected_by_the_cascade_wer_not_by_the_omni_rows():
    """Only clips the cascade transcribed imperfectly enter the comparison.

    Clip 2 is the discriminator: the cascade got it right from a perfect
    transcript and the omni model got it wrong. It must NOT count, because the
    question is what happens to actions the TRANSCRIPT put at risk. Including
    it turns a gain of +1.0 into +0.333.
    """
    cascade = [
        cascade_row(slurp_id=1, wer=0.5, ees=0),
        cascade_row(slurp_id=2, wer=0.0, ees=1),
        cascade_row(slurp_id=3, wer=0.3, ees=0),
    ]
    omni = [
        omni_row(slurp_id=1, ees=1),
        omni_row(slurp_id=2, ees=0),
        omni_row(slurp_id=3, ees=1),
    ]

    got = am.pathway_paired_stats(cascade, omni)

    assert got["n_pairs"] == 2
    assert got["cascade_rate"] == 0.0
    assert got["omni_rate"] == 1.0
    assert got["gain_pathway"] == 1.0


def test_a_pair_with_a_null_outcome_on_either_side_is_dropped_not_crashed_on():
    """`cell_stats` already filters null outcomes; this must match it.

    Without the filter the None reaches `_mean` and the whole cell dies with a
    TypeError, which in a per-cell loop looks like the analysis failing rather
    than one clip being unscored.
    """
    cascade = [
        cascade_row(slurp_id=1, wer=0.5, ees=0),
        cascade_row(slurp_id=2, wer=0.5, ees=None),
        cascade_row(slurp_id=3, wer=0.5, ees=1),
    ]
    omni = [
        omni_row(slurp_id=1, ees=1),
        omni_row(slurp_id=2, ees=1),
        omni_row(slurp_id=3, ees=None),
    ]

    got = am.pathway_paired_stats(cascade, omni)

    assert got["n_pairs"] == 1
    assert got["cascade_rate"] == 0.0
    assert got["omni_rate"] == 1.0


def test_rows_spanning_two_cells_are_refused_not_silently_collapsed():
    """The same slurp_id exists at every SNR, so a mixed call is ambiguous.

    Pairing keys on slurp_id. Hand it clip 1 at snr 20 AND at snr 0 and the
    omni lookup keeps whichever landed last, pairing a 20 dB cascade row
    against a 0 dB omni row and reporting the result as one cell. It would
    print a number, and the number would be meaningless.

    `tabulate_error_subtypes.py` already refuses a mix of routers for the same
    reason; this is that rule applied to the pathway comparison.
    """
    cascade = [
        cascade_row(slurp_id=1, wer=0.5, ees=0, snr_db=20),
        cascade_row(slurp_id=1, wer=0.5, ees=1, snr_db=0),
    ]
    omni = [
        omni_row(slurp_id=1, ees=1, snr_db=20),
        omni_row(slurp_id=1, ees=0, snr_db=0),
    ]

    try:
        am.pathway_paired_stats(cascade, omni)
    except ValueError as exc:
        assert "one cell" in str(exc).lower() or "cell" in str(exc).lower()
    else:
        raise AssertionError(
            "a call spanning two cells must raise, not return a number")


def test_underpowered_flag_trips_below_thirty_not_at_thirty():
    """Mirrors the tracker convention already pinned for the cascade metrics.

    A per-degradation pathway cell can be thin — the WER>0 subset shrinks as
    the audio gets cleaner — so the reader has to be told which cells are
    exploratory rather than left to divide n by hand.
    """
    def cell(n):
        cascade = [cascade_row(slurp_id=i, wer=0.5, ees=0) for i in range(n)]
        omni = [omni_row(slurp_id=i, ees=1) for i in range(n)]
        return am.pathway_paired_stats(cascade, omni)

    assert cell(29)["underpowered"] is True
    assert cell(30)["underpowered"] is False


def test_an_outcome_outside_the_valid_set_is_refused():
    """`pf` is a recall proportion, not a success flag, and is excluded.

    Same guard `cell_stats` carries: a typo'd outcome name must not silently
    return None for every field and read as an empty cell.
    """
    try:
        am.pathway_paired_stats([], [], outcome="pf")
    except ValueError as exc:
        assert "pf" in str(exc) or "outcome" in str(exc)
    else:
        raise AssertionError("an invalid outcome must raise")


def test_the_bootstrap_resamples_the_PAIR_not_the_two_sides_independently():
    """The clip is the unit; its cascade row and its omni row move together.

    Two clips, opposite outcomes: clip A (cascade 0, omni 1) contributes +1,
    clip B (cascade 1, omni 0) contributes -1. Drawing two clips with
    replacement can only give AA, AB, BA or BB, so the gain of any replicate
    is exactly -1, 0 or +1.

    Resample the two sides independently and the pairing dissolves: cascade
    from {A,A} against omni from {A,B} gives 0.0 - 0.5 = +0.5, a value this
    design can never produce. So a single +0.5 in the replicates proves the
    pairing broke — which is why the assertion is on the SET of values and not
    on the width of the interval.
    """
    import numpy as np

    cascade = [
        cascade_row(slurp_id="A", wer=0.5, ees=0),
        cascade_row(slurp_id="B", wer=0.5, ees=1),
    ]
    omni = [
        omni_row(slurp_id="A", ees=1),
        omni_row(slurp_id="B", ees=0),
    ]

    got = am.pathway_paired_ci(cascade, omni, np.random.default_rng(42),
                               n_boot=200)

    assert len(got["replicates"]) == 200
    assert set(got["replicates"]) <= {-1.0, 0.0, 1.0}, (
        "a replicate outside {-1, 0, +1} means the cascade and omni rows were "
        "resampled independently and the pairing broke: %r"
        % sorted(set(got["replicates"]))
    )


def test_a_duplicated_clip_is_refused_not_silently_collapsed():
    """`route_omni.py` opens its output in APPEND mode (route_omni.py:311).

    Re-run it without deleting the previous file and the clip appears twice.
    Indexing by slurp_id keeps whichever row landed last and reports a full,
    healthy-looking cell — the same silent-success failure shape as the glob
    that scored 19 of 38 files. The duplicate must stop the calculation.
    """
    cascade = [
        cascade_row(slurp_id=1, wer=0.5, ees=0),
        cascade_row(slurp_id=1, wer=0.5, ees=1),   # same clip, second run
    ]
    omni = [omni_row(slurp_id=1, ees=1)]

    try:
        am.pathway_paired_stats(cascade, omni)
    except ValueError as exc:
        assert "duplicat" in str(exc).lower()
    else:
        raise AssertionError(
            "a duplicated clip must raise, not resolve to the last row seen")


def test_sign_is_negative_when_the_text_stage_was_protective():
    """Regression pin, not a new behaviour: §8 fixes the direction.

    "Positive = the omni pathway recovers actions the cascade lost; negative =
    the text stage was protective." A sign flip here would read as the opposite
    scientific claim while every count and interval stayed plausible — the
    exact failure this project already carried to the advisor on 2026-08-07 and
    has to restate in September. Pinned deliberately.
    """
    cascade = [cascade_row(slurp_id=i, wer=0.5, ees=1) for i in range(4)]
    omni = [omni_row(slurp_id=i, ees=0) for i in range(4)]

    got = am.pathway_paired_stats(cascade, omni)

    assert got["gain_pathway"] == -1.0, "cascade beat omni; the gain must be negative"


def test_pooling_per_degradation_weights_by_n_not_by_averaging_rates():
    """"Per degradation" pools three SNR cells, which are never equal in size.

    The WER>0 subset shrinks as the audio gets cleaner, so at 20 dB a cell can
    hold a handful of clips and at 0 dB almost all of them. Averaging the three
    gains gives the tiny cell the same vote as the large one.

    Here: cell A is 10 clips with gain +1, cell B is 90 clips with gain 0.
    The mean of the gains is +0.5. The pooled truth is +0.1 — five times
    smaller, and on the other side of "roughly nothing".
    """
    cells = [
        {"n_pairs": 10, "cascade_rate": 0.0, "omni_rate": 1.0},
        {"n_pairs": 90, "cascade_rate": 1.0, "omni_rate": 1.0},
    ]

    got = am.pool_pathway_cells(cells)

    assert got["n_pairs"] == 100
    assert abs(got["cascade_rate"] - 0.9) < 1e-12
    assert abs(got["omni_rate"] - 1.0) < 1e-12
    assert abs(got["gain_pathway"] - 0.1) < 1e-12
