"""Unit tests for scripts/model_frames.py — the pure frame/coding layer the
confirmatory regressions are built from.

Every test here pins one defect found by the 2026-08-14 audit of
fit_mixed_effects.py (THESIS_TRACKER.md §8, EXECUTION_TODO.md Step 7).
conftest.py puts scripts/ on sys.path, so the module imports by bare name.
"""
import pytest

import model_frames as mf


def test_error_flags_separates_critical_from_function():
    """H1 IS the comparison of these two coefficients, so both must exist and
    must be independently derivable. The audited code built only one."""
    critical_only = [{"type": "substitute", "slot_position": "critical"}]
    function_only = [{"type": "substitute", "slot_position": "function"}]
    both = critical_only + function_only

    assert mf.error_flags(critical_only) == (1, 0)
    assert mf.error_flags(function_only) == (0, 1)
    assert mf.error_flags(both) == (1, 1)
    assert mf.error_flags([]) == (0, 0)
    assert mf.error_flags(None) == (0, 0)


def test_snr_dummies_are_both_zero_on_clean_and_on_zero_db():
    """The nested coding §8 pins. A 4-level {clean,20,10,0} factor is exactly
    collinear with (degradation == clean) — the audited code built that factor
    and the fit came out rank-deficient."""
    assert mf.snr_dummies(None) == (0, 0)      # clean
    assert mf.snr_dummies(20) == (1, 0)
    assert mf.snr_dummies(10) == (0, 1)
    assert mf.snr_dummies(0) == (0, 0)         # within-degraded reference


# --- fixtures ---------------------------------------------------------------

def cascade_row(slurp_id=1, wer=0.2, tsa=1, ees=1, pf=1.0, degradation="babble",
                snr_db=10, asr="mlx-community/whisper-large-v3-turbo",
                llm="mlx-community/Qwen3.5-9B-MLX-8bit", cats=None,
                phonetic_distance=None, full_hallucination=False,
                gold_intent="alarm_set"):
    """One cascade row shaped like the real *_scored.jsonl schema (§10)."""
    return {
        "slurp_id": slurp_id, "pathway": "cascade", "model_family": "qwen",
        "asr_model": asr, "llm_model": llm, "degradation": degradation,
        "snr_db": snr_db, "wer": wer, "tsa": tsa, "ees": ees,
        "ees_strict": ees, "pf": pf, "gold_intent": gold_intent,
        "error_categories": cats if cats is not None else [],
        "phonetic_distance": phonetic_distance,
        "full_hallucination": full_hallucination,
    }


def omni_row(slurp_id=1, ees=1, tsa=1, pf=1.0, degradation="babble", snr_db=10,
             llm="cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit",
             model_family="qwen", gold_intent="alarm_set"):
    """One omni row: every transcript-derived field is null (§10)."""
    return {
        "slurp_id": slurp_id, "pathway": "omni", "model_family": model_family,
        "asr_model": None, "llm_model": llm, "degradation": degradation,
        "snr_db": snr_db, "wer": None, "tsa": tsa, "ees": ees,
        "ees_strict": ees, "pf": pf, "gold_intent": gold_intent,
        "error_categories": [], "phonetic_distance": None,
        "full_hallucination": None,
    }


# --- outcome selection ------------------------------------------------------

def test_h2_frame_defaults_to_ees_not_tsa():
    """CLAUDE.md hard rule: the design specifies ees. The audited code was
    hardcoded to tsa with no way to change it."""
    df, _ = mf.build_frame([cascade_row(tsa=1, ees=0)], "h2")
    assert mf.MODELS["h2"].default_outcome == "ees"
    assert df["outcome"].tolist() == [0]


def test_build_frame_rejects_an_unknown_outcome():
    with pytest.raises(ValueError):
        mf.build_frame([cascade_row()], "h2", outcome="accuracy")


def test_rows_with_a_null_outcome_are_dropped_and_counted():
    rows = [cascade_row(slurp_id=1, ees=1), cascade_row(slurp_id=2, ees=None)]
    df, diag = mf.build_frame(rows, "h2")
    assert len(df) == 1
    assert diag["dropped_null_outcome"] == 1


def test_wer_is_clamped_to_one_and_the_clamp_is_counted():
    """Repetition collapse produces WER of 30-50x; uncapped it is a leverage
    point that would dominate the fit."""
    df, diag = mf.build_frame([cascade_row(wer=55.75)], "h2")
    assert df["wer"].tolist() == [1.0]
    assert diag["wer_clamped"] == 1


# --- H1: intent layer -------------------------------------------------------

def test_h1_carries_both_error_flags_as_separate_columns():
    """H1 = 'has_critical_error is less negative than has_function_error'.
    Neither flag existed in the audited code, so H1 could not be stated."""
    row = cascade_row(cats=[{"type": "substitute", "slot_position": "critical"},
                            {"type": "insert", "slot_position": "function"}])
    df, _ = mf.build_frame([row], "h1")
    assert df["has_critical_error"].tolist() == [1]
    assert df["has_function_error"].tolist() == [1]
    terms = mf.MODELS["h1"].terms
    assert "has_critical_error" in terms and "has_function_error" in terms
    assert mf.MODELS["h1"].default_outcome == "tsa"


# --- H3: parameter layer ----------------------------------------------------

def test_h3_keeps_only_critical_slot_substitutions_with_a_distance():
    """§8 restricts H3 to rows with >=1 critical-slot substitution. A critical
    DELETION destroyed the value rather than mis-hearing it: there is no (ref,
    hyp) pair to score, and phonetic_distance is null."""
    kept = cascade_row(slurp_id=1, phonetic_distance=0.4,
                       cats=[{"type": "substitute", "slot_position": "critical"}])
    deletion = cascade_row(slurp_id=2, phonetic_distance=None,
                           cats=[{"type": "delete", "slot_position": "critical"}])
    function_sub = cascade_row(slurp_id=3, phonetic_distance=0.9,
                               cats=[{"type": "substitute", "slot_position": "function"}])

    df, diag = mf.build_frame([kept, deletion, function_sub], "h3")

    assert df["slurp_id"].tolist() == [1]
    assert diag["dropped_row_filter"] == 2
    assert mf.MODELS["h3"].default_outcome == "pf"
    assert "phonetic_distance" in mf.MODELS["h3"].terms


# --- H4b: pathway -----------------------------------------------------------

def test_h4b_keeps_omni_rows_even_though_their_wer_is_null():
    """Omni rows have no transcript, so wer is null by schema. The audited
    code dropped every row with a null wer, which would have silently emptied
    the omni arm."""
    df, _ = mf.build_frame([omni_row()], "h4b")
    assert df["pathway"].tolist() == ["omni"]
    assert len(df) == 1


def test_h4b_reference_cascade_is_whisper_turbo_into_the_large_router_only():
    """§8 pins the comparison to the reference cascade. Including Whisper Base
    or the 9B router would compare the omni model against a weaker cascade and
    inflate the pathway effect."""
    reference = cascade_row(slurp_id=1, asr="mlx-community/whisper-large-v3-turbo",
                            llm="Qwen/Qwen3.5-27B-GPTQ-Int4")
    small_router = cascade_row(slurp_id=2, asr="mlx-community/whisper-large-v3-turbo",
                               llm="mlx-community/Qwen3.5-9B-MLX-8bit")
    weak_asr = cascade_row(slurp_id=3, asr="mlx-community/whisper-base",
                           llm="Qwen/Qwen3.5-27B-GPTQ-Int4")

    df, _ = mf.build_frame([reference, small_router, weak_asr, omni_row(slurp_id=4)],
                           "h4b")

    assert sorted(df["slurp_id"].tolist()) == [1, 4]
    assert "pathway" in " ".join(mf.MODELS["h4b"].terms)


def test_h4b_terms_include_the_pathway_by_degradation_interaction():
    """H4b IS the interaction: 'does the pathway advantage depend on the kind
    of corruption?' A main effect alone does not answer it."""
    assert any(":" in t or "*" in t for t in mf.MODELS["h4b"].terms
               if "pathway" in t)


# --- the rank defect --------------------------------------------------------

def _realistic_cascade_rows():
    """clean + 2 degradations x 3 SNRs, 2 ASRs, 2 routers, 8 utterances.
    Shaped like the real sweep, which is where the rank deficiency showed up."""
    rows = []
    conditions = [("clean", None)]
    for deg in ("babble", "reverb"):
        for snr in (20, 10, 0):
            conditions.append((deg, snr))
    for sid in range(8):
        for asr in ("mlx-community/whisper-large-v3-turbo", "mlx-community/whisper-base"):
            for llm in ("Qwen3.5-9B", "Qwen3.5-27B"):
                for i, (deg, snr) in enumerate(conditions):
                    # wer must vary WITHIN a condition and the two error flags
                    # must be independently on/off, or the fixture itself is
                    # collinear and the rank check tests nothing.
                    cats = []
                    if (sid + i) % 3:
                        cats.append({"type": "substitute", "slot_position": "critical"})
                    if (sid * 3 + i) % 4:
                        cats.append({"type": "insert", "slot_position": "function"})
                    rows.append(cascade_row(
                        slurp_id=sid, asr=asr, llm=llm, degradation=deg,
                        snr_db=snr, wer=((sid * 7 + i * 3) % 11) / 10.0,
                        tsa=(sid + i) % 2, ees=(sid + i) % 2, pf=0.5,
                        cats=cats,
                    ))
    return rows


def test_h1_design_matrix_is_full_rank():
    """The audited fit built a 4-level {clean,20,10,0} SNR factor: design
    matrix (23712, 18) came out rank 17, smallest singular value exactly 0.
    With the nested coding the same design must be full rank."""
    import numpy as np

    df, _ = mf.build_frame(_realistic_cascade_rows(), "h1")
    X = mf.design_matrix(mf.formula(df, "h1"), df)

    assert np.linalg.matrix_rank(X) == X.shape[1]
    assert np.linalg.svd(X, compute_uv=False)[-1] > 1e-8


def test_formula_drops_a_term_that_does_not_vary():
    """A single-level factor is a constant column: it duplicates the intercept
    and makes the design singular for a reason that has nothing to do with the
    hypothesis."""
    rows = [cascade_row(slurp_id=i, llm="only-one-router", ees=i % 2,
                        wer=i / 10.0)
            for i in range(10)]
    df, _ = mf.build_frame(rows, "h2")
    f = mf.formula(df, "h2")
    assert "C(llm_model)" not in f


def test_formula_puts_the_selected_outcome_on_the_left():
    rows = [cascade_row(slurp_id=i, wer=i / 10.0, ees=i % 2) for i in range(6)]
    df, _ = mf.build_frame(rows, "h2", outcome="ees_strict")
    assert mf.formula(df, "h2").startswith("outcome ~ ")


def test_rank_report_detects_the_forbidden_four_level_snr_factor():
    """Proves the rank check can actually see the audited defect: the same
    frame, coded the forbidden way, must come back deficient."""
    df, _ = mf.build_frame(_realistic_cascade_rows(), "h1")
    df = df.copy()
    # The coding §8 forbids: one 4-level factor {clean, 20, 10, 0}. Its three
    # dummies sum to the 'any degradation' indicator, which C(degradation)
    # already carries.
    df["snr_level"] = ["clean" if (a, b) == (0, 0) and d == "clean"
                       else ("20" if a else ("10" if b else "0"))
                       for a, b, d in zip(df["snr_20"], df["snr_10"],
                                          df["degradation"])]
    forbidden = "outcome ~ C(asr_model) + C(degradation) + C(snr_level)"

    assert mf.rank_report(forbidden, df)["deficient"] is True
    assert mf.rank_report(mf.formula(df, "h1"), df)["deficient"] is False


def test_h4b_sensitivity_refit_can_exclude_hallucinated_rows():
    """§8: H4b is re-fit without full_hallucination rows so that 'the cascade
    collapses under babble because Whisper invented text' is distinguishable
    from 'the information was destroyed'."""
    normal = cascade_row(slurp_id=1, asr="mlx-community/whisper-large-v3-turbo",
                         llm="Qwen3.5-27B-GPTQ-Int4", full_hallucination=False)
    invented = cascade_row(slurp_id=2, asr="mlx-community/whisper-large-v3-turbo",
                           llm="Qwen3.5-27B-GPTQ-Int4", full_hallucination=True)

    keep_all, d_all = mf.build_frame([normal, invented], "h4b")
    dropped, d_drop = mf.build_frame([normal, invented], "h4b",
                                     drop_hallucinations=True)

    assert len(keep_all) == 2 and d_all["dropped_hallucination"] == 0
    assert dropped["slurp_id"].tolist() == [1]
    assert d_drop["dropped_hallucination"] == 1


def test_omni_rows_are_never_counted_as_hallucinations():
    """full_hallucination is null on omni rows by schema (there is no
    transcript to hallucinate). Treating null as True would delete the entire
    omni arm from the sensitivity refit."""
    df, diag = mf.build_frame([omni_row()], "h4b", drop_hallucinations=True)
    assert len(df) == 1
    assert diag["dropped_hallucination"] == 0


def test_level_coverage_counts_the_design_cells_each_level_appears_in():
    """A router that exists on 2 conditions and one that exists on 57 are not
    comparable: the router factor is confounded with which conditions were
    run. Found on the real data 2026-08-18 — the 200-row H2a pilot made
    C(llm_model) 'vary', so H2 fitted and meant nothing."""
    rows = []
    for i, (deg, snr) in enumerate([("clean", None), ("babble", 10),
                                    ("reverb", 0), ("noise", 20)]):
        for sid in range(5):
            rows.append(cascade_row(slurp_id=sid, degradation=deg, snr_db=snr,
                                    llm="small-router"))
            if deg == "babble":                      # pilot: one condition only
                rows.append(cascade_row(slurp_id=sid, degradation=deg,
                                        snr_db=snr, llm="big-router"))

    df, _ = mf.build_frame(rows, "h2")
    cov = mf.level_coverage(df, "llm_model")

    assert cov["small-router"] == 4
    assert cov["big-router"] == 1


def test_thin_level_is_reported_against_the_fullest_one():
    rows = []
    for deg, snr in [("clean", None), ("babble", 10), ("reverb", 0), ("noise", 20)]:
        for sid in range(5):
            rows.append(cascade_row(slurp_id=sid, degradation=deg, snr_db=snr,
                                    llm="small-router"))
            if deg == "babble":
                rows.append(cascade_row(slurp_id=sid, degradation=deg,
                                        snr_db=snr, llm="big-router"))
    df, _ = mf.build_frame(rows, "h2")

    thin = mf.thin_levels(df, "h2", min_coverage=0.5)

    assert thin and thin[0]["column"] == "llm_model"
    assert thin[0]["level"] == "big-router"
    assert thin[0]["coverage"] == pytest.approx(0.25)


def test_balanced_levels_are_not_flagged():
    rows = []
    for deg, snr in [("clean", None), ("babble", 10), ("reverb", 0), ("noise", 20)]:
        for sid in range(5):
            for llm in ("small-router", "big-router"):
                rows.append(cascade_row(slurp_id=sid, degradation=deg,
                                        snr_db=snr, llm=llm))
    df, _ = mf.build_frame(rows, "h2")

    assert mf.thin_levels(df, "h2", min_coverage=0.5) == []


def test_clean_is_not_flagged_for_having_no_snr_levels():
    """'clean' appears in 1 SNR combination and every degradation in 3 — that
    is the design, not an imbalance. The coverage guard is about system arms
    (which router / ASR / pathway was run where), not about conditions."""
    rows = []
    for deg, snr in [("clean", None), ("babble", 20), ("babble", 10), ("babble", 0)]:
        for sid in range(5):
            for asr in ("mlx-community/whisper-large-v3-turbo",
                        "mlx-community/whisper-base"):
                rows.append(cascade_row(slurp_id=sid, degradation=deg,
                                        snr_db=snr, asr=asr, llm="one-router"))
    df, _ = mf.build_frame(rows, "h1")

    assert mf.thin_levels(df, "h1", min_coverage=0.5, required_only=False) == []


# --------------------------------------------------------------------------
# §8 pins clean as H4b's degradation reference. The code did not.
# --------------------------------------------------------------------------
# `METRICS.md:75` says in the H4b block: "`clean` = reference level of
# degradation", and §3's capacity-confound defence depends on it — DESIGN.md
# :114: "the clean condition anchors each pathway's baseline capability, and
# the confirmatory test is the interaction, which cancels the constant capacity
# gap". The spec pinned Treatment(reference='cascade') on pathway and left
# degradation at patsy's alphabetical default, which is `babble`. Measured
# 2026-08-27 on the real fit: with babble as reference the omni x reverb
# interaction reads -0.1553, p = 0.0045; with clean it reads -0.0984,
# p = 0.171. Same fit, different parameterisation, opposite verdict on one
# hypothesis-relevant term.

def test_h4b_pins_clean_as_the_degradation_reference():
    spec = mf.MODELS["h4b"]
    terms = " ".join(spec.terms)
    assert "C(degradation, Treatment(reference='clean'))" in terms, (
        "§8 requires clean as H4b's degradation reference; got %r" % terms)
    assert "*C(degradation)" not in terms, (
        "bare C(degradation) leaves patsy's alphabetical default (babble)")


def test_h4b_still_pins_cascade_as_the_pathway_reference():
    terms = " ".join(mf.MODELS["h4b"].terms)
    assert "C(pathway, Treatment(reference='cascade'))" in terms


def test_h4b_required_term_matches_the_terms_it_declares():
    """The `required` guard must name the same interaction the formula builds,
    or the refusal that protects H4b silently stops matching."""
    spec = mf.MODELS["h4b"]
    for req in spec.required:
        assert req in spec.terms, "required %r is not among terms %r" % (req, spec.terms)
