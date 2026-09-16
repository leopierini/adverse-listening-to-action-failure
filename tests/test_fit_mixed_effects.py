"""Unit tests for the estimator layer of scripts/fit_mixed_effects.py.

§8 pins ONE confirmatory estimator: statsmodels GEE, binomial family,
exchangeable working correlation clustered on slurp_id, robust SEs. The
pre-audit script silently fell back to smf.logit — which ignores clustering
and inflates significance — and recorded that only inside the saved text.
"""
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

import fit_mixed_effects as fme
import model_frames as mf


def rows_that_fit(n_utterances=40, seed=0):
    """A small but non-degenerate cascade sweep: n utterances x 7 conditions
    x 2 ASRs. The outcome depends on the predictors with real noise on top —
    a deterministic outcome separates perfectly and the fit is refused rather
    than estimated, which tests the guard instead of the estimator.
    """
    rng = random.Random(seed)
    conditions = [("clean", None)]
    for deg in ("babble", "reverb"):
        for snr in (20, 10, 0):
            conditions.append((deg, snr))
    rows = []
    routers = ("Qwen3.5-9B-MLX-8bit", "Qwen3.5-27B-GPTQ-Int4")
    for sid in range(n_utterances):
      for llm_i, llm in enumerate(routers):
        for asr_i, asr in enumerate(("whisper-large-v3-turbo", "whisper-base")):
            for deg, snr in conditions:
                # The weaker ASR must actually be weaker, or its coefficient
                # carries no information and the sandwich covariance collapses.
                base = 0.0 if deg == "clean" else {20: 0.15, 10: 0.30, 0: 0.55}[snr]
                wer = min(1.0, max(0.0, base + 0.15 * asr_i + rng.gauss(0, 0.08)))
                p_ok = 0.9 - 0.7 * wer + 0.04 * llm_i
                rows.append({
                    "slurp_id": sid, "pathway": "cascade", "model_family": "qwen",
                    "asr_model": asr, "llm_model": llm,
                    "degradation": deg, "snr_db": snr, "wer": wer,
                    "tsa": int(rng.random() < p_ok),
                    "ees": int(rng.random() < p_ok - 0.05),
                    "ees_strict": int(rng.random() < p_ok - 0.15),
                    "pf": 0.5, "gold_intent": "alarm_set",
                    "error_categories": [], "phonetic_distance": None,
                    "full_hallucination": False,
                })
    return rows


def test_fit_uses_gee_clustered_on_the_utterance():
    df, _ = mf.build_frame(rows_that_fit(), "h2")
    res = fme.fit_confirmatory(df, mf.formula(df, "h2"))

    assert res.estimator == "GEE"
    assert res.cov_struct == "Exchangeable"
    assert res.family == "Binomial"
    assert res.group_col == "slurp_id"
    assert res.n_clusters == 40
    assert res.n_obs == len(df)


def test_fit_refuses_a_rank_deficient_design():
    """statsmodels returns coefficients and standard errors for a singular
    design without complaint. The audited fit did exactly that."""
    df, _ = mf.build_frame(rows_that_fit(), "h2")
    df = df.copy()
    df["wer_copy"] = df["wer"]

    with pytest.raises(fme.RankDeficientDesign):
        fme.fit_confirmatory(df, "outcome ~ wer + wer_copy")


def test_fit_does_not_substitute_an_unclustered_estimator_on_failure():
    """The silent downgrade: GEE fails, smf.logit answers instead, and the
    caller cannot tell. Failure must surface as a failure."""
    df, _ = mf.build_frame(rows_that_fit(), "h2")

    with pytest.raises(fme.EstimatorFailed) as excinfo:
        fme.fit_confirmatory(df, "outcome ~ a_column_that_does_not_exist")
    assert "a_column_that_does_not_exist" in str(excinfo.value)


def test_summary_text_names_the_estimator_that_actually_ran():
    """The saved artifact is what reaches the thesis; it must state its own
    provenance rather than leaving the reader to assume GEE."""
    df, _ = mf.build_frame(rows_that_fit(), "h2")
    res = fme.fit_confirmatory(df, mf.formula(df, "h2"))
    text = res.as_text()

    assert "GEE" in text and "Exchangeable" in text
    assert "slurp_id" in text


def test_non_finite_standard_errors_are_rejected():
    """A NaN robust SE means the sandwich covariance collapsed. statsmodels
    prints the table anyway, with a blank p-value that reads as 'not
    significant' rather than 'not estimated'.

    Checked as a pure function: the numerical conditions that make GEE emit a
    NaN are knife-edge, so driving one through a real fit would be a flaky test
    of statsmodels rather than a stable test of the guard.
    """
    with pytest.raises(fme.EstimatorFailed) as excinfo:
        fme.check_standard_errors([0.11, float("nan"), 0.42],
                                  ["Intercept", "wer", "snr_20"],
                                  "outcome ~ wer + snr_20")
    assert "standard error" in str(excinfo.value).lower()
    assert "wer" in str(excinfo.value)


def test_finite_standard_errors_pass_the_guard():
    assert fme.check_standard_errors([0.11, 0.2], ["Intercept", "wer"],
                                     "outcome ~ wer") is None


def test_fit_rejects_a_perfectly_separated_fit():
    """Separation gives finite but astronomical SEs and a coefficient of ~±20.
    The table prints, looks like a huge effect, and means nothing."""
    rows = rows_that_fit()
    for r in rows:                      # outcome becomes a function of wer alone
        r["ees"] = int(r["wer"] < 0.5)
    df, _ = mf.build_frame(rows, "h2")

    with pytest.raises(fme.EstimatorFailed) as excinfo:
        fme.fit_confirmatory(df, "outcome ~ wer")
    # Separation surfaces through whichever guard sees it first: statsmodels
    # only warns when its GLM starting values detect it, otherwise it shows up
    # as a collapsed sandwich covariance. Either way the fit is refused.
    msg = str(excinfo.value).lower()
    assert "separation" in msg or "standard error" in msg


def test_h1_contrast_recovers_a_planted_difference_between_the_two_flags():
    """H1 is the COMPARISON of two coefficients, so it needs a contrast, not
    two separately-read p-values. Planted: a function-word error is far more
    damaging than a critical-slot one, so critical - function must be > 0."""
    df, _ = mf.build_frame(rows_with_error_flags(), "h1")
    res = fme.fit_confirmatory(df, mf.formula(df, "h1"), model="h1")
    diff = res.contrast("has_critical_error - has_function_error = 0")

    assert float(diff.effect[0]) > 0
    assert float(diff.pvalue) < 0.05


def test_one_broken_hypothesis_does_not_hide_the_others():
    """Five hypotheses are fitted in one run. If H3 has no rows yet, H1 and H2
    must still be reported — the audited script returned early on the first
    problem and printed nothing at all."""
    report = fme.run_models(rows_with_error_flags(), ["h1", "h2", "h3"])

    assert set(report) == {"h1", "h2", "h3"}
    assert report["h1"].fit is not None
    assert report["h2"].fit is not None
    assert report["h3"].fit is None          # no critical-slot substitutions
    assert report["h3"].error is not None


def test_h1_report_carries_the_contrast_not_just_two_coefficients():
    report = fme.run_models(rows_with_error_flags(), ["h1"])
    assert report["h1"].error is None
    assert report["h1"].contrast is not None
    assert "has_critical_error - has_function_error" in report["h1"].contrast_expr


def test_h1_fails_loudly_when_its_two_flags_do_not_vary():
    """Without both flags the fit still runs — it just answers a different
    question. Reporting it as H1 is the failure mode; refusing is correct."""
    report = fme.run_models(rows_that_fit(), ["h1"])   # no error_categories

    assert report["h1"].fit is None
    assert "has_function_error" in str(report["h1"].error)


def rows_with_error_flags(seed=7):
    """rows_that_fit() plus independently-varying critical/function error
    flags, with a mild penalty planted on critical and a heavy one on
    function — the direction H1 predicts."""
    rng = random.Random(seed)
    rows = rows_that_fit()
    for r in rows:
        critical = rng.random() < 0.5
        function = rng.random() < 0.4
        r["error_categories"] = []
        if critical:
            r["error_categories"].append(
                {"type": "substitute", "slot_position": "critical"})
        if function:
            r["error_categories"].append(
                {"type": "insert", "slot_position": "function"})
        p_ok = 0.9 - 0.15 * critical - 0.55 * function
        r["tsa"] = int(rng.random() < p_ok)
    return rows


def test_the_outcome_flag_reaches_every_model():
    report = fme.run_models(rows_that_fit(), ["h2"], outcome="ees_strict")
    assert report["h2"].outcome == "ees_strict"


def test_cli_reports_every_model_and_writes_the_fitted_ones(tmp_path):
    """End-to-end: a model that cannot be tested yet must appear in the report
    with its reason, not vanish."""
    import json
    import subprocess

    src = tmp_path / "rows.jsonl"
    with src.open("w") as f:
        for r in rows_with_error_flags():
            f.write(json.dumps(r) + "\n")
    outdir = tmp_path / "out"

    proc = subprocess.run(
        [sys.executable, "scripts/fit_mixed_effects.py", str(src),
         "--models", "h1", "h2", "h3", "--out-dir", str(outdir)],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )

    assert proc.returncode == 0, proc.stderr
    assert (outdir / "h1_tsa.txt").exists()
    assert (outdir / "h2_ees.txt").exists()
    assert not (outdir / "h3_pf.txt").exists()      # no rows for H3 in this data
    assert "H3" in proc.stdout and "NOT FITTED" in proc.stdout
    report = json.loads((outdir / "diagnostics.json").read_text())
    assert report["h3"]["error"]
    assert report["h1"]["contrast"]["expression"].startswith("has_critical_error")


def test_a_pilot_sized_router_does_not_get_fitted_as_if_it_were_a_full_arm():
    """The real failure this guard was written for: 200 pilot rows of the 27B
    on one condition sat in results/ alongside 23,712 rows of the 9B, so
    C(llm_model) 'varied' and H2 fitted a number that meant nothing."""
    rows = rows_that_fit()
    pilot = [dict(r) for r in rows
             if r["llm_model"] == "Qwen3.5-27B-GPTQ-Int4"
             and r["degradation"] == "babble" and r["snr_db"] == 10]
    full = [r for r in rows if r["llm_model"] != "Qwen3.5-27B-GPTQ-Int4"]

    report = fme.run_models(full + pilot, ["h2"])

    assert report["h2"].fit is None
    assert "Qwen3.5-27B-GPTQ-Int4" in str(report["h2"].error)
    assert "coverage" in str(report["h2"].error).lower()


def test_the_coverage_guard_can_be_relaxed_deliberately():
    """It is a methodological choice, not a law: making it explicit keeps the
    decision in the run command instead of hidden in a constant."""
    rows = rows_that_fit()
    pilot = [dict(r) for r in rows
             if r["llm_model"] == "Qwen3.5-27B-GPTQ-Int4"
             and r["degradation"] == "babble" and r["snr_db"] == 10]
    full = [r for r in rows if r["llm_model"] != "Qwen3.5-27B-GPTQ-Int4"]

    report = fme.run_models(full + pilot, ["h2"], min_level_coverage=0.0)

    assert report["h2"].fit is not None


def test_a_thin_level_on_a_nuisance_covariate_is_warned_about_not_refused():
    """In H1 the router is a nuisance covariate, not the hypothesis. A pilot-
    sized level there does not invalidate the H1 contrast, but it must not pass
    unremarked either — it is estimated from 200 rows on 2 conditions."""
    rows = rows_with_error_flags()
    pilot = [dict(r) for r in rows
             if r["llm_model"] == "Qwen3.5-27B-GPTQ-Int4"
             and r["degradation"] == "babble" and r["snr_db"] == 10]
    full = [r for r in rows if r["llm_model"] != "Qwen3.5-27B-GPTQ-Int4"]

    report = fme.run_models(full + pilot, ["h1"])

    assert report["h1"].fit is not None          # H1 is still testable
    assert report["h1"].warnings
    assert "Qwen3.5-27B-GPTQ-Int4" in " ".join(report["h1"].warnings)


def test_a_proportion_outcome_is_labelled_as_a_fractional_logit():
    """PF is recall in [0,1], not a Bernoulli draw. Fitting it with a binomial
    family is fractional logistic regression (Papke & Wooldridge), which is
    consistent — but the saved artifact must say so rather than letting a
    reader assume a binary outcome."""
    rows = rows_that_fit()
    rng = random.Random(11)
    for r in rows:
        r["pf"] = round(min(1.0, max(0.0, 0.9 - 0.7 * r["wer"] + rng.gauss(0, 0.1))), 3)
        r["error_categories"] = [{"type": "substitute", "slot_position": "critical"}]
        r["phonetic_distance"] = rng.random()
    df, _ = mf.build_frame(rows, "h3")
    res = fme.fit_confirmatory(df, mf.formula(df, "h3"), model="h3")

    assert "fractional" in res.as_text().lower()


def test_a_binary_outcome_is_not_labelled_a_fractional_logit():
    df, _ = mf.build_frame(rows_that_fit(), "h2")
    res = fme.fit_confirmatory(df, mf.formula(df, "h2"), model="h2")
    assert "fractional" not in res.as_text().lower()


# --------------------------------------------------------------------------
# H4b's confirmatory answer must live in the artifact, not only in prose
# --------------------------------------------------------------------------
# §3's operational for H4b is "the pathway x degradation interaction is
# significant", i.e. a JOINT test over the six interaction terms — not any one
# coefficient, each of which depends on which degradation is the reference.
# Until 2026-08-28 that test was computed by hand and written into §13, so the
# one number the hypothesis turns on could not be checked from any file. The
# individual coefficients were in the artifact; the answer was not.

def _toy_fit():
    df, _ = mf.build_frame(rows_that_fit(), "h2")
    return fme.fit_confirmatory(df, mf.formula(df, "h2"))


def test_the_fit_wrapper_can_jointly_test_a_set_of_terms():
    """A joint Wald over several coefficients, not a single contrast."""
    fit = _toy_fit()
    terms = [k for k in fit.result.params.index if k != "Intercept"]
    joint = fit.joint_test(terms)
    assert joint["df"] == len(terms)
    assert joint["statistic"] > 0
    assert 0.0 <= joint["p_value"] <= 1.0
    assert joint["terms"] == terms


def test_the_joint_test_refuses_a_term_that_is_not_in_the_fit():
    fit = _toy_fit()
    with pytest.raises(KeyError):
        fit.joint_test(["not_a_term"])


def test_h4b_declares_the_interaction_as_its_joint_test():
    """The spec must name the family of terms, so the artifact records the
    same test §3 pre-registers rather than whatever a reader recomputes."""
    import fit_mixed_effects as fme
    assert "h4b" in fme.JOINT_TESTS
    assert "interaction" in fme.JOINT_TESTS["h4b"].lower() or \
           ":" in fme.JOINT_TESTS["h4b"]
