"""Unit tests for the CALIBRATED NULL — scripts/absorption_metrics.py.

The null the whole absorption result now rests on (§14, decided 2026-08-14):
fit the router's own WER -> outcome curve on CLEAN rows per (asr, llm), then
predict every degraded row from it. The gain becomes observed - predicted,
identified from data instead of assumed.

Two things these tests exist to stop, both of which already happened once:

  * applying the marginal clean curve to the PAIRED subset (selected on
    `clean ees == 1`, measured ceiling 0.993) gives +0.303 against a marginal
    P(EES | WER = 0) of ~0.66 — Defect (a), the mismatched-population error the
    2026-08-12 rewrite existed to remove. The paired variant must renormalise.
  * a curve that is not identified (no outcome variation, no WER variation)
    silently returning numbers. `fit_mixed_effects.py` was rewritten in August
    for exactly this: refuse, never fall back.

conftest.py puts scripts/ on sys.path, so the module imports by bare name.
"""
import json
import os

import numpy as np
import pytest

import absorption_metrics as am


def row(slurp_id=1, wer=0.0, ees=1, degradation="noise", snr_db=10,
        asr="whisper-large-v3-turbo", llm="qwen9b"):
    """One results row, shaped like the real *_scored.jsonl schema."""
    return {
        "slurp_id": slurp_id,
        "wer": wer,
        "ees": ees,
        "tsa": ees,
        "ees_strict": ees,
        "degradation": degradation,
        "snr_db": snr_db,
        "asr_model": asr,
        "llm_model": llm,
    }


def clean_row(slurp_id=1, wer=0.0, ees=1, asr="whisper-large-v3-turbo",
              llm="qwen9b"):
    return row(slurp_id=slurp_id, wer=wer, ees=ees, degradation="clean",
               snr_db=None, asr=asr, llm=llm)


def logistic_sample(a, b, n, seed, degradation="clean", asr="whisper-large-v3-turbo",
                    llm="qwen9b", first_id=0):
    """n rows whose outcome really is drawn from sigmoid(a + b*wer).

    WER is spread over [0, 1] so the curve is identified across its range —
    the clean condition genuinely reaches WER = 1.00 for all three ASRs
    (feasibility check, 2026-08-12).
    """
    rng = np.random.default_rng(seed)
    wers = rng.uniform(0.0, 1.0, size=n)
    p = 1.0 / (1.0 + np.exp(-(a + b * wers)))
    draws = rng.uniform(0.0, 1.0, size=n) < p
    build = clean_row if degradation == "clean" else row
    out = []
    for i, (w, y) in enumerate(zip(wers, draws)):
        r = build(slurp_id=first_id + i, wer=float(w), ees=int(y), asr=asr, llm=llm)
        if degradation != "clean":
            r["degradation"] = degradation
        out.append(r)
    return out


# --------------------------------------------------------------------------
# The fit itself
# --------------------------------------------------------------------------

def test_fit_recovers_the_parameters_it_was_generated_from():
    rows = logistic_sample(a=1.2, b=-2.5, n=6000, seed=11)
    curve = am.fit_clean_calibration(rows)[("whisper-large-v3-turbo", "qwen9b")]
    assert curve["refused"] is None
    assert curve["a"] == pytest.approx(1.2, abs=0.15)
    assert curve["b"] == pytest.approx(-2.5, abs=0.30)


def test_fit_agrees_with_statsmodels_maximum_likelihood():
    """The estimator is hand-rolled IRLS so a 2000-replicate bootstrap is
    seconds rather than minutes. It has to give statsmodels' answer."""
    sm = pytest.importorskip("statsmodels.api")
    rows = logistic_sample(a=0.7, b=-3.1, n=800, seed=5)
    curve = am.fit_clean_calibration(rows)[("whisper-large-v3-turbo", "qwen9b")]

    y = np.array([r["ees"] for r in rows], dtype=float)
    x = sm.add_constant(np.array([am.cap_wer(r["wer"]) for r in rows]))
    ref = sm.Logit(y, x).fit(disp=0).params

    assert curve["a"] == pytest.approx(float(ref[0]), abs=1e-4)
    assert curve["b"] == pytest.approx(float(ref[1]), abs=1e-4)


def test_a_curve_is_fit_per_asr_and_llm_never_pooled():
    """Pooling would fit the null partly on the cells being tested."""
    rows = (logistic_sample(a=2.0, b=-1.0, n=3000, seed=1,
                            asr="whisper-large-v3-turbo")
            + logistic_sample(a=-1.0, b=+3.0, n=3000, seed=2,
                              asr="parakeet-tdt-0.6b-v3"))
    curves = am.fit_clean_calibration(rows)

    assert set(curves) == {("whisper-large-v3-turbo", "qwen9b"),
                           ("parakeet-tdt-0.6b-v3", "qwen9b")}
    assert curves[("whisper-large-v3-turbo", "qwen9b")]["b"] < 0
    assert curves[("parakeet-tdt-0.6b-v3", "qwen9b")]["b"] > 0


def test_degraded_rows_do_not_enter_the_fit():
    """Calibrate on clean only (§14). Degraded rows that behave nothing like
    the clean population must leave the curve where it was."""
    clean = logistic_sample(a=1.5, b=-3.0, n=2000, seed=3)
    key = ("whisper-large-v3-turbo", "qwen9b")
    before = am.fit_clean_calibration(clean)[key]

    poison = [row(slurp_id=90000 + i, wer=0.9, ees=1) for i in range(2000)]
    after = am.fit_clean_calibration(clean + poison)[key]

    assert after["a"] == pytest.approx(before["a"])
    assert after["b"] == pytest.approx(before["b"])


def test_prediction_caps_wer_like_every_other_baseline():
    """Repetition collapse (WER 40x) is 'completely wrong', not 40x wrong."""
    curve = am.fit_clean_calibration(
        logistic_sample(a=1.0, b=-2.0, n=2000, seed=4))[
            ("whisper-large-v3-turbo", "qwen9b")]
    assert am.predict_calibrated(curve, 40.0) == pytest.approx(
        am.predict_calibrated(curve, 1.0))


# --------------------------------------------------------------------------
# Refusals — a curve that is not identified must not return numbers
# --------------------------------------------------------------------------

def test_fit_refuses_when_every_clean_outcome_is_one():
    """Separation: the MLE runs off to infinity. statsmodels prints a
    coefficient table anyway; a blank/huge estimate reads as a result."""
    rows = [clean_row(slurp_id=i, wer=i / 100.0, ees=1) for i in range(100)]
    curve = am.fit_clean_calibration(rows)[("whisper-large-v3-turbo", "qwen9b")]
    assert curve["refused"] is not None
    assert curve["a"] is None and curve["b"] is None


def test_fit_refuses_when_clean_wer_does_not_vary():
    """No slope is identified from a single x value."""
    rows = [clean_row(slurp_id=i, wer=0.0, ees=i % 2) for i in range(100)]
    curve = am.fit_clean_calibration(rows)[("whisper-large-v3-turbo", "qwen9b")]
    assert curve["refused"] is not None
    assert curve["b"] is None


def test_a_refused_curve_yields_no_gain_rather_than_a_fallback():
    rows = [clean_row(slurp_id=i, wer=i / 100.0, ees=1) for i in range(100)]
    curves = am.fit_clean_calibration(rows)
    s = am.calibrated_stats([row(slurp_id=1, wer=0.5, ees=1)], curves)
    assert s["gain_calibrated"] is None
    assert s["refused"] is not None


def test_a_cell_with_no_curve_at_all_refuses_rather_than_borrowing_one():
    """A router the clean condition never saw must not be scored against
    another router's curve."""
    curves = am.fit_clean_calibration(
        logistic_sample(a=1.0, b=-2.0, n=500, seed=6))
    s = am.calibrated_stats(
        [row(slurp_id=1, wer=0.5, ees=1, llm="qwen27b")], curves)
    assert s["gain_calibrated"] is None
    assert s["refused"] is not None


# --------------------------------------------------------------------------
# The gain
# --------------------------------------------------------------------------

def test_gain_is_zero_when_degraded_rows_obey_their_clean_curve():
    """The definition of a null: a router that does exactly as well under
    degradation as its clean curve predicts has absorbed nothing extra."""
    clean = logistic_sample(a=1.0, b=-2.5, n=6000, seed=7)
    degraded = logistic_sample(a=1.0, b=-2.5, n=6000, seed=8,
                               degradation="noise", first_id=50000)
    curves = am.fit_clean_calibration(clean)
    s = am.calibrated_stats(degraded, curves)
    assert s["gain_calibrated"] == pytest.approx(0.0, abs=0.02)


def test_gain_is_positive_when_the_router_beats_its_clean_curve():
    clean = logistic_sample(a=1.0, b=-4.0, n=4000, seed=9)
    degraded = [row(slurp_id=50000 + i, wer=0.9, ees=1) for i in range(400)]
    curves = am.fit_clean_calibration(clean)
    s = am.calibrated_stats(degraded, curves)
    assert s["gain_calibrated"] > 0.3


def test_gain_uses_every_degraded_row_including_the_perfectly_transcribed_ones():
    """§14 fixed the population: the FULL degraded sample, not `wer > 0`.
    Conditioning on wer > 0 here is what reintroduces Defect (a)."""
    clean = logistic_sample(a=0.5, b=-3.0, n=3000, seed=10)
    curves = am.fit_clean_calibration(clean)
    errored = [row(slurp_id=i, wer=0.6, ees=0) for i in range(100)]
    perfect = [row(slurp_id=200 + i, wer=0.0, ees=1) for i in range(100)]

    s = am.calibrated_stats(errored + perfect, curves)
    assert s["n_calibrated"] == 200
    assert s["mean_observed"] == pytest.approx(0.5)


def test_clean_rows_are_not_their_own_null():
    """A clean cell is the reference for the comparison, not a result of it."""
    clean = logistic_sample(a=1.0, b=-2.0, n=1000, seed=12)
    curves = am.fit_clean_calibration(clean)
    s = am.calibrated_stats(clean, curves)
    assert s["n_calibrated"] == 0
    assert s["gain_calibrated"] is None


# --------------------------------------------------------------------------
# The paired variant — the +0.303 trap
# --------------------------------------------------------------------------

def test_paired_variant_renormalises_by_the_anchors_own_prediction():
    """On the paired subset the comparator is P(w_deg)/P(w_clean), capped at
    1 — not the marginal P(w_deg). One item, anchor at WER 0, degraded at
    WER 1: the ratio is P(1)/P(0), and a surviving item gains 1 - that."""
    clean = logistic_sample(a=1.0, b=-3.0, n=4000, seed=13)
    curves = am.fit_clean_calibration(clean)
    curve = curves[("whisper-large-v3-turbo", "qwen9b")]

    anchor = clean_row(slurp_id=777, wer=0.0, ees=1)
    degraded = row(slurp_id=777, wer=1.0, ees=1)
    index = am.build_clean_index([anchor])

    s = am.calibrated_stats_paired([degraded], index, curves)
    ratio = (am.predict_calibrated(curve, 1.0)
             / am.predict_calibrated(curve, 0.0))
    assert s["n_paired"] == 1
    assert s["mean_predicted"] == pytest.approx(ratio)
    assert s["gain_calibrated_paired"] == pytest.approx(1.0 - ratio)


def test_paired_ratio_is_capped_at_one():
    """A degraded row easier than its own anchor cannot be owed more than
    certainty."""
    clean = logistic_sample(a=-1.0, b=+3.0, n=4000, seed=14)   # rising curve
    curves = am.fit_clean_calibration(clean)
    index = am.build_clean_index([clean_row(slurp_id=777, wer=0.0, ees=1)])
    s = am.calibrated_stats_paired(
        [row(slurp_id=777, wer=1.0, ees=1)], index, curves)
    assert s["mean_predicted"] == pytest.approx(1.0)
    assert s["gain_calibrated_paired"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Thin support — reported, never hidden
# --------------------------------------------------------------------------

def test_the_curve_reports_how_little_clean_data_supports_its_top_end():
    """§14: at WER >= 0.8 the clean calibration rests on 8-31 observations per
    ASR, so the logistic is interpolating, not identifying. That number is
    part of the result."""
    rows = ([clean_row(slurp_id=i, wer=0.1, ees=i % 2) for i in range(200)]
            + [clean_row(slurp_id=500 + i, wer=0.9, ees=0) for i in range(7)])
    curve = am.fit_clean_calibration(rows)[("whisper-large-v3-turbo", "qwen9b")]
    assert curve["n_clean"] == 207
    assert curve["n_clean_at_high_wer"] == 7


def test_the_cell_reports_how_much_of_it_sits_above_that_support():
    clean = logistic_sample(a=1.0, b=-2.0, n=2000, seed=15)
    curves = am.fit_clean_calibration(clean)
    cell = ([row(slurp_id=i, wer=0.2, ees=1) for i in range(90)]
            + [row(slurp_id=500 + i, wer=0.95, ees=0) for i in range(10)])
    s = am.calibrated_stats(cell, curves)
    assert s["share_at_high_wer"] == pytest.approx(0.10)


# --------------------------------------------------------------------------
# Against the real data — the figures §14 recorded on 2026-08-14
# --------------------------------------------------------------------------

SWEEP_GLOB_SENTINEL = "results/cascade_whisper_qwen9b_clean_scored.jsonl"


def _load_sweep_rows(router="qwen9b"):
    """Rows of ONE router arm.

    §14 recorded these figures on 2026-08-14, when `qwen9b` was the only arm
    in results/ and "the sweep" and "the 9B" meant the same set of files. They
    stopped meaning the same thing on 2026-08-27, when the Qwen-27B arm landed
    and `result_sets.sweep_files()` went from 38 files to 76. The recorded
    numbers are a property of the arm they were measured on, so the arm is now
    named rather than assumed: unscoped, this helper returned six calibration
    curves where §14 reports three, and the mean gain averaged two routers.
    """
    import result_sets
    rows = []
    for path in result_sets.sweep_files():
        if router not in str(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


@pytest.mark.skipif(not os.path.exists(SWEEP_GLOB_SENTINEL),
                    reason="results/ not present in this checkout")
def test_clean_curves_reproduce_the_marginal_probabilities_recorded_in_s14():
    """§14, 2026-08-14: the curve fit on clean rows predicts
    P(EES | WER = 0) = 0.657 / 0.683 / 0.690 for the three ASRs **of the
    Qwen-9B arm**. Those three numbers are why applying it raw to a subset with
    a measured ceiling of 0.993 gives +0.303 — they are the evidence for the
    population decision."""
    curves = am.fit_clean_calibration(_load_sweep_rows())
    p0 = sorted(round(c["p_at_zero"], 3) for c in curves.values()
                if c["refused"] is None)
    assert p0 == [0.657, 0.683, 0.690]


@pytest.mark.skipif(not os.path.exists(SWEEP_GLOB_SENTINEL),
                    reason="results/ not present in this checkout")
def test_calibrated_gain_over_the_full_degraded_sample_reproduces_minus_0_016():
    """§14's reported value, and the count that goes with it: -0.016 as the
    mean over the 54 degraded cells of the **Qwen-9B arm**, 40 of them
    negative. 54 = 18 degraded (degradation, SNR) cells x 3 ASRs, one router."""
    from collections import defaultdict

    rows = _load_sweep_rows()
    curves = am.fit_clean_calibration(rows)
    buckets = defaultdict(list)
    for r in rows:
        buckets[am.cell_key(r)].append(r)

    gains = [am.calibrated_stats(cell, curves)["gain_calibrated"]
             for cell in buckets.values()]
    gains = [g for g in gains if g is not None]

    assert len(gains) == 54
    assert sum(1 for g in gains if g < 0) == 40
    assert sum(gains) / len(gains) == pytest.approx(-0.016, abs=0.001)


# --------------------------------------------------------------------------
# The bootstrap band (§14: "report the fitted curve with a bootstrap band")
# --------------------------------------------------------------------------

def _bootstrap_fixture(seed_clean=21, seed_deg=22, n=300):
    """A cell plus the clean population its curve is fit on."""
    clean = logistic_sample(a=1.0, b=-3.0, n=n, seed=seed_clean)
    degraded = logistic_sample(a=1.0, b=-3.0, n=n, seed=seed_deg,
                               degradation="noise", first_id=50000)
    return clean, degraded


def test_bootstrap_refits_the_calibration_inside_every_replicate():
    """The curve is estimated, not known. Fitting it once outside the loop
    would condition on it and report a band narrower than the truth — the same
    error the paired filter's re-application inside each replicate avoids."""
    import bootstrap_absorption as ba

    clean, degraded = _bootstrap_fixture()
    index = am.build_clean_index(clean)
    boot = ba.bootstrap_cell(degraded, index, clean, "ees", 40,
                             np.random.default_rng(3))

    assert len(set(boot["calibration_b"])) > 1
    assert len(set(boot["calibration_a"])) > 1


def test_bootstrap_band_brackets_the_point_estimate():
    import bootstrap_absorption as ba

    clean, degraded = _bootstrap_fixture()
    index = am.build_clean_index(clean)
    point = am.calibrated_stats(
        degraded, am.fit_clean_calibration(clean))["gain_calibrated"]

    boot = ba.bootstrap_cell(degraded, index, clean, "ees", 200,
                             np.random.default_rng(4))
    ci = ba.summarize(boot, "gain_calibrated")
    assert ci["lo"] < point < ci["hi"]


def test_bootstrap_reports_nothing_rather_than_zero_when_the_curve_refused():
    import bootstrap_absorption as ba

    clean = [clean_row(slurp_id=i, wer=i / 100.0, ees=1) for i in range(100)]
    degraded = [row(slurp_id=i, wer=0.5, ees=1) for i in range(100)]
    boot = ba.bootstrap_cell(degraded, am.build_clean_index(clean), clean,
                             "ees", 20, np.random.default_rng(5))
    assert ba.summarize(boot, "gain_calibrated") is None


# --------------------------------------------------------------------------
# The report — a metric nobody can see is not reported
# --------------------------------------------------------------------------

def test_analyze_absorption_writes_the_calibrated_columns(tmp_path):
    """The CSV is what the figures and the Methods tables read. A gain that
    only exists inside a module has not been reported."""
    import csv
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    rows = (logistic_sample(a=1.0, b=-3.0, n=300, seed=31)
            + logistic_sample(a=1.0, b=-3.0, n=300, seed=32,
                              degradation="noise", first_id=50000))
    src = tmp_path / "rows.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out.csv"

    subprocess.run(
        [sys.executable, str(repo / "scripts" / "analyze_absorption.py"),
         str(src), "--out", str(out)],
        cwd=str(repo), check=True, capture_output=True)

    written = list(csv.DictReader(out.open()))
    degraded = [r for r in written if r["degradation"] == "noise"][0]
    curve = am.fit_clean_calibration(rows)[("whisper-large-v3-turbo", "qwen9b")]
    expected = am.calibrated_stats(
        [r for r in rows if r["degradation"] == "noise"],
        {("whisper-large-v3-turbo", "qwen9b"): curve})

    assert float(degraded["gain_calibrated"]) == pytest.approx(
        expected["gain_calibrated"])
    assert int(degraded["n_clean_at_high_wer"]) == curve["n_clean_at_high_wer"]
