"""Unit tests for scripts/mediation_analysis.py.

§8 asks the question "of the failures, how much is the ASR's fault and how much
the router's?" and cites Imai et al. (2010). The file used to answer it with a
Baron-Kenny product: an OLS coefficient in WER units multiplied by a logistic
coefficient in log-odds, with `total = direct + indirect` asserted on top.

That identity does not hold for a logistic outcome. A logistic coefficient
changes when a covariate is added even with no confounding at all
(non-collapsibility), so "direct + indirect" is not the total effect of
anything. The potential-outcomes decomposition Imai et al. define is on the
PROBABILITY scale, where the identity does hold — and that is what these tests
pin.

conftest.py puts scripts/ on sys.path, so the module imports by bare name.
"""
import numpy as np
import pytest

import mediation_analysis as med


def synth(n_items=400, direct=0.0, mediated=True, seed=0):
    """Rows shaped like *_scored.jsonl, with a KNOWN causal structure.

    degradation -> WER  (the a-path, present only when `mediated`)
    WER         -> EES  (the b-path, always present)
    degradation -> EES  (the direct path, strength `direct`)
    """
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_items):
        for asr in ("whisper-large-v3-turbo", "parakeet-tdt-0.6b-v3"):
            for treated in (0, 1):
                base = 0.10 if asr.startswith("whisper") else 0.14
                shift = 0.35 if (treated and mediated) else 0.0
                wer = float(np.clip(rng.normal(base + shift, 0.10), 0.0, 1.0))
                eta = 1.0 - 3.0 * wer + direct * treated
                ees = int(rng.uniform() < 1.0 / (1.0 + np.exp(-eta)))
                rows.append({
                    "slurp_id": sid,
                    "asr_model": asr,
                    "degradation": "noise" if treated else "clean",
                    "snr_db": 10 if treated else None,
                    "wer": wer,
                    "ees": ees,
                    "tsa": ees,
                })
    return rows


@pytest.fixture(scope="module")
def frame():
    df = med.build_df(synth(seed=1))
    return med.contrast_frame(df, "noise")


# --------------------------------------------------------------------------
# The frame
# --------------------------------------------------------------------------

def test_contrast_frame_holds_only_clean_and_the_named_degradation():
    rows = synth(n_items=50, seed=2)
    rows += [dict(r, degradation="babble") for r in rows if r["degradation"] == "noise"]
    df = med.build_df(rows)
    f = med.contrast_frame(df, "noise")
    assert set(f["degradation"].unique()) == {"clean", "noise"}


def test_contrast_frame_codes_the_treatment_as_zero_one():
    rows = med.build_df(synth(n_items=50, seed=3))
    f = med.contrast_frame(rows, "noise")
    assert set(f["treated"].unique()) == {0, 1}
    assert (f.loc[f["degradation"] == "clean", "treated"] == 0).all()


def test_refuses_a_contrast_with_no_clean_anchor():
    """Without the clean arm there is no contrast, only one condition."""
    rows = [r for r in synth(n_items=50, seed=4) if r["degradation"] == "noise"]
    f = med.contrast_frame(med.build_df(rows), "noise")
    out = med.mediate(f, seed=1, n_rep=50)
    assert out["refused"] is not None
    assert out["acme"] is None


def test_refuses_a_degradation_that_is_not_in_the_data():
    df = med.build_df(synth(n_items=50, seed=5))
    f = med.contrast_frame(df, "farfield")
    out = med.mediate(f, seed=1, n_rep=50)
    assert out["refused"] is not None


# --------------------------------------------------------------------------
# The decomposition
# --------------------------------------------------------------------------

def test_total_effect_equals_acme_plus_ade(frame):
    """The identity Baron-Kenny broke. On the probability scale it holds."""
    out = med.mediate(frame, seed=7, n_rep=200)
    assert out["refused"] is None
    assert out["total"] == pytest.approx(out["acme"] + out["ade"], abs=1e-9)


def test_effects_are_on_the_probability_scale_not_log_odds(frame):
    out = med.mediate(frame, seed=7, n_rep=200)
    for k in ("acme", "ade", "total"):
        assert -1.0 <= out[k] <= 1.0


def test_full_mediation_leaves_essentially_no_direct_effect():
    """degradation reaches EES only through WER -> ADE ~ 0, ACME ~ total."""
    df = med.build_df(synth(direct=0.0, mediated=True, seed=8))
    out = med.mediate(med.contrast_frame(df, "noise"), seed=9, n_rep=300)
    assert out["ade"] == pytest.approx(0.0, abs=0.03)
    assert out["acme"] == pytest.approx(out["total"], abs=0.03)
    assert out["acme"] < -0.10          # degradation destroys success
    assert out["prop_mediated"] > 0.85


def test_no_mediation_when_the_treatment_does_not_move_the_mediator():
    """WER is the same in both arms, so nothing can be mediated by it."""
    df = med.build_df(synth(direct=-1.5, mediated=False, seed=10))
    out = med.mediate(med.contrast_frame(df, "noise"), seed=11, n_rep=300)
    assert out["acme"] == pytest.approx(0.0, abs=0.03)
    assert out["ade"] < -0.10


def test_a_positive_direct_path_shows_up_as_direct_not_mediated():
    df = med.build_df(synth(direct=+1.5, mediated=True, seed=12))
    out = med.mediate(med.contrast_frame(df, "noise"), seed=13, n_rep=300)
    assert out["ade"] > 0.05
    assert out["acme"] < -0.05


# --------------------------------------------------------------------------
# Clustering — rows are not independent, and the CI must know it
# --------------------------------------------------------------------------

def test_cluster_bootstrap_resamples_utterances_not_rows():
    """Each utterance contributes several rows; drawing rows independently
    would pretend they carry independent information."""
    df = med.build_df(synth(n_items=60, seed=14))
    f = med.contrast_frame(df, "noise")
    rng = np.random.default_rng(3)
    sample = med.resample_utterances(f, rng)
    counts = sample.groupby("slurp_id").size()
    per_item = f.groupby("slurp_id").size().iloc[0]
    assert (counts % per_item == 0).all()
    assert len(sample) == len(f)


def test_cluster_bootstrap_returns_a_band_around_the_point_estimate():
    df = med.build_df(synth(n_items=70, seed=15))
    f = med.contrast_frame(df, "noise")
    point = med.mediate(f, seed=16, n_rep=200)
    band = med.cluster_bootstrap(f, n_boot=25, seed=17, n_rep=80)
    assert band["acme"]["lo"] < point["acme"] < band["acme"]["hi"]
    assert band["n_ok"] >= 20


def test_cluster_bootstrap_is_deterministic_for_a_seed():
    df = med.build_df(synth(n_items=40, seed=18))
    f = med.contrast_frame(df, "noise")
    a = med.cluster_bootstrap(f, n_boot=6, seed=5, n_rep=30)
    b = med.cluster_bootstrap(f, n_boot=6, seed=5, n_rep=30)
    assert a["acme"]["lo"] == pytest.approx(b["acme"]["lo"])
    assert a["acme"]["hi"] == pytest.approx(b["acme"]["hi"])


# --------------------------------------------------------------------------
# The defect this rewrite removes
# --------------------------------------------------------------------------

def test_no_log_odds_product_is_reported_as_an_effect():
    """The old file multiplied an OLS coefficient (WER units) by a logistic one
    (log-odds) and printed the product as the indirect effect. Nothing in the
    module may expose that quantity any more."""
    import inspect
    src = inspect.getsource(med)
    assert "a * b_wer" not in src
    assert "Baron-Kenny" not in src


# --------------------------------------------------------------------------
# %MED is a ratio, and its denominator is allowed to cross zero
# --------------------------------------------------------------------------
# Measured 2026-08-26 on the real sweep: the per-(degradation, SNR) run printed
# `%MED = +177.6%` for reverb at 20 dB, a cell whose total effect is ~0. The
# share mediated is ACME / total; where the total effect is not distinguishable
# from zero, that denominator crosses zero and the ratio is not a share of
# anything. The `abs(total) > 1e-6` guard was far too loose to catch it.

def test_prop_mediated_is_refused_when_the_total_effect_straddles_zero():
    """A treatment that moves neither the mediator nor the outcome."""
    df = med.build_df(synth(n_items=300, direct=0.0, mediated=False, seed=21))
    out = med.mediate(med.contrast_frame(df, "noise"), seed=22, n_rep=300)
    assert out["refused"] is None            # the fit itself is fine
    assert out["total_lo"] < 0.0 < out["total_hi"]
    assert out["prop_mediated"] is None
    assert "total effect" in out["prop_mediated_refused"]


def test_prop_mediated_survives_a_total_effect_that_is_clearly_nonzero():
    """The guard must not swallow the cells the analysis exists to report."""
    df = med.build_df(synth(n_items=300, direct=0.0, mediated=True, seed=23))
    out = med.mediate(med.contrast_frame(df, "noise"), seed=24, n_rep=300)
    assert out["prop_mediated"] is not None
    assert out["prop_mediated_refused"] is None
    assert out["prop_mediated"] > 0.85
    assert out["total_hi"] < 0.0             # the interval excludes zero


def test_the_total_interval_brackets_the_point_estimate(frame):
    out = med.mediate(frame, seed=25, n_rep=300)
    assert out["total_lo"] < out["total"] < out["total_hi"]


# The printed column: the cluster bootstrap is this project's interval of
# record, and it is wider than Mediation's own, so a cell can pass the guard
# inside `mediate` and still be indefensible once utterances are resampled.

def _band(lo, hi):
    return {"acme": {"median": -0.02, "lo": lo, "hi": hi},
            "ade": {"median": 0.0, "lo": lo, "hi": hi},
            "total": {"median": (lo + hi) / 2.0, "lo": lo, "hi": hi},
            "n_ok": 200, "n_boot": 200}


def test_format_prop_mediated_dashes_the_cell_that_printed_177_percent():
    out = {"prop_mediated": 1.776, "prop_mediated_refused": None}
    assert med.format_prop_mediated(out, _band(-0.031, +0.024)).strip() == "—"


def test_format_prop_mediated_prints_the_share_when_the_band_excludes_zero():
    out = {"prop_mediated": 0.877, "prop_mediated_refused": None}
    assert "87.7%" in med.format_prop_mediated(out, _band(-0.152, -0.098))


def test_format_prop_mediated_honours_a_refusal_from_the_fit():
    out = {"prop_mediated": None, "prop_mediated_refused": "the total effect ..."}
    assert med.format_prop_mediated(out, _band(-0.152, -0.098)).strip() == "—"


def test_format_prop_mediated_works_without_a_bootstrap_band():
    """`--bootstrap 0` is legal; the guard inside `mediate` still applies."""
    out = {"prop_mediated": 0.877, "prop_mediated_refused": None}
    assert "87.7%" in med.format_prop_mediated(out, None)


# --------------------------------------------------------------------------
# The band was printed for ACME only
# --------------------------------------------------------------------------
# §13, 2026-08-26: "Every ADE figure above is a point estimate, so `reverb's
# -0.0471 exceeds clipping's -0.0037` is NOT established as a difference."
# cluster_bootstrap already computes the ADE band; only the report withheld it.

def test_cluster_bootstrap_bands_the_direct_effect_too():
    df = med.build_df(synth(n_items=60, direct=-1.5, mediated=False, seed=26))
    f = med.contrast_frame(df, "noise")
    point = med.mediate(f, seed=27, n_rep=200)
    band = med.cluster_bootstrap(f, n_boot=25, seed=28, n_rep=80)
    assert band["ade"]["lo"] < point["ade"] < band["ade"]["hi"]
    assert band["total"]["lo"] < point["total"] < band["total"]["hi"]


def test_ci_text_renders_a_band_with_its_replicate_count():
    text = med.ci_text(_band(-0.152, -0.098), "ade")
    assert "-0.1520" in text and "-0.0980" in text and "n=200" in text


def test_ci_text_dashes_a_band_that_could_not_be_computed():
    empty = {"ade": {"median": None, "lo": None, "hi": None},
             "n_ok": 0, "n_boot": 200}
    assert med.ci_text(empty, "ade").strip() == "—"
    assert med.ci_text(None, "ade").strip() == "—"


def test_the_table_header_prints_one_percent_sign_not_two():
    """`mediation_per_snr.txt` (2026-08-26) carried a literal `95%%`: the label
    is an ARGUMENT to the row format, so doubling the sign never gets undone."""
    h = med.table_header()
    assert "%%" not in h
    assert "ACME 95% CI" in h and "ADE 95% CI" in h


# --------------------------------------------------------------------------
# The saved artifact has to record what produced it
# --------------------------------------------------------------------------
# `mediation_per_snr.txt` recorded "B = 100, seed = 42" and nothing about
# --sim or --boot-sim. Those two set how many simulation draws happen inside
# each of the B fits, so they set both the numbers and the runtime: on
# 2026-08-27 a rerun took over six hours where §13 recorded 53 min 56 s, and
# the difference could not be attributed, because the artifact did not say
# what the first run used. An artifact that cannot be reproduced from its own
# header is not a record.

def test_the_provenance_line_names_every_parameter_that_changes_the_result():
    line = med.provenance(outcome="ees", bootstrap=100, sim=800, boot_sim=200,
                          seed=42, per_snr=True, n_files=38)
    for token in ("ees", "B = 100", "--sim 800", "--boot-sim 200",
                  "seed = 42", "per-SNR", "38"):
        assert token in line, "%r missing from: %s" % (token, line)


def test_the_provenance_line_says_when_boot_sim_defaulted_to_sim():
    line = med.provenance(outcome="ees", bootstrap=100, sim=800, boot_sim=None,
                          seed=42, per_snr=False, n_files=38)
    assert "--boot-sim 800" in line
    assert "pooled" in line
