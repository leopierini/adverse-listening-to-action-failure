"""Tests for the pure helpers behind the Step 7 figures.

Nothing here draws. The figures are only as trustworthy as the parsing between
the saved artifacts and the marks, and a silently wrong parse is the failure
mode that matters: a forest plot with a coefficient off by a column still looks
like a forest plot. Each helper is pinned against the real artifact shapes.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

import make_thesis_figures as mtf  # noqa: E402


# --- parse_gee_summary -------------------------------------------------------

GEE_SUMMARY = """\
========================================================================
MODEL      : h4b
FORMULA    : outcome ~ C(pathway)*C(degradation) + snr_20
ESTIMATOR  : GEE (Binomial family, Exchangeable working correlation, robust SEs)
CLUSTER    : slurp_id  (416 clusters, 31616 observations)
DESIGN     : 17 columns, rank 17, smallest singular value 6.589e+00
========================================================================
                               GEE Regression Results
===================================================================================
Dep. Variable:                     outcome   No. Observations:                31616
Model:                                 GEE   No. clusters:                      416
Method:                        Generalized   Min. cluster size:                  76
Family:                           Binomial   Mean cluster size:                76.0
Date:                     Thu, 27 Aug 2026   Scale:                           1.000
Covariance type:                    robust   Time:                         22:52:40
===============================================================================================
                                  coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------------------------------
Intercept                       0.5995      0.104      5.774      0.000       0.396       0.803
C(pathway)[T.omni]              0.0526      0.079      0.663      0.507      -0.103       0.208
C(degradation)[T.babble]       -0.9399      0.073    -12.838      0.000      -1.083      -0.796
snr_20                          0.6524      0.039     16.793      0.000       0.576       0.728
==============================================================================
Skew:                         -0.1621   Kurtosis:                      -1.8385
==============================================================================
"""


def test_parse_gee_summary_reads_every_coefficient_row():
    terms = mtf.parse_gee_summary(GEE_SUMMARY)
    assert list(terms) == ["Intercept", "C(pathway)[T.omni]",
                          "C(degradation)[T.babble]", "snr_20"]


def test_parse_gee_summary_keeps_all_six_columns_in_order():
    t = mtf.parse_gee_summary(GEE_SUMMARY)["C(degradation)[T.babble]"]
    assert t == {"coef": -0.9399, "se": 0.073, "z": -12.838, "p": 0.0,
                 "lo": -1.083, "hi": -0.796}


def test_parse_gee_summary_ignores_the_header_block():
    """`No. Observations: 31616` and `Scale: 1.000` are numbers too.

    They are the reason the parser demands six numeric columns rather than
    "some numbers after a label" — a looser rule reads the header as a
    coefficient and every downstream row shifts.
    """
    terms = mtf.parse_gee_summary(GEE_SUMMARY)
    for junk in ("Dep. Variable:", "No. clusters:", "Skew:", "MODEL",
                 "DESIGN", "Date:"):
        assert not any(k.startswith(junk) for k in terms)


def test_parse_gee_summary_refuses_a_block_with_no_coefficients():
    with pytest.raises(ValueError):
        mtf.parse_gee_summary("MODEL : h4b\nno table here\n")


def test_parse_gee_summary_on_the_real_h4b_artifact():
    path = os.path.join("results", "analysis", "fits", "h4b_ees.txt")
    if not os.path.exists(path):
        pytest.skip("h4b fit artifact not present")
    with open(path) as f:
        terms = mtf.parse_gee_summary(f.read())
    # The six interactions the joint Wald test is taken over must all be found,
    # under the `clean` anchor §3 and §8 pre-register.
    for deg in mtf.DEG_ORDER:
        assert mtf.H4B_INTERACTION % deg in terms
    assert mtf.H4B_MAIN in terms


# --- excludes_zero / verdict_counts ------------------------------------------

@pytest.mark.parametrize("lo,hi,expected", [
    (0.01, 0.05, True),
    (-0.05, -0.01, True),
    (-0.01, 0.05, False),
    (0.0, 0.05, False),      # an endpoint at zero does not exclude zero
    (-0.05, 0.0, False),
    (0.0, 0.0, False),
])
def test_excludes_zero(lo, hi, expected):
    assert mtf.excludes_zero(lo, hi) is expected


def test_verdict_counts_splits_into_three_disjoint_classes():
    lo = [0.01, 0.02, -0.10, -0.03, -0.01]
    hi = [0.05, 0.06, -0.02, 0.04, 0.01]
    pos, cov, neg = mtf.verdict_counts(lo, hi)
    assert (pos, cov, neg) == (2, 2, 1)
    assert pos + cov + neg == len(lo)


def test_verdict_counts_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mtf.verdict_counts([0.0, 1.0], [1.0])


def test_verdict_counts_reproduces_the_headline_on_the_real_artifacts():
    """0 of 54 positive under the calibrated null, on every arm.

    This is the claim the whole absorption story rests on, so the figure code
    that draws it is checked against the artifacts rather than against a copy
    of the number in prose.
    """
    for _key, point_csv, ci_csv, _label, llm in mtf.ARMS:
        if not os.path.exists(mtf._path(ci_csv)):
            pytest.skip("absorption artifacts not present")
        d = mtf.load_arm(point_csv, ci_csv, llm)
        pos, cov, neg = mtf.verdict_counts(d["gain_calibrated_lo"],
                                           d["gain_calibrated_hi"])
        assert pos == 0
        assert pos + cov + neg == 54
        # ...and the assumed comparator finds one in 30-34, which is the
        # contrast fig3 exists to draw.
        pos_paired, _cov, _neg = mtf.verdict_counts(d["gain_paired_lo"],
                                                    d["gain_paired_hi"])
        assert 30 <= pos_paired <= 34


# --- waterfall_segments ------------------------------------------------------

def test_waterfall_segments_chains_start_to_end():
    assert mtf.waterfall_segments([-0.2, -0.05]) == [(0.0, -0.2),
                                                     (-0.2, -0.25)]


def test_waterfall_segments_handles_components_of_opposite_sign():
    """Four of the eighteen mediation cells have a positive direct effect.

    A stacked bar would draw those as if they added to the damage; the
    waterfall walks back toward zero, which is what the decomposition says.
    """
    segments = mtf.waterfall_segments([-0.0286, +0.0125])
    assert segments[0] == (0.0, -0.0286)
    assert segments[1][0] == pytest.approx(-0.0286)
    assert segments[1][1] == pytest.approx(-0.0161)


def test_waterfall_segments_last_end_is_the_total():
    components = [-0.2242, -0.0471]
    assert mtf.waterfall_segments(components)[-1][1] == pytest.approx(
        sum(components))


def test_waterfall_segments_of_nothing_is_nothing():
    assert mtf.waterfall_segments([]) == []


# --- symmetric_limit ---------------------------------------------------------

def test_symmetric_limit_is_driven_by_the_largest_magnitude():
    assert mtf.symmetric_limit([-0.09, 0.02], pad=1.0) == pytest.approx(0.09)


def test_symmetric_limit_ignores_nan():
    assert mtf.symmetric_limit([np.nan, -0.05, 0.01],
                               pad=1.0) == pytest.approx(0.05)


def test_symmetric_limit_never_returns_zero():
    """An all-zero panel must still get a usable colour range."""
    assert mtf.symmetric_limit([0.0, 0.0]) > 0


def test_symmetric_limit_refuses_an_empty_input():
    with pytest.raises(ValueError):
        mtf.symmetric_limit([np.nan, np.nan])


# --- degradation_group / condition_columns -----------------------------------

def test_degradation_group_matches_the_three_and_three_split():
    assert [mtf.degradation_group(d) for d in ("clipping", "codec", "farfield")] \
        == ["preserving"] * 3
    assert [mtf.degradation_group(d) for d in ("babble", "noise", "reverb")] \
        == ["destroying"] * 3
    assert mtf.degradation_group("clean") == "clean"


def test_degradation_group_refuses_an_unknown_label():
    with pytest.raises(ValueError):
        mtf.degradation_group("compression")


def test_condition_columns_is_the_54_cell_layout_per_asr():
    cols = mtf.condition_columns()
    assert len(cols) == 18
    assert cols[0] == ("clipping", 20.0)
    assert cols[-1] == ("reverb", 0.0)
    # SNR runs 20 -> 0 inside each degradation: severity increases rightward.
    assert [s for _d, s in cols[:3]] == [20.0, 10.0, 0.0]


# --- parse_mediation_table ---------------------------------------------------

MEDIATION = """\
degradation    snr |      ACME       ADE     TOTAL     %MED | ACME 95% CI (cluster)    ADE 95% CI (cluster)
------------------------------------------------------------------------------------------------------
babble          20 |   -0.0257   -0.0032   -0.0289        - | [-0.0357, -0.0172] n=100 [-0.0183, +0.0158] n=100
reverb           0 |   -0.2242   -0.0471   -0.2713    82.6% | [-0.2546, -0.1966] n=100 [-0.0836, -0.0084] n=100
"""


def test_parse_mediation_table_reads_both_effects_and_both_bands():
    rows = mtf.parse_mediation_table(MEDIATION)
    assert len(rows) == 2
    assert rows[1] == {"degradation": "reverb", "snr_db": 0.0,
                       "acme": -0.2242, "ade": -0.0471, "total": -0.2713,
                       "acme_lo": -0.2546, "acme_hi": -0.1966,
                       "ade_lo": -0.0836, "ade_hi": -0.0084}


def test_parse_mediation_table_preserves_the_additive_identity():
    """`total = ACME + ADE` holds exactly in the estimator, so it must survive
    the parse — a column read into the wrong field would break it."""
    for r in mtf.parse_mediation_table(MEDIATION):
        assert r["acme"] + r["ade"] == pytest.approx(r["total"], abs=1e-4)


def test_parse_mediation_table_skips_the_header_and_the_rule():
    rows = mtf.parse_mediation_table(MEDIATION)
    assert all(r["degradation"] in mtf.DEG_LABEL for r in rows)


def test_parse_mediation_table_refuses_a_file_with_no_rows():
    with pytest.raises(ValueError):
        mtf.parse_mediation_table("degradation    snr |\n----\n")


def test_parse_mediation_table_on_the_real_artifact_covers_every_cell():
    path = os.path.join("results", "analysis", "mediation_per_snr.txt")
    if not os.path.exists(path):
        pytest.skip("mediation artifact not present")
    with open(path) as f:
        rows = mtf.parse_mediation_table(f.read())
    assert len(rows) == 18
    keys = {(r["degradation"], r["snr_db"]) for r in rows}
    assert keys == set(mtf.condition_columns())
    for r in rows:
        # `total = ACME + ADE` holds exactly in the estimator, but the artifact
        # prints all three rounded to four decimals, so the printed sum can sit
        # up to 1.5e-4 from the printed total (0.5e-4 of rounding per number).
        # noise @ 10 dB is the cell that uses that budget: -0.0925 + -0.0156
        # sums to -0.1081 against a printed total of -0.1082.
        assert r["acme"] + r["ade"] == pytest.approx(r["total"], abs=2e-4)


# --- p-value formatting ------------------------------------------------------

def test_p_label_never_prints_a_zero_p_value():
    """statsmodels rounds the saved p-column to three decimals.

    A parsed 0.000 means "below what the artifact records"; printing `0.00e+00`
    would put a precision on the page that the artifact does not carry.
    """
    assert mtf._p_label(0.0) == "p < 0.001"


def test_p_label_uses_scientific_notation_below_a_thousandth():
    assert mtf._p_label(2.9453639e-05) == "p = 2.95e-05"
    # 6.58e-04 printed with four decimals would read 0.0007 and lose two
    # significant figures against the value §13 records.
    assert mtf._p_label(0.000658127) == "p = 6.58e-04"
    assert mtf._p_label(0.001) == "p = 0.0010"
    assert mtf._p_label(0.171) == "p = 0.1710"


# --- the arm table -----------------------------------------------------------

def test_load_arm_refuses_a_file_belonging_to_a_different_router():
    """The 9B band file is named for the method, not the arm.

    `absorption_ci_calibrated.csv` says nothing in its name about which router
    it holds, and the three arms' files are otherwise interchangeable in shape.
    The `llm` assertion is the only thing standing between a mislabelled panel
    and a figure that looks perfectly fine.
    """
    _key, point_csv, ci_csv, _label, _llm = mtf.ARMS[0]
    if not os.path.exists(mtf._path(ci_csv)):
        pytest.skip("absorption artifacts not present")
    with pytest.raises(ValueError):
        mtf.load_arm(point_csv, ci_csv, "Qwen/Qwen3.5-27B-GPTQ-Int4")


def test_arms_name_three_distinct_routers():
    llms = [llm for _k, _p, _c, _l, llm in mtf.ARMS]
    assert len(set(llms)) == 3
