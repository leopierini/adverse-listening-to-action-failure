"""Absorption metrics — the single definition of every absorption number.

Pure functions only: no file I/O, no printing, no argparse. Imported by
analyze_absorption.py, bootstrap_absorption.py and compare_clean_vs_degraded.py
so one definition change propagates everywhere.

Four comparators are computed side by side (docs/design/METRICS.md §8):

  gain_paired   HEADLINE. Restricted to items the same (slurp_id, asr, llm)
                triple got right on clean audio and whose degraded row has
                WER > 0. Every retained item starts from outcome = 1, so the
                router ceiling is 1.0 by construction and (1 - WER) is an
                honest comparator.
  gain_ceiling  ROBUSTNESS. Full cell, but the baseline is scaled by the
                router's measured ceiling: absorption - ceiling * (1 - WER|err).
  gain_legacy   AUDIT TRAIL ONLY, never reported as a result. Reproduces the
                defective 2026-07-24 formula so the correction stays auditable.

  gain_calibrated
                The CALIBRATED null (§14, decided 2026-08-14). The three above
                all *assume* the comparator `1 - WER`; this one fits it. The
                router's own WER -> outcome curve is estimated on the CLEAN
                condition per (asr, llm) and used to predict every degraded
                row. Identified from data instead of assumed, and the proper
                form of the mediation analysis in §8.

Design: docs/superpowers/specs/2026-08-12-absorption-metric-design.md
"""

import math
from collections import defaultdict

import numpy as np

# n below this in the paired subset -> flagged exploratory (tracker convention)
UNDERPOWERED_N = 30

# Where the clean calibration runs thin. Measured 2026-08-14: at capped
# WER >= 0.8 the clean condition holds only 8-31 rows per ASR, while 6.5-22.2%
# of degraded rows sit there. Above this line the logistic is interpolating,
# not identifying, so every curve and every cell reports its own support (§14).
HIGH_WER_SUPPORT = 0.8

# Newton-Raphson settings for the 2-parameter logistic. Hand-rolled so that a
# 2000-replicate bootstrap that refits the curve inside every replicate costs
# seconds; pinned against statsmodels' MLE in tests/test_calibrated_null.py.
_NEWTON_STEPS = 100
_NEWTON_TOL = 1e-10

# Outcome columns these functions accept. pf is a float recall score, not a
# binary success flag, so it is deliberately excluded.
VALID_OUTCOMES = ("ees", "tsa", "ees_strict")


def cell_key(row):
    """(asr, degradation, snr_db, llm) — the analysis cell a row belongs to."""
    return (
        (row.get("asr_model") or "?").split("/")[-1],
        row.get("degradation") or "clean",
        row.get("snr_db"),
        row.get("llm_model") or "?",
    )


def item_key(row):
    """(slurp_id, asr, llm) — identifies the same utterance across conditions."""
    return (
        row.get("slurp_id"),
        (row.get("asr_model") or "?").split("/")[-1],
        row.get("llm_model") or "?",
    )


def cap_wer(wer):
    """Clamp per-utterance WER to [0, 1].

    ASR repetition collapse (small Whisper variants stuck in 'of all of all…')
    produces WER of 30-50x. Uncapped, those poison the mean and can drive a
    naive baseline negative, which inflates the gain. Capping treats them as
    'completely wrong', which is what they are for absorption purposes.
    """
    return min(wer, 1.0)


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def cell_stats(rows, outcome="ees"):
    """Cell-level absorption statistics. `rows` is one analysis cell.

    Returns a dict of floats/ints/None. Never raises on empty or degenerate
    input — callers get None and decide how to render it.
    """
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            "outcome must be one of %s, got %r" % (VALID_OUTCOMES, outcome)
        )

    with_wer = [r for r in rows if r.get("wer") is not None]
    scored = [r for r in with_wer if r.get(outcome) is not None]
    errored = [r for r in scored if r["wer"] > 0]
    perfect = [r for r in scored if r["wer"] == 0]

    capped = sorted(cap_wer(r["wer"]) for r in with_wer)
    mean_wer_all = _mean(capped)
    # Even-length cells need the mean of the two central values, not the upper
    # one (audit 2026-08-14).
    _n = len(capped)
    median_wer = (capped[_n // 2] if _n % 2
                  else (capped[_n // 2 - 1] + capped[_n // 2]) / 2.0) if capped else None
    mean_wer_err = _mean(cap_wer(r["wer"]) for r in errored)

    mean_outcome = _mean(float(r[outcome]) for r in scored)
    absorption = _mean(float(r[outcome]) for r in errored)
    ceiling = _mean(float(r[outcome]) for r in perfect)

    # Legacy: baseline built from mean WER over ALL rows while absorption is
    # conditioned on WER > 0 — the mismatched denominator being preserved here
    # on purpose, for audit only.
    naive_legacy = 1.0 - mean_wer_all if mean_wer_all is not None else None
    naive_ceiling = (
        ceiling * (1.0 - mean_wer_err)
        if (ceiling is not None and mean_wer_err is not None)
        else None
    )

    return {
        "n": len(rows),
        "n_scored": len(scored),
        "n_err": len(errored),
        "n_ceiling": len(perfect),
        "n_collapse": sum(1 for r in with_wer if r["wer"] > 1.0),
        "mean_wer_all": mean_wer_all,
        "median_wer": median_wer,
        "mean_wer_err": mean_wer_err,
        "mean_outcome": mean_outcome,
        "ceiling": ceiling,
        "absorption": absorption,
        "naive_legacy": naive_legacy,
        "naive_ceiling": naive_ceiling,
        "gain_legacy": (
            absorption - naive_legacy
            if (absorption is not None and naive_legacy is not None)
            else None
        ),
        "gain_ceiling": (
            absorption - naive_ceiling
            if (absorption is not None and naive_ceiling is not None)
            else None
        ),
    }


def build_clean_index(all_rows):
    """Map (slurp_id, asr, llm) -> that item's clean-condition row.

    Pass every row you loaded, not just the clean ones — the function filters.
    First row wins if a key repeats; the scored files hold one clean row per
    (item, asr, llm), so a repeat means two clean files were passed at once.
    """
    index = {}
    for r in all_rows:
        if (r.get("degradation") or "clean") != "clean":
            continue
        key = item_key(r)
        if key not in index:
            index[key] = r
    return index


def paired_rows(cell_rows, clean_index, outcome="ees"):
    """The Modo A subset: items the system got right on clean audio whose
    degraded transcript actually contains an ASR error.

    Returns (kept_rows, n_dropped_no_anchor). Clean rows are never kept — a
    clean cell is the reference for the comparison, not a result of it.
    """
    kept = []
    dropped = 0
    for r in cell_rows:
        if (r.get("degradation") or "clean") == "clean":
            continue
        wer = r.get("wer")
        if wer is None or wer <= 0 or r.get(outcome) is None:
            continue
        anchor = clean_index.get(item_key(r))
        if anchor is None:
            dropped += 1
            continue
        if anchor.get(outcome) != 1:
            continue
        kept.append(r)
    return kept, dropped


def paired_stats(cell_rows, clean_index, outcome="ees"):
    """Headline absorption statistics for one cell.

    Every retained item scored outcome = 1 on clean audio, so the router's
    ceiling is 1.0 by construction and (1 - WER) is a fair comparator.
    """
    kept, dropped = paired_rows(cell_rows, clean_index, outcome)
    mean_wer_paired = _mean(cap_wer(r["wer"]) for r in kept)
    retention = _mean(float(r[outcome]) for r in kept)
    naive_paired = 1.0 - mean_wer_paired if mean_wer_paired is not None else None
    return {
        "n_paired": len(kept),
        "n_dropped_no_anchor": dropped,
        "mean_wer_paired": mean_wer_paired,
        "retention": retention,
        "naive_paired": naive_paired,
        "gain_paired": (
            retention - naive_paired
            if (retention is not None and naive_paired is not None)
            else None
        ),
        "underpowered": len(kept) < UNDERPOWERED_N,
    }


def classify_pair(clean_row, deg_row, outcome="ees"):
    """Classify one item's clean -> degraded transition.

    PROPAGATED  clean ok, degraded failed, and the transcript has an error
    ABSORBED    clean ok, degraded ok, transcript has an error  (the good case)
    RECOVERED   clean failed, degraded ok
    SAME_FAIL   failed in both
    SAME_OK     ok in both, no transcript error
    NEW_ERROR   clean ok, degraded failed, but no transcript error at all —
                impossible at temperature 0, so its count is a determinism check
    """
    c = clean_row.get(outcome)
    d = deg_row.get(outcome)
    wer = deg_row.get("wer") or 0
    if c == 1 and d == 0:
        return "PROPAGATED" if wer > 0 else "NEW_ERROR"
    if c == 1 and d == 1:
        return "ABSORBED" if wer > 0 else "SAME_OK"
    if c == 0 and d == 1:
        return "RECOVERED"
    if c == 0 and d == 0:
        return "SAME_FAIL"
    return "OTHER"


def resample_by_slurp_id(rows, rng):
    """Bootstrap resample keyed on the utterance, not the row.

    Within one cell this is numerically identical to resampling rows, because
    a cell holds one row per utterance — it is NOT a source of wider intervals.
    It is the right unit because the metric is item-level, and it stays correct
    if a cell ever holds several rows per item.

    Callers must re-apply the paired filter inside each replicate rather than
    resampling a pre-computed subset: which items clear the clean-anchor
    condition is itself uncertain, and freezing it discards that uncertainty.
    """
    by_id = defaultdict(list)
    for r in rows:
        by_id[r.get("slurp_id")].append(r)
    ids = list(by_id)
    if not ids:
        return []
    drawn = rng.choice(len(ids), size=len(ids), replace=True)
    sample = []
    for i in drawn:
        sample.extend(by_id[ids[i]])
    return sample


def paired_stats_incremental(cell_rows, clean_index, outcome="ees"):
    """The INCREMENTAL null: charge each degraded row only the WER it ADDED
    over its own clean anchor, `1 - max(WER_deg - WER_clean, 0)`.

    Added 2026-08-14. Until then this null existed only in prose: §13 quoted
    -0.106 from it and `verify_tracker_claims.py` guarded only the absolute
    endpoint, so half the reported range was unprotected — the exact failure
    that script exists to prevent.

    It is NOT a bound. The calibrated null of Step 7 lands outside the range
    these two span (§14): the level is not identified, and the set of
    defensible nulls is not closed. Report retention and the ordering.
    """
    kept, dropped = paired_rows(cell_rows, clean_index, outcome)
    incs = []
    for r in kept:
        anchor = clean_index.get(item_key(r))
        w_clean = cap_wer(anchor["wer"]) if anchor and anchor.get("wer") is not None else 0.0
        incs.append(max(cap_wer(r["wer"]) - w_clean, 0.0))
    mean_inc = _mean(incs)
    retention = _mean(float(r[outcome]) for r in kept)
    naive = 1.0 - mean_inc if mean_inc is not None else None
    return {
        "n_paired": len(kept),
        "mean_incremental_wer": mean_inc,
        "retention": retention,
        "naive_incremental": naive,
        "gain_incremental": (retention - naive
                             if (retention is not None and naive is not None) else None),
    }


# ---------------------------------------------------------------------------
# The calibrated null (§14, decided 2026-08-14)
#
# gain_paired / gain_ceiling / gain_legacy all assume the comparator 1 - WER.
# This one estimates it: fit P(outcome = 1) = sigmoid(a + b * cap(WER)) on the
# CLEAN rows of each (asr, llm), then predict every DEGRADED row from that
# curve. gain = observed - predicted.
#
# Two decisions inside it are the result, not implementation detail:
#
#   * the population is the FULL degraded sample. Applying the marginal clean
#     curve to the paired subset instead — which is selected on `clean ees == 1`
#     and whose measured ceiling is 0.993, against a marginal P(EES | WER=0) of
#     ~0.66 — gives +0.303 and reintroduces the mismatched-population defect the
#     2026-08-12 rewrite existed to remove. The paired variant below therefore
#     renormalises each row by its own anchor's prediction.
#   * calibrate on clean only. Pooling across conditions would fit the null
#     partly on the cells being tested.
#
# A curve that is not identified REFUSES. There is no fallback: the level of
# this gain is the disputed quantity in the whole absorption result, and a
# separated or rank-deficient fit that silently returns numbers is exactly how
# the falsified bracket [-0.106, +0.093] came to be quoted for two weeks.
# ---------------------------------------------------------------------------


def calibration_key(row):
    """(asr, llm) — the population one calibration curve is fit on."""
    return (
        (row.get("asr_model") or "?").split("/")[-1],
        row.get("llm_model") or "?",
    )


def _sigmoid(z):
    # Split at zero so neither branch can overflow exp().
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _refused_curve(reason, n_clean=0, n_high=0):
    return {
        "a": None,
        "b": None,
        "n_clean": n_clean,
        "n_clean_at_high_wer": n_high,
        "p_at_zero": None,
        "refused": reason,
    }


def _fit_logistic(xs, ys):
    """MLE for P(y = 1) = sigmoid(a + b*x) by Newton-Raphson on a 2x2 system.

    Returns (a, b) or None if the step cannot be taken — a singular Hessian or
    a non-finite iterate both mean the data does not identify the curve.
    """
    a, b = 0.0, 0.0
    for _ in range(_NEWTON_STEPS):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            p = _sigmoid(a + b * x)
            r = y - p
            w = p * (1.0 - p)
            g0 += r
            g1 += r * x
            h00 += w
            h01 += w * x
            h11 += w * x * x
        det = h00 * h11 - h01 * h01
        if not math.isfinite(det) or abs(det) < 1e-12:
            return None
        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        a += da
        b += db
        if not (math.isfinite(a) and math.isfinite(b)):
            return None
        if max(abs(da), abs(db)) < _NEWTON_TOL:
            return a, b
    return None


def fit_clean_calibration(all_rows, outcome="ees"):
    """One clean-condition calibration curve per (asr, llm).

    Pass every row you loaded, not just the clean ones — the function filters,
    the way build_clean_index does. Degraded rows never enter a fit.

    Returns {(asr, llm): curve}, where curve carries `a`, `b`, `p_at_zero`, the
    clean sample size, how much of it sits above HIGH_WER_SUPPORT, and
    `refused` — None when the curve is usable, otherwise the reason it is not.
    """
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            "outcome must be one of %s, got %r" % (VALID_OUTCOMES, outcome)
        )

    by_key = defaultdict(list)
    for r in all_rows:
        if (r.get("degradation") or "clean") != "clean":
            continue
        if r.get("wer") is None or r.get(outcome) is None:
            continue
        by_key[calibration_key(r)].append(r)

    curves = {}
    for key, rows in by_key.items():
        xs = [cap_wer(r["wer"]) for r in rows]
        ys = [float(r[outcome]) for r in rows]
        n = len(rows)
        n_high = sum(1 for x in xs if x >= HIGH_WER_SUPPORT)

        if n < 2:
            curves[key] = _refused_curve(
                "only %d clean row(s)" % n, n, n_high)
            continue
        if len(set(ys)) < 2:
            curves[key] = _refused_curve(
                "the clean outcome does not vary (all %g)" % ys[0], n, n_high)
            continue
        if max(xs) - min(xs) < 1e-12:
            curves[key] = _refused_curve(
                "clean WER does not vary (all %.3f)" % xs[0], n, n_high)
            continue
        # Perfect separation in one dimension: a threshold splits the outcomes
        # exactly, so the MLE runs to infinity and any printed slope is an
        # artefact of where the optimiser was stopped.
        x_one = [x for x, y in zip(xs, ys) if y == 1.0]
        x_zero = [x for x, y in zip(xs, ys) if y == 0.0]
        if max(x_zero) < min(x_one) or max(x_one) < min(x_zero):
            curves[key] = _refused_curve(
                "the clean outcome separates perfectly on WER", n, n_high)
            continue

        fit = _fit_logistic(xs, ys)
        if fit is None:
            curves[key] = _refused_curve(
                "the logistic fit did not converge", n, n_high)
            continue

        a, b = fit
        curves[key] = {
            "a": a,
            "b": b,
            "n_clean": n,
            "n_clean_at_high_wer": n_high,
            "p_at_zero": _sigmoid(a),
            "refused": None,
        }
    return curves


def predict_calibrated(curve, wer):
    """P(outcome = 1) the clean curve predicts at this WER, or None if the
    curve refused. WER is capped exactly as in every other baseline."""
    if curve is None or curve.get("refused") is not None:
        return None
    return _sigmoid(curve["a"] + curve["b"] * cap_wer(wer))


def calibrated_stats(cell_rows, curves, outcome="ees"):
    """Calibrated-null statistics for one cell, over the FULL degraded sample.

    `curves` is the mapping returned by fit_clean_calibration. Clean rows are
    skipped: a clean cell is the reference the curve was fit on, not a result.
    """
    scored = [
        r for r in cell_rows
        if (r.get("degradation") or "clean") != "clean"
        and r.get("wer") is not None
        and r.get(outcome) is not None
    ]
    empty = {
        "n_calibrated": 0,
        "mean_observed": None,
        "mean_predicted": None,
        "gain_calibrated": None,
        "share_at_high_wer": None,
        "n_clean_at_high_wer": None,
        "refused": None,
    }
    if not scored:
        return empty

    keys = set(calibration_key(r) for r in scored)
    if len(keys) > 1:
        empty["n_calibrated"] = len(scored)
        empty["refused"] = "the cell mixes %d (asr, llm) populations" % len(keys)
        return empty

    key = keys.pop()
    curve = curves.get(key)
    if curve is None:
        empty["n_calibrated"] = len(scored)
        empty["refused"] = "no clean calibration for %s" % (key,)
        return empty
    if curve["refused"] is not None:
        empty["n_calibrated"] = len(scored)
        empty["refused"] = curve["refused"]
        return empty

    predicted = [predict_calibrated(curve, r["wer"]) for r in scored]
    observed = [float(r[outcome]) for r in scored]
    mean_observed = _mean(observed)
    mean_predicted = _mean(predicted)
    return {
        "n_calibrated": len(scored),
        "mean_observed": mean_observed,
        "mean_predicted": mean_predicted,
        "gain_calibrated": mean_observed - mean_predicted,
        "share_at_high_wer": _mean(
            1.0 if cap_wer(r["wer"]) >= HIGH_WER_SUPPORT else 0.0 for r in scored
        ),
        "n_clean_at_high_wer": curve["n_clean_at_high_wer"],
        "refused": None,
    }


def calibrated_stats_paired(cell_rows, clean_index, curves, outcome="ees"):
    """The calibrated null on the PAIRED subset, renormalised.

    Reported for continuity with gain_paired, never as the headline. On this
    subset the comparator is each item's own conditional survival probability,
    P(w_deg) / P(w_clean) capped at 1 — NOT the marginal P(w_deg). Applying the
    marginal curve here is what produced +0.303 on 2026-08-14, 3.3x outside the
    bracket the tracker then claimed, and it is a population error rather than
    a large effect.
    """
    kept, dropped = paired_rows(cell_rows, clean_index, outcome)
    empty = {
        "n_paired": len(kept),
        "n_dropped_no_anchor": dropped,
        "retention": None,
        "mean_predicted": None,
        "gain_calibrated_paired": None,
        "refused": None,
    }
    if not kept:
        return empty

    keys = set(calibration_key(r) for r in kept)
    if len(keys) > 1:
        empty["refused"] = "the cell mixes %d (asr, llm) populations" % len(keys)
        return empty

    curve = curves.get(keys.pop())
    if curve is None or curve["refused"] is not None:
        empty["refused"] = (
            curve["refused"] if curve is not None else "no clean calibration"
        )
        return empty

    ratios = []
    for r in kept:
        anchor = clean_index.get(item_key(r))
        w_clean = (
            anchor["wer"]
            if anchor is not None and anchor.get("wer") is not None
            else 0.0
        )
        p_clean = predict_calibrated(curve, w_clean)
        p_deg = predict_calibrated(curve, r["wer"])
        ratios.append(1.0 if p_clean <= 0 else min(p_deg / p_clean, 1.0))

    retention = _mean(float(r[outcome]) for r in kept)
    mean_predicted = _mean(ratios)
    empty.update({
        "retention": retention,
        "mean_predicted": mean_predicted,
        "gain_calibrated_paired": retention - mean_predicted,
    })
    return empty


# ---------------------------------------------------------------------------
# §8 metric 5 — paired cascade-vs-omni absorption (H4)
# ---------------------------------------------------------------------------

def _distinct(rows, getter):
    return sorted(set(getter(r) for r in rows), key=lambda v: (v is None, str(v)))


def _refuse_mixed_cell(cascade_rows, omni_rows):
    """Refuse inputs that span more than one analysis cell or one arm.

    Pairing keys on slurp_id, which repeats at every SNR and in every arm, so
    a mixed call does not fail — it silently keeps one row per id and reports
    a number computed across conditions. A wrong number that prints is worse
    than a refusal, and this project has already paid for that class of bug
    twice (the 200-row pilot entering the sweep counts; the router tag glob
    that scored 19 of 38 files and looked successful).
    """
    checks = (
        ("degradation", cascade_rows + omni_rows,
         lambda r: r.get("degradation") or "clean"),
        ("snr_db", cascade_rows + omni_rows, lambda r: r.get("snr_db")),
        ("cascade llm_model", cascade_rows, lambda r: r.get("llm_model")),
        ("cascade asr_model", cascade_rows, lambda r: r.get("asr_model")),
        ("omni llm_model", omni_rows, lambda r: r.get("llm_model")),
    )
    for label, rows, getter in checks:
        if not rows:
            continue
        values = _distinct(rows, getter)
        if len(values) > 1:
            raise ValueError(
                "pathway_paired_stats needs exactly one cell per call, but "
                "%s takes %d values here: %r. Group the rows first."
                % (label, len(values), values)
            )


def _index_one_per_id(rows, label):
    """slurp_id -> row, refusing a repeat.

    A cell holds exactly one row per utterance. A repeat means two runs landed
    in the same file — `route_omni.py` and `route_transcripts.py` both open
    their output in APPEND mode, so re-running one without deleting the old
    file is the normal way to produce this. Indexed silently, the last row
    wins and the cell still looks complete.
    """
    index = {}
    for r in rows:
        sid = r.get("slurp_id")
        if sid in index:
            raise ValueError(
                "duplicated clip in the %s rows: slurp_id %r appears more than "
                "once in one cell. Both drivers append, so this is usually a "
                "re-run written on top of an earlier file." % (label, sid)
            )
        index[sid] = r
    return index


def _pair_qualifies(cascade_row, omni_row, outcome):
    """The membership rule for §8 metric 5, defined exactly once.

    Both the point estimate and every bootstrap replicate call this, so the
    subset cannot drift between the number and its interval.
    """
    wer = cascade_row.get("wer")
    if wer is None or wer <= 0:
        return False
    # An unscored clip on either side leaves the pair, exactly as `cell_stats`
    # drops a null outcome from its own denominator.
    return (cascade_row.get(outcome) is not None
            and omni_row.get(outcome) is not None)


def pathway_paired_stats(cascade_rows, omni_rows, outcome="ees"):
    """§8 metric 5: does the omni pathway recover what the transcript lost?

    On the clips where the REFERENCE CASCADE had WER > 0,
    `mean(outcome_omni) - mean(outcome_cascade)`.

      positive -> the omni pathway recovers actions the cascade lost to
                  transcription error
      negative -> the text stage was PROTECTIVE

    The subset is selected on the CASCADE's WER because the omni side has no
    WER to select on — it has no transcript stage at all (every row of the
    real omni sweeps carries `wer = None`). Clips present on only one side are
    dropped: this is a paired statistic, and an unpaired clip would move the
    two means over different populations.
    """
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            "outcome must be one of %s, got %r" % (VALID_OUTCOMES, outcome)
        )
    _refuse_mixed_cell(cascade_rows, omni_rows)

    cas_by_id = _index_one_per_id(cascade_rows, "cascade")
    omni_by_id = _index_one_per_id(omni_rows, "omni")

    pairs = []
    for c in cas_by_id.values():
        o = omni_by_id.get(c.get("slurp_id"))
        if o is not None and _pair_qualifies(c, o, outcome):
            pairs.append((c, o))

    cascade_rate = _mean(c[outcome] for c, _ in pairs)
    omni_rate = _mean(o[outcome] for _, o in pairs)
    return {
        "n_pairs": len(pairs),
        "cascade_rate": cascade_rate,
        "omni_rate": omni_rate,
        "gain_pathway": (None if not pairs else omni_rate - cascade_rate),
        "underpowered": len(pairs) < UNDERPOWERED_N,
    }


def pathway_paired_ci(cascade_rows, omni_rows, rng, outcome="ees",
                      n_boot=2000):
    """Percentile bootstrap for §8 metric 5. Resampling unit: the CLIP.

    Two things this gets right on purpose, both of them house rules that were
    written after a real defect:

    1. **The clip's two rows move together.** The cascade row and the omni row
       for one utterance are drawn as a unit. Resampling the two sides
       independently would destroy the pairing and report an interval for a
       quantity nobody computed.
    2. **The qualifying filter is re-applied inside every replicate** (§8's
       bootstrap convention). Which clips have `WER > 0` is itself measured,
       so freezing the subset once and resampling that would condition on an
       estimate and report a band narrower than the truth.
    """
    point = pathway_paired_stats(cascade_rows, omni_rows, outcome)

    cas_by_id = _index_one_per_id(cascade_rows, "cascade")
    omni_by_id = _index_one_per_id(omni_rows, "omni")
    ids = [i for i in cas_by_id if i in omni_by_id]
    ids.sort(key=lambda v: (v is None, str(v)))

    replicates = []
    for _ in range(n_boot) if ids else ():
        drawn = rng.choice(len(ids), size=len(ids), replace=True)
        cas_vals, omni_vals = [], []
        for i in drawn:
            c = cas_by_id[ids[i]]
            o = omni_by_id[ids[i]]
            if _pair_qualifies(c, o, outcome):
                cas_vals.append(c[outcome])
                omni_vals.append(o[outcome])
        if cas_vals:
            replicates.append(_mean(omni_vals) - _mean(cas_vals))

    def q(p):
        return float(np.percentile(replicates, p)) if replicates else None

    return {
        "n_pairs": point["n_pairs"],
        "gain_pathway": point["gain_pathway"],
        "cascade_rate": point["cascade_rate"],
        "omni_rate": point["omni_rate"],
        "underpowered": point["underpowered"],
        "ci_lo": q(2.5),
        "ci_median": q(50),
        "ci_hi": q(97.5),
        "n_replicates": len(replicates),
        "replicates": replicates,
    }


def pool_pathway_cells(cells):
    """Pool per-cell §8-metric-5 results into one figure (e.g. per degradation).

    Weighted by `n_pairs`, never by averaging the per-cell gains: the WER > 0
    subset shrinks as the audio gets cleaner, so the three SNR cells of one
    degradation differ in size by an order of magnitude and an unweighted mean
    hands the smallest cell an equal vote.

    Cells with no pairs contribute nothing; if none has pairs the result is
    Nones, matching the rest of this module.
    """
    usable = [c for c in cells
              if c.get("n_pairs") and c.get("cascade_rate") is not None
              and c.get("omni_rate") is not None]
    n = sum(c["n_pairs"] for c in usable)
    if not n:
        return {"n_pairs": 0, "cascade_rate": None, "omni_rate": None,
                "gain_pathway": None, "underpowered": True, "n_cells": 0}

    cascade_rate = sum(c["cascade_rate"] * c["n_pairs"] for c in usable) / n
    omni_rate = sum(c["omni_rate"] * c["n_pairs"] for c in usable) / n
    return {
        "n_pairs": n,
        "n_cells": len(usable),
        "cascade_rate": cascade_rate,
        "omni_rate": omni_rate,
        "gain_pathway": omni_rate - cascade_rate,
        "underpowered": n < UNDERPOWERED_N,
    }
