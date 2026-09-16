"""Frame and coding layer for the confirmatory regressions (docs/design/METRICS.md §8).

Pure functions only: no file I/O, no printing, no argparse, no fitting. The
driver is fit_mixed_effects.py; keeping the coding rules here means one change
propagates to every hypothesis and the rules are unit-testable without a fit.

Written 2026-08-18 to replace the coding inside the pre-audit
fit_mixed_effects.py, which could not test 4 of the 5 hypotheses.
"""

from typing import Optional, Tuple

# --- coding rules -----------------------------------------------------------


def error_flags(error_categories) -> Tuple[int, int]:
    """(has_critical_error, has_function_error) for one utterance.

    §8: H1 is the comparison of these two coefficients, so they are two
    separate flags, not one. `wer` enters the same model as a covariate, so
    the flags measure WHERE the error landed, not how corrupted the row is.
    """
    cats = error_categories or []
    critical = any(c.get("slot_position") == "critical" for c in cats)
    function = any(c.get("slot_position") == "function" for c in cats)
    return int(critical), int(function)


def snr_dummies(snr_db: Optional[float]) -> Tuple[int, int]:
    """(snr_20, snr_10) — the nested-within-degradation coding §8 pins.

    Both are 0 on clean rows AND on 0 dB rows: 0 dB is the within-degraded
    reference. A plain 4-level {clean,20,10,0} factor would be perfectly
    collinear with (degradation == 'clean') — the sum of its three dummies is
    the 'any degradation' indicator — and the fit is then rank-deficient on
    exactly the rows H4b is anchored to.
    """
    if snr_db is None:
        return 0, 0
    snr = int(snr_db)
    return int(snr == 20), int(snr == 10)


def _short(name):
    """Basename of a HF repo id: 'mlx-community/whisper-base' -> 'whisper-base'."""
    return (name or "?").split("/")[-1]


# --- model specifications ---------------------------------------------------

# Binary outcomes plus `pf`, which is a recall proportion in [0,1] and is fitted
# as a fractional logit (binomial quasi-likelihood + robust SEs), not as a
# Bernoulli variable.
VALID_OUTCOMES = ("ees", "tsa", "ees_strict", "pf")


class ModelSpec(object):
    """One confirmatory model from §8: which rows it eats, what it predicts."""

    def __init__(self, name, default_outcome, terms, hypothesis, required,
                 row_filter=None, needs_wer=True):
        self.name = name
        self.default_outcome = default_outcome
        self.terms = terms
        # Terms without which the fit still runs but answers a different
        # question. Reporting such a fit under the hypothesis's name is the
        # failure mode; refusing it is correct.
        self.required = required
        self.hypothesis = hypothesis
        self.row_filter = row_filter
        self.needs_wer = needs_wer


def _is_cascade(row):
    return (row.get("pathway") or "cascade") == "cascade"


def _has_critical_substitution(row):
    """H3's population: the value was mis-heard, not deleted.

    A critical-slot DELETION leaves no (ref, hyp) pair to score, so
    annotate_errors.py leaves phonetic_distance null on those rows. Requiring
    both conditions keeps the filter honest if that ever changes.
    """
    if not _is_cascade(row):
        return False
    if row.get("phonetic_distance") is None:
        return False
    return any(c.get("type") == "substitute" and c.get("slot_position") == "critical"
               for c in (row.get("error_categories") or []))


# The reference cascade H4b compares the omni pathway against (§8): the best
# ASR into the family's large router. Basenames, matched after stripping the
# repo path. ASR id read from the collected rows themselves; router ids from
# the tracker table verified 2026-08-14 by scripts/verify_model_specs.py
# (exit 0). No omni or large-router rows exist yet, so the driver must report
# an empty h4b frame loudly rather than fitting nothing.
REFERENCE_ASR = "whisper-large-v3-turbo"
REFERENCE_LARGE_ROUTERS = (
    "Qwen3.5-27B-GPTQ-Int4",
    "gemma-4-31B-it-qat-w4a16-ct",
)


def make_h4b_row_filter(reference_asr=REFERENCE_ASR,
                        reference_routers=REFERENCE_LARGE_ROUTERS):
    """Keep every omni row plus ONLY the reference cascade rows.

    Letting Whisper Base or the 9B router into the comparison would pit the
    omni model against a weaker cascade and inflate the pathway effect — the
    opposite of the capacity-calibration caveat this design already carries.
    """
    routers = set(reference_routers)

    def keep(row):
        if (row.get("pathway") or "cascade") == "omni":
            return True
        return (_short(row.get("asr_model")) == reference_asr
                and _short(row.get("llm_model")) in routers)

    return keep


MODELS = {
    "h1": ModelSpec(
        name="h1",
        default_outcome="tsa",
        terms=["wer", "has_critical_error", "has_function_error",
               "C(asr_model)", "C(degradation)", "snr_20", "snr_10",
               "C(llm_model)"],
        hypothesis=("H1 — an error on a critical slot hurts the intent less "
                    "than an error on a function word (compare the two flags)"),
        required=["has_critical_error", "has_function_error"],
        row_filter=_is_cascade,
    ),
    "h2": ModelSpec(
        name="h2",
        default_outcome="ees",
        terms=["wer", "C(asr_model)", "C(degradation)", "snr_20", "snr_10",
               "C(llm_model)"],
        hypothesis="H2a/H2b — does a bigger / different-family router absorb more?",
        required=["C(llm_model)"],
        row_filter=_is_cascade,
    ),
    "h3": ModelSpec(
        name="h3",
        default_outcome="pf",
        terms=["phonetic_distance", "C(asr_model)", "C(degradation)",
               "snr_20", "snr_10", "C(llm_model)"],
        hypothesis=("H3 — the more phonetically distant the mis-hearing, the "
                    "less of the parameter survives"),
        required=["phonetic_distance"],
        row_filter=_has_critical_substitution,
    ),
    "h4b": ModelSpec(
        name="h4b",
        default_outcome="ees",
        # §8 (METRICS.md): "`clean` = reference level of degradation". Pinning
        # it is not cosmetic — §3's answer to the capacity confound is that
        # "the clean condition anchors each pathway's baseline capability, and
        # the confirmatory test is the interaction, which cancels the constant
        # capacity gap" (DESIGN.md). Left at patsy's alphabetical default the
        # reference is `babble`, so every interaction is a contrast against the
        # hardest condition instead of the anchor. Measured 2026-08-27 on the
        # real fit: omni x reverb reads -0.1553, p = 0.0045 against babble and
        # -0.0984, p = 0.171 against clean. Same fit, opposite verdict on a
        # hypothesis-relevant term. The joint Wald over the six interactions is
        # invariant either way (chi2 = 29.1127, df 6, p = 5.79e-05), which is
        # why the confirmatory answer is that test and not a single coefficient.
        terms=["C(pathway, Treatment(reference='cascade'))"
               "*C(degradation, Treatment(reference='clean'))",
               "C(model_family)", "snr_20", "snr_10"],
        hypothesis=("H4b — does the omni pathway absorb more than the cascade, "
                    "and does that depend on the kind of corruption?"),
        required=["C(pathway, Treatment(reference='cascade'))"
                  "*C(degradation, Treatment(reference='clean'))"],
        row_filter=make_h4b_row_filter(),
        needs_wer=False,
    ),
}


# --- frame construction -----------------------------------------------------


def build_frame(rows, model, outcome=None, clamp_wer=True,
                drop_hallucinations=False):
    """(pandas.DataFrame, diagnostics dict) for one confirmatory model.

    The diagnostics are returned rather than printed so the caller decides how
    to render them and so every drop is auditable: a row that silently vanishes
    from a regression is how a fit ends up answering a different question than
    the one asked.
    """
    import pandas as pd

    if model not in MODELS:
        raise ValueError("unknown model %r; expected one of %s"
                         % (model, sorted(MODELS)))
    spec = MODELS[model]
    outcome = outcome or spec.default_outcome
    if outcome not in VALID_OUTCOMES:
        raise ValueError("outcome must be one of %s, got %r"
                         % (VALID_OUTCOMES, outcome))

    diag = {
        "model": model, "outcome": outcome, "rows_in": len(rows),
        "dropped_row_filter": 0, "dropped_null_outcome": 0,
        "dropped_null_wer": 0, "wer_clamped": 0, "dropped_hallucination": 0,
    }
    records = []
    for r in rows:
        if spec.row_filter is not None and not spec.row_filter(r):
            diag["dropped_row_filter"] += 1
            continue
        # Null means "not applicable" (omni rows have no transcript), never True.
        if drop_hallucinations and r.get("full_hallucination") is True:
            diag["dropped_hallucination"] += 1
            continue
        y = r.get(outcome)
        if y is None:
            diag["dropped_null_outcome"] += 1
            continue
        wer = r.get("wer")
        if spec.needs_wer and wer is None:
            diag["dropped_null_wer"] += 1
            continue
        if wer is not None and clamp_wer and wer > 1.0:
            diag["wer_clamped"] += 1
            wer = 1.0

        critical, function = error_flags(r.get("error_categories"))
        snr_20, snr_10 = snr_dummies(r.get("snr_db"))
        records.append({
            "slurp_id": r.get("slurp_id"),
            "outcome": float(y),
            "pathway": r.get("pathway") or "cascade",
            "model_family": r.get("model_family") or "?",
            "asr_model": _short(r.get("asr_model")),
            "llm_model": _short(r.get("llm_model")),
            "degradation": r.get("degradation") or "clean",
            "snr_20": snr_20,
            "snr_10": snr_10,
            "wer": float(wer) if wer is not None else None,
            "has_critical_error": critical,
            "has_function_error": function,
            "phonetic_distance": r.get("phonetic_distance"),
            "full_hallucination": bool(r.get("full_hallucination")),
            "intent": r.get("gold_intent") or "?",
        })

    df = pd.DataFrame.from_records(records)
    diag["rows_out"] = len(df)
    return df, diag


# --- formula assembly -------------------------------------------------------


def _term_varies(df, term):
    """Does this term carry information in THIS frame?

    A factor with one level, or a numeric column that is constant, is a copy of
    the intercept. Leaving it in makes the design singular for a reason that has
    nothing to do with the hypothesis being tested.
    """
    import re

    names = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", term)
    cols = [n for n in names if n in df.columns]
    if not cols:
        return False
    return all(df[c].nunique(dropna=False) > 1 for c in cols)


def formula(df, model, outcome_col="outcome"):
    """The patsy formula for `model` on `df`, keeping only terms that vary."""
    if model not in MODELS:
        raise ValueError("unknown model %r; expected one of %s"
                         % (model, sorted(MODELS)))
    kept = [t for t in MODELS[model].terms if _term_varies(df, t)]
    if not kept:
        raise ValueError(
            "no term in the %s specification varies in this frame (%d rows) — "
            "there is nothing to estimate" % (model, len(df))
        )
    return "%s ~ %s" % (outcome_col, " + ".join(kept))


def design_matrix(formula_str, df):
    """The right-hand-side design matrix patsy will hand the estimator.

    Exposed so rank can be checked BEFORE a fit: statsmodels will happily
    return coefficients and standard errors for a rank-deficient design.
    """
    import patsy

    rhs = formula_str.split("~", 1)[1].strip()
    return patsy.dmatrix(rhs, df, return_type="dataframe").values


def rank_report(formula_str, df):
    """Rank diagnostics for a design. `deficient` True means the fit is not
    identified and its coefficients must not be interpreted."""
    import numpy as np

    X = design_matrix(formula_str, df)
    sv = np.linalg.svd(X, compute_uv=False)
    rank = int(np.linalg.matrix_rank(X))
    return {
        "n_rows": int(X.shape[0]),
        "n_cols": int(X.shape[1]),
        "rank": rank,
        "smallest_singular_value": float(sv[-1]) if len(sv) else 0.0,
        "deficient": rank < X.shape[1],
    }


def missing_required(df, model):
    """Required terms of `model` that carry no information in this frame.

    A fit missing one of these still converges and still prints a table — it
    just no longer tests the hypothesis whose name is on it.
    """
    if model not in MODELS:
        raise ValueError("unknown model %r; expected one of %s"
                         % (model, sorted(MODELS)))
    return [t for t in MODELS[model].required if not _term_varies(df, t)]


# --- design coverage --------------------------------------------------------

# The experimental cell a row sits in, for coverage purposes: which audio it
# came from. (snr_20, snr_10) both zero means clean or 0 dB, which `degradation`
# already separates.
CELL_COLUMNS = ("asr_model", "degradation", "snr_20", "snr_10")

# Columns naming which SYSTEM produced a row, as opposed to which CONDITION it
# was produced under. Only these are coverage-checked: a router run on 2 of 57
# conditions is a defect, whereas 'clean' appearing in one SNR combination
# while every degradation appears in three is the design itself.
ARM_COLUMNS = ("llm_model", "asr_model", "pathway", "model_family")


def level_coverage(df, column, cell_columns=CELL_COLUMNS):
    """{level: number of design cells that level was observed in}."""
    cells = [c for c in cell_columns if c in df.columns and c != column]
    return {
        level: int(sub.drop_duplicates(subset=cells).shape[0])
        for level, sub in df.groupby(column)
    }


def thin_levels(df, model, min_coverage=0.5, cell_columns=CELL_COLUMNS,
                required_only=True):
    """Levels of a required factor that were run on too little of the design.

    A router present on 2 conditions and one present on 57 are not comparable:
    the router coefficient absorbs whatever is special about the conditions the
    thin level happens to cover. The fit converges and prints a number anyway.

    Returns a list of dicts; empty means every factor checked is comparable.
    `required_only=False` also checks nuisance covariates, whose thin levels
    are worth a warning rather than a refusal.
    """
    if model not in MODELS:
        raise ValueError("unknown model %r; expected one of %s"
                         % (model, sorted(MODELS)))
    spec = MODELS[model]
    terms = spec.required if required_only else spec.terms
    flagged = []
    for term in terms:
        for column in [c for c in ARM_COLUMNS if c in df.columns and c in term]:
            if df[column].dtype.kind not in "OSUb":
                continue
            coverage = level_coverage(df, column, cell_columns)
            if len(coverage) < 2:
                continue
            fullest = max(coverage.values())
            for level, n_cells in sorted(coverage.items()):
                share = n_cells / float(fullest) if fullest else 0.0
                if share < min_coverage:
                    flagged.append({
                        "term": term, "column": column, "level": level,
                        "n_cells": n_cells, "n_cells_fullest": fullest,
                        "coverage": share,
                    })
    return flagged
