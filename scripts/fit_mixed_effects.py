"""Fit the confirmatory models for H1, H2a/H2b, H3 and H4b (docs/design/METRICS.md §8).

Rewritten 2026-08-18. The previous version was audited on 2026-08-14 and could
not test 4 of the 5 hypotheses: the outcome was hardcoded to `tsa` with no flag,
`has_critical_error`/`has_function_error` (H1 *is* their comparison) did not
exist, `phonetic_distance` and `PF` (H3) appeared nowhere, `pathway` (H4b) did
not exist, the one model that did run was rank-deficient by construction, and on
a GEE failure it silently substituted `smf.logit` — which ignores clustering and
inflates significance — noting it only inside the saved text.

The estimator §8 pins is GEE, binomial family, exchangeable working correlation
clustered on `slurp_id`, robust (sandwich) SEs. There is no fallback: a fit that
cannot be done that way is reported as a failure, never as a different model.

Row coding lives in model_frames.py so it is unit-testable without a fit.

Usage:
    ./venv/bin/python scripts/fit_mixed_effects.py results/*_scored.jsonl
    ./venv/bin/python scripts/fit_mixed_effects.py --models h1 h2 --outcome ees \\
        results/*_scored.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import model_frames as mf


class EstimatorFailed(Exception):
    """The pinned estimator could not be fitted. Nothing is substituted."""


class RankDeficientDesign(EstimatorFailed):
    """The design matrix is singular, so the coefficients are not identified.

    statsmodels returns coefficients and standard errors for such a design
    without complaining; interpreting them is how a collinear factor turns into
    a published effect.
    """


class FitResult(object):
    """One fitted confirmatory model, carrying its own provenance."""

    def __init__(self, model, formula, result, group_col, n_clusters, rank,
                 fractional=False):
        self.model = model
        self.formula = formula
        self.result = result
        self.group_col = group_col
        self.n_clusters = n_clusters
        self.rank = rank
        self.estimator = "GEE"
        self.cov_struct = "Exchangeable"
        self.family = "Binomial"
        # True when the outcome is a proportion rather than a 0/1 draw: the
        # binomial family is then a quasi-likelihood (fractional logit,
        # Papke & Wooldridge 1996), consistent with robust SEs but not a
        # Bernoulli model. Saying so beats letting a reader assume.
        self.fractional = fractional

    @property
    def n_obs(self):
        return int(self.result.model.exog.shape[0])

    def contrast(self, expression):
        """Wald test of a linear hypothesis, e.g. 'a - b = 0'.

        H1 is not a coefficient, it is the comparison of two: reading the two
        p-values separately answers a different question.
        """
        return self.result.t_test(expression)

    def joint_test(self, terms):
        """Joint Wald test over several coefficients at once.

        §3's operational for H4b is "the `pathway × degradation` interaction is
        significant" — a statement about the whole family of interaction terms,
        not about any single one. Reading one coefficient answers a different
        question, and which coefficient looks significant depends on which
        degradation is the reference level (measured 2026-08-27: omni × reverb
        is −0.1553, p = 0.0045 against `babble` and −0.0984, p = 0.171 against
        the pre-registered `clean`). The joint statistic is invariant to that
        choice, which is why it is the one recorded.
        """
        missing = [t for t in terms if t not in self.result.params.index]
        if missing:
            raise KeyError(
                "these terms are not in the fit: %s. Present: %s"
                % (missing, list(self.result.params.index)))
        w = self.result.wald_test(list(terms), scalar=True)
        return {
            "terms": list(terms),
            "df": len(terms),
            "statistic": float(w.statistic),
            "p_value": float(w.pvalue),
        }

    def as_text(self):
        header = [
            "=" * 72,
            "MODEL      : %s" % self.model,
            "FORMULA    : %s" % self.formula,
            "ESTIMATOR  : %s (%s family, %s working correlation, robust SEs)"
            % (self.estimator, self.family, self.cov_struct),
            "CLUSTER    : %s  (%d clusters, %d observations)"
            % (self.group_col, self.n_clusters, self.n_obs),
            "DESIGN     : %d columns, rank %d, smallest singular value %.3e"
            % (self.rank["n_cols"], self.rank["rank"],
               self.rank["smallest_singular_value"]),
            "NOTE       : no fallback estimator exists; this is the pinned "
            "confirmatory specification (tracker §8).",
            "=" * 72,
            "",
        ]
        if self.fractional:
            header.insert(-2, (
                "OUTCOME    : a proportion in [0,1], not a 0/1 draw — this is a "
                "FRACTIONAL logit\n             (binomial quasi-likelihood with "
                "robust SEs; Papke & Wooldridge 1996)."
            ))
        return "\n".join(header) + self.result.summary().as_text()


def check_standard_errors(bse, exog_names, formula_str):
    """Raise if any robust standard error is not finite.

    A NaN means the sandwich covariance collapsed — near-collinearity the rank
    check could not see, or a degenerate cluster. statsmodels prints the
    coefficient table regardless, and a blank p-value reads as "not
    significant" rather than "not estimated".
    """
    import numpy as np

    values = np.asarray(bse, dtype=float)
    finite = np.isfinite(values)
    if finite.all():
        return None
    bad = [n for n, ok in zip(exog_names, finite) if not ok]
    raise EstimatorFailed(
        "GEE returned a non-finite robust standard error for %s on %s — the "
        "sandwich covariance collapsed (near-collinearity the rank check could "
        "not see, or a degenerate cluster). Do not read the coefficient table: "
        "a blank p-value there means 'not estimated', not 'not significant'."
        % (", ".join(bad), formula_str)
    )


def fit_confirmatory(df, formula_str, group_col="slurp_id", model="(unnamed)"):
    """Fit the pinned confirmatory estimator, or raise. Never substitutes."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    try:
        rank = mf.rank_report(formula_str, df)
    except Exception as exc:
        raise EstimatorFailed(
            "could not build the design for %r: %s" % (formula_str, exc)
        )

    if rank["deficient"]:
        raise RankDeficientDesign(
            "design for %r is rank %d of %d columns (smallest singular value "
            "%.3e) — the coefficients are not identified; drop the collinear "
            "term rather than interpreting the fit"
            % (formula_str, rank["rank"], rank["n_cols"],
               rank["smallest_singular_value"])
        )

    import warnings

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = smf.gee(
                formula_str, group_col, data=df,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            ).fit()
    except Exception as exc:
        raise EstimatorFailed("GEE fit failed for %r: %s" % (formula_str, exc))

    separated = [w for w in caught
                 if "separation" in str(w.message).lower()]
    if separated:
        raise EstimatorFailed(
            "GEE reported perfect separation for %r: %s. The coefficients are "
            "not identified — separation produces a huge estimate with a huge "
            "standard error, which prints as a large effect and means nothing. "
            "Drop or collapse the separating term and refit."
            % (formula_str, str(separated[0].message).strip())
        )

    check_standard_errors(fitted.bse, fitted.model.exog_names, formula_str)

    endog = fitted.model.endog
    fractional = bool(((endog > 0) & (endog < 1)).any())

    return FitResult(
        model=model, formula=formula_str, result=fitted, group_col=group_col,
        n_clusters=int(df[group_col].nunique()), rank=rank,
        fractional=fractional,
    )


# --- orchestration ----------------------------------------------------------

# H1 is not a coefficient, it is the comparison of two (§8): "an error on a
# critical slot damages the intent LESS than an error on a function word".
# Models whose hypothesis is a JOINT statement over a family of terms rather
# than one coefficient. The value is the substring that identifies the family
# in the fitted parameter names. For h4b that is the interaction: §3 asks
# whether "the pathway x degradation interaction is significant", and the
# answer must not depend on which degradation patsy picked as the reference.
JOINT_TESTS = {
    "h4b": ":C(degradation",
}

CONTRASTS = {
    "h1": "has_critical_error - has_function_error = 0",
}


class ModelReport(object):
    """What happened to one hypothesis in one run — including nothing."""

    def __init__(self, model, outcome):
        self.model = model
        self.outcome = outcome
        self.frame = None
        self.diagnostics = None
        self.formula = None
        self.fit = None
        self.contrast = None
        self.contrast_expr = CONTRASTS.get(model)
        self.joint = None
        self.joint_marker = JOINT_TESTS.get(model)
        self.warnings = []
        self.error = None


# A level of a required factor must be observed on at least this share of the
# design cells the fullest level covers, or the factor is confounded with which
# conditions happened to be run. A methodological choice, exposed as a flag.
DEFAULT_MIN_LEVEL_COVERAGE = 0.5


def run_models(rows, models, outcome=None, drop_hallucinations=False,
               group_col="slurp_id",
               min_level_coverage=DEFAULT_MIN_LEVEL_COVERAGE):
    """Fit each requested model, collecting failures instead of raising.

    One hypothesis with no usable rows must not stop the other four from being
    reported: the pre-audit script returned on the first problem and printed
    nothing at all.
    """
    report = {}
    for model in models:
        spec = mf.MODELS[model]
        entry = ModelReport(model, outcome or spec.default_outcome)
        report[model] = entry
        try:
            df, diag = mf.build_frame(
                rows, model, outcome=outcome,
                drop_hallucinations=drop_hallucinations,
            )
            entry.frame, entry.diagnostics = df, diag
            if df.empty:
                raise EstimatorFailed(
                    "no rows survived the %s row filter (%d rows in) — this "
                    "hypothesis has no data yet" % (model, diag["rows_in"])
                )
            absent = mf.missing_required(df, model)
            if absent:
                raise EstimatorFailed(
                    "%s cannot be tested on this data: %s carr%s no variation. "
                    "The fit would still converge and still print a table, but "
                    "it would not be %s. %s"
                    % (model.upper(), ", ".join(absent),
                       "ies" if len(absent) == 1 else "y",
                       model.upper(), spec.hypothesis)
                )
            thin = mf.thin_levels(df, model, min_coverage=min_level_coverage)
            if thin:
                detail = "; ".join(
                    "%s=%r seen in %d of the %d design cells the fullest level "
                    "covers (%.0f%%)"
                    % (t["column"], t["level"], t["n_cells"],
                       t["n_cells_fullest"], 100 * t["coverage"])
                    for t in thin
                )
                raise EstimatorFailed(
                    "%s: a required factor is confounded with condition "
                    "coverage — %s. The fit would converge and print a "
                    "coefficient that absorbs whatever is special about the "
                    "conditions the thin level happens to cover. Restrict the "
                    "input files, or pass --min-level-coverage 0 deliberately."
                    % (model.upper(), detail)
                )
            for t in mf.thin_levels(df, model, min_coverage=min_level_coverage,
                                    required_only=False):
                if any(t["term"] == r["term"] and t["level"] == r["level"]
                       for r in thin):
                    continue
                entry.warnings.append(
                    "nuisance covariate %s=%r was run on only %d of the %d "
                    "design cells the fullest level covers (%.0f%%); its "
                    "coefficient is not comparable and it may absorb condition "
                    "effects"
                    % (t["column"], t["level"], t["n_cells"],
                       t["n_cells_fullest"], 100 * t["coverage"])
                )
            entry.formula = mf.formula(df, model)
            entry.fit = fit_confirmatory(df, entry.formula,
                                         group_col=group_col, model=model)
            if entry.contrast_expr:
                entry.contrast = entry.fit.contrast(entry.contrast_expr)
            if entry.joint_marker:
                terms = [k for k in entry.fit.result.params.index
                         if entry.joint_marker in k]
                if terms:
                    entry.joint = entry.fit.joint_test(terms)
        except Exception as exc:
            entry.error = exc
    return report


# --- CLI --------------------------------------------------------------------


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _contrast_dict(expression, contrast):
    return {
        "expression": expression,
        "effect": float(contrast.effect[0]),
        "std_err": float(contrast.sd[0][0]),
        "statistic": float(contrast.statistic[0][0]),
        "p_value": float(contrast.pvalue),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit the confirmatory models for H1, H2a/H2b, H3 and H4b.",
    )
    parser.add_argument("files", nargs="+", help="scored JSONL files")
    parser.add_argument("--models", nargs="+", default=["h1", "h2", "h3", "h4b"],
                        choices=sorted(mf.MODELS))
    parser.add_argument("--outcome", default=None, choices=list(mf.VALID_OUTCOMES),
                        help="override each model's default outcome (tracker §8 "
                             "defaults: h1=tsa, h2=ees, h3=pf, h4b=ees)")
    parser.add_argument("--drop-hallucinations", action="store_true",
                        help="H4b sensitivity refit: exclude full_hallucination rows")
    parser.add_argument("--min-level-coverage", type=float,
                        default=DEFAULT_MIN_LEVEL_COVERAGE,
                        help="refuse a required factor whose level was run on "
                             "less than this share of the design cells the "
                             "fullest level covers (default %(default)s; 0 "
                             "disables the check)")
    parser.add_argument("--out-dir", default="results/analysis/fits")
    args = parser.parse_args(argv)

    rows = load_rows(args.files)
    print("Loaded %d rows from %d file(s)" % (len(rows), len(args.files)))

    report = run_models(rows, args.models, outcome=args.outcome,
                        drop_hallucinations=args.drop_hallucinations,
                        min_level_coverage=args.min_level_coverage)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {}
    n_fitted = 0

    for model in args.models:
        entry = report[model]
        spec = mf.MODELS[model]
        print("\n" + "=" * 72)
        print("%s — %s" % (model.upper(), spec.hypothesis))
        print("outcome: %s" % entry.outcome)
        if entry.diagnostics:
            d = entry.diagnostics
            print("rows: %d in -> %d out  (row filter %d, null outcome %d, "
                  "null wer %d, hallucination %d; wer clamped %d)"
                  % (d["rows_in"], d["rows_out"], d["dropped_row_filter"],
                     d["dropped_null_outcome"], d["dropped_null_wer"],
                     d["dropped_hallucination"], d["wer_clamped"]))

        record = {
            "outcome": entry.outcome,
            "diagnostics": entry.diagnostics,
            "formula": entry.formula,
            "error": None if entry.error is None else str(entry.error),
            "contrast": None,
            "joint_test": None,
        }

        for warning in entry.warnings:
            print("WARNING: %s" % warning)
        record["warnings"] = entry.warnings

        if entry.fit is None:
            print("NOT FITTED: %s" % entry.error)
            diagnostics[model] = record
            continue

        n_fitted += 1
        text = entry.fit.as_text()
        if entry.joint is not None:
            record["joint_test"] = entry.joint
        if entry.contrast is not None:
            record["contrast"] = _contrast_dict(entry.contrast_expr, entry.contrast)
            c = record["contrast"]
            block = (
                "\n\nCONTRAST (%s)\n"
                "  estimate %+.4f   se %.4f   z %+.3f   p %.4g\n"
                "  H1 predicts a POSITIVE estimate: a critical-slot error should "
                "damage the intent LESS than a function-word error.\n"
                % (c["expression"], c["effect"], c["std_err"], c["statistic"],
                   c["p_value"])
            )
            text += block
            print(block.strip())
        print(text.splitlines()[0] if text else "")

        path = out_dir / ("%s_%s.txt" % (model, entry.outcome))
        path.write_text(text)
        record["summary_file"] = str(path)
        print("saved -> %s" % path)
        diagnostics[model] = record

    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print("\nsaved -> %s" % (out_dir / "diagnostics.json"))

    if n_fitted == 0:
        print("\nNo model could be fitted.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
