"""Phase 6 — mediation analysis: how much of a degradation's damage to EES
travels through the transcript, and how much does not?

§8 asks "of the failures, how much is the ASR's fault vs the router's?" and
cites Imai, Keele & Tingley (2010). This file now implements what it cites.

    a-path   degradation -> WER          (OLS)
    b-path   WER         -> EES          (logistic, given the degradation)
    ACME     the part of the total effect that travels through WER
    ADE      the part that does not — the degradation-specific residual
    total    ACME + ADE, and the identity HOLDS

⚠️ WHY THIS WAS REWRITTEN (2026-08-26). The previous version multiplied the
a-path OLS coefficient (in WER units) by the b-path logistic coefficient (in
log-odds) and printed `direct + indirect` as the total effect. That identity is
false for a logistic outcome: a logistic coefficient changes when a covariate
is added even with no confounding at all (non-collapsibility), so the two
numbers are not on a common scale and their sum is not the effect of anything.
The potential-outcomes decomposition of Imai et al. is defined on the
PROBABILITY scale, where the identity does hold, and it is what
`statsmodels.stats.mediation.Mediation` implements. The old path-B table also
carried unclustered standard errors on rows that share an utterance.

Every effect below is a difference in the probability of end-to-end success,
so -0.20 means "twenty fewer successes per hundred utterances".

CLUSTERING. Each utterance appears under many conditions, so rows are not
independent. `Mediation`'s own interval assumes they are. The interval of
record here is therefore a CLUSTER BOOTSTRAP over `slurp_id`, refitting both
models inside every replicate; the parametric interval is printed beside it.

Usage:
    python mediation_analysis.py results/*_scored.jsonl \\
        --bootstrap 200 --seed 42 --out results/analysis/mediation_summary.txt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.mediation import Mediation

# Outcome columns this analysis accepts (same set as absorption_metrics).
VALID_OUTCOMES = ("ees", "tsa", "ees_strict")


def load_rows(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def build_df(rows, outcome="ees"):
    """Rows -> a frame with wer capped at 1, the outcome as 0/1, and the
    condition columns. Rows missing either variable are dropped."""
    if outcome not in VALID_OUTCOMES:
        raise ValueError("outcome must be one of %s, got %r"
                         % (VALID_OUTCOMES, outcome))
    records = []
    for r in rows:
        wer = r.get("wer")
        if wer is None:
            continue
        y = r.get(outcome)
        if y is None:
            continue
        records.append({
            "slurp_id": r.get("slurp_id"),
            "asr_model": (r.get("asr_model") or "?").split("/")[-1],
            "llm_model": (r.get("llm_model") or "?").split("/")[-1],
            "degradation": r.get("degradation") or "clean",
            "snr_level": (str(int(r["snr_db"])) if r.get("snr_db") is not None
                          else "clean"),
            # Capped exactly as in absorption_metrics.cap_wer: repetition
            # collapse produces WER of 30-50x, which would otherwise dominate
            # the a-path regression on its own.
            "wer": float(min(wer, 1.0)),
            "ees": int(y),
        })
    df = pd.DataFrame.from_records(records)
    return df


def contrast_frame(df, degradation, snr_db=None):
    """The clean arm plus one degraded arm, with `treated` as 0/1.

    `snr_db=None` pools that degradation's three SNRs. ⚠️ SNR is then NOT a
    covariate and cannot be: it is `clean` on every control row, so the SNR
    dummies would sum to the treatment indicator and the design matrix would be
    rank-deficient — the collinearity §8 names explicitly. Pass an `snr_db` to
    ask the question one cell at a time instead.
    """
    if df.empty:
        return df
    keep = df["degradation"] == "clean"
    deg = df["degradation"] == degradation
    if snr_db is not None:
        deg = deg & (df["snr_level"] == str(int(snr_db)))
    out = df[keep | deg].copy()
    out["treated"] = (out["degradation"] != "clean").astype(int)
    return out


def _covariates(frame):
    """Condition terms that actually vary — a constant term breaks the formula,
    and SNR is deliberately never among them (see contrast_frame)."""
    terms = []
    for col in ("asr_model", "llm_model"):
        if frame[col].nunique() > 1:
            terms.append("C(%s)" % col)
    return terms


def _refused(reason, frame=None):
    return {
        "acme": None, "ade": None, "total": None, "prop_mediated": None,
        "prop_mediated_refused": reason,
        "total_lo": None, "total_hi": None,
        "n": 0 if frame is None else len(frame),
        "n_items": 0 if frame is None else frame["slurp_id"].nunique(),
        "n_treated": 0, "n_control": 0,
        "refused": reason,
    }


def mediate(frame, seed=42, n_rep=500):
    """ACME / ADE / total / proportion mediated, on the probability scale.

    Refuses rather than returning numbers when the contrast is not identified.
    There is no fallback: this decomposition is the answer to "whose fault is
    it", and a silently degenerate fit answers it wrongly with confidence.
    """
    if frame is None or len(frame) == 0:
        return _refused("no rows")
    n_treated = int((frame["treated"] == 1).sum())
    n_control = int((frame["treated"] == 0).sum())
    if n_treated == 0 or n_control == 0:
        return _refused(
            "the contrast has only one arm (%d treated, %d control)"
            % (n_treated, n_control), frame)
    if frame["ees"].nunique() < 2:
        return _refused("the outcome does not vary in this contrast", frame)
    if frame["wer"].nunique() < 2:
        return _refused("the mediator does not vary in this contrast", frame)

    cov = _covariates(frame)
    rhs = (" + " + " + ".join(cov)) if cov else ""
    try:
        outcome_model = sm.GLM.from_formula(
            "ees ~ treated + wer" + rhs, data=frame,
            family=sm.families.Binomial())
        mediator_model = sm.OLS.from_formula("wer ~ treated" + rhs, data=frame)
        # Mediation draws its parameter replicates from numpy's global stream.
        np.random.seed(seed)
        res = Mediation(outcome_model, mediator_model, "treated", "wer").fit(
            n_rep=n_rep)
    except Exception as exc:                      # noqa: BLE001 - reported, not hidden
        return _refused("the fit did not converge: %s" % exc, frame)

    acme = float(np.mean(res.ACME_avg))
    ade = float(np.mean(res.ADE_avg))
    total = acme + ade

    # The share mediated is ACME / total. A denominator near zero does not make
    # the ratio large, it makes it meaningless -- measured 2026-08-26, the real
    # sweep printed "+177.6%" for reverb at 20 dB, a cell where the degradation
    # does essentially nothing. `abs(total) > 1e-6` never caught it: the total
    # there is small, not infinitesimal. The test is therefore whether the total
    # effect is distinguishable from zero at all, on the same replicates the
    # point estimate comes from (Mediation resamples the fitted parameters, so
    # ACME_avg + ADE_avg is one total per replicate).
    total_reps = np.asarray(res.ACME_avg) + np.asarray(res.ADE_avg)
    total_lo = float(np.percentile(total_reps, 2.5))
    total_hi = float(np.percentile(total_reps, 97.5))
    if total_lo <= 0.0 <= total_hi:
        prop_mediated = None
        prop_refused = (
            "the total effect is not distinguishable from zero "
            "([%+.4f, %+.4f]), so ACME / total is a ratio over a denominator "
            "that crosses zero" % (total_lo, total_hi))
    elif abs(total) <= 1e-12:
        prop_mediated = None
        prop_refused = "the total effect is exactly zero"
    else:
        prop_mediated = acme / total
        prop_refused = None

    return {
        "acme": acme,
        "ade": ade,
        "total": total,
        "prop_mediated": prop_mediated,
        "prop_mediated_refused": prop_refused,
        "total_lo": total_lo,
        "total_hi": total_hi,
        "acme_lo": float(np.percentile(res.ACME_avg, 2.5)),
        "acme_hi": float(np.percentile(res.ACME_avg, 97.5)),
        "ade_lo": float(np.percentile(res.ADE_avg, 2.5)),
        "ade_hi": float(np.percentile(res.ADE_avg, 97.5)),
        "n": len(frame),
        "n_items": int(frame["slurp_id"].nunique()),
        "n_treated": n_treated,
        "n_control": n_control,
        "refused": None,
    }


def resample_utterances(frame, rng):
    """Draw utterances with replacement, keeping each one's rows together.

    Resampling rows instead would treat the many conditions of one utterance as
    independent observations and shrink every interval.
    """
    ids = frame["slurp_id"].dropna().unique()
    if len(ids) == 0:
        return frame.iloc[0:0]
    groups = {sid: sub for sid, sub in frame.groupby("slurp_id")}
    drawn = rng.choice(len(ids), size=len(ids), replace=True)
    return pd.concat([groups[ids[i]] for i in drawn], ignore_index=True)


def cluster_bootstrap(frame, n_boot=200, seed=42, n_rep=200):
    """Percentile intervals for ACME / ADE / total, clustered on `slurp_id`.

    Replicates whose fit refuses are counted in `n_ok` rather than silently
    dropped, so a band computed from half the replicates cannot look like one
    computed from all of them.
    """
    rng = np.random.default_rng(seed)
    keep = {"acme": [], "ade": [], "total": []}
    for b in range(n_boot):
        sample = resample_utterances(frame, rng)
        out = mediate(sample, seed=seed + b + 1, n_rep=n_rep)
        if out["refused"] is not None:
            continue
        for k in keep:
            keep[k].append(out[k])
    band = {}
    for k, vals in keep.items():
        band[k] = ({"median": float(np.median(vals)),
                    "lo": float(np.percentile(vals, 2.5)),
                    "hi": float(np.percentile(vals, 97.5))}
                   if vals else {"median": None, "lo": None, "hi": None})
    band["n_ok"] = len(keep["acme"])
    band["n_boot"] = n_boot
    return band


def fmt(v):
    return "—" if v is None else "%+.4f" % v


ROW_FORMAT = "%-12s %5s | %9s %9s %9s %8s | %-24s %-24s"


def provenance(outcome, bootstrap, sim, boot_sim, seed, per_snr, n_files):
    """One line naming everything that changes the numbers OR the runtime.

    `--boot-sim` is the one that used to go unrecorded, and it is the
    expensive one: it multiplies the work inside every one of the B bootstrap
    fits. An artifact saying only "B = 100, seed = 42" cannot be reproduced,
    and on 2026-08-27 a rerun that took over six hours could not be compared
    with a recorded 53 min 56 s because of exactly that omission.
    """
    return ("outcome = %s | %s | %d input files | B = %d | --sim %d | "
            "--boot-sim %d | seed = %d"
            % (outcome, "per-SNR (18 contrasts)" if per_snr
               else "pooled (6 contrasts)",
               n_files, bootstrap, sim,
               sim if boot_sim is None else boot_sim, seed))


def table_header():
    """⚠️ The literal `%` here is NOT doubled on purpose: these strings are
    ARGUMENTS to the format, not part of it, so `%%` would print as `%%` —
    which is exactly what `mediation_per_snr.txt` showed on 2026-08-26."""
    return ROW_FORMAT % ("degradation", "snr", "ACME", "ADE", "TOTAL", "%MED",
                         "ACME 95% CI (cluster)", "ADE 95% CI (cluster)")


def ci_text(band, key):
    """One cluster-bootstrap interval, or a dash. `n=` is the number of
    replicates that actually fitted — a band from half the replicates must not
    read like a band from all of them."""
    if not band:
        return "%-24s" % "—"
    entry = band.get(key) or {}
    if entry.get("lo") is None or entry.get("hi") is None:
        return "%-24s" % "—"
    return "%-24s" % ("[%+.4f, %+.4f] n=%d"
                      % (entry["lo"], entry["hi"], band.get("n_ok", 0)))


def format_prop_mediated(out, band=None):
    """The %MED cell: a share, or a dash when it would not be a share.

    Two independent reasons to withhold it, and either is enough:
      * `mediate` already refused it (Mediation's own replicates put zero inside
        the total effect's interval);
      * the cluster bootstrap -- this project's interval of record, because
        rows sharing an utterance are not independent -- does too.

    ⚠️ The two are NOT nested, and neither is reliably the wider one. Measured
    on babble @ 20 dB, 2026-08-27:

        ACME  simulation [-0.0436, -0.0083]   cluster [-0.0357, -0.0172]
        ADE   simulation [-0.0370, +0.0334]   cluster [-0.0183, +0.0158]

    The simulation interval is the wider of the two there, roughly 2x on the
    ADE. They measure different things: the cluster band is the spread of the
    POINT ESTIMATE across resampled utterances, so the simulation noise inside
    each replicate averages out of it, while Mediation's own interval carries
    the parameter uncertainty of the outcome model -- which is large for a
    direct effect estimated with `treated` and `wer` in the same equation.
    Requiring BOTH to exclude zero is therefore a genuine union of two
    uncertainties, not a redundant check on one of them.
    """
    if out.get("prop_mediated") is None:
        return "%8s" % "—"
    if band is not None:
        tot = band.get("total") or {}
        lo, hi = tot.get("lo"), tot.get("hi")
        if lo is not None and hi is not None and lo <= 0.0 <= hi:
            return "%8s" % "—"
    return "%7.1f%%" % (100 * out["prop_mediated"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--outcome", default="ees", choices=list(VALID_OUTCOMES))
    p.add_argument("--bootstrap", type=int, default=200,
                   help="Cluster-bootstrap replicates. 0 skips the band.")
    p.add_argument("--sim", type=int, default=500,
                   help="Simulation draws inside each Mediation fit.")
    p.add_argument("--boot-sim", type=int, default=None,
                   help="Simulation draws inside each BOOTSTRAP replicate. "
                        "Defaults to --sim; lower it to make the band affordable "
                        "(the outer resampling carries most of the uncertainty).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-snr", action="store_true",
                   help="One contrast per (degradation, SNR) cell instead of "
                        "pooling a degradation's three SNRs.")
    p.add_argument("--out", default="results/analysis/mediation_summary.txt")
    args = p.parse_args()

    df = build_df(load_rows(args.files), args.outcome)
    if df.empty:
        print("No usable rows — every row lacked `wer` or the outcome.")
        return
    print("Loaded %d rows (%d utterances), outcome = %s"
          % (len(df), df["slurp_id"].nunique(), args.outcome))
    print("RUN: " + provenance(args.outcome, args.bootstrap, args.sim,
                               args.boot_sim, args.seed, args.per_snr,
                               len(args.files)))
    if not (df["degradation"] == "clean").any():
        print("⚠️  No clean rows — every contrast is against nothing and will "
              "refuse. Pass the clean files too.")

    degradations = sorted(d for d in df["degradation"].unique() if d != "clean")
    contrasts = []
    for d in degradations:
        if args.per_snr:
            for s in sorted(df.loc[df["degradation"] == d, "snr_level"].unique(),
                            key=lambda x: -int(x)):
                contrasts.append((d, int(s)))
        else:
            contrasts.append((d, None))

    print("\nEffects are differences in P(success), so -0.20 = twenty fewer "
          "successes per hundred utterances.")
    print("ACME travels through the transcript; ADE does not; total = ACME + ADE.\n")
    header = table_header()
    print(header)
    print("-" * len(header))

    lines = []
    for deg, snr in contrasts:
        frame = contrast_frame(df, deg, snr)
        out = mediate(frame, seed=args.seed, n_rep=args.sim)
        if out["refused"] is not None:
            line = ("%-12s %5s | refused: %s"
                    % (deg, "—" if snr is None else snr, out["refused"]))
            print(line)
            lines.append(line)
            continue
        band = (cluster_bootstrap(frame, args.bootstrap, args.seed,
                                  args.boot_sim or args.sim)
                if args.bootstrap else None)
        pm = format_prop_mediated(out, band)
        line = (ROW_FORMAT
                % (deg, "—" if snr is None else snr,
                   fmt(out["acme"]), fmt(out["ade"]), fmt(out["total"]), pm,
                   ci_text(band, "acme"), ci_text(band, "ade")))
        print(line)
        lines.append(line)

    print("\nACME = average causal mediation effect (through WER).")
    print("ADE  = average direct effect (everything else about the degradation).")
    print("Both are potential-outcomes quantities on the probability scale, per")
    print("Imai, Keele & Tingley (2010), via statsmodels.stats.mediation.")
    print("Both CIs are cluster bootstraps over slurp_id — rows sharing an utterance")
    print("are not independent, and Mediation's own interval assumes they are.")
    print("%MED is ACME / total, printed only where the total effect's 95% interval")
    print("excludes zero. A dash is not a missing number: it means the degradation")
    print("did too little for `how much of it went through the transcript` to have")
    print("an answer — the ratio's denominator crosses zero.")

    text = ("Mediation analysis (Phase 6) — how much of a degradation's damage\n"
            "to %s travels through the transcript?\n" % args.outcome
            + "=" * 78 + "\n\n"
            + "N = %d rows, %d utterances\n" % (len(df), df["slurp_id"].nunique())
            + "Estimator: Imai, Keele & Tingley (2010) potential-outcomes\n"
              "decomposition via statsmodels.stats.mediation.Mediation.\n"
              "Effects are differences in P(success). CI = cluster bootstrap\n"
              "over slurp_id.\n"
            + "RUN: " + provenance(args.outcome, args.bootstrap, args.sim,
                                   args.boot_sim, args.seed, args.per_snr,
                                   len(args.files)) + "\n"
            + "%MED = ACME / total, printed only where the total effect's 95%\n"
              "interval excludes zero. A dash means the total effect is not\n"
              "distinguishable from zero, so the share has no denominator.\n\n"
            + header + "\n" + "-" * len(header) + "\n"
            + "\n".join(lines) + "\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print("\n📄 Saved to %s" % args.out)


if __name__ == "__main__":
    main()
