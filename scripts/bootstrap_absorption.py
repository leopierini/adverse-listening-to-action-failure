"""Bootstrap 95% confidence intervals for absorption rate, absorption gain, and TSA.

For each (asr_model, degradation, snr_db, llm_model) cell, resample utterances
with replacement B times (default B=2000), recompute the cell statistics on each
resample, and report the 2.5th, 50th (median), and 97.5th percentiles.

Also computes the absorption-gain difference between critical-slot errors and
function-word errors per cell (the key H1 contrast), with its own CI.

Usage:
    python bootstrap_absorption.py results/analysis/*_annotated.jsonl \\
        --bootstrap 2000 --seed 42 --out results/analysis/absorption_ci.csv
"""

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from absorption_metrics import (
    build_clean_index,
    calibrated_stats,
    calibration_key,
    cell_key,
    cell_stats,
    fit_clean_calibration,
    paired_stats,
    resample_by_slurp_id,
)

METRICS = ("outcome", "absorption", "retention",
           "gain_paired", "gain_ceiling", "gain_legacy", "gain_calibrated")


def has_critical_error(row):
    return any(c.get("slot_position") == "critical"
               for c in (row.get("error_categories") or []))


def _collect(rows, clean_index, outcome, curves=None):
    """One replicate's worth of statistics, as a flat dict."""
    s = cell_stats(rows, outcome)
    p = paired_stats(rows, clean_index, outcome)
    k = calibrated_stats(rows, curves or {}, outcome)
    return {
        "outcome": s["mean_outcome"],
        "absorption": s["absorption"],
        "retention": p["retention"],
        "gain_paired": p["gain_paired"],
        "gain_ceiling": s["gain_ceiling"],
        "gain_legacy": s["gain_legacy"],
        "gain_calibrated": k["gain_calibrated"],
    }


def bootstrap_cell(rows, clean_index, clean_rows, outcome, n_boot, rng):
    """Resample the cell n_boot times. Returns {metric: [values]}.

    The paired filter runs inside the loop, via _collect, so the uncertainty
    in which items clear the clean anchor is propagated into the intervals.

    So does the CALIBRATED NULL's curve: `clean_rows` is resampled and the
    logistic refit in every replicate. Fitting it once outside the loop would
    treat an estimated curve as known and report a band narrower than the
    truth — the same error, in the null rather than in the subset. The two
    fitted parameters are returned alongside the gain so the curve itself can
    be reported with a band (§14).
    """
    out = {m: [] for m in METRICS}
    out["gain_critical"] = []
    out["gain_function"] = []
    out["gain_critical_minus_function"] = []
    out["calibration_a"] = []
    out["calibration_b"] = []

    keys = set(calibration_key(r) for r in rows)
    cell_curve_key = keys.pop() if len(keys) == 1 else None
    own_clean = [r for r in (clean_rows or [])
                 if calibration_key(r) == cell_curve_key]

    crit_rows = [r for r in rows if has_critical_error(r)]
    func_rows = [r for r in rows
                 if r.get("error_categories") and not has_critical_error(r)]

    for _ in range(n_boot):
        curves = fit_clean_calibration(resample_by_slurp_id(own_clean, rng),
                                       outcome) if own_clean else {}
        curve = curves.get(cell_curve_key)
        out["calibration_a"].append(
            curve["a"] if curve and curve["refused"] is None else np.nan)
        out["calibration_b"].append(
            curve["b"] if curve and curve["refused"] is None else np.nan)

        sample = resample_by_slurp_id(rows, rng)
        stats = _collect(sample, clean_index, outcome, curves)
        for m in METRICS:
            out[m].append(stats[m] if stats[m] is not None else np.nan)

        gc = gf = np.nan
        if crit_rows:
            g = _collect(resample_by_slurp_id(crit_rows, rng), clean_index,
                         outcome, curves)
            gc = g["gain_paired"] if g["gain_paired"] is not None else np.nan
        if func_rows:
            g = _collect(resample_by_slurp_id(func_rows, rng), clean_index,
                         outcome, curves)
            gf = g["gain_paired"] if g["gain_paired"] is not None else np.nan
        out["gain_critical"].append(gc)
        out["gain_function"].append(gf)
        out["gain_critical_minus_function"].append(
            gc - gf if not (np.isnan(gc) or np.isnan(gf)) else np.nan)
    return out


def pct(arr, q):
    a = [x for x in arr if not np.isnan(x)]
    return float(np.percentile(a, q)) if a else np.nan


def summarize(boot, key):
    """None when the metric is undefined for this cell — e.g. gain_paired on a
    clean cell, which is the reference for the comparison rather than a result.
    Returning None keeps '—' out of the CSV as an empty field instead of 'nan'."""
    arr = boot[key]
    if not arr or all(np.isnan(x) for x in arr):
        return None
    return {
        "median": pct(arr, 50),
        "lo": pct(arr, 2.5),
        "hi": pct(arr, 97.5),
    }


def fmt_ci(s, pct_fmt=True):
    if s is None:
        return "—"
    if pct_fmt:
        return f"{100*s['median']:5.1f}% [{100*s['lo']:+5.1f}, {100*s['hi']:+5.1f}]"
    return f"{s['median']:.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", help="Optional CSV output.")
    p.add_argument("--outcome", default="ees", choices=["ees", "tsa", "ees_strict"])
    args = p.parse_args()

    all_rows = []
    for path in args.files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_rows.append(json.loads(line))

    clean_index = build_clean_index(all_rows)
    if not clean_index:
        print("⚠️  No clean-condition rows found — gain_paired cannot be computed.")
    clean_rows = [r for r in all_rows
                  if (r.get("degradation") or "clean") == "clean"]
    cells = defaultdict(list)
    for row in all_rows:
        cells[cell_key(row)].append(row)

    rng = np.random.default_rng(args.seed)
    csv_rows = []

    print(f"\nBootstrapping {len(cells)} cells with B = {args.bootstrap} resamples each.")
    print(f"Outcome: {args.outcome}   ·   clean anchors: {len(clean_index)}\n")
    header = (f"{'ASR':30s} {'DEG':10s} {'SNR':>4s} | "
              f"{'OUTCOME':24s} {'ABSORB':24s} {'GAIN_PAIRED':24s} "
              f"{'GAIN_CALIBRATED':24s} {'GAIN_P(crit−func)':24s}")
    print(header)
    print("─" * len(header))

    for key in sorted(cells):
        asr, deg, snr, llm = key
        rows = cells[key]
        if len(rows) < 10:
            print(f"{asr:30s} {deg:10s} {str(snr):>4s} | n={len(rows)} too small, skipping")
            continue
        boot = bootstrap_cell(rows, clean_index, clean_rows, args.outcome,
                              args.bootstrap, rng)
        cis = {m: summarize(boot, m) for m in METRICS}
        crit_minus_func_ci = summarize(boot, "gain_critical_minus_function")
        print(
            f"{asr:30s} {deg:10s} {str(snr):>4s} | "
            f"{fmt_ci(cis['outcome']):24s} {fmt_ci(cis['absorption']):24s} "
            f"{fmt_ci(cis['gain_paired']):24s} "
            f"{fmt_ci(cis['gain_calibrated']):24s} "
            f"{fmt_ci(crit_minus_func_ci):24s}"
        )
        entry = {"asr": asr, "degradation": deg, "snr_db": snr, "llm": llm,
                 "outcome_col": args.outcome, "n": len(rows)}
        for name, ci in cis.items():
            for stat in ("median", "lo", "hi"):
                entry[f"{name}_{stat}"] = ci[stat] if ci else ""
        # visualize.py and generate_report.py read gain_median/lo/hi and
        # tsa_median/lo/hi — keep those names and point them at the headline
        # metric so the figure scripts keep working unchanged.
        for stat in ("median", "lo", "hi"):
            entry[f"gain_{stat}"] = entry[f"gain_paired_{stat}"]
            entry[f"tsa_{stat}"] = entry[f"outcome_{stat}"]
            entry[f"gain_crit_minus_func_{stat}"] = (
                crit_minus_func_ci[stat] if crit_minus_func_ci else "")
        # The calibrated null's own curve, with its band — §14 asks for the
        # fitted curve to be reported, not only the gain it implies.
        for name in ("calibration_a", "calibration_b"):
            ci = summarize(boot, name)
            for stat in ("median", "lo", "hi"):
                entry[f"{name}_{stat}"] = ci[stat] if ci else ""
        csv_rows.append(entry)

    # H1 (tracker §3) predicts errors OUTSIDE critical slots hurt the intent more,
    # i.e. critical-slot errors are absorbed MORE. So crit-func > 0 SUPPORTS H1.
    # This banner said "H1 rejected" until 2026-08-14 — it told the analyst the
    # opposite of what the number means.
    print(f"\nReadout: GAIN_P(crit−func) > 0 with the 2.5% bound also > 0 → critical-slot errors")
    print(f"are absorbed STRICTLY MORE than function-word errors → H1 SUPPORTED for this cell.")
    print(f"gain_paired CI excluding 0 → the router's absorption differs from")
    print(f"proportional propagation in that cell (positive = real error correction).")
    print(f"GAIN_CALIBRATED replaces the ASSUMED comparator 1−WER with the router's own")
    print(f"clean WER→outcome curve, refit inside every replicate (§14, 2026-08-14).")
    print(f"It is the only one of the four whose level is identified from data.")

    if args.out:
        if not csv_rows:
            print("\n⚠️ No cell had n ≥ 10 — nothing to write, CSV skipped.")
        else:
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                w.writeheader()
                w.writerows(csv_rows)
            print(f"\n📄 CSV: {args.out}")


if __name__ == "__main__":
    main()
