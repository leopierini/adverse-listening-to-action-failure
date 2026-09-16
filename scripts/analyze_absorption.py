"""Headline absorption metrics per (asr, degradation, snr, llm) cell.

All arithmetic lives in absorption_metrics.py — this script only loads rows,
groups them, prints a table and writes a CSV.

Four comparators are reported side by side (docs/design/METRICS.md §8):
  GAIN_P  gain_paired   — the headline. Items the system got right on clean
                          audio, degraded until the transcript broke.
  GAIN_C  gain_ceiling  — robustness. Baseline scaled by the router's ceiling.
  GAIN_L  gain_legacy   — audit trail. The defective pre-2026-08-12 formula,
                          reproduced so the correction stays auditable.
                          NEVER report this as a result.
  GAIN_K  gain_calibrated — the CALIBRATED null (§14, 2026-08-14). The three
                          above assume the comparator 1−WER; this one fits it
                          on the router's own clean rows. The only one whose
                          level is identified from data rather than assumed.

Pass every condition at once — the paired metric needs the clean rows to
build its anchor index:

    python analyze_absorption.py results/*_scored.jsonl \\
        --out results/analysis/absorption_qwen9b_v2.csv
"""

import argparse
import csv
import json
from collections import defaultdict

from absorption_metrics import (
    HIGH_WER_SUPPORT,
    build_clean_index,
    calibrated_stats,
    cell_key,
    cell_stats,
    fit_clean_calibration,
    paired_stats,
)


def compute_by_slot_position(rows, outcome="ees"):
    """Stratify a cell by whether its errors landed in critical slots."""
    critical_rows = []
    function_rows = []
    for r in rows:
        cats = r.get("error_categories") or []
        if any(c.get("slot_position") == "critical" for c in cats):
            critical_rows.append(r)
        elif cats:
            function_rows.append(r)
    return {
        "critical": cell_stats(critical_rows, outcome),
        "function": cell_stats(function_rows, outcome),
    }


def fmt(v, pct=False):
    if v is None:
        return "—"
    if pct:
        return f"{100*v:5.1f}%"
    return f"{v:.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--out", help="Optional CSV output path.")
    p.add_argument("--outcome", default="ees", choices=["ees", "tsa", "ees_strict"],
                   help="Success column. Default ees (tracker §8).")
    p.add_argument("--by-slot-position", action="store_true",
                   help="Additionally stratify by critical vs function slot position.")
    args = p.parse_args()

    all_rows = []
    for path in args.files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_rows.append(json.loads(line))

    curves = fit_clean_calibration(all_rows, args.outcome)
    for key, c in sorted(curves.items()):
        if c["refused"] is not None:
            print(f"⚠️  No calibrated null for {key}: {c['refused']}")

    clean_index = build_clean_index(all_rows)
    if not clean_index:
        print("⚠️  No clean-condition rows found — gain_paired cannot be computed.")
        print("    Pass the clean file too, e.g. results/*_scored.jsonl")

    buckets = defaultdict(list)
    for row in all_rows:
        buckets[cell_key(row)].append(row)

    print(f"\nOutcome: {args.outcome}   ·   cells: {len(buckets)}   "
          f"·   clean anchors: {len(clean_index)}")
    print(f"\n{'ASR':24s} {'DEG':10s} {'SNR':>4s} "
          f"{'N':>5s} {'NERR':>5s} {'NPAIR':>5s} {'WER':>7s} {'CEIL':>7s} "
          f"{'ABSORB':>7s} {'RETAIN':>7s} {'GAIN_P':>7s} {'GAIN_C':>7s} "
          f"{'GAIN_L':>7s} {'GAIN_K':>7s} {'HIWER':>7s}")
    print("=" * 134)

    csv_rows = []
    for key in sorted(buckets, key=lambda k: (k[0], k[1], str(k[2]))):
        asr, deg, snr, llm = key
        rows = buckets[key]
        s = cell_stats(rows, args.outcome)
        pstats = paired_stats(rows, clean_index, args.outcome)
        kstats = calibrated_stats(rows, curves, args.outcome)
        flag = " ⚠" if pstats["underpowered"] and pstats["n_paired"] else ""
        print(
            f"{asr:24s} {deg:10s} {('—' if snr is None else str(snr)):>4s} "
            f"{s['n']:>5d} {s['n_err']:>5d} {pstats['n_paired']:>5d} "
            f"{fmt(s['mean_wer_all'], pct=True):>7s} "
            f"{fmt(s['ceiling'], pct=True):>7s} "
            f"{fmt(s['absorption'], pct=True):>7s} "
            f"{fmt(pstats['retention'], pct=True):>7s} "
            f"{fmt(pstats['gain_paired'], pct=True):>7s} "
            f"{fmt(s['gain_ceiling'], pct=True):>7s} "
            f"{fmt(s['gain_legacy'], pct=True):>7s} "
            f"{fmt(kstats['gain_calibrated'], pct=True):>7s} "
            f"{fmt(kstats['share_at_high_wer'], pct=True):>7s}{flag}"
        )
        csv_rows.append({
            "asr": asr, "degradation": deg, "snr_db": snr, "llm": llm,
            "outcome": args.outcome, **s, **pstats, **kstats,
        })

        if args.by_slot_position:
            by_pos = compute_by_slot_position(rows, args.outcome)
            for pos in ("critical", "function"):
                sp = by_pos[pos]
                if sp["n"] == 0:
                    continue
                print(f"   └─ {pos:20s} {sp['n']:>5d} {sp['n_err']:>5d} "
                      f"{'—':>5s} {fmt(sp['mean_wer_all'], pct=True):>7s} "
                      f"{fmt(sp['ceiling'], pct=True):>7s} "
                      f"{fmt(sp['absorption'], pct=True):>7s} "
                      f"{'—':>7s} {'—':>7s} "
                      f"{fmt(sp['gain_ceiling'], pct=True):>7s} "
                      f"{fmt(sp['gain_legacy'], pct=True):>7s}")

    print("=" * 134)
    print("GAIN_P = retention − (1−WER) on items correct when clean  ← HEADLINE")
    print("GAIN_C = absorption − ceiling×(1−WER|err)                 ← robustness")
    print("GAIN_L = absorption − (1−WER over ALL rows)               ← AUDIT ONLY, do not report")
    print("GAIN_K = observed − predicted by the router's own clean WER→outcome curve,")
    print("         over the FULL degraded sample (§14). The comparator is FITTED, not assumed.")
    print(f"HIWER  = share of the cell sitting above capped WER {HIGH_WER_SUPPORT:.1f}, where the clean")
    print("         calibration is thin (8–31 rows per ASR) and the logistic interpolates.")
    print("⚠ = fewer than 30 paired items in the cell (exploratory).")

    if args.out:
        if not csv_rows:
            print("\n⚠️ No cells to write — CSV skipped.")
        else:
            with open(args.out, "w", newline="") as fout:
                writer = csv.DictWriter(fout, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"\n📄 CSV: {args.out}")


if __name__ == "__main__":
    main()
