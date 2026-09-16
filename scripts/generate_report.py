"""Generate a thesis-ready Markdown results report from the analysis outputs.

Reads:
  - results/analysis/*_pf.jsonl        (rows with WER, slot annotations, PF, EES)
  - results/analysis/absorption_ci_final.csv  (bootstrap CIs)

Writes:
  - results/analysis/RESULTS.md   — chapter-ready prose + tables
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_all_rows(jsonl_paths):
    rows = []
    for p in jsonl_paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def cell_stats(rows):
    n = len(rows)
    if n == 0:
        return None
    wers = [min(r["wer"], 1.0) for r in rows if r.get("wer") is not None]
    tsas = [r["tsa"] for r in rows if r.get("tsa") is not None]
    pfs = [r["pf"] for r in rows if r.get("pf") is not None]
    ees = [r["ees"] for r in rows if r.get("ees") is not None]
    ees_strict = [r["ees_strict"] for r in rows if r.get("ees_strict") is not None]
    collapse = sum(1 for r in rows if (r.get("wer") or 0) > 1.0)
    return {
        "n": n,
        "mean_wer": sum(wers) / len(wers) if wers else None,
        "median_wer": sorted(wers)[len(wers) // 2] if wers else None,
        "mean_tsa": sum(tsas) / len(tsas) if tsas else None,
        "mean_pf": sum(pfs) / len(pfs) if pfs else None,
        "mean_ees": sum(ees) / len(ees) if ees else None,
        "mean_ees_strict": sum(ees_strict) / len(ees_strict) if ees_strict else None,
        "collapse": collapse,
    }


def pct(x):
    return "—" if x is None else f"{100*x:5.1f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs-glob", default="results/analysis/*_pf.jsonl")
    p.add_argument("--ci-csv", default="results/analysis/absorption_ci_final.csv")
    p.add_argument("--out", default="results/analysis/RESULTS.md")
    args = p.parse_args()

    files = sorted(Path(".").glob(args.inputs_glob))
    rows = load_all_rows([str(f) for f in files])
    print(f"Loaded {len(rows)} rows from {len(files)} files")

    # Bucket by cell
    cells = defaultdict(list)
    for r in rows:
        key = (
            (r.get("asr_model") or "?").split("/")[-1],
            r.get("degradation") or "clean",
            r.get("snr_db"),
            r.get("llm_model") or "?",
        )
        cells[key].append(r)

    # Load bootstrap CIs
    ci_rows = {}
    if Path(args.ci_csv).exists():
        with open(args.ci_csv) as f:
            for r in csv.DictReader(f):
                k = (r["asr"], r["degradation"], (None if r["snr_db"] == "" else int(r["snr_db"])), r["llm"])
                ci_rows[k] = r

    # ─── Compose Markdown ────────────────────────────────────────────────
    md = []
    md.append("# Results (auto-generated)")
    md.append("")
    md.append("> Auto-generated from `results/analysis/*_pf.jsonl` and `results/analysis/absorption_ci_final.csv`.")
    md.append("> Re-run `python generate_report.py` to refresh.")
    md.append("")
    md.append(f"**Total observations:** {len(rows):,} across {len(cells)} (ASR × degradation × SNR × LLM) cells.")
    md.append("")
    md.append("## 1. Per-cell performance summary")
    md.append("")
    md.append("| ASR | Degradation | SNR (dB) | N | WER (med) | TSA | PF | EES | EES strict | collapse |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")

    for key in sorted(cells):
        asr, deg, snr, llm = key
        s = cell_stats(cells[key])
        snr_str = "—" if snr is None else str(snr)
        md.append(
            f"| {asr} | {deg} | {snr_str} | {s['n']} | {pct(s['median_wer'])} "
            f"| {pct(s['mean_tsa'])} | {pct(s['mean_pf'])} | {pct(s['mean_ees'])} "
            f"| {pct(s['mean_ees_strict'])} | {s['collapse']} |"
        )

    md.append("")
    md.append("**Reading the table:**")
    md.append("- *TSA*: % correct intent. The headline accuracy.")
    md.append("- *PF*: mean parameter-fidelity recall (soft value matching against SLURP gold entities).")
    md.append("- *EES*: lenient end-to-end success (TSA = 1 AND PF ≥ 0.5).")
    md.append("- *EES strict*: TSA = 1 AND PF = 1.0.")
    md.append("- *collapse*: utterances with WER > 1.0 (typically ASR repetition loops).")
    md.append("")

    md.append("## 2. H1 contrast: GAIN(critical − function)")
    md.append("")
    md.append("Bootstrap 95% CIs for GAIN(critical − function). H1 predicts that critical-slot "
              "errors hurt *more* than function-word errors, i.e. **negative** values support H1; "
              "**positive** values contradict it:")
    md.append("")
    md.append("| ASR | Degradation | SNR | GAIN(crit−func) median [95% CI] | H1 status |")
    md.append("|---|---|---|---|---|")
    rejected = supported = borderline = total = 0
    for k, r in sorted(ci_rows.items()):
        asr, deg, snr, llm = k
        lo = float(r["gain_crit_minus_func_lo"])
        hi = float(r["gain_crit_minus_func_hi"])
        med = float(r["gain_crit_minus_func_median"])
        total += 1
        if lo > 0:
            status = "❌ H1 rejected (CI > 0)"
            rejected += 1
        elif hi < 0:
            status = "✓ H1 supported (CI < 0)"
            supported += 1
        else:
            status = "borderline"
            borderline += 1
        snr_str = "—" if snr is None else str(snr)
        md.append(f"| {asr} | {deg} | {snr_str} | {100*med:+.1f}% [{100*lo:+.1f}, {100*hi:+.1f}] | {status} |")
    md.append("")
    if total:
        md.append(f"**Summary:** {rejected}/{total} cells reject H1 (CI strictly > 0), "
                  f"{supported}/{total} support H1 (CI strictly < 0), "
                  f"{borderline}/{total} borderline.")
    else:
        md.append("**Summary:** no bootstrap CIs available — run bootstrap_absorption.py first.")
    md.append("")
    md.append("## 3. Dose-response (Whisper-Turbo TSA)")
    md.append("")
    md.append("| Degradation | clean | SNR=20 | SNR=10 | SNR=0 |")
    md.append("|---|---|---|---|---|")
    for deg in ["noise", "reverb", "farfield", "codec", "clipping"]:
        cells_for_deg = {}
        for snr in [None, 20, 10, 0]:
            # Pool rows across LLM routers (cells are keyed per-LLM; overwriting
            # with the last-seen LLM would silently pick an arbitrary router)
            pooled = []
            for k, v in cells.items():
                if k[0] != "whisper-large-v3-turbo":
                    continue
                if snr is None and k[1] == "clean" and k[2] is None:
                    pooled.extend(v)
                elif snr is not None and k[1] == deg and k[2] == snr:
                    pooled.extend(v)
            cells_for_deg[snr] = cell_stats(pooled) if pooled else None
        row = f"| {deg}"
        for snr in [None, 20, 10, 0]:
            s = cells_for_deg.get(snr)
            row += f" | {pct(s['mean_tsa']) if s else '—'}"
        row += " |"
        md.append(row)
    md.append("")
    md.append("(`clean` column is the Whisper-Turbo × clean cell, shared by all rows. "
              "Cells are pooled across LLM routers; per-router numbers are in §1.)")
    md.append("")
    md.append("## 4. Methodological notes")
    md.append("")
    md.append("- Per-utterance WER is capped at 1.0; the *collapse* column counts rows with raw WER > 1.0 (ASR repetition loops), whose uncapped values would poison cell means.")
    md.append("- PF uses value-based soft matching against SLURP gold entities (see compute_pf.py); strict key-equality would underestimate PF because the router and SLURP use different slot vocabularies.")
    md.append("")
    print(md_text := "\n".join(md))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(md_text)
    print(f"\n\n📄 Saved to {out}")


if __name__ == "__main__":
    main()
