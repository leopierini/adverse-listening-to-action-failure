"""§8 metric 5 — paired cascade-vs-omni absorption, per degradation.

All arithmetic lives in absorption_metrics.py; this script only loads rows,
groups them into cells, and writes a table.

§8 (docs/design/METRICS.md, metric 5): on the clips where the REFERENCE
CASCADE had WER > 0, `mean(EES_omni) - mean(EES_cascade)`.

  positive -> the omni pathway recovers actions the cascade lost to
              transcription error
  negative -> the text stage was PROTECTIVE

The reference cascade is the best available one of the SAME vendor family
(Whisper-Turbo -> Qwen-27B / Gemma-31B), taken from model_frames.py so the
comparison matches the H4b confirmatory fit rather than defining its own.
Pitting an omni model against Whisper Base or the 9B router would inflate the
pathway effect — the mirror image of the capacity-calibration caveat in §3.
The Qwen omni is ALSO reported against the 9B cascade, descriptively, exactly
as §8 asks.

    ./venv/bin/python scripts/analyze_pathway_absorption.py \\
        --out results/analysis/pathway_absorption_2026-08-28.csv
"""

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

import result_sets
from absorption_metrics import (
    pathway_paired_ci,
    pool_pathway_cells,
)
from model_frames import REFERENCE_ASR, REFERENCE_LARGE_ROUTERS


def short(name):
    return (name or "?").split("/")[-1]


def load_rows(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def comparisons(rows):
    """Yield (label, omni_llm, cascade_llm, is_confirmatory) pairings.

    Confirmatory = the same-family large router §8 names. The 9B pairing is
    emitted too and flagged descriptive, so a reader can never mistake which
    one answers the hypothesis.
    """
    omni_llms = sorted({r["llm_model"] for r in rows
                        if (r.get("pathway") or "cascade") == "omni"})
    cascade_llms = {r["llm_model"] for r in rows
                    if (r.get("pathway") or "cascade") == "cascade"}

    for omni_llm in omni_llms:
        family = next(r.get("model_family") for r in rows
                      if r.get("llm_model") == omni_llm)
        same_family = [c for c in cascade_llms
                       if next(r.get("model_family") for r in rows
                               if r.get("llm_model") == c) == family]
        for cascade_llm in sorted(same_family):
            confirmatory = short(cascade_llm) in REFERENCE_LARGE_ROUTERS
            label = "%s vs %s" % (short(omni_llm), short(cascade_llm))
            yield label, omni_llm, cascade_llm, confirmatory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*",
                   help="Default: the sweep, as scripts/result_sets.py defines it.")
    p.add_argument("--out", help="CSV output path.")
    p.add_argument("--outcome", default="ees",
                   choices=["ees", "tsa", "ees_strict"])
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    paths = args.files or [str(x) for x in result_sets.sweep_files()]
    rows = load_rows(paths)
    print("caricate %d righe da %d file" % (len(rows), len(paths)))

    rng = np.random.default_rng(args.seed)
    csv_rows = []

    for label, omni_llm, cascade_llm, confirmatory in comparisons(rows):
        # The reference cascade is one ASR only; the cascade files carry both
        # Whisper models, so this filter is not optional.
        cas = [r for r in rows
               if (r.get("pathway") or "cascade") == "cascade"
               and r.get("llm_model") == cascade_llm
               and short(r.get("asr_model")) == REFERENCE_ASR]
        omn = [r for r in rows
               if (r.get("pathway") or "cascade") == "omni"
               and r.get("llm_model") == omni_llm]

        cas_cells = defaultdict(list)
        omn_cells = defaultdict(list)
        for r in cas:
            cas_cells[(r.get("degradation") or "clean", r.get("snr_db"))].append(r)
        for r in omn:
            omn_cells[(r.get("degradation") or "clean", r.get("snr_db"))].append(r)

        per_degradation = defaultdict(list)
        for cell in sorted(set(cas_cells) & set(omn_cells), key=lambda k: (k[0], str(k[1]))):
            deg, snr = cell
            # `clean` is NOT dropped: run through the SAME selection rule
            # (cascade WER > 0) it is the capacity anchor for this pairing.
            # Any gain that is already present on clean audio is not the
            # pathway absorbing degradation — it is one model being better
            # than the other. It is emitted with scope="anchor" and kept out
            # of the per-degradation pooling.
            stats = pathway_paired_ci(cas_cells[cell], omn_cells[cell], rng,
                                      outcome=args.outcome,
                                      n_boot=args.bootstrap)
            if deg != "clean":
                per_degradation[deg].append(stats)
            csv_rows.append({
                "comparison": label, "role": "confirmatory" if confirmatory else "descriptive",
                "degradation": deg, "snr_db": snr,
                "scope": "anchor" if deg == "clean" else "cell",
                "n_pairs": stats["n_pairs"],
                "cascade_rate": stats["cascade_rate"],
                "omni_rate": stats["omni_rate"],
                "gain_pathway": stats["gain_pathway"],
                "ci_lo": stats["ci_lo"], "ci_hi": stats["ci_hi"],
                "underpowered": stats["underpowered"],
            })

        for deg in sorted(per_degradation):
            pooled = pool_pathway_cells(per_degradation[deg])
            csv_rows.append({
                "comparison": label, "role": "confirmatory" if confirmatory else "descriptive",
                "degradation": deg, "snr_db": "pooled", "scope": "degradation",
                "n_pairs": pooled["n_pairs"],
                "cascade_rate": pooled["cascade_rate"],
                "omni_rate": pooled["omni_rate"],
                "gain_pathway": pooled["gain_pathway"],
                "ci_lo": "", "ci_hi": "",
                "underpowered": pooled["underpowered"],
            })

    for r in csv_rows:
        if r["scope"] not in ("degradation", "anchor"):
            continue
        print("%-52s %-9s n=%5d  cascade=%.3f  omni=%.3f  gain=%+.4f%s"
              % (r["comparison"], r["degradation"], r["n_pairs"],
                 r["cascade_rate"], r["omni_rate"], r["gain_pathway"],
                 "  [DESCRITTIVO]" if r["role"] == "descriptive" else ""))

    if args.out and csv_rows:
        with open(args.out, "w", newline="") as fout:
            w = csv.DictWriter(fout, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print("scritto %s (%d righe)" % (args.out, len(csv_rows)))


if __name__ == "__main__":
    main()
