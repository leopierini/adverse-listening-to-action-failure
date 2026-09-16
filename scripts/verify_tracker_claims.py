#!/usr/bin/env python
"""Check the factual claims in the design record against the data and the code.

WHY THIS EXISTS
---------------
The design sections (§4 experimental design, §6 dataset, §8 metrics, §9 power,
§13 findings — now in docs/design/ and docs/log/, see THESIS_TRACKER.md for the
map) are full of numbers: N=416, 209 cells, 7,488 files,
"whisper-base has WER=0 on only 126/416 clean rows", the absorption gain
bracket, the intent inventory. Each was true when written. None of them was
re-checkable, so nobody could tell a stale number from a live one — the exact
failure that put two BASE model checkpoints in §5 for a month (see §14).

This asserts each claim against its real source: the dataset files, the scored
JSONL, the analysis CSVs, and the scoring code itself. Run it after editing the
tracker, and before quoting any number from it in the thesis.

    ./venv/bin/python scripts/verify_tracker_claims.py

Exit 0 = every checkable claim holds. 1 = at least one is stale or wrong.

MAINTENANCE: the `expected` value in each check() call is a **second copy of what
the tracker asserts**. When you change a number in THESIS_TRACKER.md you must
change it here too, and vice versa — that is the point: a drift between the two
is exactly the failure this script exists to surface, and it shows up as a
red line rather than as silence. Do not "fix" a red line by editing only the
expectation; go find out which side is wrong first.
"""

import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

results = []


def check(section, claim, expected, actual, ok=None):
    if ok is None:
        ok = (expected == actual)
    results.append((section, claim, expected, actual, ok))
    print("  [{}] {:<52} claimed={!s:<18} actual={!s}".format(
        "ok " if ok else "BAD", claim[:52], expected, actual))
    return ok


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_sweep_rows():
    """Every row of the experimental sweep — and ONLY those.

    A bare glob over results/*_scored.jsonl is not the sweep: on 2026-08-17 the
    200-row H2a pilot landed in the same directory and turned eight true claims
    in this file red without a word of the tracker changing. scripts/result_sets.py
    is the single place that says which files count.
    """
    import result_sets
    rows = []
    for path in result_sets.sweep_files(REPO / "results"):
        rows.extend(load_jsonl(path))
    return rows


def load_arm_rows(tag):
    """Rows of ONE router arm of the cascade sweep.

    Needed from 2026-08-27, when the Qwen-27B arm made "the sweep" stop
    meaning "one router". Every router re-routes the SAME transcripts, so a
    figure measured on transcripts — a WER count, a clean-row count, a
    per-cell absorption gain — is multiplied, not refined, by pooling arms.
    The symptom is unmistakable and it showed up here first: "whisper-base
    clean rows total" read 832 where §8 records 416, the same 416
    transcriptions counted once per router.
    """
    import result_sets
    rows = []
    for path in result_sets.cascade_files(REPO / "results"):
        if tag in path.name:
            rows.extend(load_jsonl(path))
    return rows


# ── §6 the dataset ───────────────────────────────────────────────────────────

def check_dataset():
    print("\n§6 — the SLURP base")
    man = load_jsonl(REPO / "dataset/manifest.jsonl")
    check("§6", "N utterances", 416, len(man))

    scen = Counter(r.get("scenario") for r in man)
    check("§6", "scenario calendar", 252, scen.get("calendar"))
    check("§6", "scenario lists", 101, scen.get("lists"))
    check("§6", "scenario alarm", 63, scen.get("alarm"))
    check("§6", "scenarios sum to N", 416, sum(scen.values()))

    intents = {r.get("gold_intent_final") for r in man}
    check("§6", "distinct gold_intent_final labels", 9, len(intents))

    # The 9 strings §6 lists, plus the `unknown` escape hatch, must match the
    # ontology the prompt actually enforces.
    import batch_evaluate as be
    claimed = {"calendar_set", "calendar_query", "calendar_remove",
               "alarm_set", "alarm_query", "alarm_remove",
               "lists_createoradd", "lists_query", "lists_remove", "unknown"}
    check("§6", "intent inventory == batch_evaluate.VALID_INTENTS",
          "match", "match" if claimed == set(be.VALID_INTENTS) else
          "diff: " + str(claimed ^ set(be.VALID_INTENTS)))
    check("§6", "gold labels are a subset of the ontology", True,
          intents <= set(be.VALID_INTENTS))

    def rows(p):
        with open(REPO / p) as f:
            return list(csv.DictReader(f))
    check("§6", "manual label corrections", 4, len(rows("dataset/corrections_log.csv")))
    check("§6", "utterances removed", 5, len(rows("dataset/removed_samples.csv")))

    # §6 read "17" until 2026-08-14 and had never been checked against the file.
    rev = rows("dataset/review_log.csv")
    amb = [r for r in rev if r.get("status") == "ambiguous_retained"]
    check("§6", "ambiguous kept + flagged", 22, len(amb))
    check("§6", "review_log rows = corrections + ambiguous", 26, len(rev))

    check("§6", "audio files in dataset/audio",
          416, len(list((REPO / "dataset/audio").glob("*.flac"))))


# ── §4 the design arithmetic ─────────────────────────────────────────────────

def check_design():
    print("\n§4 — design and cell counts")
    N, ASR, ROUTERS, OMNI = 416, 3, 3, 2
    DEG, SNR = 6, 3
    conds = 1 + DEG * SNR
    check("§4", "conditions = 1 clean + 6 deg x 3 SNR", 19, conds)
    check("§4", "cascade cells = 3 ASR x 19 x 3 routers", 171, ASR * conds * ROUTERS)
    check("§4", "omni cells = 19 x 2 models", 38, conds * OMNI)
    check("§4", "total cells", 209, ASR * conds * ROUTERS + conds * OMNI)
    check("§4", "ASR transcriptions = 3 x 19 x 416", 23712, ASR * conds * N)
    check("§4", "cascade routing calls = 23,712 x 3", 71136, ASR * conds * N * ROUTERS)
    check("§4", "omni calls = 2 x 19 x 416", 15808, OMNI * conds * N)

    bank = [p for p in (REPO / "corrupted").rglob("*")
            if p.is_file() and p.suffix in (".flac", ".wav")]
    check("§4", "corrupted bank = 416 x 6 x 3", 7488, len(bank))


# ── §8 metric definitions vs the code that computes them ─────────────────────

def check_metric_definitions():
    print("\n§8 — metric definitions vs the implementation")
    src = (REPO / "scripts/compute_pf.py").read_text()
    check("§8", "EES threshold PF >= 0.5 is the default", True,
          'default=0.5' in src and 'pf >= args.pf_threshold' in src)
    check("§8", "ees_strict requires PF == 1.0", True, 'pf >= 1.0' in src)

    # §8 claims all absorption formulas live in exactly one module.
    others = []
    for p in sorted(glob.glob(str(REPO / "scripts/*.py"))):
        if Path(p).name in ("absorption_metrics.py", "verify_tracker_claims.py"):
            continue
        t = Path(p).read_text()
        if "wer > 0" in t or "1 - mean_wer" in t or "1 - wer" in t.replace("_", " "):
            others.append(Path(p).name)
    check("§8", "absorption formulas confined to absorption_metrics.py",
          [], others)


# ── §8 / §13 numbers that came out of the data ───────────────────────────────

def check_findings():
    print("\n§8/§13 — numbers measured from the collected data")
    rows = load_sweep_rows()
    # Progress: the whole sweep, both pathways. These climbed as arms landed,
    # and they are the only numbers here that were SUPPOSED to move.
    #   57 -> 133 (2026-08-27, Qwen-27B + Qwen3-Omni)
    #   133 -> 209 (2026-08-27, Gemma-31B + Gemma-12B, on vLLM 0.28.0)
    # 209 of 209 is the complete design (§4), so from here a CHANGE in any of
    # the four numbers below is a defect, not progress: it means a file was
    # added, lost or reclassified. The second copy of each is STATUS.md's
    # generated half, which is recounted from the files at every run of
    # scripts/generate_status.py -- a drift between the two is the signal.
    check("§13", "rows collected", 86944, len(rows))

    # Measurements: ONE arm. See load_arm_rows for why pooling them is not a
    # bigger sample but a double count.
    arm = load_arm_rows("qwen9b")
    clean_base = [r for r in arm
                  if r.get("degradation") == "clean"
                  and r.get("asr_model") == "mlx-community/whisper-base-mlx"]
    zero = [r for r in clean_base if r.get("wer") == 0]
    check("§8", "whisper-base clean rows with WER == 0 (9B arm)", 126, len(zero))
    check("§8", "whisper-base clean rows total (9B arm)", 416, len(clean_base))

    cascade = [r for r in rows if (r.get("pathway") or "cascade") == "cascade"]
    asr = Counter(r.get("asr_model") for r in cascade)
    # 19 conditions x 416 utterances x 3 routers = 23,712 per ASR, which is
    # §4's own arithmetic for the COMPLETE cascade (71,136 routing calls / 3
    # ASRs). It read 15,808 while only two routers had landed.
    check("§4", "rows per ASR (cascade only)", 23712,
          min(asr.values()) if asr else None)
    # Counted over the cascade alone: the omni rows carry asr_model = None by
    # construction — audio in, intent out, no transcript — and counting that
    # None would report a fourth ASR that does not exist.
    check("§4", "distinct ASR models (cascade only)", 3, len(asr))
    # Counted over BOTH pathways, so this is 3 cascade routers + 2 omni models
    # = the 5 §4 specifies, not 3. The old label said "routers so far" while
    # the expression has always counted omni models too.
    check("§4", "distinct routers + omni models", 5,
          len({r.get("llm_model") for r in rows}))

    cells = {(r.get("asr_model"), r.get("llm_model"),
              r.get("degradation"), r.get("snr_db")) for r in rows}
    check("§0a", "cells complete", 209, len(cells))

    # §13's headline bracket, recomputed cell-by-cell from the shared module —
    # the same aggregation analyze_absorption.py performs — rather than read
    # back from the CSV it produced.
    from collections import defaultdict
    import absorption_metrics as am
    idx = am.build_clean_index(arm)
    buckets = defaultdict(list)
    for r in arm:
        if (r.get("degradation") or "clean") != "clean":
            buckets[am.cell_key(r)].append(r)
    gains = []
    for cell_rows in buckets.values():
        st = am.paired_stats(cell_rows, idx, "ees")
        if st.get("gain_paired") is not None:
            gains.append(st["gain_paired"])
    check("§13", "degraded cells with an estimable gain", 54, len(gains))
    mean_gain = round(sum(gains) / len(gains), 3) if gains else None
    check("§13", "absolute-null mean gain_paired", 0.093, mean_gain,
          ok=(mean_gain is not None and abs(mean_gain - 0.093) < 0.001))
    check("§13", "cells negative under the absolute null", 12,
          sum(1 for g in gains if g < 0))

    # The OTHER endpoint the documents quote. Until 2026-08-14 the incremental
    # null existed only in prose: -0.106 was quoted in §13 and guarded by
    # nothing, in a script whose whole purpose is to stop exactly that.
    inc = []
    for cell_rows in buckets.values():
        st = am.paired_stats_incremental(cell_rows, idx, "ees")
        if st.get("gain_incremental") is not None:
            inc.append(st["gain_incremental"])
    mean_inc = round(sum(inc) / len(inc), 3) if inc else None
    check("§13", "incremental-null mean gain", -0.106, mean_inc,
          ok=(mean_inc is not None and abs(mean_inc + 0.106) < 0.001))
    check("§13", "cells negative under the incremental null", 54,
          sum(1 for g in inc if g < 0))



def check_reported_levels():
    """The numbers the documents tell you to LEAD WITH.

    None of these were guarded until 2026-08-14, which is exactly why the
    retention floor drifted: three documents paired 18.3% with `reverb 0 dB`
    when 18.3% is `babble 0 dB` and reverb is 12.0%. A number quoted in prose
    and checked by nothing will eventually be wrong.
    """
    print("\n§13 — the levels the docs tell you to quote")

    # §13 names this one "Router ceiling on perfect transcripts (Qwen3.5-9B)",
    # and it is a statement about that router: how well it does when the ASR
    # handed it a perfect transcript. Averaged over two routers it is a
    # statement about neither. Measured over the 9B arm it reproduces 0.699;
    # pooled with the 27B arm it read 0.702 on 2026-08-27, which is not a
    # correction to the recorded figure but a different quantity.
    rows = load_arm_rows("qwen9b")

    zero = [r for r in rows if r.get("wer") == 0 and r.get("ees") is not None]
    ceiling = sum(r["ees"] for r in zero) / len(zero)
    check("§8", "router ceiling mean(EES | WER=0), 9B arm", 0.699,
          round(ceiling, 3))

    import csv as _csv
    with open(REPO / "results/analysis/absorption_qwen9b_v2.csv") as f:
        cells = [c for c in _csv.DictReader(f) if c.get("retention")]
    ret = sorted(cells, key=lambda c: float(c["retention"]))
    lo, hi = ret[0], ret[-1]
    check("§13", "retention floor value", 0.120, round(float(lo["retention"]), 3))
    check("§13", "retention floor cell", "whisper-base|reverb|0",
          "{}|{}|{}".format(lo["asr"].split("/")[-1].replace("-mlx", ""),
                            lo["degradation"], lo["snr_db"]))
    check("§13", "retention ceiling value", 0.990, round(float(hi["retention"]), 3))
    check("§13", "retention ceiling cell", "whisper-large-v3-turbo|farfield|20",
          "{}|{}|{}".format(hi["asr"].split("/")[-1], hi["degradation"], hi["snr_db"]))


def check_prompt_coverage():
    """§14 records that the 9-shot prompt covers only 8 of 9 intents. Pin it,
    so that if anyone ever edits the prompt the discrepancy surfaces here
    instead of silently invalidating the collected rows."""
    print("\n§14 — few-shot prompt coverage")
    import re
    import batch_evaluate as be
    src = (REPO / "scripts/pipeline.py").read_text()
    found = Counter(re.findall(r'"intent"\s*:\s*"([a-z_]+)"', src))
    real = {i for i in be.VALID_INTENTS if i != "unknown"}
    check("§14", "few-shot examples in the prompt", 9, sum(found.values()))
    check("§14", "distinct intents exemplified (of 9)", 8,
          len([i for i in real if found.get(i, 0) > 0]))
    check("§14", "intent with NO exemplar", ["alarm_query"],
          sorted(i for i in real if found.get(i, 0) == 0))
    check("§14", "intent exemplified twice", ["alarm_remove"],
          sorted(i for i in real if found.get(i, 0) > 1))


def check_rawdump_path():
    """The runbook tells you to open this file on a billed A100. It was wrong
    until 2026-08-14 (Path.stem REPLACES the suffix, it does not append)."""
    print("\n runbook — the raw-dump path the pilot depends on")
    # This check used to RE-IMPLEMENT the path construction and compare it to
    # itself — a compile-time tautology that stayed green when route_omni.py's
    # real line was replaced with a broken one. Read the source instead.
    from pathlib import Path as _P
    src = (REPO / "scripts/route_omni.py").read_text()
    expr = 'out_path.parent / (out_path.stem + ".rawdump.jsonl")'
    check("runbook", "route_omni.py still builds the path with .stem",
          True, expr in src)
    out = _P("results/omni_pilot_qwen.jsonl")
    actual = str(out.parent / (out.stem + ".rawdump.jsonl")) if expr in src else "UNKNOWN"
    rb = (REPO / "docs/gcp-session-1-runbook.md").read_text()
    check("runbook", "runbook opens the path route_omni actually writes",
          True, actual != "UNKNOWN" and actual in rb)


def main():
    print("=" * 84)
    print("VERIFYING THE DESIGN RECORD'S CLAIMS AGAINST DATA AND CODE")
    print("=" * 84)
    for fn in (check_dataset, check_design, check_metric_definitions, check_findings,
               check_reported_levels, check_prompt_coverage, check_rawdump_path):
        try:
            fn()
        except Exception as e:
            check("?", fn.__name__ + " crashed", "runs", str(e)[:60], ok=False)

    bad = [r for r in results if not r[4]]
    print("\n" + "=" * 84)
    print("{} claims checked, {} hold, {} do NOT".format(
        len(results), len(results) - len(bad), len(bad)))
    if bad:
        print("\nSTALE OR WRONG — fix the design record (or the code):")
        for sec, claim, exp, act, _ in bad:
            print("  {} {}: tracker says {!s}, reality is {!s}".format(sec, claim, exp, act))
        return 1
    print("\nEvery checkable claim in the design record holds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
