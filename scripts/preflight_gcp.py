#!/usr/bin/env python
"""The complete offline pre-flight for the Step-5 GCP pilot.

THE POINT OF THIS FILE
----------------------
Every audit pass on this project has found something, because each pass looked
at a different layer and none of them left a durable record of what had been
checked. This script closes that set. It enumerates *every* precondition the
session-1 runbook assumes, checks each one, and prints what remains genuinely
unknown.

If this exits 0, there is no further offline check worth running: the remaining
unknowns are the four things only a live GPU can answer, and answering them is
what the pilot IS. That is the difference between "we hope nothing breaks" and
"we are running a designed experiment with four open questions".

Usage:
    ./venv/bin/python scripts/preflight_gcp.py
    ./venv/bin/python scripts/preflight_gcp.py --skip-cloud   # no gcloud calls

Exit code 0 = cleared to boot. 1 = at least one blocking check failed.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ZONE = "europe-west4-a"
REGION = "europe-west4"
INSTANCE = "instance-pierini-gpu"
# Must equal the real suite size, not a stale floor: at 96 against a 102-test
# suite, deleting tests/test_gain_sign_and_empty_transcription.py (exactly 6
# tests — the ones pinning the headline metric's sign) left 96 and passed.
EXPECTED_TESTS = 102
EXPECTED_CORRUPTED = 7488
EXPECTED_N = 416

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
results = []


def check(name, status, detail=""):
    results.append((name, status, detail))
    mark = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn ", SKIP: " skip "}[status]
    print("[{}] {}".format(mark, name))
    if detail:
        for line in str(detail).splitlines():
            print("          {}".format(line))


def sh(cmd, timeout=180):
    """Run a shell command, return (rc, stdout+stderr)."""
    try:
        p = subprocess.run(cmd, shell=True, cwd=str(REPO), timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, "timed out after {}s".format(timeout)


# ── 1. code ──────────────────────────────────────────────────────────────────

def check_tests():
    rc, out = sh("./venv/bin/python -m pytest tests/ -q", timeout=900)
    last = out.strip().splitlines()[-1] if out.strip() else "(no output)"
    if rc != 0:
        return check("test suite green", FAIL, last)
    n = 0
    for tok in last.replace("=", " ").split():
        if tok.isdigit():
            n = int(tok)
            break
    if n < EXPECTED_TESTS:
        # WARN does not block, so a deleted test file used to yield
        # "CLEARED TO BOOT" (audit 2026-08-14). A shrunken suite blocks.
        return check("test suite green", FAIL,
                     "{} (expected >= {} — did a test file go missing?)"
                     .format(last, EXPECTED_TESTS))
    check("test suite green", PASS, last)


def check_git_clean():
    rc, out = sh("git status --porcelain")
    if out.strip():
        return check("git tree clean", WARN,
                     "uncommitted changes:\n" + out.strip())
    _, head = sh("git log --oneline -1")
    check("git tree clean", PASS, head.strip())


def check_driver_flags():
    needed = {
        "route_omni.py": ["--num-samples", "--dump-raw-responses",
                          "--degradations", "--snr-levels",
                          "--max-missing-audio-frac", "--skip-model-check"],
        "route_transcripts.py": ["--dump-raw-responses", "--skip-model-check",
                                 "--workers", "--llm-base-url"],
    }
    missing = []
    for script, flags in needed.items():
        rc, out = sh("./venv/bin/python scripts/{} --help".format(script))
        if rc != 0:
            missing.append("{}: --help failed".format(script))
            continue
        for f in flags:
            if f not in out:
                missing.append("{}: {} absent".format(script, f))
    if missing:
        return check("drivers expose the flags the runbook uses", FAIL,
                     "\n".join(missing))
    check("drivers expose the flags the runbook uses", PASS,
          "route_omni.py + route_transcripts.py")


def check_helper_scripts():
    need = ["scripts/normalize_no_transcription.py", "scripts/compare_h2a_pilot.py",
            "scripts/compute_metrics.py", "scripts/annotate_errors.py",
            "scripts/compute_pf.py", "scripts/categorize_failures.py",
            "scripts/absorption_metrics.py"]
    absent = [p for p in need if not (REPO / p).exists()]
    if absent:
        return check("helper scripts present", FAIL, "\n".join(absent))
    check("helper scripts present", PASS, "{} scripts".format(len(need)))


# ── 2. data on disk ──────────────────────────────────────────────────────────

def check_audio_bank():
    n = sum(1 for _ in (REPO / "corrupted").rglob("*")
            if _.is_file() and _.suffix in (".flac", ".wav"))
    if n != EXPECTED_CORRUPTED:
        return check("corrupted audio bank complete", FAIL,
                     "{} files, expected {}".format(n, EXPECTED_CORRUPTED))
    check("corrupted audio bank complete", PASS, "{} files".format(n))


def check_pilot_cells():
    bad = []
    for d in ["clean"] + ["babble/snr10"]:
        p = REPO / "dataset/audio" if d == "clean" else REPO / "corrupted" / d
        n = len([x for x in p.iterdir() if x.is_file()]) if p.exists() else 0
        if n != EXPECTED_N:
            bad.append("{}: {} files, expected {}".format(d, n, EXPECTED_N))
    if bad:
        return check("omni pilot cells present (clean + babble@10)", FAIL,
                     "\n".join(bad))
    check("omni pilot cells present (clean + babble@10)", PASS,
          "{} + {} files".format(EXPECTED_N, EXPECTED_N))


def check_sidecars():
    live = [p for p in glob.glob(str(REPO / "results" / "*.failures.jsonl"))
            if os.path.getsize(p) > 0]
    if live:
        return check("no live failure sidecars", FAIL,
                     "\n".join(os.path.basename(p) for p in live))
    check("no live failure sidecars", PASS,
          "'sidecar empty' is a usable completion check")


def check_pilot_input():
    p = REPO / "results/parakeet_clean.jsonl"
    if not p.exists():
        return check("H2a pilot input is pristine", FAIL, "{} missing".format(p))
    rows = [json.loads(l) for l in p.open() if l.strip()]
    routed = [r for r in rows if r.get("predicted_intent") is not None]
    empty_in_100 = [i for i, r in enumerate(rows[:100])
                    if not (r.get("transcription") or "").strip()]
    if len(rows) != EXPECTED_N or routed:
        return check("H2a pilot input is pristine", FAIL,
                     "{} rows, {} already carry a prediction".format(len(rows), len(routed)))
    check("H2a pilot input is pristine", PASS,
          "{} rows, 0 predictions, {} empty transcription(s) in the first 100"
          .format(len(rows), len(empty_in_100)))


# ── 3. scoring conventions (the 2026-08-14 findings) ─────────────────────────

def violates_no_transcription_convention(row):
    """Is this row still on the pre-2026-08-12 empty-transcript convention?

    A CASCADE row whose ASR returned nothing must be stamped
    `predicted_intent = "NO_TRANSCRIPTION"` with a scored `tsa`, so that an
    infrastructure failure is never counted as a wrong answer.

    An OMNI row has no transcript at all — that is the pathway (§10), not a
    scoring bug. Until 2026-08-18 this check did not test `pathway`, so the
    200-row omni pilot made it exit 1 with "BLOCKED — do not boot": the gate
    guarding paid GPU time, crying wolf over the design working correctly.
    """
    if (row.get("pathway") or "cascade") == "omni":
        return False
    if (row.get("transcription") or "").strip():
        return False
    return (row.get("tsa") is None
            or row.get("predicted_intent") != "NO_TRANSCRIPTION")


def check_no_transcription_convention():
    """Every local routed file must use the Step-4c convention, or the paired
    H2a comparison silently uses different denominators per router."""
    # Twice now this check has been scoped by FILENAME and twice that gave a
    # false green: "*_qwen9b.jsonl" saw 19 of 38 files (missing every file
    # batch_evaluate.py wrote — the one driver that lacked the fix), and
    # "*qwen9b*.jsonl" excluded the pilot's OWN outputs, results/h2a_pilot_27b
    # .jsonl and results/omni_pilot_qwen.jsonl, which this very session writes.
    # So: scan every .jsonl in results/ and decide what is "routed" from its
    # CONTENT — a file is routed if any row carries a predicted_intent.
    offenders = []
    checked = 0
    for f in sorted(glob.glob(str(REPO / "results" / "*.jsonl"))):
        if any(s in f for s in ("_m.jsonl", "_ann.jsonl", "_scored.jsonl",
                                ".failures.jsonl", ".rawdump.jsonl")):
            continue
        rows = []
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        if not any(r.get("predicted_intent") is not None for r in rows):
            continue          # a pristine transcript file, nothing to check
        checked += 1
        for r in rows:
            if violates_no_transcription_convention(r):
                offenders.append("{}: slurp_id={} tsa={!r} intent={!r}".format(
                    os.path.basename(f), r.get("slurp_id"), r.get("tsa"),
                    r.get("predicted_intent")))
    if offenders:
        return check("empty-transcription convention consistent", FAIL,
                     "{} row(s) still on the pre-2026-08-12 convention; run "
                     "scripts/normalize_no_transcription.py\n".format(len(offenders))
                     + "\n".join(offenders[:5]))
    check("empty-transcription convention consistent", PASS,
          "{} routed files, all on the Step-4c convention".format(checked))


def check_baseline_scored():
    p = REPO / "results/parakeet_clean_qwen9b_scored.jsonl"
    if not p.exists():
        return check("H2a baseline is scored and complete", FAIL, "{} missing".format(p))
    rows = [json.loads(l) for l in p.open() if l.strip()]
    null_ees = [r for r in rows if r.get("ees") is None]
    if len(rows) != EXPECTED_N or null_ees:
        return check("H2a baseline is scored and complete", FAIL,
                     "{} rows, {} with null ees".format(len(rows), len(null_ees)))
    ees = sum(r["ees"] for r in rows)
    tsa = sum(r["tsa"] for r in rows if r.get("tsa") is not None)
    check("H2a baseline is scored and complete", PASS,
          "EES {}/{} = {:.1%} · TSA {}/{} = {:.1%}".format(
              ees, len(rows), ees / len(rows), tsa, len(rows), tsa / len(rows)))


def check_comparison_is_order_safe():
    """Prove the join-based comparison is immune to row order, since that is
    exactly the defect found on 2026-08-14."""
    import random
    src = REPO / "results/parakeet_clean_qwen9b_scored.jsonl"
    if not src.exists():
        return check("H2a comparison is order-safe", SKIP, "baseline missing")
    rows = [json.loads(l) for l in src.open() if l.strip()]
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    tmp = REPO / "results" / ".preflight_shuffled.jsonl"
    with tmp.open("w") as f:
        for r in shuffled:
            f.write(json.dumps(r) + "\n")
    try:
        rc, out = sh("./venv/bin/python scripts/compare_h2a_pilot.py "
                     "--baseline {} --candidate {}".format(src, tmp))
    finally:
        if tmp.exists():
            tmp.unlink()
    ok = rc == 0 and "discordant pairs: 0" in out and "paired rows: 416" in out
    if not ok:
        return check("H2a comparison is order-safe", FAIL,
                     "shuffling the candidate changed the result:\n" + out[-500:])
    check("H2a comparison is order-safe", PASS,
          "416 rows shuffled -> identical result (joins on slurp_id, not position)")


# ── 4. cloud ─────────────────────────────────────────────────────────────────

def check_gcloud_config():
    rc, out = sh("gcloud config list --format=json")
    if rc != 0:
        return check("gcloud account and project", FAIL, out.strip()[:300])
    cfg = json.loads(out)
    acct = cfg.get("core", {}).get("account")
    proj = cfg.get("core", {}).get("project")
    if acct != "leonardo.pierini4@gmail.com" or proj != "geant-federated-deep-learning":
        return check("gcloud account and project", FAIL,
                     "account={} project={}".format(acct, proj))
    check("gcloud account and project", PASS, "{} / {}".format(acct, proj))


def check_vm_off():
    rc, out = sh("gcloud compute instances describe {} --zone={} "
                 "--format='value(status)'".format(INSTANCE, ZONE))
    if rc != 0:
        return check("VM exists and is not billing", FAIL, out.strip()[:300])
    status = out.strip()
    if status != "TERMINATED":
        return check("VM exists and is not billing", WARN,
                     "{} is {} — it is billing right now".format(INSTANCE, status))
    check("VM exists and is not billing", PASS, "{} TERMINATED".format(INSTANCE))


def check_quota():
    rc, out = sh("gcloud compute regions describe {} --format=json".format(REGION))
    if rc != 0:
        return check("A100 quota available", WARN, out.strip()[:300])
    quotas = {q["metric"]: q for q in json.loads(out).get("quotas", [])}
    q = quotas.get("NVIDIA_A100_GPUS")
    if not q or q["limit"] - q["usage"] < 1:
        return check("A100 quota available", FAIL,
                     "{}: {}".format(REGION, q))
    check("A100 quota available", PASS,
          "{}: limit {:.0f}, in use {:.0f}".format(REGION, q["limit"], q["usage"]))


# ── main ─────────────────────────────────────────────────────────────────────

UNKNOWABLE = """
These CANNOT be checked offline. They are what the pilot exists to answer:

  1. Does the AWQ 4-bit Qwen3-Omni quant load in vLLM and ingest audio at all?
     -> read results/omni_pilot_qwen.rawdump.jsonl (route_omni.py:306 uses
        Path.stem, so the .jsonl is REPLACED, not appended). Fallback: §15.
  2. Is Gemma 4's thinking-off convention the Qwen key, or a silent no-op?
     -> CLOSED 2026-08-17: Gemma 4's chat_template.jinja uses the same
        `enable_thinking` key as Qwen, defaulting to false (line 186); checked
        by curl on both Gemma repos, no GPU needed. Still open for Gemma: that
        vLLM forwards chat_template_kwargs to the template (confirmed for Qwen,
        reasoning_content_present 0/20).
  3. Is the backend deterministic at temperature 0?
     -> re-run 50 rows; if it drifts, tracker §15 prescribes N=3 majority vote.
  4. Does europe-west4-a have A100 capacity today?
     -> quota is free (checked above); capacity is Google-side. A failed start
        is not billed. NOT europe-west1 (A100 quota but no A100 hardware) and NOT
        us-central1 (A100 quota 2, usage 2) -- both re-checked 2026-08-26. The
        nearest untried zone is europe-west4-b: same region, same free quota.

And one that was not a check but a decision, now taken: the pilot's stop rule
("if Qwen-27B shows no gap over 9B, STOP and talk to Testolin") FIRED on
2026-08-17 and was deliberately OVERRIDDEN on 2026-08-18 -- the 27B sweep runs.
Reason in §14 (docs/log/DECISIONS.md): the pilot could only detect a gap of
>=10 points and had 7% power against a 3-point gap, and §8 makes Qwen-27B the
reference cascade for H4b. Do not re-apply the stop rule; it is reported to
Testolin in September, not re-litigated here.
"""


def main():
    ap = argparse.ArgumentParser(description="Offline pre-flight for the GCP pilot.")
    ap.add_argument("--skip-cloud", action="store_true",
                    help="skip gcloud calls (offline / slow network)")
    ap.add_argument("--skip-tests", action="store_true",
                    help="skip the pytest run (it dominates the runtime)")
    args = ap.parse_args()

    print("=" * 72)
    print("GCP PILOT PRE-FLIGHT")
    print("=" * 72)

    print("\n-- code --")
    if args.skip_tests:
        check("test suite green", SKIP, "--skip-tests")
    else:
        check_tests()
    check_git_clean()
    check_driver_flags()
    check_helper_scripts()

    print("\n-- data on disk --")
    check_audio_bank()
    check_pilot_cells()
    check_sidecars()
    check_pilot_input()

    print("\n-- scoring conventions --")
    check_no_transcription_convention()
    check_baseline_scored()
    check_comparison_is_order_safe()

    print("\n-- cloud --")
    if args.skip_cloud:
        for n in ("gcloud account and project", "VM exists and is not billing",
                  "A100 quota available"):
            check(n, SKIP, "--skip-cloud")
    else:
        check_gcloud_config()
        check_vm_off()
        check_quota()

    fails = [r for r in results if r[1] == FAIL]
    warns = [r for r in results if r[1] == WARN]
    print("\n" + "=" * 72)
    print("{} checks: {} pass, {} fail, {} warn, {} skipped".format(
        len(results),
        sum(1 for r in results if r[1] == PASS), len(fails), len(warns),
        sum(1 for r in results if r[1] == SKIP)))

    if fails:
        print("\nBLOCKED — do not boot:")
        for n, _, _ in fails:
            print("  - {}".format(n))
    else:
        print("\nCLEARED TO BOOT.")
        if warns:
            print("(warnings above are not blocking, but read them)")
    print(UNKNOWABLE)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
