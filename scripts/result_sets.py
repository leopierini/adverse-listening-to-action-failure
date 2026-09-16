"""Which files in results/ are THE SWEEP, and which are not.

Pure functions only: no printing, no argparse, no analysis.

WHY THIS EXISTS
---------------
Until 2026-08-18 nothing in the repo stated which of `results/*.jsonl` make up
the experimental sweep. Every consumer wrote its own bare glob. On 2026-08-17
the 200-row H2a pilot landed in the same directory and, without one line of
code or documentation changing:

  * eight claims in verify_tracker_claims.py went red (23,712 rows became
    23,912; 1 router became 2; 57 cells became 59), and
  * fit_mixed_effects.py fitted a router coefficient estimated from 2 of the
    57 design cells and printed it under the name "H2".

Neither the documents nor the data were wrong. The file selection was. A glob
that silently changes meaning when a file appears next to it is not a
selection — it is an accident waiting for a reader.

An unrecognised `*_scored.jsonl` raises rather than being dropped: the next arm
to land here (Gemma-31B, the omni sweeps) must force a decision about which set
it belongs to, not slip into neither.
"""

import re
from pathlib import Path

# The router arms of the cascade sweep. One tag per arm, added deliberately as
# the arm lands -- never a wildcard. A wildcard would readmit exactly what this
# module exists to keep out, and it would also swallow
# `cascade_whisper_qwen9b_babble_snr0_qwen27b_scored.jsonl`, the filename that
# announces the 76-file bug (the tag APPENDED instead of substituted).
#   qwen9b   -- local, Mac, MLX 8-bit, 57 cells, 2026-07/08
#   qwen27b  -- GCP, 57 cells, collected 2026-08-27
#   gemma31b -- GCP, 57 cells, collected 2026-08-27. Exists only because vLLM
#               was upgraded to 0.28.0 in a second environment; on the 0.27.1
#               that served the Qwen models it does not start at all (§15).
ROUTER_TAGS = "qwen9b|qwen27b|gemma31b"

# The omni pathway: one file per model, all 19 conditions inside it. The two
# names are fixed by the runbook, and only those two -- `omni_qwen3_sweep` or
# any other invention must raise, not drift into the counts. Note how close
# `omni_qwen_sweep_scored.jsonl` (the sweep, 7,904 rows) sits to
# `omni_pilot_qwen_scored.jsonl` (the 200-row session-1 probe, a PILOT): one
# word apart, opposite sets. PILOT_PATTERNS is tried after these, and neither
# pattern can match the other's name.
OMNI_PATTERNS = (
    re.compile(r"^omni_(?:qwen|gemma)_sweep_scored\.jsonl$"),
)

# The cascade sweep, one file per (ASR family, degradation, SNR).
SWEEP_PATTERNS = (
    re.compile(r"^cascade_whisper_(?:%s)_[a-z]+(_snr\d+)?_scored\.jsonl$"
               % ROUTER_TAGS),
    re.compile(r"^parakeet_[a-z]+(_snr\d+)?_(?:%s)_scored\.jsonl$"
               % ROUTER_TAGS),
)

# Deliberately outside the sweep: small runs on a subset of conditions, kept
# for the record. Counting them as sweep rows is what broke the claim checks.
PILOT_PATTERNS = (
    re.compile(r"^h2a_pilot_.*_scored\.jsonl$"),
    re.compile(r"^omni_pilot_.*_scored\.jsonl$"),
)

BASELINE_PATTERNS = (
    re.compile(r"^baseline_.*_scored\.jsonl$"),
)


class UnclassifiedResultFile(Exception):
    """A *_scored.jsonl file matching no known set.

    Raised rather than ignored: a new experimental arm must be classified
    explicitly, because every count in the tracker is derived from these sets.
    """


def _matches(name, patterns):
    return any(p.match(name) for p in patterns)


def _classified(results_dir):
    results_dir = Path(results_dir)
    sweep, pilots, baselines, unknown = [], [], [], []
    for path in sorted(results_dir.glob("*_scored.jsonl")):
        name = path.name
        if _matches(name, SWEEP_PATTERNS) or _matches(name, OMNI_PATTERNS):
            sweep.append(path)
        elif _matches(name, PILOT_PATTERNS):
            pilots.append(path)
        elif _matches(name, BASELINE_PATTERNS):
            baselines.append(path)
        else:
            unknown.append(path)
    if unknown:
        raise UnclassifiedResultFile(
            "these scored files match no known result set: %s. Add a pattern to "
            "scripts/result_sets.py saying whether each belongs to the sweep, "
            "the pilots or the baselines — every row count in THESIS_TRACKER.md "
            "is derived from that decision."
            % ", ".join(p.name for p in unknown)
        )
    return sweep, pilots, baselines


def sweep_files(results_dir="results"):
    """The experimental sweep: every row the tracker's counts refer to.

    Both pathways, cascade and omni — the design is 209 cells, 171 of them
    cascade and 38 omni, and `generate_status.py` derives its "pathways
    present" row from exactly this set. Use `cascade_files` when an analysis
    is about the cascade specifically; a cascade-only statistic computed over
    this list would silently average an omni arm into it.
    """
    return _classified(results_dir)[0]


def cascade_files(results_dir="results"):
    """The cascade half of the sweep, no omni rows."""
    return [p for p in _classified(results_dir)[0]
            if _matches(p.name, SWEEP_PATTERNS)]


def omni_files(results_dir="results"):
    """The omni half of the sweep, no cascade rows."""
    return [p for p in _classified(results_dir)[0]
            if _matches(p.name, OMNI_PATTERNS)]


def pilot_files(results_dir="results"):
    """Small exploratory runs. Real data, but not part of any cell count."""
    return _classified(results_dir)[1]


def baseline_files(results_dir="results"):
    """Early smoke-test runs kept for the record."""
    return _classified(results_dir)[2]
