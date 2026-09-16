#!/usr/bin/env python
"""Check the four project documents against reality — and against each other.

WHY THIS EXISTS
---------------
`verify_tracker_claims.py` guards THESIS_TRACKER.md. Nothing guarded the other
three, and `CLAUDE.md` is the one injected into every session. On 2026-08-18 it
still said "Step 5 is next" and "europe-west4-a has had no A100 capacity since
2026-08-12" — both false since 2026-08-17, when Step 5 ran and the VM started
first try. Every session therefore opened on a stale premise and repeated it
with confidence. That is the "every time I open the project there are errors"
complaint, and its cause was structural: RULE ZERO governs claims being
WRITTEN; nothing re-checked a claim that was true when written and became false
later.

This checks the classes of staleness that have actually occurred here:

  * a document referring to a file that no longer exists
  * a hard-coded test count that no longer matches the suite
  * STATUS.md's counted half not regenerated after the data moved
  * a §N citation that resolves to no document, or to two
  * a retention percentage written beside the wrong degradation
  * claims the hard rules forbid outright (`gain_legacy`, the falsified
    bracket, the word "desk-verified")
  * a model id that looks right and is wrong (`google/gemma-4-31b` is BASE)

    ./venv/bin/python scripts/verify_docs.py

Exit 0 = every checkable claim holds. 1 = at least one is stale or wrong.
Run by `.claude/hooks/check-docs.sh` at the start of every session, so no
session can begin without seeing what is currently false.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Every hand-written document an agent or a reader is expected to trust.
# STATUS.md's generated half is checked separately (status_drift); its
# hand-written half is checked like any other prose.
DOCUMENTS = (
    # Added 2026-08-28. It goes public with a DOI and makes more checkable
    # claims than any other file here -- every command, count and version in
    # it is a promise to a stranger reproducing the work.
    "REPRODUCIBILITY.md",
    "CLAUDE.md",
    "STATUS.md",
    "EXECUTION_TODO.md",
    "THESIS_TRACKER.md",
    "docs/design/DESIGN.md",
    "docs/design/METRICS.md",
    "docs/design/SCHEMA.md",
    "docs/design/STACK.md",
    "docs/log/FINDINGS.md",
    "docs/log/DECISIONS.md",
    "docs/log/MILESTONES.md",
    "docs/log/RISKS.md",
    "literature/READING_LIST.md",
    "literature/literature-review.md",
    "literature/search-log.md",
    "dataset/README.md",
    # Operational documents — the ones commands are pasted FROM. Added
    # 2026-08-19 after a cold-start audit found `gcp-session-1-runbook.md`
    # telling the operator to stop unless pytest reported 102 green (the suite
    # was at 213), and a session-2 runbook no file referenced. A document that
    # spends money must be checked at least as hard as one that describes it.
    "docs/gcp-session-1-runbook.md",
    "docs/gcp-session-2-runbook.md",
    "docs/gcp-session-1/RISULTATI.md",
    # Session record of 2026-08-28/29, same category as RISULTATI.md above: it
    # carries the determinism figure Methods will quote for the fifth arm, the
    # measured cost, and the timestamps the dead-man's switch is judged by.
    # Checked rather than excluded on purpose -- a number that goes stale here
    # is a number that goes stale in Methods.
    "docs/gcp-session-3/ESITO.md",
    "docs/gcp-session-3/PIANO_NOTTE.md",
    # Session 4 (2026-08-30): same category again. PIANO.md carries the
    # measured evidence the safety mechanisms are judged by, ESITO.md what
    # the session actually collected.
    "docs/gcp-session-4/PIANO.md",
    "docs/gcp-session-4/ESITO.md",
    # The degree regulation, read 2026-08-30. Checked rather than excluded
    # because it is the only document in the repository whose claims are about
    # an external authority: an invented requirement here is worse than a stale
    # number, and its "what this does NOT say" section is the guard.
    "docs/THESIS_REGULATIONS.md",
    # §18, the writing plan. Kept separate from THESIS_REGULATIONS.md on the
    # project's own rule: the constraints change yearly, the plan changes while
    # Step 9 runs, and a weekly document must not live inside a yearly one.
    "docs/WRITING_PLAN.md",
    # The three documents written 2026-09-13 that face OUTWARD -- one is the
    # summary intended for the advisor, one describes the defence slides, one
    # describes the LaTeX workspace. Checked rather than excluded for the reason
    # the runbooks were added in 2026-08-19: a document someone else reads must
    # be checked at least as hard as one only this project reads. The first audit
    # of them found a chapter described as an "advanced draft" that contained no
    # prose, appendices described as ready that were TODO stubs, and a
    # bibliography described as verified whose DOIs pointed at other papers.
    "docs/TESI_LIVING_SUMMARY.md",
    "presentation/README.md",
    "thesis/README.md",
)

# Documents deliberately NOT checked, each with the reason. Being here is a
# decision on the record; being in neither list is an oversight, and
# `uncovered_documents` refuses that third state.
EXCLUDED_DOCUMENTS = (
    # Dated snapshots of a design as it stood. They describe layouts and counts
    # that were true then and must stay quotable.
    "docs/superpowers/",
    # A frozen request, and the README that explains how the deliverable
    # departed from it: both name `docs/literature-review.md` on purpose — the
    # brief asked for that path, the review was written at
    # `literature/literature-review.md` instead. The dead path IS the point.
    "literature/literature-review-brief.md",
    "literature/README.md",
    # Per-paper reading notes, one file per source. Excluded rather than checked
    # because every claim in them is about the content of a copyrighted PDF that
    # is NOT in the repository (`papers/` and `literature/pdfs/` are gitignored),
    # so no checker here could confirm one. Each note carries its own `Verifica:`
    # line naming the check that was run instead.
    "literature/notes/",
    # Agent tooling, not project record.
    ".claude/",
    # Leonardo's own study guide to the project, in Italian, written 2026-09-13.
    # Excluded because it is a teaching narrative written for one reader and
    # nothing cites it -- but the exclusion is NOT a licence: it carried the same
    # three statistical misreadings the outward-facing documents did, and those
    # were corrected by hand on 2026-09-13. Re-read it whenever §13 changes.
    "GUIDA_COMPLETA_TESI.md",
)

# Trees that are not this project's documents at all.
# NOTE: "thesis" was in this set on 2026-09-13 and is deliberately not any more.
# It hid `thesis/README.md` -- the document describing the LaTeX workspace -- from
# the classification decision entirely. Only `.md` files are enumerated, so the
# `.tex` sources are outside this checker either way; the claims inside them are
# guarded by nothing but review, which is why the audited numbers live in
# `docs/log/FINDINGS.md` and are quoted into the chapters with their caveats.
_NOT_DOCUMENTS = frozenset((".git", ".pytest_cache", "venv", "venv312",
                            "models", "node_modules", "__pycache__",
                            "literature/pdfs"))


# --- dead file references ---------------------------------------------------

# A path-looking token: at least one slash, no whitespace, a file extension or
# a trailing slash. Globs and URLs are excluded by the caller.
_PATH = re.compile(r"`([^`\s]+/[^`\s]*)`")


def dead_paths(text, repo=REPO):
    """Paths mentioned in backticks that do not exist on disk.

    Catches the commonest silent rot: a script is renamed and four documents go
    on telling the reader to run the old name.

    Only tokens whose FIRST segment is a real entry at the repo root are
    checked. Without that restriction the first run reported 29 "dead paths"
    that were HuggingFace repo ids, DOIs, a CIDR block, `<think></think>` and a
    formula — burying the two findings that were real. A checker that cries
    wolf is worse than no checker.
    """
    repo = Path(repo)
    roots = {p.name for p in repo.iterdir()}
    dead = []
    for raw in _PATH.findall(text):
        candidate = raw.strip().rstrip(",.;:)")
        if "*" in candidate or "://" in candidate or "…" in candidate:
            continue
        if "{" in candidate or ":" in candidate:      # templates, file:line
            continue
        if candidate.startswith(("http", "~", "$", "<")):
            continue
        if candidate.split("/", 1)[0] not in roots:   # not a path into this repo
            continue
        if not (repo / candidate).exists() and candidate not in dead:
            dead.append(candidate)
    return dead


# --- the stated test count --------------------------------------------------

# A LIVE claim is the "here is how you run the suite" idiom: the command with a
# trailing `#` comment naming the count, as it appears in CLAUDE.md and §0a.
#
# A past run recorded as an outcome ("-> 34 passed", "Test suite: 34 passing",
# a dated milestone) is history and must stay writable. Flagging history would
# train everyone to ignore the checker, which is worse than not having one.
_LIVE_TEST_COUNT = re.compile(r"pytest\s+tests/.*#\s*(\d+)\s+tests?\b")


def stated_test_counts(text):
    """Every live claim in `text` about how many tests the suite has now."""
    return [int(m.group(1)) for m in _LIVE_TEST_COUNT.finditer(text)]


def actual_test_count(repo=REPO):
    """How many tests the suite really has, by collecting them.

    Returns None rather than raising when the suite cannot be collected: a
    checker that dies where it cannot check takes the whole session down with
    it, and "unknown" is a usable answer.
    """
    try:
        proc = subprocess.run(
            [str(Path(repo) / "venv/bin/python"), "-m", "pytest", "tests/",
             "-q", "--collect-only"],
            capture_output=True, text=True, cwd=str(repo),
        )
    except OSError:
        return None
    match = re.search(r"(\d+)\s+tests? collected", proc.stdout)
    if match:
        return int(match.group(1))
    tail = proc.stdout.strip().splitlines()
    match = re.search(r"^(\d+)\s*$", tail[-1]) if tail else None
    return int(match.group(1)) if match else None


# --- sync duplicates, which silently inflate every count ---------------------

# macOS and every file-sync client resolve a name collision by appending " 2"
# to the stem. The copy is untracked, so `git status` shows it as noise and it
# reads as harmless — but nothing that COUNTS files knows it is a copy.
_SYNC_DUPLICATE = re.compile(r"^(?P<stem>.+) \d+$")

# Vendored trees. Thousands of files that are not this project's to keep clean.
_UNSCANNED = frozenset((".git", "venv", "venv312", "__pycache__",
                        "node_modules", ".pytest_cache"))


def sync_duplicate_files(repo=REPO):
    """Files named `X N.ext` sitting beside the `X.ext` they were copied from.

    On 2026-08-19 three of these existed. `tests/test_model_frames 2.py` was a
    byte-identical copy carrying 21 tests, so pytest collected 201 + 21 = 222
    and the session-start check reported STATUS.md as stale — pointing at the
    one file that was right, and prescribing a regeneration that would have
    written 222 into it. The same class of copy under `results/` would enter a
    row count through `result_sets.py`.

    Each finding names the twin and says whether the two still agree: an
    identical copy is safe to delete, a diverged one may hold an edit that only
    ever landed in the copy.
    """
    repo = Path(repo)
    found = []
    for root, dirnames, filenames in os.walk(str(repo)):
        dirnames[:] = sorted(d for d in dirnames if d not in _UNSCANNED)
        for filename in sorted(filenames):
            path = Path(root) / filename
            match = _SYNC_DUPLICATE.match(path.stem)
            if not match:
                continue
            twin = path.with_name(match.group("stem") + path.suffix)
            if not twin.exists():
                continue  # a legitimate name that happens to end in a number
            try:
                agree = path.read_bytes() == twin.read_bytes()
            except OSError:
                agree = False
            if agree:
                verdict = ("byte-identical to it — delete the copy, "
                           "nothing is lost")
            else:
                verdict = ("DIVERGED from it — diff the two before deleting; "
                           "the copy may hold an edit that never landed")
            found.append("`%s` is a sync copy of `%s`, %s"
                         % (path.relative_to(repo), twin.relative_to(repo),
                            verdict))
    return found


# --- a document nobody classified is a document nobody checks ----------------

def uncovered_documents(repo=REPO, documents=None, excluded=None):
    """Project `.md` files in neither the checked list nor the excluded one.

    `DOCUMENTS` was a hand-maintained tuple, so coverage depended on someone
    remembering. On 2026-08-19 a cold-start audit found the consequence:
    `docs/gcp-session-1-runbook.md` instructed the operator to STOP unless the
    suite reported 102 green — 111 tests stale — and passed every check,
    because the one check that catches exactly that (`stated_test_counts`) only
    ever ran over the tuple. A second runbook existed that no file referenced.

    Modelled on `result_sets.py`: an unclassified file is refused rather than
    ignored, so adding a document forces a decision about checking it.
    """
    repo = Path(repo)
    documents = DOCUMENTS if documents is None else documents
    excluded = EXCLUDED_DOCUMENTS if excluded is None else excluded
    uncovered = []
    for root, dirnames, filenames in os.walk(str(repo)):
        dirnames[:] = sorted(d for d in dirnames if d not in _NOT_DOCUMENTS)
        for filename in sorted(filenames):
            if not filename.endswith(".md"):
                continue
            relative = (Path(root) / filename).relative_to(repo).as_posix()
            if any(part in _NOT_DOCUMENTS for part in relative.split("/")):
                continue
            if relative in documents:
                continue
            if any(relative == e or relative.startswith(e) for e in excluded):
                continue
            uncovered.append(relative)
    return uncovered


# --- a model id nobody classified -------------------------------------------

# Only real Hugging Face orgs. A bare `a/b` pattern matches every fraction in
# the documents ("48/100", "14/14") and every path.
_MODEL_ID = re.compile(
    r"\b(?:Qwen|google|mlx-community|cyankiwi|RedHatAI|nvidia|openai|"
    r"meta-llama|unsloth)/[A-Za-z0-9][\w.-]*")

# Models this project runs locally, outside the GCP four that
# verify_model_specs.py checks against the Hugging Face API.
LOCAL_MODELS = (
    "mlx-community/Qwen3.5-9B-MLX-8bit",       # the 9B router, MLX 8-bit
    "mlx-community/whisper-large-v3-turbo",    # ASR 1
    "mlx-community/whisper-base-mlx",          # ASR 2
    "mlx-community/parakeet-tdt-0.6b-v3",      # ASR 3
)

# Considered and rejected, each named in the documents together with the reason
# it cannot be used. They must stay quotable — that IS the record.
REJECTED_MODELS = (
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",          # BF16 70.5 GB, will not fit
    "RedHatAI/gemma-4-31b-it-FP8-dynamic",       # FP8 needs Ada/Hopper
    "google/gemma-4-31B-it-qat-q4_0-gguf",       # GGUF targets llama.cpp
    # BF16, fits a 40 GB card — rejected anyway, so the Google cascade
    # and omni pair stay quantization-matched (docs/design/STACK.md).
    "google/gemma-4-12b-it",
)


def unclassified_model_ids(texts):
    """Model ids named in a document that no list accounts for.

    `trap_model_ids` catches ids already known to be wrong. This catches the
    other direction, which is how the failure came back on 2026-08-19:
    `docs/log/DECISIONS.md` presented `google/gemma-4-31b-it` as the CORRECTED
    id while `verify_model_specs.py` lists it under "Ids that look right and
    are WRONG" — BF16 62.5 GB, does not fit the A100. Naming a model is a
    decision with a GPU sweep behind it, so an unlisted id is refused rather
    than read past.

    Struck-through ids are skipped: a superseded id kept beside its correction
    is the audit trail, not a live claim.
    """
    import verify_model_specs

    known = set(m["id"] for m in verify_model_specs.MODELS)
    known.update(t[0] for t in verify_model_specs.TRAPS)
    known.update(LOCAL_MODELS)
    known.update(REJECTED_MODELS)

    found = {}
    for name, text in sorted(texts.items()):
        for model_id in _MODEL_ID.findall(_STRUCK.sub(" ", text)):
            if model_id not in known:
                found.setdefault(model_id, set()).add(name)
    return ["`%s` in %s — classify it in verify_model_specs.py (MODELS or "
            "TRAPS) or in verify_docs.py (LOCAL_MODELS, REJECTED_MODELS)"
            % (model_id, ", ".join(sorted(names)))
            for model_id, names in sorted(found.items())]


# --- two live documents disagreeing about the same step ---------------------

_STEP = re.compile(r"\bStep\s+(\d+)\b")
_BLOCKS = re.compile(r"must not start|must not begin|cannot start|"
                     r"\bis blocked\b|blocked until|do not start", re.I)
_CLEARS = re.compile(r"\bis unblocked\b|unblocked as of|cleared to start|"
                     r"\bis cleared\b|may start|can start", re.I)
_SENTENCE = re.compile(r"(?<=[.;:!?])\s+")

# `THESIS_TRACKER.md:38` — "a superseded finding stays with its correction
# beside it". Strikethrough is this project's mark for "this no longer holds",
# so a struck instruction is not a live one. Reading it as live would punish the
# documented convention and push people to DELETE the audit trail instead.
_STRUCK = re.compile(r"~~.+?~~", re.S)

# `docs/log/` is append-only history. A milestone recording that Step 6 was
# blocked in August is not a stale claim — it is the record.
_APPEND_ONLY = "docs/log/"


def step_blocking_conflicts(texts):
    """Steps that one live document blocks and another declares clear.

    On 2026-08-19 STATUS.md said "Step 6 is unblocked as of 2026-08-18" while
    EXECUTION_TODO.md — the file whose whole job is "what do I do next" — still
    said "Step 6 must not start until this is discussed with Testolin". Both
    are hand-written prose, so nothing here could see it: the decision had been
    recorded in §14 and in STATUS.md, and the checklist was simply never
    updated. An agent reading one file stops and an agent reading the other
    proceeds, which is the failure this whole checker exists to prevent.

    Compared sentence by sentence, so "Step 6 is unblocked. Step 8 must not
    start yet." is two claims about two steps, not a contradiction.
    """
    blocked, cleared = {}, {}
    for name, text in sorted(texts.items()):
        if name.startswith(_APPEND_ONLY):
            continue
        for line in _STRUCK.sub(" ", text).splitlines():
            for sentence in _SENTENCE.split(line):
                steps = set(_STEP.findall(sentence))
                if not steps:
                    continue
                for step in steps:
                    if _BLOCKS.search(sentence):
                        blocked.setdefault(step, set()).add(name)
                    if _CLEARS.search(sentence):
                        cleared.setdefault(step, set()).add(name)
    conflicts = []
    for step in sorted(set(blocked) & set(cleared), key=int):
        conflicts.append(
            "Step %s is declared blocked by %s and clear by %s — one of them "
            "was not updated when the blocker was decided"
            % (step, ", ".join(sorted(blocked[step])),
               ", ".join(sorted(cleared[step]))))
    return conflicts


# --- next action vs the checklist -------------------------------------------

_NEXT_ACTION = re.compile(r"##\s*Next action\s*\n+.*?\bStep\s+(\d+)", re.S)
_STEP_HEADER = re.compile(r"^##\s*STEP\s+(\d+)[^\n]*", re.M)
_HEADER_DONE = re.compile(r"✅\s*DONE")


def next_action_step(text):
    """The step number CLAUDE.md's 'Next action' section names."""
    match = _NEXT_ACTION.search(text)
    return int(match.group(1)) if match else None


def first_unchecked_step(text):
    """The lowest-numbered step that is not marked done and has an open box.

    A step whose header carries "✅ DONE" is skipped even if it still holds an
    unticked line: a completed step can carry something waiting on someone
    else, and reading that as "the next piece of work" is precisely how the
    checklist and CLAUDE.md drifted apart on 2026-08-18. Deferred items belong
    in the "Waiting on the advisor" section, not in a step's box list.
    """
    headers = [(m.start(), int(m.group(1)), m.group(0))
               for m in _STEP_HEADER.finditer(text)]
    for i, (start, number, header) in enumerate(headers):
        if _HEADER_DONE.search(header):
            continue
        end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        if "- [ ]" in text[start:end]:
            return number
    return None


# --- claims the hard rules forbid -------------------------------------------

# Each: (regex that finds the claim, regex that marks a legitimate mention —
# the rule stating the prohibition must itself be writable).
# These documents legitimately DISCUSS the things they forbid: the decision log
# records that a label was false, the metric module names the defective column
# in its own definition, the correction notice quotes the error it corrected.
# What must be caught is a document USING one as a live result.
#
# Two principles do most of the work, and both are narrow on purpose — a broad
# exemption would gut the check, and a checker that cries wolf trains everyone
# to ignore it:
#
#   QUOTED   a forbidden label inside quotation marks is being quoted, not
#            asserted. "The specs were labelled 'desk-verified'" is the
#            decision log doing its job; **Desk-verified 2026-07-13** as a
#            table cell is the register the rules forbid.
#   VALUED   naming a deprecated column carries no result; naming it beside a
#            number does. `cell_stats` returning `gain_legacy` is a signature;
#            "mean gain_legacy was -0.359" is a report.

_QUOTED_SPAN = re.compile("\"[^\"]*\"|\u201c[^\u201d]*\u201d")
_NUMBER_NEAR = 60
_NUMBER = re.compile(r"[-+\u2212]?\d+\.\d+|\b\d{2,}\b")

_DISCUSSING_NOT_ASSERTING = re.compile(
    r"(never quote|never be reported|must never|do not report|forbid|"
    r"audit[- ]?(trail|record|only)|audit\b|defective|reproduces?\b|"
    r"was FALSE|falsified|not a bound|removed on|was wrong|were the BASE|"
    r"until 20\d\d-\d\d-\d\d|corrected|historical|red flag)", re.I)

# (pattern, message, mode). mode "quoted" exempts a quoted mention; "valued"
# flags only when a number sits within _NUMBER_NEAR characters.
FORBIDDEN = (
    (re.compile(r"\bgain_legacy\b"),
     "quotes a value from the deprecated `gain_legacy` column", "valued"),
    (re.compile(r"absorption_qwen9b\.csv"),
     "quotes `absorption_qwen9b.csv`, kept only as the audit record", "valued"),
    (re.compile(r"bracket(ed)?\s+(by|between|:)?\s*\[?\s*[-\u2212]0\.106", re.I),
     "revives the bracket [-0.106, +0.093], falsified 2026-08-14", "plain"),
    (re.compile(r"desk[- ]verified", re.I),
     "uses the word 'desk-verified', which the hard rules forbid", "quoted"),
)


def _is_quoted(line, match):
    """Is this match sitting inside quotation marks?

    Straight and curly double quotes only. An apostrophe is far too common in
    English prose to treat as an opening quote.
    """
    return any(span.start() <= match.start() and match.end() <= span.end()
               for span in _QUOTED_SPAN.finditer(line))


def _has_number_near(line, match):
    window = line[max(0, match.start() - _NUMBER_NEAR):
                  match.end() + _NUMBER_NEAR]
    # A path or a command is not a reported value: strip the obvious ones.
    window = re.sub(r"`[^`]*\.(csv|py|jsonl|md)`", "", window)
    return bool(_NUMBER.search(window))


def forbidden_claims(text):
    """Lines making a claim the hard rules prohibit.

    Checked line by line so that a paragraph stating the prohibition does not
    excuse a violation elsewhere in the same document.
    """
    hits = []
    for number, line in enumerate(text.splitlines(), 1):
        if _DISCUSSING_NOT_ASSERTING.search(line):
            continue
        for pattern, message, mode in FORBIDDEN:
            for match in pattern.finditer(line):
                if mode == "quoted" and _is_quoted(line, match):
                    continue
                if mode == "valued" and not _has_number_near(line, match):
                    continue
                hits.append("line %d %s: %s"
                            % (number, message, line.strip()[:90]))
                break
    return hits


# --- model ids that look right and are wrong --------------------------------

# Phrases that mark a trap being NAMED in order to warn about it, rather than
# recommended. The warning has to stay writable.
_TRAP_EXEMPTION = re.compile(
    r"(base\*{0,2} checkpoint|are \*\*base\*\*|BASE checkpoint|do \*{0,2}not\*{0,2} fit|"
    r"use the|use google/|use Qwen/|trap|wrong|⚠️|does not fit|BF16|FP8 needs)",
    re.I)


def trap_model_ids(text, repo=REPO):
    """Mentions of a model id that looks right and is wrong.

    `google/gemma-4-31b` (base) sat in THESIS_TRACKER.md for a month under the
    label "desk-verified". Serving it would have measured the wrong model
    across two arms of the study.

    Matched with a boundary so that the CORRECT id is never condemned for
    containing a trap as its prefix: `google/gemma-4-31B-it-qat-w4a16-ct`
    starts with the trap `google/gemma-4-31B-it`.
    """
    sys.path.insert(0, str(Path(repo) / "scripts"))
    import verify_model_specs

    hits = []
    for line in text.splitlines():
        if _TRAP_EXEMPTION.search(line):
            continue
        for trap, reason in verify_model_specs.TRAPS:
            pattern = re.compile(re.escape(trap) + r"(?![\w./-])", re.I)
            if pattern.search(line):
                hits.append("%s — %s" % (trap, reason))
    return hits


# --- the retention pairing that was wrong in three documents ----------------

# (value, the degradation it actually belongs to). Read from
# results/analysis/absorption_qwen9b_v2.csv on 2026-08-14 and re-checked by
# verify_tracker_claims.py, which pins the floor cell itself.
RETENTION_PAIRINGS = (
    ("18.3%", "babble"),
    ("12.0%", "reverb"),
    ("18.6%", "noise"),
    ("99.0%", "farfield"),
)
_DEGRADATIONS = ("babble", "reverb", "noise", "farfield", "codec", "clipping")


# How far from the number a degradation name still counts as being paired with
# it. A whole line is too wide: a sentence listing every degradation is not a
# claim about any one of them.
PAIRING_WINDOW = 35


def retention_pairing_errors(text):
    """Retention percentages named next to the wrong degradation.

    Until 2026-08-14 three documents read "18.3% at whisper-base reverb 0 dB".
    18.3% is babble; reverb is 12.0%, six points worse. The number and the
    condition were each defensible alone, which is why nobody caught it.
    """
    errors = []
    for number, line in enumerate(text.splitlines(), 1):
        if _DISCUSSING_NOT_ASSERTING.search(line):
            continue
        for value, correct in RETENTION_PAIRINGS:
            for match in re.finditer(re.escape(value), line):
                start = max(0, match.start() - PAIRING_WINDOW)
                window = line[start:match.end() + PAIRING_WINDOW].lower()
                # "far-field" and "far field" are the same degradation.
                window = re.sub(r"[-\s]", "", window)
                # A pairing CLAIM always names a condition, i.e. a degradation
                # together with its SNR ("reverb 0 dB"). Without the SNR the
                # word may just be English: "how many survive the noise" is not
                # a claim that this number belongs to the `noise` degradation.
                if not re.search(r"\d+db", window):
                    continue
                named = [d for d in _DEGRADATIONS if d in window]
                if named and correct not in named:
                    errors.append(
                        "line %d: %s is %s retention, but is written beside %s — %s"
                        % (number, value, correct, "/".join(named),
                           line.strip()[:80])
                    )
    return errors


# --- CLI --------------------------------------------------------------------

_failures = []


def report(document, kind, detail):
    _failures.append((document, kind, detail))
    print("  [BAD] %-28s %s" % (kind, detail))


def main(argv=None):
    print("=" * 84)
    print("VERIFYING THE PROJECT DOCUMENTS AGAINST REALITY AND EACH OTHER")
    print("=" * 84)

    texts = {}
    for name in DOCUMENTS:
        path = REPO / name
        if not path.exists():
            report(name, "missing document", "%s does not exist" % name)
            continue
        texts[name] = path.read_text()

    for name, text in texts.items():
        print("\n%s" % name)
        before = len(_failures)
        for path in dead_paths(text):
            report(name, "dead file reference", "`%s` does not exist" % path)
        for hit in forbidden_claims(text):
            report(name, "forbidden claim", hit)
        for hit in trap_model_ids(text):
            report(name, "trap model id", hit)
        for hit in retention_pairing_errors(text):
            report(name, "retention pairing", hit)
        if len(_failures) == before:
            print("  [ok ] no dead paths, forbidden claims, trap ids or bad pairings")

    print("\ncross-document")
    actual = actual_test_count()
    stated_anywhere = any(stated_test_counts(t) for t in texts.values())
    if not stated_anywhere:
        print("  [ok ] no document hard-codes a test count "
              "(it lives in STATUS.md's generated half, which is recounted)")
    elif actual is None:
        report("-", "test count", "could not collect the suite")
    else:
        for name, text in texts.items():
            for stated in stated_test_counts(text):
                if stated != actual:
                    report(name, "stale test count",
                           "says %d tests, the suite collects %d" % (stated, actual))
                else:
                    print("  [ok ] %-22s states %d tests, suite collects %d"
                          % (name, stated, actual))

    print("\nsection addressing (§N -> file)")
    problems = section_map_problems()
    for problem in problems:
        report("THESIS_TRACKER.md", "broken section map", problem)
    stale = stale_section_pointers()
    for pointer in stale:
        report("(cross-file)", "stale section pointer", pointer)
    if not problems and not stale:
        print("  [ok ] every §N cited in the repo resolves to exactly one "
              "document, and no file is paired with a section it does not own")

    print("\ndocument coverage")
    uncovered = uncovered_documents()
    for document in uncovered:
        report(document, "unclassified document",
               "in neither DOCUMENTS nor EXCLUDED_DOCUMENTS — decide whether "
               "it is checked, and say so in scripts/verify_docs.py")
    if not uncovered:
        print("  [ok ] every project .md is either checked (%d) or excluded "
              "with a reason (%d)" % (len(DOCUMENTS), len(EXCLUDED_DOCUMENTS)))

    print("\nmodel ids")
    unclassified = unclassified_model_ids(texts)
    for hit in unclassified:
        report("(cross-file)", "unclassified model id", hit)
    if not unclassified:
        print("  [ok ] every model id named in a document is classified — "
              "served, trap, local, or rejected with a reason")

    print("\nsteps one document blocks and another clears")
    conflicts = step_blocking_conflicts(texts)
    for conflict in conflicts:
        report("(cross-file)", "step contradiction", conflict)
    if not conflicts:
        print("  [ok ] no step is declared blocked in one live document and "
              "clear in another")

    # Deliberately BEFORE the counts below. A sync copy makes the counters lie,
    # and the lie surfaces as "STATUS.md is stale" — an accusation against the
    # one file that was right. Read the cause first.
    print("\nstray copies that would make the counts lie")
    duplicates = sync_duplicate_files()
    for duplicate in duplicates:
        report("(filesystem)", "sync duplicate", duplicate)
    if not duplicates:
        print("  [ok ] no `X 2.ext` copies sitting beside the `X.ext` they "
              "were copied from")

    print("\nSTATUS.md vs the files")
    try:
        import generate_status
        stated = status_numbers(texts.get("STATUS.md", ""))
        actual = generate_status.collect_facts()
        drift = status_drift(stated, actual)
        for item in drift:
            report("STATUS.md", "not regenerated", item)
        if not drift:
            print("  [ok ] STATUS.md's counted half matches the files "
                  "(%d cells, %s sweep rows, next = Step %s)"
                  % (actual["cells"], "{:,}".format(actual["sweep_rows"]),
                     actual["next_step"]))
    except Exception as exc:
        report("STATUS.md", "could not be checked", str(exc)[:120])

    print("\n" + "=" * 84)
    if _failures:
        print("%d claim(s) in the documents are stale or wrong:\n" % len(_failures))
        for document, kind, detail in _failures:
            print("  %-22s %-24s %s" % (document, kind, detail))
        print("\nFix the document, or the code — do not fix the checker.")
        return 1
    print("Every checkable claim in all %d documents holds." % len(texts))
    return 0



# --- the section addressing scheme ------------------------------------------

# `| §8 | metrics and statistics | `docs/design/METRICS.md` |` — the row form of
# the map in THESIS_TRACKER.md. Struck-through rows (~~§0a~~) record a dissolved
# section and own no file.
_MAP_ROW = re.compile(
    r"^\|\s*\**§(\d+[a-z]?)\**\s*\|[^|]*\|\s*\**`([^`]+)`\**\s*\|", re.M)
# `| ~~§0a~~ | *dissolved* — state is now STATUS.md | — |` — a section that no
# longer exists but is still cited in older documents and commit messages. The
# map must keep answering for it, or every one of those citations dangles.
_MAP_DISSOLVED = re.compile(r"^\|\s*~~§(\d+[a-z]?)~~", re.M)
_SECTION_CITATION = re.compile(r"§(\d+[a-z]?)")
_SECTION_HEADING = re.compile(r"^##\s*(\d+[a-z]?)\.", re.M)

# Where a §N citation may appear. Everything a reader or an agent might follow.
_CITING_SUFFIXES = (".md", ".py", ".sh")
_SKIP_DIRS = {"venv", "venv312", ".git", "models", "corrupted", "node_modules",
              "__pycache__", ".pytest_cache", "results", "dataset", "logs",
              # Test fixtures cite non-existent section numbers ON PURPOSE,
              # to prove the checker catches them. Not claims about the project.
              "tests"}


def section_map(repo=REPO):
    """{section number: the file that owns it}, read from THESIS_TRACKER.md.

    Section numbers are this project's permanent addressing scheme: 355
    citations of them exist across 35 files, 20 of them Python. A section may
    move file; the number does not change. This table is the indirection that
    makes that true, so it is the one thing that must stay correct.
    """
    tracker = Path(repo) / "THESIS_TRACKER.md"
    if not tracker.exists():
        return {}
    return {m.group(1): m.group(2) for m in _MAP_ROW.finditer(tracker.read_text())}


def dissolved_sections(repo=REPO):
    """Sections the map records as gone, with what replaced them."""
    tracker = Path(repo) / "THESIS_TRACKER.md"
    if not tracker.exists():
        return set()
    return {m.group(1) for m in _MAP_DISSOLVED.finditer(tracker.read_text())}


def _citing_files(repo):
    repo = Path(repo)
    for path in sorted(repo.rglob("*")):
        if not path.is_file() or path.suffix not in _CITING_SUFFIXES:
            continue
        if _SKIP_DIRS & set(path.relative_to(repo).parts):
            continue
        yield path


def section_map_problems(repo=REPO):
    """Every way the §N addressing scheme can be broken.

    A dangling §N is worse than a broken link: the citation still reads as
    authoritative, so an agent follows it, finds nothing, and either invents
    the content or silently drops the constraint it referred to.
    """
    repo = Path(repo)
    mapping = section_map(repo)
    problems = []

    # ONLY the files the map names can own a thesis section. A session record
    # and a runbook have their own `## 1.`, `## 2.` headings; reading those as
    # claims on §1 and §2 produced 11 false alarms on the first run and buried
    # the one real finding underneath them.
    owners = {}
    for declared in set(mapping.values()):
        path = repo / declared
        if not path.exists():
            continue
        for m in _SECTION_HEADING.finditer(path.read_text()):
            owners.setdefault(m.group(1), []).append(declared)

    for section, declared in sorted(mapping.items()):
        holders = owners.get(section, [])
        if declared not in holders:
            problems.append(
                "§%s: the map sends readers to `%s`, which does not contain a "
                "`## %s.` heading (found in: %s)"
                % (section, declared, section,
                   ", ".join(holders) if holders else "no mapped file"))
        elif len(holders) > 1:
            problems.append(
                "§%s is defined in more than one mapped document (%s) — a "
                "citation cannot be resolved unambiguously"
                % (section, ", ".join(holders)))

    cited = set()
    for path in _citing_files(repo):
        for m in _SECTION_CITATION.finditer(path.read_text()):
            cited.add(m.group(1))
    known = set(mapping) | dissolved_sections(repo)
    for section in sorted(cited - known):
        problems.append(
            "§%s is cited in the repo but the map in THESIS_TRACKER.md does "
            "not say where it lives" % section)
    return problems


# --- STATUS.md must have been regenerated after the data changed -------------

_STATUS_PATTERNS = {
    "cells": re.compile(r"Cells complete\D*(\d+) of \d+"),
    "sweep_rows": re.compile(r"Sweep rows collected\s*\|\s*([\d,]+)"),
    "tests": re.compile(r"Test suite\s*\|\s*([\d,]+) tests"),
    "next_step": re.compile(r"Next step in .*?\|\s*\**Step (\d+)"),
}


def status_numbers(text):
    """The countable facts STATUS.md currently asserts."""
    found = {}
    for key, pattern in _STATUS_PATTERNS.items():
        match = pattern.search(text)
        if match:
            found[key] = int(match.group(1).replace(",", ""))
    return found


def status_drift(stated, actual):
    """Where STATUS.md and the files disagree.

    STATUS.md's countable half is generated, so a disagreement never means the
    file is wrong in the ordinary sense — it means the data moved and nobody
    re-ran `scripts/generate_status.py`. Same remedy either way, but the
    message should say which.
    """
    drift = []
    for key in sorted(set(stated) & set(actual)):
        if stated[key] != actual[key]:
            drift.append(
                "STATUS.md says %s = %s; the files say %s. Run "
                "`./venv/bin/python scripts/generate_status.py`."
                % (key, "{:,}".format(stated[key]), "{:,}".format(actual[key])))
    return drift



# --- a file named together with a section it does not own --------------------

# "`docs/design/METRICS.md` §8" / "docs/design/METRICS.md §8" — an explicit pairing
# of a FILE with a SECTION. A bare nickname ("tracker §13") is not a pairing:
# it means "look §13 up in the map", which is correct and is how most of this
# repo is written.
# The file and the section must be ADJACENT — "`METRICS.md` §8", "METRICS.md,
# §8". Merely nearby is not a pairing: "...the log is `search-log.md`. See §17
# for the anchors." is two separate references that happen to sit together.
_FILE_SECTION = re.compile(r"`?([\w./-]+\.md)`?[,;:]?[ ]{0,2}§(\d+[a-z]?)")


def stale_section_pointers(repo=REPO):
    """Citations that name a file AND a section the file does not hold.

    Invisible to the plain §N check, because the section does resolve — just
    not where the sentence sends you. After the 2026-08-18 split, every
    "`docs/design/METRICS.md` §8" in the repo became one of these: authoritative-
    sounding, and pointing at a 49-line map that has not held §8 since.
    """
    repo = Path(repo)
    mapping = section_map(repo)
    if not mapping:
        return []

    problems = []
    for path in _citing_files(repo):
        rel = str(path.relative_to(repo))
        # Dated snapshots. A spec recording that §1-§10 USED TO live in
        # THESIS_TRACKER.md is correct precisely because it names the old
        # pairing; flagging it would stop the project documenting its history.
        if rel.startswith("docs/superpowers/"):
            continue
        for match in _FILE_SECTION.finditer(path.read_text()):
            named_file, section = match.group(1), match.group(2)
            owner = mapping.get(section)
            if owner is None or named_file == owner:
                continue
            # The map itself is allowed to name every section.
            if rel == "THESIS_TRACKER.md":
                continue
            if named_file.endswith(Path(owner).name):
                continue
            problems.append(
                "%s pairs `%s` with §%s, which lives in `%s`"
                % (rel, named_file, section, owner))
    return problems


if __name__ == "__main__":
    sys.exit(main())
