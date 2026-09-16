#!/usr/bin/env python
"""Check every collected row against §10, the output schema — and say which
document each rule comes from.

Pure checking. Reads JSONL, writes nothing, repairs nothing.

WHY THIS EXISTS
---------------
`EXECUTION_TODO.md` Step 6 carries an open box, "Validate the schema before
merging". Nothing in the repo could tick it: `ls scripts/ | grep -iE
"schema|validate"` returned nothing (run 2026-08-28, exit 1). Meanwhile the
sweep finished — 116 files and 86,944 rows over `result_sets.sweep_files()`
(counted 2026-08-28 by reading them) — and not one of those rows had ever been
compared against the schema every downstream script assumes.

The gap that makes this worth a file of its own is not "a field might be
missing". It is that §10 states *relations* — `ees` is `tsa == 1 AND pf >= 0.5`,
`pf` is `pf_n_matched / pf_n_gold`, `full_hallucination` requires WER >= 1 on a
non-empty hypothesis, `phonetic_distance` exists only where a critical-slot
substitution does — and a row can satisfy every type while contradicting all of
them. A scoring stage that silently did not run leaves exactly that signature.

THE RULE THIS FILE FOLLOWS, AND WHY IT MATTERS MORE THAN THE RULES IT CHECKS
---------------------------------------------------------------------------
**Every violation names the document it comes from.** §10 is a short block of
annotated JSON; it does not enumerate the degradations, it does not list the
Peng TSH/TCH labels, and it never says when those labels must be non-null.
Those rules are real, but they live in §4, §8 and `categorize_failures.py`. A
validator that printed them as "§10 violations" would be inventing schema, and
the first person to check would stop trusting the whole report. So each
`Violation` carries a `source`, `--schema-only` keeps just the §10 ones, and
everything §10 leaves genuinely open is printed under "§10 DOES NOT STATE" with
what the data actually contains — surfaced for a human, never guessed at.

THE OMNI ARM IS NOT BROKEN DATA
-------------------------------
Omni rows carry `transcription = None`, `wer`/`cer`/`slot_error_rate` null,
`asr_model = None`, `error_categories = []` and `full_hallucination = None`.
That is the pathway — audio in, intent out, no transcript ever produced — not
absent data, and §10 says so field by field. The checks are keyed on the row's
own `pathway`, and `tests/test_validate_schema.py` pins it, because a validator
that reports 15,808 legitimate rows as defects is worse than no validator.

USAGE
-----
    ./venv/bin/python scripts/validate_schema.py                 # the sweep
    ./venv/bin/python scripts/validate_schema.py --schema-only   # §10 rules only
    ./venv/bin/python scripts/validate_schema.py results/x.jsonl # named files

With no arguments it validates `result_sets.sweep_files()` — never a bare
`results/*_scored.jsonl` glob, which is how the 200-row H2a pilot walked into
the sweep counts on 2026-08-17 (`CLAUDE.md`, `scripts/result_sets.py`).

Exit 0 = every row conforms. 1 = at least one violation (or an unreadable file).
"""

import argparse
import json
import sys
from collections import Counter, namedtuple
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import result_sets  # noqa: E402
from batch_evaluate import TARGET_SCENARIOS, VALID_INTENTS  # noqa: E402

# ─── where each rule comes from ──────────────────────────────────────────────
# A rule with no source is not a rule, it is an opinion. These strings are
# printed next to every violation so a reader can go and read the sentence.
SECTION_10 = "§10 docs/design/SCHEMA.md"
SECTION_8 = "§8 docs/design/METRICS.md"
SECTION_4 = "§4 docs/design/DESIGN.md"
SRC_ONTOLOGY = "scripts/batch_evaluate.py (VALID_INTENTS, TARGET_SCENARIOS)"
SRC_PENG = "scripts/categorize_failures.py (Peng TSH/TCH taxonomy, §2)"

KNOWN_SOURCES = (SECTION_10, SECTION_8, SECTION_4, SRC_ONTOLOGY, SRC_PENG)

# ─── §10's row, field for field ──────────────────────────────────────────────
# The order is §10's own. Every field in the §10 block is required on every
# row; the nullable ones are exactly those §10 annotates as null-bearing.
SCHEMA_FIELDS = (
    "slurp_id", "audio_file", "scenario", "gold_sentence", "gold_intent",
    "gold_parameters",
    "pathway", "model_family",
    "asr_model", "degradation", "snr_db", "rng_seed",
    "transcription", "wer", "cer", "slot_error_rate", "error_categories",
    "phonetic_distance", "full_hallucination",
    "llm_model", "llm_prompt_variant", "predicted_intent_raw",
    "predicted_intent", "predicted_parameters",
    "tsa", "pf", "pf_n_gold", "pf_n_matched", "ees", "ees_strict",
    "tsh_category", "tch_category", "notes",
)

FIELD_TYPES = {
    "slurp_id": int, "audio_file": str, "scenario": str, "gold_sentence": str,
    "gold_intent": str, "gold_parameters": dict,
    "pathway": str, "model_family": str,
    "asr_model": str, "degradation": str, "snr_db": int, "rng_seed": int,
    "transcription": str, "wer": float, "cer": float, "slot_error_rate": float,
    "error_categories": list, "phonetic_distance": float,
    "full_hallucination": bool,
    "llm_model": str, "llm_prompt_variant": str, "predicted_intent_raw": str,
    "predicted_intent": str, "predicted_parameters": dict,
    "tsa": int, "pf": float, "pf_n_gold": int, "pf_n_matched": int,
    "ees": int, "ees_strict": int,
    "tsh_category": str, "tch_category": str, "notes": str,
}

# Nullable because §10 annotates them so: "null for omni", "null for clean",
# "null for omni and for rows without critical-slot substitutions". `pf` is
# nullable on §8's rule ("PF = None if the utterance has no gold entities");
# `tsh_category`/`tch_category`/`notes` are null in §10's own example row.
# `tsa` is deliberately NOT here: the `tsa = None` refuse convention was
# normalised out of the data on 2026-08-14 (EXECUTION_TODO Step 4d), so a null
# one today means a superseded writer produced the row.
NULLABLE = frozenset((
    "asr_model", "snr_db", "transcription", "wer", "cer", "slot_error_rate",
    "phonetic_distance", "full_hallucination", "pf",
    "tsh_category", "tch_category", "notes",
))

# Enumerations §10 itself gives, inline, as `"cascade" | "omni"`.
PATHWAYS = ("cascade", "omni")
MODEL_FAMILIES = ("qwen", "google")

# Enumerations §10 does NOT give. §10 writes `"degradation": "babble", // or
# "clean" for the control` and stops, so the level set comes from §4's factorial
# table (6 degradations x 3 SNR + clean). Kept as a literal list, with its
# citation, for the same reason `result_sets.py` keeps one: a wildcard here
# would readmit exactly what the check exists to catch.
DEGRADATIONS = ("clean", "noise", "reverb", "farfield", "codec", "clipping",
                "babble")
SNR_LEVELS = (20, 10, 0)

# The Peng taxonomy labels, from the writer that stamps them.
TSH_CATEGORIES = ("missing", "unknown", "hallucinated", "wrong")
TCH_CATEGORIES = ("schema_mismatch", "fabricated", "semantic_drift")

# The chunk shape §10 spells out for `error_categories`.
ERROR_CHUNK_KEYS = frozenset(("type", "error_subtype", "slot_position", "ref",
                              "hyp", "entity_type"))

# Added by Step 4c (EXECUTION_TODO), verified there as a strict superset of the
# then-current schema, and never written into §10's block. Present on the rows
# the GCP drivers wrote and absent on the local ones. Reported as drift between
# §10 and the data, not as a bad row: they are additive provenance, and the
# decision to add them is on the record.
ADDITIVE_FIELDS = ("llm_model_served", "llm_thinking_request")

PF_THRESHOLD = 0.5   # §10: `ees` is "tsa==1 AND pf >= 0.5".

Violation = namedtuple("Violation", "rule source message identity")
Identity = namedtuple(
    "Identity",
    "file line slurp_id pathway asr_model degradation snr_db llm_model")


def row_identity(row, path=None, line=None):
    """Enough of a row to find it again: file, line, and the cell it belongs to."""
    get = row.get if isinstance(row, dict) else (lambda k: None)
    return Identity(
        file=(Path(path).name if path is not None else None),
        line=line,
        slurp_id=get("slurp_id"),
        pathway=get("pathway"),
        asr_model=get("asr_model"),
        degradation=get("degradation"),
        snr_db=get("snr_db"),
        llm_model=get("llm_model"),
    )


def _is(value, expected):
    """Type test that refuses bool where int is asked for.

    `isinstance(True, int)` is True in Python, so without this a row carrying
    `tsa: true` would pass an int check and then be summed as 1.
    """
    if expected is bool:
        return isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is float:
        return isinstance(value, float) or (isinstance(value, int)
                                            and not isinstance(value, bool))
    return isinstance(value, expected)


def _num(row, field):
    """The field's value if it is a usable number, else None (so a type error
    reported once is not re-reported as a nonsensical arithmetic failure)."""
    v = row.get(field)
    return v if _is(v, float) else None


def validate_row(row, identity=None, schema_only=False):
    """Every violation in one row, each naming the document it comes from.

    `schema_only=True` keeps only the rules §10 states itself.
    """
    out = []

    def bad(rule, source, message):
        out.append(Violation(rule, source, message, identity))

    if not isinstance(row, dict):
        bad("not_an_object", SECTION_10,
            "§10 defines a per-utterance JSON object; this line is a %s"
            % type(row).__name__)
        return out

    # 1. presence and type -----------------------------------------------------
    typed_ok = set()
    for field in SCHEMA_FIELDS:
        if field not in row:
            bad("missing_field", SECTION_10,
                "`%s` is in §10's row and not in this one" % field)
            continue
        value = row[field]
        if value is None:
            if field not in NULLABLE:
                bad("null_not_allowed", SECTION_10,
                    "`%s` is null; §10 never marks it null-bearing" % field)
            continue
        if not _is(value, FIELD_TYPES[field]):
            bad("wrong_type", SECTION_10,
                "`%s` is %s; §10 shows %s" % (field, type(value).__name__,
                                              FIELD_TYPES[field].__name__))
            continue
        typed_ok.add(field)

    def ok(*fields):
        return all(f in typed_ok for f in fields)

    # 2. enumerations ----------------------------------------------------------
    if ok("pathway") and row["pathway"] not in PATHWAYS:
        bad("bad_enum", SECTION_10,
            "`pathway` is %r; §10 gives \"cascade\" | \"omni\"" % row["pathway"])
    if ok("model_family") and row["model_family"] not in MODEL_FAMILIES:
        bad("bad_enum", SECTION_10,
            "`model_family` is %r; §10 gives \"qwen\" | \"google\""
            % row["model_family"])
    if ok("degradation") and row["degradation"] not in DEGRADATIONS:
        bad("bad_enum", SECTION_4,
            "`degradation` is %r; §4's factorial gives clean + %s"
            % (row["degradation"], ", ".join(DEGRADATIONS[1:])))
    if ok("snr_db") and row["snr_db"] not in SNR_LEVELS:
        bad("bad_enum", SECTION_4,
            "`snr_db` is %r; §4 gives 20 / 10 / 0 dB" % row["snr_db"])
    if ok("gold_intent") and row["gold_intent"] not in VALID_INTENTS:
        bad("bad_enum", SRC_ONTOLOGY,
            "`gold_intent` is %r, outside the 9-intent ontology"
            % row["gold_intent"])
    if ok("scenario") and row["scenario"] not in TARGET_SCENARIOS:
        bad("bad_enum", SRC_ONTOLOGY,
            "`scenario` is %r; the dataset holds %s"
            % (row["scenario"], "/".join(TARGET_SCENARIOS)))
    if row.get("tsh_category") is not None and ok("tsh_category") \
            and row["tsh_category"] not in TSH_CATEGORIES:
        bad("bad_enum", SRC_PENG,
            "`tsh_category` is %r; the TSH labels are %s"
            % (row["tsh_category"], ", ".join(TSH_CATEGORIES)))
    if row.get("tch_category") is not None and ok("tch_category") \
            and row["tch_category"] not in TCH_CATEGORIES:
        bad("bad_enum", SRC_PENG,
            "`tch_category` is %r; the TCH labels are %s"
            % (row["tch_category"], ", ".join(TCH_CATEGORIES)))

    # 3. the two pathways ------------------------------------------------------
    pathway = row.get("pathway")
    if pathway == "omni":
        for field in ("asr_model", "transcription", "wer", "cer",
                      "slot_error_rate", "phonetic_distance",
                      "full_hallucination"):
            if row.get(field) is not None:
                bad("omni_field_not_null", SECTION_10,
                    "`%s` is %r on an omni row; §10 marks it null for omni "
                    "(the pathway has no transcript)" % (field, row[field]))
        if row.get("error_categories") not in (None, []):
            bad("omni_error_categories_not_empty", SECTION_10,
                "`error_categories` is non-empty on an omni row; §10 marks it "
                "[] for omni")
    elif pathway == "cascade":
        if row.get("asr_model") is None:
            bad("cascade_asr_model_null", SECTION_10,
                "`asr_model` is null on a cascade row; §10 marks it null for "
                "omni only, and the cascade row is the one with an ASR stage")
        if row.get("transcription") is None:
            bad("cascade_transcript_field_null", SECTION_10,
                "`transcription` is null on a cascade row; §10 marks it null "
                "for omni only. An EMPTY transcript is a different thing and "
                "is a scored outcome (NO_TRANSCRIPTION, Step 4c)")
        for field in ("wer", "cer", "full_hallucination"):
            if row.get(field) is None:
                bad("cascade_metric_null", SECTION_10,
                    "`%s` is null on a cascade row; §10 marks it null for omni "
                    "only" % field)

    # 4. clean-row coding ------------------------------------------------------
    if ok("degradation"):
        if row["degradation"] == "clean" and row.get("snr_db") is not None:
            bad("clean_row_has_snr", SECTION_10,
                "`snr_db` is %r on a clean row; §10 marks it null for clean, "
                "and §8's nested dummies depend on it" % row["snr_db"])
        if row["degradation"] != "clean" and row.get("snr_db") is None:
            bad("degraded_row_has_no_snr", SECTION_10,
                "`snr_db` is null on a %s row; §10 marks it null for clean only"
                % row["degradation"])

    # 5. error_categories chunk shape -----------------------------------------
    if ok("error_categories"):
        for chunk in row["error_categories"]:
            if not isinstance(chunk, dict) or set(chunk) != ERROR_CHUNK_KEYS:
                bad("error_chunk_keys", SECTION_10,
                    "an `error_categories` entry has keys %s; §10 gives "
                    "{type, error_subtype, slot_position, ref, hyp, entity_type}"
                    % (sorted(chunk) if isinstance(chunk, dict)
                       else type(chunk).__name__))
                break

    # 6. phonetic_distance: only where a critical substitution exists ----------
    pd = row.get("phonetic_distance")
    if pd is not None and ok("phonetic_distance"):
        if not (0.0 <= pd <= 1.0):
            bad("value_out_of_range", SECTION_8,
                "`phonetic_distance` is %r; §8 defines it normalized to [0,1]" % pd)
        chunks = row.get("error_categories") or []
        has_critical_sub = any(
            isinstance(c, dict) and c.get("type") == "substitute"
            and c.get("slot_position") == "critical" for c in chunks)
        if not has_critical_sub:
            bad("phonetic_distance_unsupported", SECTION_10,
                "`phonetic_distance` is %r with no critical-slot substitution "
                "in `error_categories`; §10 marks it null for omni AND for "
                "rows without critical-slot substitutions" % pd)

    # 7. full_hallucination's own definition ----------------------------------
    if row.get("full_hallucination") is True:
        wer = _num(row, "wer")
        if wer is None or wer < 1.0:
            bad("full_hallucination_definition", SECTION_10,
                "`full_hallucination` is true with wer=%r; §10 defines it as "
                "WER>=1 on a non-empty hypothesis" % row.get("wer"))
        if not (row.get("transcription") or "").strip():
            bad("full_hallucination_definition", SECTION_10,
                "`full_hallucination` is true on an empty transcript; §8: an "
                "empty transcription is a total deletion, not a hallucination")

    # 8. the derived metrics §10 states as formulas ----------------------------
    pf, n_gold, n_matched = row.get("pf"), row.get("pf_n_gold"), row.get("pf_n_matched")
    if ok("pf_n_gold", "pf_n_matched"):
        if n_matched > n_gold or n_matched < 0 or n_gold < 0:
            bad("pf_counters_impossible", SECTION_8,
                "pf_n_matched=%r of pf_n_gold=%r; §8 defines PF as recall over "
                "gold entities" % (n_matched, n_gold))
        elif (pf is None) != (n_gold == 0):
            bad("pf_null_mismatch", SECTION_8,
                "pf=%r with pf_n_gold=%r; §8: PF is None if and only if the "
                "utterance has no gold entities" % (pf, n_gold))
        elif pf is not None and ok("pf") and n_gold > 0:
            if not (0.0 <= pf <= 1.0):
                bad("value_out_of_range", SECTION_8,
                    "`pf` is %r; a recall lies in [0,1]" % pf)
            elif abs(pf - n_matched / float(n_gold)) > 1e-9:
                bad("pf_not_recall", SECTION_10,
                    "pf=%r but pf_n_matched/pf_n_gold = %d/%d; §10's row "
                    "annotates pf as exactly that recall"
                    % (pf, n_matched, n_gold))

    if ok("tsa") and row["tsa"] not in (0, 1):
        bad("value_out_of_range", SECTION_8,
            "`tsa` is %r; §8 defines the layer metrics as binary 0/1" % row["tsa"])
    for field, threshold, rule in (("ees", PF_THRESHOLD, "ees_formula"),
                                   ("ees_strict", 1.0, "ees_strict_formula")):
        if not ok("tsa", field) or row[field] not in (0, 1):
            if ok(field) and row[field] not in (0, 1):
                bad("value_out_of_range", SECTION_8,
                    "`%s` is %r; §8 defines EES as binary 0/1" % (field, row[field]))
            continue
        if pf is None:
            # §8: "If there are no gold entities, EES = TSA."
            expected = row["tsa"]
        elif not ok("pf"):
            continue
        elif field == "ees":
            expected = 1 if (row["tsa"] == 1 and pf >= threshold) else 0
        else:
            expected = 1 if (row["tsa"] == 1 and pf == threshold) else 0
        if row[field] != expected:
            bad(rule, SECTION_10,
                "`%s` is %d; §10 defines it as tsa==1 AND pf %s %s — here "
                "tsa=%r, pf=%r, so it should be %d"
                % (field, row[field], ">=" if field == "ees" else "==",
                   threshold, row["tsa"], pf, expected))

    # 9. rules that come from §4, not from §10 ---------------------------------
    #    §4's dependent-variable list: "TSH / TCH category if failure".
    if ok("tsa") and row["tsa"] == 0 and row.get("tsh_category") is None:
        bad("tsh_category_missing", SECTION_4,
            "tsa=0 with no `tsh_category`; §4 lists the TSH label as a "
            "dependent variable of every failure, and categorize_failures.py "
            "stamps it. A whole arm missing it means that stage never ran")
    if ok("tsa", "ees") and row["tsa"] == 1 and row["ees"] == 0 \
            and row.get("tch_category") is None:
        bad("tch_category_missing", SECTION_4,
            "tsa=1 and ees=0 with no `tch_category`; §4 lists the TCH label as "
            "a dependent variable of a parameter-layer failure")
    if ok("tsa") and row["tsa"] == 1 and row.get("tsh_category") is not None:
        bad("tsh_category_on_a_success", SRC_PENG,
            "tsa=1 carries tsh_category=%r; the TSH label is defined for "
            "tsa == 0" % row["tsh_category"])

    if schema_only:
        return [v for v in out if v.source == SECTION_10]
    return out


class Report(object):
    """What the run found. Counts, a few example identities, and the
    observations that feed the "§10 does not state" section."""

    def __init__(self):
        self.n_files = 0
        self.n_rows = 0
        self.n_bad_rows = 0
        self.rule_counts = {}
        self.rule_sources = {}
        self.rule_messages = {}
        self.rule_files = {}
        self.examples = {}
        self.extra_fields = Counter()
        self.unreadable = []
        self.observed = {
            "notes_null": 0,
            "notes_empty": 0,
            "cascade_ser_null": 0,
            "cascade_ser_null_with_gold_params": 0,
            "pf_null": 0,
            "pathway": Counter(),
            "degradation": Counter(),
            "tsh_category": Counter(),
            "tch_category": Counter(),
        }

    def add(self, violation, examples_per_rule):
        rule = violation.rule
        self.rule_counts[rule] = self.rule_counts.get(rule, 0) + 1
        self.rule_sources[rule] = violation.source
        self.rule_messages.setdefault(rule, violation.message)
        ident = violation.identity
        if ident is not None:
            self.rule_files.setdefault(rule, Counter())[ident.file] += 1
            shown = self.examples.setdefault(rule, [])
            if len(shown) < examples_per_rule:
                shown.append(ident)

    def observe(self, row):
        o = self.observed
        o["pathway"][row.get("pathway")] += 1
        o["degradation"][row.get("degradation")] += 1
        o["tsh_category"][row.get("tsh_category")] += 1
        o["tch_category"][row.get("tch_category")] += 1
        if row.get("notes") is None:
            o["notes_null"] += 1
        elif row.get("notes") == "":
            o["notes_empty"] += 1
        if row.get("pf") is None:
            o["pf_null"] += 1
        if row.get("pathway") == "cascade" and row.get("slot_error_rate") is None:
            o["cascade_ser_null"] += 1
            if row.get("gold_parameters"):
                o["cascade_ser_null_with_gold_params"] += 1
        for field in row:
            if field not in FIELD_TYPES:
                self.extra_fields[field] += 1

    @property
    def n_violations(self):
        return sum(self.rule_counts.values())


def validate_files(paths, schema_only=False, examples_per_rule=3):
    """Read every file, validate every row, and remember where the bad ones are.

    Opens each path read-only and never writes: `CLAUDE.md` forbids modifying
    anything under `results/`, and a validator with a repair mode is one
    mis-typed flag away from destroying days of compute.
    """
    report = Report()
    for path in paths:
        path = Path(path)
        report.n_files += 1
        seen_keys = {}
        try:
            handle = path.open("r")
        except IOError as exc:
            report.unreadable.append((str(path), str(exc)))
            report.add(Violation("unreadable_file", SECTION_10, str(exc),
                                 Identity(path.name, None, None, None, None,
                                          None, None, None)),
                       examples_per_rule)
            continue
        with handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    report.add(Violation(
                        "unparseable_line", SECTION_10,
                        "§10's rows are one JSON object per line: %s" % exc,
                        Identity(path.name, lineno, None, None, None, None,
                                 None, None)), examples_per_rule)
                    continue
                report.n_rows += 1
                ident = row_identity(row, path, lineno)
                if isinstance(row, dict):
                    report.observe(row)
                    # §10's resume key: any script writing omni rows uses the
                    # pathway-aware 6-tuple. A repeat means the done-set failed
                    # and one cell contributed a row twice.
                    key = (row.get("slurp_id"), row.get("pathway"),
                           row.get("asr_model"), row.get("degradation"),
                           row.get("snr_db"), row.get("llm_model"))
                    first = seen_keys.get(key)
                    if first is not None:
                        report.add(Violation(
                            "duplicate_resume_key", SECTION_10,
                            "this row's §10 resume key (slurp_id, pathway, "
                            "asr_model, degradation, snr_db, llm_model) already "
                            "appeared on line %d of the same file" % first,
                            ident), examples_per_rule)
                    else:
                        seen_keys[key] = lineno
                violations = validate_row(row, ident, schema_only=schema_only)
                if violations:
                    report.n_bad_rows += 1
                for violation in violations:
                    report.add(violation, examples_per_rule)
    return report


# ─── what §10 leaves open, printed rather than guessed at ────────────────────
# Each entry: (the silence, what this validator did about it, an observation
# key or None). "An ambiguity you surface is worth more than a rule you invent."
AMBIGUITIES = (
    ("§10's example row shows `notes: \"\"` and the field is never described as "
     "null-bearing.",
     "Accepted null. Nothing downstream reads `notes` as a string without a "
     "guard, but §10 and the data disagree on the empty value.",
     "notes"),
    ("§10 annotates `slot_error_rate` \"null for omni\" and says nothing about "
     "a cascade row whose utterance has no gold slots.",
     "Accepted null on cascade rows. The observation below is what settles "
     "whether that is the rule the writers actually follow.",
     "cascade_ser"),
    ("§10 names only \"babble\" and \"clean\" for `degradation`, and gives no "
     "level set for `snr_db`.",
     "Checked against §4's factorial (6 degradations + clean; 20/10/0 dB), "
     "sourced as §4 and not as §10.",
     "degradation"),
    ("§10 lists `tsh_category` / `tch_category` as fields, with null in its "
     "example, and never says which labels are legal or when one is required.",
     "Labels checked against categorize_failures.py; the requirement itself "
     "comes from §4's dependent-variable list. Both sourced accordingly, and "
     "both dropped by --schema-only.",
     "peng"),
    ("§10 defines `full_hallucination` as WER>=1 AND a non-empty hypothesis AND "
     "zero content overlap between reference and hypothesis.",
     "The first two clauses are checked here. The third is NOT: it needs "
     "annotate_errors.py's content-token rule, and re-implementing it in a "
     "checker would compare the code against itself.",
     None),
    ("§10's block does not contain `llm_model_served` or `llm_thinking_request`.",
     "Reported as advisory drift, never as a bad row: Step 4c added them "
     "deliberately as additive provenance fields. §10 has not caught up.",
     "extra"),
    ("§10 does not say whether `pf` may be null; §8 does "
     "(\"PF = None if the utterance has no gold entities\").",
     "Used §8's rule, sourced as §8: pf is null if and only if pf_n_gold == 0.",
     "pf_null"),
)


def _fmt(n):
    return "{:,}".format(n)


def _ident_str(ident):
    bits = ["%s:%s" % (ident.file, ident.line)]
    for label, value in (("slurp_id", ident.slurp_id),
                         ("deg", ident.degradation),
                         ("snr", ident.snr_db),
                         ("asr", ident.asr_model),
                         ("llm", ident.llm_model)):
        if value is not None:
            bits.append("%s=%s" % (label, value))
    return "  ".join(bits)


def format_report(report, examples=3, schema_only=False):
    lines = []
    add = lines.append
    add("=" * 84)
    add("SCHEMA VALIDATION — every row against §10 (docs/design/SCHEMA.md)")
    add("=" * 84)
    add("files read              : %s" % _fmt(report.n_files))
    add("rows read               : %s" % _fmt(report.n_rows))
    add("rows with a violation   : %s" % _fmt(report.n_bad_rows))
    add("violations              : %s" % _fmt(report.n_violations))
    if schema_only:
        add("rule set                : §10 only (--schema-only)")
    add("")

    if report.rule_counts:
        add("VIOLATIONS BY RULE  (rule, the document it comes from, count)")
        add("-" * 84)
        for rule in sorted(report.rule_counts,
                           key=lambda r: -report.rule_counts[r]):
            add("  [BAD] %-30s %10s   %s"
                % (rule, _fmt(report.rule_counts[rule]),
                   report.rule_sources[rule]))
            add("        %s" % report.rule_messages[rule])
            files = report.rule_files.get(rule) or Counter()
            if files:
                top = ", ".join("%s (%s)" % (n, _fmt(c))
                                for n, c in files.most_common(3))
                add("        in %d file(s); top: %s" % (len(files), top))
            for ident in report.examples.get(rule, [])[:examples]:
                add("        e.g. %s" % _ident_str(ident))
            add("")
    else:
        add("No violation. Every row read conforms to §10 and to the rules "
            "borrowed from §4/§8 with their sources named.")
        add("")

    if report.extra_fields:
        add("ADVISORY — fields in the data that §10's block does not contain")
        add("-" * 84)
        for field, count in report.extra_fields.most_common():
            known = " (added by Step 4c, additive)" if field in ADDITIVE_FIELDS \
                else " (UNRECOGNISED — decide whether §10 should carry it)"
            add("  %-24s %10s rows%s" % (field, _fmt(count), known))
        add("")

    add("§10 DOES NOT STATE — surfaced, not guessed")
    add("-" * 84)
    obs = report.observed
    for i, (silence, action, key) in enumerate(AMBIGUITIES, 1):
        add("  %d. %s" % (i, silence))
        add("     -> %s" % action)
        if key == "notes":
            add("     observed: %s rows carry null, %s carry \"\""
                % (_fmt(obs["notes_null"]), _fmt(obs["notes_empty"])))
        elif key == "cascade_ser":
            add("     observed: %s cascade rows carry a null slot_error_rate; "
                "%s of those have non-empty gold_parameters"
                % (_fmt(obs["cascade_ser_null"]),
                   _fmt(obs["cascade_ser_null_with_gold_params"])))
        elif key == "degradation":
            add("     observed: %s" % ", ".join(
                "%s (%s)" % (k, _fmt(v))
                for k, v in sorted(obs["degradation"].items(),
                                   key=lambda kv: str(kv[0]))))
        elif key == "peng":
            add("     observed tsh: %s" % ", ".join(
                "%s (%s)" % (k, _fmt(v)) for k, v in
                sorted(obs["tsh_category"].items(), key=lambda kv: str(kv[0]))))
            add("     observed tch: %s" % ", ".join(
                "%s (%s)" % (k, _fmt(v)) for k, v in
                sorted(obs["tch_category"].items(), key=lambda kv: str(kv[0]))))
        elif key == "extra":
            add("     observed: %s" % (", ".join(
                "%s (%s rows)" % (f, _fmt(c))
                for f, c in report.extra_fields.most_common()) or "none"))
        elif key == "pf_null":
            add("     observed: %s rows carry pf = null" % _fmt(obs["pf_null"]))
        add("")

    add("-" * 84)
    if report.n_violations:
        add("RESULT: %s violation(s). Nothing was modified — this script only "
            "reads." % _fmt(report.n_violations))
        add("Before treating any of them as a data defect, read the source "
            "column: a rule from §4 or §8 may be describing a stage that never "
            "ran rather than a corrupt row.")
    else:
        add("RESULT: clean. %s rows in %s files conform."
            % (_fmt(report.n_rows), _fmt(report.n_files)))
    return "\n".join(lines)


def default_files():
    """The sweep, and only the sweep.

    `result_sets.sweep_files()` is the single place that says which files those
    are; a bare `results/*_scored.jsonl` glob is what let the 200-row H2a pilot
    into the counts on 2026-08-17 (`CLAUDE.md`).
    """
    return result_sets.sweep_files("results")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate collected result rows against §10 "
                    "(docs/design/SCHEMA.md). Reads only; never repairs.")
    parser.add_argument(
        "files", nargs="*",
        help="JSONL files to check. Default: result_sets.sweep_files().")
    parser.add_argument(
        "--schema-only", action="store_true",
        help="Keep only the rules §10 states itself; drop those borrowed from "
             "§4, §8 and the scoring code.")
    parser.add_argument(
        "--examples", type=int, default=3,
        help="Example row identities to print per rule (default 3).")
    args = parser.parse_args(argv)

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        try:
            paths = default_files()
        except result_sets.UnclassifiedResultFile as exc:
            print("REFUSING to guess which files are the sweep.\n\n%s\n\n"
                  "Classify it in scripts/result_sets.py, or name the files on "
                  "the command line." % exc)
            return 1
        if not paths:
            print("REFUSING to report on nothing: result_sets.sweep_files() "
                  "returned no file. Run from the repo root, or name the files "
                  "on the command line.")
            return 1

    report = validate_files(paths, schema_only=args.schema_only,
                            examples_per_rule=max(args.examples, 1))
    print(format_report(report, examples=args.examples,
                        schema_only=args.schema_only))
    return 1 if report.n_violations else 0


if __name__ == "__main__":
    sys.exit(main())
