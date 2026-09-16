"""Unit tests for scripts/validate_schema.py — §10 conformance of the result rows.

Written 2026-08-28, test-first. `EXECUTION_TODO.md` Step 6 has carried an open
box "Validate the schema before merging" since the box was written, and
`ls scripts/ | grep -iE "schema|validate"` returned nothing (run 2026-08-28,
exit 1): 86,944 rows across 116 files had never been checked against §10.

Two properties dominate this file, because both are ways the validator itself
could be wrong in a way that costs more than the gap it closes:

1. **The omni arm is not broken data.** Omni rows carry `transcription = None`,
   `wer`/`cer` null and `asr_model = None` *by construction* — the pathway is
   audio-in/intent-out. A validator that flags them destroys the credibility of
   every other line it prints. `tests/test_gain_sign_and_empty_transcription.py`
   pins the same property from the data side.
2. **It reports, it never repairs.** `CLAUDE.md` forbids modifying anything under
   `results/`. `test_validating_never_touches_the_input_file` holds the bytes.

The fixtures below are synthetic on purpose: a test that reads `results/` would
be slow, and would go red for a *data* defect rather than a *validator* defect —
the two must stay distinguishable.
"""
import json

import pytest

import validate_schema as vs


# ─── fixtures: one conforming row per pathway ────────────────────────────────

def cascade_row(**over):
    """A §10-conforming cascade row. Field-for-field the §10 example."""
    row = {
        "slurp_id": 6925,
        "audio_file": "audio-1501414241-headset.flac",
        "scenario": "calendar",
        "gold_sentence": "is there any program for tomorrow evening",
        "gold_intent": "calendar_query",
        "gold_parameters": {"date": "tomorrow", "time_of_day": "evening"},
        "pathway": "cascade",
        "model_family": "qwen",
        "asr_model": "mlx-community/whisper-large-v3-turbo",
        "degradation": "babble",
        "snr_db": 10,
        "rng_seed": 42,
        "transcription": "is there any program for tomorrow evening",
        "wer": 0.0,
        "cer": 0.0,
        "slot_error_rate": 0.0,
        "error_categories": [],
        "phonetic_distance": None,
        "full_hallucination": False,
        "llm_model": "mlx-community/Qwen3.5-9B-MLX-8bit",
        "llm_prompt_variant": "primary_9intent_9shot",
        "predicted_intent_raw": "calendar_query",
        "predicted_intent": "calendar_query",
        "predicted_parameters": {"date": "tomorrow"},
        "tsa": 1,
        "pf": 0.5,
        "pf_n_gold": 2,
        "pf_n_matched": 1,
        "ees": 1,
        "ees_strict": 0,
        "tsh_category": None,
        "tch_category": None,
        "notes": "",
    }
    row.update(over)
    return row


def omni_row(**over):
    """A §10-conforming omni row: no transcript, no ASR, no WER — by design."""
    row = cascade_row()
    row.update({
        "pathway": "omni",
        "asr_model": None,
        "transcription": None,
        "wer": None,
        "cer": None,
        "slot_error_rate": None,
        "error_categories": [],
        "phonetic_distance": None,
        "full_hallucination": None,
        "llm_model": "cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit",
    })
    row.update(over)
    return row


def rules(violations):
    return sorted(v.rule for v in violations)


def write_jsonl(path, rows):
    with open(str(path), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


# ─── the two pathways ────────────────────────────────────────────────────────

def test_a_conforming_cascade_row_produces_no_violation():
    assert vs.validate_row(cascade_row()) == []


def test_the_omni_arm_is_not_a_violation():
    """The single most important test here. Omni rows legitimately carry
    transcription=None, wer/cer=None, asr_model=None, error_categories=[] and
    full_hallucination=None. On 2026-08-27 the omni sweep landed and a bare
    read of the cascade invariant called all 15,808 of its rows a bug."""
    assert vs.validate_row(omni_row()) == []


def test_an_omni_row_carrying_an_asr_model_is_a_violation():
    """§10: `asr_model` is null for omni. A non-null one means a cascade field
    leaked onto an omni row — the opposite defect from the one above."""
    assert "omni_field_not_null" in rules(
        vs.validate_row(omni_row(asr_model="mlx-community/whisper-base")))


def test_an_omni_row_carrying_error_categories_is_a_violation():
    """§10: `error_categories` is [] for omni."""
    chunk = {"type": "substitute", "error_subtype": "semantic_drift",
             "slot_position": "function", "ref": "a", "hyp": "b",
             "entity_type": None}
    assert "omni_error_categories_not_empty" in rules(
        vs.validate_row(omni_row(error_categories=[chunk])))


def test_a_cascade_row_without_an_asr_model_is_a_violation():
    assert "cascade_asr_model_null" in rules(
        vs.validate_row(cascade_row(asr_model=None)))


def test_a_cascade_row_without_a_transcription_is_a_violation():
    """Distinct from an EMPTY transcription, which is a real, scored outcome
    (`NO_TRANSCRIPTION`, Step 4c). Null means the field was never written."""
    assert "cascade_transcript_field_null" in rules(
        vs.validate_row(cascade_row(transcription=None)))
    assert vs.validate_row(cascade_row(transcription="", wer=1.0,
                                       full_hallucination=False)) == []


# ─── presence, types, enumerations ───────────────────────────────────────────

def test_a_missing_required_field_is_reported_by_name():
    row = cascade_row()
    del row["gold_intent"]
    v = vs.validate_row(row)
    assert rules(v) == ["missing_field"]
    assert "gold_intent" in v[0].message


def test_every_one_of_the_33_schema_fields_is_required():
    assert len(vs.SCHEMA_FIELDS) == 33
    for field in vs.SCHEMA_FIELDS:
        row = cascade_row()
        del row[field]
        assert "missing_field" in rules(vs.validate_row(row)), field


def test_a_wrong_type_is_reported():
    assert "wrong_type" in rules(vs.validate_row(cascade_row(wer="0.0")))
    assert "wrong_type" in rules(vs.validate_row(cascade_row(slurp_id="6925")))


def test_a_boolean_is_not_accepted_where_an_integer_is_required():
    """`isinstance(True, int)` is True in Python; tsa=True must still fail."""
    assert "wrong_type" in rules(vs.validate_row(cascade_row(tsa=True)))


def test_null_in_a_field_10_never_marks_nullable_is_reported():
    assert "null_not_allowed" in rules(vs.validate_row(cascade_row(tsa=None)))


def test_an_unknown_pathway_is_reported():
    assert "bad_enum" in rules(vs.validate_row(cascade_row(pathway="hybrid")))


def test_an_unknown_model_family_is_reported():
    assert "bad_enum" in rules(vs.validate_row(cascade_row(model_family="meta")))


# ─── clean / SNR coding ──────────────────────────────────────────────────────

def test_a_clean_row_carrying_an_snr_is_reported():
    assert "clean_row_has_snr" in rules(
        vs.validate_row(cascade_row(degradation="clean", snr_db=0)))


def test_a_degraded_row_without_an_snr_is_reported():
    assert "degraded_row_has_no_snr" in rules(
        vs.validate_row(cascade_row(degradation="babble", snr_db=None)))


def test_a_conforming_clean_row_passes():
    assert vs.validate_row(cascade_row(degradation="clean", snr_db=None)) == []


# ─── the derived metrics: §10 states the formulas, so they are checkable ─────

def test_ees_that_contradicts_tsa_and_pf_is_reported():
    """§10: `ees` = tsa==1 AND pf >= 0.5. Here tsa=1, pf=0.5 -> ees must be 1."""
    assert "ees_formula" in rules(vs.validate_row(cascade_row(ees=0)))
    assert "ees_formula" in rules(
        vs.validate_row(cascade_row(tsa=0, ees=1, ees_strict=0)))


def test_ees_strict_that_contradicts_tsa_and_pf_is_reported():
    assert "ees_strict_formula" in rules(vs.validate_row(cascade_row(ees_strict=1)))


def test_with_no_gold_entities_ees_equals_tsa():
    """§8: PF is None when the utterance has no gold entities, and EES = TSA."""
    ok = cascade_row(gold_parameters={}, pf=None, pf_n_gold=0, pf_n_matched=0,
                     tsa=1, ees=1, ees_strict=1)
    assert vs.validate_row(ok) == []
    bad = dict(ok, ees=0)
    assert "ees_formula" in rules(vs.validate_row(bad))


def test_pf_that_is_not_the_recall_of_its_own_counters_is_reported():
    assert "pf_not_recall" in rules(
        vs.validate_row(cascade_row(pf=1.0, pf_n_gold=2, pf_n_matched=1,
                                    ees=1, ees_strict=1)))


def test_pf_null_disagreeing_with_pf_n_gold_is_reported():
    assert "pf_null_mismatch" in rules(
        vs.validate_row(cascade_row(pf=None, pf_n_gold=2, pf_n_matched=1,
                                    ees=1, ees_strict=1)))
    assert "pf_null_mismatch" in rules(
        vs.validate_row(cascade_row(pf=0.0, pf_n_gold=0, pf_n_matched=0,
                                    ees=1, ees_strict=1)))


def test_more_matched_than_gold_entities_is_reported():
    assert "pf_counters_impossible" in rules(
        vs.validate_row(cascade_row(pf=1.0, pf_n_gold=1, pf_n_matched=2,
                                    ees=1, ees_strict=1)))


# ─── error_categories, phonetic_distance, full_hallucination ─────────────────

def test_an_error_chunk_with_the_wrong_keys_is_reported():
    assert "error_chunk_keys" in rules(
        vs.validate_row(cascade_row(error_categories=[{"type": "delete"}])))


def test_phonetic_distance_without_a_critical_substitution_is_reported():
    """§10: null for omni AND for rows without critical-slot substitutions."""
    assert "phonetic_distance_unsupported" in rules(
        vs.validate_row(cascade_row(phonetic_distance=0.4)))


def test_phonetic_distance_on_a_critical_substitution_is_accepted():
    chunk = {"type": "substitute", "error_subtype": "named_entity_error",
             "slot_position": "critical", "ref": "sara", "hyp": "sarah",
             "entity_type": "person"}
    assert vs.validate_row(
        cascade_row(error_categories=[chunk], phonetic_distance=0.0)) == []


def test_phonetic_distance_outside_zero_to_one_is_reported():
    chunk = {"type": "substitute", "error_subtype": "named_entity_error",
             "slot_position": "critical", "ref": "sara", "hyp": "sarah",
             "entity_type": "person"}
    assert "value_out_of_range" in rules(
        vs.validate_row(cascade_row(error_categories=[chunk],
                                    phonetic_distance=1.4)))


def test_full_hallucination_true_below_wer_one_is_reported():
    """§10 defines the flag as WER>=1 AND non-empty hypothesis."""
    assert "full_hallucination_definition" in rules(
        vs.validate_row(cascade_row(full_hallucination=True, wer=0.5)))


def test_full_hallucination_true_on_an_empty_transcript_is_reported():
    """§8: an empty transcription is a total deletion, never a hallucination."""
    assert "full_hallucination_definition" in rules(
        vs.validate_row(cascade_row(full_hallucination=True, wer=1.0,
                                    transcription="   ")))


# ─── rules that come from a document OTHER than §10 ──────────────────────────

def test_a_failure_without_a_tsh_category_is_reported_and_is_not_a_10_rule():
    """§4 lists "TSH / TCH category if failure" as a dependent variable;
    §10 itself never says when the field must be non-null. The violation
    therefore carries a source other than §10, and --schema-only drops it."""
    bad = cascade_row(tsa=0, ees=0, ees_strict=0, tsh_category=None)
    v = vs.validate_row(bad)
    assert "tsh_category_missing" in rules(v)
    assert all(x.source != vs.SECTION_10 for x in v if x.rule == "tsh_category_missing")
    assert vs.validate_row(bad, schema_only=True) == []


def test_a_tsh_category_outside_the_peng_ontology_is_reported():
    assert "bad_enum" in rules(
        vs.validate_row(cascade_row(tsa=0, ees=0, ees_strict=0,
                                    tsh_category="confused")))


def test_a_degradation_outside_the_six_plus_clean_is_reported():
    v = vs.validate_row(cascade_row(degradation="whisper_noise"))
    assert "bad_enum" in rules(v)
    assert all(x.source != vs.SECTION_10 for x in v if x.rule == "bad_enum")


def test_a_gold_intent_outside_the_nine_intent_ontology_is_reported():
    assert "bad_enum" in rules(vs.validate_row(cascade_row(gold_intent="music_play")))


def test_a_predicted_intent_outside_the_ontology_is_not_a_violation():
    """A hallucinated label is the FINDING (TSH `hallucinated`), not a defect."""
    assert vs.validate_row(
        cascade_row(tsa=0, ees=0, ees_strict=0, predicted_intent="pizza_order",
                    predicted_intent_raw="pizza_order", tsh_category="hallucinated",
                    pf=0.0, pf_n_matched=0)) == []


# ─── file-level checks ───────────────────────────────────────────────────────

def test_a_duplicate_resume_key_within_a_file_is_reported(tmp_path):
    """§10's resume-from-row rule keys on
    (slurp_id, pathway, asr_model, degradation, snr_db, llm_model).
    Two rows sharing one mean the done-set failed and a row was routed twice."""
    p = write_jsonl(tmp_path / "cascade_whisper_qwen9b_clean_scored.jsonl",
                    [cascade_row(degradation="clean", snr_db=None),
                     cascade_row(degradation="clean", snr_db=None)])
    report = vs.validate_files([p])
    assert "duplicate_resume_key" in report.rule_counts


def test_two_rows_differing_only_in_asr_model_are_not_duplicates(tmp_path):
    p = write_jsonl(tmp_path / "cascade_whisper_qwen9b_clean_scored.jsonl",
                    [cascade_row(degradation="clean", snr_db=None),
                     cascade_row(degradation="clean", snr_db=None,
                                 asr_model="mlx-community/whisper-base")])
    assert vs.validate_files([p]).rule_counts == {}


def test_a_malformed_json_line_is_reported_not_raised(tmp_path):
    p = tmp_path / "cascade_whisper_qwen9b_clean_scored.jsonl"
    p.write_text(json.dumps(cascade_row(degradation="clean", snr_db=None)) +
                 "\n{not json\n")
    report = vs.validate_files([p])
    assert "unparseable_line" in report.rule_counts
    assert report.n_rows == 1


def test_blank_lines_are_skipped(tmp_path):
    p = tmp_path / "cascade_whisper_qwen9b_clean_scored.jsonl"
    p.write_text(json.dumps(cascade_row(degradation="clean", snr_db=None)) + "\n\n\n")
    report = vs.validate_files([p])
    assert report.rule_counts == {}
    assert report.n_rows == 1


def test_the_additive_step_4c_fields_are_advisory_not_violations(tmp_path):
    """`llm_model_served` and `llm_thinking_request` are on 63,232 rows and in
    no §10 example (counted 2026-08-28). Step 4c added them deliberately, so
    they are surfaced as drift between §10 and the data — not as bad rows."""
    row = cascade_row(degradation="clean", snr_db=None)
    row["llm_model_served"] = "mlx-community/Qwen3.5-9B-MLX-8bit"
    row["llm_thinking_request"] = {"chat_template_kwargs": {"enable_thinking": False}}
    report = vs.validate_files([write_jsonl(tmp_path / "cascade_whisper_qwen9b_clean_scored.jsonl", [row])])
    assert report.rule_counts == {}
    assert report.extra_fields["llm_model_served"] == 1
    assert report.extra_fields["llm_thinking_request"] == 1


# ─── the report and the command line ─────────────────────────────────────────

def test_the_report_counts_files_rows_and_carries_example_identities(tmp_path):
    a = write_jsonl(tmp_path / "cascade_whisper_qwen9b_clean_scored.jsonl",
                    [cascade_row(degradation="clean", snr_db=None, ees=0)])
    b = write_jsonl(tmp_path / "omni_qwen_sweep_scored.jsonl",
                    [omni_row(degradation="clean", snr_db=None)])
    report = vs.validate_files([a, b])
    assert report.n_files == 2
    assert report.n_rows == 2
    assert report.rule_counts["ees_formula"] == 1
    ident = report.examples["ees_formula"][0]
    assert ident.file == a.name
    assert ident.line == 1
    assert ident.slurp_id == 6925
    text = vs.format_report(report)
    assert "ees_formula" in text
    assert "6925" in text


def test_the_ambiguity_section_is_printed_even_on_a_clean_run(tmp_path):
    """An ambiguity surfaced is worth more than a rule invented — so §10's
    silences are printed whether or not anything failed."""
    p = write_jsonl(tmp_path / "cascade_whisper_qwen9b_clean_scored.jsonl",
                    [cascade_row(degradation="clean", snr_db=None)])
    text = vs.format_report(vs.validate_files([p]))
    assert "§10 DOES NOT STATE" in text
    assert "notes" in text


def test_main_exits_zero_on_a_clean_file_and_one_on_a_dirty_one(tmp_path, capsys):
    clean = write_jsonl(tmp_path / "cascade_whisper_qwen9b_clean_scored.jsonl",
                        [cascade_row(degradation="clean", snr_db=None)])
    assert vs.main([str(clean)]) == 0
    dirty = write_jsonl(tmp_path / "parakeet_clean_qwen9b_scored.jsonl",
                        [cascade_row(degradation="clean", snr_db=None, ees=0)])
    assert vs.main([str(dirty)]) == 1


def test_main_with_no_arguments_uses_result_sets_sweep_files(tmp_path, monkeypatch):
    """Never a bare results/*_scored.jsonl glob (CLAUDE.md): the 200-row H2a
    pilot walked into the sweep counts through exactly such a glob."""
    p = write_jsonl(tmp_path / "cascade_whisper_qwen9b_clean_scored.jsonl",
                    [cascade_row(degradation="clean", snr_db=None)])
    called = []

    def fake_sweep_files(results_dir="results"):
        called.append(results_dir)
        return [p]

    monkeypatch.setattr(vs.result_sets, "sweep_files", fake_sweep_files)
    assert vs.main([]) == 0
    assert called == ["results"]


def test_validating_never_touches_the_input_file(tmp_path):
    """CLAUDE.md: never modify results/. Report, do not repair."""
    p = write_jsonl(tmp_path / "cascade_whisper_qwen9b_clean_scored.jsonl",
                    [cascade_row(degradation="clean", snr_db=None, ees=0),
                     cascade_row(degradation="clean", snr_db=None, slurp_id=7)])
    before = p.read_bytes()
    listing_before = sorted(x.name for x in tmp_path.iterdir())
    vs.main([str(p)])
    assert p.read_bytes() == before
    assert sorted(x.name for x in tmp_path.iterdir()) == listing_before


def test_schema_only_keeps_the_10_rules_and_drops_the_borrowed_ones(tmp_path):
    row = cascade_row(degradation="clean", snr_db=None, tsa=0, ees=0,
                      ees_strict=0, tsh_category=None)
    p = write_jsonl(tmp_path / "cascade_whisper_qwen9b_clean_scored.jsonl", [row])
    assert "tsh_category_missing" in vs.validate_files([p]).rule_counts
    assert vs.validate_files([p], schema_only=True).rule_counts == {}


def test_every_violation_names_the_document_it_comes_from():
    """RULE ZERO applied to the validator itself: no rule may be anonymous."""
    bad = cascade_row(pathway="hybrid", ees=0, wer="x", tsa=0,
                      tsh_category="confused", degradation="whisper_noise")
    v = vs.validate_row(bad)
    assert v
    for item in v:
        assert item.source, item
        assert item.source in vs.KNOWN_SOURCES, item


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
