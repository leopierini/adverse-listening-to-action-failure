import json

import route_omni


def test_build_omni_row_schema():
    sample = {"slurp_id": 5, "scenario": "alarm", "intent": "alarm_set",
              "sentence": "wake me at six", "entities": [],
              "tokens": [{"surface": "wake"}]}
    row = route_omni.build_omni_row(
        sample, audio_file="a.flac", degradation="babble", snr_db=10,
        seed=42, llm_model="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    assert row["pathway"] == "omni"
    assert row["model_family"] == "qwen"
    assert row["asr_model"] is None
    assert row["transcription"] is None
    assert row["wer"] is None
    assert row["error_categories"] == []
    assert row["phonetic_distance"] is None
    assert row["full_hallucination"] is None
    assert row["degradation"] == "babble"
    assert row["snr_db"] == 10
    assert row["llm_model"] == "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def test_resume_key_includes_pathway():
    row = {"slurp_id": 5, "pathway": "omni", "asr_model": None,
           "degradation": "clean", "snr_db": None, "llm_model": "m"}
    assert route_omni.resume_key(row) == (5, "omni", None, "clean", None, "m")


def test_load_processed_treats_missing_audio_as_NOT_done(tmp_path):
    """CONTRACT CHANGED 2026-08-12 (pre-flight audit, finding C5).

    This test previously asserted the opposite — that a missing-audio row counted
    as *done*. Combined with `_skip` writing those rows to the MAIN output, that
    made a single run with a wrong `--audio-root` (or an incomplete rsync of the
    corrupted bank) write up to 7,488 rows with `tsa: null`, exit 0, and then
    permanently skip all of them on every corrected re-run. Unrecoverable without
    hand-editing collected data.

    New contract: missing audio goes to the failure sidecar, so it never reaches
    the main output at all; and a row is `done` only if it carries a prediction,
    so anything that did reach the model-less path is retried. No omni output
    files exist yet, so there is no legacy data relying on the old behaviour."""
    rows = [
        {"slurp_id": 1, "pathway": "omni", "asr_model": None, "degradation": "clean",
         "snr_db": None, "llm_model": "m", "predicted_intent": "alarm_set", "notes": ""},
        {"slurp_id": 2, "pathway": "omni", "asr_model": None, "degradation": "clean",
         "snr_db": None, "llm_model": "m", "predicted_intent": None,
         "notes": " | missing_audio:corrupted/x.flac"},
    ]
    out = tmp_path / "o.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    done = route_omni.load_processed(str(out))
    assert (1, "omni", None, "clean", None, "m") in done
    assert (2, "omni", None, "clean", None, "m") not in done  # retried, not lost


def test_apply_prediction_fills_fields():
    row = route_omni.build_omni_row(
        {"slurp_id": 1, "scenario": "alarm", "intent": "alarm_set",
         "sentence": "x", "entities": [], "tokens": []},
        audio_file="a.flac", degradation="clean", snr_db=None, seed=42,
        llm_model="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    route_omni.apply_prediction(row, {"intent": "alarm_set", "parameters": {"time": "6 am"}})
    assert row["predicted_intent"] == "alarm_set"
    assert row["predicted_parameters"] == {"time": "6 am"}
    assert row["tsa"] == 1
