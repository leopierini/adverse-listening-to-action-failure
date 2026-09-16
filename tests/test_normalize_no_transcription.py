"""Tests for `normalize_no_transcription.py`.

The script back-fills the Step-4c empty-transcription convention onto routed
files written before 2026-08-12. The property that matters for the thesis is
that it produces *exactly* what `route_transcripts.route_one` would have
produced on the same row — otherwise the local Qwen-9B arm and the GCP
27B/Gemma arms still disagree on the denominators, which is the bias the
convention exists to remove.

Everything here is offline.
"""
import json

import normalize_no_transcription as nn
import route_transcripts as rt


def _row(**over):
    row = {
        "slurp_id": 1, "transcription": "", "asr_model": "parakeet",
        "degradation": "babble", "snr_db": 0, "llm_model": "qwen9b",
        "predicted_intent": None, "predicted_intent_raw": None,
        "predicted_parameters": {}, "tsa": None, "notes": None,
        "wer": 1.0, "gold_intent_final": "alarm_set",
    }
    row.update(over)
    return row


# ─── the core equivalence ────────────────────────────────────────────────────

def test_matches_route_one_exactly_on_an_empty_transcription():
    """The back-fill and the live driver must agree field for field."""
    via_driver = rt.route_one(_row(), "qwen9b", "http://x/v1", "primary")[1]
    via_script = nn.normalize_row(_row())[0]

    for field in ("predicted_intent", "predicted_intent_raw",
                  "predicted_parameters", "tsa"):
        assert via_script[field] == via_driver[field], field
    assert nn.NOTE_MARKER in via_script["notes"]
    assert nn.NOTE_MARKER in via_driver["notes"]


def test_route_one_sends_no_request_so_the_comparison_is_offline():
    """Guards the premise above: route_one must not need a live endpoint."""
    status, row = rt.route_one(_row(), "qwen9b", "http://127.0.0.1:1/v1", "primary")
    assert status == "ok"
    assert row["tsa"] == 0


# ─── what it must not touch ──────────────────────────────────────────────────

def test_leaves_rows_with_a_transcription_untouched():
    before = _row(transcription="set an alarm", predicted_intent="alarm_set", tsa=1)
    after, changed = nn.normalize_row(dict(before))
    assert changed is False
    assert after == before


def test_preserves_asr_stage_fields():
    after, _ = nn.normalize_row(_row(wer=1.0, cer=0.8))
    assert after["wer"] == 1.0 and after["cer"] == 0.8


def test_whitespace_only_transcription_counts_as_empty():
    after, changed = nn.normalize_row(_row(transcription="   \n"))
    assert changed is True and after["tsa"] == 0


# ─── idempotence ─────────────────────────────────────────────────────────────

def test_is_idempotent():
    once, changed_1 = nn.normalize_row(_row())
    twice, changed_2 = nn.normalize_row(dict(once))
    assert changed_1 is True and changed_2 is False
    assert once == twice
    assert twice["notes"].count(nn.NOTE_MARKER) == 1


def test_already_normalized_row_is_not_rewritten():
    row = _row(predicted_intent="NO_TRANSCRIPTION",
               predicted_intent_raw="NO_TRANSCRIPTION", tsa=0,
               notes=" | no_transcription")
    _, changed = nn.normalize_row(dict(row))
    assert changed is False


# ─── file level ──────────────────────────────────────────────────────────────

def test_normalize_file_reports_and_rewrites(tmp_path):
    src = tmp_path / "in.jsonl"
    with src.open("w") as f:
        f.write(json.dumps(_row(slurp_id=1)) + "\n")
        f.write(json.dumps(_row(slurp_id=2, transcription="hello",
                                predicted_intent="alarm_set", tsa=1)) + "\n")
        f.write(json.dumps(_row(slurp_id=3)) + "\n")

    out = tmp_path / "out.jsonl"
    rep = nn.normalize_file(str(src), str(out))

    assert rep == {"file": str(src), "rows": 3, "empty": 2, "changed": 2}
    rows = [json.loads(l) for l in out.open()]
    assert [r["tsa"] for r in rows] == [0, 1, 0]
    assert [r["slurp_id"] for r in rows] == [1, 2, 3]  # order preserved


def test_downstream_pf_scores_a_normalized_row_as_a_failure():
    """compute_pf refuses EES when tsa is None; with tsa=0 it must score 0."""
    import compute_pf  # noqa: F401  (import guard: the module loads)
    after, _ = nn.normalize_row(_row())
    assert after["tsa"] == 0
    assert after["tsa"] is not None
