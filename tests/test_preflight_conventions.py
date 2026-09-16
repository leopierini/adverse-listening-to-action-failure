"""Unit tests for the scoring-convention guard in scripts/preflight_gcp.py.

This is the gate that stands between the project and a paid A100. On
2026-08-18 it exited 1 with "BLOCKED — do not boot" over 200 rows of
results/omni_pilot_qwen.jsonl, whose `transcription` is null BY DESIGN — the
omni pathway has no transcript at all (§10). A gate that cries wolf is a gate
that gets clicked through, and this one guards money.
"""
import preflight_gcp as pf


def cascade_row(transcription="", tsa=None, predicted_intent="alarm_set"):
    return {"slurp_id": 1, "pathway": "cascade", "transcription": transcription,
            "tsa": tsa, "predicted_intent": predicted_intent}


def omni_row(predicted_intent="alarm_set", tsa=1):
    """§10: omni rows carry transcription = null. That is the pathway, not a
    scoring bug — the model never produces a transcript to begin with."""
    return {"slurp_id": 1, "pathway": "omni", "transcription": None,
            "tsa": tsa, "predicted_intent": predicted_intent}


def test_an_omni_row_without_a_transcript_is_not_an_offender():
    assert pf.violates_no_transcription_convention(omni_row()) is False


def test_a_cascade_row_with_an_empty_transcript_must_be_stamped():
    assert pf.violates_no_transcription_convention(cascade_row()) is True


def test_a_correctly_stamped_cascade_row_passes():
    row = cascade_row(tsa=0, predicted_intent="NO_TRANSCRIPTION")
    assert pf.violates_no_transcription_convention(row) is False


def test_a_cascade_row_with_a_real_transcript_is_never_checked():
    assert pf.violates_no_transcription_convention(
        cascade_row(transcription="set an alarm")) is False


def test_a_row_with_no_pathway_field_is_treated_as_cascade():
    """Rows written before the pathway stamp existed are cascade rows; the
    convention applies to them."""
    row = {"slurp_id": 1, "transcription": "", "tsa": None,
           "predicted_intent": "alarm_set"}
    assert pf.violates_no_transcription_convention(row) is True
