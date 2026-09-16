"""Two gaps found by the 2026-08-14 clean-context audit.

1. NO TEST PINNED THE SIGN OF THE HEADLINE METRIC.
   A mutation test proved it: inverting `absorption_metrics.py`'s
   `retention - naive_paired` to `naive_paired - retention` left all 25 tests in
   `test_absorption_metrics.py` green. Every `gain_paired` assertion there was
   either 0.0 (degenerate: retention 0.5 against naive 0.5) or None. A
   sign-inverted headline metric shipped clean — which is precisely the failure
   that reached the advisor on 2026-08-07.

2. `batch_evaluate.py` FED EMPTY TRANSCRIPTIONS TO THE ROUTER.
   `route_transcripts.route_one` short-circuits them with no model call;
   `batch_evaluate` passed "" straight to `get_intent_from_llm` and scored
   whatever came back. All 18 affected rows happened to receive `unknown` from
   Qwen-9B, so nothing collected is corrupt — that was luck. A router that
   guessed the gold intent would have written `tsa = 1` for an utterance whose
   ASR produced nothing.
"""
import json

import absorption_metrics as am


# ─── 1. the headline metric's SIGN and MAGNITUDE ─────────────────────────────

def _cell(n_items, retention_hits, wer):
    """A degraded cell of n_items whose clean anchors all scored ees=1."""
    deg, clean = [], []
    for i in range(n_items):
        base = {"slurp_id": i, "asr_model": "asr", "llm_model": "llm"}
        clean.append(dict(base, degradation="clean", snr_db=None, wer=0.0, ees=1))
        deg.append(dict(base, degradation="noise", snr_db=0, wer=wer,
                        ees=1 if i < retention_hits else 0))
    return deg, clean


def test_gain_paired_sign_and_value_are_pinned_non_degenerately():
    """retention 0.8, mean WER 0.5 -> naive 0.5 -> gain = +0.3.

    Non-degenerate on purpose: retention != naive, so an inverted subtraction
    yields -0.3 and fails. This is the assertion the suite lacked.
    """
    deg, clean = _cell(10, retention_hits=8, wer=0.5)
    st = am.paired_stats(deg, am.build_clean_index(clean + deg), "ees")
    assert st["retention"] == 0.8
    assert st["naive_paired"] == 0.5
    assert abs(st["gain_paired"] - 0.3) < 1e-12, st["gain_paired"]
    assert st["gain_paired"] > 0


def test_gain_paired_is_negative_when_the_router_underperforms_the_null():
    """The other side of zero, so neither sign passes by accident."""
    deg, clean = _cell(10, retention_hits=2, wer=0.5)
    st = am.paired_stats(deg, am.build_clean_index(clean + deg), "ees")
    assert abs(st["gain_paired"] - (-0.3)) < 1e-12, st["gain_paired"]
    assert st["gain_paired"] < 0


def test_gain_ceiling_sign_is_pinned_non_degenerately():
    """The FIRST version of this test asserted `ceiling` and `absorption` and
    never mentioned `gain_ceiling` at all — it narrated a plan in a comment and
    did not carry it out, so inverting the subtraction left it green. That is
    the same defect this file exists to fix, one metric over.

    Construction: 4 rows at WER=0 of which 2 succeed -> ceiling = 0.5.
    6 rows at WER=0.5 of which 5 succeed -> absorption = 5/6.
    naive_ceiling = 0.5 * (1 - 0.5) = 0.25 -> gain_ceiling = 5/6 - 0.25 = +0.5833.
    Neither operand equals the other, so an inverted subtraction fails.
    """
    rows = [{"slurp_id": i, "asr_model": "a", "llm_model": "l",
             "degradation": "noise", "snr_db": 0,
             "wer": 0.0 if i < 4 else 0.5,
             "ees": (1 if i < 2 else 0) if i < 4 else (1 if i < 9 else 0)}
            for i in range(10)]
    st = am.cell_stats(rows, "ees")
    assert st["ceiling"] == 0.5, st["ceiling"]
    assert abs(st["absorption"] - 5.0 / 6.0) < 1e-12, st["absorption"]
    assert abs(st["naive_ceiling"] - 0.25) < 1e-12, st["naive_ceiling"]
    assert abs(st["gain_ceiling"] - (5.0 / 6.0 - 0.25)) < 1e-12, st["gain_ceiling"]
    assert st["gain_ceiling"] > 0


def test_median_wer_is_the_real_median_on_an_even_cell():
    """Was the upper-middle value until 2026-08-14."""
    rows = [{"slurp_id": i, "asr_model": "a", "llm_model": "l",
             "degradation": "noise", "snr_db": 0, "wer": w, "ees": 0}
            for i, w in enumerate([0.0, 0.2, 0.4, 0.6])]
    assert am.cell_stats(rows, "ees")["median_wer"] == 0.30000000000000004 or \
        abs(am.cell_stats(rows, "ees")["median_wer"] - 0.3) < 1e-9


# ─── 2. batch_evaluate must not ask the model about nothing ──────────────────

def test_batch_evaluate_never_calls_the_router_on_an_empty_transcription(
        monkeypatch, tmp_path):
    """BEHAVIOURAL. The first version of this test grepped the source for the
    guard's ordering, so dedenting the router call out of the `else` — the
    realistic regression — left it green while the driver fabricated a success.

    Here the stub router always returns the gold intent. If it is ever asked
    about an empty transcript, the row scores tsa=1 and the assertions fail.
    """
    import argparse

    import batch_evaluate as be

    calls = []

    def fake_llm(text, **kw):
        calls.append(text)
        return {"intent": "alarm_set", "parameters": {}}

    monkeypatch.setattr(be, "transcribe_audio", lambda *a, **k: "")
    monkeypatch.setattr(be, "get_intent_from_llm", fake_llm)
    monkeypatch.setattr(be, "load_samples", lambda *a, **k: [
        {"slurp_id": 1, "scenario": "alarm", "intent": "alarm_set",
         "sentence": "set an alarm", "entities": [],
         "recordings": [{"file": "x.flac"}]}])
    monkeypatch.setattr(be, "pick_available_recording",
                        lambda s, root: ("x.flac", str(tmp_path / "x.flac")))

    out = tmp_path / "out.jsonl"
    args = argparse.Namespace(
        dataset="d", manifest="m", num_samples=1, seed=42,
        asr_models="asr", audio_root=str(tmp_path), degradation="clean",
        snr_db=None, asr_language="en", llm_model="llm",
        llm_base_url="http://x/v1", llm_prompt_variant="primary",
        out=str(out))
    be.evaluate(args)

    assert calls == [], "the router was asked about an empty transcript: {}".format(calls)
    rows = [json.loads(l) for l in out.open() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["predicted_intent"] == "NO_TRANSCRIPTION"
    assert rows[0]["tsa"] == 0
    assert "no_transcription" in (rows[0]["notes"] or "")


def test_collected_rows_never_score_an_empty_transcription_as_success():
    """The real data, as an invariant: no empty transcript may carry tsa == 1.

    Scoped to the CASCADE arm on two grounds, and the second is the reason the
    first matters. `CLAUDE.md` forbids a bare `results/*_scored.jsonl` glob --
    that is how the 200-row H2a pilot walked into the sweep counts. And the
    invariant is about the cascade pathway specifically: it says a router must
    not claim success on a transcript the ASR failed to produce. The omni
    models are handed audio and never produce a transcript at all, so
    `transcription` is null on all 7,904 of their rows by construction, and a
    success there is a real success. Read over a bare glob, this test called
    the entire omni arm a bug the moment it landed (2026-08-27).
    """
    import result_sets
    bad = []
    for f in result_sets.cascade_files("results"):
        for line in open(f):
            r = json.loads(line)
            if not (r.get("transcription") or "").strip() and r.get("tsa") == 1:
                bad.append((str(f), r.get("slurp_id")))
    assert bad == [], "empty transcription scored as a success: {}".format(bad[:5])


def test_the_omni_arm_carries_no_transcript_at_all_and_that_is_by_design():
    """The companion to the test above: the omni rows are excluded from that
    invariant because of a structural property, not to make a test pass."""
    import result_sets
    files = result_sets.omni_files("results")
    if not files:
        pytest.skip("no omni rows collected yet")
    n = with_transcript = 0
    for f in files:
        for line in open(f):
            r = json.loads(line)
            n += 1
            if (r.get("transcription") or "").strip():
                with_transcript += 1
    assert n > 0
    assert with_transcript == 0, (
        "%d omni rows carry a transcript; the omni pathway is audio-in, "
        "intent-out and should carry none" % with_transcript)
