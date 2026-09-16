"""Tests for the cascade GCP driver (`route_transcripts.py`) and the response
validation in `pipeline.py` that it depends on.

This file did not exist before the 2026-08-12 pre-flight audit — which is exactly
why the three defects it now pins (stale-prediction carryover, unusable-response
scoring, unverified endpoint) survived into a paid-GPU-ready state.

Everything here is offline: the OpenAI client is replaced by a fake. No network.
"""
import json
import sys

import pytest

import pipeline
import route_transcripts as rt


# ─── Fake OpenAI client ──────────────────────────────────────────────────────

class _Msg:
    def __init__(self, content, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


class _Choice:
    def __init__(self, content, finish_reason="stop", reasoning_content=None):
        self.message = _Msg(content, reasoning_content)
        self.finish_reason = finish_reason


class _Resp:
    def __init__(self, content, finish_reason="stop", reasoning_content=None):
        self.choices = [_Choice(content, finish_reason, reasoning_content)]


class _Completions:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp


class _Models:
    def __init__(self, ids):
        self._ids = ids

    def list(self):
        if isinstance(self._ids, Exception):
            raise self._ids
        return type("L", (), {"data": [type("M", (), {"id": i})() for i in self._ids]})()


class FakeClient:
    def __init__(self, resp=None, model_ids=()):
        self.chat = type("C", (), {})()
        self.chat.completions = _Completions(resp)
        self.models = _Models(model_ids)


@pytest.fixture
def fake_client(monkeypatch):
    """Install a FakeClient for every base_url; returns a setter."""
    holder = {}

    def _install(resp=None, model_ids=()):
        c = FakeClient(resp, model_ids)
        holder["client"] = c
        monkeypatch.setattr(pipeline, "_get_client", lambda base_url: c)
        return c

    return _install


def _routed_row(**over):
    """A row as it comes OUT of a previous router's run — the realistic GCP input,
    because the Whisper transcripts only exist inside already-routed files."""
    row = {
        "slurp_id": 10732,
        "asr_model": "mlx-community/whisper-base-mlx",
        "degradation": "babble",
        "snr_db": 0,
        "gold_intent": "lists_remove",
        "transcription": "remove pepper from my grocery list",
        "llm_model": "mlx-community/Qwen3.5-9B-MLX-8bit",
        "llm_prompt_variant": "primary_9intent_9shot",
        "predicted_intent": "unknown",
        "predicted_intent_raw": "unknown",
        "predicted_parameters": {"stale": "value"},
        "tsa": 0,
        "pf": 0.0,
        "pf_n_gold": 1,
        "pf_n_matched": 0,
        "ees": 0,
        "ees_strict": 0,
        "tsh_category": "unknown",
        "tch_category": None,
        "notes": "",
    }
    row.update(over)
    return row


STALE_FIELDS = ("pf", "pf_n_gold", "pf_n_matched", "ees", "ees_strict",
                "tsh_category", "tch_category")


# ─── 1. The no-transcription convention ──────────────────────────────────────

def test_no_transcription_is_scored_as_failure_and_calls_no_model(monkeypatch):
    """Decision 2026-08-12: an empty transcription is a real end-to-end failure
    (tsa=0), stamped deterministically — no GPU request, no model call."""
    called = []
    monkeypatch.setattr(rt, "get_intent_from_llm",
                        lambda *a, **k: called.append(1) or {"intent": "lists_remove"})

    status, row = rt.route_one(_routed_row(transcription=""),
                               "Qwen/Qwen3.5-27B", "http://vm:8000/v1", "v1")

    assert called == [], "the model must not be called on an empty transcription"
    assert status == "ok", "must be a scored row, so denominators match across routers"
    assert row["predicted_intent"] == "NO_TRANSCRIPTION"
    assert row["predicted_intent_raw"] == "NO_TRANSCRIPTION"
    assert row["predicted_parameters"] == {}
    assert row["tsa"] == 0
    assert "no_transcription" in row["notes"]


def test_whitespace_only_transcription_counts_as_empty(monkeypatch):
    monkeypatch.setattr(rt, "get_intent_from_llm",
                        lambda *a, **k: pytest.fail("model must not be called"))
    status, row = rt.route_one(_routed_row(transcription="   \n"),
                               "Qwen/Qwen3.5-27B", "http://vm:8000/v1", "v1")
    assert status == "ok" and row["tsa"] == 0
    assert row["predicted_intent"] == "NO_TRANSCRIPTION"


def test_no_transcription_clears_every_stale_field(monkeypatch):
    """The C1 regression: a row stamped with the NEW llm_model must never carry
    the PREVIOUS router's parameter-layer scores."""
    monkeypatch.setattr(rt, "get_intent_from_llm",
                        lambda *a, **k: pytest.fail("model must not be called"))
    _, row = rt.route_one(_routed_row(transcription=""),
                          "Qwen/Qwen3.5-27B", "http://vm:8000/v1", "v1")
    assert row["llm_model"] == "Qwen/Qwen3.5-27B"
    for f in STALE_FIELDS:
        assert row[f] is None, f"{f} carried over from the previous router"


# ─── 2. Stale-field clearing on the normal path ──────────────────────────────

def test_successful_route_clears_stale_parameter_layer(monkeypatch):
    monkeypatch.setattr(rt, "get_intent_from_llm",
                        lambda *a, **k: {"intent": "lists_remove", "parameters": {"item_name": "pepper"}})
    status, row = rt.route_one(_routed_row(), "Qwen/Qwen3.5-27B", "http://vm:8000/v1", "v1")
    assert status == "ok"
    assert row["predicted_intent"] == "lists_remove"
    assert row["predicted_parameters"] == {"item_name": "pepper"}
    assert row["tsa"] == 1
    for f in STALE_FIELDS:
        assert row[f] is None, f"{f} must be recomputed by compute_pf, not inherited"


def test_genuine_parse_error_is_still_scored(monkeypatch):
    """A model that really did answer, just not in JSON, is a legitimate failure
    at temp 0 — it must stay in the main output, scored."""
    monkeypatch.setattr(rt, "get_intent_from_llm",
                        lambda *a, **k: {"error": "Invalid JSON format", "raw_output": "I think..."})
    status, row = rt.route_one(_routed_row(), "Qwen/Qwen3.5-27B", "http://vm:8000/v1", "v1")
    assert status == "ok"
    assert row["predicted_intent"] == "JSON_PARSE_ERROR"
    assert row["tsa"] == 0


def test_infrastructure_failure_is_not_scored(monkeypatch):
    monkeypatch.setattr(rt, "get_intent_from_llm",
                        lambda *a, **k: {"error": "llm_request_failed", "detail": "boom"})
    status, row = rt.route_one(_routed_row(), "Qwen/Qwen3.5-27B", "http://vm:8000/v1", "v1")
    assert status == "fail"
    assert row["tsa"] is None
    assert row["predicted_intent"] is None


# ─── 3. pipeline response validation (the C3 regression) ─────────────────────

def test_empty_content_is_an_infrastructure_failure(fake_client):
    """vLLM with a reasoning parser returns content=None and stuffs everything in
    reasoning_content. Scoring that as tsa=0 would fabricate 100% wrongness."""
    fake_client(_Resp(None, finish_reason="stop", reasoning_content="<think>...</think>"))
    r = pipeline.get_intent_from_llm("set an alarm", model="m", base_url="http://vm:8000/v1")
    assert r["error"] == "llm_request_failed"
    assert "reasoning_content_present=True" in r["detail"]


def test_blank_content_is_an_infrastructure_failure(fake_client):
    fake_client(_Resp("   ", finish_reason="stop"))
    r = pipeline.get_intent_from_llm("set an alarm", model="m", base_url="http://vm:8000/v1")
    assert r["error"] == "llm_request_failed"


def test_truncated_response_is_an_infrastructure_failure(fake_client):
    """finish_reason='length' means the answer was cut off — a resource problem,
    not a model mistake."""
    fake_client(_Resp('{"intent": "alarm_se', finish_reason="length"))
    r = pipeline.get_intent_from_llm("set an alarm", model="m", base_url="http://vm:8000/v1")
    assert r["error"] == "llm_request_failed"
    assert "finish_reason='length'" in r["detail"]


def test_good_response_still_parses(fake_client):
    fake_client(_Resp('{"intent": "alarm_set", "parameters": {"time": "6 am"}}'))
    r = pipeline.get_intent_from_llm("wake me at six", model="m", base_url="http://vm:8000/v1")
    assert r == {"intent": "alarm_set", "parameters": {"time": "6 am"}}


def test_request_sends_determinism_and_budget_params(fake_client):
    c = fake_client(_Resp('{"intent": "alarm_set"}'))
    pipeline.get_intent_from_llm("wake me", model="m", base_url="http://vm:8000/v1")
    kw = c.chat.completions.calls[0]
    assert kw["temperature"] == 0.0
    assert kw["response_format"] == {"type": "json_object"}
    assert kw["max_tokens"] == pipeline.DEFAULT_MAX_TOKENS
    assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_omni_validates_responses_the_same_way(fake_client, tiny_flac):
    fake_client(_Resp(None, finish_reason="length"))
    r = pipeline.get_intent_from_omni(tiny_flac, model="m", base_url="http://vm:8000/v1")
    assert r["error"] == "llm_request_failed"


# ─── 4. Endpoint / model verification (the H2 regression) ────────────────────

def test_verify_model_served_accepts_a_match(fake_client):
    fake_client(model_ids=["Qwen/Qwen3.5-27B"])
    assert pipeline.verify_model_served("Qwen/Qwen3.5-27B", "http://vm:8000/v1") == "Qwen/Qwen3.5-27B"


def test_verify_model_served_aborts_on_mismatch(fake_client):
    """The catastrophic case: --llm-base-url forgotten, so the request would go to
    the local Qwen-9B while every row is stamped Qwen3.5-27B."""
    fake_client(model_ids=["mlx-community/Qwen3.5-9B-MLX-8bit"])
    with pytest.raises(pipeline.ModelNotServedError) as e:
        pipeline.verify_model_served("Qwen/Qwen3.5-27B", "http://localhost:8080/v1")
    assert "Qwen3.5-9B" in str(e.value)


def test_verify_model_served_aborts_when_unreachable(fake_client):
    fake_client(model_ids=ConnectionError("refused"))
    with pytest.raises(pipeline.ModelNotServedError):
        pipeline.verify_model_served("Qwen/Qwen3.5-27B", "http://vm:8000/v1")


def test_main_exits_when_model_not_served(tmp_path, monkeypatch, fake_client):
    fake_client(model_ids=["some/other-model"])
    inp = tmp_path / "in.jsonl"
    inp.write_text(json.dumps(_routed_row()) + "\n")
    monkeypatch.setattr(sys, "argv", [
        "route_transcripts.py", "--in", str(inp), "--out", str(tmp_path / "o.jsonl"),
        "--llm-model", "Qwen/Qwen3.5-27B", "--llm-base-url", "http://vm:8000/v1"])
    with pytest.raises(SystemExit) as e:
        rt.main()
    assert e.value.code != 0
    assert not (tmp_path / "o.jsonl").exists() or (tmp_path / "o.jsonl").read_text() == ""


# ─── 5. Driver-level: sidecar routing and resume ─────────────────────────────

def _run_main(tmp_path, monkeypatch, rows, out_name="o.jsonl"):
    inp = tmp_path / "in.jsonl"
    inp.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / out_name
    monkeypatch.setattr(sys, "argv", [
        "route_transcripts.py", "--in", str(inp), "--out", str(out),
        "--llm-model", "Qwen/Qwen3.5-27B", "--llm-base-url", "http://vm:8000/v1",
        "--skip-model-check"])
    rt.main()
    return out


def test_failed_rows_go_to_the_sidecar_and_never_the_main_output(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "get_intent_from_llm",
                        lambda *a, **k: {"error": "llm_request_failed", "detail": "503"})
    out = _run_main(tmp_path, monkeypatch, [_routed_row()])
    assert not out.exists() or out.read_text().strip() == ""
    sidecar = tmp_path / "o.failures.jsonl"
    assert sidecar.exists() and len(sidecar.read_text().strip().splitlines()) == 1


def test_resume_adds_no_duplicate_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "get_intent_from_llm",
                        lambda *a, **k: {"intent": "lists_remove", "parameters": {}})
    rows = [_routed_row(slurp_id=1), _routed_row(slurp_id=2)]
    out = _run_main(tmp_path, monkeypatch, rows)
    assert len(out.read_text().strip().splitlines()) == 2
    _run_main(tmp_path, monkeypatch, rows)
    assert len(out.read_text().strip().splitlines()) == 2


def test_failed_row_is_retried_on_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "get_intent_from_llm",
                        lambda *a, **k: {"error": "llm_request_failed", "detail": "503"})
    out = _run_main(tmp_path, monkeypatch, [_routed_row()])
    monkeypatch.setattr(rt, "get_intent_from_llm",
                        lambda *a, **k: {"intent": "lists_remove", "parameters": {}})
    _run_main(tmp_path, monkeypatch, [_routed_row()])
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["tsa"] == 1


# ─── 6. Downstream consistency of the NO_TRANSCRIPTION convention ────────────

def test_no_transcription_categorises_as_missing_not_hallucinated():
    """The new sentinel must not be read as a router hallucination — the ASR
    produced nothing, the router was never asked. Left unhandled this would
    inflate the TSH 'hallucinated' count in exactly the worst cells (babble/0 dB),
    where empty transcriptions concentrate."""
    import categorize_failures as cf
    assert cf.tsh_category({"predicted_intent": "NO_TRANSCRIPTION"}) == "missing"


def test_no_transcription_row_scores_ees_zero(tmp_path):
    """End-to-end through compute_pf: tsa=0 must yield ees=0, not null."""
    import subprocess
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    row = {"slurp_id": 1, "gold_intent": "lists_remove", "tsa": 0,
           "gold_parameters": {"list_name": "grocery"}, "predicted_parameters": {},
           "predicted_intent": "NO_TRANSCRIPTION"}
    inp = tmp_path / "in.jsonl"
    inp.write_text(json.dumps(row) + "\n")
    out = tmp_path / "out.jsonl"
    r = subprocess.run([str(repo / "venv" / "bin" / "python"),
                        str(repo / "scripts" / "compute_pf.py"),
                        "--in", str(inp), "--out", str(out)],
                       capture_output=True, text=True, cwd=str(repo))
    assert r.returncode == 0, r.stderr
    scored = json.loads(out.read_text().strip())
    assert scored["ees"] == 0 and scored["ees_strict"] == 0
    assert scored["pf"] == 0.0


def test_rows_record_served_model_and_thinking_request(tmp_path, monkeypatch, fake_client):
    fake_client(model_ids=["Qwen/Qwen3.5-27B"])
    monkeypatch.setattr(rt, "get_intent_from_llm",
                        lambda *a, **k: {"intent": "lists_remove", "parameters": {}})
    inp = tmp_path / "in.jsonl"
    inp.write_text(json.dumps(_routed_row()) + "\n")
    out = tmp_path / "o.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "route_transcripts.py", "--in", str(inp), "--out", str(out),
        "--llm-model", "Qwen/Qwen3.5-27B", "--llm-base-url", "http://vm:8000/v1"])
    rt.main()
    row = json.loads(out.read_text().strip())
    assert row["llm_model_served"] == "Qwen/Qwen3.5-27B"
    assert row["llm_thinking_request"] == {"chat_template_kwargs": {"enable_thinking": False}}
