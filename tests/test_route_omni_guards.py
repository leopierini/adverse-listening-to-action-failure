"""Guards added to `route_omni.py` by the 2026-08-12 pre-flight audit:
  - missing audio goes to the failure sidecar, never the main output (C5)
  - a run whose worklist is >2% missing audio hard-aborts before any GPU request
  - --num-samples exists, so the 100-utterance gating pilot can actually be run (H1)

Offline: the omni call is stubbed, no network, no audio needed.
"""
import json
import sys

import pytest

import route_omni


SAMPLE = {"slurp_id": 1, "scenario": "alarm", "intent": "alarm_set",
          "sentence": "wake me at six", "entities": [], "tokens": [],
          "recordings": [{"file": "x-headset.flac"}]}


def _args(out, *extra):
    return ["route_omni.py", "--out", str(out),
            "--llm-model", "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "--llm-base-url", "http://vm:8000/v1", "--skip-model-check"] + list(extra)


def _stub(monkeypatch, samples=None, exists=True):
    monkeypatch.setattr(route_omni, "get_intent_from_omni",
                        lambda *a, **k: {"intent": "alarm_set", "parameters": {}})
    monkeypatch.setattr(route_omni, "load_samples",
                        lambda *a, **k: samples if samples is not None else [SAMPLE])
    monkeypatch.setattr(route_omni, "pick_available_recording",
                        lambda s, root: ("x-headset.flac", "x-headset.flac"))
    if exists is not None:
        monkeypatch.setattr(route_omni.os.path, "exists",
                            exists if callable(exists) else (lambda p: exists))


# ─── C5: missing audio must not poison the main output ───────────────────────

def test_missing_audio_goes_to_sidecar_not_main_output(tmp_path, monkeypatch):
    """A FEW clips missing (below the abort threshold, which is raised here so the
    run proceeds): the missing one must land in the sidecar, so a corrected re-run
    retries it instead of skipping it forever."""
    _stub(monkeypatch, exists=lambda p: "snr10" not in p)
    out = tmp_path / "omni.jsonl"
    monkeypatch.setattr(sys, "argv", _args(out, "--degradations", "clean,babble",
                                           "--snr-levels", "10",
                                           "--max-missing-audio-frac", "0.6"))
    route_omni.main()

    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["degradation"] == "clean"
    sidecar = tmp_path / "omni.failures.jsonl"
    assert sidecar.exists()
    skipped = [json.loads(l) for l in sidecar.read_text().splitlines()]
    assert len(skipped) == 1 and "missing_audio" in skipped[0]["notes"]


def test_missing_audio_is_retried_after_the_bank_is_fixed(tmp_path, monkeypatch):
    """The whole point of C5: the poisoned row must NOT be permanently 'done'."""
    _stub(monkeypatch, exists=lambda p: "snr10" not in p)
    out = tmp_path / "omni.jsonl"
    monkeypatch.setattr(sys, "argv", _args(out, "--degradations", "clean,babble",
                                           "--snr-levels", "10",
                                           "--max-missing-audio-frac", "0.6"))
    route_omni.main()
    assert len(out.read_text().splitlines()) == 1

    _stub(monkeypatch, exists=True)  # bank repaired
    monkeypatch.setattr(sys, "argv", _args(out, "--degradations", "clean,babble",
                                           "--snr-levels", "10"))
    route_omni.main()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 2
    assert sorted(r["degradation"] for r in rows) == ["babble", "clean"]


def test_run_aborts_when_most_audio_is_missing(tmp_path, monkeypatch):
    """Wrong --audio-root / incomplete rsync: abort BEFORE burning GPU time."""
    calls = []
    _stub(monkeypatch, exists=False)
    monkeypatch.setattr(route_omni, "get_intent_from_omni",
                        lambda *a, **k: calls.append(1) or {"intent": "alarm_set"})
    out = tmp_path / "omni.jsonl"
    monkeypatch.setattr(sys, "argv", _args(out, "--degradations", "clean,babble",
                                           "--snr-levels", "10"))
    with pytest.raises(SystemExit) as e:
        route_omni.main()
    assert e.value.code != 0
    assert calls == [], "must abort before issuing any request"
    assert not out.exists() or out.read_text() == ""


# ─── H1: the pilot needs a sample limit ──────────────────────────────────────

def test_num_samples_is_passed_through(tmp_path, monkeypatch):
    seen = {}

    def fake_load(dataset, scenarios, num_samples, seed, manifest_path=None):
        seen["n"] = num_samples
        return [SAMPLE]

    monkeypatch.setattr(route_omni, "get_intent_from_omni",
                        lambda *a, **k: {"intent": "alarm_set", "parameters": {}})
    monkeypatch.setattr(route_omni, "load_samples", fake_load)
    monkeypatch.setattr(route_omni, "pick_available_recording",
                        lambda s, root: ("x-headset.flac", "x-headset.flac"))
    monkeypatch.setattr(route_omni.os.path, "exists", lambda p: True)
    out = tmp_path / "omni.jsonl"
    monkeypatch.setattr(sys, "argv", _args(out, "--degradations", "clean",
                                           "--num-samples", "100"))
    route_omni.main()
    assert seen["n"] == 100


def test_num_samples_defaults_to_all(tmp_path, monkeypatch):
    seen = {}

    def fake_load(dataset, scenarios, num_samples, seed, manifest_path=None):
        seen["n"] = num_samples
        return [SAMPLE]

    monkeypatch.setattr(route_omni, "get_intent_from_omni",
                        lambda *a, **k: {"intent": "alarm_set", "parameters": {}})
    monkeypatch.setattr(route_omni, "load_samples", fake_load)
    monkeypatch.setattr(route_omni, "pick_available_recording",
                        lambda s, root: ("x-headset.flac", "x-headset.flac"))
    monkeypatch.setattr(route_omni.os.path, "exists", lambda p: True)
    out = tmp_path / "omni.jsonl"
    monkeypatch.setattr(sys, "argv", _args(out, "--degradations", "clean"))
    route_omni.main()
    assert seen["n"] == 0


# ─── provenance fields ───────────────────────────────────────────────────────

def test_omni_rows_record_thinking_request(tmp_path, monkeypatch):
    _stub(monkeypatch)
    out = tmp_path / "omni.jsonl"
    monkeypatch.setattr(sys, "argv", _args(out, "--degradations", "clean"))
    route_omni.main()
    row = json.loads(out.read_text().strip())
    assert row["llm_thinking_request"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "llm_model_served" in row
