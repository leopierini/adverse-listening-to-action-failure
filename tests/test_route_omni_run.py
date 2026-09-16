import json

import route_omni


def test_end_to_end_with_stub(tmp_path, monkeypatch):
    # Stub the omni call so no server is needed. `--skip-model-check` added
    # 2026-08-12: the startup GET /v1/models guard (audit finding H2) would
    # otherwise abort here — and this test was quietly making a real connection
    # attempt to localhost:8000, which a unit test must never do.
    monkeypatch.setattr(route_omni, "get_intent_from_omni",
                        lambda *a, **k: {"intent": "alarm_set", "parameters": {"time": "6 am"}})
    monkeypatch.setattr(route_omni.os.path, "exists", lambda p: True)

    sample = {"slurp_id": 1, "scenario": "alarm", "intent": "alarm_set",
              "sentence": "wake me at six", "entities": [], "tokens": [],
              "recordings": [{"file": "x-headset.flac"}]}
    monkeypatch.setattr(route_omni, "load_samples", lambda *a, **k: [sample])
    monkeypatch.setattr(route_omni, "pick_available_recording",
                        lambda s, root: ("x-headset.flac", "x-headset.flac"))

    out = tmp_path / "omni.jsonl"
    import sys
    monkeypatch.setattr(sys, "argv", [
        "route_omni.py", "--out", str(out),
        "--degradations", "clean,babble", "--snr-levels", "10",
        "--llm-model", "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "--skip-model-check"])
    route_omni.main()

    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 2  # clean + babble@10
    assert all(r["pathway"] == "omni" and r["tsa"] == 1 for r in rows)

    # Resume: a second run must add zero rows.
    route_omni.main()
    rows2 = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows2) == 2
