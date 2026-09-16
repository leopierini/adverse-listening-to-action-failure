import route_omni


def test_run_worklist_processes_each_once(monkeypatch):
    monkeypatch.setattr(route_omni, "get_intent_from_omni",
                        lambda *a, **k: {"intent": "alarm_set", "parameters": {}})
    monkeypatch.setattr(route_omni.os.path, "exists", lambda p: True)

    worklist = []
    for i in range(20):
        row = route_omni.build_omni_row(
            {"slurp_id": i, "scenario": "alarm", "intent": "alarm_set",
             "sentence": "x", "entities": [], "tokens": []},
            audio_file=f"{i}.flac", degradation="clean", snr_db=None,
            seed=42, llm_model="Qwen/Qwen3-Omni-30B-A3B-Instruct")
        worklist.append((row, f"{i}.flac"))

    ok_ids = []
    route_omni.run_worklist(worklist, "m", "url", workers=8,
                            on_ok=lambda r: ok_ids.append(r["slurp_id"]),
                            on_fail=lambda r: None, on_skip=lambda r: None)
    assert sorted(ok_ids) == list(range(20))
