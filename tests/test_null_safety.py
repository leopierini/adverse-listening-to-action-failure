import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / "venv" / "bin" / "python")


def _write(tmp_path, rows):
    p = tmp_path / "in.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_compute_metrics_skips_omni(tmp_path):
    omni = {"slurp_id": 6925, "pathway": "omni", "asr_model": None,
            "transcription": None, "wer": None, "cer": None,
            "gold_parameters": {"time": "seven"}, "predicted_parameters": {}}
    inp = _write(tmp_path, [omni])
    out = tmp_path / "out.jsonl"
    r = subprocess.run([PY, str(REPO / "scripts" / "compute_metrics.py"),
                        "--in", str(inp), "--out", str(out)],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stderr
    row = json.loads(out.read_text().splitlines()[0])
    assert row["wer"] is None and row["cer"] is None  # never scored on omni


def test_annotate_errors_skips_omni(tmp_path):
    omni = {"slurp_id": 6925, "pathway": "omni", "transcription": None,
            "error_categories": []}
    inp = _write(tmp_path, [omni])
    out = tmp_path / "out.jsonl"
    r = subprocess.run([PY, str(REPO / "scripts" / "annotate_errors.py"),
                        "--in", str(inp), "--out", str(out)],
                       capture_output=True, text=True, cwd=str(REPO))
    assert r.returncode == 0, r.stderr
    row = json.loads(out.read_text().splitlines()[0])
    assert row.get("phonetic_distance") in (None,)  # not stamped for omni
    assert row["error_categories"] == []
