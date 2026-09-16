import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _row_dict_keys(script_name):
    """Statically find any dict literal in the script that has a 'slurp_id' key,
    and return its key set — avoids importing heavy ASR deps."""
    src = (REPO / "scripts" / script_name).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if "slurp_id" in keys:
                return set(keys)
    return set()


def test_batch_evaluate_row_has_pathway():
    keys = _row_dict_keys("batch_evaluate.py")
    assert "pathway" in keys and "model_family" in keys


def test_transcribe_parakeet_row_has_pathway():
    keys = _row_dict_keys("transcribe_parakeet.py")
    assert "pathway" in keys and "model_family" in keys
