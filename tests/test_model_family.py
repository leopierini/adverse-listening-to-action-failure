import pipeline


def test_qwen_family():
    assert pipeline.model_family_of("mlx-community/Qwen3.5-9B-MLX-8bit") == "qwen"
    assert pipeline.model_family_of("Qwen/Qwen3-Omni-30B-A3B-Instruct") == "qwen"


def test_google_family():
    assert pipeline.model_family_of("google/gemma-4-31b") == "google"
    assert pipeline.model_family_of("google/gemma-4-12b") == "google"


def test_unknown_and_none():
    assert pipeline.model_family_of("meta-llama/Llama-3-70B") is None
    assert pipeline.model_family_of(None) is None
