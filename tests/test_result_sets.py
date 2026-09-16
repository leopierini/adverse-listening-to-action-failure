"""Unit tests for scripts/result_sets.py — which result files are THE SWEEP.

Written 2026-08-18. Nothing in the repo said which of results/*.jsonl are the
experimental sweep and which are pilots or baselines, so every consumer used a
bare glob. When the 200-row H2a pilot landed on 2026-08-17, eight claims in
verify_tracker_claims.py went red and fit_mixed_effects.py fitted a router
coefficient off 2 of 57 design cells. Neither the docs nor the data were wrong;
the file selection was.
"""
import pytest

import result_sets as rs


def touch(d, *names):
    for n in names:
        (d / n).write_text("")
    return d


def test_sweep_is_the_cascade_and_parakeet_scored_files(tmp_path):
    touch(tmp_path,
          "cascade_whisper_qwen9b_clean_scored.jsonl",
          "cascade_whisper_qwen9b_babble_snr0_scored.jsonl",
          "parakeet_clean_qwen9b_scored.jsonl",
          "parakeet_reverb_snr10_qwen9b_scored.jsonl")

    names = [p.name for p in rs.sweep_files(tmp_path)]

    assert len(names) == 4
    assert all(n.endswith("_scored.jsonl") for n in names)


def test_pilots_and_baselines_are_not_part_of_the_sweep(tmp_path):
    touch(tmp_path,
          "cascade_whisper_qwen9b_clean_scored.jsonl",
          "h2a_pilot_27b_scored.jsonl",
          "h2a_pilot_27b_babble_scored.jsonl",
          "omni_pilot_qwen_scored.jsonl",
          "baseline_clean_n50_scored.jsonl")

    sweep = [p.name for p in rs.sweep_files(tmp_path)]
    pilots = [p.name for p in rs.pilot_files(tmp_path)]

    assert sweep == ["cascade_whisper_qwen9b_clean_scored.jsonl"]
    assert sorted(pilots) == ["h2a_pilot_27b_babble_scored.jsonl",
                              "h2a_pilot_27b_scored.jsonl",
                              "omni_pilot_qwen_scored.jsonl"]


def test_intermediate_stages_are_not_mistaken_for_results(tmp_path):
    """_m (metrics) and _ann (annotated) are half-scored intermediates. Reading
    them as results would double-count every utterance."""
    touch(tmp_path,
          "cascade_whisper_qwen9b_clean.jsonl",
          "cascade_whisper_qwen9b_clean_m.jsonl",
          "cascade_whisper_qwen9b_clean_ann.jsonl",
          "cascade_whisper_qwen9b_clean_scored.jsonl")

    assert [p.name for p in rs.sweep_files(tmp_path)] == \
        ["cascade_whisper_qwen9b_clean_scored.jsonl"]


def test_an_unrecognised_scored_file_is_reported_not_silently_dropped(tmp_path):
    """A new arm lands in results/ one day. Silently excluding it would make
    every count quietly wrong — the failure this module exists to prevent."""
    touch(tmp_path,
          "cascade_whisper_qwen9b_clean_scored.jsonl",
          "gemma31b_sweep_clean_scored.jsonl")

    with pytest.raises(rs.UnclassifiedResultFile) as excinfo:
        rs.sweep_files(tmp_path)
    assert "gemma31b_sweep_clean_scored.jsonl" in str(excinfo.value)


# --------------------------------------------------------------------------
# The Qwen-27B arm, collected 2026-08-27 (Step 6, model 1 of 4)
# --------------------------------------------------------------------------
# The naming rule in the runbook SUBSTITUTES the router tag rather than
# appending it, so the tag lands mid-name on the cascade files and at the end
# on the parakeet ones. Both shapes have to be recognised, or every count
# check raises UnclassifiedResultFile on 38 real files.

def test_the_qwen27b_arm_is_part_of_the_sweep(tmp_path):
    touch(tmp_path,
           "cascade_whisper_qwen27b_clean_scored.jsonl",
           "cascade_whisper_qwen27b_babble_snr0_scored.jsonl",
           "parakeet_clean_qwen27b_scored.jsonl",
           "parakeet_reverb_snr10_qwen27b_scored.jsonl")
    names = [p.name for p in rs.sweep_files(tmp_path)]
    assert names == sorted(names)
    assert len(names) == 4
    assert "cascade_whisper_qwen27b_babble_snr0_scored.jsonl" in names
    assert "parakeet_clean_qwen27b_scored.jsonl" in names


def test_both_router_arms_live_in_the_sweep_together(tmp_path):
    touch(tmp_path,
           "cascade_whisper_qwen9b_clean_scored.jsonl",
           "cascade_whisper_qwen27b_clean_scored.jsonl")
    assert len(rs.sweep_files(tmp_path)) == 2


def test_a_filename_naming_two_routers_is_still_refused(tmp_path):
    """`cascade_whisper_qwen9b_babble_snr0_qwen27b_scored.jsonl` is how the
    76-file bug announces itself — the tag appended instead of substituted."""
    touch(tmp_path, "cascade_whisper_qwen9b_babble_snr0_qwen27b_scored.jsonl")
    with pytest.raises(rs.UnclassifiedResultFile):
        rs.sweep_files(tmp_path)


# --------------------------------------------------------------------------
# The omni pathway, collected 2026-08-27 (Step 6, model 3 of 4)
# --------------------------------------------------------------------------
# These belong to the sweep, not to the pilots: they are 19 of the design's
# 209 cells. generate_status.py already anticipates them — it derives the
# "pathways present" row from `pathway` over sweep_files() — and TOTAL_CELLS
# counts 171 cascade + 38 omni.

def test_the_omni_sweeps_are_part_of_the_sweep(tmp_path):
    touch(tmp_path, "omni_qwen_sweep_scored.jsonl", "omni_gemma_sweep_scored.jsonl")
    names = sorted(p.name for p in rs.sweep_files(tmp_path))
    assert names == ["omni_gemma_sweep_scored.jsonl", "omni_qwen_sweep_scored.jsonl"]


def test_the_omni_PILOT_is_still_a_pilot_not_the_sweep(tmp_path):
    """`omni_pilot_qwen_scored.jsonl` is the 200-row session-1 probe. The two
    names differ by one word and belong to opposite sets."""
    touch(tmp_path, "omni_qwen_sweep_scored.jsonl", "omni_pilot_qwen_scored.jsonl")
    assert [p.name for p in rs.sweep_files(tmp_path)] == ["omni_qwen_sweep_scored.jsonl"]
    assert [p.name for p in rs.pilot_files(tmp_path)] == ["omni_pilot_qwen_scored.jsonl"]


def test_an_invented_omni_name_is_still_refused(tmp_path):
    """The runbook fixes the two output names. Anything else must force a
    decision rather than drift into the counts."""
    touch(tmp_path, "omni_qwen3_sweep_scored.jsonl")
    with pytest.raises(rs.UnclassifiedResultFile):
        rs.sweep_files(tmp_path)


# --------------------------------------------------------------------------
# The Gemma-31B arm, collected 2026-08-27 (Step 6, model 2 of 4)
# --------------------------------------------------------------------------
# It exists only because vLLM was upgraded to 0.28.0 in a second environment;
# on 0.27.1 the model would not start at all (§15). Same naming rule as every
# other cascade arm: the tag SUBSTITUTES, so it lands mid-name on the cascade
# files and at the end on the parakeet ones.

def test_the_gemma31b_arm_is_part_of_the_sweep(tmp_path):
    touch(tmp_path,
          "cascade_whisper_gemma31b_clean_scored.jsonl",
          "cascade_whisper_gemma31b_babble_snr0_scored.jsonl",
          "parakeet_clean_gemma31b_scored.jsonl",
          "parakeet_reverb_snr10_gemma31b_scored.jsonl")
    assert len(rs.sweep_files(tmp_path)) == 4


def test_all_three_router_arms_coexist(tmp_path):
    touch(tmp_path,
          "cascade_whisper_qwen9b_clean_scored.jsonl",
          "cascade_whisper_qwen27b_clean_scored.jsonl",
          "cascade_whisper_gemma31b_clean_scored.jsonl")
    assert len(rs.cascade_files(tmp_path)) == 3


def test_a_gemma_filename_naming_two_routers_is_refused(tmp_path):
    touch(tmp_path, "cascade_whisper_qwen27b_clean_gemma31b_scored.jsonl")
    with pytest.raises(rs.UnclassifiedResultFile):
        rs.sweep_files(tmp_path)
