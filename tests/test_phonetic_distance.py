import annotate_errors as ae


def test_dm_distance_near_zero_for_homophones():
    # sara / sarah share the Double-Metaphone code → distance ~0
    assert ae.double_metaphone_distance("sara", "sarah") < 0.2


def test_dm_distance_large_for_unrelated_names():
    assert ae.double_metaphone_distance("sarah", "david") > 0.6


def test_dm_distance_bounded():
    d = ae.double_metaphone_distance("seven", "heaven")
    assert 0.0 <= d <= 1.0


def test_dm_distance_empty_inputs():
    assert ae.double_metaphone_distance("", "") == 0.0
    assert ae.double_metaphone_distance("sarah", "") == 1.0


def test_row_phonetic_distance_only_critical_substitutions():
    cats = [
        {"type": "substitute", "slot_position": "critical", "ref": "sarah", "hyp": "sara"},
        {"type": "substitute", "slot_position": "function", "ref": "the", "hyp": "a"},
        {"type": "delete", "slot_position": "critical", "ref": "at", "hyp": ""},
    ]
    d = ae.row_phonetic_distance(cats)
    assert d is not None and d < 0.2  # only the critical substitution counts


def test_row_phonetic_distance_none_when_no_critical_sub():
    cats = [{"type": "delete", "slot_position": "critical", "ref": "at", "hyp": ""}]
    assert ae.row_phonetic_distance(cats) is None


def test_full_hallucination_flag():
    assert ae.is_full_hallucination(1.0, ["set", "alarm", "seven"], ["the", "weather", "today"]) is True
    assert ae.is_full_hallucination(1.0, ["set", "alarm"], ["set", "clock"]) is False  # "set" overlaps
    assert ae.is_full_hallucination(0.3, ["a"], ["b"]) is False  # WER below threshold
    assert ae.is_full_hallucination(None, ["a"], ["b"]) is False
    # Empty hypothesis = TOTAL DELETION (its own category), NOT a hallucination:
    assert ae.is_full_hallucination(1.0, ["set", "alarm"], []) is False
