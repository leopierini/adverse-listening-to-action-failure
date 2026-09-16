import annotate_errors as ae


def test_annotate_row_returns_triple(sample_gold):
    row = {"slurp_id": 999, "transcription": "set an alarm for sara at seven am",
           "wer": 0.2}
    cats, ref_t, hyp_t = ae.annotate_row(row, sample_gold)
    assert isinstance(cats, list)
    assert "sara" in hyp_t and "sarah" in ref_t
    # sarah→sara is a critical-slot substitution → phonetic_distance is defined
    pd = ae.row_phonetic_distance(cats)
    assert pd is not None
