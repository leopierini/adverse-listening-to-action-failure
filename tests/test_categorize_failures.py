import categorize_failures as cf


def test_tsh_missing():
    assert cf.tsh_category({"predicted_intent": "JSON_PARSE_ERROR"}) == "missing"
    assert cf.tsh_category({"predicted_intent": None}) == "missing"


def test_tsh_unknown_reported_separately():
    assert cf.tsh_category({"predicted_intent": "unknown"}) == "unknown"


def test_tsh_hallucinated_label_outside_ontology():
    assert cf.tsh_category({"predicted_intent": "set_alarm"}) == "hallucinated"


def test_tsh_wrong_valid_label():
    assert cf.tsh_category({"predicted_intent": "alarm_set"}) == "wrong"


def test_tch_schema_mismatch_no_params():
    assert cf.tch_category({"gold_parameters": {"time": "seven"},
                            "predicted_parameters": {}}) == "schema_mismatch"


def test_tch_fabricated_no_value_matches():
    assert cf.tch_category({"gold_parameters": {"time": "seven"},
                            "predicted_parameters": {"time": "next friday"}}) == "fabricated"


def test_tch_semantic_drift_partial_match():
    assert cf.tch_category({"gold_parameters": {"time": "seven am", "person": "sarah"},
                            "predicted_parameters": {"time": "7 am"}}) == "semantic_drift"
