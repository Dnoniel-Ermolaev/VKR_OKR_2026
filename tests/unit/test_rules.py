from src.medical.rules import evaluate_hard_rules


def test_stemi_detected_as_high_risk():
    risk, level, reasons, route_to_llm = evaluate_hard_rules(
        {
            "pain_type": "typical",
            "ecg_changes": "ST-elevation",
            "troponin": 0.03,
            "hr": 88,
        }
    )
    assert risk >= 0.9
    assert level == "high"
    assert route_to_llm is False
    assert reasons
