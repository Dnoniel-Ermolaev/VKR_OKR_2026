from src.core.case_runtime import infer_case_status, merge_payload_with_observations
from src.evaluation.metrics import auc_roc, sensitivity, specificity


def test_metrics_compute_expected_values():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]
    y_score = [0.9, 0.6, 0.4, 0.1]

    assert sensitivity(y_true, y_pred) == 0.5
    assert specificity(y_true, y_pred) == 1.0
    assert auc_roc(y_true, y_score) == 1.0


def test_case_status_and_merge_observations():
    status, stage = infer_case_status({"triage_category": "data_quality_issue", "missing_fields": ["troponin"]})
    assert status == "awaiting_labs"
    assert stage == "awaiting_labs"

    payload = {
        "name": "Ivan",
        "pain_type": "typical",
        "ecg_changes": "ST-depression",
        "troponin": 0.1,
        "hr": 100,
        "bp": "130/80",
    }
    merged = merge_payload_with_observations(
        payload,
        [
            {"category": "lab", "name": "troponin", "value_num": 0.22, "unit": "ng/mL"},
            {"category": "vital", "name": "spo2", "value_num": 93.0, "unit": "%"},
        ],
    )
    assert merged["troponin"] == 0.22
    assert merged["spo2"] == 93.0
    assert merged["vital_signs"]
