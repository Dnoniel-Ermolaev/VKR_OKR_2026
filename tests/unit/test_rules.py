"""Тесты rule-based блока (rules.py) и детерминированного классификатора
(diagnosis.py) по клиническим рекомендациям Минздрава."""
from __future__ import annotations

import pytest

from src.medical.diagnosis import (
    AcsDiagnosis,
    DiagnosisResult,
    classify_acs,
    merge_diagnoses,
)
from src.medical.protocols import (
    NSTEMI_PROTOCOL,
    STEMI_PROTOCOL,
    UA_PROTOCOL,
    GENERIC_ACS_PROTOCOL,
    select_protocol,
)
from src.medical.rules import (
    RULEBOOK,
    RULES_BY_ID,
    evaluate_hard_rules,
    evaluate_rules,
)


# =========================================================================
# Совместимость со старым тестом скрининга ИМпST по ECG ST-elevation
# =========================================================================
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


# =========================================================================
# Базовые свойства рулбука
# =========================================================================
def test_rulebook_has_typed_rules():
    assert len(RULEBOOK) >= 15
    assert all(rule.kr_reference for rule in RULEBOOK)
    assert all(rule.id in RULES_BY_ID for rule in RULEBOOK)


def test_rulebook_categories_cover_expected_set():
    categories = {rule.category for rule in RULEBOOK}
    assert {"ecg", "biomarker", "clinical", "hemodynamic", "score", "time"} <= categories


def test_evaluate_rules_returns_structured_fires():
    fires = evaluate_rules(
        {
            "pain_type": "typical",
            "ecg_changes": "подъём ST в V2-V5",
            "troponin": 0.5,
            "hr": 95,
            "bp": "120/80",
        }
    )
    assert any(f.rule_id == "ecg_st_elev_persistent" for f in fires)
    assert any(f.rule_id == "troponin_high" for f in fires)
    sample = fires[0]
    assert sample.title_ru
    assert sample.kr_reference
    assert sample.severity in {"critical", "high", "medium", "low"}


# =========================================================================
# ЭКГ-правила
# =========================================================================
def test_ecg_new_lbbb_fires():
    fires = evaluate_rules(
        {"ecg_changes": "впервые возникшая блокада ЛНПГ", "pain_type": "typical",
         "troponin": 0.0, "hr": 80, "bp": "130/80"}
    )
    assert any(f.rule_id == "ecg_new_lbbb" for f in fires)


def test_ecg_posterior_mi_fires():
    fires = evaluate_rules(
        {"ecg_changes": "признаки заднего ИМ, V7-V9", "pain_type": "typical",
         "troponin": 0.0, "hr": 80, "bp": "120/80"}
    )
    assert any(f.rule_id == "ecg_posterior_mi" for f in fires)


def test_ecg_st_depression_fires():
    fires = evaluate_rules(
        {"ecg_changes": "депрессия ST в V4-V6", "pain_type": "typical",
         "troponin": 0.0, "hr": 80, "bp": "120/80"}
    )
    assert any(f.rule_id == "ecg_st_depression_significant" for f in fires)


def test_ecg_t_inversion_fires():
    fires = evaluate_rules(
        {"ecg_changes": "глубокая инверсия T в I, aVL, V5-V6", "pain_type": "atypical",
         "troponin": 0.0, "hr": 80, "bp": "120/80"}
    )
    assert any(f.rule_id == "ecg_t_inversion_deep" for f in fires)


def test_ecg_de_winter_pattern_fires():
    fires = evaluate_rules(
        {"ecg_changes": "паттерн de Winter в V2-V4", "pain_type": "typical",
         "troponin": 0.05, "hr": 90, "bp": "120/80"}
    )
    assert any(f.rule_id == "ecg_de_winter" for f in fires)


def test_ecg_sgarbossa_pattern_fires():
    fires = evaluate_rules(
        {"ecg_changes": "критерии Sgarbossa положительные", "pain_type": "typical",
         "troponin": 0.0, "hr": 80, "bp": "120/80"}
    )
    assert any(f.rule_id == "ecg_sgarbossa" for f in fires)


# =========================================================================
# Биомаркеры
# =========================================================================
def test_troponin_99th_percentile_rule():
    fires = evaluate_rules(
        {"ecg_changes": "норма", "pain_type": "atypical",
         "troponin": 0.05, "hr": 80, "bp": "120/80"}
    )
    assert any(f.rule_id == "troponin_99th_percentile" for f in fires)


def test_troponin_high_rule():
    fires = evaluate_rules(
        {"ecg_changes": "норма", "pain_type": "typical",
         "troponin": 0.2, "hr": 80, "bp": "120/80"}
    )
    assert any(f.rule_id == "troponin_high" for f in fires)


# =========================================================================
# Клиника и гемодинамика
# =========================================================================
def test_chest_pain_duration_rule():
    fires = evaluate_rules(
        {
            "ecg_changes": "норма",
            "pain_type": "typical",
            "symptoms_text": "Стойкая загрудинная боль более 30 минут",
            "troponin": 0.02,
            "hr": 80,
            "bp": "120/80",
        }
    )
    assert any(f.rule_id == "chest_pain_duration_gt_20min" for f in fires)


def test_hemodynamic_instability_rule():
    fires = evaluate_rules(
        {"ecg_changes": "норма", "pain_type": "typical",
         "troponin": 0.02, "hr": 110, "bp": "80/40"}
    )
    assert any(f.rule_id == "hemodynamic_instability" for f in fires)


def test_killip_3_4_rule():
    fires = evaluate_rules(
        {"ecg_changes": "норма", "pain_type": "typical",
         "troponin": 0.02, "hr": 95, "bp": "120/80", "killip_class": "III"}
    )
    assert any(f.rule_id == "killip_3_4" for f in fires)


def test_life_threatening_arrhythmia_rule():
    fires = evaluate_rules(
        {"ecg_changes": "норма", "pain_type": "typical",
         "troponin": 0.02, "hr": 95, "bp": "120/80",
         "symptoms_text": "Эпизод желудочковой тахикардии при поступлении"}
    )
    assert any(f.rule_id == "life_threatening_arrhythmia" for f in fires)


def test_mechanical_complication_rule():
    fires = evaluate_rules(
        {"ecg_changes": "норма", "pain_type": "typical",
         "troponin": 0.02, "hr": 95, "bp": "120/80",
         "symptoms_text": "Острая митральная недостаточность по ЭхоКГ"}
    )
    assert any(f.rule_id == "mechanical_complication" for f in fires)


# =========================================================================
# Демография и тайминг
# =========================================================================
def test_age_ge_75_rule():
    fires = evaluate_rules(
        {"ecg_changes": "норма", "pain_type": "atypical",
         "troponin": 0.0, "hr": 80, "bp": "120/80", "age": 82}
    )
    assert any(f.rule_id == "age_ge_75" for f in fires)


def test_ckd_creatinine_high_rule():
    fires = evaluate_rules(
        {"ecg_changes": "норма", "pain_type": "atypical",
         "troponin": 0.0, "hr": 80, "bp": "120/80", "creatinine": 180.0}
    )
    assert any(f.rule_id == "ckd_creatinine_high" for f in fires)


def test_diabetes_known_rule():
    fires = evaluate_rules(
        {"ecg_changes": "норма", "pain_type": "atypical",
         "troponin": 0.0, "hr": 80, "bp": "120/80",
         "symptoms_text": "В анамнезе сахарный диабет 2 типа"}
    )
    assert any(f.rule_id == "diabetes_known" for f in fires)


def test_pain_onset_within_12h_rule():
    fires = evaluate_rules(
        {"ecg_changes": "подъём ST в V2-V5", "pain_type": "typical",
         "troponin": 0.05, "hr": 95, "bp": "120/80",
         "symptoms_text": "Боль началась в первые 12 часов"}
    )
    assert any(f.rule_id == "pain_onset_within_12h" for f in fires)


# =========================================================================
# Детерминированный классификатор по КР
# =========================================================================
def test_classifier_detects_impst_for_st_elevation_with_persistent_pain_and_troponin():
    result = classify_acs(
        {
            "pain_type": "typical",
            "ecg_changes": "стойкий подъём ST в V2-V5",
            "troponin": 0.5,
            "hr": 95,
            "bp": "120/80",
            "symptoms_text": "Боль более 30 минут",
        }
    )
    assert isinstance(result, DiagnosisResult)
    assert result.label == AcsDiagnosis.IM_PST
    assert result.confidence >= 0.85
    assert result.icd10_suggested
    assert any("ИМпST" in c or "ST" in c for c in result.criteria_fired)


def test_classifier_oks_pst_when_troponin_not_yet_elevated():
    result = classify_acs(
        {
            "pain_type": "typical",
            "ecg_changes": "подъём ST в V2-V5",
            "troponin": 0.0,
            "hr": 95,
            "bp": "120/80",
        }
    )
    assert result.label == AcsDiagnosis.OKS_PST
    assert result.needs_repeat_troponin is True


def test_classifier_oks_bpst_for_st_depression_with_elevated_troponin():
    result = classify_acs(
        {
            "pain_type": "typical",
            "ecg_changes": "депрессия ST в V4-V6",
            "troponin": 0.06,
            "hr": 95,
            "bp": "120/80",
        }
    )
    assert result.label in {AcsDiagnosis.OKS_BPST, AcsDiagnosis.IM_BPST}


def test_classifier_ns_when_clinic_present_but_normal_troponin_and_ecg():
    result = classify_acs(
        {
            "pain_type": "typical",
            "ecg_changes": "норма",
            "troponin": 0.0,
            "hr": 80,
            "bp": "130/80",
            "symptoms_text": "Типичная загрудинная боль",
        }
    )
    assert result.label == AcsDiagnosis.NS


def test_classifier_oks_unlikely_for_clean_picture():
    result = classify_acs(
        {
            "pain_type": "none",
            "ecg_changes": "норма",
            "troponin": 0.0,
            "hr": 70,
            "bp": "120/80",
        }
    )
    assert result.label == AcsDiagnosis.OKS_UNLIKELY


def test_classifier_serialization_is_jsonable():
    result = classify_acs(
        {
            "pain_type": "typical",
            "ecg_changes": "подъём ST в V2-V5",
            "troponin": 0.6,
            "hr": 90,
            "bp": "120/80",
            "symptoms_text": "стойкий приступ",
        }
    )
    data = result.to_dict()
    assert data["label"] == AcsDiagnosis.IM_PST.value
    assert "criteria_fired" in data and isinstance(data["criteria_fired"], list)
    assert "icd10_suggested" in data and isinstance(data["icd10_suggested"], list)


def test_merge_diagnoses_picks_highest_priority():
    assert merge_diagnoses(AcsDiagnosis.NS, AcsDiagnosis.IM_PST) == AcsDiagnosis.IM_PST
    assert merge_diagnoses(AcsDiagnosis.OKS_BPST, AcsDiagnosis.OKS_PST) == AcsDiagnosis.OKS_PST
    assert merge_diagnoses(None, None) == AcsDiagnosis.OKS_UNLIKELY


# =========================================================================
# select_protocol - приоритет AcsDiagnosis над МКБ
# =========================================================================
def test_select_protocol_prioritises_acs_diagnosis():
    # Даже если МКБ говорит про НС, приоритет - ИМпST из классификатора.
    protocol = select_protocol(
        ["I20.0"],
        ecg_description="",
        triage_category="",
        acs_diagnosis=AcsDiagnosis.IM_PST,
    )
    assert protocol.code == STEMI_PROTOCOL.code


def test_select_protocol_accepts_string_acs_diagnosis():
    protocol = select_protocol(
        [],
        ecg_description="",
        triage_category="",
        acs_diagnosis="im_bpst",
    )
    assert protocol.code == NSTEMI_PROTOCOL.code


def test_select_protocol_falls_back_when_diagnosis_unknown():
    protocol = select_protocol(
        ["I20.0"],
        acs_diagnosis="not-a-real-code",
    )
    # МКБ I20.0 -> UA, поэтому при некорректной метке используем fallback по МКБ.
    assert protocol.code in {UA_PROTOCOL.code, GENERIC_ACS_PROTOCOL.code}
