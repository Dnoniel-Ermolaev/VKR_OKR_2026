"""Тесты справочников клинических сущностей."""
from __future__ import annotations

from src.medical import catalog


def test_all_catalog_codes_are_unique():
    for items, key in (
        (catalog.VITALS, "code"),
        (catalog.LABS, "code"),
        (catalog.STUDIES, "code"),
        (catalog.PROCEDURES, "code"),
        (catalog.MEDICATIONS, "code"),
        (catalog.DIAGNOSES, "icd10"),
    ):
        codes = [getattr(item, key) for item in items]
        assert len(codes) == len(set(codes)), f"duplicate codes in {items[0].__class__.__name__}: {codes}"


def test_catalog_as_json_has_all_sections():
    snapshot = catalog.catalog_as_json()
    for key in ("vitals", "labs", "studies", "procedures", "medications", "diagnoses"):
        assert key in snapshot
        assert isinstance(snapshot[key], list)
        assert snapshot[key], f"section '{key}' must not be empty"


def test_resolve_lab_handles_code_name_and_alias():
    assert catalog.resolve_lab("troponin_i") == "troponin_i"
    assert catalog.resolve_lab("Тропонин I") == "troponin_i"
    assert catalog.resolve_lab("тропонин") == "troponin_i"
    assert catalog.resolve_lab("  Creatinine  ".lower()) in {None, "creatinine_blood"}
    assert catalog.resolve_lab("") is None
    assert catalog.resolve_lab("   ") is None


def test_resolve_medication_accepts_aliases():
    assert catalog.resolve_medication("aspirin") == "asa"
    assert catalog.resolve_medication("Клопидогрел") == "clopidogrel"
    assert catalog.resolve_medication("nope") is None


def test_resolve_study_and_procedure():
    assert catalog.resolve_study("экг") == "ecg_12"
    assert catalog.resolve_study("ЭхоКГ") == "echo_cg"
    assert catalog.resolve_procedure("стентирование") == "pci_stent"
    assert catalog.resolve_procedure("ИВЛ") == "mechanical_ventilation"


def test_flag_for_lab_thresholds():
    assert catalog.flag_for_lab("troponin_i", None) == "unknown"
    assert catalog.flag_for_lab("troponin_i", 0.01) == "norm"
    assert catalog.flag_for_lab("troponin_i", 0.05) == "high"
    assert catalog.flag_for_lab("troponin_i", 0.5) == "critical_high"
    assert catalog.flag_for_lab("k_blood", 2.8) == "critical_low"
    assert catalog.flag_for_lab("unknown_code", 1.0) == "unknown"


def test_flag_for_vital_thresholds():
    assert catalog.flag_for_vital("sbp", 120) == "norm"
    assert catalog.flag_for_vital("sbp", 90) == "low"
    assert catalog.flag_for_vital("sbp", 210) == "critical_high"
    assert catalog.flag_for_vital("spo2", 88) == "critical_low"


def test_diagnosis_icd_lookup():
    assert "I21.0" in catalog.DIAGNOSIS_BY_ICD
    stemi = catalog.DIAGNOSIS_BY_ICD["I21.0"]
    assert stemi.group == "STEMI"
