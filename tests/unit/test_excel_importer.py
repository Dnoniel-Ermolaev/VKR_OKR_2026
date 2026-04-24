"""Тесты Excel-импорта клинических данных."""
from __future__ import annotations

import io

import pytest

openpyxl = pytest.importorskip("openpyxl")

from src.infrastructure.importers.excel_importer import SHEETS, ExcelImportService


# -------------------- Шаблоны --------------------


def test_build_template_single_sheet_has_headers_and_example():
    svc = ExcelImportService()
    blob = svc.build_template("Labs")
    wb = openpyxl.load_workbook(filename=io.BytesIO(blob))
    assert "Labs" in wb.sheetnames
    assert "_README" in wb.sheetnames
    ws = wb["Labs"]
    headers = [c.value for c in ws[1]]
    assert headers == SHEETS["Labs"].headers
    # пример-строка присутствует
    assert ws.max_row >= 2


def test_build_template_full_contains_all_sheets():
    svc = ExcelImportService()
    blob = svc.build_template_full()
    wb = openpyxl.load_workbook(filename=io.BytesIO(blob))
    for name in SHEETS:
        assert name in wb.sheetnames, f"sheet {name} missing"
    assert "_README" in wb.sheetnames


def test_build_template_raises_for_unknown_sheet():
    svc = ExcelImportService()
    with pytest.raises(ValueError):
        svc.build_template("NotASheet")


# -------------------- Парсеры строк --------------------


def test_parse_lab_row_resolves_code_by_alias():
    svc = ExcelImportService()
    parsed = svc._parse_lab_row({
        "recorded_at": "2026-03-27T08:00",
        "code": "",
        "name": "Тропонин I",
        "value": 0.05,
        "unit": "нг/мл",
        "note": "",
    })
    assert parsed["code"] == "troponin_i"
    assert parsed["category"] == "lab"
    assert parsed["flag"] == "high"
    assert parsed["value_num"] == 0.05


def test_parse_lab_row_rejects_unknown_name():
    svc = ExcelImportService()
    with pytest.raises(ValueError):
        svc._parse_lab_row({"name": "Марсианский белок", "value": 1.0})


def test_parse_vitals_row_returns_observations_for_each_filled_code():
    svc = ExcelImportService()
    observations = svc._parse_vitals_row({
        "recorded_at": "2026-03-27T08:00",
        "sbp": 180,
        "dbp": "",
        "hr": 110,
        "spo2": 92,
        "note": "",
    })
    codes = {o["code"] for o in observations}
    assert codes == {"sbp", "hr", "spo2"}
    sbp = next(o for o in observations if o["code"] == "sbp")
    assert sbp["flag"] == "high"


def test_parse_medication_row_falls_back_to_freeform_name():
    svc = ExcelImportService()
    parsed = svc._parse_medication_row({
        "code": "",
        "name": "Новый антиагрегант",
        "med_class": "antiplatelet",
        "dose": "75",
        "unit": "мг",
        "route": "po",
        "frequency": "1 раз/сут",
        "started_at": "",
        "stopped_at": "",
        "status": "active",
    })
    assert parsed["name"] == "Новый антиагрегант"
    assert parsed["med_class"] == "antiplatelet"
    assert parsed["status"] == "active"


def test_parse_medication_row_rejects_empty():
    svc = ExcelImportService()
    with pytest.raises(ValueError):
        svc._parse_medication_row({"code": "", "name": ""})


def test_parse_study_resolves_by_russian_alias():
    svc = ExcelImportService()
    parsed = svc._parse_study_row({
        "code": "",
        "name": "ЭхоКГ",
        "status": "done",
        "started_at": "2026-03-27T10:00",
        "completed_at": "2026-03-27T10:20",
        "result_text": "ФВ 45%",
    })
    assert parsed["code"] == "echo_cg"
    assert parsed["status"] == "done"


def test_parse_diagnosis_normalizes_icd():
    svc = ExcelImportService()
    parsed = svc._parse_diagnosis_row({
        "icd10": "i21.0",
        "name": "",
        "diagnosis_type": "primary",
        "established_at": "2026-03-27",
        "note": "",
    })
    assert parsed["icd10"] == "I21.0"
    # auto-заполнение по каталогу
    assert "инфаркт" in parsed["name"].lower()


def test_parse_dt_handles_iso_and_excel_serial():
    svc = ExcelImportService()
    from datetime import datetime

    assert svc._parse_dt("2026-03-27T08:00") == datetime(2026, 3, 27, 8, 0)
    assert svc._parse_dt("27.03.2026 08:00") == datetime(2026, 3, 27, 8, 0)
    assert svc._parse_dt("") is None
    assert svc._parse_dt(None) is None
    # Excel serial date for 2026-03-27 ~ 46108
    serial_dt = svc._parse_dt(46108)
    assert serial_dt is not None and serial_dt.year == 2026


# -------------------- Import (dry_run) --------------------


def test_import_dry_run_reports_counts(in_memory_session):
    repo = _make_repo(in_memory_session)
    case = repo.create_case(
        patient_id=None, visit_id=None, title="t", llm_model="",
        initial_payload={}, latest_payload={},
    )

    svc = ExcelImportService(db_session=in_memory_session)
    blob = _build_populated_workbook()
    report = svc.import_case_data(case.id, blob, dry_run=True)

    assert report["dry_run"] is True
    assert report["imported_total"] >= 3
    for sheet in ("Labs", "Vitals", "Medications", "Diagnoses"):
        assert sheet in report["sheets"]
    # в dry-run БД не меняется
    assert repo.get_case_observations(case.id) == []
    assert repo.get_case_medications(case.id) == []
    assert repo.get_case_diagnoses(case.id) == []


def test_import_persists_rows(in_memory_session):
    repo = _make_repo(in_memory_session)
    case = repo.create_case(
        patient_id=None, visit_id=None, title="t", llm_model="",
        initial_payload={}, latest_payload={},
    )

    svc = ExcelImportService(db_session=in_memory_session)
    blob = _build_populated_workbook()
    report = svc.import_case_data(case.id, blob, dry_run=False)

    assert report["imported_total"] >= 3
    observations = repo.get_case_observations(case.id)
    meds = repo.get_case_medications(case.id)
    diags = repo.get_case_diagnoses(case.id)

    assert any(o.code == "troponin_i" for o in observations)
    assert any(o.code == "sbp" for o in observations)
    assert any(m.code == "asa" for m in meds)
    assert any(d.icd10 == "I21.0" for d in diags)


def test_import_reports_rejected_rows(in_memory_session):
    repo = _make_repo(in_memory_session)
    case = repo.create_case(
        patient_id=None, visit_id=None, title="t", llm_model="",
        initial_payload={}, latest_payload={},
    )

    svc = ExcelImportService(db_session=in_memory_session)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    ws = wb.create_sheet("Labs")
    ws.append(SHEETS["Labs"].headers)
    ws.append(["2026-03-27T08:00", "", "Абсолютно_неизвестный_анализ", 1.0, "", ""])
    buf = io.BytesIO()
    wb.save(buf)

    report = svc.import_case_data(case.id, buf.getvalue())
    labs = report["sheets"]["Labs"]
    assert labs["imported"] == 0
    assert len(labs["rejected"]) == 1


# -------------------- helpers --------------------


def _make_repo(session):
    from src.infrastructure.db.repository import sql_database_repository
    return sql_database_repository(session)


def _build_populated_workbook() -> bytes:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    labs = wb.create_sheet("Labs")
    labs.append(SHEETS["Labs"].headers)
    labs.append(["2026-03-27T08:00", "troponin_i", "", 0.05, "нг/мл", ""])

    vitals = wb.create_sheet("Vitals")
    vitals.append(SHEETS["Vitals"].headers)
    vitals.append(["2026-03-27T08:00", 180, 100, 110, 20, 36.8, 92, "", "", "", ""])

    meds = wb.create_sheet("Medications")
    meds.append(SHEETS["Medications"].headers)
    meds.append(["asa", "", "", "100", "мг", "po", "1 раз/сут", "2026-03-27T08:00", "", "active"])

    diags = wb.create_sheet("Diagnoses")
    diags.append(SHEETS["Diagnoses"].headers)
    diags.append(["I21.0", "", "primary", "2026-03-27T08:00", ""])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
