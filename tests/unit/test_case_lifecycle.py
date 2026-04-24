"""Тесты жизненного цикла кейса в репозитории."""
from __future__ import annotations

from datetime import datetime, timezone

from src.infrastructure.db.repository import sql_database_repository


def _make_patient_and_visit(session):
    from src.infrastructure.db.models import Patient, Visit

    patient = Patient(full_name="Test Patient", birth_date=datetime(1960, 1, 1).date(), gender="male")
    session.add(patient)
    session.flush()
    visit = Visit(patient_id=patient.id, admission_time=datetime.now(timezone.utc))
    session.add(visit)
    session.commit()
    return patient, visit


def _create_active_case(repo, patient_id: int, visit_id: int):
    return repo.create_case(
        patient_id=patient_id,
        visit_id=visit_id,
        title="ACS",
        llm_model="",
        initial_payload={},
        latest_payload={},
    )


def test_get_active_case_returns_none_initially(in_memory_session):
    repo = sql_database_repository(in_memory_session)
    patient, visit = _make_patient_and_visit(in_memory_session)
    assert repo.get_active_case(patient.id, visit.id) is None


def test_get_active_case_returns_existing(in_memory_session):
    repo = sql_database_repository(in_memory_session)
    patient, visit = _make_patient_and_visit(in_memory_session)
    case = _create_active_case(repo, patient.id, visit.id)
    active = repo.get_active_case(patient.id, visit.id)
    assert active is not None
    assert active.id == case.id
    assert active.status in {"active", "awaiting_labs"}


def test_close_case_sets_completed_and_closed_at(in_memory_session):
    repo = sql_database_repository(in_memory_session)
    patient, visit = _make_patient_and_visit(in_memory_session)
    case = _create_active_case(repo, patient.id, visit.id)
    closed = repo.close_case(case.id)
    assert closed is not None
    assert closed.status == "completed"
    assert closed.closed_at is not None
    # после закрытия активных кейсов нет
    assert repo.get_active_case(patient.id, visit.id) is None


def test_reopen_case_restores_active_status(in_memory_session):
    repo = sql_database_repository(in_memory_session)
    patient, visit = _make_patient_and_visit(in_memory_session)
    case = _create_active_case(repo, patient.id, visit.id)
    repo.close_case(case.id)
    reopened = repo.reopen_case(case.id)
    assert reopened is not None
    assert reopened.status == "active"
    assert reopened.closed_at is None
    assert repo.get_active_case(patient.id, visit.id).id == case.id


def test_close_case_returns_none_for_unknown_id(in_memory_session):
    repo = sql_database_repository(in_memory_session)
    assert repo.close_case("not-a-real-case") is None


def test_add_case_observations_persists_code_and_flag(in_memory_session):
    repo = sql_database_repository(in_memory_session)
    patient, visit = _make_patient_and_visit(in_memory_session)
    case = _create_active_case(repo, patient.id, visit.id)
    repo.add_case_observations(case.id, [
        {"category": "lab", "code": "troponin_i", "name": "Тропонин I",
         "value_num": 0.1, "unit": "нг/мл", "flag": "critical_high"},
    ])
    observations = repo.get_case_observations(case.id)
    assert len(observations) == 1
    assert observations[0].code == "troponin_i"
    assert observations[0].flag == "critical_high"


def test_add_case_study_and_procedure_and_medication(in_memory_session):
    repo = sql_database_repository(in_memory_session)
    patient, visit = _make_patient_and_visit(in_memory_session)
    case = _create_active_case(repo, patient.id, visit.id)

    repo.add_case_study(case.id, code="ecg_12", name="ЭКГ", status="done")
    repo.add_case_procedure(case.id, code="pci_stent", name="ЧКВ", status="done")
    repo.add_case_medication(case.id, code="asa", name="ASA", med_class="antiplatelet",
                             dose="100", unit="мг", route="po", status="active")
    repo.add_case_diagnosis(case.id, icd10="I21.0", name="ОИМ", diagnosis_type="primary")

    assert len(repo.get_case_studies(case.id)) == 1
    assert len(repo.get_case_procedures(case.id)) == 1
    assert len(repo.get_case_medications(case.id)) == 1
    assert len(repo.get_case_diagnoses(case.id)) == 1
