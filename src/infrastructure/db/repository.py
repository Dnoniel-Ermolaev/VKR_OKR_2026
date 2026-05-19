# src/infrastructure/db/repository.py
from __future__ import annotations
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from src.infrastructure.db.models import (
    CaseAssessment,
    CaseDiagnosis,
    CaseMedication,
    CaseObservation,
    CaseProcedure,
    CaseStudy,
    CaseTrackingItem,
    ClinicalReport,
    Patient,
    PatientRecord,
    TriageCase,
    Visit,
)

try:
    import pandas as pd
except Exception:
    pd = None

DEFAULT_COLUMNS = [
    "id",
    "name",
    "symptoms_json",
    "ecg_desc",
    "troponin",
    "hr",
    "bp",
    "risk_level",
    "explanation",
    "timestamp",
]


class PatientRepository:
    """ Репозиторий для хранения пациентов в CSV-файле """

    def __init__(self, csv_path: Path) -> None:
        """ Подготовка CSV-файла для хранения пациентов """
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            self._init_empty()

    def save_patient(self, record: PatientRecord) -> None:
        """ Сохранение пациента в CSV-файл """
        if pd is not None:
            df = pd.read_csv(self.csv_path)
            df = pd.concat([df, pd.DataFrame([record.model_dump()])], ignore_index=True)
            df.to_csv(self.csv_path, index=False)
            return
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DEFAULT_COLUMNS)
            writer.writerow(record.model_dump())

    def search_patients(
        self,
        *,
        name: Optional[str] = None,
        risk_level: Optional[str] = None,
        top_k: int = 20,
    ) -> List[Dict[str, object]]:
        """ Поиск пациентов в CSV-файле """
        if pd is not None:
            df = pd.read_csv(self.csv_path)
            if name:
                df = df[df["name"].astype(str).str.contains(name, case=False, na=False)]
            if risk_level:
                df = df[df["risk_level"] == risk_level]
            return df.head(top_k).to_dict(orient="records")

        with self.csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        filtered: List[Dict[str, object]] = []
        for row in rows:
            if name and name.lower() not in str(row.get("name", "")).lower():
                continue
            if risk_level and str(row.get("risk_level")) != risk_level:
                continue
            filtered.append(row)
        return filtered[:top_k]

    def _init_empty(self) -> None:
        """ Создание пустого CSV-файла с нужными колонками """
        if pd is not None:
            pd.DataFrame(columns=DEFAULT_COLUMNS).to_csv(self.csv_path, index=False)
            return
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DEFAULT_COLUMNS)
            writer.writeheader()

# ---------- Классы для использования ORM SQLAlchemy ----------

class sql_database_repository:
    """ Репозиторий для работы с базой данных через SQLAlchemy """

    def __init__(self, session: Session):
        """ Подключение сессии SQLAlchemy """
        self.session = session

    def get_all_patients(self):
        """ Список пациентов из базы """
        return self.session.query(Patient).all()

    def get_patient_full_details(self, patient_id: int):
        """ Данные о пациенте """
        return self.session.query(Patient).filter(Patient.id == patient_id).first()

    def get_visit(self, visit_id: int) -> Visit | None:
        """ Данные о визите пациента """
        return self.session.query(Visit).filter(Visit.id == visit_id).first()

    def create_case(
        self,
        *,
        patient_id: int | None,
        title: str,
        llm_model: str,
        initial_payload: Dict[str, Any],
        latest_payload: Dict[str, Any],
    ) -> TriageCase:
        """ Создание клинического случая """
        case = TriageCase(
            patient_id=patient_id,
            title=title,
            llm_model=llm_model,
            initial_payload=initial_payload,
            latest_payload=latest_payload,
        )
        self.session.add(case)
        self.session.commit()
        self.session.refresh(case)
        return case

    def get_case(self, case_id: str) -> TriageCase | None:
        """ Данные клинического случая """
        return self.session.query(TriageCase).filter(TriageCase.id == case_id).first()

    def get_case_observations(self, case_id: str) -> List[CaseObservation]:
        """ Список наблюдений клинического случая """
        return (
            self.session.query(CaseObservation)
            .filter(CaseObservation.case_id == case_id)
            .order_by(CaseObservation.recorded_at.asc(), CaseObservation.id.asc())
            .all()
        )

    def get_case_assessments(self, case_id: str) -> List[CaseAssessment]:
        """ Список оценок риска клинического случая """
        return (
            self.session.query(CaseAssessment)
            .filter(CaseAssessment.case_id == case_id)
            .order_by(CaseAssessment.created_at.asc(), CaseAssessment.id.asc())
            .all()
        )

    def get_case_reports(self, case_id: str) -> List[ClinicalReport]:
        """ Список клинических отчетов по случаю """
        return (
            self.session.query(ClinicalReport)
            .filter(ClinicalReport.case_id == case_id)
            .order_by(ClinicalReport.created_at.asc(), ClinicalReport.id.asc())
            .all()
        )

    def get_case_tracking_items(self, case_id: str) -> List[CaseTrackingItem]:
        """ Список задач наблюдения по случаю """
        return (
            self.session.query(CaseTrackingItem)
            .filter(CaseTrackingItem.case_id == case_id)
            .order_by(CaseTrackingItem.created_at.asc(), CaseTrackingItem.id.asc())
            .all()
        )

    def list_patient_cases(self, patient_id: int) -> List[TriageCase]:
        """ Список клинических случаев пациента """
        return (
            self.session.query(TriageCase)
            .filter(TriageCase.patient_id == patient_id)
            .order_by(TriageCase.created_at.desc())
            .all()
        )

    def add_case_observations(self, case_id: str, observations: List[Dict[str, Any]]) -> List[CaseObservation]:
        """ Добавление наблюдений к клиническому случаю """
        created: List[CaseObservation] = []
        for item in observations:
            recorded_at = self._parse_dt(item.get("recorded_at")) or datetime.now(timezone.utc)
            obs = CaseObservation(
                case_id=case_id,
                category=str(item.get("category", "vital")),
                code=str(item.get("code", "")).strip(),
                name=str(item.get("name", "")).strip(),
                value_num=float(item["value_num"]) if item.get("value_num") is not None else None,
                value_text=str(item["value_text"]) if item.get("value_text") is not None else None,
                unit=str(item.get("unit", "")),
                flag=str(item.get("flag", "unknown")),
                source=str(item.get("source", "manual")),
                note=str(item.get("note", "")),
                recorded_at=recorded_at,
            )
            self.session.add(obs)
            created.append(obs)
        self.session.commit()
        for obs in created:
            self.session.refresh(obs)
        return created

    def update_case_observation(self, observation_id: int, **fields: Any) -> CaseObservation | None:
        """ Обновление наблюдения клинического случая """
        obs = self.session.query(CaseObservation).filter(CaseObservation.id == observation_id).first()
        if obs is None:
            return None
        for key, value in fields.items():
            if key == "recorded_at":
                value = self._parse_dt(value) or obs.recorded_at
            if hasattr(obs, key):
                setattr(obs, key, value)
        self.session.commit()
        self.session.refresh(obs)
        return obs

    def delete_case_observation(self, observation_id: int) -> bool:
        """ Удаление наблюдения клинического случая """
        obs = self.session.query(CaseObservation).filter(CaseObservation.id == observation_id).first()
        if obs is None:
            return False
        self.session.delete(obs)
        self.session.commit()
        return True

    def save_case_assessment(
        self,
        *,
        case_id: str,
        run_kind: str,
        payload_snapshot: Dict[str, Any],
        result: Dict[str, Any],
    ) -> CaseAssessment:
        """ Сохранение оценки риска клинического случая """
        path_trace = _build_path_trace_payload(result)
        acs_diagnosis = result.get("acs_diagnosis") or {}
        assessment = CaseAssessment(
            case_id=case_id,
            run_kind=run_kind,
            payload_snapshot=payload_snapshot,
            risk=float(result.get("risk", 0.0)),
            risk_level=str(result.get("risk_level", "low")),
            triage_category=str(result.get("triage_category", "")),
            next_step=str(result.get("next_step", "")),
            route_reason=str(result.get("route_reason", "")),
            explanation=str(result.get("explanation", "")),
            citations_json=list(result.get("citations", [])),
            missing_fields_json=list(result.get("missing_fields", [])),
            acs_diagnosis_json=dict(acs_diagnosis) if isinstance(acs_diagnosis, dict) else {},
            path_trace_json=path_trace,
            llm_used=bool(result.get("llm_used", False)),
        )
        self.session.add(assessment)
        self.session.commit()
        self.session.refresh(assessment)
        return assessment

    def update_case_state(
        self,
        *,
        case_id: str,
        latest_payload: Dict[str, Any],
        result: Dict[str, Any],
        status: str,
        current_stage: str,
    ) -> TriageCase:
        """ Обновление состояния клинического случая """
        case = self.get_case(case_id)
        if case is None:
            raise ValueError("Case not found")
        case.latest_payload = latest_payload
        case.latest_risk = float(result.get("risk", 0.0))
        case.latest_risk_level = str(result.get("risk_level", "low"))
        case.latest_triage_category = str(result.get("triage_category", ""))
        case.latest_next_step = str(result.get("next_step", ""))
        case.latest_explanation = str(result.get("explanation", ""))
        case.latest_citations = list(result.get("citations", []))
        case.missing_fields_json = list(result.get("missing_fields", []))
        acs_diagnosis = result.get("acs_diagnosis") or {}
        if isinstance(acs_diagnosis, dict):
            case.latest_acs_diagnosis = dict(acs_diagnosis)
        case.latest_path_trace_json = _build_path_trace_payload(result)
        case.status = status
        case.current_stage = current_stage
        case.updated_at = datetime.now(timezone.utc)
        if status == "completed":
            case.closed_at = datetime.now(timezone.utc)
        self.session.commit()
        self.session.refresh(case)
        return case

    def save_case_report(
        self,
        *,
        case_id: str,
        report_type: str,
        content: str,
        citations: List[str],
    ) -> ClinicalReport:
        """ Сохранение клинического отчета """
        report = ClinicalReport(
            case_id=case_id,
            report_type=report_type,
            content=content,
            citations_json=citations,
        )
        self.session.add(report)
        self.session.commit()
        self.session.refresh(report)
        return report

    def replace_case_tracking_items(self, case_id: str, items: List[Dict[str, Any]]) -> List[CaseTrackingItem]:
        """ Замена списка задач наблюдения по случаю """
        existing = self.session.query(CaseTrackingItem).filter(CaseTrackingItem.case_id == case_id).all()
        for row in existing:
            self.session.delete(row)
        created: List[CaseTrackingItem] = []
        now = datetime.now(timezone.utc)
        for item in items:
            due_at = self._parse_dt(item.get("due_at"))
            completed_at = self._parse_dt(item.get("completed_at"))
            row = CaseTrackingItem(
                case_id=case_id,
                item_type=str(item.get("item_type", "task")),
                name=str(item.get("name", "")).strip(),
                status=str(item.get("status", "pending")),
                priority=str(item.get("priority", "medium")),
                due_at=due_at,
                completed_at=completed_at,
                result_summary=str(item.get("result_summary", "")),
                metadata_json=dict(item.get("metadata_json", {})),
                source_page=int(item["source_page"]) if item.get("source_page") is not None else None,
                created_at=now,
                updated_at=now,
            )
            self.session.add(row)
            created.append(row)
        self.session.commit()
        for row in created:
            self.session.refresh(row)
        return created

    # ---------- Жизненный цикл клинического случая ----------
    def get_active_case(self, patient_id: int | None) -> TriageCase | None:
        """ Активный клинический случай пациента """
        query = self.session.query(TriageCase).filter(TriageCase.status.in_(["active", "awaiting_labs"]))
        if patient_id is not None:
            query = query.filter(TriageCase.patient_id == patient_id)
        return query.order_by(TriageCase.created_at.desc()).first()

    def close_case(self, case_id: str) -> TriageCase | None:
        """ Закрытие клинического случая """
        case = self.get_case(case_id)
        if case is None:
            return None
        case.status = "completed"
        case.current_stage = "closed"
        case.closed_at = datetime.now(timezone.utc)
        case.updated_at = datetime.now(timezone.utc)
        self.session.commit()
        self.session.refresh(case)
        return case

    def reopen_case(self, case_id: str) -> TriageCase | None:
        """ Повторное открытие клинического случая """
        case = self.get_case(case_id)
        if case is None:
            return None
        case.status = "active"
        case.current_stage = "resumed"
        case.closed_at = None
        case.updated_at = datetime.now(timezone.utc)
        self.session.commit()
        self.session.refresh(case)
        return case

    def delete_case(self, case_id: str) -> bool:
        """ Удаление клинического случая """
        case = self.get_case(case_id)
        if case is None:
            return False
        self.session.delete(case)
        self.session.commit()
        return True

    # ---------- Исследования ----------
    def get_case_studies(self, case_id: str) -> List[CaseStudy]:
        """ Список исследований по клиническому случаю """
        return (
            self.session.query(CaseStudy)
            .filter(CaseStudy.case_id == case_id)
            .order_by(CaseStudy.started_at.asc().nullslast(), CaseStudy.id.asc())
            .all()
        )

    def add_case_study(self, case_id: str, **fields: Any) -> CaseStudy:
        """ Добавление исследования к клиническому случаю """
        started_at = self._parse_dt(fields.get("started_at"))
        completed_at = self._parse_dt(fields.get("completed_at"))
        study = CaseStudy(
            case_id=case_id,
            code=str(fields.get("code", "")).strip(),
            name=str(fields.get("name", "")).strip(),
            status=str(fields.get("status", "ordered")),
            started_at=started_at,
            completed_at=completed_at,
            result_text=str(fields.get("result_text", "")),
            result_json=dict(fields.get("result_json", {}) or {}),
            ordered_by=str(fields.get("ordered_by", "")),
            priority=str(fields.get("priority", "medium")),
            note=str(fields.get("note", "")),
        )
        self.session.add(study)
        self.session.commit()
        self.session.refresh(study)
        return study

    def update_case_study(self, study_id: int, **fields: Any) -> CaseStudy | None:
        """ Обновление исследования клинического случая """
        study = self.session.query(CaseStudy).filter(CaseStudy.id == study_id).first()
        if study is None:
            return None
        for key, value in fields.items():
            if key in {"started_at", "completed_at"}:
                value = self._parse_dt(value)
            if key == "result_json" and value is not None:
                value = dict(value)
            if hasattr(study, key):
                setattr(study, key, value)
        study.updated_at = datetime.now(timezone.utc)
        self.session.commit()
        self.session.refresh(study)
        return study

    def delete_case_study(self, study_id: int) -> bool:
        """ Удаление исследования клинического случая """
        study = self.session.query(CaseStudy).filter(CaseStudy.id == study_id).first()
        if study is None:
            return False
        self.session.delete(study)
        self.session.commit()
        return True

    # ---------- Процедуры ----------
    def get_case_procedures(self, case_id: str) -> List[CaseProcedure]:
        """ Список процедур по клиническому случаю """
        return (
            self.session.query(CaseProcedure)
            .filter(CaseProcedure.case_id == case_id)
            .order_by(CaseProcedure.started_at.asc().nullslast(), CaseProcedure.id.asc())
            .all()
        )

    def add_case_procedure(self, case_id: str, **fields: Any) -> CaseProcedure:
        """ Добавление процедуры к клиническому случаю """
        procedure = CaseProcedure(
            case_id=case_id,
            code=str(fields.get("code", "")).strip(),
            name=str(fields.get("name", "")).strip(),
            status=str(fields.get("status", "ordered")),
            started_at=self._parse_dt(fields.get("started_at")),
            completed_at=self._parse_dt(fields.get("completed_at")),
            operator=str(fields.get("operator", "")),
            details_json=dict(fields.get("details_json", {}) or {}),
            priority=str(fields.get("priority", "medium")),
            note=str(fields.get("note", "")),
        )
        self.session.add(procedure)
        self.session.commit()
        self.session.refresh(procedure)
        return procedure

    def update_case_procedure(self, procedure_id: int, **fields: Any) -> CaseProcedure | None:
        """ Обновление процедуры клинического случая """
        procedure = self.session.query(CaseProcedure).filter(CaseProcedure.id == procedure_id).first()
        if procedure is None:
            return None
        for key, value in fields.items():
            if key in {"started_at", "completed_at"}:
                value = self._parse_dt(value)
            if key == "details_json" and value is not None:
                value = dict(value)
            if hasattr(procedure, key):
                setattr(procedure, key, value)
        procedure.updated_at = datetime.now(timezone.utc)
        self.session.commit()
        self.session.refresh(procedure)
        return procedure

    def delete_case_procedure(self, procedure_id: int) -> bool:
        """ Удаление процедуры клинического случая """
        procedure = self.session.query(CaseProcedure).filter(CaseProcedure.id == procedure_id).first()
        if procedure is None:
            return False
        self.session.delete(procedure)
        self.session.commit()
        return True

    # ---------- Препараты ----------
    def get_case_medications(self, case_id: str) -> List[CaseMedication]:
        """ Список препаратов по клиническому случаю """
        return (
            self.session.query(CaseMedication)
            .filter(CaseMedication.case_id == case_id)
            .order_by(CaseMedication.started_at.asc().nullslast(), CaseMedication.id.asc())
            .all()
        )

    def add_case_medication(self, case_id: str, **fields: Any) -> CaseMedication:
        """ Добавление препарата к клиническому случаю """
        med = CaseMedication(
            case_id=case_id,
            code=str(fields.get("code", "")).strip(),
            name=str(fields.get("name", "")).strip(),
            med_class=str(fields.get("med_class", "")).strip(),
            dose=str(fields.get("dose", "")),
            unit=str(fields.get("unit", "")),
            route=str(fields.get("route", "po")),
            frequency=str(fields.get("frequency", "")),
            started_at=self._parse_dt(fields.get("started_at")),
            stopped_at=self._parse_dt(fields.get("stopped_at")),
            status=str(fields.get("status", "active")),
            prescribed_by=str(fields.get("prescribed_by", "")),
            note=str(fields.get("note", "")),
        )
        self.session.add(med)
        self.session.commit()
        self.session.refresh(med)
        return med

    def update_case_medication(self, medication_id: int, **fields: Any) -> CaseMedication | None:
        """ Обновление препарата клинического случая """
        med = self.session.query(CaseMedication).filter(CaseMedication.id == medication_id).first()
        if med is None:
            return None
        for key, value in fields.items():
            if key in {"started_at", "stopped_at"}:
                value = self._parse_dt(value)
            if hasattr(med, key):
                setattr(med, key, value)
        med.updated_at = datetime.now(timezone.utc)
        self.session.commit()
        self.session.refresh(med)
        return med

    def delete_case_medication(self, medication_id: int) -> bool:
        """ Удаление препарата клинического случая """
        med = self.session.query(CaseMedication).filter(CaseMedication.id == medication_id).first()
        if med is None:
            return False
        self.session.delete(med)
        self.session.commit()
        return True

    # ---------- Диагнозы ----------
    def get_case_diagnoses(self, case_id: str) -> List[CaseDiagnosis]:
        """ Список диагнозов по клиническому случаю """
        return (
            self.session.query(CaseDiagnosis)
            .filter(CaseDiagnosis.case_id == case_id)
            .order_by(CaseDiagnosis.created_at.asc(), CaseDiagnosis.id.asc())
            .all()
        )

    def add_case_diagnosis(self, case_id: str, **fields: Any) -> CaseDiagnosis:
        """ Добавление диагноза к клиническому случаю """
        diagnosis = CaseDiagnosis(
            case_id=case_id,
            icd10=str(fields.get("icd10", "")).strip().upper(),
            name=str(fields.get("name", "")).strip(),
            diagnosis_type=str(fields.get("diagnosis_type", "primary")),
            established_at=self._parse_dt(fields.get("established_at")),
            note=str(fields.get("note", "")),
        )
        self.session.add(diagnosis)
        self.session.commit()
        self.session.refresh(diagnosis)
        return diagnosis

    def update_case_diagnosis(self, diagnosis_id: int, **fields: Any) -> CaseDiagnosis | None:
        """ Обновление диагноза клинического случая """
        diagnosis = self.session.query(CaseDiagnosis).filter(CaseDiagnosis.id == diagnosis_id).first()
        if diagnosis is None:
            return None
        for key, value in fields.items():
            if key == "established_at":
                value = self._parse_dt(value)
            if key == "icd10" and isinstance(value, str):
                value = value.strip().upper()
            if hasattr(diagnosis, key):
                setattr(diagnosis, key, value)
        self.session.commit()
        self.session.refresh(diagnosis)
        return diagnosis

    def delete_case_diagnosis(self, diagnosis_id: int) -> bool:
        """ Удаление диагноза клинического случая """
        diagnosis = self.session.query(CaseDiagnosis).filter(CaseDiagnosis.id == diagnosis_id).first()
        if diagnosis is None:
            return False
        self.session.delete(diagnosis)
        self.session.commit()
        return True

    def _parse_dt(self, value: Any) -> datetime | None:
        """ Преобразование значения в дату и время """
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value.strip():
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None


def _build_path_trace_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Сформировать JSON-объект трассы пациента по графу для персистентного хранения."""
    node_trace = result.get("node_trace") or []
    rule_fires = result.get("rule_fires") or []
    rule_reasons = result.get("rule_reasons") or []
    visited_nodes = [entry.get("node") for entry in node_trace if isinstance(entry, dict)]
    return {
        "visited_nodes": visited_nodes,
        "node_trace": node_trace,
        "rule_fires": rule_fires,
        "rule_reasons": rule_reasons,
        "next_step": result.get("next_step"),
        "triage_category": result.get("triage_category"),
        "risk_level": result.get("risk_level"),
        "current_node": visited_nodes[-1] if visited_nodes else None,
    }
