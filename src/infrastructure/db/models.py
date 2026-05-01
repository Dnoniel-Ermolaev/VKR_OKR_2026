# src/infrastucture/db/models.py
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.infrastructure.db.database import Base


RiskLevel = Literal["low", "medium", "high"] # Тип риска

CaseStatus = Literal["active", "awaiting_labs", "completed"] # Тип статуса кейса

# Модель данных пациента
class PatientData(BaseModel):
    name: str = Field(min_length=1)
    age: int | None = Field(default=None, ge=0, le=120)
    gender: Literal["male", "female", "unknown"] = "unknown"
    admission_time: str = ""
    pain_onset_time: str = ""
    pain_type: Literal["typical", "atypical", "none"]
    pain_description: str = ""
    ecg_changes: str = Field(min_length=1)
    troponin: float = Field(ge=0)
    hr: int = Field(gt=0)
    bp: str = Field(min_length=3, description="Format: systolic/diastolic")
    spo2: float | None = Field(default=None, ge=0, le=100)
    rr: int | None = Field(default=None, ge=0, le=80)
    glucose: float | None = Field(default=None, ge=0)
    creatinine: float | None = Field(default=None, ge=0)
    ast_alt_ckmb: Dict[str, float] = Field(default_factory=dict)
    lipid_profile: Dict[str, float] = Field(default_factory=dict)
    potassium_sodium_magnesium: Dict[str, float] = Field(default_factory=dict)
    echo_dkg_results: str = ""
    mri_results: str = ""
    ct_coronary: str = ""
    killip_class: str = ""
    interventions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    vital_signs: list[Dict[str, Any]] = Field(default_factory=list)
    symptoms_text: str = ""

    @field_validator("bp")
    @classmethod
    def validate_bp(cls, value: str) -> str:
        parts = value.split("/")
        if len(parts) != 2:
            raise ValueError("bp must be in format systolic/diastolic")
        try:
            int(parts[0])
            int(parts[1])
        except ValueError as exc:
            raise ValueError("bp values must be integers") from exc
        return value


class PatientRecord(BaseModel):
    id: str
    name: str
    symptoms_json: str
    ecg_desc: str
    troponin: float
    hr: int
    bp: str
    risk_level: RiskLevel
    explanation: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_assessment(
        cls,
        *,
        record_id: str,
        patient_data: Dict[str, object],
        risk_level: RiskLevel,
        explanation: str,
    ) -> "PatientRecord":
        return cls(
            id=record_id,
            name=str(patient_data.get("name", "unknown")),
            symptoms_json=str(patient_data),
            ecg_desc=str(patient_data.get("ecg_changes", "")),
            troponin=float(patient_data.get("troponin", 0.0)),
            hr=int(patient_data.get("hr", 0)),
            bp=str(patient_data.get("bp", "")),
            risk_level=risk_level,
            explanation=explanation,
        )

""" Классы для использовани ORM SQLAlchemy """
 # Наследуемся от Base (класс Base из SQLAlchemy)
class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    full_name: Mapped[str] = mapped_column(String(150))
    birth_date: Mapped[date] = mapped_column(Date)
    gender: Mapped[str] = mapped_column(String(10))

     # Связь "Один ко многим": пациент -> визиты
    visits = relationship("Visit", back_populates="patient", cascade="all, delete-orphan")

     # Связь "Один ко многим": пациент -> стационарные кейсы
    cases = relationship("TriageCase", back_populates="patient", cascade="all, delete-orphan")


class Visit(Base):
    __tablename__ = "visits"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    admission_time: Mapped[datetime] = mapped_column(DateTime)

     # Связь "Многие к одному" (визиты к пациентам)
    patient = relationship("Patient", back_populates="visits")


class TriageCase(Base):
    __tablename__ = "triage_cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[int | None] = mapped_column(ForeignKey("patients.id"), nullable=True, index=True)

    # Трекинг состояния кейса (активен, закрыт, на каком этапе)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    current_stage: Mapped[str] = mapped_column(String(64), default="start")

    title: Mapped[str] = mapped_column(String(200), default="ACS case")
    llm_model: Mapped[str] = mapped_column(String(120), default="")
    initial_payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    # Последние результаты оценки
    latest_payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    latest_risk: Mapped[float | None] = mapped_column(Float, nullable=True)
    latest_risk_level: Mapped[str] = mapped_column(String(16), default="low")
    latest_triage_category: Mapped[str] = mapped_column(String(64), default="")
    latest_next_step: Mapped[str] = mapped_column(String(64), default="")
    latest_explanation: Mapped[str] = mapped_column(Text, default="")
    latest_citations: Mapped[list[str]] = mapped_column(JSON, default=list)

    missing_fields_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    patient = relationship("Patient", back_populates="cases")
    observations = relationship("CaseObservation", back_populates="case", cascade="all, delete-orphan")
    assessments = relationship("CaseAssessment", back_populates="case", cascade="all, delete-orphan")
    reports = relationship("ClinicalReport", back_populates="case", cascade="all, delete-orphan")
    tracking_items = relationship("CaseTrackingItem", back_populates="case", cascade="all, delete-orphan")
    studies = relationship("CaseStudy", back_populates="case", cascade="all, delete-orphan")
    procedures = relationship("CaseProcedure", back_populates="case", cascade="all, delete-orphan")
    medications = relationship("CaseMedication", back_populates="case", cascade="all, delete-orphan")
    diagnoses = relationship("CaseDiagnosis", back_populates="case", cascade="all, delete-orphan")

# Витальные показатели и Анализы
class CaseObservation(Base):
    __tablename__ = "case_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("triage_cases.id"), index=True)
    category: Mapped[str] = mapped_column(String(24), default="vital")
    code: Mapped[str] = mapped_column(String(48), default="", index=True)
    name: Mapped[str] = mapped_column(String(120), index=True)
    value_num: Mapped[float | None] = mapped_column(Float, nullable=True)
    value_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    unit: Mapped[str] = mapped_column(String(32), default="")
    flag: Mapped[str] = mapped_column(String(16), default="unknown")
    source: Mapped[str] = mapped_column(String(64), default="manual")
    note: Mapped[str] = mapped_column(Text, default="")
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    case = relationship("TriageCase", back_populates="observations")

# История оценок риска
class CaseAssessment(Base):
    __tablename__ = "case_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("triage_cases.id"), index=True)
    run_kind: Mapped[str] = mapped_column(String(40), default="initial")
    payload_snapshot: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    risk: Mapped[float] = mapped_column(Float, default=0.0)
    risk_level: Mapped[str] = mapped_column(String(16), default="low")
    triage_category: Mapped[str] = mapped_column(String(64), default="")
    next_step: Mapped[str] = mapped_column(String(64), default="")
    route_reason: Mapped[str] = mapped_column(Text, default="")
    explanation: Mapped[str] = mapped_column(Text, default="")
    citations_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    missing_fields_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    llm_used: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    case = relationship("TriageCase", back_populates="assessments")

# Клинические отчеты от LLM
class ClinicalReport(Base):
    __tablename__ = "clinical_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("triage_cases.id"), index=True)
    report_type: Mapped[str] = mapped_column(String(40), default="epicrisis")
    content: Mapped[str] = mapped_column(Text, default="")
    citations_json: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    case = relationship("TriageCase", back_populates="reports")

# Элементы трекинга (задачи, наблюдения, исследования)
class CaseTrackingItem(Base):
    __tablename__ = "case_tracking_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("triage_cases.id"), index=True)
    item_type: Mapped[str] = mapped_column(String(40), index=True)
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    priority: Mapped[str] = mapped_column(String(16), default="medium")
    due_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    result_summary: Mapped[str] = mapped_column(Text, default="")
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    source_page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    case = relationship("TriageCase", back_populates="tracking_items")

# Исследования
class CaseStudy(Base):
    __tablename__ = "case_studies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("triage_cases.id"), index=True)
    code: Mapped[str] = mapped_column(String(64), default="", index=True)
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(24), default="ordered", index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    result_text: Mapped[str] = mapped_column(Text, default="")
    result_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    ordered_by: Mapped[str] = mapped_column(String(120), default="")
    priority: Mapped[str] = mapped_column(String(16), default="medium")
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    case = relationship("TriageCase", back_populates="studies")

# Процедуры
class CaseProcedure(Base):
    __tablename__ = "case_procedures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("triage_cases.id"), index=True)
    code: Mapped[str] = mapped_column(String(64), default="", index=True)
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(24), default="ordered", index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    operator: Mapped[str] = mapped_column(String(120), default="")
    details_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    priority: Mapped[str] = mapped_column(String(16), default="medium")
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    case = relationship("TriageCase", back_populates="procedures")

# Лекарства
class CaseMedication(Base):
    __tablename__ = "case_medications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("triage_cases.id"), index=True)
    code: Mapped[str] = mapped_column(String(64), default="", index=True)
    name: Mapped[str] = mapped_column(String(200))
    med_class: Mapped[str] = mapped_column(String(32), default="", index=True)
    dose: Mapped[str] = mapped_column(String(64), default="")
    unit: Mapped[str] = mapped_column(String(32), default="")
    route: Mapped[str] = mapped_column(String(24), default="po")
    frequency: Mapped[str] = mapped_column(String(64), default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="active", index=True)
    prescribed_by: Mapped[str] = mapped_column(String(120), default="")
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    case = relationship("TriageCase", back_populates="medications")

# Диагнозы
class CaseDiagnosis(Base):
    __tablename__ = "case_diagnoses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("triage_cases.id"), index=True)
    icd10: Mapped[str] = mapped_column(String(16), default="", index=True)
    name: Mapped[str] = mapped_column(String(250))
    diagnosis_type: Mapped[str] = mapped_column(String(24), default="primary", index=True)
    established_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    case = relationship("TriageCase", back_populates="diagnoses")