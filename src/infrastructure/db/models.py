from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Literal

from pydantic import BaseModel, Field, field_validator


RiskLevel = Literal["low", "medium", "high"]


class PatientData(BaseModel):
    name: str = Field(min_length=1)
    pain_type: Literal["typical", "atypical", "none"]
    ecg_changes: str = Field(min_length=1)
    troponin: float = Field(ge=0)
    hr: int = Field(gt=0)
    bp: str = Field(min_length=3, description="Format: systolic/diastolic")
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
