"""Миграция старых CaseTrackingItem в новые структурированные таблицы.

Запуск:
    python -m scripts.migrate_v2

Скрипт:
1. Создаёт все таблицы (включая новые case_studies / case_procedures / case_medications / case_diagnoses).
2. Переносит записи из CaseTrackingItem с item_type='procedure'|'medication'|'study' в
   соответствующие таблицы, пытаясь сопоставить с каталогами по коду/имени.
3. Оставляет исходные tracking items нетронутыми (на всякий случай; удалить их можно вручную).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

from src.infrastructure.db.database import Base, SessionLocal, engine
from src.infrastructure.db.models import (  # noqa: F401 - register models
    CaseAssessment,
    CaseDiagnosis,
    CaseMedication,
    CaseObservation,
    CaseProcedure,
    CaseStudy,
    CaseTrackingItem,
    ClinicalReport,
    Patient,
    TriageCase,
    Visit,
)
from src.medical.catalog import (
    MEDICATION_BY_CODE,
    PROCEDURE_BY_CODE,
    STUDY_BY_CODE,
    resolve_medication,
    resolve_procedure,
    resolve_study,
)


def _status_for_tracking(status: str) -> str:
    if not status:
        return "ordered"
    normalized = status.lower()
    if normalized in {"done", "completed"}:
        return "done"
    if normalized in {"in_progress", "running"}:
        return "in_progress"
    if normalized in {"cancelled", "canceled"}:
        return "cancelled"
    return "ordered"


def migrate() -> dict[str, int]:
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    moved = {"studies": 0, "procedures": 0, "medications": 0, "skipped": 0}
    try:
        items = session.query(CaseTrackingItem).all()
        for item in items:
            kind = (item.item_type or "").lower()
            name = (item.name or "").strip()
            if not name:
                moved["skipped"] += 1
                continue
            if kind == "study":
                code = resolve_study(name) or resolve_study(item.metadata_json.get("code", "")) or ""
                study_name = STUDY_BY_CODE[code].name_ru if code else name
                session.add(
                    CaseStudy(
                        case_id=item.case_id,
                        code=code,
                        name=study_name,
                        status=_status_for_tracking(item.status),
                        started_at=item.created_at,
                        completed_at=item.completed_at,
                        result_text=item.result_summary or "",
                        priority=item.priority or "medium",
                        note="",
                    )
                )
                moved["studies"] += 1
            elif kind == "procedure":
                code = resolve_procedure(name) or ""
                proc_name = PROCEDURE_BY_CODE[code].name_ru if code else name
                session.add(
                    CaseProcedure(
                        case_id=item.case_id,
                        code=code,
                        name=proc_name,
                        status=_status_for_tracking(item.status),
                        started_at=item.created_at,
                        completed_at=item.completed_at,
                        priority=item.priority or "medium",
                        note=item.result_summary or "",
                    )
                )
                moved["procedures"] += 1
            elif kind == "medication":
                code = resolve_medication(name) or ""
                definition = MEDICATION_BY_CODE.get(code)
                session.add(
                    CaseMedication(
                        case_id=item.case_id,
                        code=code,
                        name=definition.name_ru if definition else name,
                        med_class=definition.group if definition else "",
                        dose=definition.typical_dose if definition else "",
                        unit=definition.typical_unit if definition else "",
                        route=definition.default_route if definition else "po",
                        status="active" if item.status not in {"stopped", "completed"} else "stopped",
                        started_at=item.created_at,
                        stopped_at=item.completed_at,
                        note=item.result_summary or "",
                    )
                )
                moved["medications"] += 1
            else:
                moved["skipped"] += 1
        session.commit()
    finally:
        session.close()
    return moved


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    stats = migrate()
    print("Migration v2 complete:", stats)
