"""Контроль пациента: расчёт прогресса по клиническому протоколу.

Заменяет прежний OCR-driven tracker. Работает на структурированных
сущностях из БД (CaseObservation / CaseStudy / CaseProcedure / CaseMedication /
CaseDiagnosis) и сопоставляет их с требованиями выбранного протокола.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.medical.catalog import (
    LAB_BY_CODE,
    MEDICATION_BY_CODE,
    PROCEDURE_BY_CODE,
    STUDY_BY_CODE,
    VITAL_BY_CODE,
    flag_for_lab,
    flag_for_vital,
)
from src.medical.protocols import Protocol, ProtocolRequirement, select_protocol


def _field(item: Any, name: str, default: Any = None) -> Any:
    if item is None:
        return default
    if hasattr(item, name):
        return getattr(item, name)
    if isinstance(item, dict):
        return item.get(name, default)
    return default


def _ensure_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """PostgreSQL TIMESTAMP WITHOUT TIME ZONE даёт naive datetime — приводим к UTC для сравнений."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Матрица по протоколу
# ---------------------------------------------------------------------------


def select_case_protocol(
    diagnoses: Iterable[Any] | None,
    ecg_description: str = "",
    triage_category: str = "",
) -> Protocol:
    icd_codes = [str(_field(d, "icd10", "")).upper() for d in (diagnoses or [])]
    return select_protocol(icd_codes, ecg_description=ecg_description, triage_category=triage_category)


def _requirement_kind_title(kind: str) -> str:
    return {
        "lab": "Анализ",
        "study": "Исследование",
        "procedure": "Процедура",
        "medication_class": "Назначение (класс)",
        "vital": "Наблюдение",
    }.get(kind, kind)


def _catalog_name(kind: str, code: str) -> str:
    if kind == "lab":
        entry = LAB_BY_CODE.get(code)
        if entry:
            return entry.name_ru
    if kind == "study":
        entry = STUDY_BY_CODE.get(code)
        if entry:
            return entry.name_ru
    if kind == "procedure":
        entry = PROCEDURE_BY_CODE.get(code)
        if entry:
            return entry.name_ru
    if kind == "vital":
        entry = VITAL_BY_CODE.get(code)
        if entry:
            return entry.name_ru
    return code


def _count_satisfying_observations(
    requirement: ProtocolRequirement,
    observations: List[Any],
) -> List[Any]:
    matching = [
        obs for obs in observations
        if str(_field(obs, "code", "")).lower() == requirement.code
        and _field(obs, "value_num") is not None
    ]
    matching.sort(key=lambda o: _ensure_dt(_field(o, "recorded_at")) or datetime.min.replace(tzinfo=timezone.utc))
    return matching


def _count_satisfying_studies(
    requirement: ProtocolRequirement,
    studies: List[Any],
) -> List[Any]:
    matching = [
        s for s in studies
        if str(_field(s, "code", "")).lower() == requirement.code
        and str(_field(s, "status", "")).lower() in {"done", "completed"}
    ]
    matching.sort(key=lambda s: _ensure_dt(_field(s, "completed_at") or _field(s, "started_at")) or datetime.min.replace(tzinfo=timezone.utc))
    return matching


def _count_satisfying_procedures(
    requirement: ProtocolRequirement,
    procedures: List[Any],
) -> List[Any]:
    matching = [
        p for p in procedures
        if str(_field(p, "code", "")).lower() == requirement.code
        and str(_field(p, "status", "")).lower() in {"done", "completed"}
    ]
    matching.sort(key=lambda p: _ensure_dt(_field(p, "completed_at") or _field(p, "started_at")) or datetime.min.replace(tzinfo=timezone.utc))
    return matching


def _count_satisfying_medications(
    requirement: ProtocolRequirement,
    medications: List[Any],
) -> List[Any]:
    target_class = (requirement.med_class or requirement.code).lower()
    matching: List[Any] = []
    for med in medications:
        status = str(_field(med, "status", "")).lower()
        if status and status not in {"active", "paused", "completed", "stopped"}:
            continue
        if status == "stopped":
            # Stopped still counts as previously administered but not active.
            pass
        code = str(_field(med, "code", "")).lower()
        med_class = str(_field(med, "med_class", "")).lower()
        if not med_class and code and code in MEDICATION_BY_CODE:
            med_class = MEDICATION_BY_CODE[code].group
        if med_class == target_class:
            matching.append(med)
    return matching


def derive_tracking(
    *,
    protocol: Protocol,
    observations: List[Any],
    studies: List[Any],
    procedures: List[Any],
    medications: List[Any],
    case_started_at: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Пройти по требованиям протокола и сформировать список пунктов контроля."""
    now = _as_utc(now) if now is not None else datetime.now(timezone.utc)
    started = _as_utc(case_started_at) if case_started_at is not None else now
    items: List[Dict[str, Any]] = []

    for req in protocol.requirements:
        if req.kind == "lab":
            satisfying = _count_satisfying_observations(req, observations)
        elif req.kind == "study":
            satisfying = _count_satisfying_studies(req, studies)
        elif req.kind == "procedure":
            satisfying = _count_satisfying_procedures(req, procedures)
        elif req.kind == "medication_class":
            satisfying = _count_satisfying_medications(req, medications)
        elif req.kind == "vital":
            satisfying = [
                obs for obs in observations
                if str(_field(obs, "code", "")).lower() == req.code
            ]
        else:
            satisfying = []

        done = len(satisfying)
        needed = max(1, req.min_occurrences)

        status: str
        overdue = False
        if req.kind == "medication_class" and done >= needed:
            status = "done"
        elif done >= needed:
            status = "done"
        elif done > 0:
            status = "in_progress"
        else:
            status = "pending"

        # Overdue logic — по окну от поступления.
        if status != "done" and req.window_hours is not None and started is not None:
            deadline = started + _hours(req.window_hours)
            if now > deadline:
                overdue = True

        last_at = None
        if satisfying:
            last_at = (
                _ensure_dt(_field(satisfying[-1], "recorded_at"))
                or _ensure_dt(_field(satisfying[-1], "completed_at"))
                or _ensure_dt(_field(satisfying[-1], "started_at"))
            )

        items.append(
            {
                "kind": req.kind,
                "code": req.code,
                "title": req.title or _catalog_name(req.kind, req.code),
                "priority": req.priority,
                "status": status,
                "overdue": overdue,
                "done_count": done,
                "needed_count": needed,
                "window_hours": req.window_hours,
                "last_recorded_at": last_at.isoformat() if last_at else None,
                "note": req.note,
            }
        )
    return items


def build_alerts(
    *,
    protocol: Protocol,
    tracking_items: List[Dict[str, Any]],
    observations: List[Any],
    medications: List[Any],
    procedures: List[Any] | None = None,
) -> List[str]:
    alerts: List[str] = []

    # Просроченные критические пункты.
    for item in tracking_items:
        if item["priority"] in {"critical", "high"} and item["status"] != "done":
            if item["overdue"]:
                alerts.append(
                    f"Просрочено: {item['title']} (нужно не менее {item['needed_count']}, выполнено {item['done_count']})."
                )
            elif item["priority"] == "critical" and item["status"] == "pending":
                alerts.append(f"Критический шаг не начат: {item['title']}.")

    # Отсутствие обязательной терапии при ОКС.
    active_classes = set()
    for med in medications:
        if str(_field(med, "status", "")).lower() in {"active", "paused"}:
            med_class = str(_field(med, "med_class", "")).lower()
            code = str(_field(med, "code", "")).lower()
            if not med_class and code in MEDICATION_BY_CODE:
                med_class = MEDICATION_BY_CODE[code].group
            if med_class:
                active_classes.add(med_class)
    if protocol.code in {"acs_stemi", "acs_nstemi"} and "antiplatelet" not in active_classes:
        alerts.append("Не назначен антиагрегант при ОКС — угроза тромботических осложнений.")
    if protocol.code == "acs_stemi" and "anticoag" not in active_classes:
        alerts.append("Не назначен антикоагулянт при STEMI.")

    # Критические лабораторные отклонения.
    latest_by_code: Dict[str, Any] = {}
    for obs in observations:
        code = str(_field(obs, "code", "")).lower()
        if not code:
            continue
        prev = latest_by_code.get(code)
        prev_t = _ensure_dt(_field(prev, "recorded_at")) if prev else None
        cur_t = _ensure_dt(_field(obs, "recorded_at"))
        if prev is None or (cur_t and prev_t and cur_t > prev_t):
            latest_by_code[code] = obs
    for code, obs in latest_by_code.items():
        value = _field(obs, "value_num")
        if value is None:
            continue
        if code in LAB_BY_CODE:
            flag = flag_for_lab(code, float(value))
            if flag in {"critical_low", "critical_high"}:
                alerts.append(
                    f"Критическое отклонение: {LAB_BY_CODE[code].name_ru} = {value}{LAB_BY_CODE[code].unit} ({flag})."
                )
        elif code in VITAL_BY_CODE:
            flag = flag_for_vital(code, float(value))
            if flag in {"critical_low", "critical_high"}:
                alerts.append(
                    f"Витальный показатель вне безопасного диапазона: {VITAL_BY_CODE[code].name_ru} = {value}{VITAL_BY_CODE[code].unit}."
                )

    # SpO2 < 90 без O2-терапии.
    spo2 = latest_by_code.get("spo2")
    if spo2 is not None:
        try:
            spo2_val = float(_field(spo2, "value_num"))
            oxygen_active = any(
                str(_field(p, "code", "")).lower() == "oxygen_therapy"
                and str(_field(p, "status", "")).lower() in {"ordered", "in_progress", "done", "completed"}
                for p in (procedures or [])
            )
            if spo2_val < 90 and not oxygen_active:
                alerts.append(f"SpO2 {spo2_val}% < 90% — рассмотреть оксигенотерапию.")
        except (TypeError, ValueError):
            pass

    return _dedupe(alerts)


def build_control_summary(
    tracking_items: List[Dict[str, Any]],
    alerts: List[str],
) -> Dict[str, Any]:
    total = len(tracking_items)
    done = sum(1 for item in tracking_items if item["status"] == "done")
    in_progress = sum(1 for item in tracking_items if item["status"] == "in_progress")
    pending = sum(1 for item in tracking_items if item["status"] == "pending")
    overdue = sum(1 for item in tracking_items if item["overdue"])
    critical_pending = [
        item["title"] for item in tracking_items
        if item["priority"] in {"critical", "high"} and item["status"] != "done"
    ]
    completion = round((done / total) * 100.0, 1) if total else 0.0
    return {
        "total_items": total,
        "done_items": done,
        "in_progress_items": in_progress,
        "pending_items": pending,
        "overdue_items": overdue,
        "completion_percent": completion,
        "critical_pending": critical_pending,
        "alerts": alerts,
    }


def has_pending_critical(tracking_items: List[Dict[str, Any]]) -> bool:
    return any(
        item["priority"] in {"critical", "high"} and item["status"] != "done"
        for item in tracking_items
    )


def build_case_control(
    *,
    observations: List[Any],
    studies: List[Any],
    procedures: List[Any],
    medications: List[Any],
    diagnoses: List[Any],
    case_payload: Dict[str, Any] | None = None,
    latest_result: Dict[str, Any] | None = None,
    case_started_at: Optional[datetime] = None,
) -> Tuple[Protocol, List[Dict[str, Any]], Dict[str, Any]]:
    ecg_text = ""
    triage = ""
    if case_payload:
        ecg_text = str(case_payload.get("ecg_changes", ""))
    if latest_result:
        triage = str(latest_result.get("triage_category", ""))
    protocol = select_case_protocol(diagnoses, ecg_description=ecg_text, triage_category=triage)
    tracking = derive_tracking(
        protocol=protocol,
        observations=observations,
        studies=studies,
        procedures=procedures,
        medications=medications,
        case_started_at=case_started_at,
    )
    alerts = build_alerts(
        protocol=protocol,
        tracking_items=tracking,
        observations=observations,
        medications=medications,
        procedures=procedures,
    )
    summary = build_control_summary(tracking, alerts)
    return protocol, tracking, summary


def _hours(hours: float):
    from datetime import timedelta
    return timedelta(hours=float(hours))


def _dedupe(items: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


__all__ = [
    "select_case_protocol",
    "derive_tracking",
    "build_alerts",
    "build_control_summary",
    "has_pending_critical",
    "build_case_control",
]
