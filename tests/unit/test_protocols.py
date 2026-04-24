"""Тесты протоколов ОКС и расчёта прогресса контроля."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.core.patient_control import (
    build_alerts,
    build_case_control,
    build_control_summary,
    derive_tracking,
    select_case_protocol,
)
from src.medical.protocols import (
    GENERIC_ACS_PROTOCOL,
    NSTEMI_PROTOCOL,
    STEMI_PROTOCOL,
    UA_PROTOCOL,
    select_protocol,
)


def _obs(code: str, value: float, hours_ago: float = 0.0, category: str = "lab"):
    recorded_at = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return SimpleNamespace(
        code=code,
        name=code,
        category=category,
        value_num=value,
        value_text=str(value),
        unit="",
        recorded_at=recorded_at,
        flag="norm",
        note="",
    )


def _study(code: str, status: str = "done", hours_ago: float = 0.0):
    dt = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return SimpleNamespace(code=code, name=code, status=status, started_at=dt, completed_at=dt)


def _med(code: str, med_class: str = "", status: str = "active"):
    return SimpleNamespace(code=code, name=code, med_class=med_class, status=status)


def _diag(icd10: str):
    return SimpleNamespace(icd10=icd10, name=icd10, diagnosis_type="primary")


def _procedure(code: str, status: str = "done"):
    dt = datetime.now(timezone.utc)
    return SimpleNamespace(code=code, name=code, status=status, started_at=dt, completed_at=dt)


# -------------------- select_protocol --------------------


def test_select_protocol_stemi_by_icd():
    assert select_protocol(["I21.0"]).code == STEMI_PROTOCOL.code


def test_select_protocol_nstemi_by_icd():
    assert select_protocol(["I21.4"]).code == NSTEMI_PROTOCOL.code


def test_select_protocol_ua_by_icd():
    assert select_protocol(["I20.0"]).code == UA_PROTOCOL.code


def test_select_protocol_fallback_generic():
    assert select_protocol([]).code == GENERIC_ACS_PROTOCOL.code


def test_select_protocol_by_ecg_description():
    assert select_protocol([], ecg_description="выраженная элевация ST").code == STEMI_PROTOCOL.code
    assert select_protocol([], ecg_description="st-depression в V4-V6").code == NSTEMI_PROTOCOL.code


def test_select_case_protocol_accepts_objects():
    protocol = select_case_protocol([_diag("I21.0")])
    assert protocol.code == STEMI_PROTOCOL.code


# -------------------- derive_tracking --------------------


def test_derive_tracking_flags_missing_troponin_and_ecg():
    protocol = STEMI_PROTOCOL
    started = datetime.now(timezone.utc) - timedelta(hours=5)
    items = derive_tracking(
        protocol=protocol,
        observations=[],
        studies=[],
        procedures=[],
        medications=[],
        case_started_at=started,
    )
    trop = next(i for i in items if i["code"] == "troponin_i")
    ecg = next(i for i in items if i["code"] == "ecg_12")
    assert trop["status"] == "pending"
    assert ecg["status"] == "pending"
    assert ecg["needed_count"] == 3


def test_derive_tracking_marks_done_when_enough_series():
    protocol = STEMI_PROTOCOL
    started = datetime.now(timezone.utc) - timedelta(hours=8)
    observations = [
        _obs("troponin_i", 0.02, hours_ago=8),
        _obs("troponin_i", 0.03, hours_ago=5),
        _obs("troponin_i", 0.06, hours_ago=2),
    ]
    items = derive_tracking(
        protocol=protocol,
        observations=observations,
        studies=[],
        procedures=[],
        medications=[],
        case_started_at=started,
    )
    trop = next(i for i in items if i["code"] == "troponin_i")
    assert trop["status"] == "done"
    assert trop["done_count"] == 3


def test_derive_tracking_overdue_when_window_elapsed():
    protocol = STEMI_PROTOCOL
    started = datetime.now(timezone.utc) - timedelta(hours=10)
    items = derive_tracking(
        protocol=protocol,
        observations=[],
        studies=[],
        procedures=[],
        medications=[],
        case_started_at=started,
    )
    angio = next(i for i in items if i["code"] == "coronary_angiography")
    # window_hours=2 истекло через 10 часов
    assert angio["overdue"] is True


def test_derive_tracking_medication_class_done():
    protocol = STEMI_PROTOCOL
    items = derive_tracking(
        protocol=protocol,
        observations=[],
        studies=[],
        procedures=[],
        medications=[
            _med("asa", med_class="antiplatelet"),
            _med("clopidogrel", med_class="antiplatelet"),
            _med("heparin_uf", med_class="anticoag"),
        ],
        case_started_at=datetime.now(timezone.utc),
    )
    antiplt = next(i for i in items if i["code"] == "antiplatelet")
    anticoag = next(i for i in items if i["code"] == "anticoag")
    assert antiplt["status"] == "done"
    assert anticoag["status"] == "done"


# -------------------- alerts --------------------


def test_build_alerts_requires_antiplatelet_for_stemi():
    items = derive_tracking(
        protocol=STEMI_PROTOCOL,
        observations=[],
        studies=[],
        procedures=[],
        medications=[],
        case_started_at=datetime.now(timezone.utc),
    )
    alerts = build_alerts(
        protocol=STEMI_PROTOCOL,
        tracking_items=items,
        observations=[],
        medications=[],
        procedures=[],
    )
    assert any("антиагрегант" in a.lower() for a in alerts)
    assert any("антикоагулянт" in a.lower() for a in alerts)


def test_build_alerts_spo2_without_oxygen_therapy():
    observations = [_obs("spo2", 85, category="vital")]
    items = derive_tracking(
        protocol=STEMI_PROTOCOL,
        observations=observations,
        studies=[],
        procedures=[],
        medications=[],
        case_started_at=datetime.now(timezone.utc),
    )
    alerts = build_alerts(
        protocol=STEMI_PROTOCOL,
        tracking_items=items,
        observations=observations,
        medications=[],
        procedures=[],
    )
    assert any("spo2" in a.lower() and "оксиген" in a.lower() for a in alerts)


def test_build_alerts_no_spo2_warning_when_oxygen_given():
    observations = [_obs("spo2", 85, category="vital")]
    procedures = [_procedure("oxygen_therapy", status="in_progress")]
    items = derive_tracking(
        protocol=STEMI_PROTOCOL,
        observations=observations,
        studies=[],
        procedures=procedures,
        medications=[],
        case_started_at=datetime.now(timezone.utc),
    )
    alerts = build_alerts(
        protocol=STEMI_PROTOCOL,
        tracking_items=items,
        observations=observations,
        medications=[],
        procedures=procedures,
    )
    assert not any("рассмотреть оксигенотерапию" in a.lower() for a in alerts)


def test_build_alerts_critical_lab_deviation():
    observations = [_obs("k_blood", 2.5)]
    items = derive_tracking(
        protocol=STEMI_PROTOCOL,
        observations=observations,
        studies=[],
        procedures=[],
        medications=[],
        case_started_at=datetime.now(timezone.utc),
    )
    alerts = build_alerts(
        protocol=STEMI_PROTOCOL,
        tracking_items=items,
        observations=observations,
        medications=[],
        procedures=[],
    )
    assert any("калий" in a.lower() for a in alerts)


# -------------------- summary / orchestration --------------------


def test_build_control_summary_percentages():
    items = [
        {"kind": "lab", "code": "a", "title": "a", "priority": "critical", "status": "done", "overdue": False},
        {"kind": "lab", "code": "b", "title": "b", "priority": "high", "status": "pending", "overdue": True},
        {"kind": "lab", "code": "c", "title": "c", "priority": "medium", "status": "in_progress", "overdue": False},
    ]
    summary = build_control_summary(items, alerts=["x"])
    assert summary["total_items"] == 3
    assert summary["done_items"] == 1
    assert summary["pending_items"] == 1
    assert summary["in_progress_items"] == 1
    assert summary["overdue_items"] == 1
    assert summary["completion_percent"] == pytest.approx(33.3, rel=0.01)
    assert summary["alerts"] == ["x"]


def test_build_case_control_end_to_end_stemi():
    diagnoses = [_diag("I21.0")]
    observations = [_obs("troponin_i", 0.08, hours_ago=1)]
    studies = [_study("ecg_12", status="done", hours_ago=1)]
    procedures: list = []
    medications = [_med("asa", med_class="antiplatelet")]
    protocol, tracking, summary = build_case_control(
        observations=observations,
        studies=studies,
        procedures=procedures,
        medications=medications,
        diagnoses=diagnoses,
        case_started_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    assert protocol.code == STEMI_PROTOCOL.code
    assert summary["total_items"] == len(STEMI_PROTOCOL.requirements)
    assert isinstance(summary["alerts"], list)
    # отсутствует антикоагулянт -> в алертах
    assert any("антикоагулянт" in a.lower() for a in summary["alerts"])
