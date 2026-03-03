from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict

from pydantic import ValidationError

from src.core.state import AgentState
from src.core.tools import LlmClient, build_repository, build_retriever
from src.infrastructure.db.models import PatientData, PatientRecord
from src.medical.rules import evaluate_hard_rules


BASE_DIR = Path(__file__).resolve().parents[2]
REPOSITORY = build_repository(BASE_DIR)
RETRIEVER = build_retriever(BASE_DIR)
LLM_CLIENT = LlmClient()


def llm_parse_history(state: AgentState) -> AgentState:
    free_text = str(state.get("free_text", "")).strip()
    if not free_text:
        return {"missing_fields": ["free_text"], "parse_confidence": 0.0, "llm_used": False}
    patient_data, missing_fields, confidence, llm_used = LLM_CLIENT.parse_history(
        free_text=free_text,
        require_llm=bool(state.get("require_llm", False)),
        model_name=state.get("llm_model"),
    )
    return {
        "patient_data": patient_data,
        "missing_fields": missing_fields,
        "parse_confidence": confidence,
        "llm_used": llm_used,
    }


def _safe_patient_payload(payload: Dict[str, object]) -> Dict[str, object]:
    safe = dict(payload)
    safe.setdefault("name", "Unknown")
    safe.setdefault("pain_type", "none")
    safe.setdefault("ecg_changes", "normal")
    safe.setdefault("troponin", 0.0)
    safe.setdefault("hr", 70)
    safe.setdefault("bp", "120/80")
    safe.setdefault("symptoms_text", str(payload.get("symptoms_text", "")))
    safe.setdefault("age", None)
    safe.setdefault("gender", "unknown")
    safe.setdefault("admission_time", "")
    safe.setdefault("pain_onset_time", "")
    safe.setdefault("pain_description", str(payload.get("symptoms_text", "")))
    safe.setdefault("spo2", None)
    safe.setdefault("rr", None)
    safe.setdefault("glucose", None)
    safe.setdefault("creatinine", None)
    safe.setdefault("ast_alt_ckmb", {})
    safe.setdefault("lipid_profile", {})
    safe.setdefault("potassium_sodium_magnesium", {})
    safe.setdefault("echo_dkg_results", "")
    safe.setdefault("mri_results", "")
    safe.setdefault("ct_coronary", "")
    safe.setdefault("killip_class", "")
    safe.setdefault("interventions", [])
    safe.setdefault("medications", [])
    safe.setdefault("vital_signs", [])
    return safe


def parse_input(state: AgentState) -> AgentState:
    payload = state.get("patient_data", {})
    parse_confidence = float(state.get("parse_confidence", 1.0))
    missing_fields = list(state.get("missing_fields", []))
    parsed_ok = True
    try:
        validated = PatientData(**payload)
        normalized = validated.model_dump()
    except ValidationError as exc:
        parsed_ok = False
        normalized = _safe_patient_payload(payload)
        for err in exc.errors():
            loc = err.get("loc", [])
            if loc:
                missing_fields.append(str(loc[0]))
        parse_confidence = min(parse_confidence, 0.35)
    missing_fields = sorted(set(missing_fields))
    return {
        "patient_data": normalized,
        "parsed_ok": parsed_ok,
        "free_text": str(state.get("free_text", "")),
        "missing_fields": missing_fields,
        "parse_confidence": parse_confidence,
        "clarification_attempts": int(state.get("clarification_attempts", 0)),
        "max_clarification_attempts": int(state.get("max_clarification_attempts", 2)),
        "require_llm": bool(state.get("require_llm", False)),
        "force_llm": bool(state.get("force_llm", False)),
        "llm_model": str(state.get("llm_model", "")).strip(),
    }


def router_pretriage(state: AgentState) -> AgentState:
    missing_fields = list(state.get("missing_fields", []))
    next_step, confidence, reason, llm_used = LLM_CLIENT.route_pretriage(
        patient_data=state["patient_data"],
        parse_confidence=float(state.get("parse_confidence", 1.0)),
        missing_fields=missing_fields,
        parsed_ok=bool(state.get("parsed_ok", False)),
        require_llm=bool(state.get("require_llm", False)),
        model_name=state.get("llm_model"),
    )
    # Confidence gates.
    if confidence < 0.55:
        critical = {"troponin", "ecg_changes", "hr", "bp"}
        if critical.intersection(set(missing_fields)) or not bool(state.get("parsed_ok", False)):
            next_step = "needs_more_data"
            reason = "Low confidence pretriage + missing critical fields."
        else:
            next_step = "proceed"
            reason = "Low confidence pretriage, but minimum data available."
    return {
        "next_step": next_step,
        "route_confidence": confidence,
        "route_reason": reason,
        "llm_used": llm_used,
    }


def clarify_data(state: AgentState) -> AgentState:
    attempts = int(state.get("clarification_attempts", 0)) + 1
    max_attempts = int(state.get("max_clarification_attempts", 2))
    missing = state.get("missing_fields", [])
    if attempts <= max_attempts and str(state.get("free_text", "")).strip():
        return {
            "clarification_attempts": attempts,
            "next_step": "retry_parse",
            "route_reason": f"Уточнение попытка {attempts}/{max_attempts}; missing={', '.join(missing)}",
        }
    explanation = (
        "Качество данных недостаточно для безопасного решения. "
        f"Не хватает ключевых полей: {', '.join(missing) if missing else 'неизвестно'}. "
        "Требуется ручной ввод/уточнение."
    )
    return {
        "clarification_attempts": attempts,
        "next_step": "data_quality_issue",
        "triage_category": "data_quality_issue",
        "risk": float(state.get("risk", 0.2)),
        "risk_level": str(state.get("risk_level", "low")),
        "explanation": explanation,
    }


def rule_check(state: AgentState) -> AgentState:
    risk, risk_level, reasons, route_to_llm = evaluate_hard_rules(state["patient_data"])
    if bool(state.get("force_llm")):
        route_to_llm = True
    return {
        "risk": risk,
        "risk_level": risk_level,
        "rule_reasons": reasons,
        "route_to_llm": route_to_llm,
    }


def router_diagnostic(state: AgentState) -> AgentState:
    next_step, confidence, reason, llm_used = LLM_CLIENT.route_diagnostic(
        patient_data=state["patient_data"],
        rule_reasons=state.get("rule_reasons", []),
        risk=float(state.get("risk", 0.0)),
        risk_level=str(state.get("risk_level", "low")),
        require_llm=bool(state.get("require_llm", False)),
        model_name=state.get("llm_model"),
    )
    if confidence < 0.6:
        risk = float(state.get("risk", 0.0))
        if risk >= 0.9:
            next_step = "urgent"
            reason = "Low confidence diagnostic router; escalate by risk."
        elif risk >= 0.45:
            next_step = "rag_llm"
            reason = "Low confidence diagnostic router; use uncertain path."
        else:
            next_step = "rule_only"
            reason = "Low confidence diagnostic router; use observation path."
    if bool(state.get("force_llm")) and next_step == "rule_only":
        next_step = "rag_llm"
        reason = "force_llm активен, переход в diagnostic_uncertain."
    return {"next_step": next_step, "route_confidence": confidence, "route_reason": reason, "llm_used": llm_used}


def high_risk_fast_track(state: AgentState) -> AgentState:
    explanation = (
        "Маршрут high_risk_fast_track: высокий риск по клиническим правилам, "
        "требуется срочная кардиологическая тактика."
    )
    return {"triage_category": "high_risk_fast_track", "explanation": explanation}


def diagnostic_uncertain(state: AgentState) -> AgentState:
    return {"triage_category": "diagnostic_uncertain"}


def low_risk_observation(state: AgentState) -> AgentState:
    explanation = state.get("explanation") or "Маршрут low_risk_observation: динамическое наблюдение."
    return {"triage_category": "low_risk_observation", "explanation": explanation}


def data_quality_issue(state: AgentState) -> AgentState:
    return {
        "triage_category": "data_quality_issue",
        "risk": float(state.get("risk", 0.2)),
        "risk_level": str(state.get("risk_level", "low")),
        "explanation": state.get("explanation") or "Недостаточно данных для диагностики.",
    }


def rag_retrieval(state: AgentState) -> AgentState:
    patient = state["patient_data"]
    query = f"{patient.get('pain_type', '')} {patient.get('ecg_changes', '')} troponin {patient.get('troponin', '')}"
    snippets = RETRIEVER.retrieve(query=query, top_k=3)
    return {"rag_context": "\n---\n".join(snippets)}


def llm_assess(state: AgentState) -> AgentState:
    adjustment, explanation, llm_used = LLM_CLIENT.assess(
        patient_data=state["patient_data"],
        rule_reasons=state.get("rule_reasons", []),
        rag_context=state.get("rag_context", ""),
        require_llm=bool(state.get("require_llm", False)),
        model_name=state.get("llm_model"),
    )
    new_risk = max(0.0, min(1.0, float(state.get("risk", 0.0)) + adjustment))
    if new_risk >= 0.75:
        level = "high"
    elif new_risk >= 0.45:
        level = "medium"
    else:
        level = "low"
    return {
        "risk": new_risk,
        "risk_level": level,
        "explanation": explanation,
        "llm_used": llm_used,
        "triage_category": "diagnostic_uncertain",
    }


def router_management(state: AgentState) -> AgentState:
    next_step, confidence, reason, llm_used = LLM_CLIENT.route_management(
        patient_data=state["patient_data"],
        risk=float(state.get("risk", 0.0)),
        risk_level=str(state.get("risk_level", "low")),
        explanation=str(state.get("explanation", "")),
        require_llm=bool(state.get("require_llm", False)),
        model_name=state.get("llm_model"),
    )
    if confidence < 0.55:
        if str(state.get("risk_level", "low")) == "high":
            next_step = "recommend_treatment"
            reason = "Low confidence management router; prefer treatment hints for high risk."
        else:
            next_step = "monitor"
            reason = "Low confidence management router; prefer monitoring."
    return {"next_step": next_step, "route_confidence": confidence, "route_reason": reason, "llm_used": llm_used}


def monitor_plan(state: AgentState) -> AgentState:
    append_text = "Рекомендован мониторинг: ЭКГ/АД/ЧСС/SpO2 в динамике и контроль тропонина."
    explanation = f"{state.get('explanation', '')} {append_text}".strip()
    return {"explanation": explanation}


def recommend_treatment(state: AgentState) -> AgentState:
    risk_level = str(state.get("risk_level", "low"))
    if risk_level == "high":
        plan = (
            "Предварительный план: рассмотреть urgent кардиологическую тактику, "
            "антитромботическую стратегию по локальному протоколу и контроль гемодинамики."
        )
    else:
        plan = "Предварительный план: наблюдение, повторные маркеры и контроль симптомов."
    explanation = f"{state.get('explanation', '')} {plan}".strip()
    return {"explanation": explanation}


def output_save(state: AgentState) -> AgentState:
    explanation = state.get("explanation") or "Оценка выполнена по rule-based логике."
    record_id = str(uuid.uuid4())
    record = PatientRecord.from_assessment(
        record_id=record_id,
        patient_data=state["patient_data"],
        risk_level=state["risk_level"],
        explanation=explanation,
    )
    REPOSITORY.save_patient(record)
    return {"save_id": record_id, "explanation": explanation}


def route_from_start(state: AgentState) -> str:
    return "llm_parse_history" if bool(str(state.get("free_text", "")).strip()) else "parse_input"


def route_after_pretriage(state: AgentState) -> str:
    step = str(state.get("next_step", "proceed"))
    return "clarify_data" if step == "needs_more_data" else "rule_check"


def route_after_clarify(state: AgentState) -> str:
    step = str(state.get("next_step", "data_quality_issue"))
    if step == "retry_parse":
        return "llm_parse_history"
    return "data_quality_issue"


def route_after_diagnostic(state: AgentState) -> str:
    step = str(state.get("next_step", "rag_llm"))
    if step == "urgent":
        return "high_risk_fast_track"
    if step == "rule_only":
        return "low_risk_observation"
    return "diagnostic_uncertain"


def route_after_management(state: AgentState) -> str:
    step = str(state.get("next_step", "finalize"))
    if step == "monitor":
        return "monitor_plan"
    if step == "recommend_treatment":
        return "recommend_treatment"
    return "output_save"
