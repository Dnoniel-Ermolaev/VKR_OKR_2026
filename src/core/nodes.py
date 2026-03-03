from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict

from src.core.state import AgentState
from src.core.tools import LlmClient, build_repository, build_retriever
from src.infrastructure.db.models import PatientData, PatientRecord
from src.medical.rules import evaluate_hard_rules


BASE_DIR = Path(__file__).resolve().parents[2]
REPOSITORY = build_repository(BASE_DIR)
RETRIEVER = build_retriever(BASE_DIR)
LLM_CLIENT = LlmClient()


def parse_input(state: AgentState) -> AgentState:
    payload = state.get("patient_data", {})
    validated = PatientData(**payload)
    return {
        "patient_data": validated.model_dump(),
        "parsed_ok": True,
        "require_llm": bool(state.get("require_llm", False)),
        "force_llm": bool(state.get("force_llm", False)),
        "llm_model": str(state.get("llm_model", "")).strip(),
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
    return {"risk": new_risk, "risk_level": level, "explanation": explanation, "llm_used": llm_used}


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


def should_use_llm(state: AgentState) -> str:
    return "rag_retrieval" if bool(state.get("route_to_llm") or state.get("force_llm")) else "output_save"
