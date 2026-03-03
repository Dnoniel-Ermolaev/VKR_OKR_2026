from typing import Any, Dict, List, Literal, TypedDict


RiskLevel = Literal["low", "medium", "high"]


class AgentState(TypedDict, total=False):
    free_text: str
    patient_data: Dict[str, Any]
    parsed_ok: bool
    parse_confidence: float
    missing_fields: List[str]
    clarification_attempts: int
    max_clarification_attempts: int
    require_llm: bool
    force_llm: bool
    llm_model: str
    route_to_llm: bool
    triage_category: Literal["high_risk_fast_track", "diagnostic_uncertain", "low_risk_observation", "data_quality_issue"]
    next_step: Literal[
        "needs_more_data",
        "proceed",
        "retry_parse",
        "data_quality_issue",
        "rule_only",
        "rag_llm",
        "urgent",
        "monitor",
        "recommend_treatment",
        "finalize",
    ]
    route_confidence: float
    route_reason: str
    llm_used: bool
    rule_reasons: List[str]
    rag_context: str
    db_results: List[Dict[str, Any]]
    risk: float
    risk_level: RiskLevel
    explanation: str
    save_id: str
