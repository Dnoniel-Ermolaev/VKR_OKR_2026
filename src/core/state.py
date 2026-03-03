from typing import Any, Dict, List, Literal, TypedDict


RiskLevel = Literal["low", "medium", "high"]


class AgentState(TypedDict, total=False):
    patient_data: Dict[str, Any]
    parsed_ok: bool
    require_llm: bool
    force_llm: bool
    llm_model: str
    route_to_llm: bool
    llm_used: bool
    rule_reasons: List[str]
    rag_context: str
    db_results: List[Dict[str, Any]]
    risk: float
    risk_level: RiskLevel
    explanation: str
    save_id: str
