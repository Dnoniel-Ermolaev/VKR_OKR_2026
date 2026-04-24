from __future__ import annotations

from pathlib import Path
from typing import Dict, TypedDict

from src.core.tools import LlmClient, build_retriever

BASE_DIR = Path(__file__).resolve().parents[2]
RETRIEVER = build_retriever(BASE_DIR)
LLM_CLIENT = LlmClient()


class ReportState(TypedDict, total=False):
    case_summary: str
    report_query: str
    rag_context: str
    citations: list[str]
    llm_model: str
    require_llm: bool
    report: str
    llm_used: bool


def report_rag_retrieval(state: ReportState) -> ReportState:
    hits = RETRIEVER.retrieve_hits(query=str(state.get("report_query", "")), top_k=3)
    return {
        "rag_context": "\n---\n".join(hit.formatted_text(rank=idx + 1) for idx, hit in enumerate(hits)),
        "citations": [hit.citation for hit in hits],
    }


def generate_report(state: ReportState) -> ReportState:
    report, llm_used = LLM_CLIENT.generate_clinical_report(
        case_summary=str(state.get("case_summary", "")),
        rag_context=str(state.get("rag_context", "")),
        require_llm=bool(state.get("require_llm", False)),
        model_name=str(state.get("llm_model", "")).strip() or None,
    )
    return {"report": report, "llm_used": llm_used}


def _fallback_invoke(state: ReportState) -> ReportState:
    current = dict(state)
    current.update(report_rag_retrieval(current))
    current.update(generate_report(current))
    return current


def build_report_graph():
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception:
        return _ReportFallbackGraph()

    workflow = StateGraph(ReportState)
    workflow.add_node("report_rag_retrieval", report_rag_retrieval)
    workflow.add_node("generate_report", generate_report)
    workflow.add_edge(START, "report_rag_retrieval")
    workflow.add_edge("report_rag_retrieval", "generate_report")
    workflow.add_edge("generate_report", END)
    return workflow.compile()


class _ReportFallbackGraph:
    def invoke(self, state: ReportState) -> ReportState:
        return _fallback_invoke(state)


report_graph = build_report_graph()
