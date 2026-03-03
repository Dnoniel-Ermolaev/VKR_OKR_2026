from src.core.nodes import (
    clarify_data,
    data_quality_issue,
    diagnostic_uncertain,
    high_risk_fast_track,
    llm_assess,
    llm_parse_history,
    low_risk_observation,
    monitor_plan,
    output_save,
    parse_input,
    rag_retrieval,
    recommend_treatment,
    route_after_clarify,
    route_after_diagnostic,
    route_after_management,
    route_after_pretriage,
    route_from_start,
    router_diagnostic,
    router_management,
    router_pretriage,
    rule_check,
)
from src.core.state import AgentState

try:
    from langgraph.graph import END, START, StateGraph
except Exception:
    END = "__end__"
    START = "__start__"
    StateGraph = None


def build_graph():
    if StateGraph is None:
        return _FallbackGraph()

    workflow = StateGraph(AgentState)
    workflow.add_node("llm_parse_history", llm_parse_history)
    workflow.add_node("parse_input", parse_input)
    workflow.add_node("router_pretriage", router_pretriage)
    workflow.add_node("clarify_data", clarify_data)
    workflow.add_node("data_quality_issue", data_quality_issue)
    workflow.add_node("rule_check", rule_check)
    workflow.add_node("router_diagnostic", router_diagnostic)
    workflow.add_node("high_risk_fast_track", high_risk_fast_track)
    workflow.add_node("diagnostic_uncertain", diagnostic_uncertain)
    workflow.add_node("low_risk_observation", low_risk_observation)
    workflow.add_node("rag_retrieval", rag_retrieval)
    workflow.add_node("llm_assess", llm_assess)
    workflow.add_node("router_management", router_management)
    workflow.add_node("monitor_plan", monitor_plan)
    workflow.add_node("recommend_treatment", recommend_treatment)
    workflow.add_node("output_save", output_save)

    workflow.add_conditional_edges(
        START,
        route_from_start,
        {
            "llm_parse_history": "llm_parse_history",
            "parse_input": "parse_input",
        },
    )
    workflow.add_edge("llm_parse_history", "parse_input")
    workflow.add_edge("parse_input", "router_pretriage")
    workflow.add_conditional_edges(
        "router_pretriage",
        route_after_pretriage,
        {"clarify_data": "clarify_data", "rule_check": "rule_check"},
    )
    workflow.add_conditional_edges(
        "clarify_data",
        route_after_clarify,
        {"llm_parse_history": "llm_parse_history", "data_quality_issue": "data_quality_issue"},
    )
    workflow.add_edge("rule_check", "router_diagnostic")
    workflow.add_conditional_edges(
        "router_diagnostic",
        route_after_diagnostic,
        {
            "high_risk_fast_track": "high_risk_fast_track",
            "diagnostic_uncertain": "diagnostic_uncertain",
            "low_risk_observation": "low_risk_observation",
        },
    )
    workflow.add_edge("diagnostic_uncertain", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "llm_assess")
    workflow.add_edge("llm_assess", "router_management")
    workflow.add_edge("high_risk_fast_track", "router_management")
    workflow.add_edge("low_risk_observation", "router_management")
    workflow.add_conditional_edges(
        "router_management",
        route_after_management,
        {
            "monitor_plan": "monitor_plan",
            "recommend_treatment": "recommend_treatment",
            "output_save": "output_save",
        },
    )
    workflow.add_edge("monitor_plan", "output_save")
    workflow.add_edge("recommend_treatment", "output_save")
    workflow.add_edge("data_quality_issue", "output_save")
    workflow.add_edge("output_save", END)
    return workflow.compile()


class _FallbackGraph:
    """Minimal invoke-compatible fallback when LangGraph is unavailable."""

    def invoke(self, state: AgentState) -> AgentState:
        current = dict(state)
        start_step = route_from_start(current)
        if start_step == "llm_parse_history":
            current.update(llm_parse_history(current))
        current.update(parse_input(current))
        current.update(router_pretriage(current))
        pretriage_step = route_after_pretriage(current)
        if pretriage_step == "clarify_data":
            current.update(clarify_data(current))
            clarify_step = route_after_clarify(current)
            if clarify_step == "llm_parse_history":
                current.update(llm_parse_history(current))
                current.update(parse_input(current))
                current.update(router_pretriage(current))
                pretriage_step = route_after_pretriage(current)
            if pretriage_step == "clarify_data":
                current.update(data_quality_issue(current))
                current.update(output_save(current))
                return current
        current.update(rule_check(current))
        current.update(router_diagnostic(current))
        diagnostic_step = route_after_diagnostic(current)
        if diagnostic_step == "diagnostic_uncertain":
            current.update(diagnostic_uncertain(current))
            current.update(rag_retrieval(current))
            current.update(llm_assess(current))
        elif diagnostic_step == "high_risk_fast_track":
            current.update(high_risk_fast_track(current))
        else:
            current.update(low_risk_observation(current))
        current.update(router_management(current))
        management_step = route_after_management(current)
        if management_step == "monitor_plan":
            current.update(monitor_plan(current))
        elif management_step == "recommend_treatment":
            current.update(recommend_treatment(current))
        current.update(output_save(current))
        return current


graph = build_graph()
