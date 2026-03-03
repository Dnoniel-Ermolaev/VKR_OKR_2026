from src.core.nodes import llm_assess, output_save, parse_input, rag_retrieval, rule_check, should_use_llm
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
    workflow.add_node("parse_input", parse_input)
    workflow.add_node("rule_check", rule_check)
    workflow.add_node("rag_retrieval", rag_retrieval)
    workflow.add_node("llm_assess", llm_assess)
    workflow.add_node("output_save", output_save)

    workflow.add_edge(START, "parse_input")
    workflow.add_edge("parse_input", "rule_check")
    workflow.add_conditional_edges(
        "rule_check",
        should_use_llm,
        {
            "rag_retrieval": "rag_retrieval",
            "output_save": "output_save",
        },
    )
    workflow.add_edge("rag_retrieval", "llm_assess")
    workflow.add_edge("llm_assess", "output_save")
    workflow.add_edge("output_save", END)
    return workflow.compile()


class _FallbackGraph:
    """Minimal invoke-compatible fallback when LangGraph is unavailable."""

    def invoke(self, state: AgentState) -> AgentState:
        current = dict(state)
        current.update(parse_input(state))
        current.update(rule_check(current))
        next_step = should_use_llm(current)
        if next_step == "rag_retrieval":
            current.update(rag_retrieval(current))
            current.update(llm_assess(current))
        current.update(output_save(current))
        return current


graph = build_graph()
