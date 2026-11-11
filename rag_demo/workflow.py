from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class RAGState(TypedDict, total=False):
    needs_plan: bool
    needs_audit: bool
    query: str
    history: list[dict[str, str]]
    plan: dict[str, Any]
    retrieved: list[dict[str, Any]]
    context: str
    sources: list[dict[str, Any]]
    answer: dict[str, Any]
    audit: dict[str, Any]
    diagnostics: dict[str, Any]
    settings: dict[str, Any]


def run_rag_graph(
    initial_state: RAGState,
    *,
    plan_node,
    retrieve_node,
    compress_node,
    answer_node,
    audit_node,
) -> RAGState:
    """Run explicit retrieval, compression, answer and optional audit stages."""
    graph = StateGraph(RAGState)
    graph.add_node("query_plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("compress_context", compress_node)
    graph.add_node("generate_answer", answer_node)
    graph.add_node("grade_and_verify", audit_node)
    graph.add_conditional_edges(
        START,
        lambda state: "query_plan" if state.get("needs_plan") else "retrieve",
        {"query_plan": "query_plan", "retrieve": "retrieve"},
    )
    graph.add_edge("query_plan", "retrieve")
    graph.add_edge("retrieve", "compress_context")
    graph.add_edge("compress_context", "generate_answer")
    graph.add_conditional_edges(
        "generate_answer",
        lambda state: "grade_and_verify" if state.get("needs_audit") else END,
        {"grade_and_verify": "grade_and_verify", END: END},
    )
    graph.add_edge("grade_and_verify", END)
    return graph.compile().invoke(initial_state)
