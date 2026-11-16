from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class RAGState(TypedDict, total=False):
    needs_plan: bool
    needs_audit: bool
    needs_context_grade: bool
    needs_reflection: bool
    query: str
    active_query: str
    history: list[dict[str, str]]
    plan: dict[str, Any]
    retrieved: list[dict[str, Any]]
    retry_count: int
    reflection_count: int
    audit_history: list[dict[str, Any]]
    context_grade: dict[str, Any]
    should_abstain: bool
    context: str
    sources: list[dict[str, Any]]
    answer: dict[str, Any]
    audit: dict[str, Any]
    reflection: dict[str, Any]
    reflection_unverified_draft: str
    diagnostics: dict[str, Any]
    settings: dict[str, Any]


def run_rag_graph(
    initial_state: RAGState,
    *,
    plan_node,
    retrieve_node,
    grade_context_node,
    retry_query_node,
    compress_node,
    answer_node,
    audit_node,
    revise_answer_node,
    abstain_node,
) -> RAGState:
    """Run a bounded RAG workflow with an optional corrective retrieval pass."""
    graph = StateGraph(RAGState)
    graph.add_node("query_plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_context", grade_context_node)
    graph.add_node("retry_query", retry_query_node)
    graph.add_node("compress_context", compress_node)
    graph.add_node("generate_answer", answer_node)
    graph.add_node("grade_and_verify", audit_node)
    graph.add_node("reflect_answer", revise_answer_node)
    graph.add_node("abstain", abstain_node)
    graph.add_conditional_edges(
        START,
        lambda state: "query_plan" if state.get("needs_plan") else "retrieve",
        {"query_plan": "query_plan", "retrieve": "retrieve"},
    )
    graph.add_edge("query_plan", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        lambda state: "grade_context" if state.get("needs_context_grade") else "compress_context",
        {"grade_context": "grade_context", "compress_context": "compress_context"},
    )
    graph.add_conditional_edges(
        "grade_context",
        lambda state: (
            "compress_context"
            if state.get("context_grade", {}).get("sufficient")
            else "retry_query"
            if int(state.get("retry_count", 0)) < 1
            and state.get("context_grade", {}).get("retry_query")
            else "abstain"
        ),
        {
            "compress_context": "compress_context",
            "retry_query": "retry_query",
            "abstain": "abstain",
        },
    )
    graph.add_edge("retry_query", "retrieve")
    graph.add_edge("compress_context", "generate_answer")
    graph.add_conditional_edges(
        "generate_answer",
        lambda state: "grade_and_verify" if state.get("needs_audit") else END,
        {"grade_and_verify": "grade_and_verify", END: END},
    )
    graph.add_conditional_edges(
        "grade_and_verify",
        lambda state: (
            "reflect_answer"
            if state.get("needs_reflection")
            and int(state.get("reflection_count", 0)) < 1
            and audit_needs_revision(state.get("audit", {}))
            else END
        ),
        {"reflect_answer": "reflect_answer", END: END},
    )
    graph.add_conditional_edges(
        "reflect_answer",
        lambda state: "grade_and_verify" if state.get("reflection", {}).get("status") == "revised_once" else END,
        {"grade_and_verify": "grade_and_verify", END: END},
    )
    graph.add_edge("abstain", END)
    return graph.compile().invoke(initial_state)


def audit_needs_revision(audit: dict[str, Any]) -> bool:
    """Use bounded audit signals to decide whether one repair pass is warranted."""
    if not audit or audit.get("error"):
        return False
    llm = audit.get("llm_audit", {})
    deterministic = audit.get("deterministic_citation_check", {})
    try:
        low_grade = int(llm.get("grade", 5)) < 4
    except (TypeError, ValueError):
        low_grade = True
    has_invalid_refs = bool(deterministic.get("invalid_refs"))
    has_unsupported = bool(llm.get("unsupported_claims"))
    has_citation_issues = bool(llm.get("citation_issues"))
    citation_unsupported = llm.get("citation_support") is False
    return bool(
        low_grade
        or has_unsupported
        or has_citation_issues
        or citation_unsupported
        or has_invalid_refs
    )
