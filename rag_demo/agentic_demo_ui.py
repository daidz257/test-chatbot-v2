from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from . import db
from .evaluation import case_metrics, load_jsonl_cases
from .observability import trace_span, tracing_status
from .agentic_patterns import route_with_function_calling
from .rag import (
    _document_inventory_answer,
    is_document_count_question,
    is_document_list_question,
    should_use_all_documents,
)


def _use_selected_question() -> None:
    selected = st.session_state.get("agentic-demo-case", "")
    questions = st.session_state.get("agentic-demo-case-questions", {})
    if selected in questions:
        st.session_state["agentic-demo-question"] = questions[selected]


def _run_variant(
    runner: Callable[..., dict[str, Any]],
    *,
    kb_id: str,
    api_key: str,
    question: str,
    top_k: int,
    candidate_k: int,
    context_token_budget: int,
    embedding_rpm: int,
    generation_rpm: int,
    base_flags: dict[str, bool],
    agentic_flags: dict[str, bool],
    basic_runner: Callable[..., dict[str, Any]],
    progress,
) -> dict[str, Any]:
    flags = {
        **base_flags,
        "query_decomposition": bool(agentic_flags.get("query_decomposition")),
        "reflection_loop": bool(agentic_flags.get("reflection_loop")),
        "function_call_router": bool(agentic_flags.get("function_call_router")),
    }
    selected_route = None
    route_error = None
    if agentic_flags.get("function_call_router"):
        try:
            with trace_span("rag.agentic_pattern_demo", {"rag.kb_id": kb_id, "rag.pattern": "function_calling_router"}):
                selected_route = route_with_function_calling(
                    api_key,
                    question,
                    [],
                    generation_rpm=generation_rpm,
                    progress=progress,
                )
        except Exception as error:
            route_error = str(error)[:500]
    if selected_route and selected_route["route"] in {"document_count", "document_list"}:
        result = _document_inventory_answer(
            kb_id,
            include_names=selected_route["route"] == "document_list",
        )
        result["diagnostics"] = {"pipeline": "Danh mục kho · route bằng Function Calling", "request_meter": {}}
    elif selected_route and selected_route["route"] == "all_documents":
        result = basic_runner(
            kb_id,
            api_key,
            question,
            [],
            mode="Tổng hợp toàn kho",
            top_k=top_k,
            embedding_rpm=embedding_rpm,
            generation_rpm=generation_rpm,
            progress=progress,
        )
    elif not selected_route and (is_document_count_question(question) or is_document_list_question(question)):
        result = _document_inventory_answer(
            kb_id,
            include_names=is_document_list_question(question),
        )
        result["diagnostics"] = {"pipeline": "Danh mục kho · rule-based", "request_meter": {}}
    elif not selected_route and should_use_all_documents(question, "Tự động"):
        result = basic_runner(
            kb_id,
            api_key,
            question,
            [],
            mode="Tổng hợp toàn kho",
            top_k=top_k,
            embedding_rpm=embedding_rpm,
            generation_rpm=generation_rpm,
            progress=progress,
        )
    else:
        result = runner(
            kb_id,
            api_key,
            question,
            [],
            flags=flags,
            top_k=top_k,
            candidate_k=candidate_k,
            context_token_budget=context_token_budget,
            embedding_rpm=embedding_rpm,
            generation_rpm=generation_rpm,
            use_langgraph=True,
            progress=progress,
        )
    if selected_route:
        diagnostics = result.setdefault("diagnostics", {})
        diagnostics["agentic_route"] = selected_route
        route_meter = selected_route.get("meter", {})
        meter = diagnostics.setdefault("request_meter", {})
        for key in ("api_calls", "generation_calls", "embedding_calls", "input_tokens", "output_tokens", "estimated_token_calls"):
            meter[key] = int(meter.get(key, 0)) + int(route_meter.get(key, 0))
        meter["total_tokens"] = int(meter.get("input_tokens", 0)) + int(meter.get("output_tokens", 0))
        meter["labels"] = list(route_meter.get("labels", [])) + list(meter.get("labels", []))
    diagnostics = result.setdefault("diagnostics", {})
    if route_error:
        diagnostics["agentic_route"] = {
            "route": "rule_fallback",
            "label": "Rule-based fallback",
            "error": route_error,
            "fallback": "rule-based route",
        }
    is_inventory = (
        selected_route and selected_route["route"] in {"document_count", "document_list"}
    ) or (
        not selected_route and (is_document_count_question(question) or is_document_list_question(question))
    )
    is_all_documents = (
        selected_route and selected_route["route"] == "all_documents"
    ) or (
        not selected_route and should_use_all_documents(question, "Tự động")
    )
    if is_inventory:
        router_label = "Function Calling" if selected_route else "Rule-based router"
        diagnostics["agentic_pipeline"] = f"{router_label} → SQLite inventory → Answer"
        diagnostics["skipped_patterns"] = ["decomposition", "corrective retrieval", "reflection"]
    elif is_all_documents:
        router_label = "Function Calling" if selected_route else "Rule-based router"
        diagnostics["agentic_pipeline"] = f"{router_label} → Map-reduce toàn kho → Answer"
        diagnostics["skipped_patterns"] = ["decomposition", "corrective retrieval", "reflection"]
    else:
        diagnostics["agentic_pipeline"] = _pipeline_summary(
            agentic=bool(selected_route), flags=flags
        )
    return result


def _route_preview(question: str) -> tuple[str, str]:
    if is_document_count_question(question) or is_document_list_question(question):
        return (
            "Danh mục kho · SQLite",
            "Câu hỏi đếm/liệt kê file được trả trực tiếp từ metadata; không gọi Gemini hay vector retrieval.",
        )
    if should_use_all_documents(question, "Tự động"):
        return (
            "Tổng hợp toàn kho · map-reduce",
            "Câu hỏi tổng hợp rộng được chuyển sang đọc lần lượt các tài liệu sẵn sàng rồi hợp nhất ghi chú.",
        )
    return (
        "Hỏi đáp chi tiết · RAG top-k",
        "Câu hỏi thông thường đi qua tìm kiếm vector theo top-k rồi sinh câu trả lời có nguồn.",
    )


def _pipeline_summary(*, agentic: bool, flags: dict[str, bool]) -> str:
    """Describe the configured path in concise, user-visible stages."""
    route = "Function Calling chọn route → " if agentic and flags.get("function_call_router") else "Rule-based route → "
    if agentic and flags.get("query_decomposition"):
        route += "Planner phân rã (nếu cần) → "
    route += "Retrieve top-k → "
    if flags.get("corrective_retrieval"):
        route += "Evaluate context → Retry tối đa 1 lần nếu thiếu → "
    route += "Sinh câu trả lời"
    if agentic and flags.get("reflection_loop"):
        route += " → Audit → Sửa tối đa 1 lần → Audit lại → Trả lời / Từ chối kết luận"
    elif flags.get("answer_grading") or flags.get("citation_verification"):
        route += " → Audit → Trả lời"
    else:
        route += " → Trả lời"
    return "Câu hỏi → " + route


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.caption("Không có nguồn được trả về.")
        return
    for index, source in enumerate(sources, start=1):
        title = (
            f"{source.get('ref', f'S{index}')} · {source.get('file_name', 'Tài liệu')} · "
            f"trang {source.get('page_number') or '?'} · chunk "
            f"{int(source.get('chunk_index', -1)) + 1 if source.get('chunk_index') is not None else '?'}"
        )
        with st.expander(title):
            content = source.get("context_used") or source.get("content") or ""
            if content:
                st.write(str(content)[:5000])
            st.caption(f"Chunk ID: {source.get('chunk_id', 'không có')}")


def _render_grade_and_attempts(result: dict[str, Any]) -> None:
    diagnostics = result.get("diagnostics", {})
    attempts = diagnostics.get("retrieval_attempts", [])
    grades = diagnostics.get("context_grades", [])
    if not grades and not diagnostics.get("abstained_after_retrieval"):
        st.info("Luồng tuần tự: retrieve → context → answer. Evaluator và corrective retrieval đang tắt.")
        if attempts:
            for attempt in attempts:
                st.caption(
                    f"Lượt retrieval {attempt.get('attempt', '?')}: "
                    f"{attempt.get('candidate_count', 0)} ứng viên · query: {attempt.get('query', '')}"
                )
        return

    st.markdown("**Quyết định của evaluator và lượt truy xuất**")
    grade_by_attempt = {int(item.get("attempt", index + 1)): item for index, item in enumerate(grades)}
    for attempt in attempts:
        number = int(attempt.get("attempt", 0))
        grade = grade_by_attempt.get(number, {})
        with st.container(border=True):
            st.markdown(f"**Lượt truy xuất {number}** · {attempt.get('candidate_count', 0)} ứng viên")
            st.code(attempt.get("query", ""), language=None)
            if grade:
                status = "Đủ bằng chứng" if grade.get("sufficient") else "Chưa đủ bằng chứng"
                st.write(f"**Evaluator:** {status}. {grade.get('reason', '')}")
                if grade.get("retry_query"):
                    st.write(f"**Query sửa:** `{grade['retry_query']}`")
                if grade.get("error"):
                    st.warning(f"Evaluator lỗi và được bỏ qua: {grade['error']}")
                if grade.get("evidence_sources"):
                    st.caption("Nguồn evaluator đã xem: " + " · ".join(grade["evidence_sources"]))
            else:
                st.caption("Không có context grade cho lượt này.")
    if diagnostics.get("abstained_after_retrieval"):
        st.warning("Workflow dừng trước bước sinh câu trả lời vì bằng chứng vẫn chưa đủ sau lượt thử lại.")


def _render_variant_result(title: str, result: dict[str, Any]) -> None:
    diagnostics = result.get("diagnostics", {})
    meter = diagnostics.get("request_meter", {})
    st.markdown(f"### {title}")
    st.write(result.get("answer", "(Không có câu trả lời)"))
    route = diagnostics.get("agentic_route", {})
    if diagnostics.get("agentic_pipeline"):
        st.caption("Pipeline đã chạy: " + diagnostics["agentic_pipeline"])
    if diagnostics.get("skipped_patterns"):
        st.caption("Không áp dụng ở route này: " + ", ".join(diagnostics["skipped_patterns"]))
    if route:
        if route.get("error"):
            st.warning(f"Function Calling lỗi; đã fallback về rule route. Lỗi: {route['error']}")
        else:
            st.info(
                f"Function Calling chọn: {route.get('label', route.get('route', 'không rõ'))}. "
                f"Lý do: {route.get('reason', '—')}"
            )
    plan = diagnostics.get("plan", {})
    if plan.get("decomposition_queries"):
        st.markdown("**Các câu hỏi con được planner tạo**")
        for index, subquestion in enumerate(plan["decomposition_queries"], start=1):
            st.markdown(f"{index}. {subquestion}")
        anchors = diagnostics.get("retrieval", {}).get("decomposition_anchor_labels", [])
        if anchors:
            st.caption("Chunk được giữ để bao phủ câu hỏi con: " + " · ".join(anchors))
    reflection = diagnostics.get("reflection", {})
    if diagnostics.get("reflection_status"):
        status_label = {
            "passed": "Audit đạt",
            "unresolved_after_one_repair": "Chưa đạt sau một lần sửa · đã abstain",
            "audit_unavailable": "Không audit được · đã chặn kết luận",
        }.get(diagnostics["reflection_status"], diagnostics["reflection_status"])
        st.markdown(f"**Reflection:** {status_label} · số lần sửa {diagnostics.get('reflection_count', 0)}")
        for audit_index, audit in enumerate(diagnostics.get("audit_history", []), start=1):
            llm_audit = audit.get("llm_audit", {})
            st.caption(
                f"Audit {audit_index}: điểm {llm_audit.get('grade', '—')}/5 · "
                f"{llm_audit.get('grade_reason', audit.get('error', 'Không có ghi chú'))}"
            )
        if reflection.get("feedback"):
            with st.expander("Phản hồi đã dùng khi sửa", expanded=False):
                st.json(reflection["feedback"])
    validation = diagnostics.get("answer_validation", {})
    if validation and not validation.get("passed"):
        st.warning("Citation ID/presence heuristic phát hiện vấn đề; hãy đối chiếu nội dung nguồn.")
    a, b, c, d = st.columns(4)
    a.metric("Độ trễ", f"{diagnostics.get('latency_ms', 0)} ms")
    b.metric("API calls", int(meter.get("api_calls", 0)))
    c.metric("Token", int(meter.get("total_tokens", 0)))
    d.metric("Tìm lại", int(diagnostics.get("retry_count", 0)))
    _render_grade_and_attempts(result)
    with st.expander("Nguồn và nội dung context", expanded=True):
        _render_sources(result.get("sources", []))
    if meter.get("estimated_token_calls"):
        st.caption("Một phần token được ước tính vì provider không trả usage metadata.")


def render_agentic_demo(
    *,
    root_dir: Path,
    kb_id: str,
    kb_name: str,
    api_key: str,
    top_k: int,
    candidate_k: int,
    context_token_budget: int,
    embedding_rpm: int,
    generation_rpm: int,
    base_flags: dict[str, bool],
    advanced_runner: Callable[..., dict[str, Any]],
    basic_runner: Callable[..., dict[str, Any]],
) -> None:
    st.title("🧭 Agentic Design Patterns · Demo trực tiếp")
    st.caption(
        "So sánh cùng một câu hỏi trên cùng cấu hình RAG hiện tại. Baseline tắt các pattern trong demo; "
        "nhánh thử nghiệm chỉ bật các pattern bạn chọn."
    )

    st.markdown("#### Pattern được minh họa")
    pattern_cols = st.columns(4)
    pattern_cols[0].markdown("**1 · Retrieve**\n\nTìm bằng chứng bằng dense vector.")
    pattern_cols[1].markdown("**2 · Evaluate**\n\nGemini kiểm tra context có đủ căn cứ không.")
    pattern_cols[2].markdown("**3 · Correct once**\n\nNếu thiếu, tạo query sửa và tìm lại một lần.")
    pattern_cols[3].markdown("**4 · Answer / abstain**\n\nTrả lời từ context hoặc dừng nếu vẫn thiếu.")
    st.caption(
        "Các pattern được bật riêng để so với route/workflow hiện tại: tool router, phân rã tối đa 3 truy vấn con, "
        "corrective retrieval và reflection sửa tối đa một lần. Đây là agentic workflow có giới hạn, không phải autonomous agent."
    )
    trace_state = tracing_status()
    st.caption(
        "Phoenix trace đang bật: hai lượt demo sẽ xuất span để quan sát."
        if trace_state.get("enabled")
        else "Phoenix trace đang tắt; bật PHOENIX_TRACING_ENABLED để xem hai lượt trong Phoenix."
    )

    demo_path = root_dir / "evals" / "demo_dataset.jsonl"
    try:
        cases = load_jsonl_cases(demo_path.read_text(encoding="utf-8")) if demo_path.exists() else []
    except (OSError, ValueError, json.JSONDecodeError):
        cases = []
    labels = {f"{case['id']} · {case['question']}": case for case in cases}
    question_col, choose_col = st.columns([4, 2])
    with question_col:
        st.text_area(
            "Câu hỏi",
            key="agentic-demo-question",
            height=90,
            placeholder="Nhập câu hỏi cần kiểm tra trên tài liệu trong kho…",
        )
    with choose_col:
        if labels:
            st.selectbox(
                "Hoặc chọn case demo",
                options=["Tự nhập câu hỏi", *labels],
                key="agentic-demo-case",
            )
            st.session_state["agentic-demo-case-questions"] = {
                label: case["question"] for label, case in labels.items()
            }
            st.button(
                "Dùng câu hỏi này",
                key="agentic-demo-use-case",
                on_click=_use_selected_question,
                disabled=st.session_state.get("agentic-demo-case") == "Tự nhập câu hỏi",
            )

    question = str(st.session_state.get("agentic-demo-question", "")).strip()
    if question:
        route, route_note = _route_preview(question)
        with st.expander("🔀 Routing hiện có trong chatbot · xem quyết định", expanded=False):
            st.markdown(f"**Nhánh được chọn: {route}**")
            st.write(route_note)
            st.caption(
                "Preview dùng rule/router đang có của chatbot ở chế độ Tự động. Đây là routing theo ý định đơn giản, "
                "không phải LLM tự chọn tool."
            )

    st.markdown("#### Chọn pattern cho lượt thử agentic")
    option_cols = st.columns(3)
    selected_flags = {
        "function_call_router": option_cols[0].checkbox(
            "Function Calling router", value=bool(base_flags.get("function_call_router")), key="agentic-demo-router"
        ),
        "query_decomposition": option_cols[1].checkbox(
            "Decompose multi-hop", value=bool(base_flags.get("query_decomposition")), key="agentic-demo-decomposition"
        ),
        "reflection_loop": option_cols[2].checkbox(
            "Reflection tối đa 1 lần", value=bool(base_flags.get("reflection_loop")), key="agentic-demo-reflection"
        ),
    }
    st.caption("Baseline tắt ba nâng cấp mới. Corrective Retrieval và các kỹ thuật RAG hiện có giữ nguyên trạng thái ở cả hai nhánh.")
    pattern_keys = {"function_call_router", "query_decomposition", "reflection_loop"}
    baseline_flags = {key: bool(value) for key, value in base_flags.items() if key not in pattern_keys}
    baseline_flags.update({key: False for key in pattern_keys})
    selected_base_flags = {**baseline_flags, **selected_flags}
    common_techniques = {
        "parent_child": "Parent-Child",
        "hybrid_search": "Hybrid Search",
        "reranker": "Reranker local",
        "query_rewrite": "Query Rewrite",
        "multi_query": "Multi-query",
        "hyde": "HyDE",
        "context_compression": "Context compression",
        "answer_grading": "Answer grading",
        "citation_verification": "Citation verification",
    }
    enabled_common = [label for key, label in common_techniques.items() if baseline_flags.get(key)]
    st.caption("Kỹ thuật RAG dùng chung: " + (" · ".join(enabled_common) if enabled_common else "không có kỹ thuật bổ sung"))
    pipeline_cols = st.columns(2)
    with pipeline_cols[0]:
        st.markdown("**Baseline · pattern mới tắt**")
        st.code(_pipeline_summary(agentic=False, flags=baseline_flags), language=None)
    with pipeline_cols[1]:
        st.markdown("**Nhánh thử nghiệm · theo lựa chọn**")
        st.code(_pipeline_summary(agentic=True, flags=selected_base_flags), language=None)
    preview_route, _ = _route_preview(question) if question else ("Hỏi đáp chi tiết · RAG top-k", "")
    if selected_flags.get("function_call_router") and preview_route.startswith("Danh mục"):
        st.info("Nếu Function Calling chọn đếm/liệt kê file, câu trả lời lấy trực tiếp từ SQLite; các bước RAG phía sau sẽ được bỏ qua.")
    elif selected_flags.get("function_call_router") and preview_route.startswith("Tổng hợp"):
        st.info("Nếu Function Calling chọn tổng hợp toàn kho, chatbot dùng map-reduce; decomposition và reflection của RAG top-k không áp dụng cho nhánh này.")

    ready_docs = [item for item in db.list_documents(kb_id) if item.get("status") == "Sẵn sàng"]
    if not ready_docs:
        st.warning(f"Kho “{kb_name}” chưa có tài liệu đã lập chỉ mục.")
    elif not api_key:
        st.warning("Nhập Gemini API key trong mục Kết nối AI để chạy so sánh.")
    else:
        st.caption(
            f"Kho: {kb_name} · {len(ready_docs)} tài liệu sẵn sàng · top-k {top_k} · "
            f"cùng model và cùng cấu hình retrieval cho hai biến thể."
        )

    button_cols = st.columns([2, 2, 5])
    run_compare = button_cols[0].button(
        "▶ So sánh hai workflow", type="primary", key="agentic-demo-run",
        disabled=not question or not api_key or not ready_docs,
    )
    if button_cols[1].button("Xóa kết quả", key="agentic-demo-clear"):
        st.session_state.pop("agentic-demo-last-result", None)
        st.rerun()
    button_cols[2].caption(
        "Chạy baseline và cấu hình đã chọn. Mỗi pattern có chi phí riêng; function router thêm một lượt Gemini, "
        "reflection có thể thêm audit + lần viết lại."
    )

    if run_compare:
        status = st.empty()
        results: dict[str, Any] = {}
        variants = [
            ("baseline", "Baseline · pattern mới tắt", {
                "function_call_router": False,
                "query_decomposition": False,
                "reflection_loop": False,
            }),
            ("agentic", "Agentic · pattern đã chọn", selected_flags),
        ]
        for variant_key, label, agentic_flags in variants:
            status.info(f"Đang chạy {label}…")
            progress = lambda message, label=label: status.caption(f"{label}: {message}")
            try:
                with st.spinner(f"Đang gọi Gemini cho: {label}"):
                    results[variant_key] = _run_variant(
                        advanced_runner,
                        kb_id=kb_id,
                        api_key=api_key,
                        question=question,
                        top_k=top_k,
                        candidate_k=candidate_k,
                        context_token_budget=context_token_budget,
                        embedding_rpm=embedding_rpm,
                        generation_rpm=generation_rpm,
                        base_flags=baseline_flags,
                        agentic_flags=agentic_flags,
                        basic_runner=basic_runner,
                        progress=progress,
                    )
            except Exception as error:
                results[variant_key] = {"error": str(error)[:800]}
        st.session_state["agentic-demo-last-result"] = {
            "kb_id": kb_id,
            "question": question,
            "results": results,
            "case": labels.get(st.session_state.get("agentic-demo-case", "")),
        }
        status.success("Hai cấu hình đã hoàn tất. So sánh route, planner, audit, nguồn và chi phí bên dưới.")

    saved = st.session_state.get("agentic-demo-last-result")
    if not saved:
        return
    if saved.get("kb_id") != kb_id:
        st.info("Kết quả trước thuộc kho khác. Chạy lại để so sánh trên kho đang chọn.")
        return
    st.markdown("---")
    st.markdown(f"#### Kết quả cho: {saved.get('question', '')}")
    results = saved.get("results", {})
    off = results.get("baseline", {})
    on = results.get("agentic", {})
    if off.get("error") or on.get("error"):
        for label, result in (("Baseline", off), ("Agentic", on)):
            if result.get("error"):
                st.error(f"{label}: {result['error']}")
    if off and on and not off.get("error") and not on.get("error"):
        off_diag, on_diag = off.get("diagnostics", {}), on.get("diagnostics", {})
        off_meter = off_diag.get("request_meter", {})
        on_meter = on_diag.get("request_meter", {})
        comparison = [
            {"Chỉ số": "Độ trễ (ms)", "Baseline": off_diag.get("latency_ms", 0), "Agentic": on_diag.get("latency_ms", 0)},
            {"Chỉ số": "API calls", "Baseline": off_meter.get("api_calls", 0), "Agentic": on_meter.get("api_calls", 0)},
            {"Chỉ số": "Tổng token", "Baseline": off_meter.get("total_tokens", 0), "Agentic": on_meter.get("total_tokens", 0)},
            {"Chỉ số": "Nguồn cuối", "Baseline": len(off.get("sources", [])), "Agentic": len(on.get("sources", []))},
            {"Chỉ số": "Số lượt tìm lại", "Baseline": 0, "Agentic": on_diag.get("retry_count", 0)},
        ]
        st.dataframe(comparison, hide_index=True, width="stretch")
        case = saved.get("case")
        if case:
            expected = {str(name).casefold() for name in case.get("expected_sources", [])}
            available = {str(item.get("file_name", "")).casefold() for item in ready_docs}
            missing_expected = sorted(expected - available)
            if missing_expected:
                st.warning(
                    "Nguồn kỳ vọng của case demo không có trong kho hiện tại; metric nguồn không thể diễn giải công bằng: "
                    + ", ".join(missing_expected)
                )
            st.markdown("**Metric tham chiếu cho case đã chọn**")
            metrics_off = case_metrics(case, str(off.get("answer", "")), off.get("sources", []))
            metrics_on = case_metrics(case, str(on.get("answer", "")), on.get("sources", []))
            st.dataframe(
                [
                    {"Metric": key, "Baseline": metrics_off.get(key), "Agentic": metrics_on.get(key)}
                    for key in ("retrieval_hit", "expected_source_recall", "retrieved_document_precision", "abstention_correct", "citation_refs_valid")
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption("Metric nguồn khớp theo tên file; citation/abstention có heuristic, không phải kiểm chứng ngữ nghĩa.")
        col_off, col_on = st.columns(2)
        with col_off:
            _render_variant_result("Baseline · pattern mới tắt", off)
        with col_on:
            _render_variant_result("Nhánh thử nghiệm · pattern đã chọn", on)
        st.info(
            "Cách diễn giải: evaluator bật chỉ có lợi nếu bằng chứng/đáp án cải thiện đáng kể so với phần chi phí và độ trễ tăng thêm. "
            "Một câu hỏi không đủ để kết luận pattern tốt hơn nói chung; dùng RAG Evaluation để chạy nhiều case."
        )
