from __future__ import annotations

import csv
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from . import db
from .evaluation import (
    agent_selection_metrics,
    aggregate_results,
    case_metrics,
    judge_answer,
    load_jsonl_cases,
    normalize_filename_key,
)
from .observability import trace_span
from .quota import RequestMeter


def render_evaluation_workbench(
    *,
    root_dir: Path,
    kb_id: str,
    kb_name: str,
    api_key: str,
    feature_flags: dict[str, bool],
    top_k: int,
    candidate_k: int,
    context_token_budget: int,
    embedding_rpm: int,
    generation_rpm: int,
    use_langgraph: bool,
    basic_runner: Callable[..., dict[str, Any]],
    advanced_runner: Callable[..., dict[str, Any]],
    multi_agent_runner: Callable[..., dict[str, Any]],
) -> None:
    st.title("📊 RAG Evaluation & Observability")
    st.caption(
        "Chạy cùng một bộ câu hỏi qua RAG cơ bản, nâng cao và multi-agent; đo retrieval, citations, "
        "abstention, định tuyến agent, độ trễ và quota. Đây là benchmark nhỏ để demo, không thay thế tập đánh giá đại diện."
    )
    st.info(
        "Mỗi lần chạy gọi Gemini cho từng câu hỏi được chọn. LLM Judge thêm một lượt Gemini cho mỗi kết quả; "
        "điểm judge là tín hiệu tham khảo và cần spot-check bằng người."
    )

    demo_path = root_dir / "evals" / "demo_dataset.jsonl"
    demo_text = demo_path.read_text(encoding="utf-8") if demo_path.exists() else ""
    template = (
        '{"id":"case-1","question":"Câu hỏi","reference_answer":"Đáp án kỳ vọng",'
        '"expected_sources":["ten-file.pdf"],"answerable":true,"category":"fact lookup"}\n'
        '{"id":"case-2","question":"Thông tin không có trong tài liệu là gì?",'
        '"reference_answer":"Chưa đủ căn cứ để trả lời.","expected_sources":["ten-file.pdf"],'
        '"answerable":false,"category":"abstention","expected_agents":["risk_analyst"]}\n'
    )
    with st.expander("Dataset: schema và tải mẫu", expanded=False):
        st.markdown(
            "Mỗi dòng JSONL có `id`, `question`, `reference_answer`, `expected_sources` (tên file) và "
            "`answerable` (true/false); `category` và `expected_agents` là tùy chọn. Agent ID hợp lệ: "
            "`evidence_analyst`, `table_analyst`, `cross_document_analyst`, `requirements_analyst`, `risk_analyst`. "
            "Nên có câu trả lời được/không được, "
            "câu hỏi số liệu/bảng và câu cần nhiều file."
        )
        st.download_button("Tải JSONL template", template, "rag_eval_template.jsonl", "application/jsonl")
        st.download_button("Tải bộ demo 12 case", demo_text, "demo_dataset.jsonl", "application/jsonl")

    upload = st.file_uploader(
        "Tùy chọn: dùng bộ JSONL của bạn", type=["jsonl"], key="rag-eval-upload"
    )
    payload = upload.getvalue().decode("utf-8-sig") if upload else demo_text
    try:
        cases = load_jsonl_cases(payload)
    except ValueError as error:
        st.error(str(error))
        return
    st.caption(f"Dataset: {len(cases)} case · kho đang chọn: {kb_name}")

    ready_docs = [
        item for item in db.list_documents(kb_id)
        if item.get("status") == "Sẵn sàng"
    ]
    ready_files = {
        normalize_filename_key(str(item.get("file_name", "")))
        for item in ready_docs
    }
    expected_files = {
        normalize_filename_key(str(name))
        for case in cases for name in case.get("expected_sources", [])
    }
    missing = sorted(
        name for name in expected_files
        if name not in ready_files
    )
    if missing:
        st.warning(
            "Một số expected_sources không có trong kho đã chọn; chỉ số retrieval sẽ bị ảnh hưởng: "
            + ", ".join(missing[:8])
        )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        pipelines = st.multiselect(
            "Pipeline", ["RAG cơ bản", "RAG nâng cao", "Multi-agent RAG"],
            default=["RAG cơ bản", "RAG nâng cao"], key="rag-eval-pipelines",
        )
    with col_b:
        case_cap = min(3, len(cases)) if "Multi-agent RAG" in pipelines else len(cases)
        limit = st.number_input(
            "Số case tối đa", min_value=1, max_value=case_cap,
            value=min(3 if "Multi-agent RAG" in pipelines else 6, case_cap), step=1,
            key=f"rag-eval-case-limit-{int('Multi-agent RAG' in pipelines)}",
            help="Khi có Multi-agent RAG, tối đa 3 case mỗi lượt để giới hạn quota và thời gian.",
        )
    with col_c:
        use_judge = st.checkbox(
            "Bật LLM Judge", value=False, key="rag-eval-judge",
            help="Thêm một generation call cho mỗi kết quả.",
        )
    multi_agent_max_agents = 2
    multi_agent_critic = True
    if "Multi-agent RAG" in pipelines:
        with st.expander("⚙️ Chi phí Multi-agent", expanded=True):
            ma_a, ma_b = st.columns(2)
            with ma_a:
                multi_agent_max_agents = st.number_input(
                    "Số specialist tối đa", min_value=1, max_value=3, value=2, step=1,
                    key="rag-eval-ma-agents",
                )
            with ma_b:
                multi_agent_critic = st.checkbox(
                    "Bật Critic (thêm 1 generation call)", value=True,
                    key="rag-eval-ma-critic",
                )
            st.caption(
                f"Mỗi case multi-agent: 1 embedding batch + tối đa "
                f"{1 + int(multi_agent_max_agents) + 1 + int(multi_agent_critic)} generation calls. "
                "Worker chạy tuần tự theo giới hạn RPM."
            )
    selected = cases[: int(limit)]
    has_planner = any(feature_flags.get(name) for name in ("query_rewrite", "multi_query", "hyde"))
    has_audit = bool(feature_flags.get("answer_grading") or feature_flags.get("citation_verification"))
    advanced_generations_max = (
        1 + int(has_planner) + int(has_audit)
        + 2 * int(feature_flags.get("corrective_retrieval", False))
    )
    generation_estimate = 0
    embedding_estimate = 0
    if "RAG cơ bản" in pipelines:
        generation_estimate += len(selected) * (1 + int(use_judge))
        embedding_estimate += len(selected)
    if "RAG nâng cao" in pipelines:
        generation_estimate += len(selected) * (advanced_generations_max + int(use_judge))
        embedding_estimate += len(selected) * (1 + int(feature_flags.get("corrective_retrieval", False)))
    if "Multi-agent RAG" in pipelines:
        multi_agent_generation_max = 1 + int(multi_agent_max_agents) + 1 + int(multi_agent_critic)
        generation_estimate += len(selected) * (multi_agent_generation_max + int(use_judge))
        embedding_estimate += len(selected)
    st.caption(
        f"Ước tính tối đa: khoảng {generation_estimate} generation call, {embedding_estimate} embedding batch; "
        "không tính retry do lỗi mạng/quota."
    )
    run_signature = hashlib.sha256(
        json.dumps(
            {
                "metrics_version": 3,
                "kb_id": kb_id,
                "dataset": payload,
                "corpus": [
                    {
                        "id": item.get("id"),
                        "file_name": item.get("file_name"),
                        "status": item.get("status"),
                        "created_at": item.get("created_at"),
                    }
                    for item in ready_docs
                ],
                "pipelines": pipelines,
                "case_limit": int(limit),
                "judge": use_judge,
                "flags": feature_flags,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "context_token_budget": context_token_budget,
                "embedding_rpm": embedding_rpm,
                "generation_rpm": generation_rpm,
                "use_langgraph": use_langgraph,
                "multi_agent_max_agents": int(multi_agent_max_agents),
                "multi_agent_critic": multi_agent_critic,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    if st.button(
        "▶ Chạy evaluation", type="primary", disabled=not api_key or not pipelines,
        key="rag-eval-run",
    ):
        rows: list[dict[str, Any]] = []
        total = len(selected) * len(pipelines)
        progress_bar = st.progress(0.0)
        status = st.empty()
        for index, (case, pipeline) in enumerate(
            [(case, pipeline) for case in selected for pipeline in pipelines], start=1
        ):
            status.info(f"{index}/{total} · {pipeline} · {case['id']}")
            case_progress = lambda message, prefix=f"{pipeline} · {case['id']}: ": status.caption(prefix + message)
            try:
                with trace_span(
                    "rag.evaluation.case",
                    {"rag.kb_id": kb_id, "eval.case_id": case["id"], "eval.pipeline": pipeline},
                ):
                    if pipeline == "RAG cơ bản":
                        result = basic_runner(
                            kb_id, api_key, case["question"], [],
                            mode="Hỏi đáp chi tiết", top_k=top_k,
                            embedding_rpm=embedding_rpm, generation_rpm=generation_rpm,
                            progress=case_progress,
                        )
                    elif pipeline == "RAG nâng cao":
                        result = advanced_runner(
                            kb_id, api_key, case["question"], [], flags=feature_flags,
                            top_k=top_k, candidate_k=candidate_k,
                            context_token_budget=context_token_budget,
                            embedding_rpm=embedding_rpm, generation_rpm=generation_rpm,
                            use_langgraph=use_langgraph, progress=case_progress,
                        )
                    else:
                        result = multi_agent_runner(
                            kb_id, api_key, case["question"], [], flags=feature_flags,
                            top_k=top_k, candidate_k=candidate_k,
                            context_token_budget=context_token_budget,
                            embedding_rpm=embedding_rpm, generation_rpm=generation_rpm,
                            max_agents=int(multi_agent_max_agents),
                            include_critic=multi_agent_critic,
                            progress=case_progress,
                        )
                error_text = ""
            except Exception as error:
                error_text = str(error)[:600]
                result = {
                    "answer": "", "sources": [],
                    "diagnostics": {
                        "pipeline": pipeline, "request_meter": RequestMeter().as_dict(),
                        "latency_ms": 0,
                    },
                }

            answer = str(result.get("answer", ""))
            sources = result.get("sources", [])
            metrics = case_metrics(case, answer, sources)
            route = result.get("diagnostics", {}).get("supervisor", {})
            metrics.update(
                agent_selection_metrics(case, route.get("selected_agents", []))
                if pipeline == "Multi-agent RAG"
                else {"agent_selection_precision": None, "agent_selection_recall": None, "agent_selection_f1": None}
            )
            chunk_contents = db.get_chunk_contents(
                [str(source.get("chunk_id", "")) for source in sources if source.get("chunk_id")]
            )
            contexts = [
                f"{source.get('file_name', 'Tài liệu')} · trang {source.get('page_number') or '?'}\n"
                + str(source.get("context_used") or chunk_contents.get(str(source.get("chunk_id", "")), ""))
                for source in sources
            ]
            judge_meter = RequestMeter()
            judge_result: dict[str, Any] = {}
            judge_started = time.perf_counter()
            if use_judge and not error_text:
                try:
                    with trace_span(
                        "rag.evaluation.judge",
                        {"eval.case_id": case["id"], "eval.pipeline": pipeline},
                    ):
                        judge_result = judge_answer(
                            api_key, case, answer, contexts,
                            rpm_limit=generation_rpm, meter=judge_meter, progress=case_progress,
                        )
                except Exception as error:
                    judge_result = {"error": str(error)[:400], "label": "LLM Judge lỗi"}
            judge_latency_ms = round((time.perf_counter() - judge_started) * 1000) if use_judge else 0
            rows.append(
                {
                    "pipeline": pipeline, "case": case, "answer": answer,
                    "sources": sources, "diagnostics": result.get("diagnostics", {}),
                    "metrics": metrics, "judge": judge_result,
                    "judge_meter": judge_meter.as_dict(), "judge_latency_ms": judge_latency_ms,
                    "error": error_text,
                }
            )
            progress_bar.progress(index / max(1, total))
        st.session_state["rag-eval-last-run"] = rows
        st.session_state["rag-eval-run-signature"] = run_signature
        try:
            db.save_evaluation_run(
                knowledge_base_id=kb_id,
                dataset_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                config={
                    "pipelines": pipelines,
                    "case_limit": int(limit),
                    "judge": use_judge,
                    "feature_flags": feature_flags,
                    "top_k": top_k,
                    "candidate_k": candidate_k,
                    "context_token_budget": context_token_budget,
                    "multi_agent_max_agents": int(multi_agent_max_agents),
                    "multi_agent_critic": multi_agent_critic,
                },
                summary=aggregate_results(rows),
                results=[
                    {
                        "pipeline": row["pipeline"], "case": row["case"],
                        "answer": row["answer"], "sources": row["sources"],
                        "diagnostics": row["diagnostics"], "metrics": row["metrics"],
                        "judge": row["judge"], "judge_meter": row["judge_meter"],
                        "judge_latency_ms": row["judge_latency_ms"], "error": row["error"],
                    }
                    for row in rows
                ],
            )
        except Exception as error:
            st.warning(f"Không lưu được lịch sử evaluation: {str(error)[:250]}")
        status.success("Evaluation hoàn tất. Xem từng case hoặc tải kết quả bên dưới.")

    rows = st.session_state.get("rag-eval-last-run", [])
    if not rows:
        return
    if st.session_state.get("rag-eval-run-signature") != run_signature:
        st.info("Các kết quả đang lưu thuộc kho, dataset hoặc cấu hình trước đó. Chạy lại để so sánh đúng cấu hình hiện tại.")
        return
    st.subheader("Kết quả tổng hợp")
    st.dataframe(aggregate_results(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Retrieval và citation ID được đo xác định theo dataset. Abstention dựa trên nhãn answerable "
        "và cụm từ từ chối (heuristic). Các chỉ số không tự chứng minh câu trả lời đúng; hãy spot-check nguồn."
    )

    export_rows = []
    for row in rows:
        diagnostics = row.get("diagnostics", {})
        export_rows.append(
            {
                "pipeline": row["pipeline"],
                "case_id": row["case"].get("id"),
                "category": row["case"].get("category"),
                "question": row["case"].get("question"),
                "answer": row.get("answer"),
                "expected_answerable": row["case"].get("answerable"),
                "expected_sources": row["case"].get("expected_sources"),
                "expected_agents": row["case"].get("expected_agents", []),
                "sources": [
                    {key: source.get(key) for key in ("ref", "file_name", "page_number", "chunk_index", "chunk_id")}
                    for source in row.get("sources", [])
                ],
                "metrics": row.get("metrics"), "judge": row.get("judge"),
                "supervisor": diagnostics.get("supervisor", {}),
                "agent_trace": diagnostics.get("agents", []),
                "critic": diagnostics.get("critic", {}),
                "latency_ms": diagnostics.get("latency_ms"),
                "judge_latency_ms": row.get("judge_latency_ms", 0),
                "request_meter": diagnostics.get("request_meter", {}),
                "judge_meter": row.get("judge_meter", {}), "error": row.get("error"),
            }
        )
    st.download_button(
        "Tải kết quả JSON", json.dumps(export_rows, ensure_ascii=False, indent=2),
        "rag-evaluation-results.json", "application/json", key="rag-eval-json-download",
    )
    buffer = io.StringIO()
    fields = [
        "pipeline", "case_id", "category", "question", "answer", "retrieval_hit",
        "expected_source_recall", "retrieved_document_precision", "abstention_correct",
        "citation_refs_valid", "citation_present", "citation_check_passed", "retrieved_files",
        "agent_selection_precision", "agent_selection_recall", "agent_selection_f1",
        "latency_ms", "api_calls", "input_tokens",
        "output_tokens", "judge_latency_ms", "judge_correctness", "judge_groundedness",
        "judge_answer_relevance", "error",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in export_rows:
        metrics, judge = row.get("metrics") or {}, row.get("judge") or {}
        meter, judge_meter = row.get("request_meter") or {}, row.get("judge_meter") or {}
        writer.writerow(
            {
                "pipeline": row["pipeline"], "case_id": row["case_id"], "category": row["category"],
                "question": row["question"], "answer": row["answer"],
                **{key: metrics.get(key) for key in (
                    "retrieval_hit", "expected_source_recall", "retrieved_document_precision",
                    "abstention_correct", "citation_refs_valid", "citation_present",
                    "citation_check_passed", "retrieved_files", "agent_selection_precision",
                    "agent_selection_recall", "agent_selection_f1",
                )},
                "latency_ms": row["latency_ms"],
                "judge_latency_ms": row.get("judge_latency_ms", 0),
                "api_calls": int(meter.get("api_calls", 0)) + int(judge_meter.get("api_calls", 0)),
                "input_tokens": int(meter.get("input_tokens", 0)) + int(judge_meter.get("input_tokens", 0)),
                "output_tokens": int(meter.get("output_tokens", 0)) + int(judge_meter.get("output_tokens", 0)),
                "judge_correctness": judge.get("correctness"), "judge_groundedness": judge.get("groundedness"),
                "judge_answer_relevance": judge.get("answer_relevance"), "error": row.get("error"),
            }
        )
    st.download_button(
        "Tải kết quả CSV", buffer.getvalue(), "rag-evaluation-results.csv", "text/csv",
        key="rag-eval-csv-download",
    )

    with st.expander("🕘 Lịch sử evaluation đã lưu", expanded=False):
        saved_runs = [
            item for item in db.list_evaluation_runs(30)
            if item.get("knowledge_base_id") == kb_id
        ]
        if not saved_runs:
            st.info("Chưa có run evaluation đã lưu cho kho này.")
        else:
            st.dataframe(
                [
                    {
                        "Thời gian": item["created_at"],
                        "Dataset": item["dataset_hash"][:10],
                        "Pipeline": ", ".join(item.get("config", {}).get("pipelines", [])),
                        "Case": item.get("config", {}).get("case_limit", 0),
                        "Kết quả tổng hợp": json.dumps(item.get("summary", []), ensure_ascii=False),
                    }
                    for item in saved_runs
                ],
                use_container_width=True,
                hide_index=True,
            )
            latest = saved_runs[0]
            st.download_button(
                "Tải kết quả JSON của run gần nhất",
                json.dumps(latest.get("results", []), ensure_ascii=False, indent=2),
                f"rag-evaluation-{latest['id'][:8]}.json",
                "application/json",
                key=f"rag-eval-history-download-{latest['id']}",
            )

    for row in rows:
        case = row.get("case", {})
        with st.expander(f"{row.get('pipeline')} · {case.get('id')} · {case.get('question')}"):
            if row.get("error"):
                st.error(row["error"])
            st.markdown("**Câu trả lời**")
            st.write(row.get("answer") or "(trống)")
            st.markdown("**Đáp án tham chiếu · nguồn kỳ vọng**")
            st.write(case.get("reference_answer") or "(không có)")
            st.write(case.get("expected_sources", []))
            st.markdown("**Nguồn truy xuất**")
            for source in row.get("sources", []):
                chunk = source.get("chunk_index")
                chunk_label = int(chunk) + 1 if chunk is not None else "?"
                st.write(
                    f"{source.get('ref')} · {source.get('file_name')} · "
                    f"trang {source.get('page_number') or '?'} · chunk {chunk_label}"
                )
            st.json(row.get("metrics", {}))
            diagnostics = row.get("diagnostics", {})
            if row.get("pipeline") == "Multi-agent RAG":
                with st.expander("Agent routing và kết quả chuyên viên", expanded=False):
                    st.write("Supervisor:", diagnostics.get("supervisor", {}))
                    for agent in diagnostics.get("agents", []):
                        st.markdown(f"**{agent.get('agent_name', agent.get('agent_id'))}**")
                        st.write("Phát hiện:", agent.get("findings", []))
                        st.write("Chưa đủ dữ kiện:", agent.get("unknowns", []))
                    st.write("Critic:", diagnostics.get("critic", {}))
                    st.write("Citation validation:", diagnostics.get("citation_validation", {}))
            if row.get("judge"):
                st.markdown("**LLM Judge · điểm tham khảo**")
                st.json(row["judge"])
            meter = diagnostics.get("request_meter", {})
            judge_meter = row.get("judge_meter", {})
            st.caption(
                f"{diagnostics.get('latency_ms', 0)} ms · "
                f"{int(meter.get('api_calls', 0)) + int(judge_meter.get('api_calls', 0))} API calls · "
                f"{int(meter.get('total_tokens', 0)) + int(judge_meter.get('total_tokens', 0))} tokens"
            )
