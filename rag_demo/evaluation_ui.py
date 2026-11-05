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
) -> None:
    st.title("📊 RAG Evaluation & Observability")
    st.caption(
        "Chạy cùng một bộ câu hỏi qua RAG cơ bản và nâng cao, đo retrieval, citations, abstention, "
        "độ trễ và quota. Đây là benchmark nhỏ để demo, không thay thế tập đánh giá đại diện."
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
        '"answerable":false,"category":"abstention"}\n'
    )
    with st.expander("Dataset: schema và tải mẫu", expanded=False):
        st.markdown(
            "Mỗi dòng JSONL có `id`, `question`, `reference_answer`, `expected_sources` (tên file) và "
            "`answerable` (true/false); `category` là tùy chọn. Nên có câu trả lời được/không được, "
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
            "Pipeline", ["RAG cơ bản", "RAG nâng cao"],
            default=["RAG cơ bản", "RAG nâng cao"], key="rag-eval-pipelines",
        )
    with col_b:
        limit = st.number_input(
            "Số case tối đa", min_value=1, max_value=len(cases),
            value=min(6, len(cases)), step=1, key="rag-eval-case-limit",
        )
    with col_c:
        use_judge = st.checkbox(
            "Bật LLM Judge", value=False, key="rag-eval-judge",
            help="Thêm một generation call cho mỗi kết quả.",
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
    st.caption(
        f"Ước tính tối đa: khoảng {generation_estimate} generation call, {embedding_estimate} embedding batch; "
        "không tính retry do lỗi mạng/quota."
    )
    run_signature = hashlib.sha256(
        json.dumps(
            {
                "metrics_version": 2,
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
                    else:
                        result = advanced_runner(
                            kb_id, api_key, case["question"], [], flags=feature_flags,
                            top_k=top_k, candidate_k=candidate_k,
                            context_token_budget=context_token_budget,
                            embedding_rpm=embedding_rpm, generation_rpm=generation_rpm,
                            use_langgraph=use_langgraph, progress=case_progress,
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
                "sources": [
                    {key: source.get(key) for key in ("ref", "file_name", "page_number", "chunk_index", "chunk_id")}
                    for source in row.get("sources", [])
                ],
                "metrics": row.get("metrics"), "judge": row.get("judge"),
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
                    "citation_check_passed", "retrieved_files",
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
            if row.get("judge"):
                st.markdown("**LLM Judge · điểm tham khảo**")
                st.json(row["judge"])
            diagnostics = row.get("diagnostics", {})
            meter = diagnostics.get("request_meter", {})
            judge_meter = row.get("judge_meter", {})
            st.caption(
                f"{diagnostics.get('latency_ms', 0)} ms · "
                f"{int(meter.get('api_calls', 0)) + int(judge_meter.get('api_calls', 0))} API calls · "
                f"{int(meter.get('total_tokens', 0)) + int(judge_meter.get('total_tokens', 0))} tokens"
            )
