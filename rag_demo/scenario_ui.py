from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from . import db
from .config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_RPM_LIMIT,
    GENERATION_RPM_LIMIT,
    MAX_UPLOAD_MB,
    SUPPORTED_EXTENSIONS,
)
from .multi_agent import AGENT_REGISTRY, run_multi_agent_pipeline
from .quota import GeminiRequestError
from .rag import ingest_upload
from .scenarios import SCENARIO_FIELD_TYPES, demo_cv_scenario, validate_scenario


def _doc_labels(documents: list[dict[str, Any]]) -> dict[str, str]:
    counts: dict[str, int] = {}
    for item in documents:
        name = str(item.get("file_name", "Tài liệu")).casefold().strip()
        counts[name] = counts.get(name, 0) + 1
    labels = {}
    for item in documents:
        document_id = str(item["id"])
        name = str(item.get("file_name", "Tài liệu"))
        labels[document_id] = (
            f"{name} · {document_id[:7]}" if counts.get(name.casefold().strip(), 0) > 1 else name
        )
    return labels


def _create_demo_template() -> None:
    template = demo_cv_scenario()
    scenario_id = db.save_scenario(
        name=template["name"], description=template["description"],
        steps=template["steps"], agent_ids=template["agent_ids"],
    )
    st.session_state["scenario-pending-run-id"] = scenario_id
    st.session_state["scenario-just-created"] = template["name"]


def _render_langchain_and_agent_overview() -> None:
    st.subheader("LangChain trong project này")
    st.markdown(
        "LangChain là bộ thư viện xây ứng dụng với mô hình ngôn ngữ: chuẩn hóa tài liệu, splitter, "
        "vector store, retriever và các giao diện model. Project dùng `Document`, "
        "`RecursiveCharacterTextSplitter` và `langchain-chroma`; embedding/generation Gemini được "
        "adapter nội bộ gọi trực tiếp để kiểm soát quota, retry và đo request. Phần hỏi đáp chính không "
        "dùng một chain LangChain dựng sẵn; app điều phối rõ ràng bằng Python và LangGraph."
    )
    st.markdown(
        "**Phân biệt các lớp:** LangChain cung cấp building blocks; Chroma lưu/tìm vector; "
        "LangGraph giữ state và chuyển bước; Gemini sinh nội dung; SQLite lưu trạng thái sản phẩm và lịch sử."
    )
    st.subheader("Agentic design patterns đang demo")
    st.markdown(
        "1. **Supervisor–worker:** Supervisor chọn chuyên viên từ danh sách cho phép, dựa trên yêu cầu "
        "và loại bằng chứng.\n"
        "2. **Specialists:** các agent có mục tiêu riêng (trích xuất bằng chứng, đọc bảng, đối chiếu hồ sơ, "
        "so yêu cầu, kiểm tra rủi ro). Mọi agent dùng chung một context truy xuất.\n"
        "3. **Synthesis:** một agent tổng hợp báo cáo, trả lời và gợi ý tiếp theo.\n"
        "4. **Critic / verifier:** kiểm tra câu trả lời với nguồn gốc, sửa câu trả lời thận trọng khi cần, "
        "sau đó app kiểm tra citation ID xác định.\n"
        "5. **Bounded workflow:** LangGraph giới hạn số specialist, số node và lượt gọi; worker chạy tuần tự "
        "để tuân thủ RPM, tránh burst và làm log dễ đọc. Retrieval/embedding chỉ chạy một lần."
    )
    st.info(
        "Một lượt mặc định gồm 1 embedding batch + 1 supervisor + 2 specialist + 1 synthesis + "
        "1 critic (nếu bật). Giảm số chuyên viên hoặc tắt Critic để hạ token, request và độ trễ. "
        "Đây là workflow multi-agent tường minh, không phải mỗi node LangGraph đều là agent."
    )


def _render_builder() -> None:
    scenarios = db.list_scenarios()
    ids = [item["id"] for item in scenarios]
    edit_options = ["__new__", *ids]
    edit_labels = {item["id"]: f"{item['name']} · v{item['version']}" for item in scenarios}
    edit_id = st.selectbox(
        "Kịch bản cần chỉnh sửa",
        edit_options,
        format_func=lambda value: "＋ Tạo kịch bản mới" if value == "__new__" else edit_labels.get(value, value),
        key="scenario-builder-selected-id",
    )
    current = next((item for item in scenarios if item["id"] == edit_id), None)
    template = demo_cv_scenario() if current is None else current
    identity = edit_id
    steps_default = template.get("steps", [])
    step_count = st.number_input(
        "Số bước nhập liệu",
        min_value=1,
        max_value=10,
        value=min(10, max(1, len(steps_default) or 3)),
        step=1,
        key=f"scenario-builder-count-{identity}",
        help="Mỗi bước có thể là câu hỏi ngắn, câu hỏi dài, lựa chọn hoặc bước chọn file trong kho.",
    )

    with st.form(f"scenario-builder-form-{identity}"):
        name = st.text_input("Tên kịch bản", value=template.get("name", ""), key=f"scenario-name-{identity}")
        description = st.text_area(
            "Mục tiêu / sản phẩm đầu ra",
            value=template.get("description", ""),
            key=f"scenario-description-{identity}",
            height=90,
            help="Mô tả chatbot cần tạo ra kết quả gì.",
        )
        steps: list[dict[str, Any]] = []
        for index in range(int(step_count)):
            old = steps_default[index] if index < len(steps_default) else {}
            with st.expander(f"Bước {index + 1}", expanded=index == 0):
                label = st.text_input(
                    "Câu hỏi / nhãn hiển thị",
                    value=old.get("label", f"Thông tin bước {index + 1}"),
                    key=f"scenario-step-label-{identity}-{index}",
                )
                old_type = old.get("type", "text")
                field_type = st.selectbox(
                    "Loại trường",
                    list(SCENARIO_FIELD_TYPES),
                    index=list(SCENARIO_FIELD_TYPES).index(old_type) if old_type in SCENARIO_FIELD_TYPES else 0,
                    format_func=lambda value: SCENARIO_FIELD_TYPES[value],
                    key=f"scenario-step-type-{identity}-{index}",
                )
                help_text = st.text_area(
                    "Hướng dẫn cho người dùng",
                    value=old.get("help", ""),
                    key=f"scenario-step-help-{identity}-{index}",
                    height=65,
                )
                options_text = st.text_area(
                    "Các phương án (mỗi dòng một lựa chọn)",
                    value="\n".join(old.get("options", [])),
                    key=f"scenario-step-options-{identity}-{index}",
                    disabled=field_type != "choice",
                    height=75,
                )
                required = st.checkbox(
                    "Bắt buộc nhập",
                    value=bool(old.get("required", True)),
                    key=f"scenario-step-required-{identity}-{index}",
                )
            steps.append(
                {
                    "label": label,
                    "type": field_type,
                    "help": help_text,
                    "options": [line.strip() for line in options_text.splitlines() if line.strip()],
                    "required": required,
                }
            )

        st.markdown("**Nhóm chuyên viên được Supervisor phép chọn**")
        old_agent_ids = set(template.get("agent_ids", []))
        agent_ids = [
            agent_id
            for agent_id, agent in AGENT_REGISTRY.items()
            if st.checkbox(
                f"{agent['name']} · {agent['focus']}",
                value=agent_id in old_agent_ids or not current and agent_id in ("evidence_analyst", "requirements_analyst"),
                key=f"scenario-agent-{identity}-{agent_id}",
            )
        ]
        is_active = st.checkbox(
            "Cho phép người dùng chọn kịch bản này",
            value=bool(template.get("is_active", True)),
            key=f"scenario-active-{identity}",
        )
        submitted = st.form_submit_button("Lưu kịch bản", type="primary", use_container_width=True)

    if submitted:
        try:
            cleaned_steps = validate_scenario(name, description, steps, agent_ids)
            saved_id = db.save_scenario(
                scenario_id=edit_id if current else None,
                name=name,
                description=description,
                steps=cleaned_steps,
                agent_ids=agent_ids,
                is_active=is_active,
            )
            if is_active:
                st.session_state["scenario-pending-run-id"] = saved_id
            st.success("Đã lưu kịch bản và tăng phiên bản cấu hình.")
            st.rerun()
        except ValueError as error:
            st.error(str(error))

    if current:
        st.caption(
            f"Phiên bản {current['version']} · cập nhật {current['updated_at']} · "
            "các lượt chạy cũ giữ nguyên bản snapshot kịch bản đã dùng."
        )
        if st.button("🗑️ Xóa kịch bản", key=f"delete-scenario-{current['id']}"):
            db.delete_scenario(current["id"])
            st.rerun()


def _render_run_wizard(
    *,
    scenario: dict[str, Any],
    kb_id: str,
    kb_name: str,
    api_key: str,
    embedding_rpm: int,
    generation_rpm: int,
    top_k: int,
    candidate_k: int,
    context_token_budget: int,
) -> None:
    steps = scenario.get("steps", [])
    if not steps:
        st.warning("Kịch bản này chưa có bước nhập liệu.")
        return
    active_id_key = "scenario-active-wizard-id"
    step_key = "scenario-active-wizard-step"
    wizard_identity = f"{scenario['id']}:v{scenario['version']}"
    if st.session_state.get(active_id_key) != wizard_identity:
        st.session_state[active_id_key] = wizard_identity
        st.session_state[step_key] = 0
    step_index = max(0, min(int(st.session_state.get(step_key, 0)), len(steps)))
    st.progress(min(1.0, step_index / len(steps)), text=f"Bước {min(step_index + 1, len(steps))} / {len(steps)}")

    if step_index < len(steps):
        step = steps[step_index]
        st.subheader(f"Bước {step_index + 1} · {step['label']}")
        if step.get("help"):
            st.caption(step["help"])
        field_key = f"scenario-field-{scenario['id']}-v{scenario['version']}-{step['id']}"
        ready_docs = [
            item for item in db.list_documents(kb_id)
            if item.get("status") == "Sẵn sàng"
        ]
        labels = _doc_labels(ready_docs)
        if step["type"] == "text":
            value = st.text_input(step["label"], key=field_key)
        elif step["type"] == "textarea":
            value = st.text_area(step["label"], key=field_key, height=150)
        elif step["type"] == "choice":
            options = step.get("options", [])
            choice_options = options if step.get("required", True) else ["(Bỏ qua)", *options]
            value = st.selectbox(step["label"], choice_options, key=field_key) if options else ""
            if value == "(Bỏ qua)":
                value = ""
        else:
            if not ready_docs:
                st.info("Kho chưa có tài liệu đã lập chỉ mục. Thêm file ngay tại đây hoặc qua mục Chat.")
            value = st.multiselect(
                step["label"],
                options=[item["id"] for item in ready_docs],
                format_func=lambda doc_id: labels.get(doc_id, doc_id),
                key=field_key,
                help="Chỉ các tài liệu đã chọn được dùng làm nguồn cho lượt chạy này.",
            )
            with st.expander(f"＋ Thêm file vào kho {kb_name}", expanded=not ready_docs):
                uploads = st.file_uploader(
                    f"Chọn file (tối đa {MAX_UPLOAD_MB} MB mỗi file)",
                    type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
                    accept_multiple_files=True,
                    key=f"scenario-uploader-{scenario['id']}-{kb_id}",
                )
                if st.button(
                    "Nhập file và lập chỉ mục",
                    key=f"scenario-ingest-{scenario['id']}-{kb_id}",
                    disabled=not uploads or not api_key,
                    type="primary",
                ):
                    status = st.status("Đang nhập file vào kho…", expanded=True)
                    try:
                        for file_index, upload in enumerate(uploads, start=1):
                            status.write(f"{file_index}/{len(uploads)} · {upload.name}")
                            _document_id, message = ingest_upload(
                                kb_id,
                                upload.name,
                                upload.getvalue(),
                                api_key,
                                chunk_size=900,
                                chunk_overlap=150,
                                chunk_strategy="parent_child",
                                embedding_rpm=embedding_rpm,
                                embedding_batch_size=EMBEDDING_BATCH_SIZE,
                                progress=status.write,
                            )
                            status.write(message)
                        status.update(label="Đã nhập file vào kho.", state="complete")
                        st.rerun()
                    except Exception as error:
                        status.update(label="Không nhập được file.", state="error")
                        st.error(str(error)[:700])
        required = bool(step.get("required", True))
        has_value = bool(value) if isinstance(value, list) else bool(str(value).strip())
        left, right = st.columns(2)
        with left:
            if step_index > 0 and st.button("← Quay lại", key=f"scenario-back-{scenario['id']}-{step_index}"):
                st.session_state[step_key] = step_index - 1
                st.rerun()
        with right:
            if st.button(
                "Tiếp theo →" if step_index < len(steps) - 1 else "Xem lại thông tin →",
                key=f"scenario-next-{scenario['id']}-{step_index}",
                type="primary",
                disabled=required and not has_value,
                use_container_width=True,
            ):
                st.session_state[step_key] = step_index + 1
                st.rerun()
        return

    st.subheader("Kiểm tra thông tin trước khi chạy")
    inputs: dict[str, Any] = {}
    selected_document_ids: list[str] = []
    context_lines: list[str] = []
    question_parts = [scenario.get("name", "Phân tích tài liệu"), scenario.get("description", "")]
    for step in steps:
        value = st.session_state.get(
            f"scenario-field-{scenario['id']}-v{scenario['version']}-{step['id']}", ""
        )
        if step.get("type") == "choice" and value == "(Bỏ qua)":
            value = ""
        inputs[step["id"]] = value
        if step["type"] == "documents":
            selected_document_ids = [str(value) for value in value] if isinstance(value, list) else []
            doc_names = db.get_document_names(selected_document_ids)
            st.write(f"**{step['label']}:** " + (", ".join(doc_names.get(item, item) for item in selected_document_ids) or "(chưa chọn)"))
        else:
            pretty_value = str(value).strip()
            st.write(f"**{step['label']}:** {pretty_value or '(để trống)'}")
            if pretty_value:
                context_lines.append(f"{step['label']}: {pretty_value}")
                question_parts.append(f"{step['label']}: {pretty_value}")

    run_flags = {
        "parent_child": True,
        "hybrid_search": True,
        "reranker": True,
        "context_compression": True,
    }
    with st.expander("⚙️ Chi tiết chạy / giới hạn chi phí", expanded=False):
        max_agents = st.number_input(
            "Số chuyên viên tối đa", min_value=1, max_value=3, value=min(2, len(scenario.get("agent_ids", [])) or 1),
            key=f"scenario-agent-count-{scenario['id']}",
        )
        include_critic = st.checkbox(
            "Bật Critic / kiểm tra citation (thêm 1 lượt Gemini)", value=True,
            key=f"scenario-critic-{scenario['id']}",
        )
        st.caption(
            f"Ước tính tối đa: 1 embedding batch + {1 + int(max_agents) + 1 + int(include_critic)} "
            f"generation calls; context tối đa khoảng {int(context_token_budget) * (int(max_agents) + 1 + int(include_critic)):,} "
            "token được đưa vào các lượt specialist/synthesis/critic cộng dồn. Hybrid Search, FlashRank và context compression chạy local."
        )
        use_hybrid = st.checkbox("Hybrid Search + BM25", value=True, key=f"scenario-hybrid-{scenario['id']}")
        use_reranker = st.checkbox("Reranker local", value=True, key=f"scenario-reranker-{scenario['id']}")
        use_compression = st.checkbox("Nén context local", value=True, key=f"scenario-compress-{scenario['id']}")
        run_flags.update({
            "hybrid_search": use_hybrid,
            "reranker": use_reranker,
            "context_compression": use_compression,
        })
    requires_document_selection = any(
        step["type"] == "documents" and bool(step.get("required", True)) for step in steps
    )
    if requires_document_selection and not selected_document_ids:
        st.warning("Kịch bản yêu cầu chọn ít nhất một file.")
    st.caption(
        f"Phạm vi dữ liệu: {len(selected_document_ids)} file đã chọn" if selected_document_ids
        else "Phạm vi dữ liệu: toàn bộ tài liệu đã lập chỉ mục trong kho"
    )
    run_clicked = st.button(
        "▶ Chạy kịch bản với nhóm AI",
        type="primary",
        key=f"scenario-run-{scenario['id']}-{kb_id}",
        disabled=not api_key or (requires_document_selection and not selected_document_ids),
        use_container_width=True,
    )
    if run_clicked:
        task_context = "\n".join(context_lines)
        question = "\n".join(part for part in question_parts if part).strip()[:5000]
        snapshot = {
            "id": scenario["id"],
            "name": scenario["name"],
            "version": scenario["version"],
            "description": scenario.get("description", ""),
            "steps": steps,
            "agent_ids": scenario.get("agent_ids", []),
        }
        status = st.status("Nhóm agent đang xử lý…", expanded=True)
        try:
            result = run_multi_agent_pipeline(
                kb_id,
                api_key,
                question,
                [],
                flags=run_flags,
                top_k=top_k,
                candidate_k=candidate_k,
                context_token_budget=context_token_budget,
                embedding_rpm=embedding_rpm,
                generation_rpm=generation_rpm,
                max_agents=int(max_agents),
                include_critic=include_critic,
                allowed_agent_ids=scenario.get("agent_ids", []),
                document_ids=selected_document_ids or None,
                task_context=task_context,
                progress=status.write,
            )
            db.save_scenario_run(
                scenario_id=scenario["id"], scenario_name=scenario["name"],
                knowledge_base_id=kb_id, scenario_snapshot=snapshot, inputs=inputs,
                document_ids=selected_document_ids, status="completed",
                result={key: result.get(key) for key in ("answer", "suggestions", "sources")},
                diagnostics=result.get("diagnostics", {}),
            )
            st.session_state["scenario-last-result"] = result
            st.session_state["scenario-last-result-kb"] = kb_id
            st.session_state[step_key] = 0
            status.update(label="Hoàn tất kịch bản.", state="complete")
            st.rerun()
        except Exception as error:
            error_text = str(error)[:800]
            db.save_scenario_run(
                scenario_id=scenario["id"], scenario_name=scenario["name"],
                knowledge_base_id=kb_id, scenario_snapshot=snapshot, inputs=inputs,
                document_ids=selected_document_ids, status="failed",
                result={"error": error_text}, diagnostics={},
            )
            status.update(label="Kịch bản dừng do lỗi.", state="error")
            st.error(error_text)
    if st.button("← Bắt đầu lại", key=f"scenario-reset-{scenario['id']}"):
        st.session_state[step_key] = 0
        st.rerun()


def _render_run_result(result: dict[str, Any], kb_id: str) -> None:
    st.subheader("Kết quả kịch bản")
    st.markdown(result.get("answer") or "(Chưa có câu trả lời)")
    suggestions = result.get("suggestions", [])
    if suggestions:
        st.markdown("**Câu hỏi gợi ý tiếp theo**")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")
    sources = result.get("sources", [])
    with st.expander(f"Nguồn đã dùng · {len(sources)}", expanded=True):
        for source in sources:
            st.markdown(
                f"**[{source.get('ref')}] {source.get('file_name')} · "
                f"Trang {source.get('page_number') or '?'} · Chunk {int(source.get('chunk_index', 0)) + 1}**"
            )
            st.code(str(source.get("context_used", "")), language=None)
    diagnostics = result.get("diagnostics", {})
    with st.expander("🔎 Quan sát từng agent và chi phí", expanded=False):
        st.markdown(f"**Workflow:** {diagnostics.get('pipeline')}")
        route = diagnostics.get("supervisor", {})
        st.write("Lý do điều phối:", route.get("reason", ""))
        st.write("Agent được chọn:", ", ".join(route.get("selected_agents", [])))
        for item in diagnostics.get("agents", []):
            with st.expander(item.get("agent_name", item.get("agent_id", "Agent")), expanded=False):
                st.write("Phát hiện:", item.get("findings", []))
                st.write("Chưa đủ dữ kiện:", item.get("unknowns", []))
        st.write("Critic:", diagnostics.get("critic", {}))
        st.write("Kiểm tra citation:", diagnostics.get("citation_validation", {}))
        meter = diagnostics.get("request_meter", {})
        cols = st.columns(4)
        cols[0].metric("Thời gian", f"{diagnostics.get('latency_ms', 0)} ms")
        cols[1].metric("API calls", meter.get("api_calls", 0))
        cols[2].metric("Input token", meter.get("input_tokens", 0))
        cols[3].metric("Output token", meter.get("output_tokens", 0))
        st.caption(
            f"Ước tính context được lặp qua agent: "
            f"~{diagnostics.get('estimated_repeated_context_tokens', 0):,} token tổng hợp; "
            "input/output thực tế ưu tiên usage metadata của Gemini."
        )
        st.caption("Request labels: " + " → ".join(meter.get("labels", [])))


def render_scenario_workbench(
    *,
    kb_id: str,
    kb_name: str,
    api_key: str,
    embedding_rpm: int = EMBEDDING_RPM_LIMIT,
    generation_rpm: int = GENERATION_RPM_LIMIT,
    top_k: int = 6,
    candidate_k: int = 24,
    context_token_budget: int = 4000,
) -> None:
    pending_scenario_id = st.session_state.pop("scenario-pending-run-id", None)
    if pending_scenario_id:
        # Apply the selection before the run-tab selectbox is instantiated.
        st.session_state["scenario-selected-run-id"] = pending_scenario_id
    st.title("🧭 Kịch bản sản phẩm & Multi-agent")
    st.caption(
        "Tạo luồng nhiều bước cho người dùng, giới hạn tài liệu đưa vào xử lý, rồi quan sát Supervisor, "
        "chuyên viên, bước tổng hợp, Critic, nguồn và chi phí."
    )
    just_created = st.session_state.pop("scenario-just-created", None)
    if just_created:
        st.success(f"Đã tạo kịch bản mẫu: {just_created}")
    with st.expander("🧠 LangChain, LangGraph và Agentic Design Patterns", expanded=False):
        _render_langchain_and_agent_overview()

    tab_run, tab_manage, tab_history = st.tabs(["▶ Chọn kịch bản", "⚙️ Tạo / sửa kịch bản", "🕘 Lịch sử chạy"])
    scenarios = db.list_scenarios(active_only=True)
    with tab_run:
        if not scenarios:
            st.info("Chưa có kịch bản đang bật. Tạo kịch bản mới hoặc thêm mẫu CV từ tab quản lý.")
        else:
            scenario_ids = [item["id"] for item in scenarios]
            selected_id = st.selectbox(
                "Chọn việc cần làm",
                scenario_ids,
                index=scenario_ids.index(st.session_state["scenario-selected-run-id"])
                if st.session_state.get("scenario-selected-run-id") in scenario_ids else 0,
                format_func=lambda value: next(item["name"] for item in scenarios if item["id"] == value),
                key="scenario-selected-run-id",
            )
            scenario = next(item for item in scenarios if item["id"] == selected_id)
            st.markdown(f"### {scenario['name']}")
            st.write(scenario.get("description", ""))
            st.caption(
                f"Kịch bản v{scenario['version']} · kho {kb_name} · "
                f"{len(scenario.get('steps', []))} bước · tối đa 3 agent"
            )
            _render_run_wizard(
                scenario=scenario, kb_id=kb_id, kb_name=kb_name, api_key=api_key,
                embedding_rpm=embedding_rpm, generation_rpm=generation_rpm,
                top_k=top_k, candidate_k=candidate_k, context_token_budget=context_token_budget,
            )
            last_result = st.session_state.get("scenario-last-result")
            if last_result and st.session_state.get("scenario-last-result-kb") == kb_id:
                st.divider()
                _render_run_result(last_result, kb_id)

    with tab_manage:
        if st.button("＋ Tạo bản sao kịch bản mẫu CV", key="scenario-add-cv-template"):
            _create_demo_template()
            st.rerun()
        _render_builder()
        st.caption(
            "Builder dùng các bước nhập liệu có cấu trúc; không cho cấu hình prompt thực thi tự do. "
            "Mỗi lượt chạy lưu snapshot tên, phiên bản, input, tài liệu, kết quả và diagnostics để mentor xem lại."
        )

    with tab_history:
        history = [item for item in db.list_scenario_runs(50) if item.get("knowledge_base_id") == kb_id]
        if not history:
            st.info("Chưa có lượt chạy kịch bản trong kho này.")
        else:
            st.dataframe(
                [
                    {
                        "Thời gian": item["created_at"],
                        "Kịch bản": item["scenario_name"],
                        "Trạng thái": item["status"],
                        "File": len(item.get("document_ids", [])),
                        "Agent": len(item.get("diagnostics", {}).get("agents", [])),
                        "API calls": item.get("diagnostics", {}).get("request_meter", {}).get("api_calls", 0),
                        "Token": item.get("diagnostics", {}).get("request_meter", {}).get("total_tokens", 0),
                    }
                    for item in history
                ],
                use_container_width=True,
                hide_index=True,
            )
            for run in history:
                with st.expander(f"{run['created_at']} · {run['scenario_name']} · {run['status']}"):
                    st.write("Snapshot:", run.get("scenario_snapshot", {}))
                    st.write("Inputs:", run.get("inputs", {}))
                    result = run.get("result", {})
                    if result.get("answer"):
                        st.markdown(result["answer"])
                        st.json(result.get("sources", []))
                    elif result.get("error"):
                        st.error(result["error"])
                    st.json(run.get("diagnostics", {}))
