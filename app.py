from __future__ import annotations

import importlib
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from rag_demo.config import ROOT_DIR

load_dotenv(ROOT_DIR / ".env")

from rag_demo.observability import configure_observability

OBSERVABILITY_STATUS = configure_observability()

from rag_demo import db, lab as rag_lab
from rag_demo.observability import trace_span

# Streamlit can rerun app.py while retaining imported package modules in memory.
# Reload the lab helpers when the running process still has the older module API.
if not all(hasattr(rag_lab, name) for name in ("query_strings", "embed_queries_cached")):
    rag_lab = importlib.reload(rag_lab)

from rag_demo.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_RPM_LIMIT,
    GENERATION_RPM_LIMIT,
    MAX_UPLOAD_MB,
    ROOT_DIR,
    SUPPORTED_EXTENSIONS,
)
from rag_demo.quota import GeminiRequestError
from rag_demo.quota import RequestMeter
from rag_demo.rag import (
    answer_question,
    delete_document,
    delete_knowledge_base,
    ingest_upload,
    is_document_count_question,
    is_document_list_question,
    retry_document_index,
    should_use_all_documents,
)
from rag_demo.advanced_rag import run_advanced_pipeline
from rag_demo.evaluation_ui import render_evaluation_workbench
from rag_demo.multi_agent import run_multi_agent_pipeline
from rag_demo.scenario_ui import render_scenario_workbench
from rag_demo.voice import transcribe_audio

load_dotenv(ROOT_DIR / ".env")
st.set_page_config(page_title="Chat với kho tài liệu", page_icon="💬", layout="wide")
db.init_db()

_TECHNIQUE_LABELS = {
    "parent_child": "Parent-Child",
    "hybrid_search": "Hybrid Search + BM25",
    "reranker": "FlashRank Reranker (local)",
    "query_rewrite": "Query Rewrite",
    "multi_query": "Multi-query",
    "hyde": "HyDE",
    "context_compression": "Context Compression (local)",
    "answer_grading": "Answer Grading",
    "citation_verification": "Citation Verification",
    "corrective_retrieval": "Corrective Retrieval (tối đa 1 lần tìm lại)",
    "langgraph": "LangGraph",
}
if not st.session_state.get("_incomplete_docs_recovered"):
    db.recover_incomplete_documents()
    st.session_state["_incomplete_docs_recovered"] = True


def _short_error(value: str | None) -> str:
    return (value or "").replace("\n", " ")[:650]


def _render_speech_button(text: str, key: str) -> None:
    safe_text = json.dumps(str(text), ensure_ascii=False).replace("</", "<\\/")
    safe_key = re.sub(r"[^a-zA-Z0-9_-]", "", key)
    components.html(
        f"""<button id="speak-{safe_key}" style="border:1px solid #c9cdd3;border-radius:16px;
        padding:5px 12px;background:#fff;cursor:pointer;font:13px sans-serif">🔊 Đọc câu trả lời</button>
        <script>
        const speakButton = document.getElementById("speak-{safe_key}");
        speakButton.addEventListener("click", () => {{
          if (!("speechSynthesis" in window)) {{
            speakButton.textContent = "Trình duyệt không hỗ trợ đọc văn bản";
            return;
          }}
          window.speechSynthesis.cancel();
          const utterance = new SpeechSynthesisUtterance({safe_text});
          utterance.lang = "vi-VN";
          utterance.rate = 1;
          window.speechSynthesis.speak(utterance);
        }});
        </script>""",
        height=42,
        scrolling=False,
    )


def _merge_request_meter(primary: dict, additional: dict) -> dict:
    merged = dict(primary or {})
    for key in ("api_calls", "embedding_calls", "generation_calls", "input_tokens", "output_tokens", "estimated_token_calls"):
        merged[key] = int(merged.get(key, 0)) + int(additional.get(key, 0))
    merged["total_tokens"] = int(merged.get("input_tokens", 0)) + int(merged.get("output_tokens", 0))
    merged["labels"] = list(additional.get("labels", [])) + list(merged.get("labels", []))
    merged["token_note"] = additional.get("token_note") or merged.get("token_note", "")
    return merged


def _location(page_number: int | None) -> str:
    return f"Trang {page_number}" if page_number else "Không có số trang"


_LAB_STEP_ORDER = {
    "chunking": 0,
    "planning": 1,
    "retrieval": 2,
    "reranker": 3,
    "context": 4,
    "answer": 5,
    "audit": 6,
    "complete": 7,
}


def _parent_map(parents: list[dict] | None) -> dict[str, dict]:
    return {
        str(item.get("parent_chunk_id")): item
        for item in (parents or [])
        if item.get("parent_chunk_id")
    }


def _index_from_internal_id(value: str | None, *, parent: bool = False) -> int | None:
    suffix = str(value or "").rsplit(":", 1)[-1]
    match = re.fullmatch(r"p?(\d+)", suffix)
    if not match or (parent and not suffix.startswith("p")):
        return None
    return int(match.group(1))


def _lab_chunk_label(chunk: dict, parents: dict[str, dict] | None = None) -> str:
    row = chunk.get("row", chunk)
    page = _location(row.get("page_number"))
    if "parent_index" in row and "chunk_index" not in row:
        return f"Parent {int(row.get('parent_index', 0)) + 1} · {page}"

    chunk_number = int(row.get("chunk_index", 0)) + 1
    parent_id = str(row.get("parent_chunk_id") or chunk.get("parent_id") or "")
    prefix = "Child chunk" if parent_id else "Chunk"
    label = f"{prefix} {chunk_number} · {page}"
    if row.get("content_type"):
        label += f" · {row['content_type']}"
    if row.get("file_name") or chunk.get("file_name"):
        label += f" · {row.get('file_name') or chunk.get('file_name')}"
    if parent_id:
        parent = (parents or {}).get(parent_id)
        if parent:
            label += (
                f" · thuộc Parent {int(parent.get('parent_index', 0)) + 1}"
                f" ({_location(parent.get('page_number'))})"
            )
        elif row.get("parent_index") is not None:
            label += f" · thuộc Parent {int(row['parent_index']) + 1}"
        else:
            label += " · parent không tìm thấy"
    return label


def _lab_source_label(source: dict, parents: dict[str, dict] | None = None) -> str:
    chunk_index = source.get("chunk_index")
    if chunk_index is None:
        chunk_index = _index_from_internal_id(source.get("chunk_id"))
    chunk_label = f"Chunk {int(chunk_index) + 1}" if chunk_index is not None else "Chunk nguồn"
    label = f"{chunk_label} · {_location(source.get('page_number'))}"
    if source.get("content_type"):
        label += f" · {source['content_type']}"
    if source.get("file_name"):
        label += f" · {source['file_name']}"
    parent_id = str(source.get("parent_chunk_id") or "")
    parent = (parents or {}).get(parent_id)
    if parent:
        label += (
            f" · thuộc Parent {int(parent.get('parent_index', 0)) + 1}"
            f" ({_location(parent.get('page_number'))})"
        )
    elif source.get("parent_index") is not None:
        label += (
            f" · thuộc Parent {int(source['parent_index']) + 1}"
            f" ({_location(source.get('parent_page_number'))})"
        )
    elif source.get("parent_chunk_id"):
        parent_index = _index_from_internal_id(source.get("parent_chunk_id"), parent=True)
        if parent_index is not None:
            label += f" · thuộc Parent {parent_index + 1}"
    return label


def _source_display_label(source: dict, index: int) -> str:
    chunk_index = source.get("chunk_index")
    if chunk_index is None:
        chunk_index = _index_from_internal_id(source.get("chunk_id"))
    chunk = f"Chunk {int(chunk_index) + 1}" if chunk_index is not None else "Chunk nguồn"
    label = (
        f"[{source.get('ref', f'S{index + 1}')}] {source.get('file_name', 'Tài liệu')} · "
        f"{_location(source.get('page_number'))} · {chunk}"
    )
    if source.get("content_type"):
        label += f" · {source['content_type']}"
    if source.get("parent_index") is not None:
        parent_page = _location(source.get("parent_page_number"))
        label += f" · child của Parent {int(source['parent_index']) + 1} ({parent_page})"
    elif source.get("parent_chunk_id"):
        parent_index = _index_from_internal_id(source.get("parent_chunk_id"), parent=True)
        if parent_index is not None:
            label += f" · child của Parent {parent_index + 1}"
    if source.get("is_dense_anchor"):
        label += " · Dense gốc được giữ"
    return label


def _readable_chunk_ids(values: list[str]) -> list[str]:
    labels = []
    for value in values:
        suffix = str(value).rsplit(":", 1)[-1]
        match = re.fullmatch(r"(p?)(\d+)", suffix)
        if match:
            kind = "Parent" if match.group(1) else "Chunk"
            labels.append(f"{kind} {int(match.group(2)) + 1}")
        else:
            labels.append("Chunk nguồn")
    return labels


def _render_lab_audit(audit: dict) -> None:
    deterministic = audit.get("deterministic_citation_check", {})
    llm_audit = audit.get("llm_audit", {})
    if deterministic:
        st.markdown("**Kiểm tra mã trích dẫn tự động**")
        st.write("Kết quả:", "Đạt" if deterministic.get("passed") else "Cần xem lại")
        st.caption(f"Mã trích dẫn có trong câu trả lời: {deterministic.get('cited_refs', []) or 'Không có'}")
        st.caption(f"Mã nguồn hợp lệ của context: {deterministic.get('known_refs', []) or 'Không có'}")
        if deterministic.get("invalid_refs"):
            st.error(f"Mã trích dẫn không tồn tại: {deterministic['invalid_refs']}")
        if deterministic.get("claims_without_citation"):
            st.markdown("Các câu chưa gắn mã nguồn:")
            for claim in deterministic["claims_without_citation"]:
                st.markdown(f"- {claim}")
    if llm_audit:
        st.markdown("**Đánh giá bằng Gemini**")
        if llm_audit.get("grade") is not None:
            st.metric("Điểm hỗ trợ từ tài liệu", f"{llm_audit['grade']}/5")
            if llm_audit.get("grade_reason"):
                st.write(llm_audit["grade_reason"])
        if llm_audit.get("citation_support") is not None:
            st.write(
                "Nguồn có hỗ trợ câu trả lời:",
                "Có" if llm_audit["citation_support"] else "Chưa đủ hỗ trợ",
            )
        unsupported = llm_audit.get("unsupported_claims", [])
        issues = llm_audit.get("citation_issues", [])
        if unsupported:
            st.markdown("Các ý Gemini đánh dấu là chưa được nguồn hỗ trợ:")
            for claim in unsupported:
                st.markdown(f"- {claim}")
        if issues:
            st.markdown("Các vấn đề trích dẫn Gemini phát hiện:")
            for issue in issues:
                st.markdown(f"- {issue}")


def _render_lab_chunks(
    chunks: list[dict],
    max_visible: int,
    *,
    title: str,
    parents: list[dict] | None = None,
) -> None:
    st.markdown(f"**{title} · {len(chunks):,} chunk**")
    parent_lookup = _parent_map(parents)
    for chunk in chunks[:max_visible]:
        st.caption(_lab_chunk_label(chunk, parent_lookup))
        st.code(str(chunk.get("content", "")), language=None)
    if len(chunks) > max_visible:
        st.caption(f"Đang hiển thị {max_visible} chunk đầu tiên; tổng cộng {len(chunks)}.")


def _render_rag_explainer() -> None:
    with st.expander("📘 Giải thích thuật ngữ, chỉ số, thông số và công thức", expanded=False):
        st.markdown(
            """
### Tìm kiếm và xếp hạng

| Nhãn trên giao diện | Ý nghĩa |
|---|---|
| **Dense** | Tìm kiếm ngữ nghĩa: Gemini biến câu hỏi và mỗi chunk thành vector; Chroma tìm các vector gần nhau theo cosine distance. Khoảng cách thấp hơn nghĩa là gần hơn. |
| **Dense top** | Các chunk đứng đầu từ tìm kiếm vector cho **câu hỏi gốc**. Khi có nhiều query, Dense count là số chunk duy nhất gom được qua mọi query; danh sách top hiển thị riêng query gốc để dễ đọc. |
| **Distance** | Khoảng cách cosine do Chroma trả về trong chế độ tìm kiếm vector cơ bản; thấp hơn là gần hơn. Đây không phải xác suất. |
| **BM25 / BM25 top** | Tìm kiếm từ khóa cục bộ. BM25 count là số chunk duy nhất xuất hiện qua các danh sách BM25; BM25 top là các chunk khớp từ khóa tốt nhất với query gốc. Không cần gọi Gemini. |
| **Hợp nhất / fused** | Hợp các danh sách Dense và BM25, bỏ trùng, rồi xếp theo điểm Reciprocal Rank Fusion (RRF). Đây là số ứng viên được giữ lại sau giới hạn candidate. |
| **RRF score** | Trong chat nâng cao, đây là điểm cộng dồn có trọng số theo thứ hạng, không phải xác suất: `RRF(d) = Σ_q w_q / (60 + rank_q(d))`. Hạng bắt đầu từ 1; câu gốc có trọng số cao nhất, Rewrite/Multi-query thấp hơn, HyDE thấp và chỉ dùng Dense; BM25 có trọng số thấp hơn Dense. Phòng thử nghiệm giữ RRF không trọng số để xem riêng baseline từng kỹ thuật. |
| **Reranker score** | Điểm liên quan do FlashRank chấm lại giữa câu hỏi và từng chunk. Mặc định dùng `ms-marco-MultiBERT-L-12`, giới hạn độ dài đầu vào 256 token. Dùng để sắp xếp ứng viên; không phải xác suất đúng và không so trực tiếp với RRF score. |

Với vector `q` của câu hỏi và `x` của chunk, `cosine_similarity(q,x) = (q·x) / (||q|| × ||x||)` và `cosine_distance = 1 - cosine_similarity`. Dense search xếp khoảng cách thấp trước.

BM25 dùng cách chấm điểm BM25Okapi. Dạng tổng quát cho query `q` và tài liệu/chunk `d` là `Σ IDF(t) × f(t,d) × (k1+1) / (f(t,d) + k1 × (1-b + b×|d|/avgdl))`, với `f` là số lần xuất hiện, `|d|` độ dài chunk, `avgdl` độ dài trung bình. Tham số mặc định của BM25Okapi là `k1=1.5`, `b=0.75`, `epsilon=0.25`; IDF âm được nâng lên theo `epsilon × average_IDF`. App tách từ theo chữ/số Unicode, chuyển chữ thường.

RRF chỉ dùng thứ hạng, không cộng trực tiếp cosine distance với BM25 score vì hai loại điểm có thang đo khác nhau. Mỗi danh sách lấy tối đa `candidate_k` (mặc định 24); mã đảm bảo ít nhất 8 và nhiều nhất 50, đồng thời không vượt số chunk có trong kho. Pool ứng viên luôn giữ các chunk đầu từ Dense câu gốc; context cuối dành tối đa hai nhóm nguồn riêng cho Dense gốc và không nén các nguồn được giữ này. Pipeline hiển thị riêng top sau RRF, sau reranker và đúng nguồn gửi Gemini.

### Chunk, Parent-Child và context

- **Parser / phần trích xuất**: parser chuyên biệt lấy văn bản, bảng, trang và cấu trúc trước khi chia chunk. File scan/ảnh hoặc lỗi parser có thể chuyển OCR/Docling; số trang có thể không có với định dạng văn bản thuần.
- **Số phần trích xuất** là các đoạn nội dung cấu trúc do parser trả về (ví dụ đoạn văn, tiêu đề, bảng), chưa phải số chunk.
- **Recursive Character Chunking** thử tách theo thứ tự đoạn trống, dòng mới, dấu cách rồi ký tự; tiếp tục chia cho đến khi đoạn vừa `chunk_size`. Chunk overlap lặp lại một phần cuối chunk trước để giảm mất ngữ cảnh ở ranh giới.
- **Chunk size** tính theo ký tự, không phải token. Mặc định 900 ký tự; **overlap** mặc định 150. Ở Parent-Child, kích thước parent là `min(6000, max(3×chunk_size, chunk_size+overlap))`; parent overlap không vượt `parent_size/4`. Child được tìm kiếm, parent lớn hơn được đưa vào context. Bảng được lặp lại tiêu đề trong các chunk con.
- **Chunk N / Trang N** là số thứ tự bắt đầu từ 1 trong tài liệu và số trang parser đọc được. Một child hiển thị rõ “thuộc Parent N”; parent không phải một kết quả tìm kiếm riêng.
- **Parent fallback** là số ứng viên không tìm thấy parent tương ứng nên pipeline dùng chính child làm context.
- **Candidate K** là số kết quả giữ lại sau retrieval trước reranker (mặc định 24). **Số chunk truy xuất / top-k** là số nguồn tối đa đưa vào context (mặc định 6). Với Parent-Child, các child cùng parent được gộp để tránh gửi parent lặp lại.
- **Nén context** chạy cục bộ, không gọi Gemini. Đoạn văn được tách câu/dòng, chấm bằng số từ câu hỏi trùng trong từng đoạn, giữ đoạn điểm cao trước rồi khôi phục thứ tự gốc. Với bảng, giữ dòng tiêu đề và ưu tiên các hàng có từ khóa khớp. Nếu không tìm được đoạn/hàng khớp thì dùng cắt theo giới hạn ký tự. Vì vậy đây là chọn lọc trích xuất, không phải viết lại hay tóm tắt bằng mô hình.
- Ngân sách context mặc định 4.000 token; app dành khoảng `3 × ngân sách` ký tự và trừ khoảng đệm 300 ký tự giữa nguồn. **Ký tự trước → sau** đếm phần nội dung trích đoạn, không tính đầy đủ nhãn file/trang/trích dẫn. Token hiển thị ước lượng `ceil(số ký tự nội dung / 3)`, không phải tokenizer chính xác của Gemini nên prompt thực có thể cao hơn một ít.

### Các kỹ thuật và luồng làm việc

- **Cơ bản** dùng vector search trực tiếp; **Nâng cao** thêm các công tắc RAG; **So sánh** chạy cả hai pipeline cho cùng câu hỏi nên dùng nhiều API/token hơn. **Tự động** nhận diện một số câu hỏi rộng; **Hỏi đáp chi tiết** tập trung top-k; **Tổng hợp toàn kho** đọc riêng từng file theo phần tối đa khoảng 60.000 ký tự rồi tổng hợp. Câu hỏi toàn kho trong chế độ Tự động hoặc Tổng hợp toàn kho được chuyển sang luồng đầy đủ dù đang chọn Nâng cao.
- **Tiết kiệm API** giữ các bước local và tắt planner/audit gọi thêm Gemini; **Bật mọi kỹ thuật** bật planner và audit để demo. Các checkbox bên dưới preset cho phép chỉnh từng kỹ thuật.
- **Query Rewrite** làm câu hỏi độc lập/rõ hơn trước khi tìm kiếm; **Multi-query** tạo tối đa hai cách hỏi tương đương; **HyDE** tạo đoạn trả lời giả thuyết để lấy vector truy vấn. Query planner chỉ nhận câu hỏi/lịch sử, không nhận toàn bộ tài liệu; các kỹ thuật đang bật dùng chung tối đa một generation request.
- **Query embeddings** là các câu truy vấn sau khi bỏ trùng và chuyển sang vector; danh sách thường gồm câu gốc, câu rewrite, query phụ và/hoặc đoạn HyDE. Mỗi vector tìm kiếm một danh sách Dense.
- **Parent-Child** tìm bằng child nhỏ rồi khôi phục parent lớn làm ngữ cảnh; nếu không tìm được parent, dùng nội dung child và ghi nhận fallback.
- **Context compression** chọn đoạn/hàng liên quan cục bộ. **Answer Grading** đánh giá câu trả lời theo nguồn, **Citation Verification** kiểm tra mã trích dẫn và mức hỗ trợ. Hai kiểm tra dùng chung tối đa một generation request.
- **LangGraph** giữ trạng thái và điều phối các node planner → retrieval → rerank/context → answer → audit trong chat thường. Lab thực hiện từng bước riêng để có thể xem và xác nhận từng đầu ra.
- **Multi-agent** là một pipeline chọn riêng: LangGraph điều phối Supervisor → tối đa 3 specialist → Synthesis → Critic. Các agent có prompt/vai trò khác nhau nhưng dùng chung context retrieval; chạy tuần tự theo RPM và bị giới hạn số node. Tốn nhiều generation call hơn; mở **🧭 Kịch bản sản phẩm** để tạo luồng nhiều bước và xem báo cáo agent.
- **Kịch bản sản phẩm** lưu các bước nhập text/lựa chọn/tài liệu, allowlist agent và phiên bản; khi có bước chọn file, retrieval chỉ dùng các file đó. Lịch sử chạy lưu input, snapshot, result, source và diagnostics.
- **Giọng nói**: microphone chuyển sang text bằng một lượt Gemini, người dùng duyệt/sửa trước khi gửi. Nút loa dùng giọng tiếng Việt sẵn có trong trình duyệt, không gọi Gemini TTS.
- Câu trả lời `[S1]`, `[S2]`… tham chiếu danh sách nguồn hiển thị bên dưới đáp án. Gợi ý câu hỏi tiếp theo là một phần của cùng câu trả lời Gemini, không phải request riêng.
- **Thời gian** là thời lượng xử lý của lượt. **Lịch sử hội thoại** lưu user/assistant messages theo từng chat và kho; thao tác đổi/xóa nằm ở thanh lịch sử.

### Các thông số trong phần cài đặt

| Thông số | Phạm vi / mặc định | Tác dụng |
|---|---|---|
| Độ dài chunk | 300–3.000 ký tự; mặc định 900 | Mục tiêu độ dài mỗi child/chunk. |
| Chunk overlap | 0–800 ký tự; mặc định 150; phải nhỏ hơn chunk size | Phần văn bản lặp qua ranh giới chunk. |
| Số chunk tối đa xem | 10–200; mặc định 50 | Chỉ giới hạn số chunk hiển thị trong inspector, không đổi index hay retrieval. |
| Ứng viên trước rerank (`candidate_k`) | 8–50; mặc định 24 | Độ rộng danh sách tìm kiếm trước reranker. |
| Số chunk truy xuất (`top_k`) | 2–15; mặc định 6 | Số nguồn tối đa gửi sang bước tạo câu trả lời. |
| Ngân sách context | 500–12.000 token; mặc định 4.000 | Giới hạn xấp xỉ nội dung đưa vào prompt Gemini. |
| Embedding RPM / Generation RPM | 1–600; mặc định lấy từ cấu hình quota (thường 10) | Giãn nhịp từng loại request để tránh burst/quota. |
| Batch embedding | Tối đa 32 văn bản/request theo cấu hình mặc định | Gom embedding để giảm số request. |
| Kích thước file upload | Tối đa 100 MB mặc định | Có thể đổi bằng cấu hình `MAX_UPLOAD_MB`. |

### Gemini, quota và số liệu lượt chạy

| Chỉ số | Ý nghĩa |
|---|---|
| **Embedding** | Request biến văn bản thành vector để lưu/tìm kiếm. Dùng model embedding cấu hình trong app; mặc định `gemini-embedding-001`, 768 chiều. Batch mặc định tối đa 32 văn bản/request. |
| **Generation** | Request sinh văn bản bằng model Gemini cấu hình trong app (mặc định `gemini-3.5-flash`): lập kế hoạch query, câu trả lời hoặc audit. Query Rewrite/Multi-query/HyDE dùng chung tối đa một lượt planner; Answer Grading/Citation Verification dùng chung tối đa một lượt audit. |
| **API calls** | Số lần thử gọi API, có thể gồm cả lần retry; embedding và generation được tách riêng. BM25, RRF, FlashRank và nén context chạy local. |
| **Input / output / tổng token** | Token prompt gửi lên / token câu trả lời trả về / tổng hai loại. Ưu tiên usage metadata Gemini; khi thiếu, app ước lượng từ số ký tự chia 3 và báo số lượt có ước tính. |
| **Embedding RPM / Generation RPM** | Nhịp gọi tối đa mỗi phút mà app tự giãn theo cấu hình. Đây là giới hạn pacing cục bộ; quota thực tế còn phụ thuộc project, model và tier của Google. |
| **Answer Grading** | Gemini chấm mức độ đúng, đầy đủ, có nguồn hỗ trợ từ 1–5. **Citation Verification** vừa kiểm tra mã nguồn có tồn tại/được trích dẫn, vừa có thể yêu cầu Gemini xem mức hỗ trợ nội dung. |

Planner tạo query phụ tối đa 2 câu (mỗi câu tối đa 20 từ); HyDE tạo đoạn giả thuyết tối đa 80 từ. Các query trùng được loại bỏ trước khi embedding. Câu trả lời và các câu hỏi gợi ý được trả trong cùng lượt generation.

Giới hạn đầu ra Gemini trong code: planner tối đa 350 token, câu trả lời tối đa 1.600 token, audit tối đa 250 token; model dùng thinking level `minimal`. App đặt khoảng cách request cục bộ xấp xỉ `63 / RPM` giây để chừa một khoảng đệm so với 60 giây chuẩn; khi gặp lỗi quota/temporary server, retry có backoff giới hạn theo cấu hình. Giới hạn pacing này không tăng quota của project.

Trong câu trả lời, `[S1]`, `[S2]` là mã nguồn ngắn trỏ về file/trang/chunk trong danh sách **Nguồn đã dùng**; chúng không phải chunk ID. **Thời gian** trong chẩn đoán là thời lượng xử lý lượt hỏi, còn **pipeline/kỹ thuật** cho biết chế độ và công tắc thực sự được dùng.
"""
        )


def _render_lab_checkpoints(state: dict, max_chunks: int) -> None:
    step_number = _LAB_STEP_ORDER.get(state.get("step", "chunking"), 0)
    upload = state["upload"]
    selected_strategy = state.get("selected_strategy")
    selected_label = "Chunk thường" if selected_strategy == "regular" else "Parent-Child"

    if step_number >= 1 and selected_strategy:
        with st.expander(f"✅ Bước 1 · Chia chunk — {selected_label}", expanded=step_number == 1):
            st.caption(
                f"Parser: {upload['parser']} · {len(upload['sections'])} phần trích xuất · "
                f"{len(state['selected_chunks'])} chunk · {len(state['selected_parents'])} parent"
            )
            _render_lab_chunks(
                state["selected_chunks"],
                min(max_chunks, 30),
                title="Kết quả đã chọn",
                parents=state["selected_parents"],
            )
            if state.get("index"):
                st.caption("Các embedding đã được ghi vào collection Chroma tạm trong phiên này.")

    if step_number >= 2:
        with st.expander("✅ Bước 2 · Lập kế hoạch truy vấn", expanded=step_number == 2):
            st.write("Câu hỏi:", state.get("question", ""))
            st.write("Kỹ thuật:", ", ".join(sorted(state.get("planner_methods", set()))) or "Không dùng planner")
            plan = state.get("plan", {})
            st.write("Query rewrite:", plan.get("rewritten_query") or "Không đổi")
            if plan.get("alternative_queries"):
                st.write("Multi-query:", plan["alternative_queries"])
            if plan.get("hypothetical_answer"):
                st.markdown("**HyDE**")
                st.code(plan["hypothetical_answer"], language=None)

    if step_number >= 3:
        with st.expander("✅ Bước 3 · Truy xuất", expanded=step_number == 3):
            diagnostics = state.get("retrieval_diagnostics", {})
            st.caption(
                f"Phương thức: {state.get('retrieval_method')} · "
                f"Dense: {diagnostics.get('dense_candidate_count', 0)} · "
                f"BM25: {diagnostics.get('bm25_candidate_count', 0)} · "
                f"Hợp nhất: {diagnostics.get('fused_candidate_count', 0)} ứng viên"
            )
            st.write("Các truy vấn đã dùng:", diagnostics.get("queries_used", []))
            chunk_by_id = {str(item["chunk_id"]): item for item in state["selected_chunks"]}
            parent_lookup = _parent_map(state.get("selected_parents"))
            dense_labels = [
                _lab_chunk_label(chunk_by_id[item_id], parent_lookup)
                for item_id in diagnostics.get("dense_top_ids", [])
                if item_id in chunk_by_id
            ][:10]
            if dense_labels:
                st.markdown("**Dense top · truy vấn gốc**")
                for label in dense_labels:
                    st.markdown(f"- {label}")
            else:
                st.caption("Dense top: không có kết quả.")
            if diagnostics.get("hybrid_search"):
                st.markdown("**BM25 top · truy vấn gốc**")
                st.write(
                    [
                        _lab_chunk_label(chunk_by_id[item_id], parent_lookup)
                        for item_id in diagnostics.get("bm25_top_ids", [])
                        if item_id in chunk_by_id
                    ][:10]
                )
            for candidate in state.get("initial_candidates", [])[:12]:
                st.markdown(f"**{_lab_chunk_label(candidate, parent_lookup)}** · RRF {candidate.get('rrf_score', 0):.5f}")
                st.code(candidate["document"].page_content, language=None)
            if len(state.get("initial_candidates", [])) > 12:
                st.caption("Chỉ hiển thị 12 ứng viên đầu tiên.")

    if step_number >= 4:
        with st.expander("✅ Bước 4 · Reranker", expanded=step_number == 4):
            before = state.get("rerank_before_ids", [])
            after = state.get("rerank_after_ids", [])
            st.write("Reranker:", "FlashRank local" if state.get("use_reranker") else "Tắt")
            candidate_by_id = {
                item["id"]: item
                for item in [*state.get("initial_candidates", []), *state.get("candidates", [])]
            }
            parents_by_id = _parent_map(state.get("selected_parents"))
            left, right = st.columns(2)
            left.caption("Thứ tự trước")
            right.caption("Thứ tự sau")
            for index, item_id in enumerate(before[:15], start=1):
                item = candidate_by_id.get(item_id)
                left.markdown(f"{index}. {_lab_chunk_label(item, parents_by_id) if item else 'Chunk không còn trong danh sách'}")
            for index, item_id in enumerate(after[:15], start=1):
                item = candidate_by_id.get(item_id)
                right.markdown(f"{index}. {_lab_chunk_label(item, parents_by_id) if item else 'Chunk không còn trong danh sách'}")
            if state.get("use_reranker"):
                for candidate in state.get("candidates", [])[:12]:
                    st.markdown(
                        f"**{_lab_chunk_label(candidate, parents_by_id)}** · rerank score "
                        f"{candidate.get('rerank_score', 0):.4f}"
                    )
                    st.code(candidate["document"].page_content, language=None)

    if step_number >= 5:
        with st.expander("✅ Bước 5 · Context gửi vào Gemini", expanded=step_number == 5):
            diagnostics = state.get("context_diagnostics", {})
            st.caption(
                f"Nén: {'bật' if state.get('compress_context') else 'tắt'} · "
                f"{diagnostics.get('context_chars_before', 0):,} → "
                f"{diagnostics.get('context_chars_after', 0):,} ký tự · "
                f"~{diagnostics.get('context_tokens_estimated', 0):,} token"
            )
            parent_lookup = _parent_map(state.get("selected_parents"))
            for source in state.get("sources", []):
                st.caption(f"[{source.get('ref')}] {_lab_source_label(source, parent_lookup)}")
            st.code(state.get("context", ""), language=None)

    if step_number >= 6:
        with st.expander("✅ Bước 6 · Câu trả lời", expanded=step_number == 6):
            st.markdown(state.get("answer", {}).get("answer", "Chưa có câu trả lời."))
            parent_lookup = _parent_map(state.get("selected_parents"))
            for source in state.get("sources", []):
                st.caption(f"[{source.get('ref')}] {_lab_source_label(source, parent_lookup)}")
            if state.get("answer", {}).get("suggestions"):
                st.write("Câu hỏi gợi ý:", state["answer"]["suggestions"])

    if step_number >= 7:
        with st.expander("✅ Bước 7 · Answer Grading / Citation Verification", expanded=True):
            if state.get("audit"):
                _render_lab_audit(state["audit"])
            else:
                st.caption("Đã chọn bỏ qua audit.")


def _render_stepwise_lab(
    kb_id: str,
    kb_name: str,
    api_key: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
    candidate_k: int,
    top_k: int,
    context_token_budget: int,
    embedding_rpm: int,
    generation_rpm: int,
) -> None:
    st.title("🧪 Phòng thử nghiệm RAG từng bước")
    st.caption(
        "Thử nghiệm tách biệt với lịch sử chat. Index Chroma chỉ nằm trong bộ nhớ của phiên; "
        "các bước truy xuất chỉ dùng file thử nghiệm, không đọc các file khác trong kho. "
        "Phòng thử nghiệm cho phép dừng ở từng bước để xem input/output; chat thường điều phối pipeline tự động qua LangGraph. "
        "Ở cuối quy trình có thể chọn lưu tài liệu vào kho thật."
    )
    _render_rag_explainer()
    state = st.session_state.get("stepwise-rag-lab")
    if state and state.get("kb_id") != kb_id:
        st.warning("Bạn đã đổi kho trong lúc đang thử nghiệm. Hãy quay lại kho ban đầu hoặc bắt đầu lượt thử mới.")
        st.caption(f"Lượt hiện tại gắn với kho: {state.get('kb_name', 'không rõ')}")
        if st.button("Hủy lượt thử và giải phóng index tạm", key="lab-cancel-other-kb"):
            rag_lab.cleanup_index(state.get("index"))
            st.session_state.pop("stepwise-rag-lab", None)
            st.rerun()
        return

    if state:
        if st.button("↻ Bắt đầu lượt thử nghiệm mới", key="lab-reset", type="secondary"):
            rag_lab.cleanup_index(state.get("index"))
            st.session_state.pop("stepwise-rag-lab", None)
            st.rerun()
        _render_lab_checkpoints(state, int(max_chunks))
    else:
        st.markdown(f"### File thử nghiệm · kho đích nếu lưu: {kb_name}")
        uploaded = st.file_uploader(
            "Chọn một tài liệu để chạy thử",
            type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
            accept_multiple_files=False,
            key=f"lab-upload-{kb_id}",
        )
        if st.button(
            "Đọc file và tạo bản xem trước chunk",
            type="primary",
            disabled=uploaded is None or chunk_overlap >= chunk_size,
            key="lab-prepare-upload",
        ):
            try:
                prepared = rag_lab.prepare_upload(
                    uploaded.name,
                    uploaded.getvalue(),
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                )
                st.session_state["stepwise-rag-lab"] = {
                    "id": prepared["id"],
                    "kb_id": kb_id,
                    "kb_name": kb_name,
                    "upload": prepared,
                    "meter": RequestMeter(),
                    "step": "chunking",
                }
                st.rerun()
            except Exception as error:
                st.error(f"Không đọc được file: {error}")
        st.caption("File chỉ được đọc và chia chunk để xem trước; chưa được thêm vào kho thật.")
        return

    step = state.get("step", "chunking")
    st.progress((_LAB_STEP_ORDER[step] + 1) / len(_LAB_STEP_ORDER), text=f"Bước {_LAB_STEP_ORDER[step] + 1}/8")
    upload = state["upload"]

    previous_step = {
        "planning": "chunking",
        "retrieval": "planning",
        "reranker": "retrieval",
        "context": "reranker",
        "answer": "context",
        "audit": "answer",
    }.get(step)
    if previous_step and st.button("⬅ Quay lại bước trước", key=f"lab-back-{step}-{state['id']}"):
        state["step"] = previous_step
        st.session_state["stepwise-rag-lab"] = state
        st.rerun()

    if step == "chunking":
        st.subheader("Bước 1 · Chọn và xem cách chia chunk")
        st.caption(
            f"File: {upload['file_name']} · Parser: {upload['parser']} · "
            f"{len(upload['sections'])} phần nội dung"
        )
        with st.expander("Bước 0 · Nội dung parser trích xuất trước khi chia chunk", expanded=False):
            for section_index, section in enumerate(upload["sections"][:10], start=1):
                st.caption(
                    f"Phần {section_index} · {section.kind} · {_location(section.page_number)} · "
                    f"{section.section or 'Không có tiêu đề'} · {section.parser}"
                )
                st.code(section.text[:12000], language=None)
                if len(section.text) > 12000:
                    st.caption("Phần này vượt 12.000 ký tự; chỉ hiển thị đầu nội dung.")
            if len(upload["sections"]) > 10:
                st.caption("Chỉ hiển thị 10 phần đầu tiên.")
        choice = st.selectbox(
            "Chọn một cách chia để xem trước",
            options=["Chunk thường", "Parent-Child chunking"],
            index=None,
            placeholder="Bắt buộc chọn một cách chia…",
            key=f"lab-chunk-choice-{state['id']}",
        )
        compare = st.checkbox("Đối chiếu thêm cả hai cách chia", key=f"lab-chunk-compare-{state['id']}")
        chosen_strategy = "regular" if choice == "Chunk thường" else "parent_child"
        strategy_names = ["regular", "parent_child"] if compare else ([chosen_strategy] if choice else [])
        preview_cap = st.number_input(
            "Số chunk tối đa xem",
            min_value=5,
            max_value=100,
            value=min(30, max(5, int(max_chunks))),
            step=5,
            key=f"lab-preview-count-{state['id']}",
        )
        for strategy in strategy_names:
            preview = upload["previews"][strategy]
            title = "Chunk thường" if strategy == "regular" else "Parent-Child"
            st.markdown(f"#### {title} · {len(preview['chunks']):,} chunk · {len(preview['parents']):,} parent")
            _render_lab_chunks(
                preview["chunks"],
                int(preview_cap),
                title="Nội dung sau chia",
                parents=preview["parents"],
            )
            if strategy == "parent_child" and preview["parents"]:
                with st.expander("Xem các parent chunks", expanded=False):
                    _render_lab_chunks(preview["parents"], min(int(preview_cap), 20), title="Parent chunks")
        if not choice:
            st.info("Chọn một phương pháp để xem bản chia tương ứng. Không thể đi tiếp khi chưa chọn.")
        if not api_key:
            st.warning("Nhập Gemini API key trong phần cài đặt bên trái để tạo embedding và tiếp tục.")
        if st.button(
            "Chốt cách chia, tạo embedding và chuyển sang bước tiếp theo",
            type="primary",
            disabled=not choice or not api_key,
            key=f"lab-continue-chunk-{state['id']}",
        ):
            if not api_key:
                st.error("Nhập Gemini API key trong phần cài đặt bên trái trước khi tạo embedding.")
                return
            preview = upload["previews"][chosen_strategy]
            progress_box = st.empty()
            progress_bar = st.progress(0.02)
            progress_value = {"value": 0.02}

            def progress(message: str) -> None:
                progress_box.caption(message)
                progress_value["value"] = min(0.95, progress_value["value"] + 0.05)
                progress_bar.progress(progress_value["value"])

            try:
                if state.get("selected_strategy") == chosen_strategy and state.get("index"):
                    index = state["index"]
                else:
                    index = rag_lab.create_ephemeral_index(
                        preview["chunks"],
                        api_key,
                        state["meter"],
                        embedding_rpm=int(embedding_rpm),
                        progress=progress,
                    )
                    rag_lab.cleanup_index(state.get("index"))
                state.update(
                    {
                        "selected_strategy": chosen_strategy,
                        "selected_chunks": preview["chunks"],
                        "selected_parents": preview["parents"],
                        "index": index,
                        "step": "planning",
                    }
                )
                progress_bar.progress(1.0)
                st.session_state["stepwise-rag-lab"] = state
                st.rerun()
            except Exception as error:
                st.error(f"Không tạo được embedding/index tạm: {error}")
        return

    _render_lab_meter(state["meter"])

    if step == "planning":
        st.subheader("Bước 2 · Query Rewrite / Multi-query / HyDE")
        question_key = f"lab-question-{state['id']}"
        planner_widget_key = f"lab-planner-choice-{state['id']}"
        if not state.get("question_locked"):
            question = st.text_area("Câu hỏi thử nghiệm", key=question_key, height=90)
            if st.button(
                "Xác nhận câu hỏi",
                type="primary",
                disabled=not question.strip(),
                key=f"lab-lock-question-{state['id']}",
            ):
                state["question"] = question.strip()
                state["question_locked"] = True
                state.setdefault("planner_cache", {})
                st.session_state["stepwise-rag-lab"] = state
                st.rerun()
            return

        st.info(f"Câu hỏi: {state['question']}")
        if st.button("Đổi câu hỏi", key=f"lab-unlock-question-{state['id']}"):
            state["question_locked"] = False
            state.pop("planner_preview_key", None)
            state.pop("planner_preview", None)
            st.session_state.pop(planner_widget_key, None)
            st.session_state["stepwise-rag-lab"] = state
            st.rerun()

        planner_options = {
            "Không dùng planner": set(),
            "Query Rewrite": {"Rewrite"},
            "Multi-query": {"Multi-query"},
            "HyDE": {"HyDE"},
            "Query Rewrite + Multi-query": {"Rewrite", "Multi-query"},
            "Query Rewrite + HyDE": {"Rewrite", "HyDE"},
            "Multi-query + HyDE": {"Multi-query", "HyDE"},
            "Query Rewrite + Multi-query + HyDE": {"Rewrite", "Multi-query", "HyDE"},
        }
        planner_choice = st.selectbox(
            "Chọn tổ hợp kỹ thuật; kết quả sẽ chạy và hiện ngay khi chọn",
            list(planner_options),
            index=None,
            placeholder="Bắt buộc chọn một phương án…",
            key=planner_widget_key,
        )
        planner_methods = planner_options.get(planner_choice, set())
        if planner_choice is not None:
            method_key = ",".join(sorted(planner_methods)) or "none"
            preview_key = f"{state['question']}\0{method_key}"
            planner_cache = state.setdefault("planner_cache", {})
            if preview_key not in planner_cache:
                if planner_methods and not api_key:
                    st.warning("Nhập Gemini API key để chạy và xem preview planner.")
                else:
                    progress_box = st.empty()
                    try:
                        planner_cache[preview_key] = rag_lab.plan_queries(
                            state["question"],
                            api_key,
                            planner_methods,
                            state["meter"],
                            generation_rpm=int(generation_rpm),
                            progress=progress_box.caption,
                        )
                        state["planner_cache"] = planner_cache
                        st.session_state["stepwise-rag-lab"] = state
                    except Exception as error:
                        st.error(f"Planner thất bại: {error}")
            plan = planner_cache.get(preview_key)
            if plan is not None:
                state["planner_preview_key"] = preview_key
                state["planner_preview"] = plan
                st.markdown("#### Preview đầu ra planner")
                st.write("Query dùng cho vector search:", plan.get("rewritten_query") or state["question"])
                st.write("Các query phụ:", plan.get("alternative_queries") or "Không tạo query phụ")
                if plan.get("hypothetical_answer"):
                    st.markdown("**HyDE · câu trả lời giả thuyết dùng để tìm kiếm**")
                    st.code(plan["hypothetical_answer"], language=None)
                if planner_methods:
                    st.caption("Các kỹ thuật đã chọn dùng chung tối đa một generation call. Đổi sang tổ hợp mới sẽ tạo preview mới; tổ hợp đã xem được cache trong lượt thử này.")
                _render_lab_meter(state["meter"])
                if st.button(
                    "Dùng kết quả này và tiếp tục",
                    type="primary",
                    key=f"lab-continue-plan-{state['id']}",
                ):
                    state.update(
                        {
                            "planner_methods": planner_methods,
                            "planner_choice": planner_choice,
                            "plan": plan,
                            "planner_commit_key": preview_key,
                            "step": "retrieval",
                        }
                    )
                    st.session_state["stepwise-rag-lab"] = state
                    st.rerun()
        return

    if step == "retrieval":
        st.subheader("Bước 3 · Chọn phương pháp tìm kiếm")
        method = st.selectbox(
            "Chọn retrieval; kết quả sẽ chạy và hiện ngay khi chọn",
            ["Vector search thông thường", "Hybrid Search (vector + BM25)"],
            index=None,
            placeholder="Bắt buộc chọn một phương pháp…",
            key=f"lab-retrieval-choice-{state['id']}",
        )
        if method is None:
            st.info("Chọn một phương pháp để chạy và xem ngay danh sách chunk tìm được.")
            return
        if not api_key:
            st.warning("Nhập Gemini API key để tạo embedding cho truy vấn.")
            return
        retrieval_cache = state.setdefault("retrieval_cache", {})
        retrieval_cache_key = (
            f"{state.get('planner_commit_key', '')}|{state['selected_strategy']}|"
            f"{method}|k={int(candidate_k)}"
        )
        if retrieval_cache_key not in retrieval_cache:
            progress_box = st.empty()
            try:
                queries = rag_lab.query_strings(state["question"], state["plan"])
                query_vectors = rag_lab.embed_queries_cached(
                    queries,
                    api_key,
                    state["meter"],
                    state.setdefault("query_vector_cache", {}),
                    embedding_rpm=int(embedding_rpm),
                    progress=progress_box.caption,
                )
                candidates, diagnostics = rag_lab.retrieve(
                    state["index"],
                    state["selected_chunks"],
                    state["selected_parents"],
                    file_name=upload["file_name"],
                    question=state["question"],
                    plan=state["plan"],
                    api_key=api_key,
                    meter=state["meter"],
                    use_hybrid=method.startswith("Hybrid"),
                    use_parent_child=state["selected_strategy"] == "parent_child",
                    candidate_k=int(candidate_k),
                    embedding_rpm=int(embedding_rpm),
                    query_vectors=query_vectors,
                    progress=progress_box.caption,
                )
                retrieval_cache[retrieval_cache_key] = {"candidates": candidates, "diagnostics": diagnostics}
                state["retrieval_cache"] = retrieval_cache
                state["query_vector_cache"] = state.get("query_vector_cache", {})
                st.session_state["stepwise-rag-lab"] = state
            except Exception as error:
                st.error(f"Retrieval thất bại: {error}")
        preview = retrieval_cache.get(retrieval_cache_key)
        if preview:
            candidates = preview["candidates"]
            diagnostics = preview["diagnostics"]
            state["retrieval_preview_method"] = method
            state["retrieval_preview_key"] = retrieval_cache_key
            state["retrieval_preview"] = preview
            st.markdown("#### Preview kết quả retrieval")
            st.caption(
                f"Dense: {diagnostics['dense_candidate_count']} · "
                f"BM25: {diagnostics['bm25_candidate_count']} · "
                f"Hợp nhất: {diagnostics['fused_candidate_count']} ứng viên"
            )
            st.write("Query đã dùng:", diagnostics["queries_used"])
            for candidate in candidates[:12]:
                st.markdown(
                    f"**{_lab_chunk_label(candidate, _parent_map(state.get('selected_parents')))}** · "
                    f"RRF {candidate.get('rrf_score', 0):.5f}"
                )
                st.code(candidate["document"].page_content, language=None)
            if len(candidates) > 12:
                st.caption("Chỉ hiển thị 12 ứng viên đầu tiên.")
            chunk_by_id = {str(item["chunk_id"]): item for item in state["selected_chunks"]}
            with st.expander("Xem danh sách top riêng của Dense và BM25", expanded=False):
                dense_labels = [
                    _lab_chunk_label(chunk_by_id[chunk_id], _parent_map(state.get("selected_parents")))
                    for chunk_id in diagnostics.get("dense_top_ids", [])
                    if chunk_id in chunk_by_id
                ]
                st.markdown("**Dense top · truy vấn gốc**")
                st.write(dense_labels[:10] or "Không có kết quả.")
                if method.startswith("Hybrid"):
                    bm25_labels = [
                        _lab_chunk_label(chunk_by_id[chunk_id], _parent_map(state.get("selected_parents")))
                        for chunk_id in diagnostics.get("bm25_top_ids", [])
                        if chunk_id in chunk_by_id
                    ]
                    st.markdown("**BM25 top · truy vấn gốc**")
                    st.write(bm25_labels[:10] or "Không có kết quả phù hợp từ BM25.")
            _render_lab_meter(state["meter"])
            if st.button(
                "Dùng kết quả retrieval này và tiếp tục",
                type="primary",
                key=f"lab-continue-retrieval-{state['id']}",
            ):
                state.update(
                    {
                        "retrieval_method": method,
                        "retrieval_commit_key": retrieval_cache_key,
                        "initial_candidates": candidates,
                        "candidates": candidates,
                        "retrieval_diagnostics": diagnostics,
                        "step": "reranker",
                    }
                )
                st.session_state["stepwise-rag-lab"] = state
                st.rerun()
        return

    if step == "reranker":
        st.subheader("Bước 4 · Có dùng FlashRank Reranker không?")
        method = st.selectbox(
            "Chọn cách xếp hạng; thứ tự mới sẽ hiện ngay",
            ["Giữ thứ tự từ retrieval", "Xếp hạng lại bằng FlashRank local"],
            index=None,
            placeholder="Bắt buộc chọn một phương án…",
            key=f"lab-reranker-choice-{state['id']}",
        )
        if method is None:
            st.info("Chọn một phương án để xem ngay thứ tự ứng viên.")
            return
        rerank_cache = state.setdefault("rerank_cache", {})
        rerank_cache_key = f"{state.get('retrieval_commit_key', '')}|{method}"
        if rerank_cache_key not in rerank_cache:
            try:
                before = [dict(item) for item in state["initial_candidates"]]
                use_reranker = method.startswith("Xếp hạng lại")
                with st.spinner("FlashRank đang xếp hạng lại các chunk…" if use_reranker else "Đang giữ thứ tự retrieval…"):
                    candidates = rag_lab.rerank_candidates(
                        before,
                        state["question"],
                        state["plan"].get("rewritten_query", ""),
                    ) if use_reranker else before
                rerank_cache[rerank_cache_key] = {
                    "use_reranker": use_reranker,
                    "before": before,
                    "candidates": candidates,
                }
                state["rerank_cache"] = rerank_cache
                st.session_state["stepwise-rag-lab"] = state
            except Exception as error:
                st.error(f"Reranker không chạy được: {error}")
        preview = rerank_cache.get(rerank_cache_key)
        if preview:
            before = preview["before"]
            candidates = preview["candidates"]
            st.markdown("#### Preview trước / sau rerank")
            left, right = st.columns(2)
            left.markdown("**Trước**")
            right.markdown("**Sau**")
            for index in range(min(10, max(len(before), len(candidates)))):
                if index < len(before):
                    item = before[index]
                    left.markdown(f"{index + 1}. {_lab_chunk_label(item, _parent_map(state.get('selected_parents')))} · RRF {item.get('rrf_score', 0):.5f}")
                if index < len(candidates):
                    item = candidates[index]
                    score = item.get("rerank_score")
                    score_label = f"rerank {score:.4f}" if score is not None else f"RRF {item.get('rrf_score', 0):.5f}"
                    right.markdown(f"{index + 1}. {_lab_chunk_label(item, _parent_map(state.get('selected_parents')))} · {score_label}")
            for candidate in candidates[:8]:
                st.markdown(f"**{_lab_chunk_label(candidate, _parent_map(state.get('selected_parents')))}**")
                st.code(candidate["document"].page_content, language=None)
            if preview["use_reranker"]:
                st.caption("FlashRank chạy cục bộ; sau lần tải model đầu không phát sinh Gemini request.")
            if st.button(
                "Dùng thứ tự này và tiếp tục",
                type="primary",
                key=f"lab-continue-reranker-{state['id']}",
            ):
                state.update(
                    {
                        "use_reranker": preview["use_reranker"],
                        "rerank_before_ids": [item["id"] for item in before],
                        "rerank_after_ids": [item["id"] for item in candidates],
                        "candidates": candidates,
                        "rerank_commit_key": rerank_cache_key,
                        "step": "context",
                    }
                )
                st.session_state["stepwise-rag-lab"] = state
                st.rerun()
        return

    if step == "context":
        st.subheader("Bước 5 · Chọn ngữ cảnh gửi sang Gemini")
        method = st.selectbox(
            "Chọn cách dựng context; kết quả sẽ hiện ngay",
            ["Giữ nguyên context", "Nén context theo câu hỏi"],
            index=None,
            placeholder="Bắt buộc chọn một phương án…",
            key=f"lab-context-choice-{state['id']}",
        )
        if method is None:
            st.info("Chọn một cách để dựng và xem context ngay.")
            return
        context_pair_cache = state.setdefault("context_pair_cache", {})
        context_pair_key = (
            f"{state.get('rerank_commit_key', '')}|"
            f"top_k={int(top_k)}|budget={int(context_token_budget)}"
        )
        if context_pair_key not in context_pair_cache:
            try:
                plain_context = rag_lab.build_context(
                    state["question"],
                    state["candidates"],
                    use_parent_child=state["selected_strategy"] == "parent_child",
                    compress=False,
                    top_k=int(top_k),
                    context_token_budget=int(context_token_budget),
                )
                compressed_context = rag_lab.build_context(
                    state["question"],
                    state["candidates"],
                    use_parent_child=state["selected_strategy"] == "parent_child",
                    compress=True,
                    top_k=int(top_k),
                    context_token_budget=int(context_token_budget),
                )
                context_pair_cache[context_pair_key] = {
                    "plain": plain_context,
                    "compressed": compressed_context,
                }
                state["context_pair_cache"] = context_pair_cache
                st.session_state["stepwise-rag-lab"] = state
            except Exception as error:
                st.error(f"Không dựng được context: {error}")
        pair = context_pair_cache.get(context_pair_key)
        if pair:
            plain_context, plain_sources, plain_diagnostics = pair["plain"]
            compressed_context, compressed_sources, compressed_diagnostics = pair["compressed"]
            use_compressed = method.startswith("Nén")
            preview = pair["compressed"] if use_compressed else pair["plain"]
            context, sources, diagnostics = preview
            context_cache_key = f"{context_pair_key}|{'compressed' if use_compressed else 'plain'}"
            state["context_preview_method"] = method
            state["context_preview_key"] = context_cache_key
            state["context_preview"] = preview
            st.markdown("#### So sánh context không nén và có nén")
            plain_chars = plain_diagnostics["context_chars_after"]
            compressed_chars = compressed_diagnostics["context_chars_after"]
            saved_chars = max(0, plain_chars - compressed_chars)
            savings_percent = (saved_chars / plain_chars * 100) if plain_chars else 0
            st.caption(
                f"Không nén: {plain_chars:,} ký tự · Có nén: {compressed_chars:,} ký tự · "
                f"Giảm {saved_chars:,} ký tự ({savings_percent:.1f}%) · "
                f"Ước tính token: {plain_diagnostics['context_tokens_estimated']:,} → "
                f"{compressed_diagnostics['context_tokens_estimated']:,}"
            )
            if plain_context == compressed_context:
                st.info("Hai context hiện giống nhau: nội dung đã nằm trong giới hạn hoặc không có dòng/câu đủ liên quan để loại bớt.")
            plain_col, compressed_col = st.columns(2)
            with plain_col:
                st.markdown("**Context không nén**")
                st.caption(f"Nguồn gửi: {len(plain_sources)}")
                for source in plain_sources:
                    st.caption(f"[{source.get('ref')}] {_lab_source_label(source, _parent_map(state.get('selected_parents')))}")
                st.text_area(
                    "Nội dung không nén",
                    value=plain_context,
                    height=360,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"lab-context-plain-{state['id']}-{abs(hash(context_pair_key))}",
                )
            with compressed_col:
                st.markdown("**Context có nén**")
                st.caption(f"Nguồn gửi: {len(compressed_sources)}")
                for source in compressed_sources:
                    st.caption(f"[{source.get('ref')}] {_lab_source_label(source, _parent_map(state.get('selected_parents')))}")
                st.text_area(
                    "Nội dung đã nén",
                    value=compressed_context,
                    height=360,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"lab-context-compressed-{state['id']}-{abs(hash(context_pair_key))}",
                )
            st.info(f"Đang chọn **{'context có nén' if use_compressed else 'context không nén'}** để gửi sang Gemini.")
            if st.button(
                "Dùng context này và tiếp tục",
                type="primary",
                key=f"lab-continue-context-{state['id']}",
            ):
                if state.get("context_commit_key") != context_cache_key:
                    state.pop("answer", None)
                    state.pop("audit", None)
                    state["audit_cache"] = {}
                state.update(
                    {
                        "compress_context": method.startswith("Nén"),
                        "context": context,
                        "sources": sources,
                        "context_diagnostics": diagnostics,
                        "context_commit_key": context_cache_key,
                        "step": "answer",
                    }
                )
                st.session_state["stepwise-rag-lab"] = state
                st.rerun()
        return

    if step == "answer":
        st.subheader("Bước 6 · Sinh câu trả lời có trích dẫn")
        st.caption("Gemini nhận câu hỏi cùng context đã chọn ở bước trước. Câu trả lời sẽ hiện tại đây trước khi bạn đi tiếp.")
        if not state.get("answer"):
            if st.button(
                "Gọi Gemini và tạo câu trả lời",
                type="primary",
                disabled=not api_key,
                key=f"lab-generate-answer-{state['id']}",
            ):
                progress_box = st.empty()
                try:
                    answer = rag_lab.generate_answer(
                        state["question"],
                        state["context"],
                        api_key,
                        state["meter"],
                        generation_rpm=int(generation_rpm),
                        progress=progress_box.caption,
                    )
                    state.update({"answer": answer, "audit_cache": {}})
                    st.session_state["stepwise-rag-lab"] = state
                    st.rerun()
                except Exception as error:
                    st.error(f"Sinh câu trả lời thất bại: {error}")
        else:
            st.markdown("#### Câu trả lời")
            st.markdown(state["answer"].get("answer", "Gemini không trả nội dung câu trả lời."))
            for source in state.get("sources", []):
                st.caption(f"[{source.get('ref')}] {_lab_source_label(source, _parent_map(state.get('selected_parents')))}")
            suggestions = state["answer"].get("suggestions", [])
            if suggestions:
                st.markdown("**Gợi ý câu hỏi tiếp theo**")
                for suggestion in suggestions:
                    st.markdown(f"- {suggestion}")
            _render_lab_meter(state["meter"])
            if st.button(
                "Dùng câu trả lời này và tiếp tục kiểm tra",
                type="primary",
                key=f"lab-continue-answer-{state['id']}",
            ):
                state["step"] = "audit"
                st.session_state["stepwise-rag-lab"] = state
                st.rerun()
        return

    if step == "audit":
        st.subheader("Bước 7 · Chọn kiểm tra sau câu trả lời")
        audit_choice = st.selectbox(
            "Chọn audit; kết quả sẽ chạy và hiện ngay khi chọn",
            ["Không audit", "Answer Grading", "Citation Verification", "Cả hai"],
            index=None,
            placeholder="Bắt buộc chọn một phương án…",
            key=f"lab-audit-choice-{state['id']}",
        )
        audit_methods = {
            "Không audit": set(),
            "Answer Grading": {"Answer Grading"},
            "Citation Verification": {"Citation Verification"},
            "Cả hai": {"Answer Grading", "Citation Verification"},
        }.get(audit_choice, set())
        if audit_methods:
            st.caption("Grading và citation verification dùng chung tối đa một generation call. Tổ hợp đã chạy được cache trong lượt thử.")
        if audit_choice is None:
            st.info("Chọn audit để xem ngay kết quả kiểm tra.")
            return
        audit_cache = state.setdefault("audit_cache", {})
        answer_key = str(hash(state["answer"].get("answer", "")))
        audit_cache_key = f"{state.get('context_commit_key', '')}|{answer_key}|{audit_choice}"
        if audit_cache_key not in audit_cache:
            if audit_methods and not api_key:
                st.warning("Nhập Gemini API key để chạy audit.")
            else:
                try:
                    progress_box = st.empty()
                    with st.spinner("Gemini đang kiểm tra câu trả lời và trích dẫn…" if audit_methods else "Đang bỏ qua audit…"):
                        audit_cache[audit_cache_key] = rag_lab.audit_answer(
                            state["question"],
                            state["answer"].get("answer", ""),
                            state["context"],
                            state["sources"],
                            api_key,
                            state["meter"],
                            audit_methods,
                            generation_rpm=int(generation_rpm),
                            progress=progress_box.caption,
                        ) if audit_methods else {}
                    state["audit_cache"] = audit_cache
                    st.session_state["stepwise-rag-lab"] = state
                except Exception as error:
                    st.error(f"Audit thất bại: {error}")
        audit_preview = audit_cache.get(audit_cache_key)
        if audit_preview is not None:
            st.markdown("#### Preview kết quả audit")
            if audit_preview:
                _render_lab_audit(audit_preview)
            else:
                st.caption("Đã chọn bỏ qua audit.")
            _render_lab_meter(state["meter"])
            if st.button(
                "Chốt kết quả audit và hoàn tất lượt thử",
                type="primary",
                key=f"lab-continue-audit-{state['id']}",
            ):
                state.update(
                    {
                        "audit_methods": audit_methods,
                        "audit": audit_preview,
                        "step": "complete",
                    }
                )
                st.session_state["stepwise-rag-lab"] = state
                st.rerun()
        return

    if step == "complete":
        st.success("Đã hoàn tất toàn bộ pipeline thử nghiệm.")
        meter = state["meter"].as_dict()
        st.caption(
            f"API calls: {meter['api_calls']} · Embedding: {meter['embedding_calls']} · "
            f"Generation: {meter['generation_calls']} · Tổng token: {meter['total_tokens']}"
        )
        if state.get("audit"):
            _render_lab_audit(state["audit"])
        st.markdown("#### Lưu tài liệu đã thử vào kho chat")
        st.caption(
            "Đây là bước riêng: sẽ tạo embedding lần nữa trong Chroma bền vững của kho để tài liệu dùng được trong chat."
        )
        if state.get("saved_document_id"):
            st.success(state.get("saved_message", "Đã lưu vào kho."))
        elif st.button("Lưu file với cách chia đã chọn vào kho thật", type="primary", key=f"lab-save-{state['id']}"):
            try:
                progress_box = st.empty()
                document_id, message = ingest_upload(
                    kb_id,
                    upload["file_name"],
                    upload["contents"],
                    api_key,
                    chunk_size=int(upload["chunk_size"]),
                    chunk_overlap=int(upload["chunk_overlap"]),
                    chunk_strategy=state["selected_strategy"],
                    prepared_sections=upload["sections"],
                    prepared_parser_name=upload["parser"],
                    embedding_rpm=int(embedding_rpm),
                    embedding_batch_size=EMBEDDING_BATCH_SIZE,
                    progress=progress_box.caption,
                )
                state.update({"saved_document_id": document_id, "saved_message": message})
                st.session_state["stepwise-rag-lab"] = state
                st.success(message)
            except Exception as error:
                st.error(f"Không lưu được file vào kho: {error}")


def _render_lab_meter(meter: RequestMeter) -> None:
    values = meter.as_dict()
    cols = st.columns(6)
    cols[0].metric("API calls", values["api_calls"])
    cols[1].metric("Embedding", values["embedding_calls"])
    cols[2].metric("Generation", values["generation_calls"])
    cols[3].metric("Token vào", values["input_tokens"])
    cols[4].metric("Token ra", values["output_tokens"])
    cols[5].metric("Tổng token", values["total_tokens"])
    if values["estimated_token_calls"]:
        st.caption(values["token_note"])


def _render_sources(sources: list[dict], message_key: str) -> None:
    if not sources:
        return
    deduped: list[dict] = []
    seen: set[str] = set()
    for source in sources:
        chunk_id = str(source.get("chunk_id", ""))
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            deduped.append(source)
    visible = deduped[:500]
    contents = db.get_chunk_contents([str(item.get("chunk_id", "")) for item in visible])
    with st.expander(f"📎 Nguồn đã dùng ({len(deduped)} chunk)", expanded=False):
        if len(deduped) > len(visible):
            st.caption("Đang hiển thị 500 nguồn đầu tiên; toàn bộ nguồn vẫn được lưu trong lịch sử.")
        labels = [_source_display_label(source, index) for index, source in enumerate(visible)]
        selected_index = st.selectbox(
            "Chọn chunk nguồn",
            options=list(range(len(visible))),
            format_func=lambda index: labels[index],
            key=f"source-select-{message_key}",
        )
        source = visible[selected_index]
        if source.get("distance") is not None:
            st.caption(f"Distance: {source['distance']}")
        if source.get("rerank_score") is not None:
            st.caption(f"Reranker score: {source['rerank_score']}")
        if source.get("rrf_score") is not None:
            st.caption(f"Hybrid RRF score: {source['rrf_score']}")
        st.code(
            contents.get(str(source.get("chunk_id", "")), "Chunk hiện không còn trong kho."),
            language=None,
        )
        context_used = source.get("context_used")
        original_chunk = contents.get(str(source.get("chunk_id", "")), "")
        if context_used and context_used != original_chunk:
            st.markdown("**Ngữ cảnh parent/đã nén thực sự gửi Gemini**")
            st.code(context_used, language=None)


def _render_diagnostics(diagnostics: dict, message_key: str) -> None:
    if not diagnostics:
        return
    progress_steps = diagnostics.get("progress_steps", [])
    with st.expander("🔬 Xem pipeline, nguồn và chi phí", expanded=False):
        if progress_steps:
            st.markdown("**Tiến trình xử lý**")
            for step in progress_steps:
                st.markdown(f"- {step}")
        else:
            st.caption(
                "Lượt hỏi cũ không lưu các mốc trạng thái; phần pipeline và nguồn bên dưới "
                "vẫn phản ánh chẩn đoán đã được lưu."
            )
        meter = diagnostics.get("request_meter", {})
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Thời gian", f"{diagnostics.get('latency_ms', 0)} ms")
        metric2.metric("Lượt gọi API", meter.get("api_calls", 0))
        metric3.metric("Token vào", meter.get("input_tokens", 0))
        metric4.metric("Token ra", meter.get("output_tokens", 0))
        st.caption(
            f"Pipeline: {diagnostics.get('pipeline', 'RAG cơ bản')} · "
            f"Embedding: {meter.get('embedding_calls', 0)} request · "
            f"Generation: {meter.get('generation_calls', 0)} request · "
            f"Tổng token: {meter.get('total_tokens', 0)}"
        )
        if meter.get("estimated_token_calls"):
            st.caption(meter.get("token_note", "Một phần token là ước tính vì API không trả usage metadata."))
        if meter.get("labels"):
            st.caption("Thứ tự request (có thể gồm retry): " + " → ".join(meter["labels"]))
        techniques = diagnostics.get("techniques", {})
        if techniques:
            st.markdown("**Trạng thái kỹ thuật trong lượt này**")
            st.write(
                " · ".join(
                    f"{_TECHNIQUE_LABELS.get(name, name)}: {'bật' if enabled else 'tắt'}"
                    for name, enabled in techniques.items()
                )
            )
        if diagnostics.get("ignored_advanced_flags"):
            st.caption(
                "Không áp dụng trong Multi-agent: "
                + ", ".join(diagnostics["ignored_advanced_flags"])
            )
        plan = diagnostics.get("plan", {})
        if plan:
            st.markdown("**Lập kế hoạch truy vấn**")
            st.write(f"Truy vấn sau rewrite: {plan.get('rewritten_query') or 'Không đổi'}")
            if plan.get("alternative_queries"):
                st.write("Multi-query:", plan["alternative_queries"])
            if plan.get("hypothetical_answer"):
                st.write(
                    "HyDE (giả thuyết chỉ dùng để tìm kiếm, không phải dữ kiện đã xác minh):",
                    plan["hypothetical_answer"],
                )
        retrieval = diagnostics.get("retrieval", {})
        retrieval_attempts = diagnostics.get("retrieval_attempts", [])
        context_grades = diagnostics.get("context_grades", [])
        if (
            techniques.get("corrective_retrieval")
            or int(diagnostics.get("retry_count", 0))
            or diagnostics.get("abstained_after_retrieval")
        ):
            st.markdown("**Corrective Retrieval · tối đa 1 lượt tìm lại**")
            for attempt in retrieval_attempts:
                st.write(
                    f"Lượt {attempt.get('attempt')}: `{attempt.get('query', '')}` · "
                    f"{attempt.get('candidate_count', 0)} ứng viên"
                )
                if attempt.get("top_labels"):
                    st.caption("Ứng viên top:")
                    for label in attempt["top_labels"]:
                        st.markdown(f"- {label}")
            for grade in context_grades:
                st.caption(
                    f"Đánh giá lượt {grade.get('attempt')}: "
                    f"{'đủ bằng chứng' if grade.get('sufficient') else 'chưa đủ bằng chứng'} · "
                    f"{grade.get('reason', '')} · context {grade.get('evidence_chars', 0):,} ký tự"
                )
                for label in grade.get("evidence_sources", []):
                    st.markdown(f"- Grader đã xem: {label}")
                if grade.get("error"):
                    st.warning(f"Context judge bị bỏ qua: {grade['error']}")
            if diagnostics.get("abstained_after_retrieval"):
                st.warning("Đã dừng trước bước sinh câu trả lời vì bằng chứng sau một lần tìm lại vẫn chưa đủ.")
        if retrieval:
            st.markdown("**Tìm kiếm và xếp hạng**")
            if retrieval.get("strategy") == "dense":
                st.caption(
                    f"Dense top-k: {retrieval.get('dense_candidate_count', 0)}/"
                    f"{retrieval.get('top_k', 0)} kết quả · BM25: không chạy · "
                    "Hợp nhất: không áp dụng trong pipeline cơ bản"
                )
            else:
                st.caption(
                    f"Dense: {retrieval.get('dense_candidate_count', 0)} ứng viên · "
                    f"BM25: {retrieval.get('bm25_candidate_count', 0)} · "
                    f"Hợp nhất: {retrieval.get('fused_candidate_count', 0)} · "
                    f"Pool rerank: {retrieval.get('candidate_pool_count', retrieval.get('fused_candidate_count', 0))} · "
                    f"Parent: {retrieval.get('parent_results', 0)} · "
                    f"Parent fallback: {retrieval.get('parent_fallbacks', 0)}"
                )
            dense_top = retrieval.get("dense_top_labels") or _readable_chunk_ids(retrieval.get("dense_top_ids", []))
            if dense_top:
                st.markdown("**Dense top (câu hỏi gốc)**")
                for label in dense_top:
                    st.markdown(f"- {label}")
            else:
                st.caption("Dense top: không có kết quả.")
            bm25_top = retrieval.get("bm25_top_labels") or _readable_chunk_ids(retrieval.get("bm25_top_ids", []))
            if retrieval.get("strategy") != "dense" and retrieval.get("hybrid_search"):
                if bm25_top:
                    st.markdown("**BM25 top (câu hỏi gốc)**")
                    for label in bm25_top:
                        st.markdown(f"- {label}")
                else:
                    st.caption("BM25 top: không có kết quả.")
            if retrieval.get("original_dense_anchor_labels"):
                st.markdown("**Dense gốc được giữ để tránh mất nguồn nền**")
                for label in retrieval["original_dense_anchor_labels"]:
                    st.markdown(f"- {label}")
            if retrieval.get("rrf_top_labels"):
                st.markdown("**Top sau hợp nhất RRF**")
                for label in retrieval["rrf_top_labels"]:
                    st.markdown(f"- {label}")
            if retrieval.get("reranked_top_labels"):
                st.markdown("**Top sau reranker**")
                for label in retrieval["reranked_top_labels"]:
                    st.markdown(f"- {label}")
            if retrieval.get("rerank_query"):
                st.caption(f"Câu dùng để rerank: {retrieval['rerank_query']}")
            if retrieval.get("queries_used"):
                st.write("Query embeddings:", retrieval["queries_used"])
            if retrieval.get("query_variant_labels"):
                st.markdown("**Trọng số query trong RRF**")
                for label in retrieval["query_variant_labels"]:
                    st.markdown(f"- {label}")
        coverage = diagnostics.get("document_coverage", {})
        if coverage:
            st.markdown("**Phạm vi tổng hợp toàn kho**")
            st.caption(
                f"Đã rà soát {coverage.get('ready_documents', 0)}/"
                f"{coverage.get('total_documents', coverage.get('ready_documents', 0))} file"
            )
            if coverage.get("not_ready_file_names"):
                st.caption("Chưa sẵn sàng: " + ", ".join(coverage["not_ready_file_names"]))
        compression = diagnostics.get("compression", {})
        if compression:
            st.markdown("**Nén ngữ cảnh**")
            st.caption(
                f"{compression.get('context_chars_before', 0):,} → "
                f"{compression.get('context_chars_after', 0):,} ký tự · "
                f"~{compression.get('context_tokens_estimated', 0):,} token ngữ cảnh · "
                f"ngân sách {compression.get('context_token_budget', 0):,} token · "
                f"{compression.get('dense_anchor_sources', 0)} nguồn Dense gốc được bảo toàn"
            )
            if compression.get("selected_context_labels"):
                st.markdown("**Nguồn thực sự gửi cho Gemini**")
                for label in compression["selected_context_labels"]:
                    st.markdown(f"- {label}")
        audit = diagnostics.get("audit", {})
        if audit:
            deterministic = audit.get("deterministic_citation_check", {})
            llm_audit = audit.get("llm_audit", {})
            st.markdown("**Chấm câu trả lời / kiểm tra trích dẫn**")
            st.caption(
                "Điểm grading đánh giá câu trả lời theo context đã gửi Gemini; "
                "điểm cao không chứng minh retrieval đã tìm đủ mọi đoạn liên quan trong kho."
            )
            if deterministic:
                st.write(
                    "Kiểm tra mã nguồn:",
                    "Đạt" if deterministic.get("passed") else "Cần xem lại",
                )
                if deterministic.get("invalid_refs"):
                    st.error(f"Mã nguồn không tồn tại: {deterministic['invalid_refs']}")
                if deterministic.get("claims_without_citation"):
                    st.warning("Có câu chưa gắn trích dẫn; xem danh sách trong chi tiết.")
            if llm_audit:
                st.write(
                    f"Điểm hỗ trợ: {llm_audit.get('grade', '?')}/5 · "
                    f"{llm_audit.get('grade_reason', '')}"
                )
                if llm_audit.get("unsupported_claims") or llm_audit.get("citation_issues"):
                    st.warning(
                        "Điểm cần kiểm tra: "
                        + "; ".join(
                            llm_audit.get("unsupported_claims", [])
                            + llm_audit.get("citation_issues", [])
                        )
                    )
        supervisor = diagnostics.get("supervisor", {})
        if supervisor:
            st.markdown("**Điều phối multi-agent**")
            st.caption(
                f"Supervisor chọn: {', '.join(supervisor.get('selected_agents', [])) or 'không có agent'} · "
                f"{supervisor.get('reason', '')}"
            )
            for agent in diagnostics.get("agents", []):
                with st.expander(agent.get("agent_name", agent.get("agent_id", "Specialist")), expanded=False):
                    st.write("Phát hiện:", agent.get("findings", []))
                    st.write("Chưa đủ dữ kiện:", agent.get("unknowns", []))
            critic = diagnostics.get("critic", {})
            if critic.get("enabled"):
                st.write("Critic:", "Đã kiểm tra" if critic.get("supported") else "Có điểm cần thận trọng")
                if critic.get("issues"):
                    st.warning(" · ".join(critic["issues"]))
            citation_check = diagnostics.get("citation_validation", {})
            if citation_check:
                st.write("Citation ID:", "Hợp lệ" if citation_check.get("passed") else "Có mã cần kiểm tra")
                if citation_check.get("invalid_refs"):
                    st.error(f"Citation không thuộc nguồn lượt này: {citation_check['invalid_refs']}")
            st.caption(
                f"Context được lặp qua specialist/synthesis/critic: "
                f"~{diagnostics.get('estimated_repeated_context_tokens', 0):,} token tổng hợp."
            )
        if diagnostics.get("voice_transcription_meter"):
            st.caption("Lượt gọi nhận dạng giọng nói đã được cộng vào tổng request/token ở trên.")


def _render_comparison(comparison: dict, message_key: str) -> None:
    if not comparison:
        return
    basic = comparison.get("basic", {})
    advanced = comparison.get("advanced", {})
    basic_diag = basic.get("diagnostics", {})
    advanced_diag = advanced.get("diagnostics", {})
    basic_meter = basic_diag.get("request_meter", {})
    advanced_meter = advanced_diag.get("request_meter", {})
    basic_ids = {item.get("chunk_id") for item in basic.get("sources", []) if item.get("chunk_id")}
    advanced_ids = {item.get("chunk_id") for item in advanced.get("sources", []) if item.get("chunk_id")}
    union = basic_ids | advanced_ids
    overlap = (len(basic_ids & advanced_ids) / len(union) * 100) if union else 0

    def show_sources(sources: list[dict], key: str) -> None:
        if not sources:
            return
        if not st.checkbox("Xem nội dung nguồn", key=f"{key}-show-sources"):
            return
        with st.container(border=True):
            st.markdown("**Nội dung nguồn**")
            contents_by_id = db.get_chunk_contents(
                [str(source.get("chunk_id", "")) for source in sources[:15]]
            )
            for source_index, source in enumerate(sources[:15], start=1):
                label = _source_display_label(source, source_index - 1)
                st.markdown(f"**{label}**")
                if source.get("rerank_score") is not None:
                    st.caption(f"Reranker score: {source['rerank_score']}")
                chunk_id = str(source.get("chunk_id", ""))
                text = source.get("context_used") or contents_by_id.get(
                    chunk_id, "Chunk hiện không còn trong kho."
                )
                st.code(text, language=None)
            if len(sources) > 15:
                st.caption("Chỉ hiển thị 15 nguồn đầu tiên.")

    with st.expander("⚖️ So sánh cơ bản và nâng cao", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown("**Cơ bản · vector top-k**")
            st.markdown(basic.get("answer", "Chưa có kết quả."))
            st.caption(
                f"{len(basic_ids)} nguồn · {basic_diag.get('latency_ms', 0)} ms · "
                f"{basic_meter.get('api_calls', 0)} API calls · "
                f"{basic_meter.get('total_tokens', 0)} token"
            )
            st.write("Chunk:", sorted(basic_ids))
            st.caption(
                "Nguồn: "
                + ("; ".join(
                    f"{source.get('file_name', 'Tài liệu')} · {_location(source.get('page_number'))}"
                    for source in basic.get("sources", [])
                ) or "không có")
            )
            show_sources(basic.get("sources", []), f"{message_key}-basic")
        with right:
            st.markdown("**Nâng cao · cấu hình đang chọn**")
            st.markdown(advanced.get("answer", "Chưa có kết quả."))
            st.caption(
                f"{len(advanced_ids)} nguồn · {advanced_diag.get('latency_ms', 0)} ms · "
                f"{advanced_meter.get('api_calls', 0)} API calls · "
                f"{advanced_meter.get('total_tokens', 0)} token"
            )
            st.write("Chunk:", sorted(advanced_ids))
            st.caption(
                "Nguồn: "
                + ("; ".join(
                    f"{source.get('file_name', 'Tài liệu')} · {_location(source.get('page_number'))}"
                    for source in advanced.get("sources", [])
                ) or "không có")
            )
            technique_states = advanced_diag.get("techniques", {})
            if technique_states:
                st.caption(
                    "Cấu hình: "
                    + " · ".join(
                        f"{_TECHNIQUE_LABELS.get(name, name)}: {'bật' if active else 'tắt'}"
                        for name, active in technique_states.items()
                    )
                )
            show_sources(advanced.get("sources", []), f"{message_key}-advanced")
        st.info(
            f"Độ trùng nguồn: {overlap:.0f}% · "
            f"Chênh lệch lượt gọi API: {advanced_meter.get('api_calls', 0) - basic_meter.get('api_calls', 0):+d} · "
            f"Chênh lệch token: {advanced_meter.get('total_tokens', 0) - basic_meter.get('total_tokens', 0):+d}"
        )
        if advanced_meter.get("estimated_token_calls") or basic_meter.get("estimated_token_calls"):
            st.caption("Một phần token so sánh được ước tính từ độ dài nội dung do API không trả usage metadata.")
        if comparison.get("note"):
            st.caption(comparison["note"])


def _run_basic_answer(
    kb_id: str,
    api_key: str,
    query: str,
    history: list[dict[str, str]],
    *,
    mode: str,
    top_k: int,
    embedding_rpm: int,
    generation_rpm: int,
    progress,
) -> dict:
    meter = RequestMeter()
    started = time.perf_counter()
    if progress:
        progress("Đang chạy RAG cơ bản: tìm nguồn liên quan hoặc đọc toàn kho theo phạm vi câu hỏi…")
    with trace_span(
        "rag.basic_pipeline",
        {"rag.kb_id": kb_id, "rag.mode": mode, "rag.top_k": top_k},
    ) as span:
        result = answer_question(
            kb_id,
            api_key,
            query,
            history,
            mode=mode,
            top_k=top_k,
            embedding_rpm=embedding_rpm,
            generation_rpm=generation_rpm,
            progress=progress,
            meter=meter,
        )
        if span:
            span.set_attribute("rag.source_count", len(result.get("sources", [])))
    if progress:
        progress("Đã hoàn tất truy xuất/tổng hợp và tạo câu trả lời.")
    result["diagnostics"] = {
        "pipeline": (
            "Danh mục kho · trả lời chính xác từ SQLite"
            if result.get("mode") == "inventory"
            else "Map-reduce · đọc toàn bộ tài liệu đã lập chỉ mục"
            if result.get("mode") == "all_documents"
            else "RAG cơ bản · tìm kiếm vector top-k"
        ),
        "advanced": False,
        "techniques": {},
        "retrieval": result.get("retrieval_diagnostics", {}),
        "document_coverage": result.get("document_coverage", {}),
        "request_meter": meter.as_dict(),
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }
    return result


def _progress_callback(status_box, progress_bar, index: int, total: int):
    counter = {"value": 0.02}

    def update(message: str) -> None:
        status_box.caption(message)
        match = re.search(r"(\d+)/(\d+) chunk", message)
        if match:
            current, count = int(match.group(1)), max(1, int(match.group(2)))
            counter["value"] = max(counter["value"], current / count)
        else:
            counter["value"] = min(0.9, counter["value"] + 0.02)
        progress_bar.progress(min(0.96, (index - 1 + counter["value"]) / max(1, total)))

    return update


def _new_conversation(kb_id: str) -> str:
    conversation_id = db.create_conversation(kb_id)
    st.session_state[f"active-conversation-{kb_id}"] = conversation_id
    st.session_state[f"conversation-picker-{kb_id}"] = conversation_id
    return conversation_id


def _conversation_picker_labels(conversations: list[dict]) -> dict[str, str]:
    """Return unique, stable selectbox labels even when titles are duplicated."""
    labels: dict[str, str] = {}
    for item in conversations:
        item_id = item["id"]
        suffix_length = min(8, len(item_id))
        while suffix_length < len(item_id) and any(
            other["id"] != item_id
            and other["title"] == item["title"]
            and other["id"].startswith(item_id[:suffix_length])
            for other in conversations
        ):
            suffix_length = min(suffix_length + 4, len(item_id))
        labels[item_id] = f"{item['title']} · {item_id[:suffix_length]}"
    return labels


def _document_picker_labels(documents: list[dict]) -> dict[str, str]:
    """Disambiguate same-named files in a KB while preserving their stored names."""
    normalized_names = {
        item["id"]: " ".join(
            unicodedata.normalize("NFKC", str(item.get("file_name", "")))
            .casefold()
            .split()
        )
        for item in documents
    }
    counts: dict[str, int] = {}
    for key in normalized_names.values():
        counts[key] = counts.get(key, 0) + 1
    labels: dict[str, str] = {}
    for item in documents:
        item_id = str(item["id"])
        name = str(item.get("file_name", "Tài liệu"))
        if counts[normalized_names[item_id]] <= 1:
            labels[item_id] = name
            continue
        suffix_length = min(6, len(item_id))
        while suffix_length < len(item_id) and any(
            other_id != item_id
            and normalized_names[other_id] == normalized_names[item_id]
            and other_id.startswith(item_id[:suffix_length])
            for other_id in normalized_names
        ):
            suffix_length = min(suffix_length + 2, len(item_id))
        labels[item_id] = f"{name} · {item_id[:suffix_length]}"
    return labels


def _queue_follow_up(query: str, kb_id: str, conversation_id: str) -> None:
    # Keep the exact destination with the prompt; do not infer it from a
    # global picker after Streamlit reruns the page.
    st.session_state["pending_query_request"] = {
        "query": query,
        "kb_id": kb_id,
        "conversation_id": conversation_id,
    }
    st.session_state["active_kb_id"] = kb_id
    st.session_state[f"active-conversation-{kb_id}"] = conversation_id
    st.session_state[f"conversation-picker-{kb_id}"] = conversation_id


def _queue_typed_query() -> None:
    """Pin a typed prompt to the selected conversation before the app reruns."""
    query = str(st.session_state.get("chat-input", "")).strip()
    kb_id = st.session_state.get("active_kb_id")
    if not query or not kb_id:
        return

    conversation_id = st.session_state.get(
        f"conversation-picker-{kb_id}"
    ) or st.session_state.get(f"active-conversation-{kb_id}")
    if conversation_id:
        _queue_follow_up(query, kb_id, conversation_id)


def _queue_voice_query(kb_id: str, conversation_id: str) -> None:
    query = str(st.session_state.get("voice-transcribed-text", "")).strip()
    if not query:
        return
    st.session_state["pending_query_request"] = {
        "query": query,
        "kb_id": st.session_state.get("voice-transcribed-kb", kb_id),
        "conversation_id": conversation_id,
        "voice_meter": st.session_state.get("voice-transcribed-meter", {}),
    }
    # Callbacks run before widgets are rendered again, so it is safe to reset
    # the transcript field here while preserving its conversation destination.
    st.session_state["voice-transcribed-text"] = ""
    for key in ("voice-transcribed-meter", "voice-transcribed-kb", "voice-transcribed-conversation"):
        st.session_state.pop(key, None)


def _create_kb_callback() -> None:
    name = st.session_state.get("new-kb-name", "").strip()
    if name:
        try:
            kb_id = db.create_knowledge_base(name)
            st.session_state["active_kb_id"] = kb_id
            st.session_state["kb-create-error"] = ""
        except ValueError as error:
            st.session_state["kb-create-error"] = str(error)
    else:
        st.session_state["kb-create-error"] = "Nhập tên kho trước khi tạo."


def _delete_kb_callback(kb_id: str) -> None:
    try:
        knowledge_base = db.get_knowledge_base(kb_id)
        delete_knowledge_base(kb_id)
        st.session_state["active_kb_id"] = "default"
        st.session_state["kb-delete-error"] = ""
        st.session_state["kb-delete-success"] = (
            f"Đã xóa kho “{knowledge_base['name']}” cùng file, vector và hội thoại bên trong."
            if knowledge_base
            else "Đã xóa kho tài liệu."
        )
    except Exception as error:
        st.session_state["kb-delete-error"] = str(error)


def _apply_rag_preset(preset: str) -> None:
    """Set advanced switches to a reusable low-cost or full-demo preset."""
    cheap = {
        "adv-parent-child": True,
        "adv-hybrid": True,
        "adv-reranker": True,
        "adv-query-rewrite": False,
        "adv-multi-query": False,
        "adv-hyde": False,
        "adv-compress": True,
        "adv-grade": False,
        "adv-citations": False,
        "adv-corrective-retrieval": False,
        "adv-langgraph": True,
    }
    full = {key: True for key in cheap}
    for key, enabled in (cheap if preset == "Tiết kiệm API" else full).items():
        st.session_state[key] = enabled


app_surface = st.radio(
    "Khu vực làm việc",
    ["💬 Chat", "🧭 Kịch bản sản phẩm", "🧪 Phòng thử nghiệm từng bước", "📊 RAG Evaluation"],
    horizontal=True,
    key="app-surface",
)
if app_surface == "💬 Chat":
    st.title("💬 Chat với kho tài liệu")
    st.caption("Thêm tài liệu vào kho, rồi hỏi đáp bằng ngôn ngữ tự nhiên. Câu trả lời có trích dẫn file và trang.")

with st.sidebar:
    st.header("📚 Kho tài liệu")
    knowledge_bases = db.list_knowledge_bases()
    if not knowledge_bases:
        db.init_db()
        knowledge_bases = db.list_knowledge_bases()
    kb_names = {item["id"]: item["name"] for item in knowledge_bases}
    kb_display_names = {
        item["id"]: item.get("display_name", item["name"])
        for item in knowledge_bases
    }
    if st.session_state.get("active_kb_id") not in kb_names:
        st.session_state["active_kb_id"] = "default" if "default" in kb_names else next(iter(kb_names))
    kb_id = st.selectbox(
        "Đang chat trong kho",
        options=list(kb_names),
        format_func=lambda item_id: kb_display_names[item_id],
        key="active_kb_id",
    )

    with st.form("create-knowledge-base", clear_on_submit=True):
        st.text_input("Tạo kho mới", placeholder="Ví dụ: Quy trình nhân sự", key="new-kb-name")
        st.form_submit_button("＋ Tạo kho", on_click=_create_kb_callback, use_container_width=True)
    if st.session_state.get("kb-create-error"):
        st.error(st.session_state["kb-create-error"])
    if st.session_state.get("kb-delete-success"):
        st.success(st.session_state.pop("kb-delete-success"))

    with st.expander("Quản lý kho", expanded=False):
        selected_kb_name = kb_names.get(kb_id, "Kho tài liệu")
        with st.form(f"rename-kb-form-{kb_id}"):
            st.text_input("Đổi tên kho", value=selected_kb_name, key=f"rename-kb-name-{kb_id}")
            if st.form_submit_button("Lưu tên kho", use_container_width=True):
                new_name = st.session_state.get(f"rename-kb-name-{kb_id}", "").strip()
                try:
                    db.rename_knowledge_base(kb_id, new_name)
                    st.session_state.pop(f"kb-rename-error-{kb_id}", None)
                    st.rerun()
                except ValueError as error:
                    st.session_state[f"kb-rename-error-{kb_id}"] = str(error)
        if st.session_state.get(f"kb-rename-error-{kb_id}"):
            st.error(st.session_state.pop(f"kb-rename-error-{kb_id}"))
        if kb_id == "default":
            st.caption("Kho mặc định luôn được giữ lại để ứng dụng có kho dự phòng.")
        else:
            confirm_kb_delete = st.checkbox(
                f"Tôi muốn xóa kho “{selected_kb_name}” cùng file và hội thoại bên trong.",
                key=f"confirm-delete-kb-{kb_id}",
            )
            if st.button(
                "🗑️ Xóa kho này",
                key=f"delete-kb-{kb_id}",
                disabled=not confirm_kb_delete,
                use_container_width=True,
                on_click=_delete_kb_callback,
                args=(kb_id,),
            ):
                st.rerun()
        if st.session_state.get("kb-delete-error"):
            st.error(st.session_state["kb-delete-error"])

    st.divider()
    st.subheader("💬 Hội thoại")
    conversation_key = f"conversation-picker-{kb_id}"
    active_key = f"active-conversation-{kb_id}"
    conversations = db.list_conversations(kb_id)
    conversation_ids = [item["id"] for item in conversations]
    if not conversation_ids:
        first_conversation = _new_conversation(kb_id)
        conversations = db.list_conversations(kb_id)
        conversation_ids = [item["id"] for item in conversations]
        st.session_state[conversation_key] = first_conversation
    if st.session_state.get(conversation_key) not in conversation_ids:
        preferred = st.session_state.get(active_key)
        st.session_state[conversation_key] = preferred if preferred in conversation_ids else conversation_ids[0]
    if st.session_state.get(active_key) not in conversation_ids:
        st.session_state[active_key] = st.session_state[conversation_key]

    # Streamlit serializes selectbox values using their formatted label. Titles
    # are user-editable and may repeat, so include a stable ID suffix to ensure
    # each option maps back to exactly one conversation.
    conversation_labels = _conversation_picker_labels(conversations)
    conversation_id = st.selectbox(
        "Lịch sử gần đây",
        options=conversation_ids,
        format_func=lambda item_id: conversation_labels.get(item_id, "Cuộc trò chuyện"),
        key=conversation_key,
        label_visibility="collapsed",
    )
    st.session_state[active_key] = conversation_id
    st.button(
        "＋ Cuộc trò chuyện mới",
        use_container_width=True,
        key=f"new-chat-{kb_id}",
        on_click=_new_conversation,
        args=(kb_id,),
    )

    current_conversation = next(item for item in conversations if item["id"] == conversation_id)
    with st.expander("Đổi tên / xóa hội thoại", expanded=False):
        with st.form(f"rename-chat-form-{conversation_id}"):
            st.text_input(
                "Tên cuộc trò chuyện",
                value=current_conversation["title"],
                key=f"rename-chat-name-{conversation_id}",
            )
            if st.form_submit_button("Lưu tên", use_container_width=True):
                title = st.session_state.get(f"rename-chat-name-{conversation_id}", "").strip()
                if not title:
                    st.error("Tên hội thoại không được để trống.")
                else:
                    db.rename_conversation(conversation_id, title)
                    st.rerun()
        confirm_chat_delete = st.checkbox(
            "Tôi xác nhận xóa hội thoại này và toàn bộ tin nhắn.",
            key=f"confirm-delete-chat-{conversation_id}",
        )
        if st.button(
            "🗑️ Xóa hội thoại",
            disabled=not confirm_chat_delete,
            key=f"delete-chat-{conversation_id}",
            use_container_width=True,
        ):
            db.delete_conversation(conversation_id)
            st.rerun()

    with st.expander("Kết nối AI & tùy chọn nâng cao", expanded=False):
        api_key = st.text_input(
            "Gemini API key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            key="gemini-api-key",
            help="Có thể lưu GEMINI_API_KEY trong file .env để dùng lại.",
        ).strip()
        if not api_key:
            st.warning("Cần API key Gemini để nhập tài liệu và hỏi đáp.")
        answer_mode = st.selectbox(
            "Chế độ trả lời",
            ["Cơ bản", "Nâng cao", "Multi-agent", "So sánh cơ bản và nâng cao"],
            index=1,
            key="answer-pipeline-mode-v2",
            help="Multi-agent có Supervisor, specialist, synthesis và Critic; tốn nhiều lượt Gemini hơn RAG đơn. So sánh chạy hai pipeline cùng câu hỏi.",
        )
        agent_max_workers = 2
        agent_include_critic = True
        if answer_mode == "Multi-agent":
            with st.expander("Tùy chọn nhóm AI", expanded=False):
                agent_max_workers = st.number_input(
                    "Số chuyên viên tối đa", min_value=1, max_value=3, value=2, step=1,
                    key="chat-agent-max-workers",
                )
                agent_include_critic = st.checkbox(
                    "Bật Critic / kiểm tra citation (thêm 1 generation call)",
                    value=True,
                    key="chat-agent-critic",
                )
                st.caption(
                    f"Ước tính lượt này: 1 embedding batch + tối đa "
                    f"{1 + int(agent_max_workers) + 1 + int(agent_include_critic)} generation calls. "
                    "Giảm số specialist hoặc tắt Critic để tiết kiệm token."
                )
                st.caption(
                    "Multi-agent dùng Parent-Child, Hybrid/BM25, reranker và nén context theo các công tắc tương ứng. "
                    "Query Rewrite, HyDE, Corrective Retrieval và Answer Grading không chạy trong pipeline này."
                )
        chat_mode = st.selectbox(
            "Phạm vi tìm kiếm",
            ["Tự động", "Hỏi đáp chi tiết", "Tổng hợp toàn kho"],
            help="Chế độ tự động dùng truy xuất nhanh; câu hỏi rộng sẽ đọc riêng từng file trong toàn kho.",
            key="chat-mode",
        )
        if chat_mode == "Tổng hợp toàn kho":
            st.info(
                "Câu hỏi trong phạm vi này luôn đọc mọi tài liệu đã lập chỉ mục, kể cả khi đang chọn Nâng cao. "
                "Yêu cầu đếm file được trả trực tiếp từ danh mục kho."
            )
        elif chat_mode == "Tự động":
            st.caption(
                "Câu hỏi tóm tắt toàn kho hoặc hỏi file nào chứa thông tin sẽ tự đọc mọi tài liệu đã lập chỉ mục, "
                "kể cả khi đang chọn Nâng cao. Câu hỏi thông thường vẫn dùng pipeline đã chọn."
            )
        else:
            st.caption(
                "Hỏi đáp chi tiết dùng top-k cho câu hỏi thông thường; câu yêu cầu rõ toàn bộ kho sẽ tự chuyển "
                "sang đọc mọi file đã lập chỉ mục."
            )
        top_k = st.number_input("Số chunk truy xuất", min_value=2, max_value=15, value=6, step=1, key="top-k")
        st.markdown("**Bật/tắt kỹ thuật nâng cao**")
        preset_col1, preset_col2 = st.columns(2)
        preset_col1.button(
            "⚡ Tiết kiệm API",
            key="rag-preset-low-cost",
            help="Giữ retrieval hybrid, reranker local, parent-child, nén cục bộ và LangGraph; tắt planner/audit gọi Gemini thêm.",
            on_click=_apply_rag_preset,
            args=("Tiết kiệm API",),
            use_container_width=True,
        )
        preset_col2.button(
            "🧪 Bật mọi kỹ thuật",
            key="rag-preset-full",
            help="Bật planner, audit và Corrective Retrieval; có thể thêm tối đa 2 lượt Gemini cùng một lần retrieval.",
            on_click=_apply_rag_preset,
            args=("Đầy đủ",),
            use_container_width=True,
        )
        feature_flags = {
            "parent_child": st.checkbox("Parent-Child chunking", value=True, key="adv-parent-child"),
            "hybrid_search": st.checkbox("Hybrid Search (vector + BM25)", value=True, key="adv-hybrid"),
            "reranker": st.checkbox("Reranker đa ngôn ngữ local", value=True, key="adv-reranker"),
            "query_rewrite": st.checkbox("Query Rewrite", value=False, key="adv-query-rewrite"),
            "multi_query": st.checkbox("Multi-query (tối đa 2 câu phụ)", value=False, key="adv-multi-query"),
            "hyde": st.checkbox("HyDE", value=False, key="adv-hyde"),
            "context_compression": st.checkbox("Nén ngữ cảnh trích xuất", value=True, key="adv-compress"),
            "answer_grading": st.checkbox("Chấm mức độ hỗ trợ của câu trả lời", value=False, key="adv-grade"),
            "citation_verification": st.checkbox("Xác minh trích dẫn", value=False, key="adv-citations"),
            "corrective_retrieval": st.checkbox(
                "Tự đánh giá nguồn và tìm lại tối đa 1 lần",
                value=False,
                key="adv-corrective-retrieval",
                help="Thêm 1–2 lượt Gemini để chấm độ đủ của bằng chứng và tạo truy vấn sửa. Nếu vẫn thiếu, app từ chối sinh câu trả lời.",
            ),
        }
        use_langgraph = st.checkbox("Điều phối bằng LangGraph", value=True, key="adv-langgraph")
        feature_flags["langgraph"] = use_langgraph
        st.caption(
            "Mặc định tối ưu tốc độ: dùng tìm kiếm hybrid và các bước local; planner, audit và corrective retrieval tắt cho đến khi cần demo."
        )
        settings_a, settings_b = st.columns(2)
        with settings_a:
            chunk_size = st.number_input(
                "Độ dài chunk", min_value=300, max_value=3000, value=900, step=100, key="chunk-size"
            )
            embedding_rpm = st.number_input(
                "Embedding RPM",
                min_value=1,
                max_value=600,
                value=min(600, EMBEDDING_RPM_LIMIT),
                step=1,
                key="embedding-rpm",
            )
        with settings_b:
            chunk_overlap = st.number_input(
                "Chunk overlap", min_value=0, max_value=800, value=150, step=25, key="chunk-overlap"
            )
            generation_rpm = st.number_input(
                "Generation RPM",
                min_value=1,
                max_value=600,
                value=min(600, GENERATION_RPM_LIMIT),
                step=1,
                key="generation-rpm",
            )
        max_chunks = st.number_input(
            "Số chunk tối đa xem mỗi lần", min_value=10, max_value=200, value=50, step=10, key="max-chunks"
        )
        candidate_k = st.number_input(
            "Ứng viên trước rerank", min_value=8, max_value=50, value=24, step=4, key="candidate-k"
        )
        context_token_budget = st.number_input(
            "Ngân sách ngữ cảnh gửi Gemini (token)",
            min_value=500,
            max_value=12000,
            value=4000,
            step=500,
            key="context-token-budget",
        )
        if chunk_overlap >= chunk_size:
            st.warning("Chunk overlap phải nhỏ hơn độ dài chunk.")
        st.caption(
            f"Tối đa {EMBEDDING_BATCH_SIZE} chunk/request embedding; tự giãn nhịp và retry tối đa "
            f"{int(os.getenv('GEMINI_MAX_RETRIES', '5'))} lần."
        )
        st.markdown("**Nếu gặp 429 / Peak requests per minute**")
        st.markdown(
            """
            - App gom nhiều chunk vào một request, tuần tự hóa tốc độ gọi và retry có backoff.
            - Nếu vẫn chạm giới hạn, hạ RPM xuống thấp hơn quota thực trong AI Studio.
            - Quota tính theo project và model; API key khác của cùng project cũng dùng chung quota.
            - Quota ngày, billing hoặc spend cap cần chờ reset hoặc điều chỉnh tier/billing.
            - Nếu embedding lỗi, chunk đã trích xuất được giữ lại để lập chỉ mục lại.
            """
        )
        st.markdown("[Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)")
        planner_enabled = any(
            feature_flags[name] for name in ("query_rewrite", "multi_query", "hyde")
        )
        audit_enabled = feature_flags["answer_grading"] or feature_flags["citation_verification"]
        corrective_calls = 2 * int(feature_flags["corrective_retrieval"])
        advanced_generation_calls = 1 + int(planner_enabled) + int(audit_enabled) + corrective_calls
        if answer_mode == "So sánh cơ bản và nâng cao":
            st.info(
                f"Ước tính trước khi hỏi: khoảng {1 + advanced_generation_calls} generation call "
                f"và tối đa {2 + int(feature_flags['corrective_retrieval'])} embedding call "
                "(một baseline + tối đa hai batch nâng cao nếu có corrective retrieval). "
                "Multi-query/HyDE dùng chung một planner; grading/citation dùng chung một audit."
            )
        elif answer_mode == "Nâng cao":
            st.caption(
                f"Ước tính tối đa: {advanced_generation_calls} generation call và "
                f"{1 + int(feature_flags['corrective_retrieval'])} batch embedding cho lượt hỏi; "
                "reranker, BM25 và nén ngữ cảnh chạy local. Corrective Retrieval có thể thêm tối đa "
                "2 generation call và 1 batch embedding; bật riêng khi cần demo hoặc xử lý truy xuất yếu."
            )
        if feature_flags["reranker"]:
            st.caption("Reranker đa ngôn ngữ được tải về một lần khi chạy nâng cao lần đầu.")
        with st.expander("RAG Observability · Phoenix local", expanded=False):
            if OBSERVABILITY_STATUS.get("enabled"):
                st.success(
                    f"Cấu hình trace đã bật · {OBSERVABILITY_STATUS.get('project_name')} · "
                    f"{OBSERVABILITY_STATUS.get('endpoint')}"
                )
            elif OBSERVABILITY_STATUS.get("error"):
                st.warning(OBSERVABILITY_STATUS["message"])
                st.caption(OBSERVABILITY_STATUS["error"])
            else:
                st.caption(OBSERVABILITY_STATUS["message"])
            st.caption(
                "Phoenix thu trace LLM và span retrieval/generation khi được bật; mặc định tracing tắt. "
                "Gemini vẫn được gọi qua API của Google."
            )

    if "api_key" not in locals():
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if "chat_mode" not in locals():
        chat_mode = "Tự động"
    if "top_k" not in locals():
        top_k = 6
    if "chunk_size" not in locals():
        chunk_size = 900
    if "chunk_overlap" not in locals():
        chunk_overlap = 150
    if "max_chunks" not in locals():
        max_chunks = 50
    if "embedding_rpm" not in locals():
        embedding_rpm = EMBEDDING_RPM_LIMIT
    if "generation_rpm" not in locals():
        generation_rpm = GENERATION_RPM_LIMIT


if app_surface == "🧪 Phòng thử nghiệm từng bước":
    _render_stepwise_lab(
        kb_id,
        kb_display_names.get(kb_id, "Kho tài liệu"),
        api_key,
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        max_chunks=int(max_chunks),
        candidate_k=int(candidate_k) if "candidate_k" in locals() else 24,
        top_k=int(top_k),
        context_token_budget=int(context_token_budget) if "context_token_budget" in locals() else 4000,
        embedding_rpm=int(embedding_rpm),
        generation_rpm=int(generation_rpm),
    )
    st.stop()

if app_surface == "🧭 Kịch bản sản phẩm":
    render_scenario_workbench(
        kb_id=kb_id,
        kb_name=kb_display_names.get(kb_id, "Kho tài liệu"),
        api_key=api_key,
        embedding_rpm=int(embedding_rpm),
        generation_rpm=int(generation_rpm),
        top_k=int(top_k),
        candidate_k=int(candidate_k) if "candidate_k" in locals() else 24,
        context_token_budget=int(context_token_budget) if "context_token_budget" in locals() else 4000,
    )
    st.stop()

if app_surface == "📊 RAG Evaluation":
    render_evaluation_workbench(
        root_dir=ROOT_DIR,
        kb_id=kb_id,
        kb_name=kb_display_names.get(kb_id, "Kho tài liệu"),
        api_key=api_key,
        feature_flags=feature_flags if "feature_flags" in locals() else {},
        top_k=int(top_k),
        candidate_k=int(candidate_k) if "candidate_k" in locals() else 24,
        context_token_budget=int(context_token_budget) if "context_token_budget" in locals() else 4000,
        embedding_rpm=int(embedding_rpm),
        generation_rpm=int(generation_rpm),
        use_langgraph=bool(use_langgraph) if "use_langgraph" in locals() else True,
        basic_runner=_run_basic_answer,
        advanced_runner=run_advanced_pipeline,
        multi_agent_runner=run_multi_agent_pipeline,
    )
    st.stop()


toolbar_left, toolbar_right = st.columns([5, 1])
with toolbar_left:
    st.markdown(f"#### {kb_display_names.get(kb_id, 'Kho tài liệu')}")
    st.caption(current_conversation["title"])
with toolbar_right:
    st.button(
        "＋ Chat mới",
        key=f"main-new-chat-{kb_id}",
        use_container_width=True,
        on_click=_new_conversation,
        args=(kb_id,),
    )

_render_rag_explainer()

with st.popover("📎 Thêm tài liệu", use_container_width=False):
    st.caption(
        f"Thêm file vào kho đang chọn. Tối đa {MAX_UPLOAD_MB} MB mỗi file; ảnh scan và file khó đọc sẽ tự chuyển OCR/Docling. "
        f"Cách chia hiện tại: {'Parent-Child' if feature_flags.get('parent_child', True) else 'Chunk thường'}."
    )
    with st.form(f"upload-form-{kb_id}", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Chọn một hoặc nhiều file",
            type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
            key=f"uploader-{kb_id}",
        )
        upload_clicked = st.form_submit_button(
            "Thêm vào kho", type="primary", use_container_width=True
        )
    if upload_clicked:
        if not uploaded_files:
            st.info("Chọn ít nhất một tài liệu.")
        elif not api_key:
            st.error("Mở “Kết nối AI & tùy chọn nâng cao” ở thanh bên và nhập Gemini API key.")
        elif chunk_overlap >= chunk_size:
            st.error("Chunk overlap phải nhỏ hơn độ dài chunk.")
        else:
            for file_index, uploaded in enumerate(uploaded_files, start=1):
                st.markdown(f"**{file_index}/{len(uploaded_files)} · {uploaded.name}**")
                bar = st.progress(0.02)
                status_box = st.empty()
                callback = _progress_callback(status_box, bar, file_index, len(uploaded_files))
                try:
                    _document_id, result_message = ingest_upload(
                        kb_id,
                        uploaded.name,
                        uploaded.getvalue(),
                        api_key,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                        chunk_strategy=(
                            "parent_child" if feature_flags.get("parent_child", True) else "regular"
                        ),
                        embedding_rpm=int(embedding_rpm),
                        embedding_batch_size=EMBEDDING_BATCH_SIZE,
                        progress=callback,
                    )
                    status_box.success(result_message)
                    bar.progress(1.0)
                except Exception as error:
                    status_box.error(str(error))
            st.rerun()

documents = db.list_documents(kb_id)
ready_count = sum(item["status"] == "Sẵn sàng" for item in documents)
total_chunks = sum(item["chunk_count"] for item in documents if item["status"] == "Sẵn sàng")
if not documents:
    st.info("Kho này chưa có tài liệu. Chọn **📎 Thêm tài liệu** để bắt đầu.")
elif ready_count == 0:
    st.warning("Tài liệu đã được thêm nhưng chưa lập chỉ mục thành công. Mở mục **Tài liệu trong kho** để xem lỗi hoặc thử lại.")

messages = db.get_messages(conversation_id)
if not messages:
    st.markdown("\n")
    st.markdown("### Bạn muốn tìm gì trong kho tài liệu này?")
    st.caption("Hãy đặt câu hỏi bên dưới. Ứng dụng sẽ tìm đoạn liên quan và dẫn nguồn để bạn kiểm tra.")

for message_index, message in enumerate(messages):
    role = message["role"] if message["role"] in {"user", "assistant"} else "assistant"
    with st.chat_message(role):
        st.markdown(message["content"])
        if role == "assistant":
            speech_text = re.sub(r"\[S\d+\]", "", str(message["content"]))
            speech_text = re.sub(r"[#*_`~]", "", speech_text).strip()
            _render_speech_button(speech_text, str(message.get("id", message_index)))
            saved_steps = message.get("diagnostics", {}).get("progress_steps", [])
            if saved_steps:
                st.caption(f"✓ {saved_steps[-1]}")
            elif message.get("diagnostics"):
                st.caption("ℹ️ Lượt hỏi cũ này chưa lưu tiến trình xử lý từng bước.")
            _render_sources(message.get("sources", []), f"history-{message_index}")
            _render_diagnostics(message.get("diagnostics", {}), str(message.get("id", message_index)))
            _render_comparison(message.get("comparison", {}), str(message.get("id", message_index)))
            suggestions = message.get("suggestions", [])
            if suggestions:
                st.caption("Bạn có thể hỏi tiếp:")
                for suggestion_index, suggestion in enumerate(suggestions[:3]):
                    st.button(
                        suggestion,
                        key=f"suggestion-{message['id']}-{suggestion_index}",
                        on_click=_queue_follow_up,
                        args=(suggestion, kb_id, conversation_id),
                    )
            if message["content"].startswith("Mình chưa hoàn tất được yêu cầu này."):
                previous_user = next(
                    (
                        item
                        for item in reversed(messages[:message_index])
                        if item["role"] == "user"
                    ),
                    None,
                )
                if previous_user:
                    st.button(
                        "↻ Hỏi lại sau khi kiểm tra quota",
                        key=f"retry-question-{message['id']}",
                        on_click=_queue_follow_up,
                        args=(previous_user["content"], kb_id, conversation_id),
                    )

with st.expander("🎙️ Hỏi bằng giọng nói", expanded=False):
    st.caption("Ghi âm tiếng Việt, xem lại bản chép lời, rồi xác nhận để gửi câu hỏi.")
    audio_recording = st.audio_input(
        "Ghi âm câu hỏi", sample_rate=16000,
        key=f"chat-audio-input-{kb_id}-{conversation_id}",
    )
    if st.button(
        "Chuyển giọng nói thành văn bản",
        key=f"transcribe-audio-{kb_id}-{conversation_id}",
        disabled=not audio_recording or not api_key,
    ):
        voice_meter = RequestMeter()
        try:
            with st.spinner("Gemini đang nhận dạng lời nói…"):
                transcript = transcribe_audio(
                    api_key,
                    audio_recording.getvalue(),
                    getattr(audio_recording, "type", "audio/wav") or "audio/wav",
                    rpm_limit=int(generation_rpm),
                    meter=voice_meter,
                )["text"]
            st.session_state["voice-transcribed-text"] = transcript
            voice_meter_data = voice_meter.as_dict()
            if voice_meter_data.get("estimated_token_calls"):
                voice_meter_data["token_note"] = (
                    str(voice_meter_data.get("token_note", ""))
                    + " Ước tính từ độ dài prompt không tính chính xác token của waveform audio."
                )
            st.session_state["voice-transcribed-meter"] = voice_meter_data
            st.session_state["voice-transcribed-kb"] = kb_id
            st.session_state["voice-transcribed-conversation"] = conversation_id
        except (GeminiRequestError, ValueError, RuntimeError) as error:
            st.error(str(error))
    if st.session_state.get("voice-transcribed-conversation") == conversation_id:
        st.text_area(
            "Bản chép lời (có thể sửa trước khi gửi)",
            key="voice-transcribed-text",
            height=80,
        )
        st.button(
            "Dùng câu hỏi này trong cuộc hội thoại",
            key=f"send-transcript-{conversation_id}",
            type="primary",
            disabled=not str(st.session_state.get("voice-transcribed-text", "")).strip(),
            on_click=_queue_voice_query,
            args=(kb_id, conversation_id),
        )

pending_request = st.session_state.pop("pending_query_request", None)
typed_query = st.chat_input(
    "Hỏi về nội dung trong kho tài liệu…",
    key="chat-input",
    on_submit=_queue_typed_query,
)
query = pending_request["query"] if pending_request else typed_query
target_kb_id = pending_request["kb_id"] if pending_request else kb_id
target_conversation_id = pending_request["conversation_id"] if pending_request else conversation_id
voice_meter_data = pending_request.get("voice_meter", {}) if pending_request else {}
if query:
    inventory_query = is_document_count_question(query)
    inventory_list_query = is_document_list_question(query)
    corpus_query = should_use_all_documents(query, chat_mode)
    local_inventory_query = inventory_query or inventory_list_query
    if not api_key and not local_inventory_query:
        st.error("Mở “Kết nối AI & tùy chọn nâng cao” ở thanh bên và nhập Gemini API key.")
    else:
        history = db.get_recent_turns(target_conversation_id, limit=6)
        db.save_message(target_conversation_id, "user", query)
        status_box = st.empty()
        progress_bar = st.progress(0.02)
        progress_steps: list[str] = []

        def progress(message: str) -> None:
            cleaned = str(message).strip()
            if cleaned and (not progress_steps or progress_steps[-1] != cleaned):
                progress_steps.append(cleaned)
                if len(progress_steps) > 100:
                    del progress_steps[:-100]
            status_box.caption(message)

        try:
            comparison_data = {}
            if local_inventory_query or corpus_query:
                result = _run_basic_answer(
                    target_kb_id,
                    api_key,
                    query,
                    history,
                    mode="Tổng hợp toàn kho" if corpus_query else chat_mode,
                    top_k=int(top_k),
                    embedding_rpm=int(embedding_rpm),
                    generation_rpm=int(generation_rpm),
                    progress=progress,
                )
            elif answer_mode == "Cơ bản":
                result = _run_basic_answer(
                    target_kb_id,
                    api_key,
                    query,
                    history,
                    mode=chat_mode,
                    top_k=int(top_k),
                    embedding_rpm=int(embedding_rpm),
                    generation_rpm=int(generation_rpm),
                    progress=progress,
                )
            elif answer_mode == "Multi-agent":
                result = run_multi_agent_pipeline(
                    target_kb_id,
                    api_key,
                    query,
                    history,
                    flags=feature_flags,
                    top_k=int(top_k),
                    candidate_k=int(candidate_k),
                    context_token_budget=int(context_token_budget),
                    embedding_rpm=int(embedding_rpm),
                    generation_rpm=int(generation_rpm),
                    max_agents=int(agent_max_workers),
                    include_critic=bool(agent_include_critic),
                    progress=progress,
                )
            else:
                advanced_result = run_advanced_pipeline(
                    target_kb_id,
                    api_key,
                    query,
                    history,
                    flags=feature_flags,
                    top_k=int(top_k),
                    candidate_k=int(candidate_k),
                    context_token_budget=int(context_token_budget),
                    embedding_rpm=int(embedding_rpm),
                    generation_rpm=int(generation_rpm),
                    use_langgraph=bool(use_langgraph),
                    progress=progress,
                )
                if answer_mode == "So sánh cơ bản và nâng cao":
                    basic_result = _run_basic_answer(
                        target_kb_id,
                        api_key,
                        query,
                        history,
                        mode="Hỏi đáp chi tiết",
                        top_k=int(top_k),
                        embedding_rpm=int(embedding_rpm),
                        generation_rpm=int(generation_rpm),
                        progress=progress,
                    )
                    result = advanced_result
                    comparison_data = {
                        "basic": {
                            "answer": basic_result["answer"],
                            "sources": basic_result.get("sources", []),
                            "diagnostics": basic_result.get("diagnostics", {}),
                        },
                        "advanced": {
                            "answer": advanced_result["answer"],
                            "sources": advanced_result.get("sources", []),
                            "diagnostics": advanced_result.get("diagnostics", {}),
                        },
                        "note": "So sánh ép cả hai pipeline dùng top-k trên cùng câu hỏi. Yêu cầu toàn kho được xử lý bằng map-reduce riêng để không bỏ sót tài liệu.",
                    }
                else:
                    result = advanced_result
            if comparison_data:
                progress("Đã chạy xong hai pipeline để so sánh kết quả.")
            diagnostics = result.setdefault("diagnostics", {})
            if voice_meter_data:
                diagnostics["voice_transcription_meter"] = voice_meter_data
                diagnostics["request_meter"] = _merge_request_meter(
                    diagnostics.get("request_meter", {}), voice_meter_data
                )
            diagnostics["progress_steps"] = list(progress_steps)
            db.save_message(
                target_conversation_id,
                "assistant",
                result["answer"],
                sources=result.get("sources", []),
                suggestions=result.get("suggestions", []),
                diagnostics=result.get("diagnostics", {}),
                comparison=comparison_data,
            )
            progress_bar.progress(1.0)
            status_box.success("Đã trả lời xong và lưu vào hội thoại này.")
        except (GeminiRequestError, ValueError, RuntimeError) as error:
            message = str(error)
            progress_steps.append("Lượt xử lý dừng do lỗi; xem nội dung lỗi bên dưới câu trả lời.")
            error_diagnostics = {
                "pipeline": "Multi-agent" if answer_mode == "Multi-agent" else "Nâng cao" if answer_mode != "Cơ bản" else "RAG cơ bản",
                "progress_steps": progress_steps[-100:],
            }
            if voice_meter_data:
                error_diagnostics["voice_transcription_meter"] = voice_meter_data
                error_diagnostics["request_meter"] = _merge_request_meter({}, voice_meter_data)
            db.save_message(
                target_conversation_id,
                "assistant",
                f"Mình chưa hoàn tất được yêu cầu này.\n\n{message}",
                diagnostics=error_diagnostics,
            )
            status_box.error(message)
        st.rerun()

with st.expander(f"🗂️ Tài liệu trong kho · {len(documents)} file · {total_chunks} chunk", expanded=False):
    if not documents:
        st.info("Chưa có tài liệu trong kho này.")
    else:
        stat1, stat2, stat3 = st.columns(3)
        stat1.metric("Tài liệu", len(documents))
        stat2.metric("Đã lập chỉ mục", ready_count)
        stat3.metric("Chunk có vector", total_chunks)
        document_ids = [item["id"] for item in documents]
        doc_names = {item["id"]: item["file_name"] for item in documents}
        doc_display_names = _document_picker_labels(documents)
        doc_selection_key = f"document-selection-{kb_id}"
        if st.session_state.get(doc_selection_key) not in document_ids:
            st.session_state[doc_selection_key] = document_ids[0]
        selected_doc_id = st.selectbox(
            "Chọn tài liệu",
            document_ids,
            format_func=lambda item_id: doc_display_names[item_id],
            key=doc_selection_key,
        )
        selected_doc = next(item for item in documents if item["id"] == selected_doc_id)
        info1, info2, info3, info4 = st.columns(4)
        info1.metric("Trạng thái", selected_doc["status"])
        info2.metric("Số chunk", selected_doc["chunk_count"])
        info3.metric("Parser", selected_doc.get("parser") or "Chưa xác định")
        info4.metric(
            "Cách chia",
            "Parent-Child" if selected_doc.get("chunk_strategy", "parent_child") == "parent_child" else "Chunk thường",
        )
        if selected_doc.get("error"):
            st.error(_short_error(selected_doc["error"]))

        st.markdown("#### Đổi tên tài liệu")
        with st.form(f"rename-document-form-{selected_doc_id}"):
            st.text_input(
                "Tên hiển thị",
                value=selected_doc["file_name"],
                key=f"rename-document-name-{selected_doc_id}",
            )
            if st.form_submit_button("Lưu tên tài liệu"):
                new_file_name = st.session_state.get(
                    f"rename-document-name-{selected_doc_id}", ""
                ).strip()
                if not new_file_name:
                    st.error("Tên file không được để trống.")
                else:
                    new_file_name = Path(new_file_name.replace("\\", "/")).name
                    new_file_name = re.sub(r'[<>:"|?*]', "_", new_file_name).strip(" .")
                    if not new_file_name:
                        st.error("Tên file không hợp lệ.")
                    else:
                        db.rename_document_record(selected_doc_id, new_file_name[:160])
                        st.rerun()

        if selected_doc["status"] == "Lỗi" and api_key:
            retry_label = (
                "Thử lập chỉ mục lại (dùng chunk đã lưu)"
                if selected_doc["chunk_count"]
                else "Thử đọc file gốc và lập chỉ mục lại"
            )
            if st.button(retry_label, key=f"retry-{selected_doc_id}"):
                progress_box = st.empty()
                try:
                    retry_document_index(
                        selected_doc_id,
                        api_key,
                        embedding_rpm=int(embedding_rpm),
                        embedding_batch_size=EMBEDDING_BATCH_SIZE,
                        progress=progress_box.caption,
                    )
                    st.success("Đã tạo vector lại thành công.")
                    st.rerun()
                except Exception as error:
                    st.error(str(error))

        stored_path = Path(selected_doc["stored_path"])
        if stored_path.exists():
            st.download_button(
                "Tải file gốc",
                data=stored_path.read_bytes(),
                file_name=selected_doc["file_name"],
                key=f"download-{selected_doc_id}",
            )

        chunk_total = int(selected_doc["chunk_count"] or 0)
        if chunk_total:
            offset_key = f"chunk-offset-{selected_doc_id}"
            if offset_key not in st.session_state:
                st.session_state[offset_key] = 0
            st.session_state[offset_key] = min(
                int(st.session_state[offset_key]), max(0, chunk_total - 1)
            )
            offset = st.number_input(
                "Bắt đầu từ vị trí chunk",
                min_value=0,
                max_value=max(1, chunk_total - 1),
                step=max(1, int(max_chunks)),
                key=offset_key,
            )
            if st.button("🔍 Xem chunk đã nhập", key=f"inspect-{selected_doc_id}"):
                st.session_state["chunks_doc_view"] = selected_doc_id
            if st.session_state.get("chunks_doc_view") == selected_doc_id:
                rows = db.list_chunks(selected_doc_id, limit=int(max_chunks), offset=int(offset))
                st.caption(f"Hiển thị tối đa {len(rows)} / {chunk_total} chunk.")
                row_ids = [row["id"] for row in rows]
                row_by_id = {row["id"]: row for row in rows}
                if row_ids:
                    chunk_selection_key = f"chunk-selection-{selected_doc_id}"
                    if st.session_state.get(chunk_selection_key) not in row_ids:
                        st.session_state[chunk_selection_key] = row_ids[0]
                    selected_chunk_id = st.selectbox(
                        "Chọn chunk để xem",
                        row_ids,
                        format_func=lambda row_id: (
                            f"Chunk {row_by_id[row_id]['chunk_index'] + 1} · "
                            f"{_location(row_by_id[row_id]['page_number'])} · "
                            f"{row_by_id[row_id]['content_type']}"
                        ),
                        key=chunk_selection_key,
                    )
                    st.code(row_by_id[selected_chunk_id]["content"], language=None)
                else:
                    st.warning("Không tìm thấy nội dung chunk đã lưu cho tài liệu này.")

        confirm_document_delete = st.checkbox(
            f"Tôi xác nhận xóa “{selected_doc['file_name']}” khỏi kho.",
            key=f"confirm-delete-document-{selected_doc_id}",
        )
        if st.button(
            "🗑️ Xóa tài liệu",
            key=f"delete-document-{selected_doc_id}",
            disabled=not confirm_document_delete,
        ):
            delete_document(selected_doc_id, api_key or None)
            st.session_state.pop("chunks_doc_view", None)
            st.rerun()

        st.markdown("#### Danh sách và lỗi nhập liệu")
        for item in documents:
            status_icon = "✅" if item["status"] == "Sẵn sàng" else "⚠️"
            st.markdown(
                f"{status_icon} **{item['file_name']}** — {item['status']} — "
                f"{item['chunk_count']} chunk — {item.get('parser') or 'parser chưa xác định'}"
            )
            if item.get("error"):
                st.caption(_short_error(item["error"]))

with st.expander("ℹ️ Lưu trữ và xử lý dữ liệu trên máy", expanded=False):
    st.caption(f"Thư mục dữ liệu local: {ROOT_DIR / 'data'}")
    st.caption("File, SQLite và Chroma được lưu trên máy; Gemini API nhận câu hỏi và các chunk được truy xuất để tạo câu trả lời.")
