from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Any

from flashrank import Ranker, RerankRequest
from google.genai import types
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from . import db
from .config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_RPM_LIMIT,
    MODEL_CACHE_DIR,
    RERANKER_MODEL,
    GENERATION_RPM_LIMIT,
)
from .quota import (
    GeminiEmbeddings,
    RequestMeter,
    generate_answer_json,
    generate_structured_json,
)
from .rag import _answer_prompt, _collection_name, _conversation_context, vector_store
from .workflow import RAGState, run_rag_graph

_BM25_CACHE: dict[str, tuple[list[dict[str, Any]], BM25Okapi]] = {}


def clear_search_cache(kb_id: str | None = None) -> None:
    if kb_id is None:
        _BM25_CACHE.clear()
    else:
        _BM25_CACHE.pop(kb_id, None)


@lru_cache(maxsize=2)
def _ranker(model_name: str) -> Ranker:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return Ranker(model_name=model_name, cache_dir=str(MODEL_CACHE_DIR), max_length=256)


def _terms(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)


def _subject_anchors(question: str, history: list[dict[str, str]]) -> list[str]:
    """Recover capitalized subjects from the current and recent user questions."""
    user_texts = [question]
    user_texts.extend(
        str(item.get("content", ""))
        for item in reversed(history[-6:])
        if item.get("role") == "user"
    )
    cue = re.compile(r"\b(?:của|về|cho|ứng viên|nhân viên|người)\s+([^\n,.;:!?]{1,80})", re.I)
    for text in user_texts:
        anchors: list[str] = []
        seen: set[str] = set()
        for match in cue.finditer(text):
            words: list[str] = []
            for raw_word in match.group(1).split():
                word = raw_word.strip("()[]{}\"'`“”‘’")
                first_letter = next((char for char in word if char.isalpha()), "")
                if not first_letter or not first_letter.isupper():
                    break
                words.append(word.rstrip("'’"))
                if len(words) == 4:
                    break
            anchor = " ".join(words).strip()
            normalized = anchor.casefold()
            if anchor and normalized not in seen:
                seen.add(normalized)
                anchors.append(anchor)
        if anchors:
            return anchors[-3:]
    return []


def _add_subject_anchors(query: str, anchors: list[str]) -> str:
    missing = []
    for anchor in anchors:
        explicit_subject = re.search(
            rf"\b(?:của|về|cho|ứng viên|nhân viên)\s+{re.escape(anchor)}\b",
            query,
            flags=re.IGNORECASE,
        )
        if not explicit_subject:
            missing.append(anchor)
    if not missing:
        return query
    return f"{query.rstrip(' ?.!')} (đối tượng: {', '.join(missing)})"


def _bm25_index(kb_id: str) -> tuple[list[dict[str, Any]], BM25Okapi] | None:
    cached = _BM25_CACHE.get(kb_id)
    if cached is not None:
        return cached
    rows = db.get_all_ready_chunks(kb_id)
    if not rows:
        return None
    tokenized = [_terms(row["content"]) for row in rows]
    index = (rows, BM25Okapi(tokenized))
    _BM25_CACHE[kb_id] = index
    return index


def _query_plan(
    question: str,
    history: list[dict[str, str]],
    api_key: str,
    flags: dict[str, bool],
    meter: RequestMeter,
    *,
    generation_rpm: int,
    progress=None,
) -> dict[str, Any]:
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "rewritten_query": types.Schema(type=types.Type.STRING),
            "alternative_queries": types.Schema(
                type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)
            ),
            "hypothetical_answer": types.Schema(type=types.Type.STRING),
        },
        required=["rewritten_query", "alternative_queries", "hypothetical_answer"],
    )
    requested = []
    if flags.get("query_rewrite"):
        requested.append("rewrite thành câu truy vấn độc lập, giữ nguyên ý định và thực thể")
    if flags.get("multi_query"):
        requested.append("tạo tối đa 2 cách hỏi khác nhau, không đổi ý nghĩa")
    if flags.get("hyde"):
        requested.append("tạo một câu trả lời giả định ngắn để tìm tài liệu tương tự")
    prompt = f"""Bạn là bước lập kế hoạch truy xuất cho chatbot hỏi đáp tài liệu.
Thực hiện đúng các tác vụ sau: {', '.join(requested)}.
Chỉ dùng lịch sử để hiểu đại từ/ngữ cảnh; không thêm giả định cá nhân.
Khi viết lại hoặc tạo truy vấn phụ, phải giữ nguyên tên người, tổ chức, mã số, thời gian và đối tượng cụ thể từ câu hỏi/lịch sử; không thay tên riêng bằng từ chung như "ứng viên" nếu đã biết tên.
Nếu chưa xác định được đối tượng từ lịch sử, giữ câu hỏi gốc thay vì tự suy đoán.
HyDE chỉ là văn bản giả định để tìm kiếm: không khẳng định bằng cấp, sự kiện hay thuộc tính cụ thể chưa được cung cấp; chỉ mô tả cách thông tin liên quan có thể xuất hiện trong tài liệu.
Giới hạn mỗi truy vấn 20 từ; đoạn giả định tối đa 80 từ. Nếu một trường không cần, trả chuỗi/danh sách rỗng.
Trả về JSON theo schema.

LỊCH SỬ:
{_conversation_context(history) or '(trống)'}

CÂU HỎI:
{question}
"""
    result = generate_structured_json(
        api_key,
        prompt,
        schema,
        label="Lập kế hoạch truy xuất nâng cao",
        rpm_limit=generation_rpm,
        progress=progress,
        meter=meter,
        output_tokens=350,
    )
    rewritten = str(result.get("rewritten_query", "")).strip() or question
    alternatives = result.get("alternative_queries", [])
    if not isinstance(alternatives, list):
        alternatives = []
    alternatives = [str(item).strip() for item in alternatives if str(item).strip()]
    hypothetical = str(result.get("hypothetical_answer", "")).strip()
    anchors = _subject_anchors(question, history)
    if flags.get("query_rewrite"):
        rewritten = _add_subject_anchors(rewritten, anchors)
    if flags.get("multi_query"):
        alternatives = [_add_subject_anchors(item, anchors) for item in alternatives]
    return {
        "rewritten_query": rewritten if flags.get("query_rewrite") else question,
        "alternative_queries": alternatives[:2] if flags.get("multi_query") else [],
        "hypothetical_answer": hypothetical if flags.get("hyde") else "",
    }


def _unique_queries(question: str, plan: dict[str, Any]) -> list[str]:
    values = [question]
    values.extend(plan.get("alternative_queries", []))
    if plan.get("rewritten_query") and plan["rewritten_query"] != question:
        values.append(plan["rewritten_query"])
    if plan.get("hypothetical_answer"):
        values.append(plan["hypothetical_answer"])
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = re.sub(r"\s+", " ", str(value)).strip()
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def _weighted_query_variants(question: str, plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the user's actual query dominant; HyDE is dense-only and low weight."""
    variants = [
        {"text": question, "kind": "original", "dense_weight": 2.5, "bm25_weight": 0.75}
    ]
    rewritten = str(plan.get("rewritten_query", "")).strip()
    if rewritten and rewritten.casefold() != question.strip().casefold():
        variants.append(
            {"text": rewritten, "kind": "rewrite", "dense_weight": 1.25, "bm25_weight": 0.5}
        )
    for alternative in plan.get("alternative_queries", []):
        text = str(alternative).strip()
        if text:
            variants.append(
                {"text": text, "kind": "multi_query", "dense_weight": 0.65, "bm25_weight": 0.25}
            )
    hypothetical = str(plan.get("hypothetical_answer", "")).strip()
    if hypothetical:
        variants.append(
            {"text": hypothetical, "kind": "hyde", "dense_weight": 0.3, "bm25_weight": 0.0}
        )

    deduplicated: dict[str, dict[str, Any]] = {}
    for variant in variants:
        text = re.sub(r"\s+", " ", variant["text"]).strip()
        if not text:
            continue
        key = text.casefold()
        existing = deduplicated.get(key)
        if existing:
            existing["dense_weight"] = max(existing["dense_weight"], variant["dense_weight"])
            existing["bm25_weight"] = max(existing["bm25_weight"], variant["bm25_weight"])
        else:
            deduplicated[key] = {**variant, "text": text}
    return list(deduplicated.values())


def _rank_candidates(
    kb_id: str,
    api_key: str,
    question: str,
    history: list[dict[str, str]],
    plan: dict[str, Any],
    flags: dict[str, bool],
    meter: RequestMeter,
    *,
    embedding_rpm: int,
    candidate_k: int,
    top_k: int,
    progress=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows = db.get_all_ready_chunks(kb_id)
    if not all_rows:
        raise ValueError("Kho này chưa có tài liệu đã nhập thành công.")
    row_by_id = {row["id"]: row for row in all_rows}
    anchors = _subject_anchors(question, history)
    search_plan = dict(plan)
    search_plan["rewritten_query"] = _add_subject_anchors(
        str(plan.get("rewritten_query") or question), anchors
    )
    query_variants = _weighted_query_variants(question, search_plan)
    query_strings = [item["text"] for item in query_variants]

    embedder = GeminiEmbeddings(
        api_key,
        batch_size=EMBEDDING_BATCH_SIZE,
        rpm_limit=embedding_rpm,
        progress=progress,
        meter=meter,
    )
    query_vectors = embedder.embed_queries(query_strings)
    store = vector_store(kb_id, api_key, embedding_rpm=embedding_rpm, progress=progress)
    if store._collection.count() == 0:
        raise ValueError("Kho này chưa có vector. Hãy lập chỉ mục tài liệu trước khi hỏi.")

    dense_k = min(max(candidate_k, 8), 50)
    reciprocal_scores: dict[str, float] = {}
    dense_seen: set[str] = set()
    dense_rankings: list[list[str]] = []
    for variant, vector in zip(query_variants, query_vectors):
        raw = store._collection.query(
            query_embeddings=[vector],
            n_results=min(dense_k, store._collection.count()),
            include=["metadatas", "distances"],
        )
        ids = (raw.get("ids") or [[]])[0]
        ranking = [chunk_id for chunk_id in ids if chunk_id in row_by_id]
        dense_rankings.append(ranking)
        for rank, chunk_id in enumerate(ranking, start=1):
            reciprocal_scores[chunk_id] = reciprocal_scores.get(chunk_id, 0.0) + (
                float(variant["dense_weight"]) / (60 + rank)
            )
            dense_seen.add(chunk_id)

    bm25_seen: set[str] = set()
    bm25_rankings: list[list[str]] = []
    if flags.get("hybrid_search"):
        cached = _bm25_index(kb_id)
        if cached:
            rows, bm25 = cached
            indexed_ids = [row["id"] for row in rows]
            for variant in query_variants:
                bm25_weight = float(variant["bm25_weight"])
                if bm25_weight <= 0:
                    continue
                scores = bm25.get_scores(_terms(variant["text"]))
                ranking_indexes = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:dense_k]
                ranking = [indexed_ids[index] for index in ranking_indexes if scores[index] > 0]
                bm25_rankings.append(ranking)
                for rank, chunk_id in enumerate(ranking, start=1):
                    reciprocal_scores[chunk_id] = reciprocal_scores.get(chunk_id, 0.0) + (
                        bm25_weight / (60 + rank)
                    )
                    bm25_seen.add(chunk_id)

    ranked_ids = sorted(reciprocal_scores, key=reciprocal_scores.__getitem__, reverse=True)
    rrf_candidate_ids = ranked_ids[:dense_k]
    # Keep a small baseline from the user's original Dense query in the pool
    # even when many generic expansions or HyDE results dominate RRF.
    anchor_count = min(max(3, top_k // 2), dense_k)
    original_dense_anchor_ids = dense_rankings[0][:anchor_count] if dense_rankings else []
    candidate_ids = list(dict.fromkeys(original_dense_anchor_ids + rrf_candidate_ids))[:dense_k]
    candidates: list[dict[str, Any]] = []
    rrf_rank_by_id = {chunk_id: rank for rank, chunk_id in enumerate(ranked_ids, start=1)}
    original_dense_rank_by_id = {
        chunk_id: rank for rank, chunk_id in enumerate(dense_rankings[0] if dense_rankings else [], start=1)
    }
    anchor_id_set = set(original_dense_anchor_ids)
    for chunk_id in candidate_ids:
        row = row_by_id[chunk_id]
        candidates.append(
            {
                "id": chunk_id,
                "document": Document(
                    page_content=row["content"],
                    metadata={
                        **json_load_metadata(row.get("metadata_json")),
                        "chunk_id": chunk_id,
                        "document_id": row["document_id"],
                        "parent_chunk_id": row.get("parent_chunk_id"),
                        "file_name": row["file_name"],
                        "page_number": row["page_number"],
                        "content_type": row["content_type"],
                    },
                ),
                "row": row,
                "rrf_score": reciprocal_scores.get(chunk_id, 0.0),
                "rrf_rank": rrf_rank_by_id.get(chunk_id),
                "original_dense_rank": original_dense_rank_by_id.get(chunk_id),
                "is_dense_anchor": chunk_id in anchor_id_set,
                "rerank_score": None,
            }
        )

    if flags.get("reranker") and candidates:
        if progress:
            progress(f"Đang xếp hạng lại {len(candidates)} ứng viên bằng mô hình local…")
        try:
            ranker = _ranker(RERANKER_MODEL)
            passages = [
                {"id": item["id"], "text": item["document"].page_content, "meta": item["document"].metadata}
                for item in candidates
            ]
            ranked = ranker.rerank(
                RerankRequest(
                    query=search_plan["rewritten_query"],
                    passages=passages,
                )
            )
        except Exception as error:
            raise RuntimeError(
                "Không thể khởi tạo reranker local. Kiểm tra mạng lần đầu tải model hoặc tắt Reranker trong phần nâng cao."
            ) from error
        by_id = {item["id"]: item for item in candidates}
        reranked: list[dict[str, Any]] = []
        reranked_ids: set[str] = set()
        for rerank_rank, rank_item in enumerate(ranked, start=1):
            item = by_id.get(str(rank_item.get("id", "")))
            if item:
                item["rerank_score"] = float(rank_item.get("score", 0.0))
                item["rerank_rank"] = rerank_rank
                reranked.append(item)
                reranked_ids.add(item["id"])
        # A partial or malformed reranker response must not silently discard
        # original-query candidates from the retrieval pool.
        reranked.extend(item for item in candidates if item["id"] not in reranked_ids)
        if reranked:
            candidates = reranked

    parents = db.get_parent_chunks(
        [str(item["row"].get("parent_chunk_id") or "") for item in candidates]
    ) if flags.get("parent_child") else {}
    if flags.get("parent_child"):
        for item in candidates:
            parent_id = str(item["row"].get("parent_chunk_id") or "")
            parent = parents.get(parent_id)
            item["context"] = parent.get("content") if parent else item["document"].page_content
            item["parent_id"] = parent_id or None
            item["parent_index"] = parent.get("parent_index") if parent else None
            item["parent_page_number"] = parent.get("page_number") if parent else None
            item["parent_fallback"] = not bool(parent)
    else:
        for item in candidates:
            item["context"] = item["document"].page_content
            item["parent_id"] = None
            item["parent_fallback"] = False

    doc_names = db.get_document_names([item["row"]["document_id"] for item in candidates])
    for item in candidates:
        item["file_name"] = doc_names.get(item["row"]["document_id"], item["row"]["file_name"])

    def top_labels(chunk_ids: list[str]) -> list[str]:
        labels = []
        for chunk_id in chunk_ids:
            row = row_by_id.get(chunk_id)
            if not row:
                continue
            page = f"Trang {row['page_number']}" if row.get("page_number") else "Không rõ trang"
            labels.append(
                f"Chunk {int(row['chunk_index']) + 1} · {row['file_name']} · {page}"
            )
        return labels

    dense_top_ids = dense_rankings[0][:candidate_k] if dense_rankings else []
    bm25_top_ids = bm25_rankings[0][:candidate_k] if bm25_rankings else []

    def display_label(item: dict[str, Any], rank: int | None = None) -> str:
        row = item["row"]
        page = f"Trang {row['page_number']}" if row.get("page_number") else "Không rõ trang"
        label = f"Chunk {int(row['chunk_index']) + 1} · {item['file_name']} · {page}"
        if rank is not None:
            label = f"#{rank} · {label}"
        if item.get("rerank_score") is not None:
            label += f" · rerank {float(item['rerank_score']):.4f}"
        if item.get("original_dense_rank") is not None:
            label += f" · Dense gốc #{item['original_dense_rank']}"
        if item.get("rrf_rank") is not None:
            label += f" · RRF #{item['rrf_rank']}"
        return label

    candidate_by_id = {item["id"]: item for item in candidates}
    original_anchor_labels = [
        display_label(candidate_by_id[chunk_id])
        for chunk_id in original_dense_anchor_ids
        if chunk_id in candidate_by_id
    ]
    rrf_top_labels = [
        display_label(candidate_by_id[chunk_id], rank)
        for rank, chunk_id in enumerate(rrf_candidate_ids[:12], start=1)
        if chunk_id in candidate_by_id
    ]
    reranked_top_labels = [
        display_label(item, rank) for rank, item in enumerate(candidates[:12], start=1)
    ]

    diagnostics = {
        "queries_used": query_strings,
        "query_variant_labels": [
            f"{item['kind']} · Dense ×{item['dense_weight']:.2f} · "
            f"BM25 ×{item['bm25_weight']:.2f} · {item['text']}"
            for item in query_variants
        ],
        "dense_candidate_count": len(dense_seen),
        "bm25_candidate_count": len(bm25_seen),
        "fused_candidate_count": len(candidate_ids),
        "hybrid_search": bool(flags.get("hybrid_search")),
        "reranked": bool(flags.get("reranker")),
        "parent_results": sum(1 for item in candidates if item.get("parent_id")),
        "parent_fallbacks": sum(1 for item in candidates if item.get("parent_fallback")),
        "dense_top_ids": dense_top_ids,
        "bm25_top_ids": bm25_top_ids,
        "dense_top_labels": top_labels(dense_top_ids),
        "bm25_top_labels": top_labels(bm25_top_ids),
        "original_dense_anchor_labels": original_anchor_labels,
        "rrf_top_labels": rrf_top_labels,
        "reranked_top_labels": reranked_top_labels,
        "rerank_query": search_plan["rewritten_query"],
        "candidate_pool_count": len(candidates),
    }
    return candidates, diagnostics


def json_load_metadata(value: str | None) -> dict[str, Any]:
    import json

    try:
        result = json.loads(value or "{}")
        return result if isinstance(result, dict) else {}
    except (TypeError, ValueError):
        return {}


def _token_words(text: str) -> set[str]:
    return set(_terms(text))


_COMPRESSION_STOPWORDS = {
    "bao", "các", "cho", "có", "của", "cũng", "đã", "đang", "đây",
    "đó", "để", "được", "gì", "hay", "khi", "là", "mà", "mỗi",
    "một", "nào", "này", "như", "những", "ở", "ra", "sao", "theo",
    "thì", "trong", "từ", "và", "về", "với",
}


def _compress_passage(
    question: str,
    text: str,
    max_chars: int,
) -> str:
    if len(text) <= max_chars:
        return text
    compressed = _compress_matching_context(question, text, max_chars)
    if compressed:
        return compressed
    if "|" in text:
        return _balanced_table_excerpt(text, max_chars)
    return _balanced_text_excerpt(text, max_chars)


def _balanced_text_excerpt(text: str, max_chars: int) -> str:
    """Keep both ends of a passage when lexical compression finds no match."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n…[đã lược phần giữa]…\n"
    if max_chars <= len(marker):
        return text[:max_chars]
    remaining = max_chars - len(marker)
    head_size = (remaining + 1) // 2
    tail_size = remaining - head_size
    head = text[:head_size].rsplit(" ", 1)[0].rstrip() or text[:head_size]
    tail = text[-tail_size:].lstrip(" ") if tail_size else ""
    excerpt = f"{head}{marker}{tail}"
    return excerpt if len(excerpt) <= max_chars else excerpt[:max_chars]


def _markdown_table_parts(text: str) -> tuple[list[str], list[str], list[str]]:
    """Find a Markdown header even when a caption precedes the table."""
    lines = text.splitlines()
    for index in range(len(lines) - 1):
        if (
            "|" in lines[index]
            and "|" in lines[index + 1]
            and re.search(r"\|\s*:?-{2,}", lines[index + 1])
        ):
            return lines[:index], lines[index : index + 2], lines[index + 2 :]
    return lines[:1], [], lines[1:]


def _balanced_table_excerpt(text: str, max_chars: int) -> str:
    """Keep the table caption/header and representative rows from both ends."""
    if max_chars <= 0:
        return ""
    prefix, header, rows = _markdown_table_parts(text)
    fixed_lines = prefix + header
    fixed = "\n".join(fixed_lines)
    if len(fixed) >= max_chars or not rows:
        return _balanced_text_excerpt(text, max_chars)

    marker = "…[một số hàng đã lược]…"
    selected: set[int] = set()

    def render(row_indexes: set[int]) -> str:
        result_lines = list(fixed_lines)
        ordered = sorted(row_indexes)
        if ordered and ordered[0] > 0:
            result_lines.append(marker)
        previous: int | None = None
        for row_index in ordered:
            if previous is not None and row_index > previous + 1:
                result_lines.append(marker)
            result_lines.append(rows[row_index])
            previous = row_index
        if ordered and ordered[-1] < len(rows) - 1:
            result_lines.append(marker)
        return "\n".join(result_lines)

    order: list[int] = []
    left, right = 0, len(rows) - 1
    while left <= right:
        order.append(left)
        if right != left:
            order.append(right)
        left += 1
        right -= 1

    for row_index in order:
        candidate = selected | {row_index}
        if len(render(candidate)) <= max_chars:
            selected = candidate

    if not selected:
        return _balanced_text_excerpt(text, max_chars)
    return render(selected)


def _compress_matching_context(question: str, text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    query_terms = _token_words(question) - _COMPRESSION_STOPWORDS
    if not query_terms:
        return ""
    if "|" in text:
        prefix, header, rows = _markdown_table_parts(text)
        fixed_lines = prefix + header
        fixed = "\n".join(fixed_lines)
        if len(fixed) >= max_chars:
            return _balanced_text_excerpt(fixed, max_chars)
        scored = [
            (sum(1 for term in _terms(line) if term in query_terms), index, line)
            for index, line in enumerate(rows)
        ]
        ranked = sorted((item for item in scored if item[0] > 0), key=lambda item: (-item[0], item[1]))
        chosen: set[int] = set()

        def render(row_indexes: set[int]) -> str:
            ordered = sorted(row_indexes)
            result_lines = list(fixed_lines)
            if ordered and ordered[0] > 0:
                result_lines.append("…[một số hàng đã lược]…")
            previous: int | None = None
            for index in ordered:
                if previous is not None and index > previous + 1:
                    result_lines.append("…[một số hàng đã lược]…")
                result_lines.append(rows[index])
                previous = index
            if ordered and ordered[-1] < len(rows) - 1:
                result_lines.append("…[một số hàng đã lược]…")
            return "\n".join(result_lines)

        for _score, row_index, line in ranked:
            candidate = chosen | {row_index}
            if len(render(candidate)) <= max_chars:
                chosen = candidate
        if not chosen:
            return ""
        return render(chosen)

    segments = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    scored_segments = []
    for index, segment in enumerate(segments):
        overlap = sum(1 for term in _terms(segment) if term in query_terms)
        if overlap:
            scored_segments.append((overlap, index, segment.strip()))
    if not scored_segments:
        return ""
    keep = sorted(scored_segments, key=lambda item: (-item[0], item[1]))
    chosen: set[int] = set()

    def render(segment_indexes: set[int]) -> str:
        ordered = sorted(segment_indexes)
        parts: list[str] = []
        if ordered and ordered[0] > 0:
            parts.append("…[đã lược phần đầu]…")
        previous: int | None = None
        for index in ordered:
            if previous is not None and index > previous + 1:
                parts.append("…[đã lược phần giữa]…")
            parts.append(segments[index].strip())
            previous = index
        if ordered and ordered[-1] < len(segments) - 1:
            parts.append("…[đã lược phần cuối]…")
        return " ".join(parts)

    for _score, index, _segment in keep:
        candidate = chosen | {index}
        if len(render(candidate)) <= max_chars:
            chosen = candidate
    return render(chosen) if chosen else ""


def _build_context(
    question: str,
    candidates: list[dict[str, Any]],
    flags: dict[str, bool],
    *,
    top_k: int,
    context_token_budget: int,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_parents: set[str] = set()

    def add_candidate(item: dict[str, Any]) -> None:
        if len(selected) >= top_k:
            return
        parent_id = item.get("parent_id")
        if flags.get("parent_child") and parent_id and parent_id in used_parents:
            return
        if parent_id:
            used_parents.add(parent_id)
        selected.append(item)

    # Reserve up to two distinct source groups for the original Dense query.
    # This prevents expanded queries/reranking from replacing every baseline
    # hit while leaving the remaining slots to the advanced ranking stages.
    anchor_candidates = sorted(
        (item for item in candidates if item.get("is_dense_anchor")),
        key=lambda item: int(item.get("original_dense_rank") or 10**9),
    )
    anchor_target = min(2, top_k)
    for item in anchor_candidates:
        add_candidate(item)
        if len(selected) >= anchor_target or len(selected) >= top_k:
            break
    for item in candidates:
        add_candidate(item)
        if len(selected) >= top_k:
            break
    # Vietnamese and table-heavy text can tokenize more densely than English;
    # use a conservative chars/token ratio to reduce surprise overages.
    token_budget_chars = max(300, context_token_budget * 3)
    remaining_chars = token_budget_chars
    contexts: list[str] = []
    sources: list[dict[str, Any]] = []
    original_chars = 0
    compressed_chars = 0
    for index, item in enumerate(selected, start=1):
        row = item["row"]
        text = str(item["context"])
        original_chars += len(text)
        available = max(0, remaining_chars - 300)
        if available <= 0:
            break
        is_anchor = bool(item.get("is_dense_anchor"))
        content = (
            _compress_passage(question, text, available)
            if flags.get("context_compression") and not is_anchor
            else text
        )
        if len(content) > available:
            content = (
                _balanced_table_excerpt(content, available)
                if "|" in content
                else _balanced_text_excerpt(content, available)
            )
        compressed_chars += len(content)
        ref = f"S{len(sources) + 1}"
        location = f"trang {row['page_number']}" if row.get("page_number") else "không xác định trang"
        contexts.append(
            f"[{ref}] {item['file_name']} · {location} · chunk {row['chunk_index'] + 1}\n{content}"
        )
        sources.append(
            {
                "ref": ref,
                "chunk_id": row["id"],
                "chunk_index": int(row["chunk_index"]),
                "content_type": row.get("content_type"),
                "parent_chunk_id": item.get("parent_id"),
                "parent_index": item.get("parent_index"),
                "parent_page_number": item.get("parent_page_number"),
                "document_id": row["document_id"],
                "file_name": item["file_name"],
                "page_number": row["page_number"],
                "context_used": content,
                "distance": None,
                "rrf_score": round(float(item.get("rrf_score", 0)), 6),
                "rerank_score": item.get("rerank_score"),
                "rrf_rank": item.get("rrf_rank"),
                "rerank_rank": item.get("rerank_rank"),
                "original_dense_rank": item.get("original_dense_rank"),
                "is_dense_anchor": is_anchor,
            }
        )
        remaining_chars -= len(content) + 300
    selected_context_labels = []
    for source in sources:
        page_label = (
            f"Trang {source['page_number']}"
            if source.get("page_number")
            else "Không rõ trang"
        )
        label = (
            f"Chunk {source['chunk_index'] + 1} · {source['file_name']} · {page_label}"
        )
        if source.get("is_dense_anchor"):
            label += " · Dense gốc được giữ"
        selected_context_labels.append(label)
    return "\n\n".join(contexts), sources, {
        "context_token_budget": context_token_budget,
        "context_chars_before": original_chars,
        "context_chars_after": compressed_chars,
        "context_tokens_estimated": (compressed_chars + 2) // 3,
        "sources_sent": len(sources),
        "compression_enabled": bool(flags.get("context_compression")),
        "dense_anchor_sources": sum(bool(source.get("is_dense_anchor")) for source in sources),
        "selected_context_labels": selected_context_labels,
    }


def _audit_answer(
    question: str,
    answer: str,
    context: str,
    sources: list[dict[str, Any]],
    api_key: str,
    flags: dict[str, bool],
    meter: RequestMeter,
    *,
    generation_rpm: int,
    progress=None,
) -> dict[str, Any]:
    refs = [str(source["ref"]) for source in sources]
    cited = re.findall(r"\[([SC][A-Za-z0-9_-]*)\]", answer)
    invalid = sorted({ref for ref in cited if ref not in refs})
    missing = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s+", answer)
        if sentence.strip() and not re.search(r"\[[SC][A-Za-z0-9_-]*\]", sentence)
    ]
    result: dict[str, Any] = {}
    if flags.get("citation_verification"):
        result["deterministic_citation_check"] = {
            "cited_refs": sorted(set(cited)),
            "known_refs": refs,
            "invalid_refs": invalid,
            "claims_without_citation": missing[:10],
            "passed": not invalid and not missing,
        }
    if not (flags.get("answer_grading") or flags.get("citation_verification")):
        return result

    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "grade": types.Schema(type=types.Type.INTEGER),
            "grade_reason": types.Schema(type=types.Type.STRING),
            "unsupported_claims": types.Schema(
                type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)
            ),
            "citation_support": types.Schema(type=types.Type.BOOLEAN),
            "citation_issues": types.Schema(
                type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)
            ),
        },
        required=["grade", "grade_reason", "unsupported_claims", "citation_support", "citation_issues"],
    )
    active_checks = []
    if flags.get("answer_grading"):
        active_checks.append("chấm mức độ đúng, đầy đủ và được nguồn hỗ trợ từ 1 đến 5")
    if flags.get("citation_verification"):
        active_checks.append("xác minh mã trích dẫn tồn tại và nội dung hỗ trợ khẳng định")
    prompt = f"""Đánh giá độc lập câu trả lời RAG theo đúng các đoạn nguồn đã truy xuất bên dưới.
Chỉ thực hiện các kiểm tra sau: {', '.join(active_checks)}.
Thang điểm tối đa 5:
5 = đầy đủ, đúng và mọi khẳng định quan trọng có nguồn hỗ trợ; 1 = phần lớn không được hỗ trợ.
Chỉ công nhận mã nguồn có trong danh sách. Không bổ sung thông tin ngoài tài liệu.
Trả tối đa 5 vấn đề ngắn, JSON đúng schema.

CÂU HỎI: {question}
MÃ NGUỒN HỢP LỆ: {', '.join(refs)}
CÂU TRẢ LỜI:
{answer[:6000]}

NGỮ CẢNH ĐÃ GỬI CHO MÔ HÌNH:
{context[:18000]}
"""
    if progress:
        progress("Đang chấm câu trả lời và kiểm tra mức hỗ trợ của nguồn…")
    llm_review = generate_structured_json(
        api_key,
        prompt,
        schema,
        label="Kiểm tra chất lượng và trích dẫn",
        rpm_limit=generation_rpm,
        progress=progress,
        meter=meter,
        output_tokens=250,
    )
    audited: dict[str, Any] = {}
    if flags.get("answer_grading"):
        try:
            grade = max(1, min(5, int(llm_review.get("grade", 1))))
        except (TypeError, ValueError):
            grade = 1
        grade_reason = str(llm_review.get("grade_reason", ""))[:500]
        unsupported_claims = [
            str(item) for item in llm_review.get("unsupported_claims", [])
        ][:5]
        audited.update(
            {
                "grade": grade,
                "grade_reason": grade_reason,
                "unsupported_claims": unsupported_claims[:5],
            }
        )
    if flags.get("citation_verification"):
        audited.update(
            {
                "citation_support": bool(llm_review.get("citation_support", False)),
                "citation_issues": [str(item) for item in llm_review.get("citation_issues", [])][:5],
            }
        )
    result["llm_audit"] = audited
    return result


def run_advanced_pipeline(
    kb_id: str,
    api_key: str,
    question: str,
    history: list[dict[str, str]],
    *,
    flags: dict[str, bool],
    top_k: int = 6,
    candidate_k: int = 24,
    context_token_budget: int = 4000,
    embedding_rpm: int = EMBEDDING_RPM_LIMIT,
    generation_rpm: int = GENERATION_RPM_LIMIT,
    use_langgraph: bool = True,
    progress=None,
) -> dict[str, Any]:
    meter = RequestMeter()
    started = time.perf_counter()
    needs_plan = any(flags.get(name) for name in ("query_rewrite", "multi_query", "hyde"))
    needs_audit = flags.get("answer_grading") or flags.get("citation_verification")
    initial_state: RAGState = {
        "needs_plan": bool(needs_plan),
        "needs_audit": bool(needs_audit),
        "query": question,
        "history": history,
        "plan": {},
        "retrieved": [],
        "context": "",
        "answer": {},
        "diagnostics": {},
        "settings": {
            "kb_id": kb_id,
            "api_key": api_key,
            "flags": flags,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "context_token_budget": context_token_budget,
            "embedding_rpm": embedding_rpm,
            "generation_rpm": generation_rpm,
            "meter": meter,
            "progress": progress,
        },
    }

    def plan_node(state: RAGState) -> dict[str, Any]:
        if progress:
            if needs_plan:
                progress("Đang lập kế hoạch câu hỏi (Rewrite / Multi-query / HyDE theo các công tắc đã bật)…")
            else:
                progress("Planner đã tắt; dùng nguyên câu hỏi để tìm kiếm.")
        return {
            "plan": _query_plan(
                question,
                history,
                api_key,
                flags,
                meter,
                generation_rpm=generation_rpm,
                progress=progress,
            )
        }

    def retrieve_node(state: RAGState) -> dict[str, Any]:
        if progress:
            if not needs_plan:
                progress("Planner đã tắt; giữ nguyên câu hỏi gốc cho bước tìm kiếm.")
            progress("Đang tìm ứng viên trong Chroma; BM25 sẽ chạy nếu Hybrid Search đang bật…")
        candidates, retrieval_diagnostics = _rank_candidates(
            kb_id,
            api_key,
            question,
            history,
            state.get("plan", {}),
            flags,
            meter,
            embedding_rpm=embedding_rpm,
            candidate_k=candidate_k,
            top_k=top_k,
            progress=progress,
        )
        if progress:
            progress(f"Đã hợp nhất và xếp hạng {len(candidates)} ứng viên; đang dựng context…")
        return {"retrieved": candidates, "diagnostics": {"retrieval": retrieval_diagnostics}}

    def compress_node(state: RAGState) -> dict[str, Any]:
        context_question = _add_subject_anchors(
            str(state.get("plan", {}).get("rewritten_query") or question),
            _subject_anchors(question, history),
        )
        context, sources, compression_diagnostics = _build_context(
            context_question,
            state.get("retrieved", []),
            flags,
            top_k=top_k,
            context_token_budget=context_token_budget,
        )
        if progress:
            progress(
                f"Đã chuẩn bị context gồm {len(sources)} nguồn "
                f"({compression_diagnostics.get('context_chars_after', len(context)):,} ký tự); "
                "đang tạo câu trả lời…"
            )
        return {
            "context": context,
            "sources": sources,
            "diagnostics": {
                **state.get("diagnostics", {}),
                "compression": compression_diagnostics,
            },
        }

    def answer_node(state: RAGState) -> dict[str, Any]:
        if not state.get("context"):
            raise ValueError("Không tìm thấy đoạn tài liệu phù hợp để trả lời.")
        prompt_question = str(state.get("plan", {}).get("rewritten_query") or question)
        context = str(state["context"])
        # Keep one recent turn so the answer model can resolve conversational
        # references; the prompt still requires evidence from current sources.
        answer_history = history[-2:] if flags.get("query_rewrite") else history[-4:]
        answer = generate_answer_json(
            api_key,
            _answer_prompt(prompt_question, context, answer_history),
            rpm_limit=generation_rpm,
            progress=progress,
            meter=meter,
        )
        if progress:
            progress("Gemini đã tạo câu trả lời từ context và nguồn đã chọn.")
        return {"answer": answer}

    def audit_node(state: RAGState) -> dict[str, Any]:
        return {
            "audit": _audit_answer(
                question,
                str(state.get("answer", {}).get("answer", "")),
                str(state.get("context", "")),
                state.get("sources", []),
                api_key,
                flags,
                meter,
                generation_rpm=generation_rpm,
                progress=progress,
            )
        }

    if use_langgraph:
        final_state = run_rag_graph(
            initial_state,
            plan_node=plan_node,
            retrieve_node=retrieve_node,
            compress_node=compress_node,
            answer_node=answer_node,
            audit_node=audit_node,
        )
    else:
        final_state = initial_state
        if needs_plan:
            final_state.update(plan_node(final_state))
        final_state.update(retrieve_node(final_state))
        final_state.update(compress_node(final_state))
        final_state.update(answer_node(final_state))
        if needs_audit:
            final_state.update(audit_node(final_state))

    if progress:
        progress("Đã hoàn tất pipeline nâng cao và các bước kiểm tra đang bật.")
    diagnostics = {
        "pipeline": "LangGraph" if use_langgraph else "Python tuần tự",
        "advanced": True,
        "techniques": {name: bool(flags.get(name)) for name in flags},
        "plan": final_state.get("plan", {}),
        **final_state.get("diagnostics", {}),
        "audit": final_state.get("audit", {}),
        "request_meter": meter.as_dict(),
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }
    answer = final_state.get("answer", {})
    return {
        **answer,
        "sources": final_state.get("sources", []),
        "mode": "advanced",
        "diagnostics": diagnostics,
    }
