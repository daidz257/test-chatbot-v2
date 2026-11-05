from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any

import chromadb
from flashrank import RerankRequest
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .advanced_rag import (
    _audit_answer,
    _build_context,
    _query_plan,
    _ranker,
    _terms,
    _unique_queries,
)
from .config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_RPM_LIMIT,
    GENERATION_RPM_LIMIT,
    MAX_UPLOAD_MB,
    RERANKER_MODEL,
)
from .parsers import parse_document
from .quota import GeminiEmbeddings, RequestMeter, generate_answer_json
from .rag import _answer_prompt, _chunk_sections


def prepare_upload(
    file_name: str,
    contents: bytes,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, Any]:
    """Parse one uploaded file and prepare both chunk previews without persisting them."""
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"File vượt giới hạn {MAX_UPLOAD_MB} MB.")
    suffix = Path(file_name).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="rag-lab-") as folder:
        source_path = Path(folder) / f"upload{suffix}"
        source_path.write_bytes(contents)
        sections, parser_name = parse_document(source_path)

    preview_document_id = f"lab-{uuid.uuid4().hex}"
    previews: dict[str, dict[str, Any]] = {}
    for strategy in ("regular", "parent_child"):
        parents: list[dict[str, Any]] = []
        chunks = _chunk_sections(
            sections,
            document_id=preview_document_id,
            kb_id="lab",
            file_name=file_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=strategy,
            parent_records=parents,
        )
        previews[strategy] = {"chunks": chunks, "parents": parents}

    if not sections or not any(item["chunks"] for item in previews.values()):
        raise ValueError("Parser không trích xuất được văn bản hoặc bảng có thể chia chunk.")

    return {
        "id": preview_document_id,
        "file_name": file_name,
        "contents": contents,
        "parser": parser_name,
        "sections": sections,
        "previews": previews,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def create_ephemeral_index(
    chunks: list[dict[str, Any]],
    api_key: str,
    meter: RequestMeter,
    *,
    embedding_rpm: int = EMBEDDING_RPM_LIMIT,
    progress=None,
) -> dict[str, Any]:
    """Embed selected chunks into an in-memory Chroma collection for this lab run."""
    if not chunks:
        raise ValueError("Không có chunk để lập chỉ mục.")
    embedder = GeminiEmbeddings(
        api_key,
        batch_size=EMBEDDING_BATCH_SIZE,
        rpm_limit=embedding_rpm,
        progress=progress,
        meter=meter,
    )
    vectors = embedder.embed_documents([str(item["content"]) for item in chunks])
    client = chromadb.EphemeralClient()
    collection_name = f"lab_{uuid.uuid4().hex[:24]}"
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    metadatas = []
    for item in chunks:
        metadata = item.get("metadata", {})
        metadatas.append(
            {
                str(key): value
                for key, value in metadata.items()
                if isinstance(value, (str, int, float, bool)) and value is not None
            }
        )
    collection.add(
        ids=[str(item["chunk_id"]) for item in chunks],
        documents=[str(item["content"]) for item in chunks],
        embeddings=vectors,
        metadatas=metadatas,
    )
    return {"client": client, "collection_name": collection_name}


def plan_queries(
    question: str,
    api_key: str,
    methods: set[str],
    meter: RequestMeter,
    *,
    generation_rpm: int = GENERATION_RPM_LIMIT,
    progress=None,
) -> dict[str, Any]:
    flags = {
        "query_rewrite": "Rewrite" in methods,
        "multi_query": "Multi-query" in methods,
        "hyde": "HyDE" in methods,
    }
    if not any(flags.values()):
        return {"rewritten_query": question, "alternative_queries": [], "hypothetical_answer": ""}
    return _query_plan(
        question,
        [],
        api_key,
        flags,
        meter,
        generation_rpm=generation_rpm,
        progress=progress,
    )


def query_strings(question: str, plan: dict[str, Any]) -> list[str]:
    return _unique_queries(question, plan)


def embed_queries_cached(
    queries: list[str],
    api_key: str,
    meter: RequestMeter,
    cache: dict[str, list[float]],
    *,
    embedding_rpm: int = EMBEDDING_RPM_LIMIT,
    progress=None,
) -> list[list[float]]:
    missing = list(dict.fromkeys(query for query in queries if query not in cache))
    if missing:
        embedder = GeminiEmbeddings(
            api_key,
            batch_size=EMBEDDING_BATCH_SIZE,
            rpm_limit=embedding_rpm,
            progress=progress,
            meter=meter,
        )
        vectors = embedder.embed_queries(missing)
        cache.update({query: vector for query, vector in zip(missing, vectors)})
    return [cache[query] for query in queries]


def retrieve(
    index: dict[str, Any],
    chunks: list[dict[str, Any]],
    parents: list[dict[str, Any]],
    *,
    file_name: str,
    question: str,
    plan: dict[str, Any],
    api_key: str,
    meter: RequestMeter,
    use_hybrid: bool,
    use_parent_child: bool,
    candidate_k: int = 24,
    embedding_rpm: int = EMBEDDING_RPM_LIMIT,
    query_vectors: list[list[float]] | None = None,
    progress=None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_strings = _unique_queries(question, plan)
    if query_vectors is None:
        query_vectors = embed_queries_cached(
            query_strings,
            api_key,
            meter,
            {},
            embedding_rpm=embedding_rpm,
            progress=progress,
        )
    if len(query_vectors) != len(query_strings):
        raise ValueError("Số vector truy vấn không khớp số query cần tìm.")
    collection = index["client"].get_collection(index["collection_name"])
    dense_k = min(max(candidate_k, 8), 50, collection.count())
    all_by_id = {str(item["chunk_id"]): item for item in chunks}
    dense_rankings: list[list[str]] = []
    dense_seen: set[str] = set()
    fused_scores: dict[str, float] = {}
    for vector in query_vectors:
        response = collection.query(
            query_embeddings=[vector],
            n_results=dense_k,
            include=["distances"],
        )
        ids = (response.get("ids") or [[]])[0]
        ranking = [str(item) for item in ids if str(item) in all_by_id]
        dense_rankings.append(ranking)
        for rank, chunk_id in enumerate(ranking, start=1):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
            dense_seen.add(chunk_id)

    bm25_rankings: list[list[str]] = []
    bm25_seen: set[str] = set()
    if use_hybrid:
        bm25 = BM25Okapi([_terms(str(item["content"])) for item in chunks])
        ids_by_index = [str(item["chunk_id"]) for item in chunks]
        for query in query_strings:
            scores = bm25.get_scores(_terms(query))
            indexes = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:dense_k]
            ranking = [ids_by_index[i] for i in indexes if scores[i] > 0]
            bm25_rankings.append(ranking)
            for rank, chunk_id in enumerate(ranking, start=1):
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
                bm25_seen.add(chunk_id)

    ranked_ids = sorted(fused_scores, key=fused_scores.__getitem__, reverse=True)[:dense_k]
    parent_by_id = {str(item["parent_chunk_id"]): item for item in parents}
    candidates: list[dict[str, Any]] = []
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        chunk = all_by_id[chunk_id]
        metadata = dict(chunk.get("metadata", {}))
        parent_id = str(chunk.get("parent_chunk_id") or "") if use_parent_child else ""
        parent = parent_by_id.get(parent_id)
        content = str(chunk["content"])
        row = {
            "id": chunk_id,
            "document_id": index["collection_name"],
            "file_name": file_name,
            "page_number": chunk.get("page_number"),
            "content_type": chunk.get("content_type", "text"),
            "chunk_index": int(chunk.get("chunk_index", rank - 1)),
            "content": content,
            "parent_chunk_id": parent_id or None,
        }
        candidates.append(
            {
                "id": chunk_id,
                "document": Document(page_content=content, metadata=metadata),
                "row": row,
                "rrf_score": fused_scores[chunk_id],
                "rrf_rank": rank,
                "rerank_score": None,
                "context": str(parent["content"]) if parent else content,
                "parent_id": parent_id or None,
                "parent_index": parent.get("parent_index") if parent else None,
                "parent_page_number": parent.get("page_number") if parent else None,
                "parent_fallback": bool(parent_id and not parent),
                "file_name": file_name,
            }
        )
    return candidates, {
        "queries_used": query_strings,
        "dense_candidate_count": len(dense_seen),
        "bm25_candidate_count": len(bm25_seen),
        "fused_candidate_count": len(ranked_ids),
        "dense_top_ids": dense_rankings[0][:candidate_k] if dense_rankings else [],
        "bm25_top_ids": bm25_rankings[0][:candidate_k] if bm25_rankings else [],
        "hybrid_search": use_hybrid,
    }


def rerank_candidates(
    candidates: list[dict[str, Any]],
    question: str,
    rewritten_query: str,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    ranker = _ranker(RERANKER_MODEL)
    passages = [
        {"id": item["id"], "text": item["document"].page_content, "meta": item["document"].metadata}
        for item in candidates
    ]
    ranked = ranker.rerank(
        RerankRequest(query=rewritten_query or question, passages=passages)
    )
    by_id = {item["id"]: item for item in candidates}
    result = []
    for rerank_rank, item in enumerate(ranked, start=1):
        candidate = by_id.get(str(item.get("id", "")))
        if candidate:
            candidate["rerank_score"] = float(item.get("score", 0.0))
            candidate["rerank_rank"] = rerank_rank
            result.append(candidate)
    return result or candidates


def build_context(
    question: str,
    candidates: list[dict[str, Any]],
    *,
    use_parent_child: bool,
    compress: bool,
    top_k: int,
    context_token_budget: int,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    return _build_context(
        question,
        candidates,
        {"parent_child": use_parent_child, "context_compression": compress},
        top_k=top_k,
        context_token_budget=context_token_budget,
    )


def generate_answer(
    question: str,
    context: str,
    api_key: str,
    meter: RequestMeter,
    *,
    generation_rpm: int = GENERATION_RPM_LIMIT,
    progress=None,
) -> dict[str, Any]:
    if not context.strip():
        raise ValueError("Chưa có ngữ cảnh. Hoàn tất bước tìm kiếm và nén ngữ cảnh trước.")
    return generate_answer_json(
        api_key,
        _answer_prompt(question, context, []),
        rpm_limit=generation_rpm,
        progress=progress,
        meter=meter,
    )


def audit_answer(
    question: str,
    answer: str,
    context: str,
    sources: list[dict[str, Any]],
    api_key: str,
    meter: RequestMeter,
    methods: set[str],
    *,
    generation_rpm: int = GENERATION_RPM_LIMIT,
    progress=None,
) -> dict[str, Any]:
    flags = {
        "answer_grading": "Answer Grading" in methods,
        "citation_verification": "Citation Verification" in methods,
    }
    return _audit_answer(
        question,
        answer,
        context,
        sources,
        api_key,
        flags,
        meter,
        generation_rpm=generation_rpm,
        progress=progress,
    )


def cleanup_index(index: dict[str, Any] | None) -> None:
    if not index:
        return
    try:
        index["client"].delete_collection(index["collection_name"])
    except Exception:
        pass
