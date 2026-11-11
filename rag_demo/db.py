from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from .config import DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                file_name TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                parser TEXT,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                chunk_size INTEGER NOT NULL DEFAULT 900,
                chunk_overlap INTEGER NOT NULL DEFAULT 150,
                chunk_strategy TEXT NOT NULL DEFAULT 'parent_child',
                created_at TEXT NOT NULL,
                UNIQUE(knowledge_base_id, sha256)
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                parent_chunk_id TEXT,
                page_number INTEGER,
                content_type TEXT NOT NULL DEFAULT 'text',
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                UNIQUE(document_id, chunk_index)
            );
            CREATE TABLE IF NOT EXISTS parent_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                parent_index INTEGER NOT NULL,
                page_number INTEGER,
                content_type TEXT NOT NULL DEFAULT 'text',
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                UNIQUE(document_id, parent_index)
            );
            CREATE INDEX IF NOT EXISTS idx_documents_kb ON documents(knowledge_base_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, chunk_index);
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                suggestions_json TEXT NOT NULL DEFAULT '[]',
                diagnostics_json TEXT NOT NULL DEFAULT '{}',
                comparison_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
            """
        )
        chunk_columns = {row["name"] for row in db.execute("PRAGMA table_info(chunks)").fetchall()}
        if "parent_chunk_id" not in chunk_columns:
            db.execute("ALTER TABLE chunks ADD COLUMN parent_chunk_id TEXT")
        document_columns = {row["name"] for row in db.execute("PRAGMA table_info(documents)").fetchall()}
        if "chunk_strategy" not in document_columns:
            db.execute(
                "ALTER TABLE documents ADD COLUMN chunk_strategy TEXT NOT NULL DEFAULT 'parent_child'"
            )
        message_columns = {row["name"] for row in db.execute("PRAGMA table_info(messages)").fetchall()}
        if "diagnostics_json" not in message_columns:
            db.execute("ALTER TABLE messages ADD COLUMN diagnostics_json TEXT NOT NULL DEFAULT '{}'")
        if "comparison_json" not in message_columns:
            db.execute("ALTER TABLE messages ADD COLUMN comparison_json TEXT NOT NULL DEFAULT '{}'")
        db.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_chunk_id)")
        _backfill_parent_chunks(db)
        db.execute(
            "INSERT OR IGNORE INTO knowledge_bases(id, name, created_at) VALUES(?, ?, ?)",
            ("default", "Kho tài liệu mặc định", _now()),
        )


def _backfill_parent_chunks(connection: sqlite3.Connection) -> None:
    """Give existing flat chunks parent groups so the new mode works on old uploads."""
    rows = connection.execute(
        """SELECT c.*, d.file_name, d.knowledge_base_id FROM chunks c
           JOIN documents d ON d.id = c.document_id
           WHERE c.parent_chunk_id IS NULL AND d.chunk_strategy = 'parent_child'
           ORDER BY c.document_id, c.page_number, c.chunk_index"""
    ).fetchall()
    by_document: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_document.setdefault(row["document_id"], []).append(row)

    for document_id, document_rows in by_document.items():
        parent_index = connection.execute(
            "SELECT COALESCE(MAX(parent_index), -1) + 1 FROM parent_chunks WHERE document_id = ?",
            (document_id,),
        ).fetchone()[0]
        groups: list[list[sqlite3.Row]] = []
        current: list[sqlite3.Row] = []
        current_page: int | None = None
        for row in document_rows:
            page = row["page_number"]
            if current and (len(current) >= 3 or page != current_page):
                groups.append(current)
                current = []
            if not current:
                current_page = page
            current.append(row)
        if current:
            groups.append(current)

        for group in groups:
            parent_id = f"{document_id}:p{parent_index:06d}"
            first = group[0]
            metadata = json.loads(first["metadata_json"] or "{}")
            metadata.update(
                {
                    "parent_chunk_id": parent_id,
                    "document_id": document_id,
                    "knowledge_base_id": first["knowledge_base_id"],
                    "file_name": first["file_name"],
                    "chunk_level": "parent",
                }
            )
            content = "\n\n".join(row["content"] for row in group)
            connection.execute(
                """INSERT OR IGNORE INTO parent_chunks
                   (id, document_id, parent_index, page_number, content_type, content, metadata_json)
                   VALUES(?, ?, ?, ?, ?, ?, ?)""",
                (
                    parent_id,
                    document_id,
                    parent_index,
                    first["page_number"],
                    first["content_type"],
                    content,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            for row in group:
                child_metadata = json.loads(row["metadata_json"] or "{}")
                child_metadata["parent_chunk_id"] = parent_id
                connection.execute(
                    "UPDATE chunks SET parent_chunk_id = ?, metadata_json = ? WHERE id = ?",
                    (parent_id, json.dumps(child_metadata, ensure_ascii=False), row["id"]),
                )
            parent_index += 1


def list_knowledge_bases() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT id, name, created_at FROM knowledge_bases ORDER BY created_at"
        ).fetchall()
    return [dict(row) for row in rows]


def create_knowledge_base(name: str) -> str:
    kb_id = uuid.uuid4().hex
    with connect() as db:
        db.execute(
            "INSERT INTO knowledge_bases(id, name, created_at) VALUES(?, ?, ?)",
            (kb_id, name.strip(), _now()),
        )
    return kb_id


def rename_knowledge_base(kb_id: str, name: str) -> None:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Tên kho không được để trống.")
    with connect() as db:
        cursor = db.execute(
            "UPDATE knowledge_bases SET name = ? WHERE id = ?",
            (cleaned[:100], kb_id),
        )
        if cursor.rowcount == 0:
            raise ValueError("Không tìm thấy kho tài liệu.")


def delete_knowledge_base_record(kb_id: str) -> None:
    if kb_id == "default":
        raise ValueError("Kho mặc định được giữ lại làm kho dự phòng.")
    with connect() as db:
        cursor = db.execute("DELETE FROM knowledge_bases WHERE id = ?", (kb_id,))
        if cursor.rowcount == 0:
            raise ValueError("Không tìm thấy kho tài liệu.")


def get_knowledge_base(kb_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM knowledge_bases WHERE id = ?", (kb_id,)).fetchone()
    return dict(row) if row else None


def find_duplicate(kb_id: str, sha256: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute(
            "SELECT * FROM documents WHERE knowledge_base_id = ? AND sha256 = ?",
            (kb_id, sha256),
        ).fetchone()
    return dict(row) if row else None


def create_document(
    kb_id: str,
    document_id: str,
    file_name: str,
    stored_path: Path,
    sha256: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    chunk_strategy: str = "parent_child",
) -> None:
    with connect() as db:
        db.execute(
            """INSERT INTO documents
               (id, knowledge_base_id, file_name, stored_path, sha256, status,
                chunk_size, chunk_overlap, chunk_strategy, created_at)
               VALUES(?, ?, ?, ?, ?, 'Đang nhập', ?, ?, ?, ?)""",
            (
                document_id,
                kb_id,
                file_name,
                str(stored_path),
                sha256,
                chunk_size,
                chunk_overlap,
                chunk_strategy,
                _now(),
            ),
        )


def recover_incomplete_documents() -> None:
    with connect() as db:
        db.execute(
            """UPDATE documents SET status = 'Lỗi',
               error = 'Ứng dụng đã dừng trước khi hoàn tất; có thể bấm thử lại.'
               WHERE status IN ('Đang nhập', 'Đang đọc tài liệu', 'Đang tạo embedding')"""
        )


def set_document_state(
    document_id: str,
    status: str,
    *,
    error: str | None = None,
    parser: str | None = None,
    chunk_count: int | None = None,
) -> None:
    fields = ["status = ?", "error = ?"]
    values: list[Any] = [status, error]
    if parser is not None:
        fields.append("parser = ?")
        values.append(parser)
    if chunk_count is not None:
        fields.append("chunk_count = ?")
        values.append(chunk_count)
    values.append(document_id)
    with connect() as db:
        db.execute(f"UPDATE documents SET {', '.join(fields)} WHERE id = ?", values)


def get_document(document_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
    return dict(row) if row else None


def get_document_names(document_ids: list[str]) -> dict[str, str]:
    if not document_ids:
        return {}
    result: dict[str, str] = {}
    with connect() as db:
        for start in range(0, len(document_ids), 400):
            group = document_ids[start : start + 400]
            placeholders = ",".join("?" for _ in group)
            rows = db.execute(
                f"SELECT id, file_name FROM documents WHERE id IN ({placeholders})", group
            ).fetchall()
            result.update({row["id"]: row["file_name"] for row in rows})
    return result


def list_documents(kb_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM documents WHERE knowledge_base_id = ? ORDER BY created_at DESC",
            (kb_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def save_chunks(document_id: str, chunks: list[dict[str, Any]]) -> None:
    with connect() as db:
        db.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        db.executemany(
            """INSERT INTO chunks
               (id, document_id, chunk_index, parent_chunk_id, page_number, content_type, content, metadata_json)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    item["chunk_id"],
                    document_id,
                    item["chunk_index"],
                    item.get("parent_chunk_id"),
                    item.get("page_number"),
                    item.get("content_type", "text"),
                    item["content"],
                    json.dumps(item.get("metadata", {}), ensure_ascii=False),
                )
                for item in chunks
            ],
        )


def save_parent_chunks(document_id: str, parents: list[dict[str, Any]]) -> None:
    with connect() as db:
        db.execute("DELETE FROM parent_chunks WHERE document_id = ?", (document_id,))
        db.executemany(
            """INSERT INTO parent_chunks
               (id, document_id, parent_index, page_number, content_type, content, metadata_json)
               VALUES(?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    item["parent_chunk_id"],
                    document_id,
                    item["parent_index"],
                    item.get("page_number"),
                    item.get("content_type", "text"),
                    item["content"],
                    json.dumps(item.get("metadata", {}), ensure_ascii=False),
                )
                for item in parents
            ],
        )


def get_parent_chunks(parent_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not parent_ids:
        return {}
    result: dict[str, dict[str, Any]] = {}
    with connect() as db:
        for start in range(0, len(parent_ids), 400):
            group = parent_ids[start : start + 400]
            placeholders = ",".join("?" for _ in group)
            rows = db.execute(
                f"SELECT * FROM parent_chunks WHERE id IN ({placeholders})", group
            ).fetchall()
            result.update({row["id"]: dict(row) for row in rows})
    return result


def list_chunks(document_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """SELECT c.*, d.file_name FROM chunks c
               JOIN documents d ON d.id = c.document_id
               WHERE c.document_id = ? ORDER BY c.chunk_index LIMIT ? OFFSET ?""",
            (document_id, limit, offset),
        ).fetchall()
    return [dict(row) for row in rows]


def get_all_ready_chunks(kb_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """SELECT c.*, d.file_name FROM chunks c
               JOIN documents d ON d.id = c.document_id
               WHERE d.knowledge_base_id = ? AND d.status = 'Sẵn sàng'
               ORDER BY d.file_name, c.page_number, c.chunk_index""",
            (kb_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_chunk_contents(chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    result: dict[str, str] = {}
    with connect() as db:
        for start in range(0, len(chunk_ids), 400):
            group = chunk_ids[start : start + 400]
            placeholders = ",".join("?" for _ in group)
            rows = db.execute(
                f"SELECT id, content FROM chunks WHERE id IN ({placeholders})", group
            ).fetchall()
            result.update({row["id"]: row["content"] for row in rows})
    return result


def delete_document_record(document_id: str) -> dict[str, Any] | None:
    document = get_document(document_id)
    if not document:
        return None
    with connect() as db:
        db.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    return document


def rename_document_record(document_id: str, file_name: str) -> None:
    with connect() as db:
        row = db.execute("SELECT id FROM documents WHERE id = ?", (document_id,)).fetchone()
        if not row:
            raise ValueError("Không tìm thấy tài liệu.")
        db.execute("UPDATE documents SET file_name = ? WHERE id = ?", (file_name, document_id))
        rows = db.execute(
            "SELECT id, metadata_json FROM chunks WHERE document_id = ?",
            (document_id,),
        ).fetchall()
        for chunk in rows:
            metadata = json.loads(chunk["metadata_json"] or "{}")
            metadata["file_name"] = file_name
            db.execute(
                "UPDATE chunks SET metadata_json = ? WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False), chunk["id"]),
            )
        parent_rows = db.execute(
            "SELECT id, metadata_json FROM parent_chunks WHERE document_id = ?",
            (document_id,),
        ).fetchall()
        for parent in parent_rows:
            metadata = json.loads(parent["metadata_json"] or "{}")
            metadata["file_name"] = file_name
            db.execute(
                "UPDATE parent_chunks SET metadata_json = ? WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False), parent["id"]),
            )


def create_conversation(kb_id: str, title: str = "Cuộc trò chuyện mới") -> str:
    conversation_id = uuid.uuid4().hex
    now = _now()
    with connect() as db:
        db.execute(
            """INSERT INTO conversations(id, knowledge_base_id, title, created_at, updated_at)
               VALUES(?, ?, ?, ?, ?)""",
            (conversation_id, kb_id, title, now, now),
        )
    return conversation_id


def list_conversations(kb_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """SELECT * FROM conversations WHERE knowledge_base_id = ?
               ORDER BY updated_at DESC, created_at DESC, id DESC LIMIT 100""",
            (kb_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(conversation_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    return dict(row) if row else None


def rename_conversation(conversation_id: str, title: str) -> None:
    cleaned = title.strip()
    if not cleaned:
        raise ValueError("Tên cuộc hội thoại không được để trống.")
    with connect() as db:
        cursor = db.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (cleaned[:100], _now(), conversation_id),
        )
        if cursor.rowcount == 0:
            raise ValueError("Không tìm thấy cuộc hội thoại.")


def delete_conversation(conversation_id: str) -> None:
    with connect() as db:
        cursor = db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        if cursor.rowcount == 0:
            raise ValueError("Không tìm thấy cuộc hội thoại.")


def save_message(
    conversation_id: str,
    role: str,
    content: str,
    *,
    sources: list[dict[str, Any]] | None = None,
    suggestions: list[str] | None = None,
    diagnostics: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> None:
    with connect() as db:
        db.execute(
            """INSERT INTO messages(id, conversation_id, role, content, sources_json,
               suggestions_json, diagnostics_json, comparison_json, created_at)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                uuid.uuid4().hex,
                conversation_id,
                role,
                content,
                json.dumps(sources or [], ensure_ascii=False),
                json.dumps(suggestions or [], ensure_ascii=False),
                json.dumps(diagnostics or {}, ensure_ascii=False),
                json.dumps(comparison or {}, ensure_ascii=False),
                _now(),
            ),
        )
        db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conversation_id)
        )
        if role == "user":
            conversation = db.execute(
                "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if conversation and conversation["title"] == "Cuộc trò chuyện mới":
                title = content.strip().replace("\n", " ")[:55]
                if len(content.strip()) > 55:
                    title += "…"
                db.execute(
                    "UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id)
                )


def get_messages(conversation_id: str) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid",
            (conversation_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["sources"] = json.loads(item.pop("sources_json") or "[]")
        item["suggestions"] = json.loads(item.pop("suggestions_json") or "[]")
        item["diagnostics"] = json.loads(item.pop("diagnostics_json", "{}") or "{}")
        item["comparison"] = json.loads(item.pop("comparison_json", "{}") or "{}")
        result.append(item)
    return result


def get_recent_turns(conversation_id: str, limit: int = 6) -> list[dict[str, str]]:
    messages = get_messages(conversation_id)
    return [{"role": row["role"], "content": row["content"]} for row in messages[-limit:]]
