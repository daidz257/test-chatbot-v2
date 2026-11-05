from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
MODEL_CACHE_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "app.sqlite3"

GENERATION_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIMENSIONS = int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768"))
GENERATION_RPM_LIMIT = max(1, int(os.getenv("GEMINI_GENERATION_RPM_LIMIT", "10")))
EMBEDDING_RPM_LIMIT = max(1, int(os.getenv("GEMINI_EMBEDDING_RPM_LIMIT", "10")))
EMBEDDING_BATCH_SIZE = max(1, min(100, int(os.getenv("GEMINI_EMBEDDING_BATCH_SIZE", "32"))))
MAX_RETRIES = max(1, min(8, int(os.getenv("GEMINI_MAX_RETRIES", "5"))))
MAX_UPLOAD_MB = max(1, int(os.getenv("MAX_UPLOAD_MB", "100")))
RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", "ms-marco-MultiBERT-L-12")

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".csv", ".tsv", ".docx", ".doc", ".xlsx", ".xls",
    ".pptx", ".ppt", ".rtf", ".odt", ".ods", ".odp", ".html", ".htm", ".epub",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
    ".json", ".xml", ".log", ".adoc", ".asciidoc", ".tex", ".xhtml",
    ".mht", ".mhtml", ".eml", ".msg",
}

for directory in (DATA_DIR, UPLOAD_DIR, CHROMA_DIR, MODEL_CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)
