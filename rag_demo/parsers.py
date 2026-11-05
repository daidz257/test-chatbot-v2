from __future__ import annotations

import csv
from functools import lru_cache
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ParsedSection:
    text: str
    page_number: int | None = None
    kind: str = "text"
    section: str = ""
    parser: str = ""


class NeedsDocling(Exception):
    pass


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()


def _markdown_table(rows: list[list[Any]]) -> str:
    cleaned = [[_cell(cell) for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    header = cleaned[0]
    body = cleaned[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _decode_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1258", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_pdf(path: Path) -> list[ParsedSection]:
    import pymupdf

    sections: list[ParsedSection] = []
    low_text_pages = 0
    page_count = 0
    with pymupdf.open(path) as pdf:
        page_count = len(pdf)
        for page_index, page in enumerate(pdf, start=1):
            body = (page.get_text("text") or "").strip()
            if len(body) < 40:
                low_text_pages += 1
            if body:
                sections.append(ParsedSection(body, page_index, "text", "Nội dung trang", "PyMuPDF"))
            try:
                finder = page.find_tables()
                for table_index, table in enumerate(finder.tables if finder else [], start=1):
                    rendered = _markdown_table(table.extract() or [])
                    if rendered:
                        sections.append(
                            ParsedSection(
                                f"Bảng {table_index} trên trang {page_index}\n{rendered}",
                                page_index,
                                "table",
                                f"Bảng {table_index}",
                                "PyMuPDF",
                            )
                        )
            except Exception:
                # Text extraction remains useful when a PDF table is malformed.
                pass
    if page_count and (not sections or low_text_pages / page_count >= 0.35):
        raise NeedsDocling(
            f"PDF có {low_text_pages}/{page_count} trang ít văn bản; chuyển sang OCR/Docling để đọc trang scan."
        )
    return sections


def _parse_docx(path: Path) -> list[ParsedSection]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(path)
    sections: list[ParsedSection] = []
    paragraphs: list[str] = []
    table_number = 0

    def flush_paragraphs() -> None:
        text = "\n".join(item for item in paragraphs if item.strip()).strip()
        if text:
            sections.append(ParsedSection(text, None, "text", "Nội dung", "python-docx"))
        paragraphs.clear()

    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            value = Paragraph(child, document).text.strip()
            if value:
                paragraphs.append(value)
        elif child.tag.endswith("}tbl"):
            flush_paragraphs()
            table_number += 1
            rows = [[cell.text for cell in row.cells] for row in Table(child, document).rows]
            table = _markdown_table(rows)
            if table:
                sections.append(
                    ParsedSection(table, None, "table", f"Bảng {table_number}", "python-docx")
                )
    flush_paragraphs()
    return sections


def _parse_spreadsheet(path: Path) -> list[ParsedSection]:
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    formula_workbook = load_workbook(path, read_only=True, data_only=False)
    result: list[ParsedSection] = []
    try:
        for sheet in workbook.worksheets:
            formula_sheet = formula_workbook[sheet.title]
            rows = []
            data_rows = sheet.iter_rows(values_only=True)
            formula_rows = formula_sheet.iter_rows(values_only=True)
            for row, formula_row in zip(data_rows, formula_rows):
                rendered_row = []
                for column_index, value in enumerate(row):
                    formula_value = formula_row[column_index] if column_index < len(formula_row) else None
                    if value is None and isinstance(formula_value, str) and formula_value.startswith("="):
                        value = formula_value
                    rendered_row.append(value)
                rows.append(rendered_row)
            table = _markdown_table(rows)
            if table:
                result.append(
                    ParsedSection(
                        f"Trang tính: {sheet.title}\n{table}", None, "table", sheet.title, "openpyxl"
                    )
                )
    finally:
        workbook.close()
        formula_workbook.close()
    return result


def _parse_csv(path: Path) -> list[ParsedSection]:
    text = _decode_text(path)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t;|")
        delimiter = dialect.delimiter
    except csv.Error:
        pass
    rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
    rendered = _markdown_table(rows)
    return [ParsedSection(rendered, None, "table", path.stem, "csv")] if rendered else []


def _parse_pptx(path: Path) -> list[ParsedSection]:
    from pptx import Presentation

    deck = Presentation(path)
    result: list[ParsedSection] = []
    for slide_number, slide in enumerate(deck.slides, start=1):
        text_parts: list[str] = []
        table_number = 0
        for shape in slide.shapes:
            if getattr(shape, "has_table", False):
                table_number += 1
                rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
                rendered = _markdown_table(rows)
                if rendered:
                    result.append(
                        ParsedSection(
                            f"Slide {slide_number} · Bảng {table_number}\n{rendered}",
                            slide_number,
                            "table",
                            f"Bảng {table_number}",
                            "python-pptx",
                        )
                    )
            elif getattr(shape, "has_text_frame", False) and shape.text.strip():
                text_parts.append(shape.text.strip())
        if text_parts:
            result.append(
                ParsedSection(
                    "\n".join(text_parts), slide_number, "text", "Nội dung slide", "python-pptx"
                )
            )
    return result


def _parse_html(path: Path) -> list[ParsedSection]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_decode_text(path), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    sections: list[ParsedSection] = []
    table_index = 0
    for table in list(soup.find_all("table")):
        rows = [[cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])] for row in table.find_all("tr")]
        rendered = _markdown_table(rows)
        if rendered:
            table_index += 1
            sections.append(
                ParsedSection(rendered, None, "table", f"Bảng {table_index}", "BeautifulSoup")
            )
        table.decompose()
    body_text = soup.get_text("\n", strip=True)
    if body_text:
        sections.insert(0, ParsedSection(html.unescape(body_text), None, "text", "Nội dung", "BeautifulSoup"))
    return sections


def _parse_fast(path: Path) -> list[ParsedSection]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix == ".docx":
        return _parse_docx(path)
    if suffix in {".txt", ".md"}:
        text = _decode_text(path).strip()
        return [ParsedSection(text, None, "text", path.stem, "text decoder")] if text else []
    if suffix in {".log", ".xml", ".adoc", ".asciidoc", ".tex"}:
        text = _decode_text(path).strip()
        return [ParsedSection(text, None, "text", path.stem, "text decoder")] if text else []
    if suffix == ".json":
        raw = _decode_text(path)
        try:
            text = json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            text = raw
        return [ParsedSection(text.strip(), None, "text", path.stem, "JSON/text decoder")]
    if suffix in {".csv", ".tsv"}:
        return _parse_csv(path)
    if suffix == ".xlsx":
        return _parse_spreadsheet(path)
    if suffix == ".pptx":
        return _parse_pptx(path)
    if suffix in {".html", ".htm", ".xhtml"}:
        return _parse_html(path)
    raise NeedsDocling(f"Chưa có parser nhanh cho định dạng {suffix or '(không có đuôi file)' }.")


@lru_cache(maxsize=1)
def _docling_converter():
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
        from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
    except ImportError as error:
        raise RuntimeError(
            "Không đọc được file bằng parser nhanh và Docling chưa được cài. "
            "Cài đầy đủ dependencies bằng run.ps1 hoặc `pip install -r requirements.txt`."
        ) from error

    # RapidOCR defaults to Simplified Chinese when no language is selected.
    # Explicitly choose Vietnamese for scanned PDFs and image uploads.
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        ocr_options=RapidOcrOptions(lang=["vi"]),
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }
    )


def _parse_with_docling(path: Path) -> list[ParsedSection]:
    converter = _docling_converter()
    result = converter.convert(path, raises_on_error=True, max_num_pages=1000, max_file_size=100 * 1024 * 1024)
    document = result.document
    page_numbers: set[int] = set()
    for item, _level in document.iterate_items():
        for provenance in getattr(item, "prov", []) or []:
            if getattr(provenance, "page_no", None) is not None:
                page_numbers.add(int(provenance.page_no))

    sections: list[ParsedSection] = []
    if page_numbers:
        for page_number in sorted(page_numbers):
            markdown = document.export_to_markdown(page_no=page_number, traverse_pictures=True).strip()
            if markdown:
                sections.append(
                    ParsedSection(markdown, page_number, "mixed", f"Nội dung trang {page_number}", "Docling OCR/layout")
                )
    else:
        markdown = document.export_to_markdown(traverse_pictures=True).strip()
        if markdown:
            sections.append(ParsedSection(markdown, None, "mixed", "Nội dung tài liệu", "Docling"))
    if not sections:
        raise RuntimeError("Docling không tìm thấy văn bản hoặc bảng trong file.")
    return sections


def parse_document(path: Path) -> tuple[list[ParsedSection], str]:
    fast_error: Exception | None = None
    try:
        sections = _parse_fast(path)
        combined = "\n".join(section.text for section in sections).strip()
        # Very sparse extraction from a binary Office/image-based document is a
        # strong signal that the fast parser did not recover the useful content.
        if not combined:
            raise NeedsDocling("Parser nhanh không tìm thấy nội dung có thể tìm kiếm.")
        if path.suffix.lower() == ".docx" and len(combined) < 25 and path.stat().st_size > 20_000:
            raise NeedsDocling("DOCX có rất ít văn bản trích xuất; chuyển sang Docling để đọc layout/ảnh.")
        return sections, "; ".join(sorted({section.parser for section in sections if section.parser}))
    except Exception as error:
        fast_error = error

    try:
        sections = _parse_with_docling(path)
        return sections, "Docling (fallback OCR/layout)"
    except Exception as docling_error:
        detail = str(docling_error).strip()
        if isinstance(fast_error, NeedsDocling):
            reason = str(fast_error)
        else:
            reason = f"Parser nhanh báo: {fast_error}"
        raise RuntimeError(f"{reason} Docling fallback cũng thất bại: {detail[:500]}") from docling_error
