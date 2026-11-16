from __future__ import annotations

import re
from io import BytesIO
from typing import Any

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches, Pt, RGBColor


def markdown_to_docx(md_text: str, title: str = "Báo Cáo") -> bytes:
    """Convert Markdown report (with headings, tables, bullets, and bold text) to a styled Word document (.docx)."""
    doc = docx.Document()

    # Standard margins (1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base typography
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0x1F, 0x29, 0x37)  # Slate-800

    lines = md_text.splitlines()
    in_table = False
    table_lines: list[str] = []

    def flush_table(tbl_lines: list[str]) -> None:
        if not tbl_lines:
            return
        rows: list[list[str]] = []
        for line in tbl_lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                stripped = stripped[1:-1]
            cells = [c.strip() for c in stripped.split("|")]
            # Skip separator row: |---|---|
            if all(re.match(r"^:?-+:?$", c) for c in cells if c):
                continue
            rows.append(cells)
        if not rows:
            return

        num_cols = max(len(r) for r in rows)
        table = doc.add_table(rows=len(rows), cols=num_cols)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True

        for r_idx, r_data in enumerate(rows):
            is_header = r_idx == 0
            row = table.rows[r_idx]
            for c_idx in range(num_cols):
                cell_text = r_data[c_idx] if c_idx < len(r_data) else ""
                cell = row.cells[c_idx]
                cell.text = ""
                p = cell.paragraphs[0]
                _add_formatted_runs(p, cell_text)

                if is_header:
                    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1E3A8A"/>')
                    cell._tc.get_or_add_tcPr().append(shading)
                    for run in p.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                else:
                    if r_idx % 2 == 1:
                        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F3F4F6"/>')
                        cell._tc.get_or_add_tcPr().append(shading)
                    for run in p.runs:
                        run.font.size = Pt(10)
        doc.add_paragraph()  # Spacing after table

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            in_table = True
            table_lines.append(stripped)
            continue
        elif in_table:
            flush_table(table_lines)
            in_table = False
            table_lines = []

        if not stripped:
            continue

        if stripped.startswith("# "):
            h = doc.add_heading(level=1)
            run = h.add_run(stripped[2:])
            run.font.size = Pt(18)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)  # Navy
        elif stripped.startswith("## "):
            h = doc.add_heading(level=2)
            run = h.add_run(stripped[3:])
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)  # Blue
        elif stripped.startswith("### "):
            h = doc.add_heading(level=3)
            run = h.add_run(stripped[4:])
            run.font.size = Pt(12)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x37, 0x41, 0x51)
        elif stripped.startswith(("- ", "* ")):
            p = doc.add_paragraph(style="List Bullet")
            _add_formatted_runs(p, stripped[2:])
        elif re.match(r"^\d+\.\s+", stripped):
            prefix_match = re.match(r"^(\d+\.\s+)", stripped)
            prefix = prefix_match.group(1) if prefix_match else ""
            p = doc.add_paragraph(style="List Number")
            _add_formatted_runs(p, stripped[len(prefix) :])
        else:
            p = doc.add_paragraph()
            _add_formatted_runs(p, stripped)

    if in_table:
        flush_table(table_lines)

    bio = BytesIO()
    doc.save(bio)
    return bio.getvalue()


def _add_formatted_runs(paragraph: Any, text: str) -> None:
    """Format bold markers (**text**) cleanly into Word runs."""
    parts = re.split(r"(\*\*.*?\*\*)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2])
            run.font.bold = True
        else:
            paragraph.add_run(part)
