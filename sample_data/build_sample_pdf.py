"""Build the synthetic demo PDF from synthetic_sustainability_report.txt.

Usage (from the project root, with the backend venv active):
    python sample_data/build_sample_pdf.py
Produces sample_data/nordvik_sustainability_report_2025.pdf (fictional company)."""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

HERE = Path(__file__).resolve().parent
TEXT_FILE = HERE / "synthetic_sustainability_report.txt"
DEFAULT_OUT = HERE / "nordvik_sustainability_report_2025.pdf"


def build_pdf(out_path: Path = DEFAULT_OUT, text_file: Path = TEXT_FILE) -> Path:
    pages = [p.strip() for p in text_file.read_text(encoding="utf-8").split("---PAGE---")]
    doc = fitz.open()
    doc.set_metadata({"title": "Nordvik Components AB Sustainability Report 2025 (synthetic)", "author": "ESG MVP demo"})
    for body in pages:
        page = doc.new_page(width=595, height=842)  # A4
        y = 60
        for line in body.splitlines():
            rect = fitz.Rect(50, y, 545, y + 200)
            # insert_textbox returns remaining height (negative if it did not fit)
            used = page.insert_textbox(rect, line, fontsize=10, fontname="helv")
            lines_used = max(1, int((200 - used) // 13)) if used >= 0 else 2
            y += 13 * lines_used + 4
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    doc.close()
    return out_path


if __name__ == "__main__":
    out = build_pdf(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
    print(f"Wrote {out}")
