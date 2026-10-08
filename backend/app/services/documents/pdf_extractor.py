"""PDF text extraction with PyMuPDF. OCR is not performed; image-only PDFs are flagged."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

# A page with fewer extractable characters than this is considered "image-like".
MIN_CHARS_PER_TEXT_PAGE = 40
OCR_REQUIRED_RATIO = 0.8


@dataclass
class ExtractedPage:
    page_number: int  # 1-based, as printed by PDF viewers
    text: str


@dataclass
class ExtractedDocument:
    title: str | None
    pages: list[ExtractedPage] = field(default_factory=list)
    page_count: int = 0
    ocr_required: bool = False
    report_year: int | None = None

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


def clean_text(text: str) -> str:
    # a soft hyphen at a line break means the word continues on the next line
    text = re.sub(r"\u00ad\s*\n\s*", "", text)
    text = text.replace("\u00ad", "")
    # a real hyphen at a line break is usually a compound ("climate-neutral"): keep it, join the line
    text = re.sub(r"(\w)-\n(\w)", r"\1-\2", text)
    text = re.sub(r"[ \t]+", " ", text)  # no-break spaces are kept: they separate thousands
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: str | Path) -> ExtractedDocument:
    """Extract per-page text. Raises ValueError if the file is not a readable PDF."""
    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not open PDF: {exc}") from exc
    try:
        if doc.needs_pass:
            raise ValueError("PDF is password protected")
        pages: list[ExtractedPage] = []
        empty = 0
        for i, page in enumerate(doc):
            text = clean_text(page.get_text("text") or "")
            if len(text) < MIN_CHARS_PER_TEXT_PAGE:
                empty += 1
            pages.append(ExtractedPage(page_number=i + 1, text=text))
        meta_title = (doc.metadata or {}).get("title") or None
        page_count = doc.page_count
    finally:
        doc.close()

    ocr_required = page_count > 0 and (empty / page_count) >= OCR_REQUIRED_RATIO
    result = ExtractedDocument(
        title=meta_title.strip() if meta_title and meta_title.strip() else None,
        pages=[p for p in pages if p.text],
        page_count=page_count,
        ocr_required=ocr_required,
    )
    if not result.title and result.pages:
        first_line = next((ln.strip() for ln in result.pages[0].text.splitlines() if ln.strip()), None)
        result.title = first_line[:200] if first_line else None
    result.report_year = detect_report_year(result.title or "", result.full_text[:20000])
    return result


def detect_report_year(title: str, text: str) -> int | None:
    """Best-effort reporting year: explicit 'Report 2024' style phrases, else most common year."""
    for candidate in (title, text[:3000]):
        m = re.search(
            r"(?:report|reporting year|financial year|fiscal year|FY)\s*[:\-]?\s*(20\d{2})"
            r"|(20\d{2})\s*(?:sustainability|annual|esg|integrated|impact)\s+report",
            candidate,
            re.IGNORECASE,
        )
        if m:
            return int(m.group(1) or m.group(2))
    years = [int(y) for y in re.findall(r"\b(20[0-4]\d)\b", text)]
    if not years:
        return None
    return Counter(years).most_common(1)[0][0]
