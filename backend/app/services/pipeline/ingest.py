"""Turns a Source (uploaded file or downloaded PDF/HTML) into Document/pages/chunks."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.entities import Document, DocumentChunk, DocumentPage, Source
from app.services.documents.html_extractor import extract_html
from app.services.documents.pdf_extractor import detect_report_year, extract_pdf
from app.services.documents.tables import extract_tables_as_text
from app.services.documents.text_utils import YEAR_RE
from app.services.retrieval.chunker import TABLE_MARKER, chunk_pages

log = logging.getLogger(__name__)
MAX_TABLE_PAGES = 20


def _looks_like_table_page(text: str) -> bool:
    return len(set(YEAR_RE.findall(text))) >= 2 and len(re.findall(r"\d+(?:[.,]\d+)?", text)) >= 12


def _chunk_row(document_id: int, ch, page_number: int | None) -> DocumentChunk:
    return DocumentChunk(document_id=document_id, page_number=page_number, chunk_index=ch.chunk_index, text=ch.text,
                         section_heading=(ch.section_heading or None) and ch.section_heading[:300],
                         content_kind="table" if ch.is_table else "text")


def ingest_pdf(db: Session, source: Source, pdf: Path | str | bytes, *, deadline: float | None = None) -> Document:
    """`pdf` is a stored file (uploads) or in-memory bytes (external PDFs, which are not kept)."""
    settings = get_settings()
    extracted = extract_pdf(pdf, deadline=deadline)
    doc = Document(source_id=source.id, title=extracted.title or source.title, doc_kind="pdf",
                   page_count=extracted.page_count, report_year=extracted.report_year, ocr_required=extracted.ocr_required)
    db.add(doc)
    db.flush()
    if extracted.ocr_required:
        source.status = "OCR_REQUIRED"
        source.status_detail = "PDF appears to be image-only; OCR is not enabled in this MVP."
        return doc

    # pdfplumber for table-like pages: its rows keep labels, units and periods next to the figures
    table_pages = [p.page_number for p in extracted.pages if _looks_like_table_page(p.text)]
    tables = extract_tables_as_text(pdf, table_pages[:MAX_TABLE_PAGES], deadline=deadline) if table_pages else {}

    page_texts: list[tuple[int | None, str]] = []
    for p in extracted.pages:
        text = p.text + ("\n\n" + TABLE_MARKER + "\n" + tables[p.page_number] if p.page_number in tables else "")
        db.add(DocumentPage(document_id=doc.id, page_number=p.page_number, text=text))
        page_texts.append((p.page_number, text))
    for ch in chunk_pages(page_texts, settings.chunk_size_tokens, settings.chunk_overlap_tokens):
        db.add(_chunk_row(doc.id, ch, ch.page_number))
    source.status = "PROCESSED"
    source.status_detail = f"{extracted.page_count} pages extracted" + (f", {len(tables)} table page(s) enriched" if tables else "")
    if not source.title:
        source.title = doc.title
    return doc


def ingest_html(db: Session, source: Source, html: bytes | str, url: str) -> Document | None:
    settings = get_settings()
    content = extract_html(html, url)
    if len(content.text) < 200:
        source.status = "SKIPPED"
        source.status_detail = "Page had too little text content (possibly dynamic/JavaScript-rendered)."
        return None
    doc = Document(source_id=source.id, title=content.title or source.title, doc_kind="html", page_count=1,
                   report_year=detect_report_year(content.title or "", content.text[:20000]))
    db.add(doc)
    db.flush()
    db.add(DocumentPage(document_id=doc.id, page_number=1, text=content.text))
    for ch in chunk_pages([(1, content.text)], settings.chunk_size_tokens, settings.chunk_overlap_tokens):
        db.add(_chunk_row(doc.id, ch, None))
    source.status = "PROCESSED"
    source.status_detail = f"{len(content.text)} characters of page text extracted"
    source.title = source.title or content.title
    return doc
