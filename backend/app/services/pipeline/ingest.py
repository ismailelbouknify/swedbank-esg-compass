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
from app.services.extraction.patterns import YEAR_RE, extract_series
from app.services.retrieval.chunker import chunk_pages

log = logging.getLogger(__name__)
MAX_TABLE_PAGES = 20


def _looks_like_table_page(text: str) -> bool:
    return len(set(YEAR_RE.findall(text))) >= 2 and len(re.findall(r"\d+(?:[.,]\d+)?", text)) >= 12


def ingest_pdf(db: Session, source: Source, path: Path) -> Document:
    settings = get_settings()
    extracted = extract_pdf(path)
    doc = Document(source_id=source.id, title=extracted.title or source.title, doc_kind="pdf",
                   page_count=extracted.page_count, report_year=extracted.report_year, ocr_required=extracted.ocr_required)
    db.add(doc)
    db.flush()
    if extracted.ocr_required:
        source.status = "OCR_REQUIRED"
        source.status_detail = "PDF appears to be image-only; OCR is not enabled in this MVP."
        return doc

    # pdfplumber only for table-like pages the text parser could not structure
    table_pages = [p.page_number for p in extracted.pages if _looks_like_table_page(p.text) and not extract_series(p.text)]
    tables = extract_tables_as_text(path, table_pages[:MAX_TABLE_PAGES]) if table_pages else {}

    page_texts: list[tuple[int | None, str]] = []
    for p in extracted.pages:
        text = p.text + ("\n\n[Table]\n" + tables[p.page_number] if p.page_number in tables else "")
        db.add(DocumentPage(document_id=doc.id, page_number=p.page_number, text=text))
        page_texts.append((p.page_number, text))
    for ch in chunk_pages(page_texts, settings.chunk_size_words, settings.chunk_overlap_words):
        db.add(DocumentChunk(document_id=doc.id, page_number=ch.page_number, chunk_index=ch.chunk_index, text=ch.text))
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
    for ch in chunk_pages([(1, content.text)], settings.chunk_size_words, settings.chunk_overlap_words):
        db.add(DocumentChunk(document_id=doc.id, page_number=None, chunk_index=ch.chunk_index, text=ch.text))
    source.status = "PROCESSED"
    source.status_detail = f"{len(content.text)} characters of page text extracted"
    source.title = source.title or content.title
    return doc
