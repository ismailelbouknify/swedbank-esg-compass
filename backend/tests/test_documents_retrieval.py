"""1. PDF extraction  2. chunking  3. BM25 retrieval  + URL safety."""
from __future__ import annotations

import pymupdf
import pytest

from app.services.documents.pdf_extractor import extract_pdf
from app.services.documents.url_safety import UnsafeURLError, validate_public_url
from app.services.retrieval.bm25 import BM25Index, query_for
from app.services.retrieval.chunker import chunk_pages


def test_pdf_extraction_preserves_pages(sample_pdf):
    doc = extract_pdf(sample_pdf)
    assert doc.page_count == 5
    assert not doc.ocr_required
    assert doc.report_year == 2025
    assert "Sustainability Report 2025" in doc.title
    energy_page = next(p for p in doc.pages if "reduce energy consumption" in p.text)
    assert energy_page.page_number == 2
    assert "30% by 2030" in energy_page.text


def test_image_only_pdf_flagged_ocr_required(tmp_path):
    path = tmp_path / "scan.pdf"
    d = pymupdf.open()
    for _ in range(3):
        d.new_page()
    d.save(path)
    doc = extract_pdf(path)
    assert doc.ocr_required is True
    assert doc.pages == []


def test_non_pdf_rejected(tmp_path):
    bad = tmp_path / "x.pdf"
    bad.write_text("not a pdf")
    with pytest.raises(ValueError):
        extract_pdf(bad)


def test_chunking_respects_pages_size_and_overlap():
    sentences = [f"Sentence number {i} talks about energy efficiency measures." for i in range(60)]
    page1 = " ".join(sentences)
    chunks = chunk_pages([(1, page1), (2, "Short second page about diversity.")], size_words=60, overlap_words=15)
    assert all(len(c.text.split()) <= 60 for c in chunks)
    assert {c.page_number for c in chunks} == {1, 2}
    p1 = [c for c in chunks if c.page_number == 1]
    assert len(p1) > 3
    # consecutive chunks share overlapping sentences
    assert p1[0].text.splitlines()[-1] in p1[1].text
    assert chunks[-1].page_number == 2 and "diversity" in chunks[-1].text
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunking_splits_long_tables():
    table = "\n".join(" ".join(str(n) for n in range(i, i + 20)) for i in range(0, 400, 20))
    chunks = chunk_pages([(3, table)], size_words=50, overlap_words=10)
    assert all(len(c.text.split()) <= 50 for c in chunks)


def test_bm25_retrieves_factor_relevant_chunks():
    items = [
        {"id": 1, "text": "Our diversity strategy targets 40% women in management by 2027.", "page_number": 3},
        {"id": 2, "text": "We aim to reduce energy consumption by 30% by 2030 compared with 2020. Energy intensity kWh.", "page_number": 2},
        {"id": 3, "text": "The board met eleven times during the year to discuss governance.", "page_number": 7},
        {"id": 4, "text": "Information security incidents and GDPR data protection training.", "page_number": 9},
    ]
    index = BM25Index(items)
    terms, must = query_for("energy_management", "planning")
    hits = index.search(terms, top_k=2, must_contain_any=must)
    assert hits and hits[0].chunk_id == 2
    assert all("energy" in h.text.lower() for h in hits)
    terms, must = query_for("diversity_inclusion", "planning")
    assert index.search(terms, top_k=1, must_contain_any=must)[0].chunk_id == 1
    assert BM25Index([]).search(["energy"]) == []


@pytest.mark.parametrize("url", [
    "http://localhost/admin", "http://127.0.0.1/", "http://10.0.0.5/x", "http://169.254.169.254/latest/meta-data",
    "file:///etc/passwd", "ftp://example.com/a.pdf", "http://user:pw@example.com/", "http://example.com:8080/",
    "http://[::1]/",
])
def test_unsafe_urls_rejected(url):
    with pytest.raises(UnsafeURLError):
        validate_public_url(url, resolve_dns=False)


def test_public_url_accepted():
    assert validate_public_url("https://example.com/report.pdf", resolve_dns=False)
