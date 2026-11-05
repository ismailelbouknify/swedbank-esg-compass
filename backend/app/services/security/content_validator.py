"""Validation of external and uploaded content before it reaches a parser.

Content is only ever treated as data: nothing here opens, renders or executes it. PyMuPDF parsing
is not a replacement for malware scanning; register a scanner to scan content before parsing."""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import pymupdf

from app.core.config import get_settings
from app.services.security import policy
from app.services.security.policy import SourceRejection

log = logging.getLogger(__name__)

PDF_SIGNATURE = b"%PDF-"
HTML_TYPES = ("text/html", "application/xhtml+xml")
PDF_TYPES = ("application/pdf", "application/x-pdf")
GENERIC_BINARY_TYPES = ("application/octet-stream", "binary/octet-stream")


class ContentRejected(SourceRejection):
    pass


@dataclass(frozen=True)
class ScanResult:
    status: str  # clean | infected | error | not_configured
    detail: str = ""


ContentScanner = Callable[[bytes, str, str], ScanResult]  # content, kind, sha256
_scanner: ContentScanner | None = None


def set_content_scanner(scanner: ContentScanner | None) -> None:
    """Integration point for an organisation-approved malware scanner."""
    global _scanner
    _scanner = scanner


def scan_downloaded_content(content: bytes, *, kind: str, sha256: str) -> ScanResult:
    """Run the registered scanner. A scanner failure never counts as clean."""
    if _scanner is None:
        return ScanResult("not_configured")
    try:
        return _scanner(content, kind, sha256)
    except Exception as exc:  # noqa: BLE001
        log.error("content_scan_error: error=%s sha256=%s", type(exc).__name__, sha256)
        return ScanResult("error", type(exc).__name__)


def enforce_scan(content: bytes, *, kind: str, sha256: str) -> str:
    """Must run before parsing. Returns the scan status or raises ContentRejected."""
    result = scan_downloaded_content(content, kind=kind, sha256=sha256)
    if result.status == "clean":
        return result.status
    if result.status == "not_configured" and not get_settings().malware_scan_required:
        return result.status
    raise ContentRejected(policy.MALWARE_SCAN_FAILED, f"Content scan result: {result.status}")


def sha256_hex(content: bytes) -> str:
    """For duplicate detection and audit only; a hash says nothing about whether content is safe."""
    return hashlib.sha256(content).hexdigest()


def url_path_is_pdf(url: str) -> bool:
    try:
        return urlparse(url).path.lower().endswith(".pdf")
    except ValueError:
        return False


def classify_content_type(content_type: str, pdf_expected: bool) -> str | None:
    """'pdf', 'html' or None (not processed). A PDF claim is only a claim: the signature is checked too."""
    if content_type in HTML_TYPES:
        return "html"
    if content_type in PDF_TYPES:
        return "pdf"
    if content_type in GENERIC_BINARY_TYPES and pdf_expected:
        return "pdf"
    return None


def validate_pdf_signature(content: bytes) -> None:
    if not content.startswith(PDF_SIGNATURE):
        raise ContentRejected(policy.INVALID_PDF_SIGNATURE, "Content does not have a PDF signature")


def enforce_page_limit(page_count: int) -> None:
    """Over-limit documents are rejected, never silently truncated: an ESG score must not rest on
    part of a report without anyone knowing."""
    limit = get_settings().max_pdf_pages
    if page_count > limit:
        raise ContentRejected(policy.PDF_PAGE_LIMIT_EXCEEDED, f"PDF has {page_count} pages (limit {limit})")


def validate_uploaded_pdf(content: bytes) -> tuple[int, str]:
    """Full check for a manual upload, in safe order: signature, scan, then parser.
    Returns (page count, scan status)."""
    validate_pdf_signature(content)
    scan_status = enforce_scan(content, kind="pdf", sha256=sha256_hex(content))
    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ContentRejected(policy.PDF_PARSE_FAILED, "PDF could not be opened") from exc
    try:
        if doc.needs_pass:
            raise ContentRejected(policy.PDF_ENCRYPTED, "PDF is password protected")
        enforce_page_limit(doc.page_count)
        return doc.page_count, scan_status
    finally:
        doc.close()
