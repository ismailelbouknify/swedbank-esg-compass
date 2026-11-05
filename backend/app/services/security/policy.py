"""Rejection codes and small shared policy helpers."""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

from app.core.config import get_settings

T = TypeVar("T")

# URL policy
UNSAFE_URL = "unsafe_url"
UNSAFE_SCHEME = "unsafe_scheme"
HTTP_NOT_ALLOWED = "http_not_allowed"
CREDENTIALS_IN_URL = "credentials_in_url"
UNSAFE_HOST = "unsafe_host"
UNSAFE_PORT = "unsafe_port"
PRIVATE_IP = "private_ip"
DOMAIN_NOT_ALLOWED = "domain_not_allowed"
EXTERNAL_FETCH_DISABLED = "external_fetch_disabled"
# Redirects
TOO_MANY_REDIRECTS = "too_many_redirects"
# Content policy
DOWNLOAD_SIZE_LIMIT_EXCEEDED = "download_size_limit_exceeded"
INVALID_CONTENT_TYPE = "invalid_content_type"
INVALID_PDF_SIGNATURE = "invalid_pdf_signature"
PDF_PAGE_LIMIT_EXCEEDED = "pdf_page_limit_exceeded"
PDF_ENCRYPTED = "pdf_encrypted"
MALWARE_SCAN_FAILED = "malware_scan_failed"
# Availability: the source could not be read, but no policy rejected it
DNS_RESOLUTION_FAILED = "dns_resolution_failed"
SOURCE_TIMEOUT = "source_timeout"
NETWORK_ERROR = "network_error"
HTTP_ERROR = "http_error"
PDF_PARSE_FAILED = "pdf_parse_failed"

AVAILABILITY_CODES = frozenset({DNS_RESOLUTION_FAILED, SOURCE_TIMEOUT, NETWORK_ERROR, HTTP_ERROR, PDF_PARSE_FAILED})

# Audit event per code; a rejected redirect target is always logged as redirect_rejected.
_AUDIT_EVENTS = {
    UNSAFE_URL: "unsafe_url_rejected",
    UNSAFE_SCHEME: "unsafe_url_rejected",
    HTTP_NOT_ALLOWED: "unsafe_url_rejected",
    CREDENTIALS_IN_URL: "unsafe_url_rejected",
    UNSAFE_HOST: "unsafe_url_rejected",
    UNSAFE_PORT: "unsafe_url_rejected",
    PRIVATE_IP: "private_ip_rejected",
    TOO_MANY_REDIRECTS: "redirect_rejected",
    DOWNLOAD_SIZE_LIMIT_EXCEEDED: "download_size_limit_exceeded",
    INVALID_CONTENT_TYPE: "invalid_content_type",
    INVALID_PDF_SIGNATURE: "invalid_pdf_signature",
    PDF_PAGE_LIMIT_EXCEEDED: "pdf_page_limit_exceeded",
    SOURCE_TIMEOUT: "source_timeout",
}


class SourceRejection(ValueError):
    """External content was refused or is unavailable. `code` is the audit reason; the message is
    for technical logs and audit records only and is never shown in the analyst UI."""

    def __init__(self, code: str, message: str = "", *, retryable: bool = False, redirected: bool = False):
        super().__init__(message or code)
        self.code = code
        self.retryable = retryable
        self.redirected = redirected


def is_security_rejection(code: str) -> bool:
    return code not in AVAILABILITY_CODES


def audit_event_for(exc: SourceRejection) -> str:
    if exc.redirected and is_security_rejection(exc.code):
        return "redirect_rejected"
    return _AUDIT_EVENTS.get(exc.code, "external_source_rejected")


def browser_automation_allowed() -> bool:
    """Browser automation is an exceptional fallback and needs an explicit ENABLE_PLAYWRIGHT=true.
    Any future browser fetcher must check this first and meet the README's browser requirements."""
    return get_settings().enable_playwright


def run_bounded(fn: Callable[[], T], timeout: float) -> T:
    """Run a blocking call that has no timeout of its own (DNS, third-party search client).
    Raises TimeoutError when it does not finish in time; the abandoned daemon thread ends on its own."""
    outcome: dict[str, T | BaseException] = {}

    def _target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised in the caller
            outcome["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"Call did not finish within {timeout}s")
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome["value"]  # type: ignore[return-value]
