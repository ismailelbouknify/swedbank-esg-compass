"""Safe HTTP downloading: timeouts, size limits, content-type checks, validated redirects."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from app.core.config import get_settings
from app.services.documents.url_safety import UnsafeURLError, validate_public_url

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; ESG-Assessment-MVP/0.1; +local research tool)"
PDF_TYPES = ("application/pdf", "application/x-pdf", "application/octet-stream", "binary/octet-stream")
HTML_TYPES = ("text/html", "application/xhtml+xml")


@dataclass
class FetchResult:
    url: str
    final_url: str
    content_type: str
    content: bytes
    kind: str  # pdf | html


class FetchError(Exception):
    pass


def fetch(url: str, *, expect: str = "any", client: httpx.Client | None = None) -> FetchResult:
    """Download a public PDF or HTML page. `expect` is 'pdf', 'html' or 'any'."""
    settings = get_settings()
    own_client = client is None
    client = client or httpx.Client(
        timeout=settings.http_timeout_seconds,
        follow_redirects=False,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.5"},
    )
    try:
        current = url
        for _ in range(settings.max_redirects + 1):
            try:
                validate_public_url(current)
            except UnsafeURLError as exc:
                raise FetchError(f"Blocked URL {current}: {exc}") from exc
            with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        raise FetchError("Redirect without location")
                    current = urljoin(current, location)
                    continue
                if resp.status_code != 200:
                    raise FetchError(f"HTTP {resp.status_code} for {current}")
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                kind = _classify(ctype, current)
                if kind is None or (expect != "any" and kind != expect):
                    raise FetchError(f"Unexpected content-type {ctype!r} for {current}")
                limit_mb = settings.max_pdf_size_mb if kind == "pdf" else settings.max_html_size_mb
                limit = limit_mb * 1024 * 1024
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > limit:
                    raise FetchError(f"File too large ({int(declared)} bytes > {limit_mb} MB)")
                buf = bytearray()
                for chunk in resp.iter_bytes():
                    buf.extend(chunk)
                    if len(buf) > limit:
                        raise FetchError(f"Download exceeded {limit_mb} MB limit")
                content = bytes(buf)
                if kind == "pdf" and not content.startswith(b"%PDF"):
                    raise FetchError("Downloaded file is not a PDF (bad magic bytes)")
                return FetchResult(url=url, final_url=current, content_type=ctype, content=content, kind=kind)
        raise FetchError("Too many redirects")
    except httpx.HTTPError as exc:
        raise FetchError(f"Network error for {url}: {exc}") from exc
    finally:
        if own_client:
            client.close()


def _classify(ctype: str, url: str) -> str | None:
    if ctype in HTML_TYPES:
        return "html"
    if ctype in PDF_TYPES[:2]:
        return "pdf"
    if ctype in PDF_TYPES[2:] and url.lower().split("?")[0].endswith(".pdf"):
        return "pdf"
    return None
