"""The only path for retrieving external pages and documents.

Normal public HTTP/HTTPS access with basic SSRF protection: every URL and every redirect target is checked
against localhost/private/link-local/metadata destinations, and the connection goes to the address that
was checked. Time, size and redirects are bounded. The response is held in memory only: nothing is written
to disk, and the caller drops the bytes once text has been extracted."""
from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from app.core.config import Settings, get_settings
from app.services.security import policy
from app.services.security.content_validator import (
    PDF_SIGNATURE,
    classify_content_type,
    enforce_scan,
    sha256_hex,
    sniff_kind,
)
from app.services.security.policy import SourceRejection
from app.services.security.url_validator import UnsafeURLError, ValidatedURL, validate_external_url

USER_AGENT = "Mozilla/5.0 (compatible; ESG-Assessment-MVP/0.1; +local research tool)"
READ_CHUNK_BYTES = 64 * 1024
MAX_CONNECT_ADDRESSES = 3
RETRY_BACKOFF_SECONDS = 0.5
RETRYABLE_STATUS = {429, 502, 503, 504}


class FetchError(SourceRejection):
    pass


@dataclass
class FetchResult:
    url: str
    final_url: str
    content_type: str
    content: bytes
    kind: str  # pdf | html
    sha256: str
    size: int
    scan_status: str


def bounded_timeout(settings: Settings | None = None) -> httpx.Timeout:
    s = settings or get_settings()
    return httpx.Timeout(connect=s.http_connect_timeout_seconds, read=s.http_read_timeout_seconds,
                         write=s.http_read_timeout_seconds, pool=s.http_connect_timeout_seconds)


def source_deadline() -> float:
    return time.monotonic() + get_settings().max_source_processing_seconds


def build_client() -> httpx.Client:
    # trust_env=False: proxy variables and .netrc must not silently change where requests go
    return httpx.Client(
        timeout=bounded_timeout(),
        follow_redirects=False,
        trust_env=False,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.1", "Accept-Encoding": "identity"},
    )


def fetch(url: str, *, expect: str = "any", client: httpx.Client | None = None, deadline: float | None = None) -> FetchResult:
    """Retrieve one public PDF or HTML page into memory. `expect` is 'pdf', 'html' or 'any'.
    Raises FetchError, whose `code` says why; nothing is written to disk."""
    settings = get_settings()
    if not settings.enable_external_fetch:
        raise FetchError(policy.EXTERNAL_FETCH_DISABLED, "External retrieval is disabled by configuration")
    deadline = deadline if deadline is not None else source_deadline()
    own_client = client is None
    client = client or build_client()
    try:
        attempt = 0
        while True:  # bounded: every path either returns, raises, or increments towards max_http_retries
            try:
                return _follow_redirects(client, url, expect, deadline, settings)
            except FetchError as exc:
                wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                if not exc.retryable or attempt >= settings.max_http_retries or time.monotonic() + wait >= deadline:
                    raise
            time.sleep(wait)
            attempt += 1
    finally:
        client.cookies.clear()  # cookies live for one redirect chain only
        if own_client:
            client.close()


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise FetchError(policy.SOURCE_TIMEOUT, "Source processing time limit reached")


def _follow_redirects(client: httpx.Client, url: str, expect: str, deadline: float, settings: Settings) -> FetchResult:
    """Redirects to any public host are followed (corporate site -> CDN or filing host is normal);
    each target is validated again, so a redirect can never reach a local or internal address."""
    current = url
    for hop in range(settings.max_redirects + 1):
        _check_deadline(deadline)
        try:
            target = validate_external_url(current)
        except UnsafeURLError as exc:
            raise FetchError(exc.code, str(exc), retryable=exc.retryable, redirected=hop > 0) from exc
        try:
            resp = _send_pinned(client, target)
            try:
                if resp.is_redirect:
                    current = urljoin(current, resp.headers["location"])
                    continue
                return _read_body(resp, url, current, expect, deadline, settings)
            finally:
                resp.close()
        except SourceRejection:
            raise
        except httpx.TimeoutException as exc:
            raise FetchError(policy.SOURCE_TIMEOUT, "Request timed out", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise FetchError(policy.NETWORK_ERROR, f"Network error: {type(exc).__name__}", retryable=True) from exc
        except (httpx.InvalidURL, ValueError) as exc:  # malformed Location or host
            raise FetchError(policy.UNSAFE_URL, "Malformed URL", redirected=hop > 0) from exc
    raise FetchError(policy.TOO_MANY_REDIRECTS, f"More than {settings.max_redirects} redirects", redirected=True)


def _send_pinned(client: httpx.Client, target: ValidatedURL) -> httpx.Response:
    """Connect to an address that was just validated, not to whatever a second DNS lookup returns
    (DNS rebinding). The Host header, SNI and certificate check still use the real hostname."""
    last_error: httpx.TransportError | None = None
    host_header = target.host if target.port is None else f"{target.host}:{target.port}"
    for address in target.addresses[:MAX_CONNECT_ADDRESSES]:
        request = client.build_request(
            "GET", httpx.URL(target.url).copy_with(host=address),
            headers={"Host": host_header}, extensions={"sni_hostname": target.host},
        )
        try:
            return client.send(request, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            last_error = exc
    raise last_error or httpx.ConnectError("No address to connect to")


def _read_body(resp: httpx.Response, url: str, current: str, expect: str, deadline: float,
               settings: Settings) -> FetchResult:
    if not 200 <= resp.status_code < 300:
        raise FetchError(policy.HTTP_ERROR, f"HTTP {resp.status_code}", retryable=resp.status_code in RETRYABLE_STATUS)
    content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    declared = classify_content_type(content_type)
    if declared is None:  # image, archive, media...: not a document this tool reads, so not downloaded
        raise FetchError(policy.INVALID_CONTENT_TYPE, f"Unexpected content type {content_type!r}")

    html_limit = min(settings.max_download_mb, settings.max_html_size_mb) * 1024 * 1024
    limit = html_limit if declared == "html" else settings.max_download_mb * 1024 * 1024
    declared_size = resp.headers.get("content-length", "")
    if declared_size.isdigit() and int(declared_size) > limit:
        raise FetchError(policy.DOWNLOAD_SIZE_LIMIT_EXCEEDED, f"Declared size {declared_size} exceeds {limit} bytes")

    # Content-Length may be absent or wrong, and the limit applies to decoded bytes so that a
    # compressed response cannot expand past it unnoticed.
    buffer = bytearray()
    for chunk in resp.iter_bytes(READ_CHUNK_BYTES):
        buffer.extend(chunk)
        if len(buffer) > limit:
            raise FetchError(policy.DOWNLOAD_SIZE_LIMIT_EXCEEDED, f"Response exceeds {limit} bytes")
        _check_deadline(deadline)
        # a declared PDF that does not start like one: stop reading now
        if declared == "pdf" and not bytes(buffer[:len(PDF_SIGNATURE)]).startswith(PDF_SIGNATURE[:len(buffer)]):
            raise FetchError(policy.INVALID_PDF_SIGNATURE, "Content does not have a PDF signature")
    content = bytes(buffer)
    del buffer
    # The bytes decide what the document is, not the URL or a generic header: a ".pdf" link may land on an
    # HTML page, and download endpoints serve PDFs as octet-stream. A PDF needs its signature.
    kind = sniff_kind(content)
    if declared == "html" and kind is None:
        kind = "html"  # HTML fragments without an <html> tag are still pages
    if declared == "pdf" and kind != "pdf":
        raise FetchError(policy.INVALID_PDF_SIGNATURE, "Content does not have a PDF signature")
    if kind is None or (expect != "any" and kind != expect):
        raise FetchError(policy.INVALID_CONTENT_TYPE, f"Unexpected content {content_type!r}")
    if kind == "html" and len(content) > html_limit:
        raise FetchError(policy.DOWNLOAD_SIZE_LIMIT_EXCEEDED, f"Page exceeds {html_limit} bytes")
    digest = sha256_hex(content)
    try:
        scan_status = enforce_scan(content, kind=kind, sha256=digest)
    except SourceRejection as exc:
        raise FetchError(exc.code, str(exc)) from exc
    return FetchResult(url=url, final_url=current, content_type=content_type, content=content, kind=kind,
                       sha256=digest, size=len(content), scan_status=scan_status)
