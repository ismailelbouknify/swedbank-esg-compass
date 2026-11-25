"""The only path for retrieving external pages and documents.

Normal public HTTP/HTTPS access with basic SSRF protection: every URL and every redirect target is checked
against localhost/private/link-local/metadata destinations. A direct connection goes to the address that was
checked; a request through the configured proxy (see network.py) goes by hostname, because the proxy makes the
connection. Certificates are always verified. Time, size and redirects are bounded. The response is held in
memory only: nothing is written to disk, and the caller drops the bytes once text has been extracted."""
from __future__ import annotations

import ssl
import time
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from app.core.config import Settings, get_settings
from app.services.security import network, policy, windows_proxy
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


def build_client(proxy: str | None = None, proxy_headers: dict[str, str] | None = None) -> httpx.Client:
    """trust_env=False: the proxy is chosen explicitly by network.proxy_for() from the approved settings, so it is
    known per request (a direct connection is pinned, a proxied one is not); .netrc credentials are never used.
    `proxy_headers` carries an integrated sign-in token, sent only to the proxy (CONNECT), never to the site."""
    return httpx.Client(
        timeout=bounded_timeout(),
        follow_redirects=False,
        trust_env=False,
        proxy=httpx.Proxy(proxy, headers=proxy_headers) if proxy and proxy_headers else proxy,
        verify=network.ssl_context(),
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.1", "Accept-Encoding": "identity"},
    )


def fetch(url: str, *, expect: str = "any", client: httpx.Client | None = None, deadline: float | None = None,
          purpose: str = "retrieval") -> FetchResult:
    """Retrieve one public PDF or HTML page into memory. `expect` is 'pdf', 'html', 'any', or 'text' for a small
    text document such as robots.txt or a sitemap (never a PDF; HTML page size limit).
    `purpose='search'` is a search-results page: it is governed by ENABLE_WEB_SEARCH instead of ENABLE_EXTERNAL_FETCH,
    and goes through exactly the same proxy, sign-in, certificate, address and size checks.
    Raises FetchError, whose `code` says why; nothing is written to disk."""
    settings = get_settings()
    allowed = settings.enable_web_search if purpose == "search" else settings.enable_external_fetch
    if not allowed:
        raise FetchError(policy.EXTERNAL_FETCH_DISABLED, f"External {purpose} is disabled by configuration")
    deadline = deadline if deadline is not None else source_deadline()
    clients = _Clients(client)
    try:
        attempt = 0
        while True:  # bounded: every path either returns, raises, or increments towards max_http_retries
            try:
                return _follow_redirects(clients, url, expect, deadline, settings)
            except FetchError as exc:
                wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                if not exc.retryable or attempt >= settings.max_http_retries or time.monotonic() + wait >= deadline:
                    raise
            time.sleep(wait)
            attempt += 1
    finally:
        clients.close()  # cookies live for one redirect chain only


class _Clients:
    """The caller's client for every request, or one client per route (direct, or a configured proxy).

    A signed-in proxy route gets a new client, so a new connection and a fresh Kerberos ticket, for every request:
    a proxy refuses a ticket it has already seen (replay protection). Cookies are carried between all clients of
    one redirect chain and dropped afterwards."""

    def __init__(self, given: httpx.Client | None):
        self.given = given
        self.own: dict[str | None, httpx.Client] = {}
        self.single_use: list[httpx.Client] = []
        self.cookies = httpx.Cookies()

    def for_route(self, proxy: str | None) -> httpx.Client:
        if self.given is not None:
            return self.given
        if proxy not in self.own:
            self.own[proxy] = self._build(proxy)
        return self.own[proxy]

    def can_sign_in(self, proxy: str | None) -> bool:
        return self.given is None and network.integrated_signin_enabled(proxy)

    def signed_in(self, proxy: str) -> httpx.Client:
        try:
            headers = network.proxy_signin_header(proxy)
        except windows_proxy.IntegratedAuthUnavailable as exc:
            raise FetchError(policy.PROXY_SIGNIN_UNAVAILABLE, f"Integrated proxy sign-in unavailable: {exc.reason}") from exc
        client = self._build(proxy, headers)
        self.single_use.append(client)
        return client

    def _build(self, proxy: str | None, headers: dict[str, str] | None = None) -> httpx.Client:
        try:
            return build_client(proxy, headers) if proxy else build_client()
        except (OSError, ssl.SSLError) as exc:  # e.g. WEB_CA_BUNDLE points at a missing or invalid file
            raise FetchError(policy.TLS_VERIFICATION_FAILED,
                             f"Certificate configuration unusable: {type(exc).__name__}") from exc

    def before(self, client: httpx.Client) -> None:
        if client is not self.given:
            client.cookies.update(self.cookies)

    def after(self, client: httpx.Client) -> None:
        if client is not self.given:
            self.cookies.update(client.cookies)

    def close(self) -> None:
        self.cookies.clear()
        if self.given is not None:
            self.given.cookies.clear()
        for c in [*self.own.values(), *self.single_use]:
            c.cookies.clear()
            c.close()


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise FetchError(policy.SOURCE_TIMEOUT, "Source processing time limit reached")


def _follow_redirects(clients: _Clients, url: str, expect: str, deadline: float, settings: Settings) -> FetchResult:
    """Redirects to any public host are followed (corporate site -> CDN or filing host is normal);
    each target is validated again, so a redirect can never reach a local or internal address."""
    current = url
    for hop in range(settings.max_redirects + 1):
        _check_deadline(deadline)
        proxy = network.proxy_for(current)
        try:
            target = validate_external_url(current, allow_unresolved=proxy is not None)
        except UnsafeURLError as exc:
            raise FetchError(exc.code, str(exc), retryable=exc.retryable, redirected=hop > 0) from exc
        try:
            client, resp = _send_route(clients, proxy, target)
            try:
                clients.after(client)
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
        except httpx.ProxyError as exc:  # the proxy refused CONNECT, e.g. "407 Proxy Authentication Required"
            code = policy.PROXY_AUTH_REQUIRED if "407" in str(exc) else policy.PROXY_CONNECTION_FAILED
            raise FetchError(code, f"Proxy refused the connection: {str(exc)[:80]}") from exc
        except httpx.HTTPError as exc:
            if network.is_certificate_failure(exc):
                raise FetchError(policy.TLS_VERIFICATION_FAILED, "Server certificate could not be verified") from exc
            if proxy and isinstance(exc, httpx.ConnectError):
                raise FetchError(policy.PROXY_CONNECTION_FAILED, "Proxy could not be reached", retryable=True) from exc
            raise FetchError(policy.NETWORK_ERROR, f"Network error: {type(exc).__name__}", retryable=True) from exc
        except (httpx.InvalidURL, ValueError) as exc:  # malformed Location or host
            raise FetchError(policy.UNSAFE_URL, "Malformed URL", redirected=hop > 0) from exc
    raise FetchError(policy.TOO_MANY_REDIRECTS, f"More than {settings.max_redirects} redirects", redirected=True)


def _is_407(exc: httpx.ProxyError) -> bool:
    return "407" in str(exc)


def _send_route(clients: _Clients, proxy: str | None, target: ValidatedURL) -> tuple[httpx.Client, httpx.Response]:
    """Direct (pinned), through the proxy, or through the proxy with integrated sign-in. Sign-in is only attempted
    when WEB_PROXY_AUTH=integrated and the proxy asked for it (407); afterwards that proxy is signed in to at once."""
    if not proxy:
        client = clients.for_route(None)
        clients.before(client)
        return client, _send_pinned(client, target)
    if not (clients.can_sign_in(proxy) and network.signin_known_required(proxy)):
        client = clients.for_route(proxy)
        clients.before(client)
        try:
            resp = _send_via_proxy(client, target)
        except httpx.ProxyError as exc:
            if not (_is_407(exc) and clients.can_sign_in(proxy)):
                raise
        else:
            if not (resp.status_code == 407 and clients.can_sign_in(proxy)):
                return client, resp
            resp.close()
        network.remember_signin_required(proxy)
    client = clients.signed_in(proxy)
    clients.before(client)
    try:
        resp = _send_via_proxy(client, target)
    except httpx.ProxyError as exc:
        if _is_407(exc):
            raise FetchError(policy.PROXY_SIGNIN_REFUSED, "The proxy refused the integrated Windows sign-in") from exc
        raise
    if resp.status_code == 407:
        resp.close()
        raise FetchError(policy.PROXY_SIGNIN_REFUSED, "The proxy refused the integrated Windows sign-in")
    return client, resp


def _send_via_proxy(client: httpx.Client, target: ValidatedURL) -> httpx.Response:
    """The proxy connects to the host by name; the certificate is still checked against that name."""
    return client.send(client.build_request("GET", target.url), stream=True)


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
        refused = network.classify_block_page(resp.status_code, _peek(resp, deadline))
        if refused:
            raise FetchError(refused, f"HTTP {resp.status_code} from the network proxy or web filter")
        raise FetchError(policy.HTTP_ERROR, f"HTTP {resp.status_code}", retryable=resp.status_code in RETRYABLE_STATUS)
    content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    declared = classify_content_type(content_type)
    if declared is None:  # image, archive, media...: not a document this tool reads, so not downloaded
        raise FetchError(policy.INVALID_CONTENT_TYPE, f"Unexpected content type {content_type!r}")

    html_limit = min(settings.max_download_mb, settings.max_html_size_mb) * 1024 * 1024
    limit = html_limit if declared == "html" or expect == "text" else settings.max_download_mb * 1024 * 1024
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
    if expect == "text" and kind != "pdf" and declared != "pdf":
        kind = "text"  # robots.txt and sitemaps: read as text only, never parsed as a document
    if declared == "html" and kind is None:
        kind = "html"  # HTML fragments without an <html> tag are still pages
    if declared == "pdf" and kind != "pdf":
        raise FetchError(policy.INVALID_PDF_SIGNATURE, "Content does not have a PDF signature")
    if kind is None or (expect != "any" and kind != expect):
        raise FetchError(policy.INVALID_CONTENT_TYPE, f"Unexpected content {content_type!r}")
    if kind in ("html", "text") and len(content) > html_limit:
        raise FetchError(policy.DOWNLOAD_SIZE_LIMIT_EXCEEDED, f"Page exceeds {html_limit} bytes")
    if kind == "html" and len(content) <= network.BLOCK_PAGE_SCAN_BYTES:
        refused = network.classify_block_page(resp.status_code, content)  # some filters answer 200 with their own page
        if refused:
            raise FetchError(refused, "Page served by the network proxy or web filter instead of the site")
    digest = sha256_hex(content)
    try:
        scan_status = enforce_scan(content, kind=kind, sha256=digest)
    except SourceRejection as exc:
        raise FetchError(exc.code, str(exc)) from exc
    return FetchResult(url=url, final_url=current, content_type=content_type, content=content, kind=kind,
                       sha256=digest, size=len(content), scan_status=scan_status)


def _peek(resp: httpx.Response, deadline: float) -> bytes:
    """The start of an error page, enough to recognise a proxy or web filter page; never the whole body."""
    content_type = resp.headers.get("content-type", "").lower()
    if content_type and "html" not in content_type and "text" not in content_type:
        return b""
    buffer = bytearray()
    try:
        for chunk in resp.iter_bytes(READ_CHUNK_BYTES):
            buffer.extend(chunk)
            if len(buffer) >= network.BLOCK_PAGE_SCAN_BYTES or time.monotonic() >= deadline:
                break
    except httpx.HTTPError:
        pass
    return bytes(buffer[:network.BLOCK_PAGE_SCAN_BYTES])
