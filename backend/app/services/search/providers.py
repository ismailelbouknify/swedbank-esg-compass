"""Concrete search providers. DDGS (DuckDuckGo) needs no API key; SerpAPI is optional.

Transports. The DDGS library sends its requests through its own HTTP client (primp), which accepts a proxy only as
a URL: it cannot add the integrated Windows (Kerberos) proxy sign-in. Where that sign-in is needed, searches go
through the application's own retrieval transport instead (DuckDuckGoLiteSearchProvider): the same PAC routing,
Kerberos sign-in, certificate verification, address checks and limits as page retrieval. SerpAPI signs in itself."""
from __future__ import annotations

import logging
import re
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
from bs4 import BeautifulSoup

from app.core.config import get_settings
from app.services.search.base import NullSearchProvider, SearchProvider, SearchResult
from app.services.security import network, policy
from app.services.security.policy import SourceRejection, run_bounded
from app.services.security.safe_fetch import FetchError, bounded_timeout, fetch

log = logging.getLogger(__name__)

DDG_LITE_URL = "https://lite.duckduckgo.com/lite/"
DDGS_ROUTE_PROBE_URL = "https://duckduckgo.com/"
# DuckDuckGo answers suspected automation with a challenge page instead of results; it is reported, never solved
_CHALLENGE = re.compile(rb"anomaly|captcha|challenge-form|bots use duckduckgo", re.I)


class SearchPacer:
    """Process-wide pacing for the keyless DuckDuckGo providers, which have no published rate limit and answer bursts
    of queries with a challenge page. Queries are spaced by SEARCH_MIN_INTERVAL_SECONDS, and after a refusal no query
    is sent until SEARCH_REFUSAL_COOLDOWN_SECONDS have passed. This only ever slows the application down."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last = 0.0
        self._declined_until = 0.0
        self.sleep = time.sleep

    def cooldown_remaining(self) -> float:
        return max(0.0, self._declined_until - time.monotonic())

    def wait_turn(self) -> None:
        with self._lock:
            gap = get_settings().search_min_interval_seconds - (time.monotonic() - self._last)
            if gap > 0:
                self.sleep(gap)
            self._last = time.monotonic()

    def declined(self) -> None:
        self._declined_until = time.monotonic() + get_settings().search_refusal_cooldown_seconds

    def reset(self) -> None:
        self._last = self._declined_until = 0.0


pacer = SearchPacer()


def _cooling_down(provider: SearchProvider) -> bool:
    """During the cool-down after a refusal, a search is answered as refused without sending anything."""
    if pacer.cooldown_remaining() <= 0:
        return False
    provider.failed_searches += 1
    provider.last_error = policy.SEARCH_REFUSED
    log.info("search_skipped: provider=%s reason=cooldown remaining_s=%d", provider.name, pacer.cooldown_remaining())
    return True


def _result_url(href: str) -> str | None:
    """The target of a DuckDuckGo result link (`//duckduckgo.com/l/?uddg=<url>`); None for ads and internal links."""
    href = (href or "").strip()
    if href.startswith("//"):
        href = "https:" + href
    try:
        parts = urlsplit(href)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    if host == "duckduckgo.com" or host.endswith(".duckduckgo.com"):
        target = parse_qs(parts.query).get("uddg", [None])[0] if parts.path == "/l/" else None
        return _result_url(target) if target else None
    return href if parts.scheme in ("http", "https") and host else None


def parse_duckduckgo_lite(content: bytes, query: str) -> list[SearchResult]:
    soup = BeautifulSoup(content, "html.parser")
    out: list[SearchResult] = []
    for link in soup.select("a.result-link"):
        url = _result_url(link.get("href", ""))
        if not url or any(r.url == url for r in out):
            continue
        row = link.find_parent("tr")
        snippet_cell = None
        for sibling in (row.find_next_siblings("tr", limit=2) if row else []):
            snippet_cell = sibling.select_one("td.result-snippet")
            if snippet_cell:
                break
        out.append(SearchResult(title=link.get_text(" ", strip=True), url=url,
                                snippet=snippet_cell.get_text(" ", strip=True) if snippet_cell else "", query=query))
    return out


class DuckDuckGoLiteSearchProvider(SearchProvider):
    """DuckDuckGo's lite results page read through the application's retrieval transport (safe_fetch): PAC routing,
    integrated Windows sign-in, certificate verification, SSRF checks and size/time limits, exactly as for pages.
    The request identifies this application honestly; no browser is imitated. A challenge or refusal page from
    DuckDuckGo is reported as `search_service_refused`, never worked around."""

    name = "duckduckgo_lite"

    def __init__(self, client: httpx.Client | None = None):
        self._client = client  # tests only; a given client never signs in to a proxy

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        if _cooling_down(self):
            return []
        url = f"{DDG_LITE_URL}?{urlencode({'q': query})}"
        pacer.wait_turn()
        try:
            res = fetch(url, expect="html", client=self._client, purpose="search",
                        deadline=time.monotonic() + get_settings().search_timeout_seconds)
            results = parse_duckduckgo_lite(res.content, query)
            if not results and _CHALLENGE.search(res.content[:200_000]):
                raise FetchError(policy.SEARCH_REFUSED, "The search service answered with a challenge page")
        except SourceRejection as exc:
            code = exc.code
            if code == policy.HTTP_ERROR and "429" in str(exc):
                code = policy.SEARCH_RATE_LIMITED
            if code in policy.SEARCH_PROVIDER_RESTRICTION_CODES:
                pacer.declined()
            self.failed_searches += 1
            self.last_error = code
            log.warning("search_failed: provider=%s reason=%s kind=%s", self.name, code, policy.search_failure_kind(code))
            return []
        self.last_error = None
        return results[:max_results]


class DDGSSearchProvider(SearchProvider):
    name = "ddgs"

    def __init__(self) -> None:
        self._approved: DuckDuckGoLiteSearchProvider | None = None
        self.transport = "ddgs"

    def _needs_approved_transport(self) -> bool:
        """The DDGS client cannot sign in to a proxy: when integrated sign-in applies to the search route, use the
        application's own transport, which can."""
        return network.integrated_signin_enabled(network.proxy_for(DDGS_ROUTE_PROBE_URL))

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        if self._needs_approved_transport():
            if self._approved is None:
                self._approved = DuckDuckGoLiteSearchProvider()
                log.info("search_transport: provider=ddgs transport=application reason=integrated_proxy_signin")
            self.transport = "application"
            before = self._approved.failed_searches
            results = self._approved.search(query, max_results)
            self.failed_searches += self._approved.failed_searches - before
            self.last_error = self._approved.last_error
            return results
        self.transport = "ddgs"
        if _cooling_down(self):
            return []
        pacer.wait_turn()
        try:
            try:
                from ddgs import DDGS
            except ImportError:  # older package name
                from duckduckgo_search import DDGS  # type: ignore
            settings = get_settings()
            # the search client has its own HTTP stack: it is given the same approved proxy and CA bundle
            proxy = network.proxy_for("https://duckduckgo.com/")

            def _search() -> list[dict]:
                with DDGS(proxy=proxy, timeout=int(settings.http_read_timeout_seconds), verify=network.search_client_verify()) as ddgs:
                    return list(ddgs.text(query, max_results=max_results) or [])

            rows = run_bounded(_search, settings.search_timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            self.failed_searches += 1
            self.last_error = network.classify_search_error(exc)
            if self.last_error in policy.SEARCH_PROVIDER_RESTRICTION_CODES:
                pacer.declined()
            log.warning("search_failed: provider=ddgs error=%s reason=%s kind=%s", type(exc).__name__, self.last_error,
                        policy.search_failure_kind(self.last_error))
            return []
        self.last_error = None
        return [
            SearchResult(title=r.get("title", ""), url=r.get("href") or r.get("url", ""), snippet=r.get("body", ""), query=query)
            for r in rows
            if r.get("href") or r.get("url")
        ]


class SerpAPISearchProvider(SearchProvider):
    name = "serpapi"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def _get(self, url: str, query: str, max_results: int, proxy: str | None,
             proxy_headers: dict[str, str] | None) -> httpx.Response:
        try:
            return httpx.get(
                url,
                params={"q": query, "api_key": self.api_key, "num": max_results, "engine": "google"},
                timeout=bounded_timeout(),
                trust_env=False,
                proxy=httpx.Proxy(proxy, headers=proxy_headers) if proxy and proxy_headers else proxy,
                verify=network.ssl_context(),
            )
        except httpx.ProxyError as exc:
            if "407" in str(exc) and proxy_headers is not None:
                raise httpx.ProxyError("407 integrated sign-in refused") from exc
            if "407" in str(exc) and network.integrated_signin_enabled(proxy):
                return httpx.Response(407)  # CONNECT refused: let the caller sign in
            raise

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        url = "https://serpapi.com/search.json"
        try:
            proxy = network.proxy_for(url)
            resp = self._get(url, query, max_results, proxy, None)
            if resp.status_code == 407 and network.integrated_signin_enabled(proxy):
                # the same integrated Windows sign-in as page retrieval: only after the proxy asked for it
                resp = self._get(url, query, max_results, proxy, network.proxy_signin_header(proxy))
                if resp.status_code == 407:
                    raise httpx.ProxyError("407 integrated sign-in refused")
            if resp.status_code == 407:
                raise httpx.ProxyError("407 Proxy Authentication Required")
            resp.raise_for_status()
            rows = resp.json().get("organic_results", [])
        except Exception as exc:  # noqa: BLE001
            self.failed_searches += 1
            self.last_error = network.classify_search_error(exc)
            # never the exception text: httpx errors contain the request URL, including the API key
            log.warning("search_failed: provider=serpapi error=%s reason=%s", type(exc).__name__, self.last_error)
            return []
        self.last_error = None
        return [
            SearchResult(title=r.get("title", ""), url=r.get("link", ""), snippet=r.get("snippet", ""), query=query)
            for r in rows[:max_results]
            if r.get("link")
        ]


def get_search_provider() -> SearchProvider:
    """Search providers receive company names and query terms, never document content."""
    settings = get_settings()
    if not settings.enable_web_search:
        return NullSearchProvider()
    name = settings.search_provider.lower()
    if name == "ddgs":
        return DDGSSearchProvider()
    if name == "duckduckgo_lite":
        return DuckDuckGoLiteSearchProvider()
    if name == "serpapi" and settings.serpapi_api_key:
        return SerpAPISearchProvider(settings.serpapi_api_key)
    return NullSearchProvider()
