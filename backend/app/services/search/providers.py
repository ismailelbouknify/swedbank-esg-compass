"""Concrete search providers. DDGS (DuckDuckGo) needs no API key; SerpAPI is optional."""
from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings
from app.services.search.base import NullSearchProvider, SearchProvider, SearchResult
from app.services.security import network
from app.services.security.policy import run_bounded
from app.services.security.safe_fetch import bounded_timeout

log = logging.getLogger(__name__)


class DDGSSearchProvider(SearchProvider):
    name = "ddgs"

    def search(self, query: str, max_results: int) -> list[SearchResult]:
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
            log.warning("search_failed: provider=ddgs error=%s reason=%s", type(exc).__name__, self.last_error)
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
    if name == "serpapi" and settings.serpapi_api_key:
        return SerpAPISearchProvider(settings.serpapi_api_key)
    return NullSearchProvider()
