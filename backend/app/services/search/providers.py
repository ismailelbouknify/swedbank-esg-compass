"""Concrete search providers. DDGS (DuckDuckGo) needs no API key; SerpAPI is optional."""
from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings
from app.services.search.base import NullSearchProvider, SearchProvider, SearchResult
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

            def _search() -> list[dict]:
                with DDGS(timeout=int(settings.http_read_timeout_seconds)) as ddgs:
                    return list(ddgs.text(query, max_results=max_results) or [])

            rows = run_bounded(_search, settings.max_source_processing_seconds)
        except Exception as exc:  # noqa: BLE001
            self.failed_searches += 1
            log.warning("search_failed: provider=ddgs error=%s", type(exc).__name__)
            return []
        return [
            SearchResult(title=r.get("title", ""), url=r.get("href") or r.get("url", ""), snippet=r.get("body", ""), query=query)
            for r in rows
            if r.get("href") or r.get("url")
        ]


class SerpAPISearchProvider(SearchProvider):
    name = "serpapi"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            resp = httpx.get(
                "https://serpapi.com/search.json",
                params={"q": query, "api_key": self.api_key, "num": max_results, "engine": "google"},
                timeout=bounded_timeout(),
                trust_env=False,
            )
            resp.raise_for_status()
            rows = resp.json().get("organic_results", [])
        except Exception as exc:  # noqa: BLE001
            self.failed_searches += 1
            # never the exception text: httpx errors contain the request URL, including the API key
            log.warning("search_failed: provider=serpapi error=%s", type(exc).__name__)
            return []
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
