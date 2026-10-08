"""Concrete search providers. DDGS (DuckDuckGo) needs no API key; SerpAPI is optional."""
from __future__ import annotations

import logging

import httpx

from app.core.config import get_settings
from app.services.search.base import NullSearchProvider, SearchProvider, SearchResult

log = logging.getLogger(__name__)


class DDGSSearchProvider(SearchProvider):
    name = "ddgs"

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            try:
                from ddgs import DDGS
            except ImportError:  # older package name
                from duckduckgo_search import DDGS  # type: ignore
            with DDGS() as ddgs:
                rows = list(ddgs.text(query, max_results=max_results) or [])
        except Exception as exc:  # noqa: BLE001
            log.warning("DDGS search failed for %r: %s", query, exc)
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
                timeout=get_settings().http_timeout_seconds,
            )
            resp.raise_for_status()
            rows = resp.json().get("organic_results", [])
        except Exception as exc:  # noqa: BLE001
            log.warning("SerpAPI search failed for %r: %s", query, exc)
            return []
        return [
            SearchResult(title=r.get("title", ""), url=r.get("link", ""), snippet=r.get("snippet", ""), query=query)
            for r in rows[:max_results]
            if r.get("link")
        ]


def get_search_provider() -> SearchProvider:
    settings = get_settings()
    name = settings.search_provider.lower()
    if name == "ddgs":
        return DDGSSearchProvider()
    if name == "serpapi" and settings.serpapi_api_key:
        return SerpAPISearchProvider(settings.serpapi_api_key)
    return NullSearchProvider()
