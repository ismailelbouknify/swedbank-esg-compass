"""Search provider interface + query builder."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    query: str


class SearchProvider(ABC):
    name: str = "base"

    @abstractmethod
    def search(self, query: str, max_results: int) -> list[SearchResult]:
        """Return up to max_results results. Must not raise on network errors: return []."""


class NullSearchProvider(SearchProvider):
    name = "none"

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        return []


def build_queries(company: str, year: int) -> list[str]:
    c = company.strip()
    return [
        f"{c} sustainability report {year}",
        f"{c} ESG report {year}",
        f"{c} annual report {year}",
        f"{c} sustainability report {year - 1}",
        f"{c} sustainability",
        f"{c} energy target",
        f"{c} diversity sustainability",
        f"{c} data security ESG",
    ]
