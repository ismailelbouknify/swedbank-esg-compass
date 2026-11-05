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
    failed_searches: int = 0  # searches that ended in an error rather than in an empty answer

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


RECOVERY_SITE_TEMPLATES = ("site:{domain} sustainability report", "site:{domain} annual report", "site:{domain} ESG")
RECOVERY_TEMPLATES = (
    "{company} sustainability report {year}", "{company} annual report {year}", "{company} ESG report", "{company} environmental report",
    "{company} sustainability", "{company} investor relations annual report", "{company} climate target", "{company} diversity report",
    "{company} cybersecurity report",
)


def build_recovery_queries(company: str, aliases: list[str], domain: str | None, year: int, already_run: list[str], limit: int) -> list[str]:
    """The official domain comes first: first-party documents are what the recovery pass is looking for."""
    names = [*aliases, company]
    queries = [t.format(domain=domain) for t in RECOVERY_SITE_TEMPLATES] if domain else []
    queries += [t.format(company=names[0], year=year) for t in RECOVERY_TEMPLATES]
    done = {q.lower() for q in already_run}
    return list(dict.fromkeys(q for q in queries if q.lower() not in done))[:limit]
