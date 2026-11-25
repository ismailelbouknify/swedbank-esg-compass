"""Small, generic text helpers shared by discovery, ingestion and source-coverage checks.

They carry no ESG interpretation: years, whitespace and whole-word term matching only."""
from __future__ import annotations

import re

YEAR_RE = re.compile(r"\b(20\d{2})\b")


def normalize_ws(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_term(text: str, term: str) -> bool:
    """Whole-word match: an acronym only in capitals ("LED" is not the verb "led"), inflections allowed."""
    if term.isupper() and len(term) <= 4 and any(c.isupper() for c in text):
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    t, low = term.lower(), text.lower()
    if len(t) <= 4 and re.fullmatch(r"[a-z0-9]+", t):
        return re.search(rf"\b{re.escape(t)}\b", low) is not None
    return re.search(rf"(?<![a-z0-9]){re.escape(t)}(?:s|es|d|ed|ing|ers?)?(?![a-z0-9])", low) is not None


def matched_terms(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms if contains_term(text, t)]
