"""Keyword/BM25 retrieval over document chunks (no vector DB needed for the MVP)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from app.config.loader import dimension_config, get_factor, load_rules

TOKEN_RE = re.compile(r"[a-z0-9À-ɏ]+(?:[./][a-z0-9]+)*%?|%")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by", "with", "is", "are", "was",
    "were", "be", "been", "as", "at", "that", "this", "these", "those", "it", "its", "our", "we",
    "from", "has", "have", "had", "their", "they", "which", "will", "also", "not", "all", "per",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class RetrievedChunk:
    chunk_id: Any
    text: str
    page_number: int | None
    score: float
    meta: dict[str, Any]


class BM25Index:
    def __init__(self, items: list[dict[str, Any]]):
        """items: dicts with at least 'id', 'text', 'page_number'; other keys kept as meta."""
        self.items = items
        self._tokens = [tokenize(it["text"]) for it in items]
        self._bm25 = BM25Okapi(self._tokens) if items else None

    def search(self, query_terms: list[str], top_k: int = 5, must_contain_any: list[str] | None = None) -> list[RetrievedChunk]:
        if not self._bm25:
            return []
        q = [t for term in query_terms for t in tokenize(term)]
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        must = [m.lower() for m in (must_contain_any or [])]
        ranked = sorted(range(len(self.items)), key=lambda i: scores[i], reverse=True)
        out: list[RetrievedChunk] = []
        for i in ranked:
            if scores[i] <= 0:
                break
            text = self.items[i]["text"]
            if must and not contains_any(text, must):
                continue
            meta = {k: v for k, v in self.items[i].items() if k not in ("id", "text", "page_number")}
            out.append(RetrievedChunk(self.items[i]["id"], text, self.items[i].get("page_number"), float(scores[i]), meta))
            if len(out) >= top_k:
                break
        return out


def contains_any(text: str, terms: list[str]) -> bool:
    low = text.lower()
    for term in terms:
        t = term.lower()
        if len(t) <= 4 and t.isalnum():
            if re.search(rf"\b{re.escape(t)}\b", low):
                return True
        elif t in low:
            return True
    return False


def query_for(factor_key: str, dimension: str) -> tuple[list[str], list[str]]:
    """Return (query terms, must-contain terms) for a factor/dimension question."""
    dim_terms = dimension_config(dimension)["query_terms"]
    if factor_key == "reporting":
        return dim_terms, []
    factor = get_factor(factor_key)
    keywords = factor["keywords"]
    metric_terms = [k for m in factor.get("metrics", []) for k in m["keywords"]] if dimension == "performance" else []
    # factor keywords weighted double so the factor dominates the ranking
    return keywords * 2 + metric_terms + dim_terms, keywords


def all_factor_keys() -> list[str]:
    return [f["key"] for f in load_rules()["factors"]]
