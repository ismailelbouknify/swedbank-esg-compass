"""Passages and retrieval.

Passages are the chunks of the assessment's accepted sources only (processed, tied to the company by the identity
check, not blocked). Retrieval is BM25 over all of them, with a query built from the question, the factor and the
level definitions. Nothing is filtered out for lacking a particular keyword or number pattern: ranking decides, then
near-duplicates are removed and the best passages that fit the model's context window are kept."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.loader import OFFICIAL_SOURCE_TYPES, source_priority
from app.services.company.source_identity import PARENT_BOUNDARY
from app.models.entities import Assessment, Document, DocumentChunk, Source
from app.services.documents.text_utils import normalize_ws
from app.services.pipeline.source_gate import source_blocks_evidence
from app.services.rag.questions import Question
from app.services.retrieval.bm25 import BM25Index, tokenize
from app.services.retrieval.chunker import approx_tokens

DUPLICATE_OVERLAP = 0.8  # share of a passage's distinct words already in a chosen passage


@dataclass(frozen=True)
class Passage:
    """One chunk, with the metadata Python (never the model) uses for citations."""
    chunk_id: int
    source_id: int
    document_id: int
    source_title: str | None
    source_url: str | None
    source_type: str
    page_number: int | None
    section: str | None
    report_year: int | None
    is_table: bool
    text: str
    # a third-party page, or a parent group's report: "we" there is not the assessed company, so a quotation from it
    # must name the company (in the quotation or its passage)
    requires_company_name: bool = False

    @property
    def label(self) -> str:
        return f"chunk_{self.chunk_id}"

    @property
    def tokens(self) -> int:
        return approx_tokens(self.text)


def accepted_sources(a: Assessment) -> dict[int, Source]:
    """Sources whose content may be used as evidence: processed, with a recorded identity decision, not blocked."""
    return {s.id: s for s in a.sources
            if s.status == "PROCESSED" and s.company_match is not None and not source_blocks_evidence(s)}


def load_passages(db: Session, a: Assessment) -> tuple[list[Passage], int]:
    """All passages of accepted sources, best source first; byte-identical passages (mirrored pages) kept once.
    Returns (passages, duplicates skipped)."""
    accepted = accepted_sources(a)
    if not accepted:
        return [], 0
    rows = db.execute(
        select(DocumentChunk, Document).join(Document, DocumentChunk.document_id == Document.id)
        .where(Document.source_id.in_(list(accepted)))
    ).all()
    rows.sort(key=lambda r: (source_priority(accepted[r[1].source_id].source_type)["priority"], r[1].source_id, r[0].chunk_index))
    out: list[Passage] = []
    seen: set[str] = set()
    duplicates = 0
    for ch, doc in rows:
        digest = hashlib.sha1(normalize_ws(ch.text).encode()).hexdigest()
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        src = accepted[doc.source_id]
        out.append(Passage(
            chunk_id=ch.id, source_id=src.id, document_id=doc.id, source_title=doc.title or src.title, source_url=src.url,
            source_type=src.source_type, page_number=ch.page_number if doc.doc_kind == "pdf" else None,
            section=ch.section_heading, report_year=doc.report_year, is_table=ch.content_kind == "table", text=ch.text,
            requires_company_name=src.source_type not in OFFICIAL_SOURCE_TYPES or src.company_match == PARENT_BOUNDARY,
        ))
    return out, duplicates


def question_query(q: Question) -> list[str]:
    """The question, the factor and the level definitions, plus the configured vocabulary of factor and dimension."""
    terms = [q.text, q.factor_label, q.dimension_label, *q.levels.values(), *q.query_terms]
    return terms + list(q.keywords) * 2  # the factor's own words weigh most: a planning question is about this factor


class Retriever:
    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.index = BM25Index([{"id": i, "text": p.text, "page_number": p.page_number} for i, p in enumerate(passages)])

    def retrieve(self, q: Question, k: int, token_budget: int, candidates: int = 40) -> tuple[list[Passage], int]:
        """Up to `k` relevant, non-overlapping passages within `token_budget`. Returns (passages, relevant candidates):
        the candidate count (passages with any query term) is a retrieval diagnostic for the coverage decision."""
        ranked = [h.chunk_id for h in self.index.search(question_query(q), top_k=candidates)]
        if not ranked and self.passages:
            # In a corpus of one or two chunks BM25's IDF is negative for every term; rank by shared query words instead.
            query = {t for term in question_query(q) for t in tokenize(term)}
            overlap = [(len(query & set(tokenize(p.text))), i) for i, p in enumerate(self.passages)]
            ranked = [i for n, i in sorted(overlap, reverse=True) if n > 0][:candidates]
        chosen: list[Passage] = []
        chosen_words: list[set[str]] = []
        used = 0
        for idx in ranked:
            p = self.passages[idx]
            words = set(tokenize(p.text))
            if words and any(len(words & w) / len(words) >= DUPLICATE_OVERLAP for w in chosen_words):
                continue  # the same content (overlap window, mirrored page) is already in the prompt
            if chosen and used + p.tokens > token_budget:
                continue
            chosen.append(p)
            chosen_words.append(words)
            used += p.tokens
            if len(chosen) >= k:
                break
        return chosen, len(ranked)
