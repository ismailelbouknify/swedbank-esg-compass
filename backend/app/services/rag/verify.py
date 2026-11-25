"""Lightweight verification of one model answer. Generic integrity checks only, no ESG rules:

- the response is valid JSON with a known status;
- a recommended level is within the question's scale, and is never given without supporting evidence;
- every cited chunk_id is one of the passages given to the model;
- every quotation exists in the chunk it cites (only whitespace, typographic quotes/dashes, soft hyphens and
  end-of-line hyphenation are normalised; an ellipsis may join fragments that appear in order);
- the cited chunk belongs to an accepted source of the assessed company, and a quotation from a third-party page or
  from a parent group's report names the assessed company (in the quotation or its passage).

A quotation that exists in a document does not prove the level is right: that judgement stays with the analyst
(and with benchmarking). Verification only guarantees that the cited words are really in the company's sources."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

from app.services.llm.json_utils import LLMOutputError, parse_llm_json
from app.services.rag.prompt import EVIDENCE_FOUND, INSUFFICIENT_EVIDENCE
from app.services.rag.retrieval import Passage

MIN_QUOTE_WORDS = 4  # a quotation must say something: "energy" alone supports nothing
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")
_TRANSLATE = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"',
                            "„": '"', "–": "-", "—": "-", "−": "-", " ": " ", "­": ""})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_TRANSLATE)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # "manage-\nment" -> "management"
    return re.sub(r"\s+", " ", text).strip()


def _strip_quote(quote: str) -> str:
    return normalize(quote).strip(" \"'").rstrip(".;,: ")


def quote_in(quote: str, chunk_text: str) -> bool:
    """True when the quotation (or its ellipsis-separated fragments, in order) is in the chunk."""
    haystack = normalize(chunk_text)
    position = 0
    fragments = [f for f in _ELLIPSIS.split(_strip_quote(quote)) if f]
    if not fragments:
        return False
    for fragment in fragments:
        found = haystack.find(fragment, position)
        if found < 0:
            return False
        position = found + len(fragment)
    return True


@dataclass
class Quote:
    passage: Passage | None
    chunk_ref: str
    text: str
    verified: bool


@dataclass
class Verification:
    ok: bool
    status: str | None = None
    level: int | None = None
    explanation: str = ""
    quotes: list[Quote] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # plain sentences, also sent back to the model on a retry
    codes: list[str] = field(default_factory=list)  # machine-readable, for the audit trail

    def fail(self, code: str, message: str) -> None:
        self.ok = False
        self.codes.append(code)
        self.errors.append(message)


def verify_answer(raw: str, *, max_level: int, passages: list[Passage], accepted_source_ids: set[int],
                  names_company: Callable[[str], bool] | None = None) -> Verification:
    v = Verification(ok=True)
    try:
        data = parse_llm_json(raw)
    except LLMOutputError:
        v.fail("invalid_json", "The response was not a valid JSON object.")
        return v
    by_label = {p.label: p for p in passages}
    v.status = data.get("status")
    v.explanation = str(data.get("explanation") or "").strip()[:2000]
    level = data.get("recommended_level")
    evidence = data.get("evidence")
    if not isinstance(evidence, list):
        evidence = [] if evidence is None else [evidence]

    for item in evidence:
        if not isinstance(item, dict):
            v.fail("invalid_evidence", "Each evidence item must be an object with chunk_id and quote.")
            continue
        ref, quote = str(item.get("chunk_id") or "").strip(), str(item.get("quote") or "").strip()
        passage = by_label.get(ref)
        found = bool(passage) and quote_in(quote, passage.text)
        v.quotes.append(Quote(passage, ref, normalize(quote), found))

    if v.status == INSUFFICIENT_EVIDENCE:
        if level is not None:
            v.fail("level_without_evidence", "INSUFFICIENT_EVIDENCE must have recommended_level null.")
        v.quotes = []  # an abstention cites nothing
        return v
    if v.status != EVIDENCE_FOUND:
        v.fail("invalid_status", "status must be EVIDENCE_FOUND or INSUFFICIENT_EVIDENCE.")
        return v

    if isinstance(level, bool) or not isinstance(level, int):
        if isinstance(level, str) and level.strip().isdigit():
            level = int(level.strip())
        else:
            v.fail("invalid_level", f"recommended_level must be a whole number from 1 to {max_level}.")
            level = None
    if level is not None and not 1 <= level <= max_level:
        v.fail("level_out_of_range", f"recommended_level {level} is outside the scale 1 to {max_level}.")
    v.level = level
    if not v.explanation:
        v.fail("missing_explanation", "An explanation is required.")
    if not v.quotes:
        v.fail("level_without_evidence", "A recommended level needs at least one supporting quotation.")
    for q in v.quotes:
        if q.passage is None:
            v.fail("unknown_chunk", f"chunk_id {q.chunk_ref or '(empty)'} is not one of the provided passages.")
        elif q.passage.source_id not in accepted_source_ids:
            v.fail("source_not_accepted", f"{q.chunk_ref} is not from an accepted source of the assessed company.")
        elif len(q.text.split()) < MIN_QUOTE_WORDS:
            v.fail("quote_too_short", f"The quote cited from {q.chunk_ref} is too short to support a level.")
        elif not q.verified:
            v.fail("quote_not_found", f"The quote cited from {q.chunk_ref} does not appear in that passage.")
        elif (q.passage.requires_company_name and names_company is not None
              and not (names_company(q.text) or names_company(q.passage.text))):
            v.fail("company_not_named", f"{q.chunk_ref} is from a third-party or group source and does not name the "
                                        "assessed company, so it cannot show what this company does.")
    if not v.ok:
        for q in v.quotes:
            q.verified = q.verified and q.passage is not None and q.passage.source_id in accepted_source_ids
    return v
