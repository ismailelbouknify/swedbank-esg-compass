"""Evidence verification. Before any evidence may support a recommendation we check:
- the source exists and was processed
- the quote exists in the extracted text of the stated page
- numbers / years in the structured facts appear in the source text
- the evidence is relevant to the factor and to the dimension

Anything that fails is marked REQUIRES_REVIEW (never silently used)."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from app.config.loader import get_factor
from app.services.extraction import patterns as P
from app.services.extraction.extractor import ACTIVITY_RE, STRONG_GOAL_RE, Candidate

VERIFIED = "VERIFIED"
REQUIRES_REVIEW = "REQUIRES_REVIEW"
REJECTED = "REJECTED"

FUZZY_THRESHOLD = 0.9


@dataclass
class VerificationResult:
    status: str
    notes: list[str] = field(default_factory=list)
    quote: str = ""


def number_in_text(value: float, text: str) -> bool:
    """True if the number appears in the text in any common formatting (30, 30.0, 1,234, 1 234, 12,5)."""
    variants = set()
    if float(value).is_integer():
        iv = int(value)
        variants |= {str(iv), f"{iv:,}", f"{iv:,}".replace(",", " "), f"{iv:,}".replace(",", "\u00a0")}
    else:
        s = f"{value:.6f}".rstrip("0").rstrip(".")
        variants |= {s, s.replace(".", ",")}
    for v in variants:
        if re.search(rf"(?<![\d.,]){re.escape(v)}(?![\d]|[.,]\d)", text):
            return True
    if value == 0 and re.search(r"\b(zero|no)\b", text, re.IGNORECASE):
        return True
    return False


def locate_quote(quote: str, page_text: str) -> tuple[bool, str, str | None]:
    """Return (found, exact_source_text, note). Accepts whitespace/case differences and, at
    >=90% similarity, a lightly paraphrased quote (the exact source sentence is then shown)."""
    if not quote or len(quote.strip()) < 8:
        return False, quote, "Quote missing or too short."
    nq = P.normalize_ws(quote)
    npage = P.normalize_ws(page_text)
    if nq in npage:
        return True, quote, None
    best, best_ratio = None, 0.0
    candidates = P.split_sentences(page_text)
    # also consider windows of 2 consecutive sentences for multi-sentence quotes
    candidates += [a + " " + b for a, b in zip(candidates, candidates[1:])]
    for s in candidates:
        r = difflib.SequenceMatcher(None, nq, P.normalize_ws(s)).ratio()
        if r > best_ratio:
            best, best_ratio = s, r
    if best is not None and best_ratio >= FUZZY_THRESHOLD:
        return True, best, f"Quote matched source text at {best_ratio:.0%} similarity; exact source wording shown."
    return False, quote, f"Quote not found on the stated page (best similarity {best_ratio:.0%})."


def _numbers_to_check(c: Candidate) -> list[tuple[str, float]]:
    f = c.facts
    out: list[tuple[str, float]] = []
    if c.dimension == "planning":
        for k in ("target_value", "target_year", "baseline_year"):
            if f.get(k) is not None:
                out.append((k, float(f[k])))
    elif c.claim_type == "performance_series":
        for y, v in f.get("points", []):
            out.append(("year", float(y)))
            out.append(("value", float(v)))
    elif c.claim_type == "relative_change":
        out.append(("change_pct", float(f["change_pct"])))
        if f.get("base_year"):
            out.append(("base_year", float(f["base_year"])))
    elif c.dimension == "execution" and f.get("activity_year"):
        out.append(("activity_year", float(f["activity_year"])))
    return out


def _dimension_relevant(c: Candidate, text: str) -> bool:
    if c.dimension == "planning":
        return bool(STRONG_GOAL_RE.search(text))
    if c.dimension == "execution":
        return bool(ACTIVITY_RE.search(text))
    if c.claim_type == "unquantified_improvement_claim":
        return True  # recorded for transparency only; never scored
    if c.dimension == "performance":
        return bool(re.search(r"\d", text))
    if c.dimension == "reporting":
        return True
    return False


def verify_candidate(c: Candidate, page_text: str | None, source_ok: bool = True) -> VerificationResult:
    notes = list(c.notes)
    if not source_ok or page_text is None:
        return VerificationResult(REJECTED, notes + ["Source document not available for verification."], c.evidence_text)

    # Reporting 'report_published' evidence is document-level: quote the title line instead.
    if c.claim_type in ("report_published", "voluntary_disclosure"):
        first = next((ln.strip() for ln in page_text.splitlines() if len(ln.strip()) > 10), page_text[:200])
        return VerificationResult(VERIFIED, notes + ["Document-level evidence (source was retrieved and processed)."], first[:400])

    found, quote, note = locate_quote(c.evidence_text, page_text)
    if note:
        notes.append(note)
    ok = found

    for name, value in _numbers_to_check(c):
        if not number_in_text(value, page_text):
            ok = False
            notes.append(f"{name}={value:g} not found in the source text.")

    check_text = quote if found else page_text
    if c.factor != "reporting":
        factor = get_factor(c.factor)
        metric_terms = [k for m in factor.get("metrics", []) for k in m["keywords"]]
        if not P.matched_terms(check_text, factor["keywords"] + metric_terms) and not (
            c.claim_type == "performance_series" and P.matched_terms(c.facts.get("metric_label", ""), factor["keywords"] + metric_terms)
        ):
            ok = False
            notes.append(f"Evidence text does not mention {factor['label']} terms.")
    if not _dimension_relevant(c, check_text if c.claim_type != "performance_series" else page_text):
        ok = False
        notes.append(f"Evidence text does not look relevant to the {c.dimension} dimension.")

    if c.method == "heuristic":
        notes.append("Facts extracted by keyword/regex fallback (local LLM not used).")
    return VerificationResult(VERIFIED if ok else REQUIRES_REVIEW, notes, quote)


def verification_summary(results: list[VerificationResult]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts
