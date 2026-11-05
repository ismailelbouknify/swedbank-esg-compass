"""Evidence verification. Before any evidence may support a recommendation we check:
- the source exists and was processed
- the quote exists in the extracted text of the stated page
- numbers / years in the structured facts appear in the source text
- the evidence is relevant to the factor and to the dimension
- the evidence subject is plausibly the assessed company (not a government or unrelated entity)
- the factor's exclusion keywords do not dominate the passage (e.g. biodiversity != D&I)
- the actor is the company, and the goal or activity is really in force or carried out
- multi-year values are demonstrably aligned with their years

Anything that fails a HARD gate is marked REJECTED (lifecycle="rejected").
Soft failures (fuzzy quote, unclear context) yield lifecycle="candidate" (REQUIRES_REVIEW).
Only lifecycle="accepted" (VERIFIED) evidence may influence questionnaire scoring."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.config.loader import OFFICIAL_SOURCE_TYPES, get_factor, load_rules
from app.services.company.source_identity import PARENT_BOUNDARY, CompanyIdentity
from app.services.extraction import gates
from app.services.extraction import patterns as P
from app.services.extraction.extractor import ACTIVITY_RE, STRONG_GOAL_RE, Candidate

VERIFIED = "VERIFIED"
REQUIRES_REVIEW = "REQUIRES_REVIEW"
REJECTED = "REJECTED"

# Lifecycle values (stored on Evidence.evidence_lifecycle)
LIFECYCLE_ACCEPTED = "accepted"
LIFECYCLE_CANDIDATE = "candidate"
LIFECYCLE_REJECTED = "rejected"


@dataclass
class VerificationResult:
    status: str
    notes: list[str] = field(default_factory=list)
    quote: str = ""
    lifecycle: str = LIFECYCLE_CANDIDATE  # accepted | candidate | rejected
    rejection_codes: list[str] = field(default_factory=list)


def _number_variants(value: float) -> set[str]:
    variants = set()
    if float(value).is_integer():
        iv = int(value)
        variants |= {str(iv), f"{iv:,}", f"{iv:,}".replace(",", " "), f"{iv:,}".replace(",", " ")}
    else:
        s = f"{value:.6f}".rstrip("0").rstrip(".")
        variants |= {s, s.replace(".", ",")}
    return variants


def number_in_text(value: float, text: str) -> bool:
    """True if the number appears in the text in any common formatting (30, 30.0, 1,234, 1 234, 12,5)."""
    variants = _number_variants(value)
    for v in variants:
        if re.search(rf"(?<![\d.,]){re.escape(v)}(?![\d]|[.,]\d)", text):
            return True
    if value == 0 and re.search(r"\b(zero|no)\b", text, re.IGNORECASE):
        return True
    return False


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


def _point_stated_with_year(year: int, value: float, text: str) -> bool:
    """The value sits right next to its year, with no other number between them."""
    number = "|".join(re.escape(v) for v in _number_variants(value))
    return bool(re.search(rf"\b{year}\b[^\d]{{0,25}}(?<![\d.,])(?:{number})(?![\d]|[.,]\d)", text)
                or re.search(rf"(?<![\d.,])(?:{number})(?![\d]|[.,]\d)[^\d]{{0,25}}\b{year}\b", text))


def series_alignment_established(points: list[list[float]], quote: str, passage_text: str) -> bool:
    """Numbers appearing on a page do not make a time series. A model-reported series must either be
    what the table parser reads under explicit year headers, or be stated value by value with its years."""
    wanted = {(int(y), float(v)) for y, v in points}
    if any(wanted <= set(s.points) for s in P.extract_series(passage_text)):
        return True
    return all(_point_stated_with_year(y, v, quote) for y, v in wanted)


def _dimension_relevant(c: Candidate, text: str) -> bool:
    if c.dimension == "planning":
        # Require forward-looking language, not just goal vocabulary.
        # A retrospective mention of a past goal is not planning evidence.
        return bool(STRONG_GOAL_RE.search(text)) and bool(gates.FORWARD_PLANNING_RE.search(text))
    if c.dimension == "execution":
        return bool(ACTIVITY_RE.search(text))
    if c.claim_type == "unquantified_improvement_claim":
        return True  # recorded for transparency only; never scored
    if c.dimension == "performance":
        return bool(re.search(r"\d", text))
    if c.dimension == "reporting":
        return True
    return False


def _check_exclusion_keywords(factor_key: str, text: str) -> bool:
    """Return True if the text is dominated by exclusion keywords for this factor.

    For example, a biodiversity passage containing 'species' and 'habitat' should
    not qualify as workforce Diversity & Inclusion evidence just because 'diversity'
    appears somewhere in the text."""
    if factor_key == "reporting":
        return False
    try:
        factor = get_factor(factor_key)
    except KeyError:
        return False
    exclusions = factor.get("exclusion_keywords", [])
    if not exclusions:
        return False
    low = text.lower()
    matched = [kw for kw in exclusions if kw.lower() in low]
    if not matched:
        return False
    # Only reject when exclusion terms appear AND none of the factor's own positive
    # keywords produce a stronger contextual match.  Specifically: if the passage
    # matches ≥2 exclusion keywords AND has fewer factor keyword hits than exclusion
    # hits, the passage is more about the excluded topic than this factor.
    factor_hits = sum(1 for kw in factor.get("keywords", []) if kw.lower() in low)
    return len(matched) >= 2 or (len(matched) >= 1 and factor_hits <= len(matched))


def _factor_terms(factor: dict[str, Any]) -> list[str]:
    return factor["keywords"] + [k for m in factor.get("metrics", []) for k in m["keywords"]]


def _factor_supported(factor: dict[str, Any], text: str) -> bool:
    """One specific factor term, or two terms that are too general to identify the factor alone."""
    hits = P.matched_terms(text, _factor_terms(factor))
    weak = {w.lower() for w in factor.get("weak_keywords", [])}
    specific = [h for h in hits if h.lower() not in weak]
    return bool(specific) or len(hits) >= 2


def _dominated_by_other_factor(factor_key: str, text: str) -> str | None:
    """The other factor this text is really about: its terms outnumber a single incidental match."""
    own = len(P.matched_terms(text, get_factor(factor_key)["keywords"]))
    if own > 1:
        return None
    for other in load_rules()["factors"]:
        if other["key"] != factor_key and len(P.matched_terms(text, other["keywords"])) >= own + 2:
            return other["key"]
    return None


def _check_company_subject(c: Candidate, text: str) -> bool:
    """Return True when the passage is plausibly about the assessed company.

    Passages where government / external legislation is clearly the grammatical
    subject — and where first-person company language is absent — are rejected
    when the source is not an official company document.

    Official company sources (sustainability reports, uploaded documents) are given
    the benefit of the doubt: their passages are presumed to be about the company.
    """
    # Official sources are presumed to discuss the company itself.
    if c.passage.source_type in OFFICIAL_SOURCE_TYPES:
        return True

    rules = load_rules()
    indicators = rules.get("company_subject_indicators", {})
    first_person = indicators.get("first_person_pronouns", [])
    govt_terms = indicators.get("government_subject_terms", [])

    low = text.lower()

    # First-person company language strongly suggests the company is the subject.
    if any(fp.lower() in low for fp in first_person):
        return True

    # If government/external terms dominate without any first-person voice, reject.
    govt_hits = sum(1 for t in govt_terms if t.lower() in low)
    if govt_hits >= 2:
        return False
    if govt_hits == 1 and not any(fp.lower() in low for fp in first_person):
        # One government term + no first-person = likely about external context.
        return False

    return True


# Failing any of these rejects the evidence outright; each failure is recorded.
HARD_GATE_CODES = {
    "exclusion_keyword_match", "company_mismatch", "external_actor", "factor_mismatch", "cross_factor_mismatch",
    "numeric_not_in_source", "ambiguous_table_alignment", "inactive_goal", "activity_not_performed", "fallback_quote_not_verbatim",
    gates.NUMERIC_SUBJECT_MISMATCH, gates.NUMERIC_CONTEXT_INVALID,
}


def _state_and_actor_codes(c: Candidate, quote: str, company: CompanyIdentity | None) -> list[str]:
    is_company = company.is_named_in if company else None
    if c.dimension == "planning":
        codes = ["inactive_goal"] if gates.has_no_active_goal(quote) else []
        if gates.external_goal_only(quote, is_company):
            codes.append("external_actor")
        value = c.facts.get("target_value")
        if value is not None and c.factor != "reporting":
            issue = gates.numeric_target_issue(quote, float(value), _number_variants(float(value)), c.facts.get("target_unit"),
                                               c.facts.get("target_year"), _factor_terms(get_factor(c.factor)), is_company)
            if issue:
                codes.append(issue)
        return codes
    if c.dimension == "execution":
        codes = ["activity_not_performed"] if gates.activity_not_performed(quote) else []
        return codes + (["external_actor"] if gates.external_actor_only(quote, ACTIVITY_RE, is_company) else [])
    if c.claim_type == "relative_change" and gates.external_actor_only(quote, P.CHANGE_VERB_RE, is_company):
        return ["external_actor"]
    if c.claim_type in ("performance_series", "relative_change"):
        scope = c.facts.get("metric_label") or quote
        return ["external_actor"] if gates.EXTERNAL_SCOPE_RE.search(scope) else []
    return []


def verify_candidate(c: Candidate, page_text: str | None, source_ok: bool = True, company: CompanyIdentity | None = None) -> VerificationResult:
    notes = list(c.notes)
    rejection_codes: list[str] = []

    if not source_ok or page_text is None:
        return VerificationResult(
            REJECTED, notes + ["Source document not available for verification."],
            c.evidence_text, LIFECYCLE_REJECTED, ["source_unavailable"],
        )

    # Reporting 'report_published' evidence is document-level: quote the title line instead.
    if c.claim_type in ("report_published", "voluntary_disclosure"):
        first = next((ln.strip() for ln in page_text.splitlines() if len(ln.strip()) > 10), page_text[:200])
        return VerificationResult(
            VERIFIED, notes + ["Document-level evidence (source was retrieved and processed)."],
            first[:400], LIFECYCLE_ACCEPTED, [],
        )

    # ── GATE 1: factor exclusion keywords ──────────────────────────────────────
    if c.factor != "reporting" and _check_exclusion_keywords(c.factor, c.evidence_text or page_text):
        rejection_codes.append("exclusion_keyword_match")
        notes.append(
            f"The passage appears to be about a topic excluded from {c.factor} evidence "
            f"(exclusion keyword matched). Not accepted."
        )

    # ── GATE 2: company subject ─────────────────────────────────────────────────
    subject_ok = _check_company_subject(c, c.evidence_text or page_text)
    if subject_ok and company and (c.passage.source_type not in OFFICIAL_SOURCE_TYPES or c.passage.identity_basis == PARENT_BOUNDARY):
        # "we" in a third-party page is its author: the company must be named next to the claim
        subject_ok = company.is_named_in(c.evidence_text) or company.is_named_in(c.passage.text)
    if not subject_ok:
        rejection_codes.append("company_mismatch")
        notes.append(
            "The passage appears to describe a government, regulatory body or external entity "
            "rather than the assessed company. Not accepted."
        )

    # ── GATE 3: quote in source ─────────────────────────────────────────────────
    found, quote, note = gates.locate_quote(c.evidence_text, page_text)
    if note:
        notes.append(note)
    ok = found
    if not found:
        rejection_codes.append("quote_not_found")
    if c.evidence_origin and P.normalize_ws(c.evidence_text) not in P.normalize_ws(page_text):
        rejection_codes.append("fallback_quote_not_verbatim")  # no fuzzy match for a sentence nobody quoted

    # ── GATE 4: numeric grounding and value-year alignment ──────────────────────
    for name, value in _numbers_to_check(c):
        if not number_in_text(value, page_text):
            ok = False
            rejection_codes.append("numeric_not_in_source")
            notes.append(f"{name}={value:g} not found in the source text.")
    if c.claim_type == "performance_series" and (
        c.facts.get("ambiguous_alignment")
        or (c.method != "regex" and not series_alignment_established(c.facts.get("points", []), c.evidence_text, c.passage.text))
    ):
        rejection_codes.append("ambiguous_table_alignment")
        notes.append("The values cannot be tied to specific years, so no time series was formed.")

    # ── GATE 5: factor relevance ────────────────────────────────────────────────
    check_text = quote if found else page_text
    if c.factor != "reporting":
        factor = get_factor(c.factor)
        label = c.facts.get("metric_label", "") if c.claim_type == "performance_series" else ""
        if not _factor_supported(factor, check_text) and not (label and P.matched_terms(label, _factor_terms(factor))):
            rejection_codes.append("factor_mismatch")
            notes.append(f"Evidence text does not mention {factor['label']} terms.")
        elif found and not label and (other := _dominated_by_other_factor(c.factor, quote)):
            rejection_codes.append("cross_factor_mismatch")
            notes.append(f"Evidence text is about {other}, not {factor['label']}.")

    # ── GATE 6: dimension relevance ─────────────────────────────────────────────
    if not _dimension_relevant(c, check_text if c.claim_type != "performance_series" else page_text):
        ok = False
        rejection_codes.append("dimension_mismatch")
        notes.append(f"Evidence text does not look relevant to the {c.dimension} dimension.")

    # ── GATE 7: actor and state of the goal or activity ─────────────────────────
    state_codes = _state_and_actor_codes(c, quote if found else c.evidence_text, company)
    rejection_codes.extend(state_codes)
    if "external_actor" in state_codes:
        notes.append("The action or goal belongs to an external actor, not the assessed company; usable as context only.")
    if {gates.NUMERIC_SUBJECT_MISMATCH, gates.NUMERIC_CONTEXT_INVALID} & set(state_codes):
        notes.append("The figure does not belong to a target of the assessed company for this topic.")
    if "inactive_goal" in state_codes:
        notes.append("The goal is discontinued, absent or only being considered, so it is not an active commitment.")
    if "activity_not_performed" in state_codes:
        notes.append("The activity was not carried out, or has not happened yet.")

    if c.method == "heuristic":
        notes.append("Facts extracted by keyword/regex fallback (local LLM not used).")

    # ── Determine final lifecycle ───────────────────────────────────────────────
    if HARD_GATE_CODES & set(rejection_codes):
        lifecycle = LIFECYCLE_REJECTED
        status = REJECTED
    elif ok:
        lifecycle = LIFECYCLE_ACCEPTED
        status = VERIFIED
    else:
        # Soft failures (e.g. paraphrased quote, single dimension hint missing).
        lifecycle = LIFECYCLE_CANDIDATE
        status = REQUIRES_REVIEW

    return VerificationResult(status, notes, quote, lifecycle, rejection_codes)


def verification_summary(results: list[VerificationResult]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts
