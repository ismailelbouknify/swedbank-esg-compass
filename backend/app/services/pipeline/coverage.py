"""Source coverage: can the reviewed sources support a conclusion at all? A failed search and a company with no
relevant disclosure are different findings, and only the second may be called "no evidence"."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config.loader import OFFICIAL_SOURCE_TYPES, REPORT_SOURCE_TYPES
from app.core.config import Settings
from app.models.entities import Assessment, DocumentPage
from app.services.company.resolution import AMBIGUOUS
from app.services.company.source_identity import MISMATCH, REPORT_DOCUMENT_RE, SOURCE_COMPANY_MISMATCH, _domain_labels
from app.services.pipeline.source_gate import identity_for, source_blocks_evidence
from app.services.security import policy
from app.services.security.content_validator import url_path_is_pdf

EVIDENCE_FOUND = "EVIDENCE_FOUND"
NO_EVIDENCE_FOUND = "NO_EVIDENCE_FOUND"
INSUFFICIENT_SOURCE_COVERAGE = "INSUFFICIENT_SOURCE_COVERAGE"
NOT_ASSESSED_LEVEL = 0  # stored instead of a level: an assessment gap is never a Level 1 answer

# A source says something about a "no evidence" answer only if it is about sustainability at all.
_ESG_TERMS_RE = re.compile(
    r"\b(sustainab\w*|esg|climate|emissions?|environment\w*|renewable|diversity|inclusion|human rights|working conditions|"
    r"data protection|cybersecurity|circular\w*|recycl\w*|carbon|greenhouse)\b",
    re.IGNORECASE,
)
MIN_ESG_TERMS = 2

INSUFFICIENT_COVERAGE_MESSAGE = ("Insufficient source coverage — the system could not access enough reliable company information. "
                                 "Upload the company's sustainability or annual report to continue the assessment.")
BLOCKED_REPORT_REASON = ("Insufficient source coverage \u2014 a primary company report was found, but it could not be processed "
                         "because of processing limits.")
BLOCKED_REPORT_MESSAGE = (BLOCKED_REPORT_REASON + " Upload the report to continue the assessment.")
_NOT_A_LIMIT = {policy.INVALID_CONTENT_TYPE, policy.INVALID_PDF_SIGNATURE, policy.EXTERNAL_FETCH_DISABLED}
AMBIGUOUS_COMPANY_MESSAGE = ("The company name could refer to more than one company, so their information was not combined. "
                             "Enter the company website or organisation number to identify the right one.")


@dataclass
class CoverageCheck:
    valid_sources: int = 0
    official_sources: int = 0
    relevant_official_sources: int = 0
    primary_reports: int = 0
    rejected_company: int = 0
    failed_official: int = 0
    factors_without_candidates: int = 0
    total_factors: int = 0
    min_valid_sources: int = 3
    recovery_reasons: list[str] = field(default_factory=list)
    blocked_primary_reports: list[dict[str, str | None]] = field(default_factory=list)

    @property
    def sufficient(self) -> bool:
        """Enough to say "no evidence": a real report of the company, or several of its own sustainability sources.
        Third-party pages and first-party pages on other subjects cannot show that the company disclosed nothing."""
        if self.blocked_primary_reports:
            return False  # the document most likely to hold the answer was never read
        return self.primary_reports >= 1 or self.relevant_official_sources >= self.min_valid_sources


def _is_primary_report(db: Session, s, settings: Settings) -> bool:
    """A report document rather than a page merely listing reports."""
    if s.source_type not in REPORT_SOURCE_TYPES or s.document is None:
        return False
    if s.document.doc_kind == "pdf":
        return True
    chars = db.scalar(select(func.sum(func.length(DocumentPage.text))).where(DocumentPage.document_id == s.document.id)) or 0
    return chars >= settings.coverage_min_report_page_chars


def _is_about_sustainability(db: Session, s) -> bool:
    if s.document is None:
        return False
    text = "\n".join(t for (t,) in db.execute(select(DocumentPage.text).where(DocumentPage.document_id == s.document.id)))
    return len({m.group(1).lower() for m in _ESG_TERMS_RE.finditer(text)}) >= MIN_ESG_TERMS


def _is_blocked_primary_report(s, a: Assessment) -> bool:
    """This company's report, stopped by a security or processing limit. A PDF refused for its length was
    identified from its first pages (company_match). One never fully received can only be judged from its address
    and title; a similar name is not enough, so another company's oversized report does not count."""
    code = s.fetch_outcome
    if s.rejection_reason or s.company_match == MISMATCH:
        return False
    if not code or code == "ok" or code in _NOT_A_LIMIT or (code in policy.AVAILABILITY_CODES and code != policy.SOURCE_TIMEOUT):
        return False
    if s.source_type not in REPORT_SOURCE_TYPES and not url_path_is_pdf(s.url or ""):
        return False
    described = re.sub(r"[-_]+", " ", f"{s.title or ''} {s.url or ''}")
    if s.source_type not in REPORT_SOURCE_TYPES and not REPORT_DOCUMENT_RE.search(described):
        return False
    if s.company_match:
        return s.source_type in REPORT_SOURCE_TYPES
    identity = identity_for(a)
    if _domain_labels(s.url) & {e.replace(" ", "") for e in identity.excluded}:
        return False
    named = bool(identity.full_re and identity.full_re.search(described)) or any(r.search(described) for r in identity.alias_res)
    return bool(s.is_official or named or (s.discovered_via or "").startswith("report link on"))


def assess_coverage(db: Session, a: Assessment, settings: Settings, factors_without_candidates: int, total_factors: int) -> CoverageCheck:
    db.flush()
    valid = [s for s in a.sources if s.status == "PROCESSED" and s.company_match is not None and not source_blocks_evidence(s)]
    relevant_official = [s for s in valid if s.source_type in OFFICIAL_SOURCE_TYPES and _is_about_sustainability(db, s)]
    rejected = [s for s in a.sources if s.rejection_reason in (SOURCE_COMPANY_MISMATCH, AMBIGUOUS)]
    official = [s for s in a.sources if s.source_type in OFFICIAL_SOURCE_TYPES and not s.rejection_reason]
    failed_official = [s for s in official if s.status in ("FAILED", "REJECTED", "OCR_REQUIRED")]
    check = CoverageCheck(
        valid_sources=len(valid), official_sources=sum(s.source_type in OFFICIAL_SOURCE_TYPES for s in valid),
        relevant_official_sources=len(relevant_official),
        primary_reports=sum(_is_primary_report(db, s, settings) for s in valid), rejected_company=len(rejected),
        failed_official=len(failed_official), factors_without_candidates=factors_without_candidates, total_factors=total_factors,
        min_valid_sources=settings.coverage_min_valid_sources,
        blocked_primary_reports=[{"url": s.url, "title": s.title, "reason": s.fetch_outcome} for s in a.sources if _is_blocked_primary_report(s, a)],
    )
    reasons = check.recovery_reasons
    if not valid:
        reasons.append("no_valid_company_sources")
    elif len(relevant_official) < settings.coverage_min_valid_sources:
        reasons.append("few_valid_company_sources")
    if check.primary_reports == 0:
        reasons.append("no_official_report")
    if total_factors and factors_without_candidates / total_factors > settings.coverage_max_factors_without_candidates:
        reasons.append("most_factors_without_candidates")
    judged = len(valid) + len(rejected)
    if judged and len(rejected) / judged > settings.coverage_max_company_rejection_share:
        reasons.append("most_sources_wrong_company")
    if failed_official and not check.official_sources:
        reasons.append("official_sources_failed")
    return check


def coverage_state(evidence_ids: list[int], check: CoverageCheck) -> str:
    if evidence_ids:
        return EVIDENCE_FOUND
    return NO_EVIDENCE_FOUND if check.sufficient else INSUFFICIENT_SOURCE_COVERAGE
