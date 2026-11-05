"""The one place that decides whether a source may contribute evidence. Runs on web sources as soon as they
are read, so a wrong-company document costs no further downloads or extraction, and once more over all sources
before retrieval."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.loader import source_priority
from app.models.entities import Assessment, Document, DocumentPage, Source
from app.services.audit import log_event
from app.services.company.identify import classify_source
from app.services.company.resolution import AMBIGUOUS, identity_from_profile
from app.services.company.source_identity import (
    FIRST_PARTY_BASES, MISMATCH, PARENT_BOUNDARY, SOURCE_COMPANY_MISMATCH, CompanyIdentity, assess_source_identity, looks_like_company_report,
    on_supplied_domain,
)
from app.services.documents.url_safety import registrable_domain


def source_blocks_evidence(source: Source | None) -> bool:
    """Invariant: nothing from such a source may be accepted, cited in a recommendation or scored."""
    return source is None or source.status == "REJECTED" or source.company_match == MISMATCH


def identity_for(a: Assessment) -> CompanyIdentity:
    return identity_from_profile(a.company.name, a.company.organisation_number, a.company_identity)


def record_blocked_pdf_identity(db: Session, a: Assessment, s: Source, title: str | None, text: str) -> None:
    """A report refused for its length was still opened, so its first pages can say whose report it is."""
    if not text.strip():
        return
    company = a.company
    full_title = title or ""
    result = assess_source_identity(identity_for(a), title=full_title, url=s.url, text=text,
                                    on_official_domain=on_supplied_domain(s.url, company.name, company.website, company.domain))
    s.company_match = result.basis
    if not result.matched:
        s.rejection_reason = SOURCE_COMPANY_MISMATCH
        log_event(db, SOURCE_COMPANY_MISMATCH, assessment_id=a.id, entity_type="source", entity_id=s.id,
                  details={"reason": SOURCE_COMPANY_MISMATCH, "detail": result.detail[:300], "url": s.url})
    elif result.basis != PARENT_BOUNDARY and looks_like_company_report(full_title, text):
        s.source_type = "sustainability_report"
        s.priority = source_priority(s.source_type)["priority"]


def apply_source_identity(db: Session, a: Assessment, s: Source) -> bool:
    """Record the identity decision on the source. Returns False when it was rejected."""
    db.flush()
    # queried, not read from the relationship: right after ingestion that attribute can still hold an old None
    document = db.scalar(select(Document).where(Document.source_id == s.id))
    if document is None:
        return True
    company = a.company
    text = "\n".join(t for (t,) in db.execute(
        select(DocumentPage.text).where(DocumentPage.document_id == document.id).order_by(DocumentPage.page_number)))
    title = " ".join(t for t in (document.title, s.original_filename) if t)
    result = assess_source_identity(identity_for(a), title=title, url=s.url, text=text,
                                    uploaded=s.source_type == "uploaded_report",
                                    on_official_domain=on_supplied_domain(s.url, company.name, company.website, company.domain))
    s.company_match, s.rejection_reason = result.basis, None
    if (a.company_identity or {}).get("status") == AMBIGUOUS and s.source_type != "uploaded_report" and result.basis != "organisation_number":
        # no entity has been chosen, so nothing found on the web can be attributed to "the" company
        s.status, s.company_match, s.rejection_reason = "REJECTED", MISMATCH, AMBIGUOUS
        s.status_detail = "The company name could not be resolved to one company."
        log_event(db, AMBIGUOUS, assessment_id=a.id, entity_type="source", entity_id=s.id, details={"reason": AMBIGUOUS, "url": s.url})
        return False
    if not result.matched:
        s.status, s.status_detail, s.rejection_reason = "REJECTED", result.detail[:500], SOURCE_COMPANY_MISMATCH
        log_event(db, SOURCE_COMPANY_MISMATCH, assessment_id=a.id, entity_type="source", entity_id=s.id,
                  details={"reason": SOURCE_COMPANY_MISMATCH, "detail": result.detail[:300], "url": s.url})
        return False
    if result.basis == PARENT_BOUNDARY:
        return True
    if s.source_type == "other_web" and result.basis in FIRST_PARTY_BASES:
        # its own web address ties it to the company: an official source, whose passages inherit that identity
        s.source_type, s.is_official = classify_source(s.url, document.title or "", "", company.name, registrable_domain(s.url))
        s.priority = source_priority(s.source_type)["priority"]
    if s.source_type in ("company_website", "investor_relations", "policy_page", "other_web") and document.doc_kind == "pdf" and looks_like_company_report(title, text):
        # judged from the document itself: its web address may not have said it is a report
        s.source_type = "sustainability_report"
        s.priority = source_priority(s.source_type)["priority"]
    return True
