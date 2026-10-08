"""ORM -> response model helpers (analyst-facing; technical metadata is left out here, not deleted)."""
from __future__ import annotations

import re
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.loader import dimension_config, factor_label, source_priority
from app.models.entities import AnalystDecision, Assessment, Evidence, Recommendation, Source
from app.schemas.api import AssessmentListItem, AssessmentOut, EvidenceAuditOut, EvidenceOut, RecommendationOut, SourceOut
from app.services.presentation import SOURCE_STATUS_LABEL, analyst_notes, analyst_notices, is_technical, review_state


def public_error(error: str | None) -> str | None:
    if not error:
        return None
    # exception text ("ValueError: ...") is internal; plain messages (e.g. server restart) are shown as-is
    if re.match(r"^\w*(?:Error|Exception)", error) or is_technical(error):
        return "The analysis could not be completed. Try again, or upload the sustainability report as a PDF."
    return error


def assessment_out(a: Assessment) -> AssessmentOut:
    out = AssessmentOut.model_validate(a)
    out.notices = analyst_notices(a.warnings)
    out.error = public_error(a.error)
    return out


def review_counts(db: Session, assessment_ids: list[int]) -> dict[int, tuple[int, int, int]]:
    """assessment id -> (reviewed, total, data gaps)."""
    counts: dict[int, list[int]] = {i: [0, 0, 0] for i in assessment_ids}
    if not assessment_ids:
        return {}
    rows = db.execute(select(Recommendation.assessment_id, Recommendation.review_status, Recommendation.evidence_status)
                      .where(Recommendation.assessment_id.in_(assessment_ids)))
    for aid, rs, es in rows:
        c = counts[aid]
        c[1] += 1
        c[0] += rs in ("APPROVED", "OVERRIDDEN")
        c[2] += es == "NO_EVIDENCE_FOUND"
    return {k: (v[0], v[1], v[2]) for k, v in counts.items()}


def assessment_list_item(a: Assessment, counts: tuple[int, int, int] | None) -> AssessmentListItem:
    reviewed, total, gaps = counts or (0, 0, 0)
    return AssessmentListItem(**assessment_out(a).model_dump(), reviewed=reviewed, total=total, data_gaps=gaps)


def source_out(s: Source) -> SourceOut:
    out = SourceOut.model_validate(s)
    out.source_type_label = source_priority(s.source_type)["label"]
    out.status_label = SOURCE_STATUS_LABEL.get(s.status, s.status.title())
    out.has_file = bool(s.file_path)
    if s.document is not None:
        out.page_count = s.document.page_count
        out.report_year = s.document.report_year
    return out


def evidence_out(e: Evidence) -> EvidenceOut:
    return EvidenceOut(
        id=e.id, factor=e.factor, dimension=e.dimension, claim_text=e.claim_text, source_id=e.source_id,
        source_title=e.source_title, source_url=e.source_url, source_type=e.source.source_type if e.source else None,
        source_has_file=bool(e.source and e.source.file_path), page_number=e.page_number, evidence_text=e.evidence_text,
        verified=e.verification_status == "VERIFIED",
    )


def evidence_audit_out(e: Evidence) -> EvidenceAuditOut:
    out = EvidenceAuditOut.model_validate(e)
    out.source_type = e.source.source_type if e.source else None
    return out


def latest_comment(db: Session, r: Recommendation) -> str | None:
    if r.review_status == "PENDING":
        return None
    return db.scalar(select(AnalystDecision.comment).where(AnalystDecision.recommendation_id == r.id)
                     .order_by(AnalystDecision.id.desc()).limit(1))


def recommendation_fields(r: Recommendation, comment: str | None = None) -> dict:
    dim = dimension_config(r.dimension)
    level = r.final_level or r.recommended_level
    return {
        **{c: getattr(r, c) for c in (
            "id", "assessment_id", "factor", "dimension", "recommended_level", "max_level", "evidence_status",
            "reason", "conditions", "next_level_missing", "evidence_ids", "confidence_label", "review_status",
            "final_level", "updated_at")},
        "notes": analyst_notes(r.notes),
        "review_state": review_state(r.review_status, r.evidence_status),
        "analyst_comment": comment,
        "factor_label": factor_label(r.factor),
        "dimension_label": dim["label"],
        "level_description": dim["levels"].get(str(level), ""),
    }


def recommendation_out(r: Recommendation, comment: str | None = None) -> RecommendationOut:
    return RecommendationOut(**recommendation_fields(r, comment))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
