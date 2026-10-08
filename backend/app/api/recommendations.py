"""Recommendation detail + analyst review (approve / override)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.serializers import evidence_out, latest_comment, recommendation_fields, recommendation_out, source_out
from app.config.loader import dimension_config
from app.db.session import get_db
from app.models.entities import AnalystDecision, Assessment, Evidence, Recommendation
from app.schemas.api import DecisionOut, RecommendationDetail, RecommendationOut, ReviewRequest
from app.services.audit import log_event

router = APIRouter(prefix="/api/recommendations", tags=["recommendations"])


def _get(db: Session, rec_id: int) -> Recommendation:
    r = db.get(Recommendation, rec_id)
    if r is None:
        raise HTTPException(404, "Recommendation not found")
    return r


@router.get("/{rec_id}", response_model=RecommendationDetail)
def detail(rec_id: int, db: Session = Depends(get_db)) -> RecommendationDetail:
    r = _get(db, rec_id)
    ev = db.scalars(select(Evidence).where(Evidence.id.in_(r.evidence_ids or [-1]))).all()
    order = {eid: i for i, eid in enumerate(r.evidence_ids or [])}
    ev = sorted(ev, key=lambda e: order.get(e.id, 1_000_000))
    decisions = db.scalars(select(AnalystDecision).where(AnalystDecision.recommendation_id == r.id).order_by(AnalystDecision.id)).all()
    a = db.get(Assessment, r.assessment_id)
    return RecommendationDetail(
        **recommendation_fields(r, latest_comment(db, r)),
        evidence=[evidence_out(e) for e in ev],
        decisions=[DecisionOut.model_validate(d) for d in decisions],
        level_definitions=dimension_config(r.dimension)["levels"],
        sources_reviewed=[source_out(s) for s in sorted(a.sources, key=lambda s: (s.priority, s.id)) if s.status == "PROCESSED"],
    )


@router.post("/{rec_id}/review", response_model=RecommendationOut)
def review(rec_id: int, body: ReviewRequest, db: Session = Depends(get_db)) -> RecommendationOut:
    r = _get(db, rec_id)
    previous = r.final_level or r.recommended_level
    if body.action == "OVERRIDE":
        if body.selected_level is None:
            raise HTTPException(422, "Override requires a selected level")
        if not 1 <= body.selected_level <= r.max_level:
            raise HTTPException(422, f"Level must be between 1 and {r.max_level}")
        if not body.comment or not body.comment.strip():
            raise HTTPException(422, "Override requires an analyst comment")
        r.final_level = body.selected_level
        r.review_status = "OVERRIDDEN"
        event = "recommendation_overridden"
    else:
        r.final_level = r.recommended_level
        r.review_status = "APPROVED"
        event = "recommendation_approved"
    d = AnalystDecision(recommendation_id=r.id, assessment_id=r.assessment_id, action=body.action, previous_level=previous,
                        selected_level=r.final_level, comment=(body.comment or "").strip() or None, analyst=body.analyst)
    db.add(d)
    db.flush()
    log_event(db, event, assessment_id=r.assessment_id, entity_type="recommendation", entity_id=r.id,
              actor=body.analyst or "analyst",
              details={"previous_level": previous, "selected_level": r.final_level, "recommended_level": r.recommended_level,
                       "comment": d.comment, "decision_id": d.id})
    db.commit()
    db.refresh(r)
    return recommendation_out(r, d.comment)
