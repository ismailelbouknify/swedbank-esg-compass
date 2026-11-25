"""Assessment export: JSON, CSV and Excel (openpyxl)."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.loader import dimension_config, factor_label, source_priority
from app.models.entities import AnalystDecision, Assessment, Evidence, Recommendation, Source
from app.services.presentation import (
    PRODUCT_NAME, PRODUCT_SUBTITLE, REVIEW_STATE_LABEL, SOURCE_STATUS_LABEL, analyst_notices, review_state,
)


def _iso(d: datetime | None) -> str | None:
    return d.isoformat() if d else None


def _level_text(dim: str, level: int | None) -> str | None:
    return dimension_config(dim)["levels"].get(str(level)) if level else None


def _lifecycle(e: Evidence) -> str:
    if e.evidence_lifecycle:
        return e.evidence_lifecycle
    return {"VERIFIED": "accepted", "REJECTED": "rejected"}.get(e.verification_status, "candidate")


def evidence_validation_fields(e: Evidence, used_ids: set[int]) -> dict[str, Any]:
    """Why each quotation did or did not support the AI-recommended level: the lightweight verification result."""
    codes = list(e.rejection_codes or [])
    lifecycle = _lifecycle(e)
    used = e.id in used_ids and lifecycle == "accepted"
    source = e.source.company_match if e.source else None
    return {
        "evidence_lifecycle": lifecycle,
        "rejection_codes": codes,  # verification codes of a withheld answer, e.g. quote_not_found, unknown_chunk
        "company_match": False if source == "mismatch" or "source_not_accepted" in codes else (True if source else None),
        "used_for_scoring": used,
        "scoring_exclusion_reason": None if used else (", ".join(codes) or "requires_review"),
    }


EVIDENCE_VALIDATION_COLUMNS = ["evidence_lifecycle", "rejection_codes", "company_match", "used_for_scoring",
                               "scoring_exclusion_reason"]


def build_export(db: Session, a: Assessment) -> dict[str, Any]:
    """Analyst export: company, year, scores, evidence, sources, analyst decisions and data gaps.
    Technical metadata (model, extraction method, rule ids, internal scores) stays in the database."""
    recs = db.scalars(select(Recommendation).where(Recommendation.assessment_id == a.id).order_by(Recommendation.id)).all()
    evidence = db.scalars(select(Evidence).where(Evidence.assessment_id == a.id).order_by(Evidence.id)).all()
    sources = db.scalars(select(Source).where(Source.assessment_id == a.id).order_by(Source.priority, Source.id)).all()
    decisions = db.scalars(select(AnalystDecision).where(AnalystDecision.assessment_id == a.id).order_by(AnalystDecision.id)).all()
    rec_by_id = {r.id: r for r in recs}
    latest_comment: dict[int, str | None] = {d.recommendation_id: d.comment for d in decisions}
    processed = [s for s in sources if s.status == "PROCESSED"]
    used_ids = {i for r in recs for i in r.evidence_ids or []}

    def factor_name(key: str) -> str:
        return "Reporting" if key == "reporting" else factor_label(key)

    def question(r: Recommendation) -> str:
        dim = dimension_config(r.dimension)["label"]
        return "Reporting" if r.dimension == "reporting" else f"{factor_label(r.factor)} - {dim}"

    def state(r: Recommendation) -> str:
        if r.evidence_status == "INSUFFICIENT_SOURCE_COVERAGE" and r.review_status == "PENDING":
            return "Insufficient source coverage"
        return REVIEW_STATE_LABEL[review_state(r.review_status, r.evidence_status)]

    def level(value: int | None) -> int | None:
        return value or None  # 0 means not assessed: an export must not read it as a score

    return {
        "report": {"title": PRODUCT_NAME, "subtitle": PRODUCT_SUBTITLE},
        "audit": {"coverage_metrics": a.coverage_metrics, "company_identity": a.company_identity},
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "assessment": {
            "company": a.company.name, "organisation_number": a.company.organisation_number, "website": a.company.website,
            "country": a.company.country, "industry": a.company.industry, "year": a.year,
            "status": a.status.title(), "last_analysed": _iso(a.completed_at), "analysis_duration_seconds": a.duration_seconds,
            "sources_reviewed": len(processed),
            "reviewed": sum(r.review_status != "PENDING" for r in recs), "questions": len(recs),
            "notices": analyst_notices(a.warnings),
        },
        "recommendations": [
            {
                "factor": factor_name(r.factor),
                "dimension": dimension_config(r.dimension)["label"],
                "ai_recommended_level": level(r.recommended_level),
                "final_level": level(r.final_level if r.final_level is not None else r.recommended_level),
                "coverage_state": r.coverage_state,
                "max_level": r.max_level,
                "review_status": state(r),
                "level_description": _level_text(r.dimension, r.final_level or r.recommended_level),
                "ai_reason": r.ai_reason or r.reason,
                # Legacy compat: reason = AI reason (unchanged semantics for existing callers).
                "reason": r.reason,
                "confidence": r.confidence_label, "missing_for_next_level": r.next_level_missing,
                # verified | not_verified | not_applicable: whether the AI answer's quotations were found in its sources
                "evidence_verification": {"met": "verified", "not_found": "not_verified"}.get(
                    ((r.conditions or {}).get("evidence_verification") or {}).get("status"), "not_applicable"),
                "analyst_comment": latest_comment.get(r.id) if r.review_status != "PENDING" else None,
                # Surfaced explicitly so exports clearly distinguish AI from analyst.
                "changed_by_analyst": r.review_status == "OVERRIDDEN",
                "evidence_ids": r.evidence_ids,
            }
            for r in recs
        ],
        "evidence": [
            {
                "id": e.id, "factor": factor_name(e.factor), "dimension": dimension_config(e.dimension)["label"],
                "finding": e.claim_text, "quote": e.evidence_text, "source": e.source_title, "page": e.page_number,
                "url": e.source_url, "quote_confirmed_in_source": e.verification_status == "VERIFIED",
                **evidence_validation_fields(e, used_ids),
            }
            for e in evidence
        ],
        "data_gaps": [
            {"question": question(r), "finding": r.reason, "review_status": state(r)}
            for r in recs if r.evidence_status in ("NO_EVIDENCE_FOUND", "INSUFFICIENT_SOURCE_COVERAGE")
        ],
        "sources": [
            {
                "title": s.title or s.original_filename, "type": source_priority(s.source_type)["label"], "url": s.url,
                "file": s.original_filename, "public": s.is_public,
                "status": SOURCE_STATUS_LABEL.get(s.status, s.status.title()),
                "company_match": s.company_match, "rejection_reason": s.rejection_reason,
            }
            for s in sources
        ],
        "analyst_decisions": [
            {
                "question": question(rec_by_id[d.recommendation_id]) if d.recommendation_id in rec_by_id else None,
                "decision": "Approved" if d.action == "APPROVE" else "Changed",
                "ai_recommended_level": rec_by_id[d.recommendation_id].recommended_level if d.recommendation_id in rec_by_id else None,
                "previous_level": d.previous_level, "analyst_final_level": d.selected_level, "comment": d.comment,
                "analyst": d.analyst, "date": _iso(d.created_at),
            }
            for d in decisions
        ],
    }


def to_json(data: dict[str, Any]) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=False, default=str).encode("utf-8")


RECOMMENDATION_COLUMNS = ["factor", "dimension", "ai_recommended_level", "final_level", "max_level", "review_status",
                          "changed_by_analyst", "evidence_verification", "reason", "level_description", "confidence",
                          "missing_for_next_level", "analyst_comment"]


def _cell(v: Any) -> Any:
    if isinstance(v, (list, dict)):
        return "; ".join(map(str, v)) if isinstance(v, list) else json.dumps(v, ensure_ascii=False)
    return v


def to_csv(data: dict[str, Any]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    a = data["assessment"]
    w.writerow(["company", "year"] + RECOMMENDATION_COLUMNS)
    for r in data["recommendations"]:
        w.writerow([a["company"], a["year"]] + [_cell(r.get(c)) for c in RECOMMENDATION_COLUMNS])
    return ("﻿" + buf.getvalue()).encode("utf-8")  # BOM so Excel opens UTF-8 correctly


def to_xlsx(data: dict[str, Any]) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font

    wb = Workbook()
    ws = wb.active
    ws.title = "Assessment"
    wb.properties.title = f"{PRODUCT_NAME} - {data['assessment']['company']} {data['assessment']['year']}"
    wb.properties.subject = PRODUCT_SUBTITLE
    a = data["assessment"]
    ws.append([PRODUCT_NAME])
    ws.cell(row=1, column=1).font = Font(bold=True, size=14)
    ws.append([PRODUCT_SUBTITLE])
    ws.append([])
    for k in ("company", "year", "organisation_number", "website", "country", "industry", "status", "last_analysed",
              "sources_reviewed", "reviewed", "questions"):
        ws.append([k, _cell(a.get(k))])
    ws.append([])
    ws.append(RECOMMENDATION_COLUMNS)
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True)
    for r in data["recommendations"]:
        ws.append([_cell(r.get(c)) for c in RECOMMENDATION_COLUMNS])

    def sheet(title: str, rows: list[dict[str, Any]], cols: list[str]) -> None:
        s = wb.create_sheet(title)
        s.append(cols)
        for c in s[1]:
            c.font = Font(bold=True)
        for row in rows:
            s.append([_cell(row.get(c)) for c in cols])

    sheet("Evidence", data["evidence"], ["id", "factor", "dimension", "finding", "quote", "source", "page", "url",
                                         "quote_confirmed_in_source", *EVIDENCE_VALIDATION_COLUMNS])
    sheet("Data Gaps", data["data_gaps"], ["question", "finding", "review_status"])
    sheet("Sources", data["sources"], ["title", "type", "url", "file", "public", "status", "company_match", "rejection_reason"])
    sheet("Analyst Decisions", data["analyst_decisions"], ["question", "decision", "ai_recommended_level", "previous_level",
                                                           "analyst_final_level", "comment", "analyst", "date"])
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()
