"""Assessment endpoints."""
from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.serializers import (
    assessment_list_item, assessment_out, evidence_audit_out, public_error, recommendation_out, review_counts, source_out,
    utcnow,
)
from app.config.loader import source_priority
from app.core.config import get_settings
from app.db.session import get_db
from app.models.entities import AnalystDecision, Assessment, AuditEvent, Company, Evidence, Recommendation, Source
from app.schemas.api import (
    AssessmentCreate, AssessmentListItem, AssessmentLookupOut, AssessmentOut, AssessmentSummary, AuditEventOut,
    EvidenceAuditOut, RecommendationOut, SourceOut, StageOut, StatusOut,
)
from app.services.assessments import registry
from app.services.audit import log_event
from app.services.company.identify import normalize_company_name, normalize_org_number
from app.services.documents.url_safety import UnsafeURLError, normalize_website, registrable_domain, validate_public_url
from app.services.export import exporter
from app.services.pipeline.runner import run_assessment
from app.services.presentation import EXPORT_FILE_PREFIX, PUBLIC_STAGES, public_stage

router = APIRouter(prefix="/api/assessments", tags=["assessments"])


def _get(db: Session, assessment_id: int) -> Assessment:
    a = db.get(Assessment, assessment_id)
    if a is None:
        raise HTTPException(404, "Assessment not found")
    return a


def sanitize_filename(name: str | None) -> str:
    name = (name or "report.pdf").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .") or "report.pdf"
    return name[:150]


@router.get("/lookup", response_model=AssessmentLookupOut)
def lookup(company: str = Query(..., min_length=1, max_length=255), year: int = Query(..., ge=2000, le=2100),
           organisation_number: str | None = Query(None, max_length=50), country: str | None = Query(None, max_length=100),
           db: Session = Depends(get_db)) -> AssessmentLookupOut:
    """Is there already a saved assessment for this company + year? Nothing is re-analysed here."""
    res = registry.lookup(db, company, year, organisation_number, country)
    shown = ([res.completed] if res.completed else []) + res.candidates
    counts = review_counts(db, [a.id for a in shown])
    return AssessmentLookupOut(
        found=res.completed is not None or bool(res.candidates),
        ambiguous=res.ambiguous,
        completed=assessment_list_item(res.completed, counts.get(res.completed.id)) if res.completed else None,
        in_progress=assessment_out(res.in_progress) if res.in_progress else None,
        candidates=[assessment_list_item(a, counts.get(a.id)) for a in res.candidates],
    )


def _new_version(db: Session, company: Company, year: int, enable_web_search: bool, previous: Assessment | None) -> Assessment:
    a = Assessment(company_id=company.id, year=year, enable_web_search=enable_web_search, warnings=[],
                   normalized_company_name=company.normalized_name,
                   assessment_version=(previous.assessment_version or 1) + 1 if previous else 1,
                   supersedes_id=previous.id if previous else None)
    db.add(a)
    db.flush()
    return a


@router.post("", response_model=AssessmentOut, status_code=201)
def create_assessment(body: AssessmentCreate, db: Session = Depends(get_db)) -> AssessmentOut:
    website = normalize_website(body.website)
    if website:
        try:
            validate_public_url(website, resolve_dns=False)
        except UnsafeURLError as exc:
            raise HTTPException(422, f"Invalid website: {exc}") from exc
    existing = registry.current_matches(db, body.company_name, body.year, body.organisation_number, body.country)
    running = [a for a in existing if a.status == "RUNNING"]
    if running:
        raise HTTPException(409, {"message": "An analysis for this company and year is already running.",
                                  "assessment_id": running[0].id})
    completed = [a for a in existing if a.status == "COMPLETED"]
    if completed and not body.confirm_new:
        # never create a second completed assessment for the same company + year by accident
        raise HTTPException(409, {"message": "An assessment for this company and year already exists.",
                                  "assessment_id": completed[0].id})
    company = Company(name=body.company_name, normalized_name=normalize_company_name(body.company_name), website=website,
                      domain=registrable_domain(website), country=body.country, industry=body.industry,
                      organisation_number=normalize_org_number(body.organisation_number))
    db.add(company)
    db.flush()
    # an unambiguous saved assessment is replaced by this new version once it completes
    previous = completed[0] if len({registry.identity(a) for a in completed}) == 1 else None
    a = _new_version(db, company, body.year, body.enable_web_search, previous)
    registry.abandon_unfinished(a, existing)
    log_event(db, "assessment_created", assessment_id=a.id, entity_type="assessment", entity_id=a.id, actor="analyst",
              details={**body.model_dump(), "assessment_version": a.assessment_version, "supersedes_id": a.supersedes_id})
    db.commit()
    db.refresh(a)
    return assessment_out(a)


@router.get("", response_model=list[AssessmentListItem])
def list_assessments(q: str | None = Query(None, max_length=255), limit: int = Query(200, ge=1, le=500),
                     include_archived: bool = False, db: Session = Depends(get_db)) -> list[AssessmentListItem]:
    """Saved assessments (current versions only unless include_archived), newest first."""
    query = select(Assessment).join(Company, Company.id == Assessment.company_id)
    if not include_archived:
        query = query.where(Assessment.superseded_by_id.is_(None))
    if q and q.strip():
        norm = normalize_company_name(q)
        cond = Company.name.ilike(f"%{q.strip()}%")
        if norm:
            cond = cond | Assessment.normalized_company_name.like(f"%{norm}%")
        query = query.where(cond)
    rows = list(db.scalars(query.order_by(Assessment.id.desc()).limit(limit)))
    counts = review_counts(db, [a.id for a in rows])
    return [assessment_list_item(a, counts.get(a.id)) for a in rows]


@router.post("/{assessment_id}/documents", response_model=SourceOut, status_code=201)
async def upload_document(
    assessment_id: int,
    file: UploadFile = File(...),
    is_public: bool = Form(True),
    db: Session = Depends(get_db),
) -> SourceOut:
    a = _get(db, assessment_id)
    if a.status in ("RUNNING", "COMPLETED"):
        raise HTTPException(409, "Documents can only be added before the analysis runs. Use 'Run new analysis' instead.")
    filename = sanitize_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(415, "Only PDF files are accepted")
    if file.content_type not in ("application/pdf", "application/x-pdf", "application/octet-stream"):
        raise HTTPException(415, f"Unsupported content type: {file.content_type}")
    limit = get_settings().max_pdf_size_mb * 1024 * 1024
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > limit:
            raise HTTPException(413, f"File exceeds {get_settings().max_pdf_size_mb} MB limit")
    if not bytes(data[:5]).startswith(b"%PDF"):
        raise HTTPException(415, "File is not a valid PDF")
    # stored under a generated name: the user-supplied name never touches the filesystem
    path = get_settings().upload_dir / f"{uuid.uuid4().hex}.pdf"
    path.write_bytes(bytes(data))
    pr = source_priority("uploaded_report")
    s = Source(assessment_id=a.id, source_type="uploaded_report", priority=pr["priority"], title=filename.rsplit(".", 1)[0],
               file_path=str(path), original_filename=filename, is_public=is_public, is_official=True,
               discovered_via="upload", status="DISCOVERED", content_type="application/pdf")
    db.add(s)
    db.flush()
    log_event(db, "document_uploaded", assessment_id=a.id, entity_type="source", entity_id=s.id, actor="analyst",
              details={"filename": filename, "bytes": len(data), "is_public": is_public})
    db.commit()
    db.refresh(s)
    return source_out(s)


def _start(db: Session, a: Assessment, background: BackgroundTasks) -> None:
    # the backend clock is authoritative for the timer and the stored duration
    a.status, a.stage, a.progress, a.stage_message, a.error = "RUNNING", 0, 1, "Queued", None
    a.started_at, a.completed_at, a.duration_seconds = utcnow(), None, None
    log_event(db, "analysis_started", assessment_id=a.id, actor="analyst")
    db.commit()
    background.add_task(run_assessment, a.id)


@router.post("/{assessment_id}/run", response_model=StatusOut, status_code=202)
def run(assessment_id: int, background: BackgroundTasks, db: Session = Depends(get_db)) -> StatusOut:
    """Start a new (CREATED) assessment, or retry a FAILED one. A completed assessment is never
    silently re-analysed: use /rerun, which creates a new version."""
    a = _get(db, assessment_id)
    if a.status == "RUNNING":
        raise HTTPException(409, "Assessment is already running")
    if a.status == "COMPLETED":
        raise HTTPException(409, "Assessment is already completed. Use 'Run new analysis' to analyse it again.")
    _start(db, a, background)
    return _status(a)


@router.post("/{assessment_id}/rerun", response_model=AssessmentOut, status_code=202)
def rerun(assessment_id: int, background: BackgroundTasks, db: Session = Depends(get_db)) -> AssessmentOut:
    """Run a new analysis for a saved assessment (the UI asks for confirmation first). Creates a new
    version that reuses the uploaded reports; the saved result is replaced only when it completes."""
    prev = _get(db, assessment_id)
    if prev.superseded_by_id is not None:
        raise HTTPException(409, "This is an archived version. Open the current assessment instead.")
    newer = db.scalars(select(Assessment).where(Assessment.supersedes_id == prev.id,
                                                Assessment.status.in_(("CREATED", "RUNNING")))).first()
    if newer is not None or prev.status == "RUNNING":
        raise HTTPException(409, {"message": "A new analysis is already running.", "assessment_id": (newer or prev).id})
    c = prev.company
    company = Company(name=c.name, normalized_name=c.normalized_name, website=c.website, domain=c.domain, country=c.country,
                      industry=c.industry, organisation_number=c.organisation_number)
    db.add(company)
    db.flush()
    if prev.status == "COMPLETED":
        a = _new_version(db, company, prev.year, prev.enable_web_search, prev)
    else:  # a failed / never-run attempt holds no result worth keeping: it is replaced right away
        base = db.get(Assessment, prev.supersedes_id) if prev.supersedes_id else None
        a = _new_version(db, company, prev.year, prev.enable_web_search, base)
        prev.superseded_by_id = a.id
    for s in prev.sources:
        if s.source_type == "uploaded_report" and s.file_path:
            db.add(Source(assessment_id=a.id, source_type=s.source_type, priority=s.priority, title=s.title,
                          file_path=s.file_path, original_filename=s.original_filename, is_public=s.is_public,
                          is_official=s.is_official, discovered_via="upload", status="DISCOVERED", content_type=s.content_type))
    log_event(db, "assessment_created", assessment_id=a.id, entity_type="assessment", entity_id=a.id, actor="analyst",
              details={"rerun_of": prev.id, "assessment_version": a.assessment_version})
    log_event(db, "new_analysis_requested", assessment_id=prev.id, entity_type="assessment", entity_id=prev.id,
              actor="analyst", details={"new_assessment_id": a.id})
    _start(db, a, background)
    db.refresh(a)
    return assessment_out(a)


def _status(a: Assessment) -> StatusOut:
    current = public_stage(a.stage)
    stages = []
    for i, label in enumerate(PUBLIC_STAGES, start=1):
        if a.status == "COMPLETED" or i < current:
            state = "done"
        elif i == current:
            state = "failed" if a.status == "FAILED" else "active"
        else:
            state = "pending"
        stages.append(StageOut(index=i, label=label, state=state))
    return StatusOut(id=a.id, company_name=a.company.name, year=a.year, status=a.status, stage=current,
                     stage_label=PUBLIC_STAGES[current - 1] if current else "Starting analysis",
                     progress=a.progress, error=public_error(a.error), notices=assessment_out(a).notices, stages=stages,
                     started_at=a.started_at, completed_at=a.completed_at, duration_seconds=a.duration_seconds,
                     server_time=utcnow())


@router.get("/{assessment_id}/status", response_model=StatusOut)
def status(assessment_id: int, db: Session = Depends(get_db)) -> StatusOut:
    return _status(_get(db, assessment_id))


@router.get("/{assessment_id}", response_model=AssessmentSummary)
def get_assessment(assessment_id: int, db: Session = Depends(get_db)) -> AssessmentSummary:
    """Load a saved assessment. Pure read: nothing is searched, extracted or scored again."""
    a = _get(db, assessment_id)
    processed = [s for s in a.sources if s.status == "PROCESSED"]
    reviewed, total, gaps = review_counts(db, [a.id]).get(a.id, (0, 0, 0))
    newer = db.scalar(select(Assessment.id).where(Assessment.supersedes_id == a.id,
                                                  Assessment.status.in_(("CREATED", "RUNNING"))).limit(1))
    return AssessmentSummary(
        assessment=assessment_out(a),
        sources_reviewed=len(processed),
        sources_total=len(a.sources),
        sustainability_report_found=any(s.source_type in ("uploaded_report", "sustainability_report", "annual_report") for s in processed),
        reviewed=reviewed, total=total, data_gaps=gaps, newer_version_id=newer,
    )


@router.get("/{assessment_id}/recommendations", response_model=list[RecommendationOut])
def recommendations(assessment_id: int, db: Session = Depends(get_db)) -> list[RecommendationOut]:
    _get(db, assessment_id)
    recs = db.scalars(select(Recommendation).where(Recommendation.assessment_id == assessment_id).order_by(Recommendation.id)).all()
    comments: dict[int, str | None] = {}
    for rid, comment in db.execute(select(AnalystDecision.recommendation_id, AnalystDecision.comment)
                                   .where(AnalystDecision.assessment_id == assessment_id).order_by(AnalystDecision.id)):
        comments[rid] = comment  # latest decision wins
    return [recommendation_out(r, comments.get(r.id) if r.review_status != "PENDING" else None) for r in recs]


@router.get("/{assessment_id}/evidence", response_model=list[EvidenceAuditOut])
def evidence(assessment_id: int, factor: str | None = None, dimension: str | None = None,
             db: Session = Depends(get_db)) -> list[EvidenceAuditOut]:
    """Full evidence records incl. extraction metadata, for audit/debugging. The analyst UI does not use it."""
    _get(db, assessment_id)
    q = select(Evidence).where(Evidence.assessment_id == assessment_id)
    if factor:
        q = q.where(Evidence.factor == factor)
    if dimension:
        q = q.where(Evidence.dimension == dimension)
    return [evidence_audit_out(e) for e in db.scalars(q.order_by(Evidence.id))]


@router.get("/{assessment_id}/sources", response_model=list[SourceOut])
def sources(assessment_id: int, db: Session = Depends(get_db)) -> list[SourceOut]:
    a = _get(db, assessment_id)
    return [source_out(s) for s in sorted(a.sources, key=lambda s: (s.priority, s.id))]


@router.get("/{assessment_id}/audit", response_model=list[AuditEventOut])
def audit(assessment_id: int, db: Session = Depends(get_db)) -> list[AuditEvent]:
    _get(db, assessment_id)
    return list(db.scalars(select(AuditEvent).where(AuditEvent.assessment_id == assessment_id).order_by(AuditEvent.id)))


@router.delete("/{assessment_id}", status_code=204)
def delete_assessment(assessment_id: int, db: Session = Depends(get_db)) -> Response:
    """Permanently delete a saved assessment, including its earlier (archived) versions."""
    a = _get(db, assessment_id)
    versions = registry.version_chain(db, a)
    if any(v.status == "RUNNING" for v in versions):
        raise HTTPException(409, "An analysis of this assessment is running. Wait until it has finished before deleting.")
    summary = {"company": a.company.name, "organisation_number": a.company.organisation_number, "year": a.year,
               "deleted_assessment_ids": [v.id for v in versions]}
    removed = registry.delete_assessments(db, versions)
    # the assessment's own audit trail goes with it; this record of the deletion is kept
    log_event(db, "assessment_deleted", assessment_id=None, entity_type="assessment", entity_id=a.id, actor="analyst",
              details={**summary, "files_removed": len(removed)})
    db.commit()
    for path in removed:  # only after the database change is committed
        try:
            Path(path).unlink()
        except OSError:
            pass
    return Response(status_code=204)


@router.get("/{assessment_id}/export")
def export(assessment_id: int, format: str = Query("json", pattern="^(json|csv|xlsx)$"), db: Session = Depends(get_db)) -> Response:
    a = _get(db, assessment_id)
    data = exporter.build_export(db, a)
    slug = re.sub(r"[^a-z0-9]+", "-", a.company.name.lower()).strip("-") or "company"
    name = f"{EXPORT_FILE_PREFIX}-{slug}-{a.year}"
    if format == "json":
        body, media = exporter.to_json(data), "application/json"
    elif format == "csv":
        body, media = exporter.to_csv(data), "text/csv; charset=utf-8"
    else:
        body, media = exporter.to_xlsx(data), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    log_event(db, "export_generated", assessment_id=a.id, actor="analyst", details={"format": format, "bytes": len(body)})
    db.commit()
    return Response(body, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}.{format}"'})
