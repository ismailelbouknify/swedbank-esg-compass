"""Saved-assessment identity, lookup and versioning.

Identity of an assessment = (company identity, assessment year). Company identity is the
organisation number when known, otherwise the normalized company name ("Google Inc." == "google").
Two candidates with different organisation numbers or different countries are never merged: the
lookup reports them as ambiguous and the analyst decides.

"Run new analysis" never overwrites a saved result in place. It creates a new version
(assessment_version + 1, supersedes_id -> previous). The previous version stays the current one
until the new version completes; only then is it marked superseded (superseded_by_id), so a failed
re-run never destroys a completed assessment or its analyst decisions."""
from __future__ import annotations

from dataclasses import dataclass, field

from pathlib import Path

from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.entities import AnalystDecision, Assessment, AuditEvent, Company, Evidence, Recommendation, Source
from app.services.company.identify import normalize_company_name, normalize_org_number


@dataclass
class LookupResult:
    completed: Assessment | None = None  # current completed assessment for this identity + year
    in_progress: Assessment | None = None  # a newer version being created / analysed
    candidates: list[Assessment] = field(default_factory=list)  # current completed ones, when ambiguous

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1


def _same(a: str | None, b: str | None) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()


def _conflicts(c: Company, org: str | None, country: str | None) -> bool:
    if org and c.organisation_number and c.organisation_number != org:
        return True
    if country and c.country and not _same(country, c.country):
        return True
    return False


def identity(a: Assessment) -> str:
    return f"org:{a.company.organisation_number}" if a.company.organisation_number else f"name:{a.normalized_company_name}"


def current_matches(db: Session, company_name: str, year: int, organisation_number: str | None = None,
                    country: str | None = None) -> list[Assessment]:
    """Non-superseded assessments that plausibly are the same company + year, newest first."""
    norm = normalize_company_name(company_name)
    org = normalize_org_number(organisation_number)
    cond = [Assessment.normalized_company_name == norm]
    if org:
        cond.append(Company.organisation_number == org)
    q = (select(Assessment).join(Company, Company.id == Assessment.company_id)
         .where(Assessment.year == year, Assessment.superseded_by_id.is_(None), or_(*cond))
         .order_by(Assessment.id.desc()))
    rows = [a for a in db.scalars(q) if not _conflicts(a.company, org, country)]
    if org and any(a.company.organisation_number == org for a in rows):
        rows = [a for a in rows if a.company.organisation_number == org]  # an exact org-number match wins
    return rows


def lookup(db: Session, company_name: str, year: int, organisation_number: str | None = None,
           country: str | None = None) -> LookupResult:
    rows = current_matches(db, company_name, year, organisation_number, country)
    completed = [a for a in rows if a.status == "COMPLETED"]
    by_identity: dict[str, Assessment] = {}
    for a in completed:
        by_identity.setdefault(identity(a), a)  # newest per identity
    res = LookupResult()
    if len(by_identity) > 1:
        res.candidates = list(by_identity.values())
        return res
    res.completed = completed[0] if completed else None
    res.in_progress = next((a for a in rows if a.status in ("CREATED", "RUNNING")
                            and (res.completed is None or a.id > res.completed.id)), None)
    return res


def abandon_unfinished(new: Assessment, previous: list[Assessment]) -> None:
    """Earlier CREATED/FAILED attempts for the same identity are replaced by the new one right away."""
    for a in previous:
        if a.id != new.id and a.status in ("CREATED", "FAILED"):
            a.superseded_by_id = new.id


def supersede_previous(db: Session, a: Assessment) -> Assessment | None:
    """Called when a new version completes: the version it replaces is archived (kept, not deleted)."""
    if not a.supersedes_id:
        return None
    prev = db.get(Assessment, a.supersedes_id)
    if prev is not None and prev.superseded_by_id is None:
        prev.superseded_by_id = a.id
    return prev


def version_chain(db: Session, a: Assessment) -> list[Assessment]:
    """All versions linked to this assessment (earlier archived ones and any newer re-run)."""
    seen: dict[int, Assessment] = {a.id: a}
    todo = [a]
    while todo:
        cur = todo.pop()
        linked = [cur.supersedes_id, cur.superseded_by_id]
        linked += list(db.scalars(select(Assessment.id).where(or_(Assessment.supersedes_id == cur.id,
                                                                 Assessment.superseded_by_id == cur.id))))
        for i in linked:
            if i and i not in seen and (nxt := db.get(Assessment, i)) is not None:
                seen[i] = nxt
                todo.append(nxt)
    return sorted(seen.values(), key=lambda x: x.id)


def delete_assessments(db: Session, versions: list[Assessment]) -> list[str]:
    """Permanently delete assessments with their sources, evidence, recommendations, analyst decisions
    and audit events. Stored PDFs are removed once no remaining source uses them. Returns removed files."""
    ids = [a.id for a in versions]
    company_ids = {a.company_id for a in versions}
    files = {s.file_path for a in versions for s in a.sources if s.file_path}
    db.execute(delete(AnalystDecision).where(AnalystDecision.assessment_id.in_(ids)))
    db.execute(delete(Recommendation).where(Recommendation.assessment_id.in_(ids)))
    db.execute(delete(Evidence).where(Evidence.assessment_id.in_(ids)))
    db.execute(delete(AuditEvent).where(AuditEvent.assessment_id.in_(ids)))
    for a in versions:
        db.delete(a)  # cascades to sources -> documents -> pages/chunks
    db.flush()
    for cid in company_ids:
        if db.scalar(select(Assessment.id).where(Assessment.company_id == cid).limit(1)) is None:
            db.execute(delete(Company).where(Company.id == cid))
    removed: list[str] = []
    data_dir = Path(get_settings().data_dir).resolve()
    for f in files:
        if db.scalar(select(Source.id).where(Source.file_path == f).limit(1)) is not None:
            continue  # still used by another assessment
        path = Path(f).resolve()
        if path.is_relative_to(data_dir) and path.is_file():
            removed.append(str(path))
    return removed
