"""Pydantic request/response models."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

# SQLite returns naive datetimes; they are stored in UTC, so say so explicitly in the API.
UTCDateTime = Annotated[datetime, AfterValidator(lambda d: d if d.tzinfo else d.replace(tzinfo=timezone.utc))]


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------- requests
class AssessmentCreate(BaseModel):
    company_name: str = Field(min_length=1, max_length=255)
    year: int = Field(ge=2000, le=2100)
    website: str | None = Field(default=None, max_length=500)
    country: str | None = Field(default=None, max_length=100)
    industry: str | None = Field(default=None, max_length=150)
    organisation_number: str | None = Field(default=None, max_length=50)
    enable_web_search: bool = True
    # a completed assessment for the same company + year already exists: the analyst must
    # explicitly ask for a new analysis (it becomes a new version of that assessment)
    confirm_new: bool = False

    @field_validator("company_name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Company name is required")
        return v

    @field_validator("website", "country", "industry", "organisation_number")
    @classmethod
    def blank_to_none(cls, v: str | None) -> str | None:
        return v.strip() or None if v else None


class ReviewRequest(BaseModel):
    action: Literal["APPROVE", "OVERRIDE"]
    selected_level: int | None = Field(default=None, ge=1, le=6)
    comment: str | None = Field(default=None, max_length=4000)
    analyst: str | None = Field(default=None, max_length=255)


# ---------------------------------------------------------------- responses
class CompanyOut(ORM):
    id: int
    name: str
    website: str | None
    domain: str | None
    country: str | None
    industry: str | None
    organisation_number: str | None = None


# Analyst-facing models below deliberately omit technical metadata (model names, extraction method,
# rule ids, internal confidence scores, extraction statistics). It stays in the database, in the
# /evidence audit endpoint and in the audit trail.
class SourceOut(ORM):
    id: int
    source_type: str
    source_type_label: str | None = None
    priority: int
    title: str | None
    url: str | None
    original_filename: str | None
    is_public: bool
    is_official: bool
    status: str
    status_label: str | None = None
    page_count: int | None = None
    report_year: int | None = None
    has_file: bool = False
    created_at: UTCDateTime


class SourceDetail(SourceOut):
    assessment_id: int
    evidence_count: int = 0


class AssessmentOut(ORM):
    id: int
    year: int
    status: str
    progress: float
    error: str | None
    notices: list[str] = []
    enable_web_search: bool
    assessment_version: int = 1
    supersedes_id: int | None = None
    superseded_by_id: int | None = None
    created_at: UTCDateTime
    started_at: UTCDateTime | None
    completed_at: UTCDateTime | None
    duration_seconds: float | None = None
    company: CompanyOut


class AssessmentListItem(AssessmentOut):
    reviewed: int = 0
    total: int = 0
    data_gaps: int = 0


class AssessmentLookupOut(BaseModel):
    found: bool
    ambiguous: bool = False
    completed: AssessmentListItem | None = None
    in_progress: AssessmentOut | None = None
    candidates: list[AssessmentListItem] = []


class AssessmentSummary(BaseModel):
    assessment: AssessmentOut
    sources_reviewed: int
    sources_total: int
    sustainability_report_found: bool
    reviewed: int
    total: int
    data_gaps: int
    newer_version_id: int | None = None  # a re-run of this assessment that is in progress


class StageOut(BaseModel):
    index: int
    label: str
    state: Literal["pending", "active", "done", "failed"]


class StatusOut(BaseModel):
    id: int
    company_name: str
    year: int
    status: str
    stage: int
    stage_label: str | None
    progress: float
    error: str | None
    notices: list[str]
    stages: list[StageOut]
    started_at: UTCDateTime | None
    completed_at: UTCDateTime | None
    duration_seconds: float | None
    server_time: UTCDateTime  # lets the UI timer align with the backend clock


class EvidenceOut(BaseModel):
    """Evidence as shown to analysts: where it is, what it says, whether the quote was confirmed."""

    id: int
    factor: str
    dimension: str
    claim_text: str
    source_id: int
    source_title: str | None
    source_url: str | None
    source_type: str | None = None
    source_has_file: bool = False
    page_number: int | None
    evidence_text: str
    verified: bool


class EvidenceAuditOut(ORM):
    id: int
    factor: str
    dimension: str
    claim_type: str
    claim_text: str
    facts: dict[str, Any]
    extracted_value: float | None
    extracted_unit: str | None
    target_year: int | None
    baseline_year: int | None
    source_id: int
    source_title: str | None
    source_url: str | None
    source_type: str | None = None
    page_number: int | None
    evidence_text: str
    extraction_method: str
    model_name: str | None
    llm_confidence: float | None
    verification_status: str
    verification_notes: list[str]
    created_at: UTCDateTime


class DecisionOut(ORM):
    id: int
    recommendation_id: int
    action: str
    previous_level: int
    selected_level: int
    comment: str | None
    analyst: str | None
    created_at: UTCDateTime


class RecommendationOut(ORM):
    id: int
    assessment_id: int
    factor: str
    factor_label: str
    dimension: str
    dimension_label: str
    recommended_level: int
    max_level: int
    level_description: str
    evidence_status: str
    reason: str
    conditions: dict[str, Any]
    next_level_missing: list[str]
    notes: list[str]
    evidence_ids: list[int]
    confidence_label: str
    review_status: str
    review_state: Literal["review", "approved", "changed", "no_evidence"]
    final_level: int | None
    analyst_comment: str | None = None
    updated_at: UTCDateTime


class RecommendationDetail(RecommendationOut):
    evidence: list[EvidenceOut]
    decisions: list[DecisionOut]
    level_definitions: dict[str, str]
    sources_reviewed: list[SourceOut]


class AuditEventOut(ORM):
    id: int
    event_type: str
    entity_type: str | None
    entity_id: int | None
    actor: str
    details: dict[str, Any]
    created_at: UTCDateTime


class LLMStatusOut(BaseModel):
    available: bool
    provider: str
    model: str | None
    message: str
    heuristic_fallback: bool
