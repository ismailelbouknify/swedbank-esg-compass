"""Database entities. Evidence is the central entity: every recommendation points
at evidence IDs (or is explicitly NO_EVIDENCE_FOUND)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    normalized_name: Mapped[str] = mapped_column(String(255), index=True)
    website: Mapped[str | None] = mapped_column(String(500))
    domain: Mapped[str | None] = mapped_column(String(255))
    country: Mapped[str | None] = mapped_column(String(100))
    industry: Mapped[str | None] = mapped_column(String(150))
    # preferred stable identity when known (normalized: digits/letters only)
    organisation_number: Mapped[str | None] = mapped_column(String(50), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Assessment(Base):
    __tablename__ = "assessments"

    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"))
    year: Mapped[int] = mapped_column(Integer)
    # identity used to find saved assessments (normalized company name + year, or organisation number)
    normalized_company_name: Mapped[str | None] = mapped_column(String(255))
    # "Run new analysis" creates a new version; the previous one is superseded once the new one completes
    assessment_version: Mapped[int] = mapped_column(Integer, default=1)
    supersedes_id: Mapped[int | None] = mapped_column(Integer)
    superseded_by_id: Mapped[int | None] = mapped_column(Integer, index=True)
    enable_web_search: Mapped[bool] = mapped_column(Boolean, default=True)

    # CREATED | RUNNING | COMPLETED | FAILED
    status: Mapped[str] = mapped_column(String(20), default="CREATED")
    stage: Mapped[int] = mapped_column(Integer, default=0)
    stage_message: Mapped[str | None] = mapped_column(Text)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text)
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)

    llm_provider: Mapped[str | None] = mapped_column(String(50))
    llm_model: Mapped[str | None] = mapped_column(String(255))
    extraction_mode: Mapped[str | None] = mapped_column(String(20))  # llm | heuristic

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[float | None] = mapped_column(Float)  # total analysis time (backend clock)

    company: Mapped[Company] = relationship()
    sources: Mapped[list[Source]] = relationship(back_populates="assessment", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_assessments_identity_year", "normalized_company_name", "year"),)


class Source(Base):
    """An origin of information: an uploaded file or a URL."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    assessment_id: Mapped[int] = mapped_column(ForeignKey("assessments.id"), index=True)
    # uploaded_report | sustainability_report | annual_report | company_website |
    # regulatory_source | external_source | other_web
    source_type: Mapped[str] = mapped_column(String(40))
    priority: Mapped[int] = mapped_column(Integer)  # 1 = best
    title: Mapped[str | None] = mapped_column(String(500))
    url: Mapped[str | None] = mapped_column(String(2000))
    file_path: Mapped[str | None] = mapped_column(String(1000))
    original_filename: Mapped[str | None] = mapped_column(String(500))
    is_public: Mapped[bool] = mapped_column(Boolean, default=True)
    is_official: Mapped[bool] = mapped_column(Boolean, default=False)
    discovered_via: Mapped[str | None] = mapped_column(String(255))  # search query / upload / link
    # DISCOVERED | DOWNLOADED | PROCESSED | OCR_REQUIRED | SKIPPED | FAILED
    status: Mapped[str] = mapped_column(String(20), default="DISCOVERED")
    status_detail: Mapped[str | None] = mapped_column(Text)
    content_type: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    assessment: Mapped[Assessment] = relationship(back_populates="sources")
    document: Mapped[Document | None] = relationship(back_populates="source", uselist=False, cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"), index=True)
    title: Mapped[str | None] = mapped_column(String(500))
    doc_kind: Mapped[str] = mapped_column(String(10))  # pdf | html
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    report_year: Mapped[int | None] = mapped_column(Integer)
    ocr_required: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    source: Mapped[Source] = relationship(back_populates="document")
    pages: Mapped[list[DocumentPage]] = relationship(cascade="all, delete-orphan", order_by="DocumentPage.page_number")
    chunks: Mapped[list[DocumentChunk]] = relationship(cascade="all, delete-orphan")


class DocumentPage(Base):
    """Full page text, kept so quotes and numbers can be verified against the source."""

    __tablename__ = "document_pages"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    page_number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    page_number: Mapped[int | None] = mapped_column(Integer)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(primary_key=True)
    assessment_id: Mapped[int] = mapped_column(ForeignKey("assessments.id"), index=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"))

    factor: Mapped[str] = mapped_column(String(80))  # factor key or "reporting"
    dimension: Mapped[str] = mapped_column(String(30))  # reporting|planning|execution|performance

    claim_type: Mapped[str] = mapped_column(String(60))
    claim_text: Mapped[str] = mapped_column(Text)
    facts: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # structured facts

    extracted_value: Mapped[float | None] = mapped_column(Float)
    extracted_unit: Mapped[str | None] = mapped_column(String(50))
    target_year: Mapped[int | None] = mapped_column(Integer)
    baseline_year: Mapped[int | None] = mapped_column(Integer)

    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    chunk_id: Mapped[int | None] = mapped_column(ForeignKey("document_chunks.id"))
    source_title: Mapped[str | None] = mapped_column(String(500))
    source_url: Mapped[str | None] = mapped_column(String(2000))
    page_number: Mapped[int | None] = mapped_column(Integer)
    evidence_text: Mapped[str] = mapped_column(Text)  # verbatim quote from the source

    extraction_method: Mapped[str] = mapped_column(String(20))  # llm | heuristic | regex
    model_name: Mapped[str | None] = mapped_column(String(255))
    llm_confidence: Mapped[float | None] = mapped_column(Float)
    # VERIFIED | REQUIRES_REVIEW | REJECTED
    verification_status: Mapped[str] = mapped_column(String(20), default="REQUIRES_REVIEW")
    verification_notes: Mapped[list[str]] = mapped_column(JSON, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    source: Mapped[Source] = relationship()


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(primary_key=True)
    assessment_id: Mapped[int] = mapped_column(ForeignKey("assessments.id"), index=True)
    factor: Mapped[str] = mapped_column(String(80))
    dimension: Mapped[str] = mapped_column(String(30))

    recommended_level: Mapped[int] = mapped_column(Integer)
    max_level: Mapped[int] = mapped_column(Integer)
    # FOUND | NO_EVIDENCE_FOUND | CONFLICTING_EVIDENCE | REQUIRES_REVIEW
    evidence_status: Mapped[str] = mapped_column(String(30))
    triggered_rule: Mapped[str] = mapped_column(String(120))
    reason: Mapped[str] = mapped_column(Text)
    conditions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    next_level_missing: Mapped[list[str]] = mapped_column(JSON, default=list)
    notes: Mapped[list[str]] = mapped_column(JSON, default=list)
    evidence_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    confidence_label: Mapped[str] = mapped_column(String(10), default="Low")
    confidence_breakdown: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    # PENDING | APPROVED | OVERRIDDEN
    review_status: Mapped[str] = mapped_column(String(20), default="PENDING")
    final_level: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class AnalystDecision(Base):
    __tablename__ = "analyst_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    recommendation_id: Mapped[int] = mapped_column(ForeignKey("recommendations.id"), index=True)
    assessment_id: Mapped[int] = mapped_column(ForeignKey("assessments.id"), index=True)
    action: Mapped[str] = mapped_column(String(20))  # APPROVE | OVERRIDE
    previous_level: Mapped[int] = mapped_column(Integer)
    selected_level: Mapped[int] = mapped_column(Integer)
    comment: Mapped[str | None] = mapped_column(Text)
    analyst: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    assessment_id: Mapped[int | None] = mapped_column(ForeignKey("assessments.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(60))
    entity_type: Mapped[str | None] = mapped_column(String(40))
    entity_id: Mapped[int | None] = mapped_column(Integer)
    actor: Mapped[str] = mapped_column(String(100), default="system")
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
