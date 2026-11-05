"""Audit trail helper."""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from app.models.entities import AuditEvent
from app.services.security.policy import SourceRejection, audit_event_for

log = logging.getLogger(__name__)


def log_event(
    db: Session,
    event_type: str,
    *,
    assessment_id: int | None = None,
    entity_type: str | None = None,
    entity_id: int | None = None,
    actor: str = "system",
    details: dict[str, Any] | None = None,
    commit: bool = False,
) -> AuditEvent:
    event = AuditEvent(
        assessment_id=assessment_id,
        event_type=event_type,
        entity_type=entity_type,
        entity_id=entity_id,
        actor=actor,
        details=details or {},
    )
    db.add(event)
    if commit:
        db.commit()
    return event


def log_source_rejection(
    db: Session,
    exc: SourceRejection,
    *,
    assessment_id: int | None,
    url: str | None,
    source_id: int | None = None,
    stage: str = "fetch",
) -> AuditEvent:
    """One audit record and one operational log line per rejection: codes and hostnames, no content."""
    try:
        host = urlparse(url).hostname if url else None
    except ValueError:
        host = None
    event_type = audit_event_for(exc)
    log.info("source_fetch_rejected: event=%s reason=%s host=%s assessment_id=%s source_id=%s stage=%s",
             event_type, exc.code, host, assessment_id, source_id, stage)
    return log_event(db, event_type, assessment_id=assessment_id, entity_type="source" if source_id else None,
                     entity_id=source_id, details={"reason": exc.code, "host": host, "stage": stage, "detail": str(exc)[:300]})
