"""Audit trail helper."""
from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.models.entities import AuditEvent


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
