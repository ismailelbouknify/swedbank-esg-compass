"""Shared types for the deterministic ESG rule engine.

Scoring functions receive structured evidence (facts dicts + IDs), never raw LLM prose.
Only VERIFIED evidence can raise a level; unverified evidence is reported, not used."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FOUND = "FOUND"
NO_EVIDENCE_FOUND = "NO_EVIDENCE_FOUND"
CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
REQUIRES_REVIEW = "REQUIRES_REVIEW"

MET = "met"
NOT_FOUND = "not_found"
UNKNOWN = "unknown"


@dataclass
class EvidenceItem:
    id: int
    claim_type: str
    facts: dict[str, Any]
    verified: bool = True
    source_id: int = 0
    source_quality: float = 1.0
    extraction_certainty: float = 0.7
    method: str = "llm"
    page_number: int | None = None


@dataclass
class Condition:
    label: str
    status: str  # met | not_found | unknown
    evidence_ids: list[int] = field(default_factory=list)
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "status": self.status, "evidence_ids": self.evidence_ids, "detail": self.detail}


@dataclass
class ScoreResult:
    recommended_level: int
    max_level: int
    evidence_status: str
    triggered_rule: str
    reason: str
    conditions: dict[str, Condition] = field(default_factory=dict)
    evidence_ids: list[int] = field(default_factory=list)
    next_level_missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rule_completeness: float = 1.0
    conflicting: bool = False
    unverified_ids: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "recommended_level": self.recommended_level,
            "max_level": self.max_level,
            "evidence_status": self.evidence_status,
            "triggered_rule": self.triggered_rule,
            "reason": self.reason,
            "conditions": {k: c.as_dict() for k, c in self.conditions.items()},
            "evidence_ids": self.evidence_ids,
            "next_level_missing": self.next_level_missing,
            "notes": self.notes,
        }


def split_verified(items: list[EvidenceItem]) -> tuple[list[EvidenceItem], list[int]]:
    verified = [e for e in items if e.verified]
    unverified = [e.id for e in items if not e.verified]
    return verified, unverified


def finalize_status(result: ScoreResult) -> ScoreResult:
    """Derive the evidence status. NO_EVIDENCE_FOUND is only used when nothing at all was found."""
    if result.conflicting:
        result.evidence_status = CONFLICTING_EVIDENCE
    elif not result.evidence_ids and not result.unverified_ids:
        result.evidence_status = NO_EVIDENCE_FOUND
    elif result.unverified_ids or not result.evidence_ids:
        result.evidence_status = REQUIRES_REVIEW
    elif result.evidence_status not in (REQUIRES_REVIEW,):
        result.evidence_status = FOUND
    if result.unverified_ids:
        result.notes.append(
            f"{len(result.unverified_ids)} evidence item(s) could not be verified against the source and were not used for the level."
        )
    return result


def no_evidence_reason(subject: str) -> str:
    """Never state that something does not exist; only that it was not found."""
    return f"No evidence of {subject} was found in the reviewed sources."
