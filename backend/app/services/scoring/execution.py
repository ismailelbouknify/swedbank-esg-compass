"""Execution ambition level (1-3), per factor, over the last three years.

1 no evidence of activities to manage the factor
2 evidence of activities to manage the factor
3 evidence of activities that execute the strategy/plans towards the stated goals"""
from __future__ import annotations

from app.services.scoring.common import (
    MET, NOT_FOUND, UNKNOWN, Condition, EvidenceItem, ScoreResult, finalize_status, no_evidence_reason, split_verified,
)

MAX_LEVEL = 3


def decide_execution_level(activity_in_window: bool, linked_to_goal: bool, formal_goal_exists: bool) -> tuple[int, str, list[str]]:
    if not activity_in_window:
        return 1, "E1_NO_ACTIVITY_EVIDENCE", ["No evidence of activities to manage the factor in the last three years."]
    if not (linked_to_goal and formal_goal_exists):
        missing = []
        if not formal_goal_exists:
            missing.append("No formalized goal/strategy was found for activities to implement.")
        if not linked_to_goal:
            missing.append("No activity explicitly implementing the strategy/goals was found.")
        return 2, "E2_ACTIVITIES_FOUND", missing
    return 3, "E3_ACTIVITIES_EXECUTE_STRATEGY", []


def score_execution(evidence: list[EvidenceItem], factor_label: str = "the factor", formal_goal_exists: bool = False) -> ScoreResult:
    """formal_goal_exists: whether planning found a verified formalized goal (level >= 3)."""
    acts = [e for e in evidence if e.claim_type == "activity"]
    verified, unverified = split_verified(acts)
    in_window = [e for e in verified if e.facts.get("in_window") is not False]
    outside = [e for e in verified if e.facts.get("in_window") is False]
    linked = [e for e in in_window if e.facts.get("linked_to_goal")]

    level, rule, missing = decide_execution_level(bool(in_window), bool(linked), formal_goal_exists)
    if level == 1:
        reason = no_evidence_reason(f"activities to manage {factor_label} during the last three years")
        ids: list[int] = []
    elif level == 2:
        reason = f"Activities to manage {factor_label} were identified in the last three years."
        ids = [e.id for e in in_window]
    else:
        reason = f"Activities explicitly implementing the {factor_label} strategy/goals were identified in the last three years."
        ids = [e.id for e in linked] + [e.id for e in in_window if e not in linked]

    r = ScoreResult(level, MAX_LEVEL, "", rule, reason, evidence_ids=ids[:8], next_level_missing=missing, unverified_ids=unverified)
    r.conditions = {
        "activity": Condition("Activity in last 3 years", MET if in_window else NOT_FOUND, [e.id for e in in_window][:5]),
        "linked": Condition("Linked to stated goals/strategy", MET if linked else NOT_FOUND, [e.id for e in linked][:5]),
        "formal_goal": Condition("Formalized goal exists (Planning)", MET if formal_goal_exists else NOT_FOUND),
    }
    undated = [e for e in in_window if e.facts.get("in_window") is None]
    if in_window and len(undated) == len(in_window):
        r.conditions["activity"].status = UNKNOWN
        r.conditions["activity"].detail = "Activity year could not be determined"
        r.notes.append("The timing of the activities could not be confirmed as within the last three years.")
        r.rule_completeness = 0.6
    if outside and not in_window:
        # Evidence exists, it just does not satisfy the time window: keep it visible.
        r.reason = f"Activities related to {factor_label} were found, but none within the last three years."
        r.evidence_ids = [e.id for e in outside][:5]
        r.notes.append(f"{len(outside)} activity item(s) fall outside the three-year window and were not counted.")
    return finalize_status(r)
