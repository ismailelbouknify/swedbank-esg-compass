"""Combined confidence heuristic. The 1B model's self-reported confidence is NOT calibrated,
so it is only one of several inputs; the result is shown as High / Medium / Low."""
from __future__ import annotations

from typing import Any

from app.config.loader import load_rules
from app.services.scoring.common import CONFLICTING_EVIDENCE, NO_EVIDENCE_FOUND, EvidenceItem, ScoreResult


def label_for(score: float) -> str:
    t = load_rules()["confidence"]["thresholds"]
    return "High" if score >= t["high"] else "Medium" if score >= t["medium"] else "Low"


def compute_confidence(result: ScoreResult, evidence: list[EvidenceItem], coverage: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
    """coverage: {'sources_reviewed': int, 'has_report': bool} used for NO_EVIDENCE results."""
    cfg = load_rules()["confidence"]
    if result.evidence_status == NO_EVIDENCE_FOUND or not result.evidence_ids:
        # Confidence that "not found" is a fair statement depends on how much was reviewed.
        if coverage.get("has_report"):
            score = 0.6
        elif coverage.get("sources_reviewed", 0) > 0:
            score = 0.4
        else:
            score = 0.2
        breakdown = {"basis": "search coverage", **coverage}
        return score, label_for(score), breakdown

    used = [e for e in evidence if e.id in set(result.evidence_ids)]
    if not used:
        return 0.3, label_for(0.3), {"basis": "evidence not loaded"}
    w = cfg["weights"]
    source_quality = max(e.source_quality for e in used)
    n_sources = len({e.source_id for e in used})
    supporting = min(1.0, 0.5 + 0.25 * (n_sources - 1))
    extraction = sum(e.extraction_certainty for e in used) / len(used)
    completeness = result.rule_completeness
    score = (w["source_quality"] * source_quality + w["supporting_sources"] * supporting
             + w["extraction_certainty"] * extraction + w["rule_completeness"] * completeness)
    if result.evidence_status == CONFLICTING_EVIDENCE or result.conflicting:
        score -= cfg["conflict_penalty"]
    if result.unverified_ids:
        score -= cfg["unverified_penalty"]
    score = round(max(0.05, min(0.99, score)), 3)
    breakdown = {
        "source_quality": round(source_quality, 2), "supporting_sources": n_sources,
        "extraction_certainty": round(extraction, 2), "rule_completeness": round(completeness, 2),
        "conflicting": result.conflicting, "unverified_items": len(result.unverified_ids),
    }
    return score, label_for(score), breakdown
