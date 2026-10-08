"""Reporting ambition level (1-6), company level. Highest level whose conditions are all
supported by verified evidence (levels are cumulative):

1 no evidence of sustainability reporting
2 reporting appears limited to regulatory / compliance requirements
3 voluntary and/or ad-hoc sustainability disclosure
4 sustainability performance reported regularly (at least annually) in a public source
5 public reporting explicitly follows a recognised framework (GRI, SASB, ESRS, <IR>, ...)
6 level 5 + disclosures independently assured by a qualified third party"""
from __future__ import annotations

from typing import Any

from app.services.scoring.common import (
    MET, NOT_FOUND, Condition, EvidenceItem, ScoreResult, finalize_status, no_evidence_reason, split_verified,
)

MAX_LEVEL = 6


def decide_reporting_level(s: dict[str, Any]) -> tuple[int, str, list[str]]:
    """s: regulatory_only_evidence, voluntary_disclosure, public_regular_report, framework_explicit, independent_assurance"""
    if s.get("public_regular_report"):
        if s.get("framework_explicit"):
            if s.get("independent_assurance"):
                return 6, "R6_INDEPENDENT_ASSURANCE", []
            return 5, "R5_RECOGNISED_FRAMEWORK", ["No independent third-party assurance of the sustainability disclosures was found."]
        missing = ["No explicit reference to a recognised reporting framework (e.g. GRI, SASB, ESRS, Integrated Reporting) was found."]
        if s.get("independent_assurance"):
            missing.append("Assurance was found, but level 6 also requires framework-based reporting.")
        return 4, "R4_REGULAR_PUBLIC_REPORTING", missing
    if s.get("voluntary_disclosure"):
        return 3, "R3_VOLUNTARY_DISCLOSURE", ["No regular (annual) publicly available sustainability report was found."]
    if s.get("regulatory_only_evidence"):
        return 2, "R2_REGULATORY_ONLY", ["No voluntary or regular public sustainability reporting was found."]
    return 1, "R1_NO_REPORTING_EVIDENCE", ["No sustainability reporting was found in the reviewed sources."]


REASONS = {
    "R6_INDEPENDENT_ASSURANCE": "Public framework-based sustainability reporting with independent third-party assurance.",
    "R5_RECOGNISED_FRAMEWORK": "Public sustainability reporting explicitly follows a recognised reporting framework.",
    "R4_REGULAR_PUBLIC_REPORTING": "Sustainability performance is reported regularly in a publicly available report.",
    "R3_VOLUNTARY_DISCLOSURE": "Voluntary / ad-hoc sustainability disclosures were found.",
    "R2_REGULATORY_ONLY": "Sustainability reporting appears limited to regulatory requirements.",
}


def score_reporting(evidence: list[EvidenceItem]) -> ScoreResult:
    verified, unverified = split_verified(evidence)
    reports = [e for e in verified if e.claim_type == "report_published"]
    public_regular = [e for e in reports if e.facts.get("public") and e.facts.get("regular")]
    voluntary = [e for e in verified if e.claim_type == "voluntary_disclosure"] + [e for e in reports if e not in public_regular]
    frameworks = [e for e in verified if e.claim_type == "framework_reference" and e.facts.get("explicit") and e.facts.get("level5_eligible")]
    framework_mentions = [e for e in verified if e.claim_type == "framework_reference" and e not in frameworks]
    assurance = [e for e in verified if e.claim_type == "assurance_statement" and e.facts.get("sustainability_assurance")
                 and (e.facts.get("provider") or e.facts.get("standard"))]
    regulatory = [e for e in verified if e.claim_type == "regulatory_reporting"]

    signals = {
        "public_regular_report": bool(public_regular),
        "voluntary_disclosure": bool(voluntary or framework_mentions),
        "framework_explicit": bool(frameworks),
        "independent_assurance": bool(assurance),
        "regulatory_only_evidence": bool(regulatory),
    }
    level, rule, missing = decide_reporting_level(signals)
    by_level = {
        6: assurance + frameworks + public_regular,
        5: frameworks + public_regular,
        4: public_regular,
        3: voluntary + framework_mentions,
        2: regulatory,
        1: [],
    }
    ids = list(dict.fromkeys(e.id for e in by_level[level]))
    reason = REASONS.get(rule) or no_evidence_reason("sustainability reporting")
    r = ScoreResult(level, MAX_LEVEL, "", rule, reason, evidence_ids=ids[:10], next_level_missing=missing, unverified_ids=unverified)
    r.conditions = {
        "public_regular_report": Condition("Regular public report", MET if public_regular else NOT_FOUND, [e.id for e in public_regular][:5]),
        "framework": Condition("Recognised framework", MET if frameworks else NOT_FOUND, [e.id for e in frameworks][:5],
                               ", ".join(sorted({e.facts.get("framework") for e in frameworks})) or None),
        "assurance": Condition("Independent assurance", MET if assurance else NOT_FOUND, [e.id for e in assurance][:5],
                               ", ".join(sorted({e.facts.get("provider") or e.facts.get("standard") for e in assurance})) or None),
    }
    if framework_mentions and not frameworks:
        r.notes.append("Frameworks are mentioned but no explicit statement of reporting in accordance with / with reference to them was found.")
    if level == 2:
        r.notes.append("Distinguishing regulatory-only from voluntary reporting from public text is uncertain; analyst should confirm.")
    if level >= 4 and any(e.facts.get("report_type") == "uploaded_report" for e in public_regular):
        r.notes.append("Public availability of the uploaded report was declared at upload; confirm it is published.")
    return finalize_status(r)
