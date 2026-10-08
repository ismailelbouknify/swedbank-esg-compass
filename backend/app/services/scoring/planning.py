"""Planning ambition level (1-6), per factor.

1 no evidence of goals/strategy/plans
2 informal goals (normally only obtainable in dialogue with the company)
3 formalized qualitative goals
4 formalized goals that are publicly available
5 quantitative, time-bound goals (clear ownership: configurable requirement)
6 goals shown to be ambitious against an explicit external benchmark"""
from __future__ import annotations

from typing import Any

from app.config.loader import load_rules
from app.services.scoring.common import (
    MET, NOT_FOUND, UNKNOWN, Condition, EvidenceItem, ScoreResult, finalize_status, no_evidence_reason, split_verified,
)

MAX_LEVEL = 6


def _require_owner() -> bool:
    return bool(load_rules()["planning_rules"].get("require_clear_ownership_for_level_5", False))


def decide_planning_level(flags: dict[str, Any], require_owner: bool | None = None) -> tuple[int, str, list[str]]:
    """Pure rule table over one goal's flags -> (level, rule id, missing conditions for next level)."""
    require_owner = _require_owner() if require_owner is None else require_owner
    if not flags.get("goal_found", True if flags.get("formalized") else False):
        return 1, "P1_NO_GOAL_EVIDENCE", ["No evidence of goals, strategy or plans was found in the reviewed sources."]
    if not flags.get("formalized"):
        return 2, "P2_INFORMAL_GOAL", ["No formalized (documented) goal, strategy or plan was found."]
    if not flags.get("public"):
        return 3, "P3_FORMAL_NOT_PUBLIC", ["The formalized goal was not found in a publicly available source."]
    missing = []
    if not flags.get("quantitative"):
        missing.append("No quantitative goal (numeric target) was found.")
    if not flags.get("time_bound"):
        missing.append("No time-bound goal (target year) was found.")
    if require_owner and not flags.get("clear_owner"):
        missing.append("No clear ownership of the goal was found.")
    if missing:
        return 4, "P4_FORMAL_PUBLIC_QUALITATIVE", missing
    if not flags.get("benchmark_found"):
        return 5, "P5_QUANTITATIVE_TIME_BOUND", ["No sufficient external benchmark evidence found (e.g. SBTi, Paris Agreement, EU Taxonomy, peers)."]
    return 6, "P6_BENCHMARKED_AMBITION", []


RULE_REASONS = {
    "P2_INFORMAL_GOAL": "Only informal / non-documented intentions were identified.",
    "P3_FORMAL_NOT_PUBLIC": "A formalized goal was identified in a non-public document.",
    "P4_FORMAL_PUBLIC_QUALITATIVE": "A formalized goal/strategy was identified in a public source, but not a quantitative time-bound target.",
    "P5_QUANTITATIVE_TIME_BOUND": "Public quantitative time-bound target identified.",
    "P6_BENCHMARKED_AMBITION": "Public quantitative time-bound target identified, explicitly benchmarked against an external reference.",
}


def score_planning(evidence: list[EvidenceItem] | dict[str, Any], factor_label: str = "the factor") -> ScoreResult | tuple[int, str, list[str]]:
    """Score planning from structured goal evidence.

    Accepts a list of EvidenceItem (full scoring) or a plain flags dict (rule table only)."""
    if isinstance(evidence, dict):
        return decide_planning_level(evidence)

    goals = [e for e in evidence if e.claim_type == "goal"]
    verified, unverified = split_verified(goals)
    require_owner = _require_owner()
    if not verified:
        r = ScoreResult(1, MAX_LEVEL, "", "P1_NO_GOAL_EVIDENCE", no_evidence_reason(f"{factor_label} goals, strategy or plans"),
                        next_level_missing=["No formalized goal, strategy or plan was found in the reviewed sources."],
                        unverified_ids=unverified)
        r.conditions = _conditions({}, [], require_owner)
        return finalize_status(r)

    # Level per goal; the best-supported goal determines the level.
    scored = []
    for e in verified:
        level, rule, missing = decide_planning_level(e.facts, require_owner)
        scored.append((level, e, rule, missing))
    scored.sort(key=lambda t: (t[0], t[1].source_quality, t[1].extraction_certainty), reverse=True)
    level, best, rule, missing = scored[0]

    # A level-5 goal reaches 6 only if the benchmark is about that goal: stated in the goal itself or on
    # the same page of the same source (an SBTi climate target elsewhere does not benchmark a waste goal).
    if level == 5:
        bench = [e for e in verified if e.facts.get("benchmark_found") and e.source_id == best.source_id
                 and (e.id == best.id or e.page_number == best.page_number)]
        if bench:
            level, rule, missing = 6, "P6_BENCHMARKED_AMBITION", []
            best_bench = bench[0]
        else:
            best_bench = None
    else:
        best_bench = best if best.facts.get("benchmark_found") else None

    supporting = [best.id] + ([best_bench.id] if best_bench and best_bench.id != best.id else [])
    # other goals at the same level corroborate
    supporting += [e.id for lvl, e, _, _ in scored[1:] if lvl == level and e.id not in supporting]

    r = ScoreResult(level, MAX_LEVEL, "", rule, RULE_REASONS.get(rule, ""), evidence_ids=supporting,
                    next_level_missing=missing, unverified_ids=unverified)
    r.conditions = _conditions(best.facts, [best.id], require_owner, best_bench)
    r.rule_completeness = 1.0
    if level >= 5 and not best.facts.get("clear_owner"):
        r.notes.append("Clear ownership of the goal was not evidenced; the questionnaire text for level 5 mentions ownership \u2014 analyst should confirm.")
        r.rule_completeness *= 0.9
    if level == 6:
        r.notes.append("Level 6 relies on the company's own benchmark statement; verify the benchmark is credible.")
    if level in (2, 3):
        r.notes.append("Levels 2-3 usually require information from dialogue with the company.")
    return finalize_status(r)


def _conditions(f: dict[str, Any], ids: list[int], require_owner: bool, bench: EvidenceItem | None = None) -> dict[str, Condition]:
    def c(label: str, key: str, missing_status: str = NOT_FOUND) -> Condition:
        return Condition(label, MET if f.get(key) else missing_status, ids if f.get(key) else [])

    conds = {
        "formalized": c("Formal goal", "formalized"),
        "public": c("Public", "public"),
        "quantitative": c("Quantitative", "quantitative"),
        "time_bound": c("Time-bound", "time_bound"),
        "clear_owner": c("Clear ownership", "clear_owner", NOT_FOUND if require_owner else UNKNOWN),
        "benchmark": Condition("External benchmark", MET if bench else NOT_FOUND, [bench.id] if bench else [],
                               (bench.facts.get("benchmark_type") if bench else None)),
    }
    if f.get("target_value") is not None:
        conds["quantitative"].detail = f"{f['target_value']:g}{f.get('target_unit') or ''}"
    if f.get("target_year"):
        conds["time_bound"].detail = f"target year {f['target_year']}" + (f", baseline {f['baseline_year']}" if f.get("baseline_year") else "")
    return conds
