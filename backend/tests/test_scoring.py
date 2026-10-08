"""5-8. reporting / planning / execution / performance scoring  9. no-evidence behaviour."""
from __future__ import annotations

import re

from app.services.scoring.common import CONFLICTING_EVIDENCE, FOUND, NO_EVIDENCE_FOUND, REQUIRES_REVIEW, EvidenceItem
from app.services.scoring.confidence import compute_confidence
from app.services.scoring.execution import score_execution
from app.services.scoring.performance import evaluate_series, score_performance
from app.services.scoring.planning import decide_planning_level, score_planning
from app.services.scoring.reporting import decide_reporting_level, score_reporting


def E(id, claim_type, verified=True, source_id=1, **facts):
    return EvidenceItem(id=id, claim_type=claim_type, facts=facts, verified=verified, source_id=source_id)


# ------------------------------------------------------------------ reporting
def test_reporting_levels():
    assert decide_reporting_level({})[0] == 1
    assert decide_reporting_level({"regulatory_only_evidence": True})[0] == 2
    assert decide_reporting_level({"voluntary_disclosure": True, "regulatory_only_evidence": True})[0] == 3
    assert decide_reporting_level({"public_regular_report": True})[0] == 4
    assert decide_reporting_level({"public_regular_report": True, "framework_explicit": True})[0] == 5
    assert decide_reporting_level({"public_regular_report": True, "framework_explicit": True, "independent_assurance": True})[0] == 6
    # assurance without framework-based reporting is not level 6
    lvl, _, missing = decide_reporting_level({"public_regular_report": True, "independent_assurance": True})
    assert lvl == 4 and any("framework" in m for m in missing)


def test_score_reporting_keeps_evidence_per_level():
    ev = [
        E(1, "report_published", public=True, regular=True, report_type="sustainability_report"),
        E(2, "framework_reference", framework="GRI", explicit=True, level5_eligible=True),
        E(3, "assurance_statement", sustainability_assurance=True, provider="KPMG", standard="ISAE 3000"),
    ]
    r = score_reporting(ev)
    assert r.recommended_level == 6 and r.evidence_status == FOUND
    assert set(r.evidence_ids) == {1, 2, 3}
    # a mere mention of GRI (not "in accordance with") does not reach level 5
    r2 = score_reporting([ev[0], E(4, "framework_reference", framework="GRI", explicit=False, level5_eligible=True)])
    assert r2.recommended_level == 4
    # financial-audit-only assurance does not count
    r3 = score_reporting(ev[:2] + [E(5, "assurance_statement", sustainability_assurance=False, provider="EY")])
    assert r3.recommended_level == 5


# ------------------------------------------------------------------ planning
def test_planning_rule_table_example_from_spec():
    level, rule, missing = score_planning({"formalized": True, "public": True, "quantitative": True, "time_bound": True,
                                           "benchmark_found": False})
    assert level == 5 and rule == "P5_QUANTITATIVE_TIME_BOUND"
    assert missing == ["No sufficient external benchmark evidence found (e.g. SBTi, Paris Agreement, EU Taxonomy, peers)."]


def test_planning_levels():
    base = {"goal_found": True, "formalized": True, "public": True, "quantitative": True, "time_bound": True}
    assert decide_planning_level({"goal_found": False})[0] == 1
    assert decide_planning_level({"goal_found": True, "formalized": False})[0] == 2
    assert decide_planning_level({**base, "public": False})[0] == 3
    assert decide_planning_level({**base, "quantitative": False})[0] == 4
    assert decide_planning_level({**base, "time_bound": False})[0] == 4
    assert decide_planning_level(base)[0] == 5
    assert decide_planning_level({**base, "benchmark_found": True})[0] == 6
    assert decide_planning_level(base, require_owner=True)[0] == 4


def test_planning_uses_best_verified_goal_and_ignores_unverified():
    ev = [
        E(1, "goal", goal_found=True, formalized=True, public=True, quantitative=False, time_bound=False),
        E(2, "goal", goal_found=True, formalized=True, public=True, quantitative=True, time_bound=True, target_value=30, target_year=2030),
        E(3, "goal", verified=False, goal_found=True, formalized=True, public=True, quantitative=True, time_bound=True, benchmark_found=True),
    ]
    r = score_planning(ev, "Energy Management")
    assert r.recommended_level == 5  # unverified benchmark goal (id 3) not used
    assert r.evidence_ids[0] == 2 and 3 not in r.evidence_ids
    assert r.evidence_status == REQUIRES_REVIEW and r.unverified_ids == [3]
    assert r.conditions["time_bound"].status == "met" and r.conditions["clear_owner"].status == "unknown"
    assert r.conditions["benchmark"].status == "not_found"


def test_planning_benchmark_requires_same_source():
    goal = E(1, "goal", goal_found=True, formalized=True, public=True, quantitative=True, time_bound=True)
    bench = E(2, "goal", goal_found=True, formalized=True, public=True, benchmark_found=True, benchmark_type="science_based")
    assert score_planning([goal, bench], "Energy").recommended_level == 6
    bench_other = E(3, "goal", source_id=9, goal_found=True, formalized=True, public=True, benchmark_found=True)
    assert score_planning([goal, bench_other], "Energy").recommended_level == 5
    far = E(4, "goal", goal_found=True, formalized=True, public=True, benchmark_found=True)
    goal.page_number, bench.page_number, far.page_number = 184, 184, 165
    assert score_planning([goal, far], "Energy").recommended_level == 5  # benchmark on another page
    assert score_planning([goal, bench], "Energy").recommended_level == 6


# ------------------------------------------------------------------ execution
def test_execution_levels():
    act = E(1, "activity", activity_found=True, in_window=True, linked_to_goal=False)
    linked = E(2, "activity", activity_found=True, in_window=True, linked_to_goal=True)
    old = E(3, "activity", activity_found=True, in_window=False, linked_to_goal=True)
    assert score_execution([], "Energy").recommended_level == 1
    assert score_execution([act], "Energy", formal_goal_exists=True).recommended_level == 2
    assert score_execution([linked], "Energy", formal_goal_exists=False).recommended_level == 2  # no goal to execute
    assert score_execution([linked, act], "Energy", formal_goal_exists=True).recommended_level == 3
    r = score_execution([old], "Energy", formal_goal_exists=True)
    assert r.recommended_level == 1 and r.evidence_status == FOUND and "none within the last three years" in r.reason


# ------------------------------------------------------------------ performance
def test_evaluate_series_computes_improvement():
    ev = evaluate_series([(2023, 124), (2024, 116), (2025, 108)], "lower", 2025)
    assert ev["improved"] and ev["from_year"] == 2023 and ev["to_year"] == 2025
    assert round(ev["change_pct"], 1) == -12.9
    assert evaluate_series([(2023, 124), (2025, 130)], "lower", 2025)["improved"] is False
    assert evaluate_series([(2023, 28), (2025, 34)], "higher", 2025)["improved"] is True
    assert evaluate_series([(2015, 100), (2025, 50)], "lower", 2025) is None  # outside the 3-year window


def test_performance_levels():
    s = E(1, "performance_series", metric="energy intensity", better="lower", unit="kWh/unit",
          points=[[2023, 124], [2024, 116], [2025, 108]])
    r = score_performance([s], "Energy Management", 2025)
    assert r.recommended_level == 2 and r.evidence_ids == [1] and "calculated" in r.reason
    worse = E(2, "performance_series", metric="energy intensity", better="lower", unit="kWh/unit", points=[[2023, 100], [2025, 120]])
    r = score_performance([worse], "Energy Management", 2025)
    assert r.recommended_level == 1 and r.evidence_status == FOUND and "does not show improvement" in r.reason
    sector = E(3, "sector_comparison", statement="Our intensity is 20% below the sector average of 130 kWh.", favorable=True)
    r = score_performance([s, sector], "Energy Management", 2025)
    assert r.recommended_level == 3 and r.evidence_status == REQUIRES_REVIEW


def test_vague_improvement_claims_not_accepted():
    vague = E(1, "unquantified_improvement_claim", statement="We significantly improved energy efficiency.")
    r = score_performance([vague], "Energy Management", 2025)
    assert r.recommended_level == 1 and r.evidence_status == NO_EVIDENCE_FOUND
    assert any("unquantified" in n for n in r.notes)


def test_conflicting_performance_evidence():
    a = E(1, "performance_series", source_id=1, metric="energy intensity", better="lower", unit="kWh", points=[[2023, 124], [2025, 108]])
    b = E(2, "performance_series", source_id=2, metric="energy intensity", better="lower", unit="kWh", points=[[2023, 150], [2025, 108]])
    r = score_performance([a, b], "Energy Management", 2025)
    assert r.evidence_status == CONFLICTING_EVIDENCE and r.conflicting


# ------------------------------------------------------------------ no-evidence behaviour
NEGATIVE_EXISTENCE = re.compile(r"\b(has no|have no|does not have|doesn't have|lacks|there is no)\b", re.I)


def test_no_evidence_is_not_absence():
    results = [
        score_reporting([]), score_planning([], "Data Security"), score_execution([], "Data Security"),
        score_performance([], "Data Security", 2025),
    ]
    for r in results:
        assert r.recommended_level == 1
        assert r.evidence_status == NO_EVIDENCE_FOUND
        assert r.evidence_ids == []
        assert "No evidence of" in r.reason and "reviewed sources" in r.reason
        assert not NEGATIVE_EXISTENCE.search(r.reason)
        for m in r.next_level_missing:
            assert not NEGATIVE_EXISTENCE.search(m)


def test_unverified_only_evidence_requires_review():
    r = score_planning([E(1, "goal", verified=False, goal_found=True, formalized=True, public=True)], "Energy")
    assert r.recommended_level == 1 and r.evidence_status == REQUIRES_REVIEW


def test_confidence_categories():
    s = E(1, "performance_series", metric="energy intensity", better="lower", points=[[2023, 124], [2025, 108]])
    s.source_quality, s.extraction_certainty = 1.0, 0.8
    r = score_performance([s], "Energy", 2025)
    score, label, _ = compute_confidence(r, [s], {"sources_reviewed": 1, "has_report": True})
    assert label in ("High", "Medium") and 0 < score < 1
    r0 = score_performance([], "Energy", 2025)
    assert compute_confidence(r0, [], {"sources_reviewed": 0, "has_report": False})[1] == "Low"
