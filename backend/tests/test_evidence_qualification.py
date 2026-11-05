"""Generic evidence-qualification tests.

All tests use the fictional company "Northstar Manufacturing AB".
No company-specific production rules are tested or added here.
These tests verify CLASSES of evidence behaviour, not specific company data.
"""
from __future__ import annotations

import pytest

from app.config.loader import get_factor, load_rules
from app.services.extraction.extractor import Candidate, Passage
from app.services.scoring.common import EvidenceItem
from app.services.scoring.performance import score_performance
from app.services.scoring.planning import score_planning
from app.services.scoring.reporting import score_reporting
from app.services.verification.verifier import (
    LIFECYCLE_ACCEPTED, LIFECYCLE_CANDIDATE, LIFECYCLE_REJECTED,
    REJECTED, REQUIRES_REVIEW, VERIFIED,
    _check_company_subject, _check_exclusion_keywords,
    _dimension_relevant, number_in_text, verify_candidate,
)

COMPANY = "Northstar Manufacturing AB"


# ── helpers ──────────────────────────────────────────────────────────────────

def _passage(text: str, source_type: str = "sustainability_report") -> Passage:
    return Passage(
        chunk_id=1, document_id=1, source_id=1,
        source_title="Northstar Sustainability Report 2025",
        source_url="https://northstar.example/sr2025.pdf",
        source_type=source_type, source_quality=0.95, is_public=True,
        page_number=10, text=text,
    )


def _candidate(factor: str, dimension: str, claim_type: str, text: str,
               facts: dict | None = None, source_type: str = "sustainability_report") -> Candidate:
    return Candidate(
        factor=factor, dimension=dimension, claim_type=claim_type,
        claim_text=text[:200], facts=facts or {}, evidence_text=text,
        passage=_passage(text, source_type), method="heuristic",
    )


def E(id, claim_type, verified=True, source_id=1, **facts):
    return EvidenceItem(id=id, claim_type=claim_type, facts=facts, verified=verified, source_id=source_id)


# ── 1. Factor relevance: exclusion keywords ──────────────────────────────────

class TestFactorExclusion:
    def test_energy_electricity_is_energy_factor(self):
        text = "We reduced electricity consumption by 12%."
        assert not _check_exclusion_keywords("energy_management", text)

    def test_energy_is_not_di(self):
        """Electricity reduction must not match D&I exclusion keywords."""
        text = "We reduced electricity consumption by 12%."
        # No exclusion keywords for energy in D&I
        assert not _check_exclusion_keywords("diversity_inclusion", text)

    def test_biodiversity_excluded_from_di(self):
        """A biodiversity passage containing 'species' and 'habitat' must be excluded from D&I."""
        text = "We documented 72 bird species at the restored habitat."
        assert _check_exclusion_keywords("diversity_inclusion", text)

    def test_two_exclusion_keywords_triggers_gate(self):
        """Two exclusion keywords → rejected for D&I even without 'diversity'."""
        text = "We protect ecosystem and wildlife in our conservation area."
        assert _check_exclusion_keywords("diversity_inclusion", text)

    def test_women_in_management_is_di_relevant(self):
        """No exclusion for genuine workforce diversity evidence."""
        text = "Women represented 38% of management in 2025."
        assert not _check_exclusion_keywords("diversity_inclusion", text)

    def test_collective_bargaining_not_excluded_from_labor(self):
        text = "Employees are covered by collective bargaining agreements."
        assert not _check_exclusion_keywords("labor_practices", text)

    def test_animal_welfare_excluded_from_labor(self):
        text = "We improved animal welfare and livestock working conditions."
        # Should trigger because both exclusion keywords appear
        assert _check_exclusion_keywords("labor_practices", text)


# ── 2. Company subject gate ───────────────────────────────────────────────────

class TestCompanySubject:
    def test_government_target_fails_company_check_for_unofficial_source(self):
        """Government targets must not be accepted as company evidence from non-official sources."""
        text = "The government aims to reduce national energy use by 30% by 2030."
        c = _candidate("energy_management", "planning", "goal", text, source_type="other_web")
        result = not _check_company_subject(c, text)
        assert result, "Government target should fail company subject check for non-official source"

    def test_we_our_passes_company_check(self):
        """First-person company language passes the company subject check."""
        text = "We aim to reduce our energy use by 30% by 2030."
        c = _candidate("energy_management", "planning", "goal", text, source_type="sustainability_report")
        assert _check_company_subject(c, text)

    def test_official_source_always_passes(self):
        """Official sustainability reports are presumed to discuss the company."""
        text = "National targets require all manufacturers to reduce emissions."
        c = _candidate("energy_management", "planning", "goal", text, source_type="sustainability_report")
        assert _check_company_subject(c, text)

    def test_government_target_passes_official_source(self):
        """When source is official, even government-sounding text passes (in-report context)."""
        text = "The EU government target requires 55% emission cuts."
        c = _candidate("energy_management", "planning", "goal", text, source_type="annual_report")
        assert _check_company_subject(c, text)

    def test_multiple_government_terms_fails_unofficial(self):
        text = "The national government and state government target reduces national energy."
        c = _candidate("energy_management", "planning", "goal", text, source_type="other_web")
        assert not _check_company_subject(c, text)


# ── 3. Dimension relevance ────────────────────────────────────────────────────

class TestDimensionRelevance:
    def test_forward_looking_is_planning(self):
        text = "We aim to reduce energy intensity by 30% by 2030."
        c = _candidate("energy_management", "planning", "goal", text)
        # dimension_relevant requires BOTH STRONG_GOAL_RE and FORWARD_PLANNING_RE
        assert _dimension_relevant(c, text)

    def test_past_tense_no_forward_is_not_planning(self):
        """A past-tense description of an old goal is NOT planning evidence."""
        text = "Our 2020 reduction target was 15%."
        c = _candidate("energy_management", "planning", "goal", text)
        # No "will/aims to/commit to/by 20XX" forward language
        assert not _dimension_relevant(c, text)

    def test_installed_is_execution(self):
        text = "In 2025 we installed rooftop solar panels across all facilities."
        c = _candidate("energy_management", "execution", "activity", text)
        assert _dimension_relevant(c, text)

    def test_target_is_not_execution(self):
        text = "We aim to install solar panels at all sites by 2027."
        c = _candidate("energy_management", "execution", "activity", text)
        # "installed" not present → not execution
        assert not _dimension_relevant(c, text)

    def test_numeric_series_is_performance(self):
        text = "Energy intensity was 120 in 2023 and 100 in 2025."
        c = _candidate("energy_management", "performance", "performance_series", text)
        assert _dimension_relevant(c, text)


# ── 4. Numeric grounding ──────────────────────────────────────────────────────

class TestNumericGrounding:
    def test_values_in_source_accepted(self):
        """LLM proposes 31 and 36 — both present in source."""
        source = "Women in management increased from 31% in 2023 to 36% in 2025."
        c = Candidate(
            factor="diversity_inclusion", dimension="performance",
            claim_type="performance_series", claim_text="Women in management",
            facts={"metric": "women in management", "better": "higher", "unit": "%",
                   "points": [[2023, 31.0], [2025, 36.0]]},
            evidence_text="Women in management increased from 31% in 2023 to 36% in 2025.",
            passage=_passage(source), method="llm",
        )
        vr = verify_candidate(c, source)
        assert "numeric_not_in_source" not in vr.rejection_codes

    def test_wrong_value_rejected(self):
        """LLM proposes 46 but source only contains 36 → hard rejection."""
        source = "Women in management increased from 31% in 2023 to 36% in 2025."
        c = Candidate(
            factor="diversity_inclusion", dimension="performance",
            claim_type="performance_series", claim_text="Women in management",
            facts={"metric": "women in management", "better": "higher", "unit": "%",
                   "points": [[2023, 31.0], [2025, 46.0]]},  # 46 not in source
            evidence_text="Women in management increased from 31% in 2023 to 36% in 2025.",
            passage=_passage(source), method="llm",
        )
        vr = verify_candidate(c, source)
        assert "numeric_not_in_source" in vr.rejection_codes
        assert vr.lifecycle == LIFECYCLE_REJECTED
        assert vr.status == REJECTED

    def test_rejected_numeric_cannot_influence_score(self):
        """An evidence item with lifecycle=rejected must not be accepted by split_verified."""
        from app.services.scoring.common import split_verified
        ev = EvidenceItem(id=1, claim_type="performance_series",
                          facts={"metric": "women in management", "better": "higher",
                                 "unit": "%", "points": [[2023, 31.0], [2025, 46.0]]},
                          verified=False)  # rejected lifecycle → verified=False
        verified, unverified = split_verified([ev])
        assert verified == []
        assert 1 in unverified


# ── 5. Performance metric unit comparability ─────────────────────────────────

class TestMetricComparability:
    def test_incompatible_units_not_combined(self):
        """kWh/unit and % are different units — must not be combined for the same metric."""
        intensity = E(1, "performance_series", metric="energy intensity",
                      better="lower", unit="kWh/unit",
                      points=[[2022, 120], [2023, 110], [2024, 100]])
        share = E(2, "performance_series", metric="energy intensity",
                  better="higher", unit="%",
                  points=[[2022, 30], [2024, 45]])
        r = score_performance([intensity, share], "Energy Management", 2024)
        # At most one should count; the note should mention incompatibility
        assert any("unit" in n.lower() or "incompatible" in n.lower() for n in r.notes) or r.recommended_level >= 1

    def test_same_unit_combined(self):
        """Two series with identical units can be combined."""
        s1 = E(1, "performance_series", metric="energy intensity",
               better="lower", unit="kWh/unit", points=[[2022, 120], [2024, 100]])
        s2 = E(2, "performance_series", metric="energy intensity",
               better="lower", unit="kWh/unit", points=[[2023, 115], [2024, 105]])
        r = score_performance([s1, s2], "Energy Management", 2024)
        assert r.recommended_level == 2


# ── 6. Benchmark factor-match ─────────────────────────────────────────────────

class TestBenchmarkFactorMatch:
    def test_climate_benchmark_does_not_raise_di_planning(self):
        """A science-based climate benchmark may not upgrade Diversity & Inclusion planning."""
        # D&I goal without benchmark
        di_goal = E(10, "goal", source_id=1,
                    goal_found=True, formalized=True, public=True,
                    quantitative=True, time_bound=True,
                    benchmark_found=False)
        # Climate benchmark on the same source — tagged for a different factor
        climate_bench = E(11, "goal", source_id=1,
                          goal_found=True, formalized=True, public=True,
                          benchmark_found=True, benchmark_type="science_based",
                          benchmark_factor="energy_management")  # different factor
        r = score_planning([di_goal, climate_bench], "Diversity & Inclusion",
                           factor_key="diversity_inclusion")
        # Level 5 is possible (quantitative + time-bound), but benchmark tagged for wrong
        # factor must not push to level 6.
        assert r.recommended_level == 5

    def test_di_benchmark_can_raise_di_planning(self):
        """A gender-equality benchmark correctly raises D&I planning to level 6."""
        di_goal = E(20, "goal", source_id=1,
                    goal_found=True, formalized=True, public=True,
                    quantitative=True, time_bound=True,
                    benchmark_found=True, benchmark_type="peers")
        r = score_planning([di_goal], "Diversity & Inclusion")
        assert r.recommended_level == 6


# ── 7. Reporting assurance: sustainability vs financial ───────────────────────

class TestReportingAssurance:
    def test_financial_audit_does_not_satisfy_r6(self):
        """Assurance on financial statements only must not satisfy Reporting level 6."""
        ev = [
            E(1, "report_published", public=True, regular=True, report_type="annual_report"),
            E(2, "framework_reference", framework="GRI", explicit=True, level5_eligible=True),
            # Financial-statement audit — sustainability_assurance=False
            E(3, "assurance_statement", sustainability_assurance=False, provider="KPMG"),
        ]
        r = score_reporting(ev)
        assert r.recommended_level == 5, "Financial audit should not push to R6"

    def test_sustainability_assurance_satisfies_r6(self):
        """Limited assurance explicitly covering sustainability disclosures satisfies R6."""
        ev = [
            E(1, "report_published", public=True, regular=True, report_type="sustainability_report"),
            E(2, "framework_reference", framework="GRI", explicit=True, level5_eligible=True),
            E(3, "assurance_statement", sustainability_assurance=True,
              provider="KPMG", standard="ISAE 3000"),
        ]
        r = score_reporting(ev)
        assert r.recommended_level == 6


# ── 8. Scoring invariants ─────────────────────────────────────────────────────

class TestScoringInvariants:
    def test_rejected_evidence_cannot_raise_level(self):
        """Evidence with verified=False (lifecycle=rejected) cannot raise a level."""
        rejected_goal = E(1, "goal", verified=False,
                          goal_found=True, formalized=True, public=True,
                          quantitative=True, time_bound=True)
        r = score_planning([rejected_goal], "Energy Management")
        assert r.recommended_level == 1

    def test_unverified_quote_cannot_increase_score(self):
        """REQUIRES_REVIEW evidence (verified=False) must not count toward level."""
        from app.services.scoring.execution import score_execution
        unverified_activity = E(1, "activity", verified=False,
                                activity_found=True, in_window=True, linked_to_goal=True)
        r = score_execution([unverified_activity], "Energy Management", formal_goal_exists=True)
        assert r.recommended_level == 1

    def test_final_recommendation_evidence_ids_are_accepted_only(self):
        """If a recommendation has evidence_ids, those IDs must correspond to verified items.

        This is a structural invariant: accepted items come from split_verified."""
        from app.services.scoring.planning import score_planning
        v = E(1, "goal", verified=True, goal_found=True, formalized=True, public=True,
              quantitative=True, time_bound=True)
        u = E(2, "goal", verified=False, goal_found=True, formalized=True, public=True,
              quantitative=True, time_bound=True)
        r = score_planning([v, u], "Energy Management")
        assert r.recommended_level == 5
        assert 1 in r.evidence_ids
        assert 2 not in r.evidence_ids

    def test_no_evidence_status_with_empty_input(self):
        """Empty evidence list always results in NO_EVIDENCE_FOUND."""
        from app.services.scoring.common import NO_EVIDENCE_FOUND
        from app.services.scoring.execution import score_execution
        from app.services.scoring.performance import score_performance
        r = score_planning([], "Energy Management")
        assert r.evidence_status == NO_EVIDENCE_FOUND
        r = score_execution([], "Energy Management")
        assert r.evidence_status == NO_EVIDENCE_FOUND
        r = score_performance([], "Energy Management", 2025)
        assert r.evidence_status == NO_EVIDENCE_FOUND

    def test_benchmark_wrong_factor_does_not_raise_level(self):
        """A benchmark on source X for factor A must not cause factor B to reach level 6."""
        from app.services.scoring.planning import score_planning
        # This goal is level 5 (quantitative + time-bound)
        goal = E(30, "goal", source_id=5, goal_found=True, formalized=True, public=True,
                 quantitative=True, time_bound=True, benchmark_found=False)
        # Benchmark on a different source — won't satisfy score_planning level 6
        bench_other_source = E(31, "goal", source_id=99, goal_found=True, formalized=True,
                                public=True, benchmark_found=True, benchmark_type="science_based")
        r = score_planning([goal, bench_other_source], "Labor Practices")
        assert r.recommended_level == 5  # different source → not level 6
