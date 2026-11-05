"""Holdout evaluation runner using synthetic fixtures.

Run with: pytest tests/evaluation/ -v

Reports: total cases, accepted correctly, rejected correctly,
factor classification accuracy, dimension classification accuracy.

All fixtures use fictional companies and data — no real company names or domains.
"""
from __future__ import annotations

import pytest

from app.services.verification.verifier import (
    LIFECYCLE_ACCEPTED, LIFECYCLE_REJECTED, _check_exclusion_keywords,
    _check_company_subject, _dimension_relevant, verify_candidate,
)
from app.services.extraction.extractor import Candidate, Passage


def _p(text: str, source_type: str = "sustainability_report") -> Passage:
    return Passage(
        chunk_id=1, document_id=1, source_id=1,
        source_title="Acme Corp Sustainability Report 2025",
        source_url="https://acme.example/sr2025.pdf",
        source_type=source_type, source_quality=0.95,
        is_public=True, page_number=1, text=text,
    )


def _c(factor: str, dim: str, ctype: str, text: str,
       facts: dict | None = None, source_type: str = "sustainability_report") -> Candidate:
    return Candidate(
        factor=factor, dimension=dim, claim_type=ctype,
        claim_text=text[:200], facts=facts or {}, evidence_text=text,
        passage=_p(text, source_type), method="heuristic",
    )


# ── Synthetic evaluation fixture ─────────────────────────────────────────────
# Each case: (description, candidate, page_text, expected_lifecycle)
EVAL_CASES: list[tuple[str, Candidate, str, str]] = [
    # === SHOULD ACCEPT ===
    (
        "Electricity reduction goal with forward language",
        _c("energy_management", "planning", "goal",
           "We aim to reduce electricity consumption by 20% by 2030.",
           {"goal_found": True, "formalized": True, "public": True,
            "quantitative": True, "time_bound": True,
            "target_value": 20.0, "target_year": 2030}),
        "We aim to reduce electricity consumption by 20% by 2030.",
        LIFECYCLE_ACCEPTED,
    ),
    (
        "Women in management performance series",
        _c("diversity_inclusion", "performance", "performance_series",
           "Women in management increased from 31% in 2023 to 36% in 2025.",
           {"metric": "women in management", "better": "higher", "unit": "%",
            "points": [[2023, 31.0], [2025, 36.0]]}),
        "Women in management increased from 31% in 2023 to 36% in 2025.",
        LIFECYCLE_ACCEPTED,
    ),
    (
        "Executed solar installation activity",
        _c("energy_management", "execution", "activity",
           "In 2024 we installed rooftop solar panels on all manufacturing sites.",
           {"activity_found": True, "in_window": True, "linked_to_goal": False}),
        "In 2024 we installed rooftop solar panels on all manufacturing sites.",
        LIFECYCLE_ACCEPTED,
    ),
    # === SHOULD REJECT ===
    (
        "Biodiversity passage incorrectly targeting D&I (exclusion keyword)",
        _c("diversity_inclusion", "planning", "goal",
           "We documented 72 bird species at the restored habitat and conservation area.",
           {"goal_found": True}),
        "We documented 72 bird species at the restored habitat and conservation area.",
        LIFECYCLE_REJECTED,
    ),
    (
        "Wrong numeric value (46 not in source containing 36)",
        _c("diversity_inclusion", "performance", "performance_series",
           "Women in management increased from 31% in 2023 to 36% in 2025.",
           {"metric": "women in management", "better": "higher", "unit": "%",
            "points": [[2023, 31.0], [2025, 46.0]]}),
        "Women in management increased from 31% in 2023 to 36% in 2025.",
        LIFECYCLE_REJECTED,
    ),
    (
        "Government target from unofficial source (company mismatch)",
        _c("energy_management", "planning", "goal",
           "The government aims to reduce national energy use by 30% by 2030. The national government target requires mandatory reporting.",
           {"goal_found": True}, source_type="other_web"),
        "The government aims to reduce national energy use by 30% by 2030. The national government target requires mandatory reporting.",
        LIFECYCLE_REJECTED,
    ),
]


class TestEvaluationHoldout:
    """Run the full synthetic evaluation set and report aggregate results."""

    def _run_case(self, case):
        description, candidate, page_text, expected_lifecycle = case
        vr = verify_candidate(candidate, page_text)
        return vr.lifecycle == expected_lifecycle, vr.lifecycle, expected_lifecycle, description

    def test_all_evaluation_cases(self):
        total = len(EVAL_CASES)
        correct = 0
        accepted_correct = 0
        accepted_total = sum(1 for _, _, _, exp in EVAL_CASES if exp == LIFECYCLE_ACCEPTED)
        rejected_correct = 0
        rejected_total = sum(1 for _, _, _, exp in EVAL_CASES if exp == LIFECYCLE_REJECTED)
        failures = []

        for case in EVAL_CASES:
            passed, got, expected, desc = self._run_case(case)
            if passed:
                correct += 1
                if expected == LIFECYCLE_ACCEPTED:
                    accepted_correct += 1
                elif expected == LIFECYCLE_REJECTED:
                    rejected_correct += 1
            else:
                failures.append(f"FAIL [{desc}]: expected={expected}, got={got}")

        print(f"\n=== Evaluation Results ===")
        print(f"Total cases:      {total}")
        print(f"Correct:          {correct}/{total}")
        print(f"Accepted correctly: {accepted_correct}/{accepted_total}")
        print(f"Rejected correctly: {rejected_correct}/{rejected_total}")
        if failures:
            print("Failures:")
            for f in failures:
                print(f"  {f}")

        assert correct == total, f"{len(failures)} case(s) failed:\n" + "\n".join(failures)


class TestFactorClassificationAccuracy:
    """Factor exclusion classification accuracy on synthetic inputs."""

    FACTOR_CASES: list[tuple[str, str, bool]] = [
        ("We reduced electricity consumption by 12%.", "energy_management", False),
        ("We documented 72 bird species at the restored habitat.", "diversity_inclusion", True),
        ("Women represented 38% of management in 2025.", "diversity_inclusion", False),
        ("Employees are covered by collective bargaining agreements.", "labor_practices", False),
        ("We protect ecosystem and wildlife in conservation areas.", "diversity_inclusion", True),
        ("The company improved its cybersecurity controls.", "data_security", False),
        ("We improved animal welfare and livestock conditions.", "labor_practices", True),
    ]

    def test_factor_exclusion_accuracy(self):
        total = len(self.FACTOR_CASES)
        correct = 0
        failures = []
        for text, factor, expect_excluded in self.FACTOR_CASES:
            got = _check_exclusion_keywords(factor, text)
            if got == expect_excluded:
                correct += 1
            else:
                failures.append(f"[{factor}] '{text[:60]}': expected excluded={expect_excluded}, got={got}")
        print(f"\nFactor exclusion accuracy: {correct}/{total}")
        if failures:
            for f in failures:
                print(f"  {f}")
        assert correct == total, "\n".join(failures)
