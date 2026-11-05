"""Deterministic quote fallback: strict conditions, no extra model call, normal verification, nothing rescued."""
from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.entities import Evidence, Recommendation
from app.services.company.source_identity import CompanyIdentity
from app.services.extraction import gates
from app.services.extraction.extractor import QUOTE_FALLBACK, FactExtractor, Passage
from app.services.pipeline.runner import run_assessment
from app.services.search.base import NullSearchProvider
from app.services.verification.verifier import LIFECYCLE_ACCEPTED, LIFECYCLE_REJECTED, verify_candidate
from tests.conftest import FakeLLM

COMPANY = CompanyIdentity("Northstar Manufacturing AB")
GOAL = "We aim to reduce energy consumption by 30% by 2030 compared with a 2020 baseline."
ACTIVITY = "In 2025 we completed energy audits at all sites as part of our energy efficiency roadmap."


def _passage(text: str) -> Passage:
    return Passage(chunk_id=1, document_id=1, source_id=1, source_title="Report 2025", source_url=None, source_type="sustainability_report",
                   source_quality=0.95, is_public=True, page_number=2, text=text, doc_year=2025)


class _Quoting(FakeLLM):
    """Always reports a goal, with a chosen quote and optionally chosen values."""

    def __init__(self, quote: str, **extra):
        super().__init__()
        self.quote, self.extra = quote, extra

    def complete_json(self, system, user, schema=None, max_tokens=384) -> str:
        self.calls.append(user)
        return json.dumps({"goal_found": True, "formalized": True, "quantitative": False, "evidence_quote": self.quote, "confidence": 0.9, **self.extra})


def _plan(text: str, llm: FakeLLM, factor: str = "energy_management", year: int = 2025):
    ex = FactExtractor(llm, year)
    return ex, ex.planning(factor, [_passage(text)])


def _fallback(candidates):
    return [c for c in candidates if c.evidence_origin == QUOTE_FALLBACK]


# ---------------------------------------------------------------- when it applies
@pytest.mark.parametrize("quote, failure", [
    ("", "missing"),
    ("We promise to reach 90% of everything by 2030 without fail.", "not_in_source"),
    (ACTIVITY, "not_a_goal_statement"),
])
def test_unusable_quote_falls_back_to_the_prefilter_sentence_without_another_call(quote, failure):
    llm = _Quoting(quote)
    ex, out = _plan(f"{GOAL} {ACTIVITY}", llm)
    assert ex.llm_calls == 1 and len(llm.calls) == 1
    [fb] = _fallback(out)
    assert fb.evidence_text == GOAL and fb.llm_quote_failed and fb.facts["llm_quote_failure"] == failure
    vr = verify_candidate(fb, f"{GOAL} {ACTIVITY}", company=COMPANY)
    assert vr.lifecycle == LIFECYCLE_ACCEPTED and vr.rejection_codes == []


def test_values_come_from_the_sentence_never_from_the_failed_answer():
    _, out = _plan(GOAL, _Quoting("We promise to reach 90% by 2040 without any doubt.", target_value=90, target_year=2040, quantitative=True))
    [fb] = _fallback(out)
    assert (fb.facts["target_value"], fb.facts["target_year"], fb.facts["baseline_year"]) == (30.0, 2030, 2020)


def test_valid_model_quote_needs_no_fallback():
    _, out = _plan(GOAL, _Quoting(GOAL))
    assert _fallback(out) == [] and len(out) == 1


def test_slightly_paraphrased_quote_that_verifies_needs_no_fallback():
    _, out = _plan(GOAL, _Quoting(GOAL.replace("consumption", "consumptions")))
    assert _fallback(out) == []


def test_model_finding_no_goal_is_respected():
    class NoGoal(FakeLLM):
        def complete_json(self, system, user, schema=None, max_tokens=384) -> str:
            return json.dumps({"goal_found": False, "evidence_quote": "", "confidence": 0.3})

    _, out = _plan(GOAL, NoGoal())
    assert out == []


# ---------------------------------------------------------------- what it must not rescue
@pytest.mark.parametrize("quote", [
    "Our energy target of 30% by 2030 was discontinued in 2024.",  # state
    "The EU has set a target to reduce energy use by 32% by 2030.",  # actor
])
def test_answer_about_an_inactive_or_external_goal_is_not_rescued(quote):
    _, out = _plan(f"{quote} {GOAL}", _Quoting(quote))
    assert _fallback(out) == []


def test_answer_about_another_factor_is_not_rescued():
    quote = "Our board reviewed the diversity figures during the year."
    _, out = _plan(f"{quote} {GOAL}", _Quoting(quote))
    assert _fallback(out) == []


def test_numeric_failure_of_a_good_quote_is_not_rescued():
    _, out = _plan(GOAL, _Quoting(GOAL, target_value=45, quantitative=True))
    assert _fallback(out) == []
    [c] = out
    assert verify_candidate(c, GOAL, company=COMPANY).lifecycle == LIFECYCLE_REJECTED


@pytest.mark.parametrize("text", [
    "We aim to reduce energy use by 30% by 2030, a target that was discontinued in 2024.",  # state
    "The EU aims to reduce energy use across all members by 30% by 2030.",  # actor
    "Our energy goal was reached and we report energy results to the board each year.",  # no forward language
    "We aim to reduce energy consumption by 30% by 2020 compared with a 2015 baseline.",  # deadline has passed
    "Energy goal | we will reduce | 30% | by 2030 | owner | operations",  # table fragment
])
def test_ineligible_sentences_never_become_fallback_evidence(text):
    _, out = _plan(text, _Quoting("We promise to reach 90% of everything by 2030 without fail."))
    assert _fallback(out) == []


def test_fallback_still_has_to_pass_verification():
    """Exists in the passage but not on the stated page: the fallback gets no fuzzy match."""
    _, out = _plan(GOAL, _Quoting(""))
    [fb] = _fallback(out)
    page = GOAL.replace("30%", "35%")
    vr = verify_candidate(fb, page, company=COMPANY)
    assert vr.lifecycle == LIFECYCLE_REJECTED and "fallback_quote_not_verbatim" in vr.rejection_codes


def test_fallback_rejected_when_the_goal_belongs_to_a_named_other_party_under_identity_check():
    assert gates.goal_quote_failure("", "text") == "missing"
    assert gates.goal_quote_failure(GOAL, GOAL) is None


# ---------------------------------------------------------------- pipeline
class _WrongSentenceLLM(FakeLLM):
    """The observed 1B behaviour: says a goal exists, quotes a real sentence that is not the goal."""

    def complete_json(self, system, user, schema=None, max_tokens=384) -> str:
        self.calls.append(user)
        if "DIMENSION:\nPlanning" in user:
            return json.dumps({"goal_found": True, "formalized": True, "quantitative": False,
                               "evidence_quote": "The energy target is owned by our Chief Operating Officer and progress is reviewed quarterly",
                               "confidence": 0.8})
        return super().complete_json(system, user, schema, max_tokens)


def _run(client, sample_pdf, llm, name="Nordvik Components AB") -> int:
    r = client.post("/api/assessments", json={"company_name": name, "year": 2025, "enable_web_search": False, "confirm_new": True})
    aid = r.json()["id"]
    with open(sample_pdf, "rb") as f:
        assert client.post(f"/api/assessments/{aid}/documents", files={"file": ("report.pdf", f, "application/pdf")}).status_code == 201
    run_assessment(aid, llm=llm, search=NullSearchProvider())
    return aid


def _rows(aid):
    with SessionLocal() as db:
        return (list(db.scalars(select(Evidence).where(Evidence.assessment_id == aid))),
                {(r.factor, r.dimension): r for r in db.scalars(select(Recommendation).where(Recommendation.assessment_id == aid))})


def test_fallback_restores_a_real_goal_flagged_for_review_and_exported(client, sample_pdf):
    aid = _run(client, sample_pdf, _WrongSentenceLLM())
    evidence, recs = _rows(aid)
    fallback = [e for e in evidence if e.factor == "energy_management" and e.evidence_origin == QUOTE_FALLBACK]
    assert fallback and all(e.llm_quote_failed and e.extraction_method == "quote_fallback" and e.evidence_lifecycle == LIFECYCLE_ACCEPTED for e in fallback)
    rec = recs[("energy_management", "planning")]
    assert rec.recommended_level == 5 and {e.id for e in fallback} & set(rec.evidence_ids)
    assert rec.evidence_status == "REQUIRES_REVIEW" and rec.confidence <= 0.7

    accepted = {e.id for e in evidence if e.evidence_lifecycle == LIFECYCLE_ACCEPTED}
    assert all(set(r.evidence_ids) <= accepted for r in recs.values())

    data = client.get(f"/api/assessments/{aid}/export", params={"format": "json"}).json()
    exported = [e for e in data["evidence"] if e["evidence_origin"] == QUOTE_FALLBACK]
    assert exported and all(e["llm_quote_failed"] is True and e["quote_confirmed_in_source"] is True for e in exported)
    assert any(e["used_for_scoring"] for e in exported)


def test_wrong_company_source_stays_blocked_with_no_model_call_and_no_fallback(client, sample_pdf):
    llm = _WrongSentenceLLM()
    aid = _run(client, sample_pdf, llm, name="Bluefin Foods Ltd")
    evidence, recs = _rows(aid)
    assert llm.calls == [] and evidence == [] and all(r.recommended_level == 0 and r.evidence_ids == [] for r in recs.values())
