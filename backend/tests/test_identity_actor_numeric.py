"""Source identity as a hard gate, actor attribution, and whether a figure really is the company's target.
All companies are fictional."""
from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.entities import Assessment, Evidence, Recommendation, Source
from app.services.company.source_identity import CompanyIdentity, assess_source_identity
from app.services.export import exporter
from app.services.extraction import patterns as P
from app.services.extraction.extractor import Candidate, Passage
from app.services.llm.base import NullLLMProvider
from app.services.pipeline.runner import AssessmentRunner, run_assessment
from app.services.pipeline.source_gate import source_blocks_evidence
from app.services.search.base import NullSearchProvider
from app.services.verification.verifier import LIFECYCLE_ACCEPTED, LIFECYCLE_REJECTED, verify_candidate
from tests.conftest import FakeLLM

COMPANY = CompanyIdentity("Northstar Manufacturing AB")
SAMPLE_COMPANY = "Nordvik Components AB"


def _assess(text, title=None, url=None, company=COMPANY, **kw):
    return assess_source_identity(company, title=title, url=url, text=text, **kw)


def _passage(text, source_type="sustainability_report"):
    return Passage(chunk_id=1, document_id=1, source_id=1, source_title="Report", source_url=None, source_type=source_type,
                   source_quality=0.95, is_public=True, page_number=1, text=text, doc_year=2025)


def _goal(text, factor="energy_management", value=None, unit="%", year=None, source_type="sustainability_report"):
    facts = {"goal_found": True, "formalized": True, "public": True, "quantitative": value is not None, "time_bound": year is not None,
             "target_value": value, "target_unit": unit, "target_year": year}
    c = Candidate(factor, "planning", "goal", text[:80], facts, text, _passage(text, source_type), "llm")
    return verify_candidate(c, text, company=COMPANY)


def _activity(text, factor="energy_management"):
    c = Candidate(factor, "execution", "activity", text[:80], {"activity_found": True}, text, _passage(text), "llm")
    return verify_candidate(c, text, company=COMPANY)


# ---------------------------------------------------------------- identity
def test_document_dominated_by_a_different_company_is_rejected():
    text = "Bluefin Foods annual filing. " * 30 + "Bluefin Foods reported results."
    assert not _assess(text, title="Form 10-K").matched
    # a report with no legal suffix anywhere and a passing mention of the assessed company
    assert not _assess(("Bluefin reduced energy use. " * 40) + "Northstar Manufacturing supplies parts.", title="Annual filing").matched


@pytest.mark.parametrize("title, text", [
    ("Northstar Global AS", "We trade commodities worldwide. Northstar Global trades."),
    ("Sustainability reports | Northstar Group", "Reports of the Northstar Group are published here. Northstar employs many."),
    ("Northstar og miljoet", "Northstar er miljofyrtarnsertifisert. Northstar jobber med miljo."),
    ("Northstars Sustainability Report", "Northstars is a different business. Northstars grows."),
])
def test_similar_company_name_is_rejected(title, text):
    assert not _assess(text, title=title).matched


def test_shared_first_word_or_address_is_not_identity():
    assert not _assess("We trade goods.", url="https://northstar.example/northstar-manufacturing.pdf").matched
    assert not _assess("Northstar sells tools. Northstar grows.", url="https://northstar.example/").matched


def test_same_company_in_title_with_legal_form_or_plural_still_matches():
    assert _assess("We reduced energy use.", title="Northstar Manufacturing AB | Sustainability").matched
    cars = CompanyIdentity("Polar Cars")
    assert assess_source_identity(cars, title="Climate Action | Polar Cars", url=None, text="We reduced emissions.").matched
    assert assess_source_identity(CompanyIdentity("Polar Car AB"), title="Polar Cars", url=None, text="We reduced emissions.").matched


def test_division_on_the_companys_own_exact_domain_is_not_another_company():
    acme = CompanyIdentity("Acme")
    text = "Acme Cloud helps. Acme Cloud grows. Acme Cloud scales. Acme Cloud runs. Acme Cloud is fast."
    assert assess_source_identity(acme, title="Sustainability", url="https://cloud.acme.com/sustainability", text=text).basis == "name_domain"
    assert not assess_source_identity(acme, title="Sustainability", url="https://acmecloud.example/", text=text).matched


def _assessment(client, name=SAMPLE_COMPANY) -> int:
    r = client.post("/api/assessments", json={"company_name": name, "year": 2025, "enable_web_search": False, "confirm_new": True})
    assert r.status_code == 201
    return r.json()["id"]


def _upload(client, aid, pdf):
    with open(pdf, "rb") as f:
        assert client.post(f"/api/assessments/{aid}/documents", files={"file": ("report.pdf", f, "application/pdf")}).status_code == 201


def _rows(aid):
    with SessionLocal() as db:
        return (list(db.scalars(select(Source).where(Source.assessment_id == aid))),
                list(db.scalars(select(Evidence).where(Evidence.assessment_id == aid))),
                list(db.scalars(select(Recommendation).where(Recommendation.assessment_id == aid))))


def test_wrong_company_source_cannot_generate_evidence_or_affect_scoring(client, sample_pdf):
    aid = _assessment(client, "Bluefin Foods Ltd")
    _upload(client, aid, sample_pdf)
    llm = FakeLLM()
    run_assessment(aid, llm=llm, search=NullSearchProvider())
    sources, evidence, recs = _rows(aid)
    assert sources[0].status == "REJECTED" and sources[0].rejection_reason == "source_company_mismatch"
    assert source_blocks_evidence(sources[0]) and llm.calls == [] and evidence == []
    assert all(r.recommended_level == 0 and r.evidence_ids == [] for r in recs)  # a gap, not a Level 1 answer


def test_invariant_blocks_evidence_from_a_rejected_source_even_if_it_reaches_verification(client, sample_pdf):
    """Defence in depth: a candidate from a rejected source, however good it looks, is never accepted or scored."""
    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    with SessionLocal() as db:
        runner = AssessmentRunner(db, aid, llm=NullLLMProvider(), search=NullSearchProvider())
        runner._process_documents()
        runner._validate_source_identity()
        source = runner.a.sources[0]
        page = source.document.pages[1]
        goal = "We aim to reduce energy consumption per unit produced by 30% by 2030 compared with a 2020 baseline."
        passage = Passage(None, source.document.id, source.id, "Report", None, source.source_type, 0.95, True, page.page_number, page.text, 2025)
        candidate = Candidate("energy_management", "planning", "goal", "goal", {"goal_found": True, "formalized": True, "public": True,
                              "quantitative": True, "time_bound": True, "target_value": 30, "target_unit": "%", "target_year": 2030}, goal, passage, "llm")

        accepted_first = runner._persist_evidence([candidate])
        assert accepted_first[0].evidence_lifecycle == LIFECYCLE_ACCEPTED  # the same candidate passes on a valid source

        source.status, source.company_match = "REJECTED", "mismatch"
        blocked = runner._persist_evidence([candidate])
        assert blocked[0].evidence_lifecycle == LIFECYCLE_REJECTED and blocked[0].rejection_codes == ["source_company_mismatch"]

        # already-accepted evidence from a source rejected later cannot reach scoring either
        runner._score(accepted_first + blocked)
        recs = list(db.scalars(select(Recommendation).where(Recommendation.assessment_id == aid)))
    assert all(r.evidence_ids == [] and r.recommended_level == 0 for r in recs)  # no valid source remains


# ---------------------------------------------------------------- actors
@pytest.mark.parametrize("text", [
    "Authorities adopted regulations on energy efficiency in 2024.",
    "The industry association launched an energy efficiency programme in 2024.",
    "Survey respondents implemented energy efficiency measures in 2024.",
    "Market participants adopted energy targets in 2024.",
    "Banks implemented energy efficiency programmes in 2024.",
    "Bluefin Foods AB implemented an energy programme in 2024.",
])
def test_external_actor_blocks_company_execution(text):
    vr = _activity(text)
    assert vr.lifecycle == LIFECYCLE_REJECTED and "external_actor" in vr.rejection_codes


@pytest.mark.parametrize("text", [
    "Banks are expected to set energy targets by 2030.",
    "The requirement for financial institutions is to set climate and energy targets by 2030.",
    "The association aims to reduce energy use by 2030.",
])
def test_generic_obligation_or_external_goal_is_not_the_companys_target(text):
    vr = _goal(text)
    assert vr.lifecycle == LIFECYCLE_REJECTED and "external_actor" in vr.rejection_codes


def test_factor_mismatch_blocks_even_when_the_actor_is_right():
    vr = _activity("We implemented a new property valuation methodology in 2024.", "labor_practices")
    assert vr.lifecycle == LIFECYCLE_REJECTED and "factor_mismatch" in vr.rejection_codes


@pytest.mark.parametrize("text, factor", [
    ("The registered unemployment rate and labour market trends were assessed in 2024.", "labor_practices"),
    ("We implemented a target for operational emissions from offices in 2024.", "product_lifecycle"),
    ("Employment in public cloud roles was assessed by 86% of organizations in 2024.", "labor_practices"),
])
def test_factor_and_subject_are_coupled(text, factor):
    assert _activity(text, factor).lifecycle == LIFECYCLE_REJECTED


# ---------------------------------------------------------------- numeric target tuple
@pytest.mark.parametrize("text, value, factor, code", [
    ("We joined the climate alliance for energy goals, which represents 40% of global banking assets.", 40, "energy_management", "numeric_target_subject_mismatch"),
    ("Our energy goal follows an association that represents 40% of global assets.", 40, "energy_management", "numeric_target_subject_mismatch"),
    ("Our goal for employment follows research where 86% of organizations use public cloud.", 86, "labor_practices", "numeric_target_subject_mismatch"),
])
def test_statistics_about_others_cannot_become_a_company_target(text, value, factor, code):
    vr = _goal(text, factor, value=value)
    assert vr.lifecycle == LIFECYCLE_REJECTED and code in vr.rejection_codes


@pytest.mark.parametrize("text, value, unit, year, factor", [
    ("Rising energy prices are, however, expected to reach our target, leading the central bank to raise the policy rate to 2% by 2026.", 2, "%", 2026, "energy_management"),
    ("Our 2030 target for packaging waste sits beside 13% taxonomy eligible activities; we will publish it.", 13, "%", 2030, "energy_management"),
    ("With 5,000 employees, we aim to reduce energy use by 2030.", 5000, "employees", 2030, "energy_management"),
    ("We aim to reduce CO2 emissions and energy use by 2030.", 2, "emissions", 2030, "energy_management"),
    ("We work towards our goal of 24/7 carbon-free energy by 2030.", 7, "carbon-free", 2030, "energy_management"),
    ("We aim to reduce energy use by 30% in the coming years.", 30, "%", 2030, "energy_management"),  # the deadline is not in the quote
    ("We aim to cut energy use by 30 tonnes by 2030.", 30, "%", 2030, "energy_management"),  # the unit is not a percentage
])
def test_unrelated_nearby_number_cannot_become_the_target(text, value, unit, year, factor):
    vr = _goal(text, factor, value=value, unit=unit, year=year)
    assert vr.lifecycle == LIFECYCLE_REJECTED
    assert {"numeric_target_context_invalid", "numeric_target_subject_mismatch"} & set(vr.rejection_codes)


@pytest.mark.parametrize("sentence", [
    "We aim to reduce CO2 emissions and energy use by 2030.",
    "We work towards our goal of 24/7 carbon-free energy by 2030.",
    "With 5,000 employees, we aim to reduce energy use by 2030.",
    "Our alliance, which represents 40% of global assets, will set energy goals by 2030.",
])
def test_stray_figures_are_never_read_as_the_target_value(sentence):
    t = P.detect_target(sentence, 2025)
    assert t is not None and t.target_value is None and not t.quantitative


@pytest.mark.parametrize("text, value, unit, year, factor", [
    ("We aim to reduce energy consumption by 30% by 2030 compared with a 2020 baseline.", 30, "%", 2030, "energy_management"),
    ("Our goal is to reach 40% women in management positions by 2027.", 40, "%", 2027, "diversity_inclusion"),
    ("We commit to switch to 100% renewable electricity by 2028.", 100, "%", 2028, "energy_management"),
    ("Our target is net-zero energy emissions by 2040.", 0, "emissions", 2040, "energy_management"),
    ("By 2030 we commit to reduce energy use to 80 GWh.", 80, "GWh", 2030, "energy_management"),
    ("For energy, our target is 30% lower use by 2030.", 30, "%", 2030, "energy_management"),
    ("Our goal is to reach a 0% gender pay gap by 2030.", 0, "%", 2030, "diversity_inclusion"),
])
def test_valid_official_company_target_still_passes(text, value, unit, year, factor):
    vr = _goal(text, factor, value=value, unit=unit, year=year)
    assert vr.lifecycle == LIFECYCLE_ACCEPTED, vr.rejection_codes


@pytest.mark.parametrize("text", [
    "We implemented an energy management system at all plants in 2024.",
    "LED lighting was installed at all plants in 2024.",
    "The company completed energy audits at five sites in 2024.",
    "Northstar Manufacturing upgraded its heating and energy systems in 2024.",
])
def test_valid_official_company_execution_still_passes(text):
    assert _activity(text).lifecycle == LIFECYCLE_ACCEPTED


def test_first_party_sentences_do_not_need_the_company_name_once_identity_is_established():
    assert _goal("We aim to reduce energy consumption by 2030.").lifecycle == LIFECYCLE_ACCEPTED
    assert _goal("We aim to reduce energy consumption by 2030.", source_type="other_web").lifecycle == LIFECYCLE_REJECTED  # "we" is the author there


# ---------------------------------------------------------------- recommendation invariant + export
def test_recommendations_cite_only_accepted_scoring_evidence(client, sample_pdf):
    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    run_assessment(aid, llm=FakeLLM(), search=NullSearchProvider())
    _, evidence, recs = _rows(aid)
    accepted = {e.id for e in evidence if e.evidence_lifecycle == LIFECYCLE_ACCEPTED}
    cited = {i for r in recs for i in r.evidence_ids}
    assert cited and cited <= accepted
    data = client.get(f"/api/assessments/{aid}/export", params={"format": "json"}).json()
    used = {e["id"] for e in data["evidence"] if e["used_for_scoring"]}
    assert cited == used


def test_invariant_failure_fails_the_scoring_step(client, sample_pdf, monkeypatch):
    from app.services.pipeline import runner as runner_module
    from app.services.scoring.common import ScoreResult

    def citing_unknown_evidence(*_a, **_k):
        return ScoreResult(1, 6, "FOUND", "R", "reason", evidence_ids=[987654])

    monkeypatch.setattr(runner_module, "score_reporting", citing_unknown_evidence)
    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    run_assessment(aid, llm=FakeLLM(), search=NullSearchProvider())
    with SessionLocal() as db:
        a = db.get(Assessment, aid)
        assert a.status == "FAILED" and "ScoringInvariantError" in a.error
    assert _rows(aid)[2] == []
    assert "987654" not in client.get(f"/api/assessments/{aid}/status").text  # analysts see plain wording only


def test_export_contains_audit_validation_fields(client, sample_pdf):
    class WrongQuote(FakeLLM):
        def complete_json(self, system, user, schema=None, max_tokens=384):
            self.calls.append(user)
            if "DIMENSION:\nPlanning" in user:
                return json.dumps({"goal_found": True, "formalized": True, "quantitative": False, "evidence_quote": "This sentence was never in the report at all.", "confidence": 0.9})
            return super().complete_json(system, user, schema, max_tokens)

    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    run_assessment(aid, llm=WrongQuote(), search=NullSearchProvider())
    data = client.get(f"/api/assessments/{aid}/export", params={"format": "json"}).json()
    required = {"evidence_lifecycle", "rejection_codes", "company_match", "actor_match", "factor_match", "dimension_match", "numeric_match",
                "used_for_scoring", "scoring_exclusion_reason", "evidence_origin"}
    assert data["evidence"] and all(required <= set(e) for e in data["evidence"])
    fallback = [e for e in data["evidence"] if e["evidence_origin"] == "deterministic_quote_fallback"]
    assert fallback and all(e["llm_quote_failure"] == "not_in_source" for e in fallback)
    assert set(exporter.EVIDENCE_VALIDATION_COLUMNS) >= required | {"llm_quote_failure"}


def test_audit_fields_stay_out_of_the_normal_ui_endpoints(client, sample_pdf):
    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    run_assessment(aid, llm=FakeLLM(), search=NullSearchProvider())
    recs = client.get(f"/api/assessments/{aid}/recommendations").json()
    body = client.get(f"/api/assessments/{aid}/sources").text + json.dumps(recs) + client.get(f"/api/recommendations/{recs[0]['id']}").text
    for word in ("actor_match", "company_match", "rejection_codes", "scoring_exclusion_reason", "evidence_origin"):
        assert word not in body
