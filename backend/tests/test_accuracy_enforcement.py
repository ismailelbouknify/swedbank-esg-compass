"""Accuracy-enforcement gates: source identity, accepted-only scoring, actor and state validation, factor
gate, table alignment, reporting aggregation, export fields. All companies here are fictional."""
from __future__ import annotations

import httpx
import pytest
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.entities import AuditEvent, Evidence, Recommendation, Source
from app.services.company.source_identity import CompanyIdentity, assess_source_identity, looks_like_company_report
from app.services.export import exporter
from app.services.extraction import gates
from app.services.extraction import patterns as P
from app.services.extraction.extractor import Candidate, FactExtractor, Passage
from app.services.pipeline.runner import COMPANY_MISMATCH_MESSAGE, run_assessment
from app.services.scoring.common import ScoringInvariantError, assert_scoring_evidence
from app.services.search.base import NullSearchProvider, SearchProvider, SearchResult
from app.services.security import safe_fetch
from app.services.verification.verifier import (
    LIFECYCLE_ACCEPTED, LIFECYCLE_CANDIDATE, LIFECYCLE_REJECTED, series_alignment_established, verify_candidate,
)
from tests.conftest import FakeLLM

COMPANY = CompanyIdentity("Northstar Manufacturing AB", "556677-8899".replace("-", ""))
SAMPLE_COMPANY = "Nordvik Components AB"


def _passage(text: str, source_type: str = "sustainability_report") -> Passage:
    return Passage(chunk_id=1, document_id=1, source_id=1, source_title="Sustainability Report 2025", source_url="https://example.org/r.pdf",
                   source_type=source_type, source_quality=0.95, is_public=True, page_number=3, text=text, doc_year=2025)


def _candidate(factor: str, dimension: str, claim_type: str, text: str, facts: dict | None = None, *,
               source_type: str = "sustainability_report", method: str = "llm", page: str | None = None) -> Candidate:
    return Candidate(factor, dimension, claim_type, text[:100], facts or {}, text, _passage(page or text, source_type), method)


def _verify(c: Candidate, page: str | None = None, company: CompanyIdentity | None = COMPANY):
    return verify_candidate(c, page or c.passage.text, company=company)


def _activity(text: str, factor: str = "energy_management", **kw):
    return _verify(_candidate(factor, "execution", "activity", text, {"activity_found": True}, **kw))


def _goal(text: str, factor: str = "energy_management", **kw):
    return _verify(_candidate(factor, "planning", "goal", text, {"goal_found": True, "formalized": True}, **kw))


# ---------------------------------------------------------------- 1. source identity
REPORT_TEXT = "Northstar Manufacturing AB\nSustainability Report 2025\nWe reduced energy use. Northstar Manufacturing owns five plants."


def _assess(text: str, title: str | None = None, url: str | None = None, **kw):
    return assess_source_identity(COMPANY, title=title, url=url, text=text, **kw)


def test_own_report_matches_on_name():
    assert _assess(REPORT_TEXT).matched
    assert _assess("We reduced energy use.", title="Northstar Manufacturing Sustainability Report 2025").basis == "name_in_title"
    # an address alone proves nothing without the document's own text
    assert not _assess("We reduced energy use.", url="https://x.example/northstar-manufacturing-2025.pdf").matched
    assert _assess("Northstar Manufacturing reduced energy use.", url="https://x.example/northstar-manufacturing-2025.pdf").matched


def test_short_name_and_organisation_number_identify_the_company():
    text = "Northstar is a producer of tools. Northstar reduced its energy use. Northstar Manufacturing is its legal name, used in filings."
    assert _assess(text).basis in ("alias", "name_in_text")
    assert not _assess("Northstar is a producer of tools. Northstar reduced its energy use.").matched  # the first word alone is not enough
    assert _assess("Org. nr 556677-8899. Annual figures follow.").basis == "organisation_number"


def test_wrong_company_source_rejected():
    result = _assess("Bluefin Foods Ltd\nSustainability Report 2025\nBluefin Foods Ltd reduced energy use. Bluefin Foods grows.",
                     title="Bluefin Foods Sustainability Report 2025")
    assert not result.matched and result.basis == "mismatch"


def test_source_never_naming_the_company_rejected():
    assert not _assess("We reduced energy use across all of our plants this year.").matched


def test_similar_company_name_rejected():
    text = "Northstar Shipping AB\nNorthstar Shipping AB annual review.\nNorthstar Shipping operates vessels. Northstar Shipping grew."
    assert not _assess(text, title="Northstar Shipping Sustainability Report 2025").matched
    # a name continuing the assessed single-word name is another company too
    acme = CompanyIdentity("Acme")
    words = "Acme Foods sells. Acme Foods grew. Acme Foods hired. Acme Foods invested. Acme Foods expanded."
    assert not assess_source_identity(acme, title=None, url=None, text=words).matched
    # a name that only begins the same way is no match at all
    assert not assess_source_identity(CompanyIdentity("Northstar"), title=None, url=None, text="Northstarlings reduced energy use.").matched


def test_generic_first_word_alone_does_not_identify_a_company():
    identity = CompanyIdentity("Nordic Steel AB")
    assert not assess_source_identity(identity, title=None, url=None, text="Nordic Paper produces paper. Nordic Paper grew.").matched


def test_passing_mention_in_another_companys_report_rejected():
    filler = "Bluefin Foods reduced energy use. " * 10
    text = filler + "Among our suppliers is Northstar Manufacturing AB, and Northstar Manufacturing delivers tools."
    assert not _assess(text, title="Bluefin Foods Sustainability Report 2025").matched


def test_analyst_supplied_domain_establishes_identity_but_a_guess_does_not():
    assert _assess("We reduced energy use.", url="https://own.example/r.pdf", on_official_domain=True).basis == "official_domain"
    assert not _assess("We reduced energy use.", url="https://own.example/r.pdf", on_official_domain=False).matched


def test_upload_without_a_name_is_kept_but_upload_of_another_company_is_not():
    assert _assess("We reduced energy use.", uploaded=True).basis == "analyst_upload"
    other = "Bluefin Foods AB reduced energy use. Bluefin Foods AB grew."
    assert not _assess(other, uploaded=True).matched


def test_auditor_and_title_noise_are_not_treated_as_another_company():
    text = REPORT_TEXT + " KPMG AB assured this report. KPMG AB issued a statement. KPMG AB signed. KPMG AB reviewed."
    assert _assess(text, title="Towards Net Zero Sustainability Report 2025").matched


class _Search(SearchProvider):
    name = "test"

    def __init__(self, urls):
        self.urls = urls

    def search(self, query, max_results):
        return [SearchResult(f"{SAMPLE_COMPANY} sustainability report 2025", u, "sustainability report", query) for u in self.urls]


def _assessment(client, name: str = SAMPLE_COMPANY, **kw) -> int:
    body = {"company_name": name, "year": 2025, "enable_web_search": False, "confirm_new": True, **kw}
    r = client.post("/api/assessments", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _upload(client, aid: int, pdf) -> None:
    with open(pdf, "rb") as f:
        assert client.post(f"/api/assessments/{aid}/documents", files={"file": ("report.pdf", f, "application/pdf")}).status_code == 201


def _recs(aid: int) -> dict[tuple[str, str], Recommendation]:
    with SessionLocal() as db:
        return {(r.factor, r.dimension): r for r in db.scalars(select(Recommendation).where(Recommendation.assessment_id == aid))}


def _evidence(aid: int) -> list[Evidence]:
    with SessionLocal() as db:
        return list(db.scalars(select(Evidence).where(Evidence.assessment_id == aid)))


def test_wrong_company_upload_never_reaches_extraction_or_scoring(client, sample_pdf):
    aid = _assessment(client, name="Bluefin Foods Ltd")
    _upload(client, aid, sample_pdf)
    llm = FakeLLM()
    run_assessment(aid, llm=llm, search=NullSearchProvider())
    assert llm.calls == [], "no model time is spent on the wrong company"
    assert _evidence(aid) == []
    assert all(r.recommended_level == 0 and r.coverage_state == "INSUFFICIENT_SOURCE_COVERAGE" for r in _recs(aid).values())
    with SessionLocal() as db:
        source = db.scalars(select(Source).where(Source.assessment_id == aid)).one()
        events = {e.event_type for e in db.scalars(select(AuditEvent).where(AuditEvent.assessment_id == aid))}
    assert (source.status, source.company_match, source.rejection_reason) == ("REJECTED", "mismatch", "source_company_mismatch")
    assert "source_company_mismatch" in events
    st = client.get(f"/api/assessments/{aid}/status").json()
    assert st["status"] == "COMPLETED" and COMPANY_MISMATCH_MESSAGE in st["notices"]


# ---------------------------------------------------------------- 3. actors
@pytest.mark.parametrize("text", [
    "The government launched a national energy efficiency programme in 2024.",
    "The authorities have introduced new energy efficiency rules.",
    "Customers installed solar panels on their roofs in 2024.",
    "Our customers implemented energy efficiency measures in 2024.",
    "Energy labelling was introduced by the European Commission in 2024.",
    "Bluefin Foods AB installed LED lighting in 2024.",
    "The industry recently adopted a new energy standard.",
])
def test_external_actor_cannot_count_as_company_execution(text):
    vr = _activity(text)
    assert vr.lifecycle == LIFECYCLE_REJECTED and "external_actor" in vr.rejection_codes


@pytest.mark.parametrize("text", [
    "We installed LED lighting at five plants in 2024.",
    "Northstar Manufacturing installed LED lighting in 2024.",
    "LED lighting was installed at all plants in 2024.",
    "Together with customers, we implemented an energy efficiency programme in 2024.",
    "Following the government's guidance, we upgraded our energy systems in 2024.",
])
def test_company_execution_still_accepted(text):
    assert _activity(text).lifecycle == LIFECYCLE_ACCEPTED


@pytest.mark.parametrize("text, factor", [
    ("Customer data encryption was implemented in 2024.", "data_security"),
    ("Supplier audits of data security were conducted in 2024.", "data_security"),
])
def test_actor_words_used_as_modifiers_do_not_trigger_the_gate(text, factor):
    assert _activity(text, factor).lifecycle == LIFECYCLE_ACCEPTED


def test_privacy_regulation_is_not_company_data_security_execution():
    for text in ("The GDPR was adopted to strengthen data protection in Europe.",
                 "New data protection legislation was adopted in 2023.",
                 "Companies must have implemented data protection measures under the GDPR."):
        vr = _activity(text, "data_security")
        assert vr.lifecycle == LIFECYCLE_REJECTED and "external_actor" in vr.rejection_codes


@pytest.mark.parametrize("text", [
    "The EU has set a target to reduce energy use by 32% by 2030.",
    "The government's energy goal will cut consumption by 20% by 2035.",
    "The national energy target is to reach 50% renewable energy by 2030.",
])
def test_external_goal_cannot_count_as_company_planning(text):
    vr = _goal(text)
    assert vr.lifecycle == LIFECYCLE_REJECTED and "external_actor" in vr.rejection_codes


def test_benchmark_stated_inside_the_companys_own_goal_is_kept():
    text = "We aim to reduce energy use by 30% by 2030, in line with the EU target and the Paris Agreement."
    assert _goal(text).lifecycle == LIFECYCLE_ACCEPTED


def test_sector_figures_cannot_count_as_company_performance():
    row = "Sector average energy intensity 124 116"
    vr = _verify(_candidate("energy_management", "performance", "performance_series", row,
                            {"metric": "energy intensity", "metric_label": "Sector average energy intensity", "unit": "kWh",
                             "points": [[2023, 124.0], [2024, 116.0]]}, method="regex", page=f"Indicator 2023 2024\n{row}"))
    assert "external_actor" in vr.rejection_codes and vr.lifecycle == LIFECYCLE_REJECTED


# ---------------------------------------------------------------- 4. state + negation
@pytest.mark.parametrize("text", [
    "Our energy target of 30% by 2030 was discontinued in 2024.",
    "We no longer have a target to reduce energy use by 2030.",
    "We abandoned our goal to reduce energy consumption by 2030.",
    "We do not have a target for energy consumption at this stage, and will not by 2030.",
    "We are considering setting an energy target to apply by 2030.",
])
def test_inactive_goal_cannot_count_as_planning(text):
    vr = _goal(text)
    assert vr.lifecycle == LIFECYCLE_REJECTED and "inactive_goal" in vr.rejection_codes


def test_active_goal_next_to_a_discontinued_one_is_kept():
    text = "Our earlier energy goal was discontinued. We will now aim to reduce energy use by 20% by 2030."
    assert _goal(text).lifecycle == LIFECYCLE_ACCEPTED


@pytest.mark.parametrize("text", [
    "The energy audit was not carried out in 2024.",
    "The LED programme was not implemented during the year.",
    "The energy efficiency project has not yet been completed.",
    "We did not conduct an energy audit in 2024.",
    "No energy audits were conducted in 2024.",
    "The energy audit is to be completed in 2026.",
    "We will have completed the LED rollout by 2027.",
    "The installation of LED lighting was started and then cancelled.",
])
def test_activity_not_carried_out_cannot_count_as_execution(text):
    vr = _activity(text)
    assert vr.lifecycle == LIFECYCLE_REJECTED and "activity_not_performed" in vr.rejection_codes


@pytest.mark.parametrize("text", [
    "We conducted energy audits at five sites but did not complete the sixth.",
    "We not only installed LED lighting but also upgraded heating.",
    "Not all sites have been audited, but we audited energy use at five plants in 2024.",
])
def test_partly_negated_sentences_keep_the_activity_that_happened(text):
    assert _activity(text).lifecycle == LIFECYCLE_ACCEPTED


def test_negated_and_external_sentences_are_filtered_before_the_model_is_called():
    llm = FakeLLM()
    ex = FactExtractor(llm, 2025)
    text = "The energy audit was not carried out in 2024. The government launched an energy programme in 2024."
    assert ex.execution("energy_management", [_passage(text)]) == []
    goal_text = "Our energy target of 30% by 2030 was discontinued in 2024 after the review of the plan."
    assert ex.planning("energy_management", [_passage(goal_text)]) == []
    assert llm.calls == [] and ex.llm_calls == 0


def test_valid_passage_needs_exactly_one_extraction_call():
    llm = FakeLLM()
    ex = FactExtractor(llm, 2025)
    out = ex.execution("energy_management", [_passage("We installed LED lighting and completed an energy audit in 2024.")])
    assert len(out) == 1 and ex.llm_calls == 1


# ---------------------------------------------------------------- 5. factor gate
@pytest.mark.parametrize("factor, text", [
    ("diversity_inclusion", "We installed LED lighting and reduced energy consumption in 2024."),
    ("diversity_inclusion", "We implemented a carbon reduction plan for fleet fuel in 2024."),
    ("product_lifecycle", "We implemented our sustainability framework and updated our ESG governance in 2024."),
    ("data_security", "We conducted a sustainability review of our operations in 2024."),
    ("labor_practices", "We launched a collaboration with laboratory partners in 2024."),  # "labor" inside other words
    ("labor_practices", "We upgraded the sewage plant in 2024."),  # "wage" inside "sewage"
    ("energy_management", "The team led a diversity programme and completed it in 2024."),  # the verb "led" is not LED
])
def test_cross_factor_evidence_rejected(factor, text):
    vr = _activity(text, factor)
    assert vr.lifecycle == LIFECYCLE_REJECTED and "factor_mismatch" in vr.rejection_codes


def test_incidental_mention_of_a_factor_does_not_outweigh_the_real_topic():
    text = "Women engineers installed LED lighting, reducing energy consumption and electricity use in 2024."
    vr = _activity(text, "diversity_inclusion")
    assert vr.lifecycle == LIFECYCLE_REJECTED and "cross_factor_mismatch" in vr.rejection_codes


@pytest.mark.parametrize("text", [
    "We updated the employee lifecycle process in 2024.",
    "We updated our materiality assessment in 2024.",
    "We updated the training materials for our staff in 2024.",
])
def test_general_text_does_not_support_product_lifecycle(text):
    assert _activity(text, "product_lifecycle").lifecycle == LIFECYCLE_REJECTED


@pytest.mark.parametrize("factor, text", [
    ("product_lifecycle", "We launched a packaging recycling programme in 2024."),
    ("diversity_inclusion", "We launched a mentoring programme for women in 2024."),
    ("labor_practices", "We signed a new collective agreement with the trade union in 2024."),
    ("data_security", "We completed ISO 27001 certification in 2024."),
])
def test_on_factor_evidence_still_accepted(factor, text):
    assert _activity(text, factor).lifecycle == LIFECYCLE_ACCEPTED


def test_acronyms_match_in_capitals_only_when_the_text_has_capitals():
    assert P.contains_term("We installed LED lights.", "LED")
    assert not P.contains_term("The manager led the work.", "LED")
    assert P.contains_term("we installed led lights", "LED")  # lower-cased input


# ---------------------------------------------------------------- 6. tables + numeric grounding
def _table(*rows: str) -> str:
    return "\n".join(rows)


def test_row_filling_the_year_columns_forms_a_series():
    series = P.extract_series(_table("Indicator 2023 2024 2025", "Energy intensity (kWh/unit) 124 116 108"))
    assert series[0].points == [(2023, 124.0), (2024, 116.0), (2025, 108.0)] and series[0].unit == "kWh/unit"


def test_number_inside_the_label_is_not_a_cell():
    series = P.extract_series(_table("Indicator 2023 2024", "Scope 1 emissions 124 116"))
    assert series[0].metric_label == "Scope 1 emissions" and series[0].points == [(2023, 124.0), (2024, 116.0)]


def test_same_unit_in_every_cell_is_still_aligned():
    series = P.extract_series(_table("Indicator 2023 2024", "Energy use 124 kWh 116 kWh"))
    assert series[0].points == [(2023, 124.0), (2024, 116.0)] and series[0].unit == "kWh"


def test_values_without_year_headers_create_no_series():
    assert P.extract_series(_table("Women in management 52 48", "Energy use 124 116")) == []


@pytest.mark.parametrize("rows", [
    ("Indicator 2023 2024 2025", "Energy use 52 48"),  # fewer values than columns
    ("Indicator 2023 2024", "Energy use 12 52 48"),  # more values than columns
])
def test_row_not_filling_the_columns_is_rejected_with_a_reason(rows):
    series, rejected = P.extract_series_with_rejections(_table(*rows))
    assert series == []
    assert [r.metric_label for r in rejected] == ["Energy use"]


def test_change_column_is_not_read_as_a_year():
    # the earlier parser took the last two numbers: 2024=12, and the change column as 2025
    series = P.extract_series(_table("Indicator 2024 2025 Change", "Energy use 124 116 8"))
    assert series[0].points == [(2024, 124.0), (2025, 116.0)]


def test_target_column_is_not_an_actual_value():
    series = P.extract_series(_table("Indicator 2024 2025 Target 2030", "Energy use 124 116 80"))
    assert series[0].points == [(2024, 124.0), (2025, 116.0)]


def test_prose_under_a_header_is_not_a_table_row():
    text = _table("Indicator 2023 2024", "Energy use 124 116",
                  "Our energy programme now covers 12 sites across 8 countries and continues to expand further.",
                  "Water use 90 85")
    assert [s.metric_label for s in P.extract_series(text)] == ["Energy use"]


def test_sentence_with_two_metrics_for_the_same_years_is_rejected():
    series, rejected = P.extract_series_with_rejections("Energy 2023: 100, 2024: 90, water 2023: 5, 2024: 4.")
    assert series == [] and rejected


def test_ambiguous_row_is_recorded_as_rejected_evidence_and_never_scored():
    text = _table("Indicator 2023 2024 2025", "Energy intensity 52 48")
    out = FactExtractor(None, 2025).performance("energy_management", [_passage(text)])
    assert [c.facts.get("ambiguous_alignment") for c in out] == ["2_values_for_3_columns"]
    vr = _verify(out[0], text)
    assert vr.lifecycle == LIFECYCLE_REJECTED and "ambiguous_table_alignment" in vr.rejection_codes


def test_model_cannot_assign_years_to_bare_numbers():
    text = "Energy intensity 52 48. This report covers 2023 and 2024."
    c = _candidate("energy_management", "performance", "performance_series", "Energy intensity 52 48",
                   {"metric": "energy intensity", "metric_label": "Energy intensity", "unit": None, "points": [[2024, 52.0], [2023, 48.0]]},
                   page=text)
    vr = _verify(c, text)
    assert vr.lifecycle == LIFECYCLE_REJECTED and "ambiguous_table_alignment" in vr.rejection_codes


def test_model_series_stated_with_years_or_under_headers_is_kept():
    points = [[2023, 124.0], [2024, 116.0]]
    assert series_alignment_established(points, "Energy use was 124 kWh in 2023 and 116 kWh in 2024.", "")
    assert series_alignment_established(points, "see table", _table("Indicator 2023 2024", "Energy use 124 116"))
    assert not series_alignment_established([[2023, 116.0], [2024, 124.0]], "Energy use was 124 kWh in 2023 and 116 kWh in 2024.", "")


# ---------------------------------------------------------------- 2 + 7. accepted-only scoring, reporting
def test_invariant_helper_stops_scoring_on_non_accepted_evidence(caplog):
    assert assert_scoring_evidence([1, 3], {1, 3}) == [1, 3]
    with pytest.raises(ScoringInvariantError):
        assert_scoring_evidence([1, 2, 3], {1, 3})
    assert "scoring_invariant_violation" in caplog.text


class _FabricatingLLM(FakeLLM):
    """Answers every question with a quote that is not in any document."""

    def complete_json(self, system, user, schema=None, max_tokens=384) -> str:
        self.calls.append(user)
        if "DIMENSION:\nPlanning" in user:
            return ('{"goal_found": true, "formalized": true, "quantitative": true, "target_value": 90, "target_unit": "%", '
                    '"target_year": 2030, "evidence_quote": "We promise to reach 90% of everything by 2030 without fail.", "confidence": 0.9}')
        if "DIMENSION:\nExecution" in user:
            return ('{"activity_found": true, "activity_description": "x", "activity_year": 2025, "linked_to_goal": true, '
                    '"evidence_quote": "We installed energy systems on the moon in 2025 and audited privacy data security diversity.", "confidence": 0.9}')
        return super().complete_json(system, user, schema, max_tokens)


def _accepted_ids(evidence: list[Evidence]) -> set[int]:
    return {e.id for e in evidence if e.evidence_lifecycle == LIFECYCLE_ACCEPTED}


def test_recommendation_evidence_is_always_a_subset_of_accepted_evidence(client, sample_pdf):
    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    run_assessment(aid, llm=FakeLLM(), search=NullSearchProvider())
    evidence = _evidence(aid)
    accepted = _accepted_ids(evidence)
    recs = _recs(aid)
    assert accepted and recs
    for r in recs.values():
        assert set(r.evidence_ids) <= accepted
        assert set(r.confidence_breakdown["used_evidence_ids"]) <= accepted
        for cond in (r.conditions or {}).values():
            assert set(cond["evidence_ids"]) <= accepted


def test_unconfirmed_evidence_cannot_appear_in_recommendations_or_raise_levels(client, sample_pdf):
    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    run_assessment(aid, llm=_FabricatingLLM(), search=NullSearchProvider())
    evidence = _evidence(aid)
    fabricated = [e for e in evidence if "without fail" in e.evidence_text or "on the moon" in e.evidence_text]
    assert fabricated, "the fabricated quotes were extracted"
    assert all(e.evidence_lifecycle != LIFECYCLE_ACCEPTED and "quote_not_found" in e.rejection_codes for e in fabricated)
    ids = {e.id for e in fabricated}
    recs = _recs(aid)
    assert not ids & {i for r in recs.values() for i in r.evidence_ids}
    # the level rests on the real goal sentence in the report, never on the invented 90% by 2030
    rec = recs[("energy_management", "planning")]
    used = [e for e in evidence if e.id in rec.evidence_ids]
    assert rec.recommended_level == 5 and used and all(e.evidence_origin and "30%" in e.evidence_text for e in used)
    assert ids & {i for r in recs.values() for i in r.confidence_breakdown["review_evidence_ids"]}


def test_rejected_evidence_cannot_affect_scores(client, sample_pdf):
    """The same run, with and without extra rejected evidence, gives the same levels."""
    baseline = _assessment(client)
    _upload(client, baseline, sample_pdf)
    run_assessment(baseline, llm=FakeLLM(), search=NullSearchProvider())

    class _ExternalActorLLM(FakeLLM):
        def complete_json(self, system, user, schema=None, max_tokens=384) -> str:
            self.calls.append(user)
            answer = super().complete_json(system, user, schema, max_tokens)
            if "DIMENSION:\nExecution" in user and '"activity_found": true' in answer:
                return ('{"activity_found": true, "activity_description": "x", "activity_year": 2025, "linked_to_goal": true, '
                        '"evidence_quote": "The government launched an energy and data security and diversity programme in 2025.", "confidence": 0.9}')
            return answer

    other = _assessment(client)
    _upload(client, other, sample_pdf)
    run_assessment(other, llm=_ExternalActorLLM(), search=NullSearchProvider())
    for key, rec in _recs(other).items():
        assert not set(rec.evidence_ids) - _accepted_ids(_evidence(other)), key
    assert all(r.recommended_level == 1 for k, r in _recs(other).items() if k[1] == "execution")
    assert any(r.recommended_level > 1 for k, r in _recs(baseline).items() if k[1] == "execution")
    assert all("external_actor" in e.rejection_codes for e in _evidence(other) if e.dimension == "execution" and e.extraction_method == "llm")


def test_accepted_valid_evidence_still_reaches_scoring(client, sample_pdf):
    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    run_assessment(aid, llm=FakeLLM(), search=NullSearchProvider())
    recs = _recs(aid)
    assert recs[("energy_management", "planning")].recommended_level >= 4
    assert recs[("energy_management", "execution")].recommended_level >= 2
    assert recs[("energy_management", "performance")].recommended_level == 2
    assert recs[("diversity_inclusion", "planning")].recommended_level >= 4
    assert sum(1 for r in recs.values() if r.evidence_ids) >= 8


def test_processed_report_prevents_false_no_reporting(client, sample_pdf):
    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    run_assessment(aid, llm=FakeLLM(), search=NullSearchProvider())
    rec = _recs(aid)[("reporting", "reporting")]
    assert rec.recommended_level >= 4 and rec.triggered_rule != "R1_NO_REPORTING_EVIDENCE" and rec.evidence_ids


def test_report_found_at_a_neutral_web_address_counts_as_a_report(client, sample_pdf, monkeypatch):
    """Classified from the document: the web address gave no hint that this is a report."""
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))])
    pdf = sample_pdf.read_bytes()
    handler = lambda _r: httpx.Response(200, headers={"content-type": "application/pdf"}, content=pdf)  # noqa: E731
    monkeypatch.setattr(safe_fetch, "build_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    url = "https://files.hosting.example/documents/doc-7781.pdf"

    class Plain(_Search):
        def search(self, query, max_results):
            return [SearchResult("Nordvik Components document", u, "document", query) for u in self.urls]

    aid = _assessment(client, enable_web_search=True)
    run_assessment(aid, llm=FakeLLM(), search=Plain([url]))
    with SessionLocal() as db:
        source = db.scalars(select(Source).where(Source.assessment_id == aid, Source.url == url)).one()
    assert source.source_type == "sustainability_report" and source.status == "PROCESSED" and source.company_match
    rec = _recs(aid)[("reporting", "reporting")]
    assert rec.recommended_level >= 4 and rec.triggered_rule != "R1_NO_REPORTING_EVIDENCE"


def test_only_a_published_report_is_reclassified():
    assert looks_like_company_report("Sustainability Report 2025", "We reduced energy use.")
    assert looks_like_company_report(None, "Environmental Report 2024\nWe reduced energy use.")
    assert not looks_like_company_report("Press release", "We reduced energy use.")
    assert not looks_like_company_report("Sustainability Report 2025", "An analysis of a company by outside researchers.")


# ---------------------------------------------------------------- 8. export
def test_export_includes_validation_fields(client, sample_pdf):
    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    run_assessment(aid, llm=_FabricatingLLM(), search=NullSearchProvider())
    data = client.get(f"/api/assessments/{aid}/export", params={"format": "json"}).json()
    fields = set(exporter.EVIDENCE_VALIDATION_COLUMNS) - {"llm_quote_failed", "llm_quote_failure"}  # present only on the items concerned
    assert fields == {"evidence_lifecycle", "evidence_origin", "rejection_codes", "company_match", "actor_match", "factor_match", "dimension_match", "numeric_match",
                      "used_for_scoring", "scoring_exclusion_reason"}
    assert all(fields <= set(e) for e in data["evidence"])

    used = [e for e in data["evidence"] if e["used_for_scoring"]]
    assert used and all(e["evidence_lifecycle"] == LIFECYCLE_ACCEPTED and e["scoring_exclusion_reason"] is None for e in used)
    excluded = [e for e in data["evidence"] if "quote_not_found" in e["rejection_codes"]]
    assert excluded and all(not e["used_for_scoring"] and "quote_not_found" in e["scoring_exclusion_reason"] for e in excluded)
    assert data["sources"][0]["company_match"] == "name_in_text" or data["sources"][0]["company_match"] == "name_in_title"
    assert client.get(f"/api/assessments/{aid}/export", params={"format": "xlsx"}).status_code == 200


def test_analyst_endpoints_do_not_expose_validation_internals(client, sample_pdf):
    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    run_assessment(aid, llm=FakeLLM(), search=NullSearchProvider())
    body = client.get(f"/api/assessments/{aid}/sources").text + client.get(f"/api/assessments/{aid}/recommendations").text
    for word in ("company_match", "rejection_reason", "rejection_codes", "review_evidence_ids", "used_for_scoring"):
        assert word not in body


def test_soft_candidate_is_still_only_a_candidate():
    # a paraphrased quote is not rejected outright, but it is not accepted either
    page = "Energy figures appear elsewhere in this report."
    vr = _verify(_candidate("energy_management", "execution", "activity", "We installed LED lights at the plant in 2024.",
                            {"activity_found": True}, page=page), page)
    assert vr.lifecycle == LIFECYCLE_CANDIDATE and "quote_not_found" in vr.rejection_codes


def test_gates_have_no_opinion_without_any_activity_or_goal_words():
    assert not gates.activity_not_performed("Energy figures appear elsewhere.")
    assert not gates.external_actor_only("Energy figures appear elsewhere.")
