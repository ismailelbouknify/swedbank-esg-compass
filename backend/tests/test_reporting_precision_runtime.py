"""Reporting semantics, assurance scope, framework use, table row association and bounded extraction work.
Every company, domain, report and sentence here is invented: the rules are generic, not tuned to any real company.
No network: DNS and HTTP are mocked."""
from __future__ import annotations

import pymupdf
import pytest

from app.core.config import get_settings
from app.services.company.source_identity import CompanyIdentity
from app.services.extraction import patterns as P
from app.services.extraction.extractor import FactExtractor, Passage
from app.services.pipeline.runner import run_assessment
from app.services.scoring.common import EvidenceItem
from app.services.scoring.reporting import score_reporting
from app.services.verification.verifier import verify_candidate
from tests.conftest import FakeLLM
from tests.test_source_coverage import Search, _create, _page, _state, web  # noqa: F401

SITE = "https://www.kestreltiles.example"
NAME = "Kestrel Tiles AB"
COMPANY = CompanyIdentity(NAME)


def _passage(text: str, source_type: str = "company_website", page: int | None = 1, doc_year: int | None = 2025) -> Passage:
    return Passage(chunk_id=1, document_id=1, source_id=1, source_title="Page", source_url=f"{SITE}/p", source_type=source_type,
                   source_quality=0.8, is_public=True, page_number=page, text=text, doc_year=doc_year, identity_basis="official_domain")


def _item(i, claim_type, **facts) -> EvidenceItem:
    return EvidenceItem(i, claim_type, facts)


REGULAR = dict(public=True, regular=True)


# ---------------------------------------------------------------- 1. reporting semantics
@pytest.mark.parametrize("text", [
    "Our products offer some important advantages that play a key role for buyers who look for packaging that is affordable, "
    "sustainable and offers striking results.",
    "Our tiles are sustainable, durable and deliver excellent results in every environment.",
    "We combine ESG thinking with craftsmanship to create value for customers and the environment.",
    "Our sustainable production line delivers great results and an environmentally friendly finish.",
    "The new glazing machine reduces waste and gives our customers striking results.",
    "Sustainability is at the heart of everything we do at Kestrel Tiles.",
])
def test_marketing_and_generic_sustainability_claims_are_not_reporting(text):
    assert P.reporting_statement(text) is None
    p = _passage(text)
    assert not [c for c in FactExtractor(None, 2025).reporting([p], [p]) if c.claim_type == "voluntary_disclosure"]


@pytest.mark.parametrize("text", [
    "Kestrel Tiles publishes an annual sustainability report covering its environmental and social performance.",
    "We disclose our greenhouse gas emissions and energy use to stakeholders every year.",
    "Our ESG data for 2025 is available in the sustainability section of our annual report.",
    "In 2025 our Scope 1 and 2 emissions decreased by 9% compared with 2024.",
    "Our sustainability statement is prepared in accordance with ESRS and has been subject to limited assurance.",
    "Water consumption at our plants fell to 41,200 m3 in 2025, as reported in our sustainability statement.",
])
def test_sustainability_disclosure_qualifies_as_reporting(text):
    assert P.reporting_statement(text)


def test_marketing_only_website_cannot_create_reporting_level_3(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    pages = {
        f"{SITE}/": "Kestrel Tiles AB makes ceramic tiles. Our tiles are sustainable, durable and deliver striking results. " * 3,
        f"{SITE}/products": "Our products offer advantages for buyers who want affordable, sustainable and beautiful tiles. " * 3,
        f"{SITE}/production": "Our sustainable production line delivers great results and an environmentally friendly finish. " * 3,
    }
    for url, body in pages.items():
        web[url] = ("text/html", _page(f"{NAME}", body))
    aid = _create(client, name=NAME)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(u, NAME) for u in pages]))
    _, _, recs, evidence = _state(aid)
    assert not [e for e in evidence if e.factor == "reporting" and e.evidence_lifecycle == "accepted"]
    assert recs[("reporting", "reporting")].recommended_level != 3


# ---------------------------------------------------------------- 2. assurance scope
@pytest.mark.parametrize("sentence", [
    "Grantwell Audit AB provided limited assurance of the Sustainability Report and the Green Bond Report.",
    "We have conducted a limited assurance engagement of the sustainability statement for Kestrel Tiles AB for 2025.",
    "Selected sustainability metrics in this report were subject to limited assurance by an independent auditor.",
    "Our greenhouse gas emissions data was subject to reasonable assurance by an independent third party under ISAE 3410.",
])
def test_report_level_sustainability_assurance_qualifies(sentence):
    [m] = P.detect_assurance(sentence)
    assert m.covers_sustainability


@pytest.mark.parametrize("sentence", [
    # a certification scheme for production
    "The Ridgeline Mark is an independent assurance framework that verifies and certifies that clay extraction and production meet strict standards.",
    # the format of the annual report
    "Our responsibility is to obtain reasonable assurance whether the ESEF report is prepared in a format that meets the requirements.",
    "We performed reasonable assurance procedures on the iXBRL tagging of the consolidated financial statements.",
    # a facility, product, project, asset or credits
    "The kiln facility in Lund received independent third-party verification of its zero-waste performance.",
    "The Aurora tile range was externally verified by an independent third party.",
    "Each financed solar project undergoes independent third-party verification before disbursement.",
    "Our carbon removal credits will undergo independent third-party verification prior to issuance.",
    # someone else's report, or the financial statements
    "We have been engaged to undertake a limited assurance engagement of the Green Bond Report for 2025.",
    "The auditor provided reasonable assurance on the consolidated financial statements.",
])
def test_unrelated_certification_esef_facility_and_product_assurance_do_not_qualify(sentence):
    assert not any(m.covers_sustainability for m in P.detect_assurance(sentence))


def test_level_6_needs_qualifying_assurance_and_unrelated_assurance_stays_at_level_5():
    base = [_item(1, "report_published", **REGULAR), _item(2, "framework_reference", framework="ESRS", explicit=True, level5_eligible=True)]
    unrelated = [_item(3, "assurance_statement", sustainability_assurance=False, provider=None, standard=None, third_party=True)]
    assert score_reporting(base + unrelated).recommended_level == 5
    qualifying = [_item(4, "assurance_statement", sustainability_assurance=True, provider="Grantwell Audit", standard=None, third_party=True)]
    assert score_reporting(base + qualifying).recommended_level == 6


# ---------------------------------------------------------------- 3. framework use
def test_framework_use_from_a_whole_report_scores_level_5_but_references_do_not():
    use = "Kestrel Tiles' Sustainability Statement has been prepared in accordance with ESRS."
    refs = "Energy consumption is calculated as defined by GRI 302-1. Water withdrawal follows GRI 303-3. Index of GRI disclosures."
    ex = FactExtractor(None, 2025)

    def level(text):
        p = _passage(text, source_type="sustainability_report")
        cands = ex.reporting([p], [p])
        items = [EvidenceItem(i, c.claim_type, c.facts, source_id=1) for i, c in enumerate(cands, 1)
                 if verify_candidate(c, text, company=COMPANY).lifecycle == "accepted"]
        return score_reporting(items).recommended_level

    assert level("Sustainability report 2025. " + use) == 5
    assert level("Sustainability report 2025. " + refs) == 4


# ---------------------------------------------------------------- 4. table rows
STACKED = """Energy
Energy consumption by source
Unit
2022
2023
2024
2025
Fuel
MWh
12,400
11,900
11,300
10,800
Purchased electricity12
MWh
88,100
86,000
83,700
80,200
On-site renewable electricity
MWh
1,200
1,900
2,600
3,400
Total energy consumption
MWh
101,700
99,800
97,600
94,400
"""
INLINE = """Indicator Unit 2022 2023 2024 2025
Total electricity consumption MWh 88,100 86,000 83,700 80,200
Renewable electricity share % 41 48 55 61
Energy intensity MWh/tonne 1.92 1.85 1.80 1.71
"""


def _series(text):
    return {s.metric_label: s for s in P.extract_series(text)}


def test_each_mwh_row_keeps_its_own_label_unit_and_values():
    rows = _series(STACKED)
    assert rows["Total energy consumption (MWh)"].points == [(2022, 101700.0), (2023, 99800.0), (2024, 97600.0), (2025, 94400.0)]
    assert rows["Purchased electricity (MWh)"].points[0] == (2022, 88100.0)  # footnote '12' removed from the label
    assert rows["Fuel (MWh)"].points[-1] == (2025, 10800.0)
    assert rows["On-site renewable electricity (MWh)"].unit == "MWh"
    assert not any(label.strip().lower() == "mwh" for label in rows)


def test_only_unambiguous_relevant_rows_become_performance_evidence():
    cands = FactExtractor(None, 2025).performance("energy_management", [_passage(STACKED + "\n" + INLINE, "sustainability_report")])
    got = {c.facts["metric_label"]: c for c in cands if c.claim_type == "performance_series"}
    assert "Total energy consumption (MWh)" in got and got["Total energy consumption (MWh)"].facts["metric"] == "energy consumption"
    assert "Total electricity consumption MWh" in got or any("Total electricity consumption" in k for k in got)
    assert any("Renewable electricity share" in k for k in got)  # a real share, in %
    assert not any("On-site renewable electricity" in k for k in got)  # MWh is not a share
    assert not any(k.startswith("Fuel") or "Purchased electricity" in k for k in got)  # no configured metric: abstain
    assert all("unit n/a" not in c.claim_text and "[" in c.claim_text for c in got.values())


def test_chart_axis_scale_is_not_a_time_series():
    chart = "Figure 7.\nWaste performance\nLegend\nShare of waste recycled\n2022\n2023\n2024\n2025\n100\n75\n50\n25\n0\n"
    assert not P.extract_series(chart)


def test_reduction_labels_do_not_feed_a_lower_is_better_metric():
    text = "Unit 2022 2023 2024 2025\nPercentage of waste diverted from landfill % 70 74 79 83\n"
    cands = FactExtractor(None, 2025).performance("product_lifecycle", [_passage(text, "sustainability_report")])
    assert not [c for c in cands if c.claim_type == "performance_series" and c.facts.get("metric") == "waste to landfill"]


def test_unlabelled_unit_row_in_stacked_layout_is_rejected():
    text = "Unit\n2023\n2024\n2025\nMWh\n300\n280\n250\n"
    aligned, rejected = P.extract_series_with_rejections(text)
    assert not aligned and any(r.meta.get("reason") == "metric_label_missing" for r in rejected)


# ---------------------------------------------------------------- 5. bounded model work
GOAL = "We aim to reduce energy consumption at our plants by 20% by 2030 compared with 2024."


def test_overlapping_chunks_are_not_sent_to_the_model_twice():
    llm = FakeLLM()
    ex = FactExtractor(llm, 2025)
    a = _passage(f"Energy at Kestrel. {GOAL}")
    b = _passage(f"{GOAL} Our tiles are fired in electric kilns.")
    ex.planning("energy_management", [a])
    ex.planning("energy_management", [b])
    assert len(llm.calls) == 1 and ex.skipped_llm["overlap"] == 1


def test_expired_goals_and_activities_before_the_window_cost_no_model_call():
    llm = FakeLLM()
    ex = FactExtractor(llm, 2025)
    ex.planning("energy_management", [_passage("Our goal is to reduce energy consumption at our plants by 10% by 2015 compared with 2010.")])
    ex.execution("energy_management", [_passage("In 2016 we installed LED lighting at our plants to cut energy use.", doc_year=2016)])
    assert llm.calls == [] and ex.skipped_llm["deadline_passed"] == 1 and ex.skipped_llm["outside_window"] == 1


def test_model_budget_is_strict_and_the_rest_is_still_extracted_deterministically():
    llm = FakeLLM()
    ex = FactExtractor(llm, 2025, llm_budget=1)
    goals = [_passage(f"Plant {i}. We aim to reduce energy consumption at plant {i} by {10 + i}% by 2030 compared with 2024.") for i in range(3)]
    cands = ex.planning("energy_management", goals)
    assert len(llm.calls) == 1 and ex.budget_exhausted == 2
    assert len([c for c in cands if c.method == "heuristic"]) == 2  # strong, quantified goals are not lost


def _goal_page(topic_sentence: str) -> str:
    return _page(f"{NAME}", f"Kestrel Tiles AB. {topic_sentence} " * 2)


def test_budget_is_shared_round_robin_so_every_factor_gets_its_best_passage(client, web, monkeypatch):
    """Two goal passages per factor; a budget of 3 calls. Read question by question, all 3 calls would go to the
    first factor. Round-robin gives each of 3 different factors its best passage first."""
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    monkeypatch.setattr(get_settings(), "max_llm_calls_per_assessment", 3)
    topics = {
        "energy": ("We aim to reduce energy consumption at our plants by 20% by 2030.",
                   "Our target is to source 80% renewable electricity for our kilns by 2028."),
        "security": ("Our goal is ISO 27001 certification for our information security management by 2027.",
                     "We aim to complete cybersecurity training for all staff and improve data protection by 2026."),
        "diversity": ("Our goal is to reach 40% women in management positions by 2030.",
                      "We aim to close the gender pay gap and improve inclusion by 2028."),
    }
    urls = []
    for key, sentences in topics.items():
        url = f"{SITE}/{key}"
        body = f"Kestrel Tiles AB. {sentences[0]}" + "</p><p>" + "Background. " * 220 + "</p><p>" + sentences[1]
        web[url] = ("text/html", _page(NAME, body))
        urls.append((url, f"{NAME} {key}"))
    llm = FakeLLM()
    aid = _create(client, name=NAME)
    run_assessment(aid, llm=llm, search=Search(first=urls))
    a, *_ = _state(aid)
    assert len(llm.calls) == 3 and a.coverage_metrics["model_calls"] == 3
    factors = {c.split("FACTOR:\n", 1)[1].split("\n", 1)[0] for c in llm.calls}
    assert len(factors) == 3  # three different factors, not three calls on the first one
    assert a.coverage_metrics["budget_exhausted_passages"] >= 1


def _pdf(company: str) -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 780), f"{company}\nSustainability Report 2025\n"
                                  "We aim to reduce energy consumption by 30% by 2030 compared with 2020.\n" * 3, fontsize=11)
    doc.set_metadata({"title": f"{company} Sustainability Report 2025"})
    data = doc.tobytes()
    doc.close()
    return data


def test_the_same_pdf_under_two_addresses_is_parsed_once_and_timings_are_recorded(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    pdf = _pdf(NAME)
    first, mirror = f"{SITE}/files/sustainability-report-2025.pdf", "https://cdn.tilefiles.example/k/sustainability-report-2025.pdf"
    web[first] = ("application/pdf", pdf)
    web[mirror] = ("application/pdf", pdf)
    aid = _create(client, name=NAME)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(first, f"{NAME} report"), (mirror, f"{NAME} report")]))
    a, sources, *_ = _state(aid)
    assert sorted(s.status for s in sources) == ["PROCESSED", "SKIPPED"]
    m = a.coverage_metrics
    assert m["duplicate_documents_skipped"] == 1
    for key in ("model_calls", "model_seconds", "fetch_seconds", "pdf_parse_seconds", "total_seconds", "model_calls_avoided"):
        assert key in m
    assert {"retrieval", "grounding_validation", "extraction"} <= set(m["stage_seconds"])


# ---------------------------------------------------------------- metric meaning (not keyword length)
@pytest.mark.parametrize("label, unit, expected", [
    ("Share of renewable sources in total energy consumption", None, "renewable energy share"),
    ("Renewable electricity share", "%", "renewable energy share"),
    ("Share of fossil sources in total energy consumption", None, None),  # a share, but of no configured metric
    ("Total renewable energy consumption", "GWh", None),  # a component of the total, not the total
    ("Total fossil energy consumption", "GWh", None),
    ("Purchased energy consumption", "MWh", None),
    ("Renewable energy production", None, None),  # a quantity, never a share
    ("Self-generated renewable energy", "MWh", None),
    ("Total energy consumption per net revenue", None, "energy intensity"),  # per-unit = intensity
    ("Total energy consumption", "MWh", "energy consumption"),
    ("Energy intensity", "MWh/tonne", "energy intensity"),
])
def test_rows_map_to_the_metric_their_label_means(label, unit, expected):
    from app.config.loader import get_factor
    from app.services.extraction.extractor import _row_metric

    m = _row_metric(P.Series(label, unit, [(2024, 1.0), (2025, 2.0)], label), get_factor("energy_management"))
    assert (m["name"] if m else None) == expected


def test_a_rising_renewable_share_is_scored_as_improvement_not_deterioration():
    text = "Indicator 2023 2024 2025\nShare of renewable sources in total energy consumption % 58 64 76\n"
    [c] = [c for c in FactExtractor(None, 2025).performance("energy_management", [_passage(text, "sustainability_report")])
           if c.claim_type == "performance_series"]
    assert c.facts["metric"] == "renewable energy share" and c.facts["better"] == "higher"


# ---------------------------------------------------------------- negated assurance
@pytest.mark.parametrize("sentence", [
    "Prior year's sustainability statement has not been subject to limited assurance procedures.",
    "The sustainability report has not been externally assured.",
    "No external assurance was obtained for the sustainability metrics in this report.",
    "These sustainability disclosures are published without independent assurance.",
])
def test_negated_assurance_is_no_assurance(sentence):
    assert not any(m.covers_sustainability for m in P.detect_assurance(sentence))


def test_positive_report_level_assurance_stays_valid_next_to_a_negated_one():
    text = ("Grantwell Audit AB has performed a limited assurance engagement of the sustainability statement for 2025. "
            "The prior year's sustainability statement has not been subject to limited assurance.")
    assert [m.covers_sustainability for m in P.detect_assurance(text)] == [True, False]


# ---------------------------------------------------------------- reporting noise
@pytest.mark.parametrize("text", [
    "Our ambition is by 2030 to reduce CO2 emissions per tile by 65-75 per cent.",
    "We aim to cut Scope 1 and 2 emissions by 50% by 2030 compared with 2020.",
    "Our target is 100% renewable electricity at all plants by 2027.",
])
def test_an_emissions_target_alone_is_not_reporting_disclosure(text):
    assert P.reporting_statement(text) is None


@pytest.mark.parametrize("sentence", [
    "Evaluate whether the sustainability statement is prepared in accordance with ESRS in all material respects.",
    "By inquiries obtain an understanding of the reporting process for the statement prepared in accordance with ESRS.",
    "Nothing has come to our attention that causes us to believe the statement is not prepared in accordance with ESRS.",
])
def test_an_auditors_esrs_mention_alone_is_not_framework_adoption(sentence):
    assert not any(m.explicit for m in P.detect_frameworks(sentence))
    items = [_item(1, "report_published", **REGULAR),
             _item(2, "framework_reference", framework="ESRS", explicit=False, level5_eligible=True)]
    assert score_reporting(items).recommended_level == 4


def test_the_companys_own_esrs_statement_is_still_framework_adoption():
    assert P.detect_frameworks("Kestrel Tiles' sustainability statement is prepared in accordance with ESRS.")[0].explicit


# ---------------------------------------------------------------- budget exhausted -> gap, never false Level 1
def _report_with_goals() -> bytes:
    doc = pymupdf.open()
    body = (f"{NAME}\nSustainability Report 2025\n{NAME} publishes this sustainability report every year.\n"
            "We aim to reduce energy consumption at our plants by 20% by 2030.\n"
            "Our goal is ISO 27001 certification for our information security management by 2027.\n")
    doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 780), body, fontsize=11)
    doc.set_metadata({"title": f"{NAME} Sustainability Report 2025"})
    data = doc.tobytes()
    doc.close()
    return data


def _run_with_budget(client, web, monkeypatch, budget: int):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    monkeypatch.setattr(get_settings(), "allow_heuristic_fallback", False)
    monkeypatch.setattr(get_settings(), "max_llm_calls_per_assessment", budget)
    url = f"{SITE}/files/sustainability-report-2025.pdf"
    web[url] = ("application/pdf", _report_with_goals())
    nothing = FakeLLM({"DIMENSION:\nPlanning": '{"goal_found": false, "evidence_quote": "", "confidence": 0.3}'})
    aid = _create(client, name=NAME)
    run_assessment(aid, llm=nothing, search=Search(first=[(url, f"{NAME} sustainability report")]))
    return _state(aid)


def test_an_unread_question_is_insufficient_coverage_never_a_false_level_1(client, web, monkeypatch):
    a, _, recs, _ = _run_with_budget(client, web, monkeypatch, budget=1)
    assert a.coverage_metrics["primary_reports_processed"] == 1  # coverage itself is sufficient
    cut = a.coverage_metrics["questions_not_fully_analysed"]
    assert cut
    for key in cut:
        factor, dim = key.split(":")
        rec = recs[(factor, dim)]
        assert rec.coverage_state == "INSUFFICIENT_SOURCE_COVERAGE" and rec.recommended_level == 0 and rec.recommended_level != 1
        assert "processing limit" in rec.reason


def test_without_the_budget_the_same_question_is_an_honest_no_evidence(client, web, monkeypatch):
    _, _, recs, _ = _run_with_budget(client, web, monkeypatch, budget=0)  # 0 = no limit
    rec = recs[("data_security", "planning")]
    assert rec.coverage_state == "NO_EVIDENCE_FOUND" and rec.recommended_level == 1
