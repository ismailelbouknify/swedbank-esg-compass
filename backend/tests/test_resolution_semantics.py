"""Source resolution, factor-level coverage, reporting semantics and aggregation, factor/actor/purpose
qualification, execution grounding, quote alignment and table alignment. Generic: all companies are fictional
and nothing is hard-coded per company. No network: DNS and HTTP are mocked."""
from __future__ import annotations

import pymupdf
import pytest
from sqlalchemy import select

from app.config.loader import get_factor
from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models.entities import Assessment, Evidence, Recommendation, Source
from app.services.company.source_identity import CompanyIdentity, assess_source_identity
from app.services.documents.html_extractor import extract_html
from app.services.documents.report_kind import ANNUAL_REPORT, SUSTAINABILITY_REPORT, primary_report_kind
from app.services.extraction import gates
from app.services.extraction import patterns as P
from app.services.extraction.extractor import Candidate, FactExtractor, Passage
from app.services.pipeline import discovery
from app.services.pipeline.coverage import INSUFFICIENT_SOURCE_COVERAGE, NO_EVIDENCE_FOUND, CoverageCheck, coverage_state
from app.services.pipeline.runner import run_assessment
from app.services.scoring.common import EvidenceItem
from app.services.scoring.reporting import score_reporting
from app.services.verification.verifier import verify_candidate
from tests.conftest import FakeLLM
from tests.test_source_coverage import NAME, SITE, Search, _create, _page, _report_pdf, _state, web  # noqa: F401

COMPANY = CompanyIdentity("Northstar Manufacturing AB")


def _passage(text: str, source_type: str = "sustainability_report", **kw) -> Passage:
    return Passage(chunk_id=1, document_id=1, source_id=1, source_title=kw.pop("title", "Report"), source_url=f"{SITE}/r",
                   source_type=source_type, source_quality=0.9, is_public=True, page_number=kw.pop("page", 3), text=text,
                   doc_year=kw.pop("doc_year", 2025), identity_basis="official_domain", **kw)


def _cand(factor, dim, claim_type, quote, facts, passage, method="llm", **kw) -> Candidate:
    return Candidate(factor, dim, claim_type, kw.pop("claim", quote[:80]), facts, quote, passage, method, **kw)


def _item(i, claim_type, **facts) -> EvidenceItem:
    return EvidenceItem(i, claim_type, facts)


# ---------------------------------------------------------------- 5. primary report classification
@pytest.mark.parametrize("title, url, kind", [
    ("Annual and Sustainability Report 2025", None, SUSTAINABILITY_REPORT),
    ("Sustainability Report 2025", None, SUSTAINABILITY_REPORT),
    ("ESG Report 2024", None, SUSTAINABILITY_REPORT),
    ("Integrated Report 2025", None, SUSTAINABILITY_REPORT),
    ("Sustainability Statement 2025", None, SUSTAINABILITY_REPORT),
    ("2026 Environmental Report", None, SUSTAINABILITY_REPORT),
    ("Annual Report 2025", None, ANNUAL_REPORT),
    (None, "https://cdn.example/files/acme-annual-report-2025.pdf", ANNUAL_REPORT),
])
def test_company_wide_reports_are_primary(title, url, kind):
    assert primary_report_kind(title, "", url) == kind


@pytest.mark.parametrize("title, url", [
    (None, "https://www.example.com/files/acme_ex60_carbonfootprintreport.pdf"),
    ("Model X Carbon Footprint Report", None),
    ("Product Life Cycle Assessment report", None),
    ("Environmental Product Declaration", None),
    ("Position paper on circular economy", None),
    ("Sustainability Update Q3 2025", None),
    ("Capital Markets Day presentation", None),
    ("Interim report January-September 2025", None),
    ("Climate action - company website", "https://www.example.com/report-link.pdf"),
])
def test_product_and_other_documents_are_not_primary_reports(title, url):
    assert primary_report_kind(title, "", url) is None


def test_document_title_on_the_cover_decides_for_an_opaque_file_name():
    cover = "Annual and Sustainability Report 2025\nFreedom to move"
    assert primary_report_kind("Freedom to", cover, "https://filings.example/afw/files/press/acme/202603050673-1.pdf") == SUSTAINABILITY_REPORT
    assert primary_report_kind("EX60", "Carbon footprint report\nEX60", "https://www.example.com/x.pdf") is None


def test_classify_source_does_not_call_any_report_pdf_an_annual_report():
    from app.services.company.identify import classify_source

    stype, _ = classify_source(f"{SITE}/files/model_x_carbonfootprintreport.pdf", "", "", NAME, "northstarmanufacturing.example")
    assert stype not in ("annual_report", "sustainability_report")
    stype, _ = classify_source(f"{SITE}/files/annual-and-sustainability-report-2025.pdf", "", "", NAME, "northstarmanufacturing.example")
    assert stype == "sustainability_report"


def _pdf(title: str, body: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), body, fontsize=11)
    doc.set_metadata({"title": title})
    data = doc.tobytes()
    doc.close()
    return data


PRODUCT_PDF = _pdf("Model X Carbon Footprint Report",
                   "Model X Carbon Footprint Report\nNorthstar Manufacturing AB\nThis report describes the life cycle "
                   "carbon footprint of the Model X product. We installed new recycling equipment in 2025. " * 3)


def test_product_report_is_not_counted_as_primary_coverage(client, web, monkeypatch):
    """Case C: a product-specific PDF is processed, the company-wide report is not."""
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    url = f"{SITE}/files/model_x_carbonfootprintreport.pdf"
    web[url] = ("application/pdf", PRODUCT_PDF)
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(url, f"{NAME} carbon footprint report")]))
    a, sources, recs, _ = _state(aid)
    src = next(s for s in sources if s.url == url)
    assert src.status == "PROCESSED" and src.source_type not in ("annual_report", "sustainability_report")
    assert a.coverage_metrics["primary_reports_processed"] == 0
    assert recs[("data_security", "planning")].coverage_state == INSUFFICIENT_SOURCE_COVERAGE


# ---------------------------------------------------------------- 6. landing page -> primary report
def test_generic_link_label_takes_the_report_name_from_its_context():
    html = ("<html><body><div class='card'><h3>Annual and Sustainability Report 2025</h3>"
            "<a href='https://filings.example/afw/files/press/acme/202603050673-1.pdf'>View report</a></div>"
            "<div><h3>Careers</h3><a href='/jobs'>View</a></div></body></html>")
    content = extract_html(html, SITE + "/investors")
    label = dict(content.links)["https://filings.example/afw/files/press/acme/202603050673-1.pdf"]
    assert "Annual and Sustainability Report 2025" in label and "View report" in label
    assert discovery._find_report_links(content, "northstarmanufacturing.example", 2025)[0] == \
        "https://filings.example/afw/files/press/acme/202603050673-1.pdf"


def test_primary_report_is_preferred_over_a_product_report_on_the_same_page():
    html = ("<html><body><a href='/files/model_x_carbonfootprintreport.pdf'>Carbon footprint report Model X</a>"
            "<a href='/files/annual-and-sustainability-report-2025.pdf'>Annual and Sustainability Report 2025</a></body></html>")
    links = discovery._find_report_links(extract_html(html, SITE + "/"), "northstarmanufacturing.example", 2025)
    assert links[0].endswith("annual-and-sustainability-report-2025.pdf")


def test_landing_page_with_view_report_link_resolves_to_the_report_hosted_elsewhere(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    landing = f"{SITE}/investors/results-and-reports"
    report = "https://filings.example/afw/files/press/northstar/202603050673-1.pdf"
    web[landing] = ("text/html", _page(f"Results and reports | {NAME}", "Results and reports. " * 5,
                                       f"<div><h3>Annual Report</h3><a href='{report}'>View report</a></div>"))
    web[report] = ("application/pdf", _pdf("Freedom to", "Annual and Sustainability Report 2025\nNorthstar Manufacturing AB\n"
                                                         "We aim to reduce energy consumption by 30% by 2030 compared with a 2020 baseline.\n"
                                                         "Northstar Manufacturing AB publishes this report annually. " * 2))
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(landing, f"{NAME} results and reports")]))
    a, sources, *_ = _state(aid)
    src = next(s for s in sources if s.url == report)
    assert src.status == "PROCESSED" and src.source_type == "sustainability_report" and src.is_official
    assert a.coverage_metrics["primary_reports_processed"] == 1


def test_http_and_https_versions_of_one_page_are_fetched_once(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    page = f"{SITE}/sustainability"
    web[page] = ("text/html", _page(f"Sustainability | {NAME}", "We publish our sustainability report every year. " * 5))
    aid = _create(client)
    plain = page.replace("https://", "http://")
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(plain, f"{NAME} sustainability"), (page + "/", f"{NAME} sustainability")]))
    _, sources, *_ = _state(aid)
    assert len(sources) == 1


# ---------------------------------------------------------------- 7. reporting semantics
@pytest.mark.parametrize("text", [
    "X4 Electromechanical Actuator",
    "Anna Lindqvist, Chief Executive Officer",
    "We manufacture stretch sleeve machines and shrink sleeves for the beverage industry.",
    "Stretch Sleeve Machines",
    "Net sales increased by 12% to SEK 4.2 billion in 2025.",
    "Download the ebook and learn how to build software faster.",
])
def test_ordinary_company_text_is_not_reporting_evidence(text):
    assert P.reporting_statement(text) is None
    p = _passage(text, "company_website", title="Home")
    assert not [c for c in FactExtractor(None, 2025).reporting([p], [p]) if c.claim_type == "voluntary_disclosure"]
    forged = _cand("reporting", "reporting", "voluntary_disclosure", text, {"public": True}, p, method="regex")
    assert verify_candidate(forged, text, company=COMPANY).lifecycle == "rejected"


@pytest.mark.parametrize("text", [
    "We publish a sustainability report every year describing our environmental and social performance.",
    "Our ESG disclosures cover greenhouse gas emissions, energy use and workforce diversity.",
    "In 2025 our Scope 1 and 2 emissions decreased by 12% compared with 2024.",
])
def test_sustainability_disclosure_text_is_reporting_evidence(text):
    assert P.reporting_statement(text)
    p = _passage(text, "company_website", title="Sustainability")
    found = [c for c in FactExtractor(None, 2025).reporting([p], [p]) if c.claim_type == "voluntary_disclosure"]
    assert found and verify_candidate(found[0], text, company=COMPANY).lifecycle == "accepted"


def test_report_published_requires_a_report_document():
    text = "Products and services. Contact us."
    p = _passage(text, "company_website")
    forged = _cand("reporting", "reporting", "report_published", text, {"public": True, "regular": True}, p, method="regex")
    assert verify_candidate(forged, text, company=COMPANY).lifecycle == "rejected"


# ---------------------------------------------------------------- 8. reporting aggregation
REGULAR = dict(public=True, regular=True)


def test_regular_report_only_is_level_4():
    assert score_reporting([_item(1, "report_published", **REGULAR)]).recommended_level == 4


@pytest.mark.parametrize("sentence", [
    "This report has been prepared in accordance with the GRI Standards.",
    "The Sustainability Statement has been prepared in accordance with European Sustainability Reporting Standards.",
    "Our sustainability report is prepared with reference to the GRI Standards 2021.",
    "We report in accordance with the SASB standards for our industry.",
])
def test_explicit_reporting_framework_use_lifts_reporting_to_level_5(sentence):
    fm = P.detect_frameworks(sentence)
    assert fm and fm[0].explicit
    level5 = [_item(2, "framework_reference", framework=fm[0].framework, explicit=True, level5_eligible=True)]
    assert score_reporting([_item(1, "report_published", **REGULAR), *level5]).recommended_level == 5


@pytest.mark.parametrize("sentence", [
    "We calculate total energy consumption as defined by GRI Disclosure 302-1.",  # a metric definition
    "Our climate disclosures follow ESRS E1-5 on energy consumption and mix.",  # a disclosure code
    "ESRS E1-5",  # an index / table code
    "Index of ESRS Disclosure Requirements",  # an index heading
    "Last year's report was prepared in accordance with the GRI Standards.",  # history
    "We encourage our suppliers to report in accordance with GRI.",  # someone else's reporting
    "Consistent with GRI reporting, on-site generation is reported under fuel.",  # terminology
])
def test_framework_citations_and_references_are_not_framework_use(sentence):
    assert not any(m.explicit for m in P.detect_frameworks(sentence))


def _ref(i, framework="GRI", source=1, **flags):
    item = _item(i, "framework_reference", framework=framework, explicit=False, level5_eligible=True, **flags)
    item.source_id = source
    return item


def test_references_alone_do_not_lift_reporting_to_level_5():
    refs = [_ref(i, code_reference=True) for i in range(2, 10)]
    assert score_reporting([_item(1, "report_published", **REGULAR), *refs]).recommended_level == 4


def test_a_content_index_with_several_disclosure_codes_together_show_framework_adoption():
    refs = [_ref(2, content_index=True), _ref(3, code_reference=True), _ref(4, code_reference=True), _ref(5, code_reference=True)]
    assert score_reporting([_item(1, "report_published", **REGULAR), *refs]).recommended_level == 5
    scattered = [_ref(2, content_index=True, source=1), *[_ref(i, code_reference=True, source=i) for i in (3, 4, 5)]]
    assert score_reporting([_item(1, "report_published", **REGULAR), *scattered]).recommended_level == 4


def test_a_passing_framework_mention_is_not_explicit():
    assert not P.detect_frameworks("We support the work of the GRI and attended its annual conference.")[0].explicit


@pytest.mark.parametrize("sentence", [
    "2025 metrics were subject to third-party limited assurance procedures.",
    "Selected sustainability data in this report have been subject to limited assurance by an independent auditor.",
    "KPMG provided limited assurance over our greenhouse gas emissions data in accordance with ISAE 3410.",
])
def test_report_level_sustainability_assurance_is_level_6(sentence):
    am = P.detect_assurance(sentence)[0]
    assert am.covers_sustainability and (am.provider or am.standard or am.third_party)
    items = [_item(1, "report_published", **REGULAR),
             _item(2, "framework_reference", framework="GRI", explicit=True, level5_eligible=True),
             _item(3, "assurance_statement", sustainability_assurance=True, provider=am.provider, standard=am.standard,
                   third_party=am.third_party)]
    assert score_reporting(items).recommended_level == 6


@pytest.mark.parametrize("sentence", [
    "We adopted the UL 2799 certification, a rigorous third-party verification of zero-waste performance at the facility.",
    "In 2023 we completed third-party verification for our five data centers in Europe.",
    "Both carbon removal credits will undergo independent third-party verification prior to registry issuance.",
    "The product was externally verified by an accredited certification body.",
    "The auditor provided reasonable assurance on the consolidated financial statements.",
])
def test_unrelated_verification_or_certification_never_creates_level_6(sentence):
    mentions = P.detect_assurance(sentence)
    assert all(not m.covers_sustainability for m in mentions)


def test_scorer_uses_all_accepted_reporting_evidence_not_only_the_first_items():
    noise = [_item(i, "framework_reference", framework="CDP", explicit=False, level5_eligible=False) for i in range(2, 30)]
    items = [_item(1, "report_published", **REGULAR), *noise,
             _item(40, "framework_reference", framework="GRI", explicit=True, level5_eligible=True),
             _item(41, "assurance_statement", sustainability_assurance=True, provider=None, standard=None, third_party=True)]
    r = score_reporting(items)
    assert r.recommended_level == 6 and {40, 41} <= set(r.evidence_ids)


def test_extractor_does_not_spend_model_calls_on_assurance_the_keyword_check_already_refuses():
    llm = FakeLLM()
    text = "In 2023 we completed third-party verification for our five data centers in Europe. " * 3
    p = _passage(text)
    FactExtractor(llm, 2025).reporting([p], [p])
    assert llm.calls == []


# ---------------------------------------------------------------- 9. factor / actor / purpose
@pytest.mark.parametrize("sentence", [
    "By bringing this long-duration storage technology to market faster, we aim to share its potential with communities "
    "everywhere, making reliable electricity available and supporting the resilience of grids.",
    "We aim to help our customers reduce their energy consumption by 20% by 2030.",
    "Our policy recommendations aim to accelerate renewable energy deployment across the European energy system.",
])
def test_external_benefit_goals_are_not_the_companys_own_planning(sentence):
    assert gates.external_purpose_only(sentence)
    c = _cand("energy_management", "planning", "goal", sentence, {"goal_found": True}, _passage(sentence))
    assert "external_purpose" in verify_candidate(c, sentence, company=COMPANY).rejection_codes


@pytest.mark.parametrize("sentence", [
    "We aim to reduce energy consumption in our own operations by 20% by 2030.",
    "Our goal is to source 100% renewable electricity for our data centers by 2030.",
    "We aim to help our customers reduce emissions while cutting the energy use of our own facilities by 15% by 2028.",
])
def test_own_operational_goals_remain_planning(sentence):
    assert not gates.external_purpose_only(sentence)


# ---------------------------------------------------------------- 10. execution numeric grounding
def test_valid_non_numeric_activity_is_accepted_even_if_the_model_invented_a_year():
    sentence = "We launched two packaging recovery pilots."
    text = f"Circular packaging. {sentence} The pilots test reuse of transport packaging."
    llm = FakeLLM({"DIMENSION:\nExecution": '{"activity_found": true, "activity_description": "launch two packaging recovery pilots", '
                                             '"activity_year": 2026, "linked_to_goal": false, "evidence_quote": "' + sentence + '", "confidence": 0.8}'})
    p = _passage(text, doc_year=2025)
    [c] = FactExtractor(llm, 2025).execution("product_lifecycle", [p])
    assert c.facts["activity_year"] is None and c.facts["effective_year"] == 2025
    vr = verify_candidate(c, text, company=COMPANY)
    assert vr.lifecycle == "accepted" and "numeric_not_in_source" not in vr.rejection_codes


def test_execution_numeric_claim_still_needs_grounding():
    text = "We installed solar panels at our plant in 2024."
    c = _cand("energy_management", "execution", "activity", text, {"activity_found": True, "activity_year": 2019,
                                                                  "effective_year": 2019, "in_window": False}, _passage(text))
    assert "numeric_not_in_source" in verify_candidate(c, text, company=COMPANY).rejection_codes


def test_planning_target_still_needs_numeric_grounding():
    text = "We aim to reduce energy consumption by 30% by 2030."
    c = _cand("energy_management", "planning", "goal", text, {"goal_found": True, "target_value": 45.0, "target_year": 2030}, _passage(text))
    assert "numeric_not_in_source" in verify_candidate(c, text, company=COMPANY).rejection_codes


# ---------------------------------------------------------------- 11. finding / quote alignment
ISO = "In 2025, we expanded our ISO 50001 certification scope to include eight data centers in Europe and Chile."
EU = "In 2025, five data centers in Europe were participants in the EU Code of Conduct on Data Centre Energy Efficiency."


def _execution_llm(description: str, quote: str) -> FakeLLM:
    return FakeLLM({"DIMENSION:\nExecution": '{"activity_found": true, "activity_description": "' + description + '", '
                                             '"activity_year": 2025, "linked_to_goal": false, "evidence_quote": "' + quote + '", "confidence": 0.8}'})


def test_mismatched_finding_and_quote_is_recovered_from_the_verbatim_sentence():
    text = f"Energy management. {EU} {ISO}"
    [c] = FactExtractor(_execution_llm(ISO, EU), 2025).execution("energy_management", [_passage(text)])
    assert c.evidence_text == ISO
    assert verify_candidate(c, text, company=COMPANY).lifecycle == "accepted"


def test_mismatched_finding_and_quote_is_rejected_when_the_supporting_sentence_is_absent():
    text = f"Energy management. {EU} We installed LED lighting in 2025."
    desc = "Expanded ISO 50001 certification scope to eight data centers in Chile"
    [c] = FactExtractor(_execution_llm(desc, EU), 2025).execution("energy_management", [_passage(text)])
    vr = verify_candidate(c, text, company=COMPANY)
    assert vr.lifecycle == "rejected" and "claim_quote_mismatch" in vr.rejection_codes


def test_claim_support_check_is_about_the_facts_not_shared_boilerplate():
    assert gates.claim_supported_by_quote("installed LED lighting at all plants", "We installed LED lighting at all plants in 2025.")
    assert not gates.claim_supported_by_quote(ISO, EU)


# ---------------------------------------------------------------- 12. table alignment
TABLE = """Energy (MWh) 2021 2022 2023 2024 2025
Renewable electricity procured (PPAs)
MWh 14,313,200 16,565,400 19,161,300 24,111,400 32,986,300
Fuel consumption
MWh 210,500 380,300 309,300 288,700 318,300
Total electricity consumption
MWh 18,002,100 21,481,000 24,931,500 31,641,700 43,476,000
"""


def test_table_rows_keep_their_metric_labels():
    labels = [s.metric_label for s in P.extract_series(TABLE)]
    assert any("Renewable electricity procured" in lab for lab in labels)
    assert any("Total electricity consumption" in lab for lab in labels)
    assert not any(lab.strip().lower() == "mwh" for lab in labels)


def test_only_clearly_labelled_rows_score_and_units_alone_never_identify_a_metric():
    cands = FactExtractor(None, 2025).performance("energy_management", [_passage(TABLE)])
    labels = {c.facts.get("metric_label") for c in cands if c.claim_type == "performance_series"}
    assert any("Total electricity consumption" in (lab or "") for lab in labels)
    assert not any("Fuel consumption" in (lab or "") for lab in labels)  # no configured metric: abstain
    share = [c for c in cands if c.facts.get("metric") == "renewable energy share"]
    assert not share  # an absolute MWh row is not a share
    assert all("unit n/a" not in c.claim_text for c in cands)


def test_unlabelled_unit_only_row_is_not_scored():
    text = "2023 2024 2025\nMWh 300 280 250\n"
    cands = FactExtractor(None, 2025).performance("energy_management", [_passage(text)])
    assert not [c for c in cands if c.claim_type == "performance_series" and c.facts.get("points")]


# ---------------------------------------------------------------- 3/13. factor-level coverage
def test_factor_coverage_needs_a_primary_report_or_enough_factor_sources():
    weak = CoverageCheck(valid_sources=3, relevant_official_sources=3, primary_reports=0,
                         factor_official_sources={"data_security": 0, "energy_management": 3})
    assert coverage_state([], weak, "data_security") == INSUFFICIENT_SOURCE_COVERAGE
    assert coverage_state([], weak, "energy_management") == NO_EVIDENCE_FOUND
    strong = CoverageCheck(valid_sources=1, primary_reports=1, factor_official_sources={"data_security": 0})
    assert coverage_state([], strong, "data_security") == NO_EVIDENCE_FOUND
    assert coverage_state([7], weak, "data_security") == "EVIDENCE_FOUND"


def test_generic_pages_without_a_report_leave_factors_insufficient(client, web, monkeypatch):
    """Case B / D: a few generic company pages, no primary report."""
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    generic = _page(f"Products | {NAME}", "Northstar Manufacturing AB builds electromechanical actuators for industry. " * 4
                    + "Our sustainability and climate work is described on our website.")
    urls = [(f"{SITE}/products/{i}", f"{NAME} products {i}") for i in range(3)]
    for url, _ in urls:
        web[url] = ("text/html", generic)
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=urls))
    a, _, recs, _ = _state(aid)
    for f in ("energy_management", "product_lifecycle", "data_security", "labor_practices", "diversity_inclusion"):
        for d in ("planning", "execution", "performance"):
            assert recs[(f, d)].coverage_state == INSUFFICIENT_SOURCE_COVERAGE and recs[(f, d)].recommended_level != 1
    assert a.coverage_metrics["factor_coverage"]["data_security"]["sufficient"] is False


def test_enough_factor_specific_official_sources_without_evidence_is_no_evidence(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    security = _page(f"Information security | {NAME}",
                     "Northstar Manufacturing AB describes its information security and data protection organisation, "
                     "cybersecurity responsibilities and privacy practices on this page. " * 3)
    urls = [(f"{SITE}/security/{i}", f"{NAME} security {i}") for i in range(3)]
    for url, _ in urls:
        web[url] = ("text/html", security)
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=urls))
    a, _, recs, _ = _state(aid)
    assert recs[("data_security", "planning")].coverage_state == NO_EVIDENCE_FOUND
    assert recs[("data_security", "planning")].recommended_level == 1
    assert recs[("diversity_inclusion", "planning")].coverage_state == INSUFFICIENT_SOURCE_COVERAGE


# ---------------------------------------------------------------- 4. targeted factor recovery
def test_targeted_recovery_runs_once_for_weak_factors_and_uses_factor_terms(client, web):
    web[f"{SITE}/sustainability"] = ("text/html", _page(f"Sustainability | {NAME}", "We aim to reduce energy consumption by 30% by 2030. " * 3))
    search = Search(first=[(f"{SITE}/sustainability", f"{NAME} sustainability")], later=[])
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=search)
    a, *_ = _state(aid)
    recovery = search.queries[8:]
    assert a.coverage_metrics["recovery_search_runs"] == 1
    assert len(recovery) <= get_settings().recovery_max_queries
    assert any("cybersecurity" in q or "information security" in q for q in recovery)
    assert any("diversity" in q for q in recovery)
    assert all("Northstar Manufacturing" in q or "site:" in q for q in recovery)


def test_targeted_recovery_results_pass_the_same_identity_gate(client, web):
    other = "https://www.northstar-security.example/information-security"
    web[f"{SITE}/sustainability"] = ("text/html", _page(f"Sustainability | {NAME}", "We aim to reduce energy consumption by 30% by 2030. " * 3))
    web[other] = ("text/html", _page("Northstar Security AS | Information security",
                                     "Northstar Security AS provides information security services. Northstar Security AS aims "
                                     "to reach ISO 27001 certification for all clients by 2027. " * 3))
    search = Search(first=[(f"{SITE}/sustainability", f"{NAME} sustainability")], later=[(other, "Northstar Security information security")])
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=search)
    a, sources, recs, evidence = _state(aid)
    assert next(s for s in sources if s.url == other).rejection_reason == "source_company_mismatch"
    assert not [e for e in evidence if e.source_url == other and e.evidence_lifecycle == "accepted"]
    assert recs[("data_security", "planning")].evidence_ids == []
    assert (a.company_identity or {}).get("official_domain") == "northstarmanufacturing.example"


# ---------------------------------------------------------------- 16. identity
def test_alternate_domain_with_the_same_name_is_not_first_party_when_another_domain_is_official():
    identity = CompanyIdentity("Helvar Packaging AB", official_domain="helvarpackaging.example")
    text = "Helvar Packaging produces shrink sleeves. Helvar Packaging India."
    alt = assess_source_identity(identity, title="Helvar Packaging", url="https://helvarpackaging.example.in/about", text=text)
    assert alt.basis not in ("official_domain", "name_domain")
    own = assess_source_identity(identity, title="Helvar Packaging", url="https://www.helvarpackaging.example/about", text=text)
    assert own.basis == "name_domain"


def test_brand_top_level_domain_stays_first_party():
    identity = CompanyIdentity("Gizmo", official_domain="gizmo.com")
    r = assess_source_identity(identity, title="Gizmo 2026 Environmental Report", url="https://sustainability.gizmo/report.pdf",
                               text="Gizmo 2026 Environmental Report. Gizmo publishes this report annually.")
    assert r.matched and r.basis == "name_domain"


def test_wrong_company_pdf_is_rejected_from_its_first_pages_without_full_extraction(client, web, monkeypatch):
    calls = []
    real = discovery.ingest_pdf
    monkeypatch.setattr(discovery, "ingest_pdf", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    url = "https://filings.example/northstar-electronics-annual-report-2025.pdf"
    web[url] = ("application/pdf", _report_pdf(company="Northstar Electronics Inc."))
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(url, "Northstar Electronics annual report")]))
    _, sources, *_ = _state(aid)
    assert sources[0].rejection_reason == "source_company_mismatch" and calls == []


# ---------------------------------------------------------------- 14. bounded work
def test_duplicate_passages_from_mirrored_pages_are_extracted_once(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    body = _page(f"Energy | {NAME}", "We aim to reduce energy consumption by 30% by 2030 compared with a 2020 baseline. " * 3)
    urls = [(f"{SITE}/en/energy", f"{NAME} energy"), (f"{SITE}/intl/energy", f"{NAME} energy")]
    for url, _ in urls:
        web[url] = ("text/html", body)
    llm = FakeLLM()
    aid = _create(client)
    run_assessment(aid, llm=llm, search=Search(first=urls))
    planning_calls = [c for c in llm.calls if "DIMENSION:\nPlanning" in c and "Energy Management" in c]
    assert len(planning_calls) == 1


def test_performance_model_call_is_skipped_when_no_value_could_be_tied_to_a_year():
    llm = FakeLLM()
    text = "Our energy work began in 2012 and continued in 2020. Energy efficiency remains important across the company. " * 3
    FactExtractor(llm, 2025).performance("energy_management", [_passage(text)])
    assert not [c for c in llm.calls if "DIMENSION:\nPerformance" in c]


def test_stage_timings_and_model_calls_are_recorded(client, web):
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search())
    a, *_ = _state(aid)
    stages = a.coverage_metrics["stage_seconds"]
    assert {"discovery", "retrieval", "extraction", "scoring"} <= set(stages)
    assert "model_calls" in a.coverage_metrics and "model_seconds" in a.coverage_metrics
