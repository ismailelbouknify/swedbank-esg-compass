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


from app.services.pipeline import discovery


from app.services.pipeline.coverage import INSUFFICIENT_SOURCE_COVERAGE, NO_EVIDENCE_FOUND, CoverageCheck, coverage_state


from app.services.pipeline.runner import run_assessment


from tests.conftest import FakeLLM, prompt_passages


from tests.test_source_coverage import NAME, SITE, Search, _create, _page, _report_pdf, _state, web  # noqa: F401


COMPANY = CompanyIdentity("Northstar Manufacturing AB")


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


REGULAR = dict(public=True, regular=True)


ISO = "In 2025, we expanded our ISO 50001 certification scope to include eight data centers in Europe and Chile."


EU = "In 2025, five data centers in Europe were participants in the EU Code of Conduct on Data Centre Energy Efficiency."


def _execution_llm(description: str, quote: str) -> FakeLLM:
    return FakeLLM({"DIMENSION:\nExecution": '{"activity_found": true, "activity_description": "' + description + '", '
                                             '"activity_year": 2025, "linked_to_goal": false, "evidence_quote": "' + quote + '", "confidence": 0.8}'})


TABLE = """Energy (MWh) 2021 2022 2023 2024 2025
Renewable electricity procured (PPAs)
MWh 14,313,200 16,565,400 19,161,300 24,111,400 32,986,300
Fuel consumption
MWh 210,500 380,300 309,300 288,700 318,300
Total electricity consumption
MWh 18,002,100 21,481,000 24,931,500 31,641,700 43,476,000
"""


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
    assert recs[("data_security", "planning")].recommended_level == 0  # not assessed: never an automatic Level 1
    assert recs[("diversity_inclusion", "planning")].coverage_state == INSUFFICIENT_SOURCE_COVERAGE


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


def test_duplicate_passages_from_mirrored_pages_are_extracted_once(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    body = _page(f"Energy | {NAME}", "We aim to reduce energy consumption by 30% by 2030 compared with a 2020 baseline. " * 3)
    urls = [(f"{SITE}/en/energy", f"{NAME} energy"), (f"{SITE}/intl/energy", f"{NAME} energy")]
    for url, _ in urls:
        web[url] = ("text/html", body)
    llm = FakeLLM()
    aid = _create(client)
    run_assessment(aid, llm=llm, search=Search(first=urls))
    planning_calls = [c for c in llm.calls if "DIMENSION: Planning" in c and "ESG FACTOR: Energy Management" in c]
    assert len(planning_calls) == 1  # one call per question
    # the mirrored page's passage is given to the model once, not twice
    blocks = [text for _, text in prompt_passages(planning_calls[0]) if "reduce energy consumption by 30%" in text]
    assert len(blocks) == 1


def test_stage_timings_and_model_calls_are_recorded(client, web):
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search())
    a, *_ = _state(aid)
    stages = a.coverage_metrics["stage_seconds"]
    assert {"discovery", "retrieval", "answering", "persisting"} <= set(stages)
    assert "model_calls" in a.coverage_metrics and "model_seconds" in a.coverage_metrics

