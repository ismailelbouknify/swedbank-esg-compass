"""Source coverage states, one bounded recovery search, entity resolution and primary-document resolution.
No network: DNS and HTTP are mocked. All companies are fictional."""
from __future__ import annotations

import socket

import httpx
import pymupdf
import pytest
from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models.entities import Assessment, Evidence, Recommendation, Source
from app.services.company.resolution import AMBIGUOUS, identity_from_profile, resolve_company
from app.services.company.source_identity import CompanyIdentity, assess_source_identity
from app.services.documents.html_extractor import extract_html
from app.services.pipeline import discovery
from app.services.pipeline.coverage import INSUFFICIENT_COVERAGE_MESSAGE, INSUFFICIENT_SOURCE_COVERAGE, NO_EVIDENCE_FOUND, NOT_ASSESSED_LEVEL
from app.services.pipeline.runner import run_assessment
from app.services.search.base import NullSearchProvider, SearchProvider, SearchResult, build_recovery_queries
from app.services.security import safe_fetch
from tests.conftest import FakeLLM

NAME = "Northstar Manufacturing AB"
SITE = "https://www.northstarmanufacturing.example"


def _page(title: str, sentences: str, links: str = "") -> str:
    return f"<html><head><title>{title}</title></head><body><p>{sentences}</p>{links}</body></html>"


ABOUT = "Northstar Manufacturing reports on its sustainability and climate work every year, and Northstar Manufacturing publishes this openly. " * 4
ENERGY_PAGE = _page("Energy | Northstar Manufacturing", ABOUT + "We aim to reduce energy consumption by 30% by 2030 compared with a 2020 baseline.")


def _report_pdf(company: str = "Northstar Manufacturing AB", body: str | None = None) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    text = body or (f"{company}\nSustainability Report 2025\n"
                    "We aim to reduce energy consumption by 30% by 2030 compared with a 2020 baseline.\n"
                    "We installed LED lighting at all plants in 2025 as part of our energy strategy.\n"
                    "Our goal is to reach 40% women in management positions by 2030.\n"
                    f"{company} publishes this report annually.\n" + "We publish our results each year in this report. " * 3)
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    doc.set_metadata({"title": f"{company} Sustainability Report 2025"})
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture()
def web(monkeypatch):
    """url -> response. Anything else is a 404."""
    pages: dict[str, tuple[str, bytes | str]] = {}
    requested: list[str] = []

    def handler(request):
        path = request.url.raw_path.decode()
        url = f"https://{request.headers['host']}{path}"
        requested.append(url)
        if url not in pages:
            return httpx.Response(404)
        ctype, body = pages[url]
        return httpx.Response(200, headers={"content-type": ctype}, content=body)

    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))])
    monkeypatch.setattr(safe_fetch, "build_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr(discovery.time, "sleep", lambda _s: None)
    pages["__requested__"] = requested  # type: ignore[assignment]
    return pages


class Search(SearchProvider):
    """First N calls give the first answer list; everything after gives the second."""
    name = "test"

    def __init__(self, first=(), later=(), first_calls: int = 8):
        self.first, self.later, self.first_calls = list(first), list(later), first_calls
        self.queries: list[str] = []

    def search(self, query, max_results):
        self.queries.append(query)
        urls = self.first if len(self.queries) <= self.first_calls else self.later
        return [SearchResult(title, url, "sustainability report", query) for url, title in urls]


def _create(client, name=NAME, **kw) -> int:
    body = {"company_name": name, "year": 2025, "enable_web_search": True, "confirm_new": True, **kw}
    r = client.post("/api/assessments", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _state(aid):
    with SessionLocal() as db:
        a = db.get(Assessment, aid)
        return (a, list(db.scalars(select(Source).where(Source.assessment_id == aid))),
                {(r.factor, r.dimension): r for r in db.scalars(select(Recommendation).where(Recommendation.assessment_id == aid))},
                list(db.scalars(select(Evidence).where(Evidence.assessment_id == aid))))


REPORT_URL = f"{SITE}/sustainability-report-2025.pdf"


# ---------------------------------------------------------------- recovery
def test_zero_search_results_trigger_one_recovery_search_that_finds_valid_sources(client, web):
    web[REPORT_URL] = ("application/pdf", _report_pdf())
    search = Search(first=[], later=[(REPORT_URL, f"{NAME} sustainability report")])
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=search)
    a, sources, recs, evidence = _state(aid)
    assert len(search.queries) > 8
    assert [s.status for s in sources] == ["PROCESSED"] and sources[0].source_type == "sustainability_report"
    assert a.coverage_metrics["recovery_search_runs"] == 1 and "no_valid_company_sources" in a.coverage_metrics["recovery_reasons"]
    assert recs[("energy_management", "planning")].recommended_level >= 4
    assert recs[("reporting", "reporting")].recommended_level >= 4
    assert a.coverage_metrics["primary_reports_processed"] == 1 and a.coverage_metrics["factors_with_evidence"] >= 1


def test_recovery_search_runs_at_most_once_and_stays_within_its_budget(client, web):
    search = Search(first=[], later=[])
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=search)
    a, *_ = _state(aid)
    settings = get_settings()
    assert a.coverage_metrics["recovery_search_runs"] == 1
    assert len(search.queries) <= 8 + settings.recovery_max_queries


def test_recovery_source_and_time_budgets_are_bounded(client, web, configure_recovery):
    urls = [(f"{SITE}/page-{i}", f"{NAME} page {i}") for i in range(30)]
    for url, _ in urls:
        web[url] = ("text/html", ENERGY_PAGE)
    search = Search(first=[], later=urls)
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=search)
    _, sources, *_ = _state(aid)
    assert len(sources) <= configure_recovery


def test_time_budget_stops_the_recovery_search(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_time_budget_seconds", 0)
    search = Search(first=[], later=[])
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=search)
    assert len(search.queries) == 8  # the initial queries only


@pytest.fixture()
def configure_recovery(monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_max_pages", 2)
    monkeypatch.setattr(get_settings(), "recovery_max_pdfs", 1)
    return 3


def test_normal_assessment_with_good_sources_does_not_pay_for_recovery(client, web):
    web[REPORT_URL] = ("application/pdf", _report_pdf())
    for i in range(3):
        web[f"{SITE}/sustainability/{i}"] = ("text/html", ENERGY_PAGE)
    results = [(REPORT_URL, f"{NAME} sustainability report")] + [(f"{SITE}/sustainability/{i}", f"{NAME} sustainability {i}") for i in range(3)]
    search = Search(first=results, later=[])
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=search)
    a, *_ = _state(aid)
    assert a.coverage_metrics["recovery_search_runs"] == 0 and len(search.queries) == 8


def test_no_recovery_when_search_is_not_allowed(client, web, sample_pdf):
    search = Search()
    aid = _create(client, name="Nordvik Components AB", enable_web_search=False)
    with open(sample_pdf, "rb") as f:
        client.post(f"/api/assessments/{aid}/documents", files={"file": ("r.pdf", f, "application/pdf")})
    run_assessment(aid, llm=FakeLLM(), search=search)
    assert search.queries == []


def test_recovery_queries_put_the_official_domain_first_and_skip_repeats():
    queries = build_recovery_queries("Acme", [], "acme.example", 2025, ["Acme sustainability report 2025"], 10)
    assert queries[0] == "site:acme.example sustainability report"
    assert "Acme sustainability report 2025" not in queries and len(queries) == 10
    assert any("climate target" in q for q in build_recovery_queries("Acme", [], None, 2025, [], 10))


def test_official_domain_source_is_taken_before_a_secondary_source(client, web):
    secondary = "https://www.newsportal.example/northstar-manufacturing-sustainability"
    web[REPORT_URL] = ("application/pdf", _report_pdf())
    web[secondary] = ("text/html", ENERGY_PAGE)
    web[f"{SITE}/"] = ("text/html", _page(f"{NAME}", ABOUT))
    search = Search(first=[(secondary, f"{NAME} sustainability news"), (REPORT_URL, f"{NAME} sustainability report 2025"), (f"{SITE}/", NAME)])
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=search)
    requested = [u for u in web["__requested__"] if u in (REPORT_URL, secondary)]
    assert requested[0] == REPORT_URL
    _, sources, *_ = _state(aid)
    by_url = {s.url: s for s in sources}
    assert by_url[REPORT_URL].priority < by_url[secondary].priority


# ---------------------------------------------------------------- coverage states
def test_zero_sources_is_insufficient_coverage_not_no_evidence(client, web):
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search())
    a, _, recs, _ = _state(aid)
    assert len(recs) == 16
    for r in recs.values():
        assert r.coverage_state == INSUFFICIENT_SOURCE_COVERAGE == r.evidence_status
        assert r.recommended_level == NOT_ASSESSED_LEVEL and r.recommended_level != 1 and r.evidence_ids == []
    st = client.get(f"/api/assessments/{aid}/status").json()
    assert INSUFFICIENT_COVERAGE_MESSAGE in st["notices"] and "Upload the company's sustainability or annual report" in INSUFFICIENT_COVERAGE_MESSAGE
    rec = client.get(f"/api/assessments/{aid}/recommendations").json()[0]
    assert rec["coverage_state"] == INSUFFICIENT_SOURCE_COVERAGE and rec["reason"].startswith("Insufficient source coverage")
    assert any("Upload" in m for m in rec["next_level_missing"])


def test_gap_counts_and_export_never_read_not_assessed_as_a_level(client, web):
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search())
    data = client.get(f"/api/assessments/{aid}/export", params={"format": "json"}).json()
    assert all(r["ai_recommended_level"] is None and r["final_level"] is None for r in data["recommendations"])
    assert all(r["coverage_state"] == INSUFFICIENT_SOURCE_COVERAGE and r["review_status"] == "Insufficient source coverage" for r in data["recommendations"])
    assert len(data["data_gaps"]) == 16
    assert client.get(f"/api/assessments/{aid}").json()["data_gaps"] == 16
    for fmt in ("csv", "xlsx"):
        assert client.get(f"/api/assessments/{aid}/export", params={"format": fmt}).status_code == 200


def test_not_assessed_answer_cannot_be_approved_but_an_analyst_can_set_a_level(client, web):
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search())
    rec = client.get(f"/api/assessments/{aid}/recommendations").json()[0]
    assert client.post(f"/api/recommendations/{rec['id']}/review", json={"action": "APPROVE"}).status_code == 422
    r = client.post(f"/api/recommendations/{rec['id']}/review", json={"action": "OVERRIDE", "selected_level": 2, "comment": "From the printed report."})
    assert r.status_code == 200 and r.json()["final_level"] == 2


def test_sufficient_sources_without_evidence_is_no_evidence(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    neutral = _page("About | Northstar Manufacturing", ABOUT)
    urls = [(f"{SITE}/about/{i}", f"{NAME} about {i}") for i in range(3)]
    for url, _ in urls:
        web[url] = ("text/html", neutral)
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=urls))
    a, sources, recs, _ = _state(aid)
    assert sum(s.status == "PROCESSED" for s in sources) == 3
    gap = recs[("data_security", "planning")]
    assert gap.coverage_state == NO_EVIDENCE_FOUND == gap.evidence_status
    assert gap.recommended_level == 1 and gap.reason.startswith("No evidence of")


def test_one_real_report_is_enough_to_say_no_evidence_for_a_missing_topic(client, sample_pdf):
    aid = _create(client, name="Nordvik Components AB", enable_web_search=False)
    with open(sample_pdf, "rb") as f:
        client.post(f"/api/assessments/{aid}/documents", files={"file": ("r.pdf", f, "application/pdf")})
    run_assessment(aid, llm=FakeLLM(), search=NullSearchProvider())
    a, _, recs, _ = _state(aid)
    assert recs[("data_security", "performance")].coverage_state == NO_EVIDENCE_FOUND
    assert recs[("energy_management", "planning")].coverage_state == "EVIDENCE_FOUND"
    assert INSUFFICIENT_COVERAGE_MESSAGE not in (a.warnings or [])


def test_one_web_page_with_evidence_keeps_that_answer_and_marks_the_rest_as_gaps(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    url = f"{SITE}/energy"
    web[url] = ("text/html", ENERGY_PAGE)
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(url, f"{NAME} energy")]))
    _, _, recs, _ = _state(aid)
    assert recs[("energy_management", "planning")].recommended_level >= 4
    assert recs[("energy_management", "planning")].coverage_state == "EVIDENCE_FOUND"
    assert recs[("data_security", "planning")].coverage_state == INSUFFICIENT_SOURCE_COVERAGE


def test_wrong_company_sources_do_not_count_toward_coverage_or_evidence(client, web):
    wrong_pdf = _report_pdf("Bluefin Foods Ltd")
    urls = []
    for i in range(4):
        url = f"https://www.bluefin{i}.example/report-{i}.pdf"
        web[url] = ("application/pdf", wrong_pdf)
        urls.append((url, f"{NAME} sustainability report"))
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=urls, later=urls))
    a, sources, recs, evidence = _state(aid)
    assert sources and all(s.status == "REJECTED" and s.rejection_reason == "source_company_mismatch" for s in sources)
    assert evidence == [] and all(r.coverage_state == INSUFFICIENT_SOURCE_COVERAGE and r.evidence_ids == [] for r in recs.values())
    assert a.coverage_metrics["sources_validated"] == 0 and a.coverage_metrics["sources_rejected_company"] == len(sources)
    assert "most_sources_wrong_company" in a.coverage_metrics["recovery_reasons"]


def test_recovery_does_not_bypass_evidence_gates(client, web):
    """Sources found in recovery reach the same validation: a false statement on a valid page is not accepted."""
    url = f"{SITE}/news"
    web[url] = ("text/html", _page(f"News | {NAME}", ABOUT + "The government launched an energy efficiency programme in 2025."))
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[], later=[(url, f"{NAME} news")]))
    _, _, recs, evidence = _state(aid)
    assert all("government" not in e.evidence_text.lower() or e.evidence_lifecycle != "accepted" for e in evidence)
    assert recs[("energy_management", "execution")].evidence_ids == []


def test_no_extra_model_calls_are_made_for_recovery_or_resolution(client, web):
    web[REPORT_URL] = ("application/pdf", _report_pdf())
    direct, recovered = FakeLLM(), FakeLLM()
    a1 = _create(client)
    run_assessment(a1, llm=direct, search=Search(first=[(REPORT_URL, f"{NAME} sustainability report")]))
    a2 = _create(client)
    run_assessment(a2, llm=recovered, search=Search(first=[], later=[(REPORT_URL, f"{NAME} sustainability report")]))
    assert len(recovered.calls) == len(direct.calls) and direct.calls


def test_audit_export_has_recall_metrics_and_normal_endpoints_do_not(client, web):
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search())
    data = client.get(f"/api/assessments/{aid}/export", params={"format": "json"}).json()
    required = {"sources_discovered", "sources_validated", "sources_rejected_company", "official_sources_processed", "primary_reports_processed",
                "candidate_evidence_count", "accepted_evidence_count", "factors_with_evidence", "factors_with_insufficient_coverage"}
    assert required <= set(data["audit"]["coverage_metrics"])
    assert data["audit"]["coverage_metrics"]["factors_with_insufficient_coverage"] == 5
    body = client.get(f"/api/assessments/{aid}").text + client.get(f"/api/assessments/{aid}/status").text
    assert "coverage_metrics" not in body and "company_identity" not in body


# ---------------------------------------------------------------- entity resolution
def _results(*urls):
    return [SearchResult("x", u, "x", "q") for u in urls]


BRAND_RESULTS = _results("https://www.polarcars.example/a", "https://www.polarcars.example/b", "https://www.polargroup.example/a", "https://www.polargroup.example/b")


def test_ambiguous_brand_name_is_detected():
    profile = resolve_company("Polar", None, None, BRAND_RESULTS)
    assert profile["status"] == AMBIGUOUS and set(profile["excluded"]) == {"polar cars", "polar group"} and profile["official_domain"] is None


def test_supplied_website_resolves_the_brand_and_excludes_the_other_entity():
    profile = resolve_company("Polar", None, "polarcars.example", BRAND_RESULTS)
    assert profile["status"] == "resolved" and profile["aliases"] == ["polar cars"] and profile["excluded"] == ["polar group"]


def test_exact_domain_and_dominant_results_resolve_the_name():
    exact = resolve_company("Polar", None, None, BRAND_RESULTS + _results(*[f"https://polar.example/{i}" for i in range(5)]))
    assert exact["status"] == "resolved" and exact["official_domain"] == "polar.example"
    # a brand word's own domain does not settle it when several other entities outweigh it in the results
    contested = resolve_company("Polar", None, None, BRAND_RESULTS + _results("https://polar.example/"))
    assert contested["status"] == AMBIGUOUS and contested["official_domain"] is None
    # a full name stands for itself, whatever else shares its first word
    full = resolve_company("Polar Cars", None, None, BRAND_RESULTS)
    assert full["status"] == "resolved" and full["official_domain"] == "polarcars.example" and full["excluded"] == ["polar group"]
    many = _results(*[f"https://polarcars.example/{i}" for i in range(6)]) + _results("https://polargroup.example/a", "https://polargroup.example/b")
    dominant = resolve_company("Polar", None, None, many)
    assert dominant["basis"] == "dominant_in_results" and dominant["aliases"] == ["polar cars"]


def test_organisation_number_resolves_the_name():
    results = BRAND_RESULTS + [SearchResult("Polar Group", "https://polargroup.example/ir", "Org. nr 556677-8899", "q")]
    profile = resolve_company("Polar", "5566778899", None, results)
    assert profile["basis"] == "organisation_number" and profile["official_domain"] == "polargroup.example"


def test_division_on_the_companys_own_domain_is_not_a_competing_entity():
    profile = resolve_company("Polar", None, None, _results("https://cloud.polar.example/a", "https://polar.example/b", "https://blog.polar.example/c"))
    assert profile["status"] == "resolved" and profile["excluded"] == []


def test_fan_sites_do_not_make_a_name_ambiguous():
    results = _results("https://polarforum.example/a", "https://polarforum.example/b", "https://polarclub.example/a", "https://polarclub.example/b")
    assert resolve_company("Polar", None, None, results)["status"] != AMBIGUOUS


def _polar(**profile):
    return identity_from_profile("Polar", None, {"status": "resolved", **profile})


def test_other_entitys_documents_are_not_mixed_in():
    identity = _polar(aliases=["polar cars"], excluded=["polar group"])
    other = "Polar Group makes trucks. Polar Group grew. Polar Group invests. Polar makes trucks."
    assert not assess_source_identity(identity, title="Sustainability", url="https://reports.example/x", text=other).matched
    assert not assess_source_identity(identity, title="Report", url="https://www.polargroup.example/x", text="Polar makes trucks. Polar grew.").matched
    own = "Polar Cars makes cars. Polar Cars grew. Polar invests."
    assert assess_source_identity(identity, title="Sustainability", url="https://reports.example/y", text=own).matched


def test_ambiguous_assessment_does_not_mix_entities_and_asks_for_identification(client, web):
    urls = []
    for host, label in (("polarcars", "Polar Cars"), ("polargroup", "Polar Group")):
        for i in range(2):
            url = f"https://www.{host}.example/sustainability-{i}"
            web[url] = ("text/html", _page(f"{label} sustainability", f"{label} reports here. {label} aims high. We aim to reduce energy consumption by 30% by 2030. " * 8))
            urls.append((url, f"{label} sustainability"))
    aid = _create(client, name="Polar")
    run_assessment(aid, llm=FakeLLM(), search=Search(first=urls, later=urls))
    a, sources, recs, evidence = _state(aid)
    assert a.company_identity["status"] == AMBIGUOUS
    assert evidence == [] and not [s for s in sources if s.status == "PROCESSED"]
    assert all(r.coverage_state == INSUFFICIENT_SOURCE_COVERAGE for r in recs.values())
    assert any("more than one company" in n for n in client.get(f"/api/assessments/{aid}/status").json()["notices"])


def test_identity_stays_the_same_after_recovery_search(client, web):
    """Once settled, a recovery search that surfaces another entity cannot change which company is assessed."""
    cars = "https://www.polarcars.example/sustainability"
    web[cars] = ("text/html", _page("Polar Cars sustainability", "Polar Cars reports here. Polar Cars aims high. " * 10))
    group = [(f"https://www.polargroup.example/{i}", "Polar Group") for i in range(8)]
    for url, _ in group:
        web[url] = ("text/html", _page("Polar Group", "Polar Group reports here. " * 12))
    aid = _create(client, name="Polar")
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(cars, "Polar Cars"), (cars + "#2", "Polar Cars")], later=group))
    a, sources, *_ = _state(aid)
    assert a.company_identity["aliases"] == ["polar cars"]
    assert a.coverage_metrics["recovery_search_runs"] == 1
    assert not [s for s in sources if "polargroup" in (s.url or "") and s.status == "PROCESSED"]


# ---------------------------------------------------------------- parent / subsidiary
def test_parent_report_supports_a_subsidiary_only_inside_its_stated_boundary():
    identity = CompanyIdentity("Polar Cars AB")
    filler = "Polar Holdings AB reports here. " * 10
    inside = filler + "The reporting boundary includes our subsidiaries, among them Polar Cars AB."
    result = assess_source_identity(identity, title="Polar Holdings Sustainability Report 2025", url=None, text=inside)
    assert result.matched and result.basis == "parent_boundary"
    outside = filler + "Polar Cars AB is a customer of ours and buys parts."
    assert not assess_source_identity(identity, title="Polar Holdings Sustainability Report 2025", url=None, text=outside).matched


def test_subsidiary_report_does_not_support_the_parent():
    identity = CompanyIdentity("Polar")
    text = "Polar Cars AB reports here. Polar Cars grew. Polar Cars invests. Polar Cars hires. The owner is Polar."
    assert not assess_source_identity(identity, title="Polar Cars Sustainability Report 2025", url=None, text=text).matched


def test_evidence_from_a_parent_report_must_name_the_subsidiary():
    from app.services.extraction.extractor import Candidate, Passage
    from app.services.verification.verifier import verify_candidate

    identity = CompanyIdentity("Polar Cars AB")

    def goal(text):
        passage = Passage(1, 1, 1, "Group report", None, "uploaded_report", 1.0, True, 1, text, 2025, identity_basis="parent_boundary")
        c = Candidate("energy_management", "planning", "goal", "g", {"goal_found": True, "formalized": True}, text, passage, "llm")
        return verify_candidate(c, text, company=identity)

    assert "company_mismatch" in goal("We aim to reduce energy use by 2030.").rejection_codes
    assert goal("Polar Cars AB will aim to reduce energy use by 2030.").lifecycle == "accepted"


# ---------------------------------------------------------------- landing page -> primary document
def _landing(*links):
    anchors = "".join(f'<a href="{u}">{t}</a>' for u, t in links)
    return _page(f"Annual and sustainability reports | {NAME}", ABOUT, anchors)


def test_latest_report_is_preferred_and_cdn_hosted_pdfs_are_allowed():
    content = extract_html(_landing(("/reports/sustainability-report-2019.pdf", "Sustainability report 2019"),
                                    ("https://cdn.filings.example/nsm/sustainability-report-2025.pdf", "Sustainability report 2025"),
                                    ("https://other.example/annual-report-page", "Annual report")), SITE)
    links = discovery._find_report_links(content, "northstarmanufacturing.example", 2025)
    assert links[0] == "https://cdn.filings.example/nsm/sustainability-report-2025.pdf"
    assert "https://other.example/annual-report-page" not in links  # a page off the official domain is not followed


def test_landing_page_resolves_to_the_primary_report_and_improves_coverage(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    landing = f"{SITE}/reports/annual-and-sustainability-reports"
    web[landing] = ("text/html", _landing((REPORT_URL, "Sustainability report 2025")))
    web[REPORT_URL] = ("application/pdf", _report_pdf())
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(landing, f"{NAME} sustainability reports")]))
    a, sources, recs, _ = _state(aid)
    report = next(s for s in sources if s.url == REPORT_URL)
    assert report.status == "PROCESSED" and report.source_type == "sustainability_report" and report.company_match
    assert a.coverage_metrics["primary_reports_processed"] == 1
    assert recs[("energy_management", "planning")].recommended_level >= 4
    assert a.coverage_metrics["factors_with_evidence"] >= 1
    assert recs[("data_security", "planning")].coverage_state == NO_EVIDENCE_FOUND  # a report was read: an honest "no evidence"


def test_landing_page_reaches_the_report_through_a_reports_page(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    first, second = f"{SITE}/investors", f"{SITE}/investors/annual-report"
    web[first] = ("text/html", _page(f"Investors | {NAME}", ABOUT, f'<a href="{second}">Annual report</a>'))
    web[second] = ("text/html", _page(f"Annual report | {NAME}", ABOUT, f'<a href="{REPORT_URL}">Sustainability report 2025</a>'))
    web[REPORT_URL] = ("application/pdf", _report_pdf())
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(first, f"{NAME} investors")]))
    _, sources, *_ = _state(aid)
    assert {s.url: s.status for s in sources}[REPORT_URL] == "PROCESSED"


def test_wrong_company_report_behind_a_landing_page_stays_blocked(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    landing = f"{SITE}/reports"
    pdf = "https://cdn.filings.example/sustainability-report-2025.pdf"
    web[landing] = ("text/html", _landing((pdf, "Sustainability report 2025")))
    web[pdf] = ("application/pdf", _report_pdf("Bluefin Foods Ltd"))
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(landing, f"{NAME} reports")]))
    _, sources, _, evidence = _state(aid)
    blocked = next(s for s in sources if s.url == pdf)
    assert blocked.status == "REJECTED" and blocked.rejection_reason == "source_company_mismatch"
    assert not [e for e in evidence if e.source_id == blocked.id]


# ---------------------------------------------------------------- first-party recall + reporting
def test_first_party_source_inherits_identity_without_repeating_the_name(client, web, monkeypatch):
    """The address carries the exact company name, so "we" in a passage means the company."""
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    url = "https://blog.northstarmanufacturing.example/energy"
    text = "Northstar Manufacturing writes here. " + "Our operations run across five sites. " * 8 + "We aim to reduce energy consumption by 30% by 2030."
    web[url] = ("text/html", _page("Energy update", text))
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(url, f"{NAME} energy")]))
    _, sources, recs, _ = _state(aid)
    assert sources[0].company_match == "name_domain" and sources[0].is_official and sources[0].source_type != "other_web"
    assert recs[("energy_management", "planning")].recommended_level >= 4


def test_third_party_author_does_not_inherit_identity(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    url = "https://www.newsportal.example/article"
    text = "Northstar Manufacturing was covered here. Northstar Manufacturing is large. " + "Our readers enjoy this. " * 80 + "We aim to reduce energy consumption by 30% by 2030."
    web[url] = ("text/html", _page("Northstar Manufacturing article", text))
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(url, f"{NAME} news")]))
    _, sources, recs, _ = _state(aid)
    assert sources[0].status == "PROCESSED" and sources[0].source_type == "other_web"
    assert recs[("energy_management", "planning")].evidence_ids == []


def test_processed_environmental_report_counts_as_regular_public_reporting(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    url = "https://files.host.example/doc-1.pdf"
    body = f"{NAME}\nEnvironmental Report 2025\nWe report our environmental results every year in this report.\n{NAME} publishes this annually.\n" + "We publish these results each year. " * 3
    web[url] = ("application/pdf", _report_pdf(body=body))
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(url, f"{NAME} document")]))
    _, _, recs, _ = _state(aid)
    rec = recs[("reporting", "reporting")]
    assert rec.recommended_level >= 4 and rec.triggered_rule != "R1_NO_REPORTING_EVIDENCE"


def test_required_term_keeps_a_chunk_even_when_a_small_corpus_scores_it_zero():
    from app.services.retrieval.bm25 import BM25Index

    index = BM25Index([{"id": 1, "text": "We aim to reduce energy use.", "page_number": 1}, {"id": 2, "text": "Our energy policy is public.", "page_number": 1}])
    assert {h.chunk_id for h in index.search(["energy"], top_k=5, must_contain_any=["energy"])} == {1, 2}
    assert index.search(["energy"], top_k=5, must_contain_any=["water"]) == []


def test_first_party_pages_on_other_subjects_do_not_make_no_evidence_honest(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    urls = [(f"{SITE}/products/{i}", f"{NAME} product {i}") for i in range(4)]
    spec = _page(f"Product | {NAME}", "Northstar Manufacturing builds this machine. The engine has six cylinders and a long service interval. " * 5)
    for url, _ in urls:
        web[url] = ("text/html", spec)
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=urls))
    a, sources, recs, _ = _state(aid)
    assert sum(s.status == "PROCESSED" for s in sources) == 4 and a.coverage_metrics["relevant_official_sources"] == 0
    assert all(r.coverage_state == INSUFFICIENT_SOURCE_COVERAGE for r in recs.values())


def test_third_party_pages_alone_do_not_make_no_evidence_honest(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    urls = [(f"https://www.news{i}.example/article", f"{NAME} article {i}") for i in range(4)]
    for url, _ in urls:
        web[url] = ("text/html", _page(f"{NAME} in the news", ABOUT))
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=urls))
    a, sources, recs, _ = _state(aid)
    assert sum(s.status == "PROCESSED" for s in sources) == 4
    assert recs[("data_security", "planning")].coverage_state == INSUFFICIENT_SOURCE_COVERAGE


def test_ambiguous_name_is_not_searched_again(client, web):
    urls = [(f"https://www.{host}.example/{i}", f"{host}") for host in ("polarcars", "polargroup") for i in range(2)]
    search = Search(first=urls, later=urls)
    aid = _create(client, name="Polar")
    run_assessment(aid, llm=FakeLLM(), search=search)
    a, *_ = _state(aid)
    assert a.company_identity["status"] == AMBIGUOUS and a.coverage_metrics["recovery_search_runs"] == 0 and len(search.queries) == 8


def test_ambiguous_name_attributes_nothing_from_the_web(client, web):
    urls = [(f"https://www.{host}.example/{i}", host) for host in ("polarcars", "polargroup") for i in range(2)]
    third = "https://www.polartrucks.example/sustainability"  # one result only: too few to be known as a separate entity
    web[third] = ("text/html", _page("Polar sustainability", "Polar reports its sustainability and climate work here. Polar aims high. " * 10))

    class OneQueryFindsTheThird(Search):
        def search(self, query, max_results):
            found = super().search(query, max_results)
            if len(self.queries) == 1:
                found.append(SearchResult("Polar sustainability", third, "sustainability", query))
            return found

    aid = _create(client, name="Polar")
    run_assessment(aid, llm=FakeLLM(), search=OneQueryFindsTheThird(first=urls))
    a, sources, recs, evidence = _state(aid)
    assert a.company_identity["status"] == AMBIGUOUS and evidence == []
    assert {s.url: s.rejection_reason for s in sources}[third] == AMBIGUOUS
    assert all(r.recommended_level == NOT_ASSESSED_LEVEL for r in recs.values())


def test_ambiguous_name_can_still_be_assessed_from_an_uploaded_report(client, sample_pdf):
    aid = _create(client, name="Nordvik", enable_web_search=False)
    with SessionLocal() as db:
        db.get(Assessment, aid).company_identity = {"status": AMBIGUOUS, "aliases": [], "excluded": []}
        db.commit()
    with open(sample_pdf, "rb") as f:
        assert client.post(f"/api/assessments/{aid}/documents", files={"file": ("r.pdf", f, "application/pdf")}).status_code == 201
    from app.services.pipeline import source_gate

    with SessionLocal() as db:
        source = db.scalars(select(Source).where(Source.assessment_id == aid)).one()
        assert source.source_type == "uploaded_report"
        assert source.rejection_reason != AMBIGUOUS and not source_gate.source_blocks_evidence(source)


# ---------------------------------------------------------------- primary report blocked by a limit
def _good_pages(web):
    urls = [(f"{SITE}/sustainability/{i}", f"{NAME} sustainability {i}") for i in range(3)]
    for url, _ in urls:
        web[url] = ("text/html", ENERGY_PAGE)
    return urls


def _two_page_pdf(company: str = NAME) -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), f"{company} Annual Report 2025")
    doc.new_page().insert_text((72, 72), "We report each year.")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.mark.parametrize("limit, value, reason", [
    ("max_pdf_pages", 1, "pdf_page_limit_exceeded"),
    ("max_download_mb", 1, "download_size_limit_exceeded"),
])
def test_blocked_primary_report_is_a_coverage_gap_not_no_evidence(client, web, monkeypatch, limit, value, reason):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    monkeypatch.setattr(get_settings(), limit, value)
    report = f"{SITE}/annual-report-2025.pdf"
    web[report] = ("application/pdf", _two_page_pdf() + b"x" * (2 * 1024 * 1024))
    urls = [(report, f"{NAME} annual report 2025")]
    urls += _good_pages(web)  # three good official pages: enough for "no evidence" if the report were not missing
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=urls))
    a, sources, recs, _ = _state(aid)
    assert {s.url: s.fetch_outcome for s in sources}[report] == reason
    gap = recs[("data_security", "planning")]
    assert gap.coverage_state == INSUFFICIENT_SOURCE_COVERAGE and gap.recommended_level == NOT_ASSESSED_LEVEL
    assert "primary company report was found" in gap.reason and any("Upload" in m for m in gap.next_level_missing)
    assert a.coverage_metrics["primary_reports_blocked"] == 1
    assert a.coverage_metrics["blocked_primary_reports"][0] == {"url": report, "title": f"{NAME} annual report 2025", "reason": reason}
    notices = client.get(f"/api/assessments/{aid}/status").json()["notices"]
    assert any("Upload the report" in n for n in notices)
    assert reason not in str(notices)  # the technical reason stays in the audit data
    export = client.get(f"/api/assessments/{aid}/export", params={"format": "json"}).json()
    assert export["audit"]["coverage_metrics"]["primary_reports_blocked"] == 1


def test_answers_with_accepted_evidence_keep_their_level_when_a_report_is_blocked(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    monkeypatch.setattr(get_settings(), "max_pdf_pages", 1)
    report = f"{SITE}/sustainability-report-2025.pdf"
    web[report] = ("application/pdf", _two_page_pdf())
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(report, f"{NAME} sustainability report"), *_good_pages(web)]))
    _, _, recs, _ = _state(aid)
    assert recs[("energy_management", "planning")].coverage_state == "EVIDENCE_FOUND"
    assert recs[("energy_management", "planning")].recommended_level >= 4
    assert recs[("data_security", "planning")].coverage_state == INSUFFICIENT_SOURCE_COVERAGE


def test_blocked_report_overrides_another_processed_report(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    monkeypatch.setattr(get_settings(), "max_pdf_pages", 1)
    readable = _report_pdf()
    doc = pymupdf.open(stream=readable, filetype="pdf")
    assert doc.page_count == 1
    doc.close()
    blocked = f"{SITE}/annual-report-2025.pdf"
    web[blocked] = ("application/pdf", _two_page_pdf())
    web[REPORT_URL] = ("application/pdf", readable)
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(REPORT_URL, f"{NAME} sustainability report 2025"), (blocked, f"{NAME} annual report 2025")]))
    a, _, recs, _ = _state(aid)
    assert a.coverage_metrics["primary_reports_processed"] == 1 and a.coverage_metrics["primary_reports_blocked"] == 1
    assert recs[("data_security", "performance")].coverage_state == INSUFFICIENT_SOURCE_COVERAGE


def test_another_companys_blocked_report_does_not_create_a_gap(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    monkeypatch.setattr(get_settings(), "max_pdf_pages", 1)
    other = "https://www.bluefinfoods.example/annual-report-2025.pdf"
    web[other] = ("application/pdf", _two_page_pdf("Bluefin Foods Ltd"))
    similar = "https://www.northstarshipping.example/annual-report-2025.pdf"
    web[similar] = ("application/pdf", _two_page_pdf("Northstar Shipping AB"))
    urls = [*_good_pages(web), (other, "Bluefin Foods annual report 2025"), (similar, "Northstar Shipping annual report 2025")]
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=urls))
    a, sources, recs, _ = _state(aid)
    assert a.coverage_metrics["primary_reports_blocked"] == 0
    assert recs[("data_security", "planning")].coverage_state == NO_EVIDENCE_FOUND


def test_unavailable_or_invalid_report_is_not_called_blocked_by_a_limit(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    missing, fake = f"{SITE}/annual-report-2024.pdf", f"{SITE}/annual-report-2025.pdf"  # the first is a 404
    web[fake] = ("application/pdf", b"not really a pdf")
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(missing, f"{NAME} annual report 2024"), (fake, f"{NAME} annual report 2025"), *_good_pages(web)]))
    a, *_ = _state(aid)
    assert a.coverage_metrics["primary_reports_blocked"] == 0


def test_blocking_a_report_adds_no_model_calls_and_keeps_the_limits(client, web, monkeypatch):
    from app.core.config import Settings

    defaults = Settings(_env_file=None)
    assert defaults.max_pdf_pages == 300 and defaults.max_download_mb == 20
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    pages = _good_pages(web)
    plain, with_blocked = FakeLLM(), FakeLLM()
    run_assessment(_create(client), llm=plain, search=Search(first=pages))
    monkeypatch.setattr(get_settings(), "max_pdf_pages", 1)
    report = f"{SITE}/annual-report-2025.pdf"
    web[report] = ("application/pdf", _two_page_pdf())
    run_assessment(_create(client), llm=with_blocked, search=Search(first=[(report, f"{NAME} annual report 2025"), *pages]))
    assert len(with_blocked.calls) == len(plain.calls)


def test_blocked_report_with_a_neutral_title_is_identified_from_its_first_pages(client, web, monkeypatch):
    """Search often describes a filing-service PDF without any company name."""
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    monkeypatch.setattr(get_settings(), "max_pdf_pages", 1)
    own, other = "https://mb.filings.example/main/1/2/3.pdf", "https://mb.filings.example/main/4/5/6.pdf"
    web[own] = ("application/pdf", _two_page_pdf())
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(own, "Annual and Sustainability Report"), *_good_pages(web)]))
    a, sources, recs, evidence = _state(aid)
    source = {s.url: s for s in sources}[own]
    assert source.status == "REJECTED" and source.fetch_outcome == "pdf_page_limit_exceeded" and source.company_match
    assert a.coverage_metrics["primary_reports_blocked"] == 1
    assert recs[("data_security", "planning")].coverage_state == INSUFFICIENT_SOURCE_COVERAGE
    assert not [e for e in evidence if e.source_id == source.id]

    web[other] = ("application/pdf", _two_page_pdf("Bluefin Foods Ltd"))
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(other, "Annual and Sustainability Report"), *_good_pages(web)]))
    a, sources, recs, _ = _state(aid)
    source = {s.url: s for s in sources}[other]
    assert source.company_match == "mismatch" and source.rejection_reason == "source_company_mismatch"
    assert a.coverage_metrics["primary_reports_blocked"] == 0
    assert recs[("data_security", "planning")].coverage_state == NO_EVIDENCE_FOUND


def test_report_too_large_to_download_with_a_neutral_title_cannot_be_attributed(client, web, monkeypatch):
    """Nothing was received, so nothing proves whose it is: this stays a known limitation, not a false gap."""
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    monkeypatch.setattr(get_settings(), "max_download_mb", 1)
    url = "https://mb.filings.example/main/7/8/9.pdf"
    web[url] = ("application/pdf", _two_page_pdf() + b"x" * (2 * 1024 * 1024))
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(url, "Annual and Sustainability Report"), *_good_pages(web)]))
    a, *_ = _state(aid)
    assert a.coverage_metrics["primary_reports_blocked"] == 0
