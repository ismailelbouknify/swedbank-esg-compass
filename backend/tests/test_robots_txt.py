"""robots.txt is honoured on every automated retrieval path: search results, sitemap candidates, links found on pages,
report PDFs and redirect targets. Rules are cached per origin. A source a site excludes is recorded as such in the audit
trail and leads to "insufficient source coverage" (not assessed), never to a low ESG level. Manual upload still works,
and search keeps working independently. Deterministic: nothing connects anywhere."""
from __future__ import annotations

import httpx
import pytest
from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models.entities import AuditEvent
from app.services.pipeline import network_notices as notices
from app.services.pipeline.coverage import INSUFFICIENT_SOURCE_COVERAGE, NOT_ASSESSED_LEVEL
from app.services.pipeline.runner import run_assessment
from app.services.security import policy, robots, safe_fetch
from app.services.security.safe_fetch import FetchError, fetch
from tests.conftest import FakeLLM
from tests.test_security import _no_sleep_no_scanner, client_for, configure, dns  # noqa: F401 — fixtures
from tests.test_source_coverage import (  # noqa: F401
    NAME, SITE, Search, _create, _good_pages, _report_pdf, _state, _two_page_pdf, web,
)
from tests.test_windows_proxy import PROXY, TOKEN, FakeProxy, signin  # noqa: F401

pytestmark = pytest.mark.robots

HOST = SITE.removeprefix("https://")


def robots_txt(body: str) -> tuple[str, bytes]:
    return ("text/plain", body.encode())


def events(aid: int, kind: str) -> list[AuditEvent]:
    with SessionLocal() as db:
        return list(db.scalars(select(AuditEvent).where(AuditEvent.assessment_id == aid, AuditEvent.event_type == kind)))


# ---------------------------------------------------------------- every path
def test_search_results_excluded_by_robots_are_not_fetched_and_are_audited(client, web, configure):
    configure(recovery_search_enabled=False)
    web[f"{SITE}/robots.txt"] = robots_txt("User-agent: *\nDisallow: /private/\n")
    blocked = f"{SITE}/private/sustainability-report-2025.pdf"
    web[blocked] = ("application/pdf", _two_page_pdf())
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(blocked, f"{NAME} sustainability report 2025"), *_good_pages(web)]))
    _, sources, _, _ = _state(aid)
    s = {x.url: x for x in sources}[blocked]
    assert s.status == "REJECTED" and s.fetch_outcome == policy.ROBOTS_DISALLOWED
    assert blocked not in web["__requested__"]
    audited = [e for e in events(aid, "robots_disallowed") if e.entity_id == s.id]
    assert audited and audited[0].details["reason"] == policy.ROBOTS_DISALLOWED and audited[0].details["host"] == HOST
    assert "robots.txt" in audited[0].details["detail"] and "robots.txt" in s.status_detail
    assert all(x.status == "PROCESSED" for x in sources if "/sustainability/" in x.url), "allowed pages are still read"


def test_links_and_report_pdfs_found_on_pages_honour_robots(client, web, configure):
    """The validation case: a homepage link and a linked report PDF in excluded paths are no longer fetched."""
    configure(recovery_search_enabled=False)
    web[f"{SITE}/robots.txt"] = robots_txt("User-agent: *\nDisallow: /files/\nDisallow: /en/about/\n")
    web[f"{SITE}/"] = ("text/html", f"<html><title>{NAME}</title><body><p>{NAME} sustainability. " * 10
                       + '<a href="/en/about/sustainability">Sustainability</a>'
                       + '<a href="/en/sustainability/">Our sustainability work</a></p></body></html>')
    web[f"{SITE}/en/sustainability/"] = ("text/html", f"<html><title>{NAME} Sustainability</title><body><p>"
                                         + f"{NAME} sustainability and climate work. " * 10
                                         + '<a href="/files/sustainability-report-2025.pdf">Sustainability report 2025</a></p></body></html>')
    aid = _create(client, website=SITE)
    run_assessment(aid, llm=FakeLLM(), search=Search())
    requested = web["__requested__"]
    assert f"{SITE}/en/about/sustainability" not in requested
    assert f"{SITE}/files/sustainability-report-2025.pdf" not in requested
    assert f"{SITE}/en/sustainability/" in requested
    outcomes = {s.url: s.fetch_outcome for s in _state(aid)[1]}
    assert outcomes.get(f"{SITE}/files/sustainability-report-2025.pdf") in (None, policy.ROBOTS_DISALLOWED)


def test_redirect_into_an_excluded_path_is_refused(web):
    web[f"{SITE}/robots.txt"] = robots_txt("User-agent: *\nDisallow: /internal/\n")

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"User-agent: *\nDisallow: /internal/\n")
        if request.url.path == "/report":
            return httpx.Response(302, headers={"location": "/internal/report.pdf"})
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.7 x")

    with client_for(handler) as c, pytest.raises(FetchError) as e:
        fetch(f"{SITE}/report", client=c)
    assert e.value.code == policy.ROBOTS_DISALLOWED and e.value.redirected
    assert policy.audit_event_for(e.value) == "robots_disallowed"


def test_redirect_to_another_site_uses_that_sites_robots(dns):
    seen = []

    def handler(request):
        host, path = request.headers["host"], request.url.path
        seen.append(f"{host}{path}")
        if path == "/robots.txt":
            body = b"User-agent: *\nDisallow: /\n" if host == "cdn.files.example" else b""
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=body)
        if host == HOST:
            return httpx.Response(302, headers={"location": "https://cdn.files.example/report.pdf"})
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.7 x")

    with client_for(handler) as c, pytest.raises(FetchError) as e:
        fetch(f"{SITE}/report.pdf", client=c)
    assert e.value.code == policy.ROBOTS_DISALLOWED
    assert "cdn.files.example/report.pdf" not in seen and "cdn.files.example/robots.txt" in seen


def test_the_typed_website_is_fetched_but_its_redirects_are_checked(dns):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"User-agent: *\nDisallow: /\n")
        if request.url.path == "/":
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html><body>Home</body></html>")
        return httpx.Response(302, headers={"location": "/"})

    with client_for(handler) as c:
        assert fetch(f"{SITE}/", client=c, robots_txt="typed_address").kind == "html"
        with pytest.raises(FetchError) as e:
            fetch(f"{SITE}/old-home", client=c, robots_txt="typed_address")
        assert e.value.code == policy.ROBOTS_DISALLOWED and e.value.redirected
        with pytest.raises(FetchError):
            fetch(f"{SITE}/", client=c)  # the same address found automatically is not read


# ---------------------------------------------------------------- cache and RFC 9309 cases
def test_rules_are_cached_per_origin(dns):
    calls = []

    def handler(request):
        calls.append(f"{request.url.scheme}://{request.headers['host']}{request.url.path}")
        if request.url.path == "/robots.txt":
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"User-agent: *\nDisallow: /x/\n")
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html><body>p</body></html>")

    with client_for(handler) as c:
        for path in ("/a", "/b", "/c", "/d"):
            fetch(f"{SITE}{path}", client=c)
        fetch("https://other.example/a", client=c)
    assert calls.count(f"{SITE}/robots.txt") == 1 and calls.count("https://other.example/robots.txt") == 1
    assert robots.cache.fetches == 2


@pytest.mark.parametrize("status,allowed", [(404, True), (403, True), (410, True), (503, False), (500, False)])
def test_robots_txt_answers_follow_rfc_9309(dns, configure, status, allowed):
    configure(max_http_retries=0)

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(status)
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html><body>p</body></html>")

    with client_for(handler) as c:
        if allowed:
            assert fetch(f"{SITE}/page", client=c).kind == "html"
        else:
            with pytest.raises(FetchError) as e:
                fetch(f"{SITE}/page", client=c)
            assert e.value.code == policy.ROBOTS_DISALLOWED


def test_a_network_failure_reading_robots_is_reported_as_the_network_failure(dns, configure):
    """The proxy refusing robots.txt is the network's answer, not the site's: diagnostics must say so."""
    configure(max_http_retries=0)
    blue_coat = (b"<html><body><h1>Access Denied (authentication_failed)</h1><p>Credentials are missing.</p>"
                 b"<p>Blue Coat Web Filter</p></body></html>")
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(407 if request.url.path == "/robots.txt" else 200,
                              headers={"content-type": "text/html"}, content=blue_coat)

    with client_for(handler) as c:
        for _ in range(2):
            with pytest.raises(FetchError) as e:
                fetch(f"{SITE}/page", client=c)
            assert e.value.code == policy.PROXY_AUTH_REQUIRED
    assert calls == ["/robots.txt"], "the failure is cached briefly; the page itself is not attempted"


def test_robots_txt_goes_through_the_signed_in_proxy(signin, monkeypatch):
    proxy = FakeProxy(monkeypatch)
    proxy.handler = lambda request: httpx.Response(
        200, headers={"content-type": "text/plain" if request.url.path == "/robots.txt" else "text/html"},
        content=b"User-agent: *\nAllow: /\n" if request.url.path == "/robots.txt" else b"<html><body>p</body></html>")
    assert fetch("https://www.company.example/page").kind == "html"
    signed = {url.split("company.example", 1)[1]: ok for url, ok in proxy.seen if ok}
    assert "/robots.txt" in signed and "/page" in signed


# ---------------------------------------------------------------- coverage, not low scores
def test_report_excluded_by_robots_is_a_coverage_gap_not_a_low_score(client, web, configure):
    configure(recovery_search_enabled=False)
    report = f"{SITE}/reports/annual-report-2025.pdf"
    web[f"{SITE}/robots.txt"] = robots_txt("User-agent: *\nDisallow: /reports/\n")
    web[report] = ("application/pdf", _two_page_pdf())
    urls = [(report, f"{NAME} annual report 2025"), *_good_pages(web)]
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=urls))
    a, sources, recs, _ = _state(aid)
    assert {s.url: s.fetch_outcome for s in sources}[report] == policy.ROBOTS_DISALLOWED
    gap = recs[("data_security", "planning")]
    assert gap.coverage_state == INSUFFICIENT_SOURCE_COVERAGE and gap.recommended_level == NOT_ASSESSED_LEVEL
    assert a.coverage_metrics["blocked_primary_reports"][0]["reason"] == policy.ROBOTS_DISALLOWED
    st = client.get(f"/api/assessments/{aid}/status").json()
    assert notices.ROBOTS_MESSAGE in st["notices"] and any("Upload the report" in n for n in st["notices"])
    no_evidence = [r for r in recs.values() if r.coverage_state != "EVIDENCE_FOUND"]
    assert no_evidence and all(r.recommended_level == NOT_ASSESSED_LEVEL for r in no_evidence), \
        "no answer is scored as absent activity while the company's report was excluded"


def test_manual_upload_still_works_when_the_site_excludes_everything(client, web, configure):
    configure(recovery_search_enabled=False)
    web[f"{SITE}/robots.txt"] = robots_txt("User-agent: *\nDisallow: /\n")
    aid = _create(client, website=SITE)
    upload = {"file": ("report.pdf", _report_pdf(), "application/pdf")}
    assert client.post(f"/api/assessments/{aid}/documents", files=upload).status_code == 201
    run_assessment(aid, llm=FakeLLM(), search=Search())
    _, sources, recs, _ = _state(aid)
    assert any(s.source_type == "uploaded_report" and s.status == "PROCESSED" for s in sources)
    assert any((r.recommended_level or 0) > 0 for r in recs.values())


def test_excluded_candidates_leave_the_budget_to_the_next_one(client, web, configure):
    configure(recovery_search_enabled=False, max_downloaded_pdfs=1)
    excluded, allowed = f"{SITE}/private/sustainability-report-2025.pdf", f"{SITE}/public/sustainability-report-2025.pdf"
    web[f"{SITE}/robots.txt"] = robots_txt("User-agent: *\nDisallow: /private/\n")
    web[excluded] = web[allowed] = ("application/pdf", _two_page_pdf())
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(excluded, f"{NAME} sustainability report 2025"),
                                                           (allowed, f"{NAME} sustainability report 2025")]))
    assert allowed in web["__requested__"] and excluded not in web["__requested__"]


def test_search_keeps_working_when_company_sites_exclude_everything(client, web, configure):
    """Independent components: a site's robots.txt says nothing about the search service."""
    configure(recovery_search_enabled=False)
    web[f"{SITE}/robots.txt"] = robots_txt("User-agent: *\nDisallow: /\n")
    search = Search(first=[(f"{SITE}/sustainability", f"{NAME} sustainability")])
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=search)
    assert len(search.queries) == len(set(search.queries)) >= 1
    st = client.get(f"/api/assessments/{aid}/status").json()
    assert st["status"] == "COMPLETED" and notices.ROBOTS_MESSAGE in st["notices"]
    assert notices.STOPPED_MESSAGE not in st["notices"]
