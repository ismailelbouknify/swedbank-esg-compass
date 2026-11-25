"""Source discovery when the search provider declines automated searches but websites are reachable.

The bank-laptop case: a single diagnostic search succeeded, but an assessment's burst of queries made DuckDuckGo answer
with a challenge page; the assessment then had no sources. Now search stops at the first refusal (and pauses), the
refusal never stops website retrieval, and the official website is read through its navigation and sitemaps within
fixed limits, robots.txt respected, every document through the same identity and evidence gates.
Deterministic: nothing connects anywhere."""
from __future__ import annotations

import httpx
import pytest

from app import network_check
from app.services.company.resolution import AMBIGUOUS, RESOLVED, UNVERIFIED
from app.services.pipeline import network_notices as notices
from app.services.pipeline import site_discovery as sd
from app.services.pipeline.discovery import DiscoveryReport, WebAccessState, validated_official_domain
from app.services.search import providers
from app.services.search.base import SearchProvider, SearchResult
from app.services.security import policy, safe_fetch
from app.services.security.safe_fetch import FetchError, fetch
from tests.test_security import (  # noqa: F401 — fixtures used by name
    COMPANY, SITE, _assessment, _events, _no_sleep_no_scanner, _run, _sources, client_for, configure, dns,
)

DOMAIN = "nordvikcomponents.example"
PAGE = ("<html><head><title>Nordvik Components AB - {title}</title></head><body><h1>{title}</h1><p>"
        + "Nordvik Components AB sustainability work: energy, climate, diversity and data security. " * 15
        + "</p>{links}</body></html>")


def page(title: str, links: str = "") -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/html"}, content=PAGE.format(title=title, links=links).encode())


class FakeSite:
    """A company website: homepage, robots.txt, sitemap(s), pages and the report PDF, with every request recorded."""

    def __init__(self, report_pdf: bytes, robots: str | None = None, sitemap: str | None = None):
        self.requests: list[str] = []
        self.robots = robots if robots is not None else f"User-agent: *\nDisallow: /internal/\nSitemap: {SITE}/sitemap.xml\n"
        self.sitemap = sitemap if sitemap is not None else f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>{SITE}/en/news/2025/new-office</loc></url>
  <url><loc>{SITE}/en/products/widgets</loc></url>
  <url><loc>{SITE}/en/investors/reports-and-presentations/</loc></url>
  <url><loc>{SITE}/en/sustainability/</loc></url>
  <url><loc>{SITE}/internal/sustainability-report-draft.pdf</loc></url>
  <url><loc>{SITE}/files/annual-and-sustainability-report-2025.pdf</loc></url>
  <url><loc>https://www.other-company.example/sustainability-report-2025.pdf</loc></url>
  <url><loc>{SITE}/en/careers/sustainability-jobs</loc></url>
</urlset>"""
        self.pdf = report_pdf
        self.extra: dict[str, httpx.Response] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        host, path = request.headers.get("host", ""), request.url.path
        self.requests.append(f"{host}{path}")
        if f"{host}{path}" in self.extra:
            return self.extra[f"{host}{path}"]
        if host != f"www.{DOMAIN}":
            return httpx.Response(404)
        if path in ("", "/"):
            return page("Home", '<a href="/en/about/">About us</a><a href="/en/contact/">Contact</a>')
        if path == "/robots.txt":
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=self.robots.encode())
        if path == "/sitemap.xml":
            return httpx.Response(200, headers={"content-type": "application/xml"}, content=self.sitemap.encode())
        if path.endswith(".pdf"):
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=self.pdf)
        return page(path.strip("/").replace("/", " ").title() or "Page")


class RefusingSearch(SearchProvider):
    """Answers the first `answer` queries, then declines like DuckDuckGo's challenge page."""
    name = "ddgs"

    def __init__(self, answer: int = 0, results: list[SearchResult] | None = None):
        self.answer, self.results, self.queries = answer, results or [], []

    def search(self, query, max_results):
        self.queries.append(query)
        if len(self.queries) <= self.answer:
            self.last_error = None
            return list(self.results)
        self.failed_searches += 1
        self.last_error = policy.SEARCH_REFUSED
        return []


@pytest.fixture()
def site(monkeypatch, dns, sample_pdf):
    fake = FakeSite(sample_pdf.read_bytes())
    monkeypatch.setattr(safe_fetch, "build_client", lambda *a, **k: client_for(fake))
    return fake


# ---------------------------------------------------------------- the regression: search refused, website reachable
def test_search_refusal_with_a_known_website_still_reads_the_official_site(client, site):
    aid = _assessment(client, website=SITE)
    search = RefusingSearch()
    st = _run(client, aid, search)
    assert st["status"] == "COMPLETED"

    assert search.queries == [search.queries[0]], "searching stops at the first refusal, also in the recovery pass"
    sources = _sources(aid)
    report_pdf = f"{SITE}/files/annual-and-sustainability-report-2025.pdf"
    assert sources[report_pdf].status == "PROCESSED" and sources[report_pdf].discovered_via.startswith("official website")
    assert sources[f"{SITE}/en/sustainability/"].status == "PROCESSED"
    # robots.txt is respected, other sites in the sitemap are ignored, low-value sections come last
    assert f"www.{DOMAIN}/internal/sustainability-report-draft.pdf" not in site.requests
    assert not any("other-company" in r for r in site.requests)
    refused = [n for n in st["notices"] if n.startswith(notices.SEARCH_REFUSED_MESSAGE)]
    assert refused and refused[0].endswith(notices.SITE_READ_SUFFIX)
    assert notices.STOPPED_MESSAGE not in st["notices"]
    # evidence from the report found on the official site reaches the assessment
    recs = client.get(f"/api/assessments/{aid}/recommendations").json()
    assert any((r["recommended_level"] or 0) > 0 for r in recs)
    explored = [e.details for e in _events(aid) if e.event_type == "official_site_explored"]
    assert explored and explored[0]["domain"] == DOMAIN and explored[0]["sitemap_files"] == 1
    diag = [e.details for e in _events(aid) if e.event_type == "web_access_diagnostics"][0]
    assert diag["search_declined_by_provider"] is True and diag["search_failure_kinds"] == {"provider_restriction": 1}
    assert diag["fetch_stopped"] is False


def test_search_refused_after_it_settled_the_identity_reads_the_validated_official_site(client, site):
    """No website given: the first answer settles the official domain; the refusal after it stops searching only."""
    results = [SearchResult(f"{COMPANY} - Home", f"{SITE}/", "Nordvik Components AB", "q"),
               SearchResult(f"{COMPANY} news", f"{SITE}/en/news/2025/new-office", "Nordvik Components AB", "q")]
    aid = _assessment(client)
    search = RefusingSearch(answer=1, results=results)
    st = _run(client, aid, search)
    assert st["status"] == "COMPLETED"
    assert len(search.queries) == 2, "one answer, one refusal, then no more queries"
    sources = _sources(aid)
    assert sources[f"{SITE}/files/annual-and-sustainability-report-2025.pdf"].status == "PROCESSED"
    assert any(e.event_type == "official_site_explored" for e in _events(aid))


def test_without_a_website_or_settled_identity_no_site_is_guessed(client, site):
    aid = _assessment(client)
    st = _run(client, aid, RefusingSearch())
    assert st["status"] == "COMPLETED"
    assert site.requests == [], "no official domain is known for certain: nothing is fetched"
    refused = [n for n in st["notices"] if n.startswith(notices.SEARCH_REFUSED_MESSAGE)]
    assert refused and "Enter the company website address" in refused[0]


def test_search_failures_never_stop_website_retrieval():
    web = WebAccessState(limit=2)
    for _ in range(5):
        web.record_search(policy.SEARCH_REFUSED)
        web.record_search(policy.PROXY_AUTH_REQUIRED)
    assert web.search_stopped and web.search_declined and not web.fetch_stopped
    web.record_fetch(None)
    assert not web.fetch_stopped


def test_a_refusal_after_successful_searches_still_stops_searching():
    web = WebAccessState(limit=3)
    web.record_search(None)
    web.record_search(None)
    assert not web.search_stopped
    web.record_search(policy.SEARCH_RATE_LIMITED)
    assert web.search_stopped and web.search_unavailable


def test_only_search_stopped_is_named_as_such():
    web = WebAccessState(limit=2)
    web.record_fetch(None)
    web.record_search(policy.NETWORK_ERROR)
    web.record_search(policy.NETWORK_ERROR)
    out = notices.network_notices(web)
    assert notices.SEARCH_STOPPED_MESSAGE in out and notices.STOPPED_MESSAGE not in out


@pytest.mark.parametrize("code,kind", [
    (policy.SEARCH_REFUSED, "provider_restriction"), (policy.SEARCH_RATE_LIMITED, "provider_restriction"),
    (policy.PROXY_AUTH_REQUIRED, "corporate_network"), (policy.BLOCKED_BY_WEB_FILTER, "corporate_network"),
    (policy.PROXY_SIGNIN_REFUSED, "corporate_network"), (policy.TLS_VERIFICATION_FAILED, "corporate_network"),
    (policy.NETWORK_ERROR, "connectivity"), (policy.SOURCE_TIMEOUT, "connectivity"), (None, "other"),
])
def test_search_failures_are_told_apart(code, kind):
    assert policy.search_failure_kind(code) == kind


# ---------------------------------------------------------------- official site stage: limits and rules
def test_website_stage_stays_within_its_limits(client, site, configure):
    configure(website_max_pages=1, website_max_pdfs=1, max_pages_per_site=1)
    many = "".join(f"<url><loc>{SITE}/en/sustainability/topic-{i}/</loc></url>" for i in range(40))
    site.sitemap = f"<urlset>{many}<url><loc>{SITE}/a/sustainability-report-2025.pdf</loc></url>" \
                   f"<url><loc>{SITE}/b/annual-report-2025.pdf</loc></url></urlset>"
    aid = _assessment(client, website=SITE)
    _run(client, aid, RefusingSearch())
    via_site = [s for s in _sources(aid).values() if (s.discovered_via or "").startswith("official website")]
    assert len([s for s in via_site if s.url.endswith(".pdf")]) <= 1
    assert len([s for s in via_site if not s.url.endswith(".pdf")]) <= 1


def test_robots_server_error_means_no_exploration(client, site):
    site.extra[f"www.{DOMAIN}/robots.txt"] = httpx.Response(503)
    aid = _assessment(client, website=SITE)
    _run(client, aid, RefusingSearch())
    assert f"www.{DOMAIN}/sitemap.xml" not in site.requests
    assert not any((s.discovered_via or "").startswith("official website") for s in _sources(aid).values())


def test_sitemap_index_and_default_location(client, site):
    site.robots = "User-agent: *\nDisallow:\n"  # names no sitemap: /sitemap.xml is tried
    site.sitemap = f"""<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>{SITE}/sitemap-products.xml</loc></sitemap>
      <sitemap><loc>{SITE}/sitemap-pages.xml</loc></sitemap>
      <sitemap><loc>{SITE}/sitemap-archive.xml.gz</loc></sitemap></sitemapindex>"""
    site.extra[f"www.{DOMAIN}/sitemap-pages.xml"] = httpx.Response(
        200, headers={"content-type": "text/xml"},
        content=f"<urlset><url><loc>{SITE}/reports/sustainability-report-2025.pdf</loc></url></urlset>".encode())
    aid = _assessment(client, website=SITE)
    _run(client, aid, RefusingSearch())
    assert site.requests.index(f"www.{DOMAIN}/sitemap-pages.xml") < site.requests.index(f"www.{DOMAIN}/sitemap-products.xml") \
        if f"www.{DOMAIN}/sitemap-products.xml" in site.requests else True
    assert f"www.{DOMAIN}/sitemap-archive.xml.gz" not in site.requests
    assert _sources(aid)[f"{SITE}/reports/sustainability-report-2025.pdf"].status == "PROCESSED"


def test_website_discovery_can_be_switched_off(client, site, configure):
    configure(website_discovery_enabled=False)
    aid = _assessment(client, website=SITE)
    _run(client, aid, RefusingSearch())
    assert f"www.{DOMAIN}/robots.txt" not in site.requests and f"www.{DOMAIN}/sitemap.xml" not in site.requests


def test_a_given_website_that_cannot_be_read_is_not_retried_under_other_addresses(client, dns, monkeypatch):
    requests = []
    monkeypatch.setattr(safe_fetch, "build_client",
                        lambda *a, **k: client_for(lambda r: (requests.append(r), httpx.Response(503))[1]))
    aid = _assessment(client, website=SITE)
    _run(client, aid, RefusingSearch())
    assert {r.headers["host"] for r in requests} == {f"www.{DOMAIN}"}
    assert all(r.url.path in ("", "/") for r in requests)


@pytest.mark.parametrize("profile,website,expected", [
    ({"status": RESOLVED, "official_domain": DOMAIN}, None, DOMAIN),
    ({"status": AMBIGUOUS, "official_domain": DOMAIN}, None, None),
    ({"status": UNVERIFIED, "official_domain": None}, None, None),
    ({"status": RESOLVED, "official_domain": None}, None, None),
    (None, SITE, DOMAIN),
])
def test_only_a_validated_domain_is_read_as_official(profile, website, expected):
    class C:
        pass

    a, a.company = C(), C()
    a.company_identity, a.company.website = profile, website
    assert validated_official_domain(a, DiscoveryReport(official_domain=DOMAIN if website else None)) == expected


# ---------------------------------------------------------------- helpers
def test_robots_rules():
    rules = sd.parse_robots("User-agent: *\nDisallow: /internal/\n\nUser-agent: BadBot\nDisallow: /\n"
                            "Sitemap: /sitemap_index.xml\nSitemap: https://cdn.example/s.xml\n", f"{SITE}/")
    assert rules.allows(f"{SITE}/en/sustainability/") and not rules.allows(f"{SITE}/internal/x.pdf")
    assert rules.sitemaps == [f"{SITE}/sitemap_index.xml", "https://cdn.example/s.xml"]
    assert sd.SiteRules().allows(f"{SITE}/anything")


def test_sitemap_parsing_is_pattern_based_and_bounded():
    text = ('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><ns:urlset xmlns:ns="s">'
            '<ns:url><ns:loc> https://a.example/x?a=1&amp;b=2 </ns:loc></ns:url><url><loc>&a;</loc></url></ns:urlset>')
    is_index, locs = sd.parse_sitemap(text)
    assert not is_index and locs[0] == "https://a.example/x?a=1&b=2"
    assert len(sd.parse_sitemap("<urlset>" + "<url><loc>https://a.example/p</loc></url>" * 20 + "</urlset>", limit=5)[1]) == 5


def test_ranking_prefers_current_reports_and_esg_sections():
    ranked = [u for _, u in sd.rank_site_urls([
        (f"{SITE}/en/careers/sustainability-jobs", ""), (f"{SITE}/files/annual-report-2019.pdf", ""),
        (f"{SITE}/files/annual-and-sustainability-report-2025.pdf", ""), (f"{SITE}/en/products/widget", ""),
        (f"{SITE}/en/investors/reports/", ""), ("https://other.example/sustainability-report-2025.pdf", ""),
        (f"{SITE}/img/sustainability.jpg", ""), (f"{SITE}/en/about/", "Sustainability"),
    ], DOMAIN, 2025)]
    assert ranked[0] == f"{SITE}/files/annual-and-sustainability-report-2025.pdf"
    assert f"{SITE}/en/investors/reports/" in ranked and f"{SITE}/en/careers/sustainability-jobs" not in ranked
    assert ranked.index(f"{SITE}/files/annual-and-sustainability-report-2025.pdf") < ranked.index(f"{SITE}/files/annual-report-2019.pdf")
    assert f"{SITE}/en/about/" in ranked  # navigation anchor text counts
    assert not any("other.example" in u or u.endswith((".jpg", "/widget")) for u in ranked)


def test_text_documents_through_the_retrieval_transport(dns):
    with client_for(lambda r: httpx.Response(200, headers={"content-type": "application/xml"}, content=b"<urlset/>")) as c:
        assert fetch(f"{SITE}/sitemap.xml", expect="text", client=c).kind == "text"
    with client_for(lambda r: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.7 x")) as c, \
            pytest.raises(FetchError):
        fetch(f"{SITE}/sitemap.xml", expect="text", client=c)


# ---------------------------------------------------------------- search pacing and cool-down (keyless DuckDuckGo)
LITE = b"<html><body><a class='result-link' href='https://www.nordvikcomponents.example/'>x</a></body></html>"
CHALLENGE = b"<html><body><div class='anomaly-modal'>Unfortunately, bots use DuckDuckGo too.</div></body></html>"


def test_queries_are_paced(dns, monkeypatch, configure):
    configure(search_min_interval_seconds=5)
    waits = []
    monkeypatch.setattr(providers.pacer, "sleep", waits.append)
    with client_for(lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=LITE)) as c:
        p = providers.DuckDuckGoLiteSearchProvider(c)
        p.search("a", 5)
        p.search("b", 5)
    assert len(waits) == 1 and 4 < waits[0] <= 5


@pytest.mark.parametrize("response,code", [
    (httpx.Response(200, headers={"content-type": "text/html"}, content=CHALLENGE), policy.SEARCH_REFUSED),
    (httpx.Response(429, headers={"content-type": "text/html"}, content=b"<html>slow down</html>"), policy.SEARCH_RATE_LIMITED),
])
def test_a_refusal_pauses_searching_without_further_requests(dns, configure, response, code):
    configure(max_http_retries=0)
    calls = []
    with client_for(lambda r: (calls.append(r), response)[1]) as c:
        p = providers.DuckDuckGoLiteSearchProvider(c)
        assert p.search("a", 5) == [] and p.last_error == code
        assert providers.pacer.cooldown_remaining() > 0
        assert p.search("b", 5) == [] and p.last_error == policy.SEARCH_REFUSED
    assert len(calls) == 1, "nothing is sent during the cool-down"


def test_ddgs_client_rate_limit_is_recognised():
    from app.services.security import network

    assert network.classify_search_error(RuntimeError("RatelimitException: 202 Ratelimit")) == policy.SEARCH_RATE_LIMITED
    assert network.classify_search_error(RuntimeError("ProxyError 407 Proxy Authentication Required")) == policy.PROXY_AUTH_REQUIRED


# ---------------------------------------------------------------- diagnostics: one query vs sustained discovery
class ScriptedSearch(SearchProvider):
    name = "duckduckgo_lite"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def search(self, query, max_results):
        self.last_error = self.outcomes.pop(0) if self.outcomes else None
        return [] if self.last_error else [SearchResult("t", "https://x.example/", "", query)]


def test_sustained_search_check_reports_a_provider_limit(monkeypatch, capsys):
    import app.services.search.providers as sp

    monkeypatch.setattr(sp, "get_search_provider", lambda: ScriptedSearch([None, None, policy.SEARCH_REFUSED]))
    assert network_check.check_sustained_search() is False
    out = capsys.readouterr().out
    assert "query 3 FAILED (search_service_refused, provider_restriction) after 2 answered" in out
    assert "read the company's official website instead" in out


def test_sustained_search_check_ok(monkeypatch, capsys):
    import app.services.search.providers as sp

    monkeypatch.setattr(sp, "get_search_provider", lambda: ScriptedSearch([]))
    assert network_check.check_sustained_search() is True
    assert f"OK: {len(network_check.SUSTAINED_QUERIES)}/{len(network_check.SUSTAINED_QUERIES)} answered" in capsys.readouterr().out


def test_diagnostics_summary_separates_one_search_from_sustained_discovery(monkeypatch, capsys):
    for name, value in (("check_configuration", None), ("check_route", None), ("check_proxy_signin", None),
                        ("check_local_model", None)):
        monkeypatch.setattr(network_check, name, lambda *a, v=value: v)
    monkeypatch.setattr(network_check, "check_fetch", lambda url: True)
    monkeypatch.setattr(network_check, "check_search", lambda: True)
    monkeypatch.setattr(network_check, "check_sustained_search", lambda: False)
    monkeypatch.setattr(network_check, "check_website_discovery", lambda url: True)
    assert network_check.main(["x", "--website", SITE]) == 1
    out = capsys.readouterr().out
    assert "Web search (one query)     OK" in out and "Web search (sustained)     LIMITED" in out
    assert "Official-site discovery    OK" in out and "enter its address on the New assessment page" in out


def test_website_discovery_check(site, capsys):
    assert network_check.check_website_discovery(SITE + "/") is True
    out = capsys.readouterr().out
    assert "1 file(s) read" in out and f"{SITE}/files/annual-and-sustainability-report-2025.pdf" in out
    assert "excluded by robots.txt" in out
