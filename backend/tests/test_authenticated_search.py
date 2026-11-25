"""Web search on networks whose proxy requires integrated Windows sign-in.

The DDGS library sends requests through its own HTTP client (primp), which takes a proxy only as a URL and cannot add
a Kerberos sign-in. When integrated sign-in applies to the search route, searches therefore go through the
application's own retrieval transport (PAC, Kerberos, certificate and address checks), reading DuckDuckGo's lite
results page. A challenge page from DuckDuckGo is reported, never worked around. Deterministic: nothing connects."""
from __future__ import annotations

import sys
import types

import httpx
import pytest

from app import network_check
from app.services.pipeline import network_notices as notices
from app.services.pipeline.discovery import WebAccessState
from app.services.presentation import analyst_notices
from app.services.search import providers
from app.services.security import network, policy, safe_fetch, windows_proxy
from app.services.security.safe_fetch import FetchError, fetch
from app.services.security.windows_proxy import IntegratedAuthUnavailable
from tests.test_security import _no_sleep_no_scanner, configure, dns  # noqa: F401 — fixtures
from tests.test_windows_proxy import PROXY, TOKEN, FakeProxy, signin  # noqa: F401 — fixtures

LITE_PAGE = b"""<!DOCTYPE html><html><body><form action="/lite/" method="post"></form><table>
<tr><td>1.&nbsp;</td><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.company.example%2Fsustainability%2Freport%2D2025.pdf&amp;rut=abc" class='result-link'>Sustainability Report 2025 - Company</a></td></tr>
<tr><td>&nbsp;</td><td class='result-snippet'>Our <b>sustainability</b> report covers energy and diversity.</td></tr>
<tr><td>&nbsp;</td><td><span class='link-text'>www.company.example</span></td></tr>
<tr><td>2.&nbsp;</td><td><a rel="nofollow" href="https://duckduckgo.com/y.js?ad_domain=ads.example&amp;u3=x" class='result-link'>Sponsored</a></td></tr>
<tr><td>&nbsp;</td><td class='result-snippet'>An advert.</td></tr>
<tr><td>3.&nbsp;</td><td><a rel="nofollow" href="https://www.annualreport.company.example/2025/" class='result-link'>Annual Report 2025</a></td></tr>
<tr><td>&nbsp;</td><td class='result-snippet'>Integrated annual report.</td></tr>
<tr><td>4.&nbsp;</td><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.company.example%2Fsustainability%2Freport%2D2025.pdf" class='result-link'>Duplicate</a></td></tr>
<tr><td>5.&nbsp;</td><td><a rel="nofollow" href="javascript:alert(1)" class='result-link'>Not a link</a></td></tr>
</table></body></html>"""
CHALLENGE_PAGE = (b"<html><body><div class='anomaly-modal__title'>Unfortunately, bots use DuckDuckGo too.</div>"
                  b"<form id='challenge-form'></form></body></html>")
EMPTY_PAGE = b"<html><body><form action='/lite/'></form><table><tr><td>No results.</td></tr></table></body></html>"


def lite(body=LITE_PAGE, status=200) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": "text/html; charset=UTF-8"}, content=body)


@pytest.fixture()
def no_ddgs_library(monkeypatch):
    """Fails the test if the DDGS client is used."""
    class Forbidden:
        def __init__(self, *a, **k):
            raise AssertionError("the DDGS client must not be used when integrated proxy sign-in is needed")

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=Forbidden))


@pytest.fixture()
def fake_ddgs(monkeypatch):
    calls = []

    class FakeDDGS:
        def __init__(self, **kw):
            calls.append(kw)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results):
            return [{"title": "T", "href": "https://www.company.example/esg", "body": "B"}]

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=FakeDDGS))
    return calls


# ---------------------------------------------------------------- the results page
def test_lite_results_are_parsed_without_ads_duplicates_or_non_web_links():
    results = providers.parse_duckduckgo_lite(LITE_PAGE, "q")
    assert [r.url for r in results] == ["https://www.company.example/sustainability/report-2025.pdf",
                                        "https://www.annualreport.company.example/2025/"]
    assert results[0].title == "Sustainability Report 2025 - Company"
    assert results[0].snippet == "Our sustainability report covers energy and diversity."
    assert results[0].query == "q"


def test_lite_search_through_the_retrieval_transport(dns):
    seen = []

    def handler(request):
        seen.append(request)
        return lite()

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        p = providers.DuckDuckGoLiteSearchProvider(client)
        results = p.search("Company AB sustainability report 2025", 1)
    assert len(results) == 1 and p.last_error is None and p.failed_searches == 0
    url = httpx.URL(str(seen[0].url))
    assert seen[0].headers["host"] == "lite.duckduckgo.com" and url.params["q"] == "Company AB sustainability report 2025"


def test_search_identifies_the_application_honestly(dns, monkeypatch):
    """The real client of the retrieval transport: its own User-Agent, certificate verification on, no imitation."""
    seen = []
    original = safe_fetch.build_client

    def build(proxy=None, proxy_headers=None):
        client = original(proxy, proxy_headers)
        client._transport = httpx.MockTransport(lambda r: (seen.append(r), lite())[1])
        return client

    monkeypatch.setattr(safe_fetch, "build_client", build)
    p = providers.DuckDuckGoLiteSearchProvider()
    assert p.search("q", 5) and seen[0].headers["user-agent"] == safe_fetch.USER_AGENT


def test_a_challenge_page_is_reported_and_not_worked_around(dns):
    calls = []

    def handler(request):
        calls.append(request)
        return lite(CHALLENGE_PAGE)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        p = providers.DuckDuckGoLiteSearchProvider(client)
        assert p.search("q", 5) == []
    assert p.last_error == policy.SEARCH_REFUSED and p.failed_searches == 1 and len(calls) == 1


def test_no_results_is_an_empty_answer_not_a_failure(dns):
    with httpx.Client(transport=httpx.MockTransport(lambda r: lite(EMPTY_PAGE))) as client:
        p = providers.DuckDuckGoLiteSearchProvider(client)
        assert p.search("q", 5) == []
    assert p.last_error is None and p.failed_searches == 0


def test_search_pages_follow_the_search_switch_not_the_retrieval_switch(configure, dns):
    configure(enable_external_fetch=False, enable_web_search=True)
    with httpx.Client(transport=httpx.MockTransport(lambda r: lite())) as client:
        assert fetch(providers.DDG_LITE_URL + "?q=x", expect="html", client=client, purpose="search").kind == "html"
        with pytest.raises(FetchError) as e:
            fetch("https://www.company.example/", client=client)
    assert e.value.code == policy.EXTERNAL_FETCH_DISABLED
    configure(enable_external_fetch=True, enable_web_search=False)
    with pytest.raises(FetchError) as e:
        fetch(providers.DDG_LITE_URL + "?q=x", expect="html", purpose="search")
    assert e.value.code == policy.EXTERNAL_FETCH_DISABLED


def test_search_requests_keep_the_address_checks(dns):
    """A redirect from the search service to an internal address is refused like any other."""
    def handler(request):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        p = providers.DuckDuckGoLiteSearchProvider(client)
        assert p.search("q", 5) == []
    assert p.last_error == policy.PRIVATE_IP


# ---------------------------------------------------------------- DDGS: which transport
def test_ddgs_uses_the_signed_in_transport_when_the_proxy_needs_windows_signin(signin, monkeypatch, no_ddgs_library):
    proxy = FakeProxy(monkeypatch)
    proxy.handler = lambda request: lite()
    p = providers.DDGSSearchProvider()
    results = p.search("Company AB sustainability report", 10)
    assert [r.url for r in results][0] == "https://www.company.example/sustainability/report-2025.pdf"
    assert p.transport == "application" and p.last_error is None
    assert signin == ["proxy.corp.example"]  # Kerberos for HTTP/<proxy>, only after the proxy asked
    assert proxy.built[0] is None and proxy.built[1] == {"Proxy-Authorization": f"Negotiate {TOKEN}1"}
    assert all(url.startswith(providers.DDG_LITE_URL) for url, _ in proxy.seen)


def test_ddgs_client_still_used_without_integrated_signin(configure, monkeypatch, fake_ddgs):
    configure(web_proxy_auth="none")
    monkeypatch.setattr(network, "proxy_for", lambda url: PROXY)
    p = providers.DDGSSearchProvider()
    assert len(p.search("q", 5)) == 1 and p.transport == "ddgs"
    assert fake_ddgs[0]["proxy"] == PROXY and fake_ddgs[0]["verify"] is not False


def test_ddgs_client_still_used_on_a_direct_route(configure, monkeypatch, fake_ddgs):
    configure(web_proxy_auth="integrated")
    monkeypatch.setattr(network, "proxy_for", lambda url: None)
    p = providers.DDGSSearchProvider()
    assert len(p.search("q", 5)) == 1 and p.transport == "ddgs" and fake_ddgs[0]["proxy"] is None


def test_ddgs_without_signin_reports_the_proxy_requirement(configure, monkeypatch):
    configure(web_proxy_auth="none")
    monkeypatch.setattr(network, "proxy_for", lambda url: PROXY)

    class ProxyRefuses:
        def __init__(self, **kw):
            raise RuntimeError("DDGSException: ProxyError('407 Proxy Authentication Required')")

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=ProxyRefuses))
    p = providers.DDGSSearchProvider()
    assert p.search("q", 5) == [] and p.last_error == policy.PROXY_AUTH_REQUIRED and p.failed_searches == 1


def test_signed_in_search_refused_by_the_proxy(signin, monkeypatch, no_ddgs_library):
    FakeProxy(monkeypatch, accept=False)
    p = providers.DDGSSearchProvider()
    assert p.search("q", 5) == []
    assert p.last_error == policy.PROXY_SIGNIN_REFUSED and p.failed_searches == 1


def test_search_when_windows_cannot_provide_kerberos(signin, monkeypatch, no_ddgs_library):
    def unavailable(host):
        raise IntegratedAuthUnavailable("Windows could only offer NTLM for this proxy, not Kerberos")

    monkeypatch.setattr(windows_proxy, "kerberos_proxy_token", unavailable)
    FakeProxy(monkeypatch)
    p = providers.DDGSSearchProvider()
    assert p.search("q", 5) == [] and p.last_error == policy.PROXY_SIGNIN_UNAVAILABLE


def test_search_then_retrieval_both_sign_in_with_fresh_tickets(signin, monkeypatch, no_ddgs_library):
    """The regression from the bank computer: retrieval worked, search did not. Both now use one transport."""
    proxy = FakeProxy(monkeypatch)
    proxy.handler = lambda request: lite() if request.url.host == "lite.duckduckgo.com" else httpx.Response(
        200, headers={"content-type": "text/html"}, content=b"<html><body>Report</body></html>")
    p = providers.DDGSSearchProvider()
    results = p.search("Company AB sustainability report", 10)
    page = fetch(results[1].url)
    assert page.kind == "html" and p.last_error is None
    tickets = [h["Proxy-Authorization"] for h in proxy.built if h]
    assert tickets == [f"Negotiate {TOKEN}1", f"Negotiate {TOKEN}2"]  # second request signs in at once, new ticket
    assert all(signed for _, signed in proxy.seen[1:])


def test_signin_is_never_sent_to_the_search_service(signin, monkeypatch, no_ddgs_library):
    proxy = FakeProxy(monkeypatch)
    sent = []
    proxy.handler = lambda request: (sent.append(dict(request.headers)), lite())[1]
    providers.DDGSSearchProvider().search("q", 5)
    assert sent and all("proxy-authorization" not in h for h in sent)


def test_provider_selection(configure):
    configure(search_provider="duckduckgo_lite", enable_web_search=True)
    assert providers.get_search_provider().name == "duckduckgo_lite"
    configure(search_provider="ddgs")
    assert providers.get_search_provider().name == "ddgs"
    configure(enable_web_search=False)
    assert providers.get_search_provider().name == "none"


# ---------------------------------------------------------------- notices
def test_refused_search_notice_is_plain_and_points_to_approved_options():
    web = WebAccessState()
    web.record_search(policy.SEARCH_REFUSED)
    out = notices.network_notices(web)
    assert len(out) == 1 and out[0].startswith(notices.SEARCH_REFUSED_MESSAGE) and "not worked around" in out[0]
    assert analyst_notices(out) == out


def test_search_signin_notice_names_the_supported_providers(configure):
    configure(web_proxy_auth="integrated")
    web = WebAccessState()
    web.record_search(policy.PROXY_AUTH_REQUIRED)
    out = notices.network_notices(web)
    assert notices.SEARCH_SIGNIN_MESSAGE in out and "duckduckgo_lite" in notices.SEARCH_SIGNIN_MESSAGE


# ---------------------------------------------------------------- network_check: components checked independently
@pytest.fixture()
def checks(monkeypatch):
    ran = []
    monkeypatch.setattr(network_check, "check_configuration", lambda: ran.append("config"))
    monkeypatch.setattr(network_check, "check_route", lambda url: ran.append("route"))
    monkeypatch.setattr(network_check, "check_proxy_signin", lambda proxy, url: ran.append("signin"))
    monkeypatch.setattr(network_check, "check_local_model", lambda: ran.append("model"))
    monkeypatch.setattr(network_check, "check_sustained_search", lambda: ran.append("sustained") or True)
    results = {"fetch": True, "search": True}

    def fetch_check(url):
        ran.append("fetch")
        return results["fetch"]

    def search_check():
        ran.append("search")
        return results["search"]

    monkeypatch.setattr(network_check, "check_fetch", fetch_check)
    monkeypatch.setattr(network_check, "check_search", search_check)
    return ran, results


def test_network_check_runs_each_component_on_its_own(checks, capsys):
    ran, results = checks
    assert network_check.main(["x", "--search-only"]) == 0
    assert "fetch" not in ran and "search" in ran
    ran.clear()
    assert network_check.main(["x", "--retrieval-only", "https://www.company.example/"]) == 0
    assert "fetch" in ran and "search" not in ran


def test_network_check_reports_each_component_separately(checks, capsys):
    _, results = checks
    results["search"] = False
    assert network_check.main(["x"]) == 1
    out = capsys.readouterr().out
    assert "Website retrieval          OK" in out and "Web search (one query)     FAILED" in out
    assert "Websites can be read, but search does not work" in out


def test_network_check_shows_the_search_transport(signin, monkeypatch, configure, capsys, no_ddgs_library):
    configure(search_provider="ddgs", enable_web_search=True)
    proxy = FakeProxy(monkeypatch)
    proxy.handler = lambda request: lite()
    assert network_check.check_search() is True
    out = capsys.readouterr().out
    assert "application transport (PAC + Windows sign-in)" in out and "cannot sign in to a proxy" in out
    assert TOKEN not in out
