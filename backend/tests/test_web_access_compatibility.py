"""Web access on managed networks: approved proxies, the organisation's certificates, clear reasons when the network
refuses a request, early stop instead of long waits, and manual upload as the way forward.
The SSRF, redirect and timeout protections stay exactly as strict. Deterministic: nothing connects anywhere."""
from __future__ import annotations

import socket
import ssl
import urllib.request

import certifi
import httpcore
import httpx
import pytest

from app.services.llm.base import LLMProvider, LLMStatus
from app.services.pipeline import network_notices as notices
from app.services.pipeline.discovery import WebAccessState
from app.services.presentation import analyst_notices
from app.services.search import providers
from app.services.search.base import SearchProvider
from app.services.security import network, policy, safe_fetch
from app.services.security.safe_fetch import FetchError, fetch
from tests.conftest import FakeLLM
from tests.test_security import (  # noqa: F401 — fixtures used by name
    HTML, PUBLIC_IP, SITE, SpySearch, _assessment, _events, _no_sleep_no_scanner, _run, _sources, client_for, configure, dns,
    pdf_response,
)

PROXY = "http://proxy.bank.example:8080"
BLUECOAT_AUTH_PAGE = (b"<html><head><title>Access Denied</title></head><body><h1>Access Denied (authentication_failed)</h1>"
                      b"<p>Credentials are missing.</p><p>Blue Coat Web Filter</p></body></html>")
BLUECOAT_POLICY_PAGE = (b"<html><body><h1>Access Denied (policy_denied)</h1><p>Your organisation's policy does not allow "
                        b"access to this site.</p><p>Blue Coat ProxySG</p></body></html>")


def html_response(body=HTML, status=200) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": "text/html"}, content=body)


@pytest.fixture()
def via_proxy(monkeypatch):
    monkeypatch.setattr(network, "proxy_for", lambda url: PROXY)


@pytest.fixture()
def no_local_dns(monkeypatch):
    def fail(*_a, **_k):
        raise socket.gaierror("name not known on this network")
    monkeypatch.setattr(socket, "getaddrinfo", fail)


# ---------------------------------------------------------------- proxy configuration
def test_proxy_modes(configure, monkeypatch):
    monkeypatch.setattr(network, "_system_proxies", lambda: {"https": PROXY, "http": PROXY})
    monkeypatch.setattr(network, "_system_bypass", lambda host: False)
    configure(web_proxy_mode="none")
    assert network.proxy_for("https://www.company.example/") is None
    configure(web_proxy_mode="system")
    assert network.proxy_for("https://www.company.example/") == PROXY
    configure(web_proxy_mode="explicit", web_proxy_url="http://other.example:3128")
    assert network.proxy_for("https://www.company.example/") == "http://other.example:3128"
    configure(web_proxy_mode="explicit", web_proxy_url="")
    assert network.proxy_for("https://www.company.example/") is None


def test_system_proxy_respects_the_bypass_list(configure, monkeypatch):
    configure(web_proxy_mode="system")
    monkeypatch.setattr(network, "_system_proxies", lambda: {"https": PROXY})
    monkeypatch.setattr(network, "_system_bypass", lambda host: host.endswith(".bank.example"))
    assert network.proxy_for("https://intranet.bank.example/") is None
    assert network.proxy_for("https://www.company.example/") == PROXY


def test_system_mode_reads_environment_and_windows_settings_via_the_standard_library(configure, monkeypatch):
    configure(web_proxy_mode="system")
    monkeypatch.setattr(urllib.request, "getproxies", lambda: {"https": PROXY})
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda host: False)
    assert network.proxy_for("https://www.company.example/") == PROXY


def test_unsupported_proxy_types_are_not_used(configure, monkeypatch):
    configure(web_proxy_mode="system")
    monkeypatch.setattr(network, "_system_proxies", lambda: {"https": "socks4://proxy.example:1080"})
    assert network.proxy_for("https://www.company.example/") is None


def test_proxy_credentials_are_never_shown(configure):
    assert network.mask_proxy("http://user:s3cret@proxy.example:8080") == "http://***@proxy.example:8080"
    assert network.mask_proxy(None) == "none"
    configure(web_proxy_mode="explicit", web_proxy_url="http://user:s3cret@proxy.example:8080")
    assert "s3cret" not in str(network.describe())


def test_client_is_built_for_the_chosen_proxy_without_environment_lookups():
    with safe_fetch.build_client(PROXY) as client:
        assert client._trust_env is False
        assert any(isinstance(getattr(t, "_pool", None), httpcore.HTTPProxy) for t in client._mounts.values())
    with safe_fetch.build_client() as client:
        assert not client._mounts


# ---------------------------------------------------------------- certificates: always verified
@pytest.mark.parametrize("settings", [
    {"web_ca_bundle": "", "use_system_trust_store": True},
    {"web_ca_bundle": "", "use_system_trust_store": False},
    {"web_ca_bundle": certifi.where(), "use_system_trust_store": True},
])
def test_certificates_are_always_verified(configure, settings):
    configure(**settings)
    ctx = network.ssl_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname


def test_the_operating_system_store_is_used_by_default(configure):
    configure(web_ca_bundle="", use_system_trust_store=True)
    assert network.trust_source() == "system"
    configure(web_ca_bundle=certifi.where())
    assert network.trust_source() == "ca_bundle"


def test_an_unusable_ca_bundle_is_reported_and_nothing_is_requested(configure, tmp_path, dns):
    configure(web_ca_bundle=str(tmp_path / "missing.pem"))
    with pytest.raises(FetchError) as e:
        fetch("https://www.company.example/")
    assert e.value.code == policy.TLS_VERIFICATION_FAILED


def test_certificate_failure_is_reported_and_not_retried(dns):
    calls = {"n": 0}

    def handler(_r):
        calls["n"] += 1
        try:
            raise ssl.SSLCertVerificationError("certificate verify failed: unable to get local issuer certificate")
        except ssl.SSLError as exc:
            raise httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED]") from exc

    with client_for(handler) as client, pytest.raises(FetchError) as e:
        fetch("https://www.company.example/", client=client)
    assert e.value.code == policy.TLS_VERIFICATION_FAILED and calls["n"] == 1


# ---------------------------------------------------------------- through a proxy: SSRF protection unchanged
def test_proxied_request_goes_by_hostname_when_the_name_only_resolves_at_the_proxy(via_proxy, no_local_dns):
    seen = []

    def handler(request):
        seen.append(request)
        return html_response()

    with client_for(handler) as client:
        res = fetch("https://www.company.example/sustainability", client=client)
    assert res.kind == "html"
    assert seen[0].url.host == "www.company.example"  # the proxy connects; nothing to pin


def test_direct_request_is_still_pinned_and_needs_local_dns(no_local_dns):
    with client_for(lambda r: html_response()) as client, pytest.raises(FetchError) as e:
        fetch("https://www.company.example/", client=client)
    assert e.value.code == policy.DNS_RESOLUTION_FAILED


@pytest.mark.parametrize("url,code", [
    ("http://127.0.0.1/", policy.PRIVATE_IP),
    ("http://169.254.169.254/latest/meta-data", policy.PRIVATE_IP),
    ("http://localhost/", policy.UNSAFE_HOST),
    ("https://intranet.corp/", policy.UNSAFE_HOST),
    ("https://user:pw@www.company.example/", policy.CREDENTIALS_IN_URL),
    ("https://www.company.example:8443/", policy.UNSAFE_PORT),
    ("file:///etc/passwd", policy.UNSAFE_SCHEME),
])
def test_proxy_route_keeps_every_url_check(url, code, via_proxy, dns):
    def never(_r):
        raise AssertionError("must not be requested")

    with client_for(never) as client, pytest.raises(FetchError) as e:
        fetch(url, client=client)
    assert e.value.code == code


def test_proxy_route_still_rejects_names_that_resolve_to_private_addresses(via_proxy, dns):
    dns["sneaky.example"] = ["10.0.0.5"]
    with client_for(lambda r: html_response()) as client, pytest.raises(FetchError) as e:
        fetch("https://sneaky.example/", client=client)
    assert e.value.code == policy.PRIVATE_IP


def test_proxy_route_still_validates_redirect_targets(via_proxy, no_local_dns):
    def handler(request):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data"})

    with client_for(handler) as client, pytest.raises(FetchError) as e:
        fetch("https://www.company.example/report.pdf", client=client)
    assert e.value.code == policy.PRIVATE_IP and e.value.redirected


# ---------------------------------------------------------------- refusals by the network, told apart
def test_proxy_sign_in_required_on_connect(via_proxy, no_local_dns):
    calls = {"n": 0}

    def handler(_r):
        calls["n"] += 1
        raise httpx.ProxyError("407 Proxy Authentication Required")

    with client_for(handler) as client, pytest.raises(FetchError) as e:
        fetch("https://www.company.example/", client=client)
    assert e.value.code == policy.PROXY_AUTH_REQUIRED and calls["n"] == 1  # never retried
    assert not policy.is_security_rejection(e.value.code)


def test_proxy_sign_in_required_as_http_response(dns):
    with client_for(lambda r: httpx.Response(407)) as client, pytest.raises(FetchError) as e:
        fetch("http://www.company.example/", client=client)
    assert e.value.code == policy.PROXY_AUTH_REQUIRED


@pytest.mark.parametrize("status,body,code", [
    (403, BLUECOAT_AUTH_PAGE, policy.PROXY_AUTH_REQUIRED),
    (403, BLUECOAT_POLICY_PAGE, policy.BLOCKED_BY_WEB_FILTER),
    (200, BLUECOAT_AUTH_PAGE, policy.PROXY_AUTH_REQUIRED),
    (200, BLUECOAT_POLICY_PAGE, policy.BLOCKED_BY_WEB_FILTER),
])
def test_web_filter_pages_are_recognised(status, body, code, dns):
    with client_for(lambda r: html_response(body, status)) as client, pytest.raises(FetchError) as e:
        fetch("https://www.company.example/", client=client)
    assert e.value.code == code and not e.value.retryable


def test_ordinary_errors_and_articles_are_not_mistaken_for_filter_pages(dns):
    with client_for(lambda r: html_response(b"<html><body>Forbidden</body></html>", 403)) as client, pytest.raises(FetchError) as e:
        fetch("https://www.company.example/", client=client)
    assert e.value.code == policy.HTTP_ERROR
    article = b"<html><body><p>Our web filter and Zscaler rollout blocked phishing.</p>" + b"<p>Sustainability.</p>" * 50 + b"</body></html>"
    with client_for(lambda r: html_response(article)) as client:
        assert fetch("https://www.company.example/", client=client).kind == "html"


def test_unreachable_proxy_is_reported(via_proxy, no_local_dns):
    with client_for(lambda r: (_ for _ in ()).throw(httpx.ConnectError("connection refused"))) as client, \
            pytest.raises(FetchError) as e:
        fetch("https://www.company.example/", client=client)
    assert e.value.code == policy.PROXY_CONNECTION_FAILED


@pytest.mark.parametrize("exc,code", [
    (Exception("407 Proxy Authentication Required"), policy.PROXY_AUTH_REQUIRED),
    (Exception("error sending request: invalid peer certificate: UnknownIssuer"), policy.TLS_VERIFICATION_FAILED),
    (Exception("proxy connection refused"), policy.PROXY_CONNECTION_FAILED),
    (TimeoutError(), policy.SOURCE_TIMEOUT),
    (Exception("connection reset"), policy.NETWORK_ERROR),
])
def test_search_errors_are_classified(exc, code):
    assert network.classify_search_error(exc) == code


# ---------------------------------------------------------------- search clients use the same settings
def test_ddgs_gets_the_approved_proxy_and_ca_bundle(configure, monkeypatch):
    import ddgs

    seen = {}

    class FakeDDGS:
        def __init__(self, proxy=None, timeout=None, *, verify=True):
            seen.update(proxy=proxy, verify=verify)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results):
            return [{"title": "t", "href": "https://www.company.example/", "body": ""}]

    monkeypatch.setattr(ddgs, "DDGS", FakeDDGS)
    configure(web_proxy_mode="explicit", web_proxy_url=PROXY, web_ca_bundle=certifi.where())
    provider = providers.DDGSSearchProvider()
    assert len(provider.search("x", 3)) == 1
    assert seen == {"proxy": PROXY, "verify": certifi.where()} and provider.last_error is None


def test_failed_search_records_why(monkeypatch):
    import ddgs

    class Refused:
        def __init__(self, *a, **k):
            raise RuntimeError("407 Proxy Authentication Required")

    monkeypatch.setattr(ddgs, "DDGS", Refused)
    provider = providers.DDGSSearchProvider()
    assert provider.search("x", 3) == [] and provider.last_error == policy.PROXY_AUTH_REQUIRED


def test_local_ollama_never_goes_through_a_proxy(monkeypatch):
    from app.services.llm.ollama_provider import OllamaProvider

    seen = {}

    def get(url, **kw):
        seen.update(kw)
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", get)
    OllamaProvider("http://localhost:11434", "llama3.2:1b").status()
    assert seen.get("trust_env") is False


# ---------------------------------------------------------------- stop early instead of waiting
def test_access_stops_after_repeated_failures_when_nothing_got_through():
    web = WebAccessState(limit=3)
    for _ in range(2):
        web.record_fetch(policy.PROXY_AUTH_REQUIRED)
    assert not web.fetch_stopped
    web.record_fetch(policy.NETWORK_ERROR)
    assert web.fetch_stopped


def test_access_continues_once_something_worked_and_filters_do_not_count():
    web = WebAccessState(limit=2)
    web.record_fetch(None)
    for _ in range(5):
        web.record_fetch(policy.SOURCE_TIMEOUT)
    assert not web.fetch_stopped
    filtered = WebAccessState(limit=2)
    for _ in range(5):
        filtered.record_fetch(policy.BLOCKED_BY_WEB_FILTER)
    assert not filtered.fetch_stopped
    unlimited = WebAccessState(limit=0)
    for _ in range(10):
        unlimited.record_fetch(policy.NETWORK_ERROR)
        unlimited.record_search(policy.NETWORK_ERROR)
    assert not unlimited.fetch_stopped and not unlimited.search_stopped


class FailingSearch(SearchProvider):
    name = "failing"

    def __init__(self, code: str):
        self.code = code
        self.queries: list[str] = []

    def search(self, query, max_results):
        self.queries.append(query)
        self.failed_searches += 1
        self.last_error = self.code
        return []


def test_assessment_with_refused_web_access_completes_quickly_with_a_clear_reason(client, dns, monkeypatch, configure):
    """The bank-laptop case: the proxy wants sign-in for everything. The assessment does not hang; it says why."""
    configure(web_access_failure_limit=3)
    requests = []

    def handler(request):
        requests.append(request)
        return html_response(BLUECOAT_AUTH_PAGE, 403)

    monkeypatch.setattr(safe_fetch, "build_client", lambda: client_for(handler))
    aid = _assessment(client, website=SITE)
    search = FailingSearch(policy.PROXY_AUTH_REQUIRED)
    st = _run(client, aid, search)

    assert st["status"] == "COMPLETED"
    assert len(search.queries) == 3, "searching stops after the failure limit, also in the recovery pass"
    assert len(requests) == 1, "only the company website was attempted"
    assert notices.PROXY_AUTH_MESSAGE in st["notices"]
    # search hit its failure limit; website retrieval was not stopped (it was refused once), so only search is named
    assert notices.SEARCH_STOPPED_MESSAGE in st["notices"] and notices.STOPPED_MESSAGE not in st["notices"]
    assert any("bypass" in n for n in st["notices"])
    diag = next(e for e in _events(aid) if e.event_type == "web_access_diagnostics")
    assert diag.details["search_stopped"] is True and diag.details["fetch_issues"] == {policy.PROXY_AUTH_REQUIRED: 1}


def test_fetching_search_results_stops_after_repeated_network_failures(client, dns, monkeypatch, configure):
    configure(web_access_failure_limit=2, max_downloaded_pdfs=10, max_pages_per_site=10)
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ConnectError("blocked")

    monkeypatch.setattr(safe_fetch, "build_client", lambda: client_for(handler))
    urls = [f"https://www.nordvikcomponents.example/report-{i}-2025.pdf" for i in range(6)]
    aid = _assessment(client)
    st = _run(client, aid, SpySearch(urls))
    assert st["status"] == "COMPLETED"
    attempted = {s.url for s in _sources(aid).values() if s.fetch_outcome}
    assert len(attempted) == 2
    assert notices.STOPPED_MESSAGE in st["notices"]


def test_manual_upload_continues_the_assessment_when_web_access_fails(client, dns, monkeypatch, sample_pdf):
    monkeypatch.setattr(safe_fetch, "build_client", lambda: client_for(lambda r: html_response(BLUECOAT_AUTH_PAGE, 403)))
    aid = _assessment(client, website=SITE)
    with open(sample_pdf, "rb") as f:
        assert client.post(f"/api/assessments/{aid}/documents", files={"file": ("report.pdf", f, "application/pdf")}).status_code == 201
    llm = FakeLLM()
    st = _run(client, aid, SpySearch(), llm)
    assert st["status"] == "COMPLETED" and llm.calls
    uploaded = [s for s in _sources(aid).values() if s.source_type == "uploaded_report"]
    assert uploaded and uploaded[0].status == "PROCESSED"
    assert notices.PROXY_AUTH_MESSAGE in st["notices"]
    recs = client.get(f"/api/assessments/{aid}/recommendations").json()
    assert any((r["recommended_level"] or 0) > 0 for r in recs)


def test_limit_notices_name_the_setting_to_change():
    web = WebAccessState()
    web.issues[policy.DOWNLOAD_SIZE_LIMIT_EXCEEDED] += 1
    web.issues[policy.PDF_PAGE_LIMIT_EXCEEDED] += 1
    out = notices.network_notices(web)
    assert notices.SIZE_LIMIT_MESSAGE in out and notices.PAGE_LIMIT_MESSAGE in out


def test_no_proxy_and_nothing_reachable_suggests_configuring_one(configure, monkeypatch):
    configure(web_proxy_mode="none")
    monkeypatch.setattr(network, "windows_autoconfig_url", lambda: "http://wpad.bank.example/proxy.pac")
    web = WebAccessState()
    web.record_fetch(policy.NETWORK_ERROR)
    web.record_fetch(policy.SOURCE_TIMEOUT)
    out = notices.network_notices(web)
    assert out and out[0].startswith(notices.DIRECT_UNREACHABLE_MESSAGE) and "configuration script" in out[0]


def test_network_notices_reach_the_analyst_unchanged():
    msgs = [notices.PROXY_AUTH_MESSAGE, notices.TLS_MESSAGE, notices.WEB_FILTER_MESSAGE, notices.DIRECT_UNREACHABLE_MESSAGE,
            notices.SEARCH_MESSAGE, notices.SIZE_LIMIT_MESSAGE, notices.PAGE_LIMIT_MESSAGE]
    assert analyst_notices(msgs) == msgs


def test_successful_access_adds_no_network_notice():
    web = WebAccessState(limit=3)
    web.record_fetch(None)
    web.record_search(None)
    assert notices.network_notices(web) == []


# ---------------------------------------------------------------- first stage cannot hang or crash on the model check
class BrokenStatusLLM(LLMProvider):
    provider_name = "llama_cpp"

    @property
    def model_name(self):
        return "llama_cpp:model.gguf"

    def status(self) -> LLMStatus:
        raise OSError("DLL load failed while importing llama_cpp: blocked by policy")

    def complete_json(self, *a, **k):
        raise AssertionError("must not be used")


def test_assessment_continues_when_the_local_model_library_cannot_load(client, configure, sample_pdf):
    configure(enable_web_search=False, enable_external_fetch=False, allow_heuristic_fallback=True)
    aid = _assessment(client)
    with open(sample_pdf, "rb") as f:
        client.post(f"/api/assessments/{aid}/documents", files={"file": ("report.pdf", f, "application/pdf")})
    from app.services.pipeline.runner import run_assessment

    run_assessment(aid, llm=BrokenStatusLLM(), search=SpySearch())
    st = client.get(f"/api/assessments/{aid}/status").json()
    assert st["status"] == "COMPLETED"


def test_network_check_reports_a_refusal_without_credentials(configure, monkeypatch, capsys, via_proxy, no_local_dns):
    from app import network_check

    configure(web_proxy_mode="explicit", web_proxy_url="http://user:s3cret@proxy.bank.example:8080", search_provider="none")
    monkeypatch.setattr(safe_fetch, "build_client", lambda *a: client_for(lambda r: html_response(BLUECOAT_AUTH_PAGE, 403)))
    monkeypatch.setattr(network_check, "check_local_model", lambda: None)
    assert network_check.main(["network_check", "https://www.example.com/"]) == 1
    out = capsys.readouterr().out
    assert policy.PROXY_AUTH_REQUIRED in out and "s3cret" not in out
    assert "upload" in out.lower()
