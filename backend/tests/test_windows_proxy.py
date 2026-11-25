"""Windows system proxy support: the PAC script / WPAD answer as Windows evaluates it, and integrated proxy sign-in
with a Kerberos ticket from the signed-in Windows session (opt-in, only after the proxy asks, never NTLM, no
credentials stored). The SSRF, certificate and redirect protections are unchanged.

Most tests are deterministic and connect nowhere. Tests marked windows_only call the real WinHTTP / SSPI functions
against a PAC script served on 127.0.0.1, and still send nothing to any proxy."""
from __future__ import annotations

import base64
import http.server
import logging
import sys
import threading

import httpx
import pytest

from app.services.pipeline import network_notices as notices
from app.services.pipeline.discovery import WebAccessState
from app.services.presentation import analyst_notices
from app.services.search import providers
from app.services.security import network, policy, safe_fetch, windows_proxy
from app.services.security.safe_fetch import FetchError, fetch
from app.services.security.windows_proxy import IntegratedAuthUnavailable, PacResult, WindowsProxyConfig
from tests.test_security import HTML, PUBLIC_IP, _no_sleep_no_scanner, configure, dns  # noqa: F401 — fixtures

PROXY = "http://proxy.corp.example:8080"
TOKEN = base64.b64encode(b"kerberos-ticket").decode()
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="uses Windows WinHTTP / SSPI")


def html_response(status=200, body=HTML, headers=None) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": "text/html", **(headers or {})}, content=body)


@pytest.fixture()
def windows_pac(configure, monkeypatch):
    """System mode with a PAC script in Windows Internet Options; the evaluator is replaced by `answers`."""
    configure(web_proxy_mode="system", web_proxy_pac_url="")
    monkeypatch.setattr(network, "_environment_proxies", lambda: {})
    monkeypatch.setattr(network, "_system_proxies", lambda: {})
    monkeypatch.setattr(network, "windows_config", lambda: WindowsProxyConfig(
        auto_detect=False, pac_url="http://pac.corp.example/proxy.pac", static_proxy=None, bypass=None))
    calls: list[str] = []
    answers: dict[str, PacResult] = {}

    def evaluate(url, *, pac_url, auto_detect, timeout_seconds):
        calls.append(url)
        host = httpx.URL(url).host
        return answers.get(host, PacResult(True, ("proxy.corp.example:8080",)))

    monkeypatch.setattr(windows_proxy, "evaluate_pac", evaluate)
    answers["__calls__"] = calls  # type: ignore[assignment]
    return answers


# ---------------------------------------------------------------- PAC: Windows decides the route
@pytest.mark.parametrize("value,scheme,expected", [
    ("proxy.corp.example:8080", "https", ("proxy.corp.example:8080",)),
    ("a.example:8080;b.example:3128", "https", ("a.example:8080", "b.example:3128")),
    ("PROXY a.example:8080; PROXY b.example:8080; DIRECT", "https", ("a.example:8080", "b.example:8080")),
    ("http=h.example:80;https=s.example:443", "https", ("s.example:443",)),
    ("http=h.example:80;https=s.example:443", "http", ("h.example:80",)),
    ("SOCKS s.example:1080; PROXY p.example:8080", "https", ("p.example:8080",)),
    ("DIRECT", "https", ()),
    ("", "https", ()),
])
def test_proxy_lists_from_windows_are_read_in_order(value, scheme, expected):
    assert windows_proxy.parse_proxy_list(value, scheme) == expected


def test_pac_answer_decides_the_route(windows_pac):
    windows_pac["intranet.corp.example"] = PacResult(True)  # DIRECT
    assert network.proxy_for("https://www.company.example/report.pdf") == PROXY
    assert network.proxy_for("https://intranet.corp.example/") is None
    assert network.proxy_source() == "windows_pac"


def test_pac_answer_is_cached_per_host(windows_pac):
    for path in ("/a", "/b", "/c"):
        network.proxy_for(f"https://www.company.example{path}")
    network.proxy_for("https://cdn.company.example/x.pdf")
    assert [httpx.URL(u).host for u in windows_pac["__calls__"]] == ["www.company.example", "cdn.company.example"]


def test_failed_pac_falls_back_to_the_static_windows_proxy_like_windows(windows_pac, monkeypatch):
    windows_pac["www.company.example"] = PacResult(False, error="the PAC script could not be downloaded")
    monkeypatch.setattr(network, "_system_proxies", lambda: {"https": "http://static.corp.example:3128"})
    monkeypatch.setattr(network, "_system_bypass", lambda host: False)
    assert network.proxy_for("https://www.company.example/") == "http://static.corp.example:3128"
    assert network.pac_failed() and "could not be downloaded" in network.pac_status()["pac_last_error"]


def test_wpad_finding_nothing_is_not_a_failure(configure, monkeypatch):
    configure(web_proxy_mode="system", web_proxy_pac_url="")
    monkeypatch.setattr(network, "_environment_proxies", lambda: {})
    monkeypatch.setattr(network, "_system_proxies", lambda: {})
    monkeypatch.setattr(network, "windows_config", lambda: WindowsProxyConfig(True, None, None, None))
    monkeypatch.setattr(windows_proxy, "evaluate_pac", lambda *a, **k: PacResult(False, error="none", not_found=True))
    assert network.proxy_for("https://www.company.example/") is None
    assert not network.pac_failed() and network.proxy_source() == "windows_auto_detect"


def test_environment_variables_take_precedence_over_the_pac_script(windows_pac, monkeypatch):
    monkeypatch.setattr(network, "_environment_proxies", lambda: {"https": "http://env.corp.example:8080"})
    monkeypatch.setattr(network, "_system_proxies", lambda: {"https": "http://env.corp.example:8080"})
    monkeypatch.setattr(network, "_system_bypass", lambda host: False)
    assert network.proxy_for("https://www.company.example/") == "http://env.corp.example:8080"
    assert windows_pac["__calls__"] == []


@pytest.mark.parametrize("mode", ["none", "explicit"])
def test_pac_is_only_evaluated_in_system_mode(mode, windows_pac, configure):
    configure(web_proxy_mode=mode, web_proxy_url="http://explicit.corp.example:8080")
    network.proxy_for("https://www.company.example/")
    assert windows_pac["__calls__"] == []


def test_configured_pac_url_replaces_the_windows_setting(windows_pac, configure):
    configure(web_proxy_pac_url="http://other-pac.corp.example/p.pac")
    assert network.pac_configuration() == ("http://other-pac.corp.example/p.pac", False)


def test_pac_routed_requests_keep_every_ssrf_check(windows_pac, dns):
    """The PAC can route a request, never permit one: a local or private target is refused before any request."""
    dns["internal.company.example"] = ["10.0.0.5"]
    for url in ("http://127.0.0.1/", "http://169.254.169.254/latest/meta-data/", "https://internal.company.example/"):
        with pytest.raises(FetchError) as e:
            fetch(url)
        assert e.value.code in (policy.PRIVATE_IP, policy.UNSAFE_HOST)


def test_pac_status_masks_proxy_credentials(windows_pac):
    windows_pac["www.company.example"] = PacResult(True, ("user:secret@proxy.corp.example:8080",))
    network.proxy_for("https://www.company.example/")
    assert "secret" not in str(network.describe())


# ---------------------------------------------------------------- integrated sign-in in page retrieval
class FakeProxy:
    """Stands in for build_client: a proxy that wants a Negotiate sign-in, recorded per connection."""

    def __init__(self, monkeypatch, *, accept=True, connect_407=True):
        self.accept, self.connect_407 = accept, connect_407
        self.built: list[dict | None] = []
        self.seen: list[tuple[str, bool]] = []
        self.handler = lambda request: html_response()
        monkeypatch.setattr(safe_fetch, "build_client", self.build)

    def build(self, proxy=None, proxy_headers=None):
        self.built.append(proxy_headers)
        signed = bool(proxy_headers) and proxy_headers.get("Proxy-Authorization", "").startswith("Negotiate ")

        def transport(request: httpx.Request) -> httpx.Response:
            self.seen.append((str(request.url), signed))
            if proxy and not (signed and self.accept):
                if self.connect_407:
                    raise httpx.ProxyError("407 Proxy Authentication Required")
                return httpx.Response(407, headers={"proxy-authenticate": "Negotiate"})
            return self.handler(request)

        return httpx.Client(transport=httpx.MockTransport(transport), follow_redirects=False)


@pytest.fixture()
def signin(configure, monkeypatch, dns):
    configure(web_proxy_auth="integrated")
    monkeypatch.setattr(network, "proxy_for", lambda url: PROXY)
    issued: list[str] = []

    def token(host):
        issued.append(host)
        return f"{TOKEN}{len(issued)}"

    monkeypatch.setattr(windows_proxy, "kerberos_proxy_token", token)
    return issued


@pytest.mark.parametrize("connect_407", [True, False], ids=["https-connect", "http-forward"])
def test_integrated_signin_after_the_proxy_asks(signin, monkeypatch, connect_407):
    proxy = FakeProxy(monkeypatch, connect_407=connect_407)
    assert fetch("https://www.company.example/").kind == "html"
    assert signin == ["proxy.corp.example"]  # Kerberos service name HTTP/<proxy host>
    assert proxy.built[0] is None and proxy.built[1]["Proxy-Authorization"] == f"Negotiate {TOKEN}1"


def test_no_signin_is_sent_unless_the_proxy_asks(signin, monkeypatch):
    built = []

    def build(proxy_url=None, proxy_headers=None):
        built.append(proxy_headers)
        return httpx.Client(transport=httpx.MockTransport(lambda r: html_response()))

    monkeypatch.setattr(safe_fetch, "build_client", build)
    fetch("https://www.company.example/")
    assert signin == [] and built == [None]


def test_each_connection_gets_a_fresh_ticket_and_cookies_survive_the_chain(signin, monkeypatch):
    proxy = FakeProxy(monkeypatch)

    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://cdn.company.example/doc", "set-cookie": "c=1; Domain=company.example; Path=/"})
        assert "c=1" in request.headers.get("cookie", "")
        return html_response()

    proxy.handler = handler
    assert fetch("https://www.company.example/start").final_url == "https://cdn.company.example/doc"
    assert signin == ["proxy.corp.example", "proxy.corp.example"]  # one ticket per hop: no replayed tickets
    assert [h["Proxy-Authorization"] for h in proxy.built if h] == [f"Negotiate {TOKEN}1", f"Negotiate {TOKEN}2"]


def test_later_fetches_sign_in_at_once(signin, monkeypatch):
    proxy = FakeProxy(monkeypatch)
    fetch("https://www.company.example/")
    proxy.built.clear()
    fetch("https://www.company.example/other")
    assert proxy.built == [{"Proxy-Authorization": f"Negotiate {TOKEN}2"}]


def test_signin_off_by_default_reports_that_the_proxy_requires_it(configure, monkeypatch, dns):
    configure(web_proxy_auth="none")
    monkeypatch.setattr(network, "proxy_for", lambda url: PROXY)
    called = []
    monkeypatch.setattr(windows_proxy, "kerberos_proxy_token", lambda host: called.append(host))
    FakeProxy(monkeypatch)
    with pytest.raises(FetchError) as e:
        fetch("https://www.company.example/")
    assert e.value.code == policy.PROXY_AUTH_REQUIRED and called == []


def test_refused_signin_is_reported_and_not_retried(signin, monkeypatch):
    proxy = FakeProxy(monkeypatch, accept=False)
    with pytest.raises(FetchError) as e:
        fetch("https://www.company.example/")
    assert e.value.code == policy.PROXY_SIGNIN_REFUSED and not e.value.retryable
    assert len(signin) == 1 and len(proxy.built) == 2


def test_unavailable_kerberos_is_reported_with_its_reason(signin, monkeypatch):
    def unavailable(host):
        raise IntegratedAuthUnavailable("Windows could only offer NTLM for this proxy, not Kerberos")

    monkeypatch.setattr(windows_proxy, "kerberos_proxy_token", unavailable)
    FakeProxy(monkeypatch)
    with pytest.raises(FetchError) as e:
        fetch("https://www.company.example/")
    assert e.value.code == policy.PROXY_SIGNIN_UNAVAILABLE
    assert "NTLM" in network.last_signin_problem()


def test_the_signin_is_never_logged_or_sent_to_the_site(signin, monkeypatch, caplog):
    proxy = FakeProxy(monkeypatch)
    sent_headers = []
    proxy.handler = lambda r: (sent_headers.append(dict(r.headers)), html_response())[1]
    with caplog.at_level(logging.DEBUG):
        fetch("https://www.company.example/")
    assert TOKEN not in caplog.text
    assert all("proxy-authorization" not in h for h in sent_headers)


def test_signin_only_for_http_proxies(configure):
    configure(web_proxy_auth="integrated")
    assert network.integrated_signin_enabled(PROXY)
    assert not network.integrated_signin_enabled("socks5://proxy.corp.example:1080")
    assert not network.integrated_signin_enabled(None)
    configure(web_proxy_auth="none")
    assert not network.integrated_signin_enabled(PROXY)


def test_real_client_puts_the_signin_on_the_proxy_only():
    client = safe_fetch.build_client(PROXY, {"Proxy-Authorization": f"Negotiate {TOKEN}"})
    try:
        assert "proxy-authorization" not in client.headers
        assert client._transport_for_url(httpx.URL("https://www.company.example/")) is not client._transport
    finally:
        client.close()


# ---------------------------------------------------------------- search
def test_serpapi_signs_in_with_the_same_integrated_signin(configure, monkeypatch):
    configure(web_proxy_auth="integrated")
    monkeypatch.setattr(network, "proxy_for", lambda url: PROXY)
    monkeypatch.setattr(windows_proxy, "kerberos_proxy_token", lambda host: TOKEN)
    calls = []

    def get(url, *, proxy=None, **kw):
        signed = isinstance(proxy, httpx.Proxy) and "proxy-authorization" in proxy.headers
        calls.append(signed)
        if not signed:
            raise httpx.ProxyError("407 Proxy Authentication Required")
        return httpx.Response(200, json={"organic_results": [{"title": "t", "link": "https://x.example/", "snippet": ""}]},
                              request=httpx.Request("GET", url))

    monkeypatch.setattr(providers.httpx, "get", get)
    p = providers.SerpAPISearchProvider("key")
    assert len(p.search("q", 3)) == 1 and calls == [False, True] and p.last_error is None


def test_search_signin_failures_are_classified():
    assert network.classify_search_error(IntegratedAuthUnavailable("x")) == policy.PROXY_SIGNIN_UNAVAILABLE
    assert network.classify_search_error(httpx.ProxyError("407 integrated sign-in refused")) == policy.PROXY_SIGNIN_REFUSED
    assert network.classify_search_error(httpx.ProxyError("407 Proxy Authentication Required")) == policy.PROXY_AUTH_REQUIRED


# ---------------------------------------------------------------- analyst notices
def test_proxy_auth_notice_names_the_approved_option(configure):
    configure(web_proxy_auth="none")
    web = WebAccessState()
    web.record_fetch(policy.PROXY_AUTH_REQUIRED)
    out = notices.network_notices(web)
    assert out[0] == notices.PROXY_AUTH_MESSAGE and "WEB_PROXY_AUTH=integrated" in out[0]


def test_signin_notices(configure):
    configure(web_proxy_auth="integrated")
    network.note_signin_problem("the proxy is given as an IP address")
    web = WebAccessState()
    web.record_fetch(policy.PROXY_SIGNIN_UNAVAILABLE)
    web.record_fetch(policy.PROXY_SIGNIN_REFUSED)
    web.record_search(policy.PROXY_AUTH_REQUIRED)
    out = notices.network_notices(web)
    assert "IP address" in out[0] and out[1] == notices.SIGNIN_REFUSED_MESSAGE and notices.SEARCH_SIGNIN_MESSAGE in out
    assert notices.PROXY_AUTH_MESSAGE not in out
    assert analyst_notices(out) == out  # plain language: not collapsed into a technical notice


def test_failed_pac_notice(windows_pac):
    windows_pac["www.company.example"] = PacResult(False, error="the PAC script has an error")
    network.proxy_for("https://www.company.example/")
    web = WebAccessState()
    web.record_fetch(policy.NETWORK_ERROR)
    web.record_fetch(policy.SOURCE_TIMEOUT)
    out = notices.network_notices(web)
    assert out and out[0].startswith("Windows' automatic proxy configuration script could not be evaluated (the PAC script has an error)")


def test_audit_details_carry_pac_and_signin_state_without_credentials(windows_pac, configure):
    configure(web_proxy_auth="integrated")
    network.proxy_for("https://www.company.example/")
    details = notices.network_audit_details(WebAccessState())
    assert details["network"]["proxy_auth"] == "integrated" and details["network"]["pac"] == "script"
    assert details["network"]["pac_last_result"] == PROXY


# ---------------------------------------------------------------- the local model stays local
def test_local_model_requests_never_use_a_proxy(monkeypatch, windows_pac):
    from app.services.llm import ollama_provider

    seen = {}

    def get(url, **kw):
        seen.update(kw, url=url)
        return httpx.Response(200, json={"models": []}, request=httpx.Request("GET", url))

    monkeypatch.setattr(ollama_provider.httpx, "get", get)
    ollama_provider.OllamaProvider("http://localhost:11434", "llama3.2:1b").status()
    assert seen["trust_env"] is False and "proxy" not in seen and windows_pac["__calls__"] == []


# ---------------------------------------------------------------- real Windows functions (no proxy is contacted)
PAC_SCRIPT = (b'function FindProxyForURL(url, host) {'
              b' if (dnsDomainIs(host, ".company.example")) return "PROXY proxy-a.corp.example:8080; PROXY proxy-b.corp.example:8080; DIRECT";'
              b' return "DIRECT"; }')


@pytest.fixture()
def pac_server():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ns-proxy-autoconfig")
            self.end_headers()
            self.wfile.write(PAC_SCRIPT)

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/proxy.pac"
    server.shutdown()


@windows_only
def test_windows_evaluates_a_real_pac_script(pac_server, configure, monkeypatch):
    configure(web_proxy_mode="system", web_proxy_pac_url=pac_server)
    monkeypatch.setattr(network, "_environment_proxies", lambda: {})
    assert network.proxy_for("https://www.company.example/report.pdf") == "http://proxy-a.corp.example:8080"
    assert network.proxy_for("https://www.other.example/") is None  # DIRECT


@windows_only
def test_windows_reports_an_unreachable_pac_script():
    res = windows_proxy.evaluate_pac("https://www.company.example/", pac_url="http://127.0.0.1:9/none.pac",
                                     auto_detect=False, timeout_seconds=3)
    assert not res.ok and not res.not_found and res.error


@windows_only
def test_windows_reads_the_internet_options():
    cfg = windows_proxy.read_windows_config()
    assert cfg is not None and isinstance(cfg.auto_detect, bool)


@windows_only
def test_kerberos_needs_a_proxy_host_name():
    with pytest.raises(IntegratedAuthUnavailable) as e:
        windows_proxy.kerberos_proxy_token("10.1.2.3")
    assert "IP address" in e.value.reason


@windows_only
def test_sspi_never_hands_out_ntlm():
    """On a domain computer with a Kerberos service name this is a ticket; anywhere else SPNEGO would fall back to
    NTLM, which must be refused."""
    try:
        token = base64.b64decode(windows_proxy.kerberos_proxy_token("proxy.corp.example"))
    except IntegratedAuthUnavailable as exc:
        assert exc.reason
    else:
        assert not windows_proxy.is_ntlm_token(token)


def test_ntlm_messages_are_recognised():
    assert windows_proxy.is_ntlm_token(b"\x60\x48\x06\x06+\x06\x01\x05\x05\x02NTLMSSP\x00\x01\x00\x00\x00")
    assert not windows_proxy.is_ntlm_token(b"\x60\x82\x05\x00\x06\x09\x2a\x86\x48\x86\xf7\x12\x01\x02\x02")


def test_not_windows_means_unavailable(monkeypatch):
    monkeypatch.setattr(windows_proxy, "IS_WINDOWS", False)
    assert windows_proxy.read_windows_config() is None
    assert not windows_proxy.evaluate_pac("https://x.example/", pac_url="http://p/p.pac", auto_detect=False,
                                          timeout_seconds=1).ok
    with pytest.raises(IntegratedAuthUnavailable):
        windows_proxy.kerberos_proxy_token("proxy.corp.example")
