"""Restricted security mode: URL/SSRF policy, redirects, content validation, PDF handling, manual-upload
mode, logging. Deterministic: DNS and HTTP are mocked, nothing ever connects anywhere."""
from __future__ import annotations

import gzip
import logging
import socket
import tempfile
import time
from pathlib import Path

import httpx
import pymupdf
import pytest
from sqlalchemy import select

from app.api.serializers import public_error
from app.core.config import Settings, get_settings
from app.db.session import SessionLocal
from app.models.entities import AuditEvent, Evidence, Source
from app.services.documents.pdf_extractor import extract_pdf
from app.services.llm.base import NullLLMProvider
from app.services.llm.factory import get_llm_provider, set_llm_provider
from app.services.llm.ollama_provider import OllamaProvider
from app.services.pipeline import discovery
from app.services.pipeline.runner import COVERAGE_LIMITED_MESSAGE, EXTERNAL_ACCESS_OFF_MESSAGE, run_assessment
from app.services.search.base import NullSearchProvider, SearchProvider, SearchResult
from app.services.search.providers import SerpAPISearchProvider, get_search_provider
from app.services.security import content_validator, policy, safe_fetch
from app.services.security.content_validator import ScanResult, set_content_scanner
from app.services.security.policy import browser_automation_allowed
from app.services.security.safe_fetch import FetchError, fetch
from app.services.security.url_validator import UnsafeURLError, validate_external_url
from tests.conftest import FakeLLM

PUBLIC_IP = "93.184.216.34"
PDF_BYTES = b"%PDF-1.7\n" + b"x" * 100
HTML = "<html><head><title>Sustainability</title></head><body><p>" + "Our sustainability work continues. " * 20 + "</p></body></html>"


@pytest.fixture()
def dns(monkeypatch):
    """Hostname -> addresses. Unknown names resolve to one public address."""
    table: dict[str, list[str]] = {}
    calls: list[str] = []

    def getaddrinfo(host, port, *a, **kw):
        calls.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in table.get(host, [PUBLIC_IP])]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    table["__calls__"] = calls  # type: ignore[assignment]
    return table


@pytest.fixture()
def configure(monkeypatch):
    def _set(**values):
        for key, value in values.items():
            monkeypatch.setattr(get_settings(), key, value)
    return _set


@pytest.fixture(autouse=True)
def _no_sleep_no_scanner(monkeypatch):
    monkeypatch.setattr(safe_fetch.time, "sleep", lambda _s: None)
    set_content_scanner(None)
    yield
    set_content_scanner(None)


def client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def pdf_response(body: bytes = PDF_BYTES, **headers) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "application/pdf", **headers}, content=body)


def never_called(_request):
    raise AssertionError("no request may be made")


def make_pdf(pages: int) -> bytes:
    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"Page {i + 1} of a sustainability report with enough text to count.")
    data = doc.tobytes()
    doc.close()
    return data


# ---------------------------------------------------------------- URL + SSRF
@pytest.mark.parametrize("url", [
    "http://localhost/", "http://127.0.0.1/", "http://10.0.0.1/", "http://192.168.1.1/", "http://172.16.0.1/",
    "http://169.254.169.254/",
])
def test_internal_http_urls_rejected_even_when_http_is_allowed(url, dns, configure):
    configure(allow_http=True)
    with pytest.raises(UnsafeURLError) as e:
        validate_external_url(url)
    assert e.value.code in (policy.PRIVATE_IP, policy.UNSAFE_HOST)


@pytest.mark.parametrize("url, code", [
    ("file:///etc/passwd", policy.UNSAFE_SCHEME),
    ("ftp://example.com/file", policy.UNSAFE_SCHEME),
    ("data:text/html,hello", policy.UNSAFE_SCHEME),
    ("javascript:alert(1)", policy.UNSAFE_SCHEME),
    ("gopher://example.com/", policy.UNSAFE_SCHEME),
    ("smb://server/share", policy.UNSAFE_SCHEME),
    ("//server/share", policy.UNSAFE_SCHEME),
    ("/etc/passwd", policy.UNSAFE_SCHEME),
    ("C:/Windows/system32", policy.UNSAFE_SCHEME),
    ("\\\\server\\share\\report.pdf", policy.UNSAFE_URL),
    ("https://example.com\\@127.0.0.1/", policy.UNSAFE_URL),
    ("https://exa mple.com/", policy.UNSAFE_URL),
    ("https://[not-an-ip/", policy.UNSAFE_URL),
    ("https://user:pw@example.com/", policy.CREDENTIALS_IN_URL),
    ("https://example.com:8443/", policy.UNSAFE_PORT),
    ("", policy.UNSAFE_URL),
])
def test_non_approved_urls_rejected(url, code, dns):
    with pytest.raises(UnsafeURLError) as e:
        validate_external_url(url)
    assert e.value.code == code


@pytest.mark.parametrize("host", [
    "[::1]", "[::]", "[fe80::1]", "[fc00::1]", "[fd00:ec2::254]", "[ff02::1]", "[2001:db8::1]",
    "[::ffff:127.0.0.1]", "[::ffff:10.0.0.1]", "[::ffff:169.254.169.254]", "[2002:7f00:1::]", "[64:ff9b::7f00:1]",
])
def test_ipv6_local_ranges_and_embedded_ipv4_rejected(host, dns):
    with pytest.raises(UnsafeURLError) as e:
        validate_external_url(f"https://{host}/")
    assert e.value.code == policy.PRIVATE_IP


@pytest.mark.parametrize("ip", [
    "0.0.0.1", "127.8.8.8", "10.1.1.1", "100.64.0.1", "172.31.0.1", "192.168.0.1", "169.254.0.1", "198.18.0.1",
    "224.0.0.1", "240.0.0.1", "255.255.255.255", "168.63.129.16",
])
def test_non_public_ipv4_ranges_rejected(ip, dns):
    with pytest.raises(UnsafeURLError) as e:
        validate_external_url(f"https://{ip}/")
    assert e.value.code == policy.PRIVATE_IP


@pytest.mark.parametrize("host", [
    "localhost", "LOCALHOST.", "foo.localhost", "metadata.google.internal", "server.internal", "printer.local",
    "intranet", "2130706433", "0x7f.0.0.1", "127.1", "0177.0.0.1", "\uff4c\uff4f\uff43\uff41\uff4c\uff48\uff4f\uff53\uff54",
])
def test_local_names_and_obfuscated_ips_rejected_without_dns(host, dns):
    with pytest.raises(UnsafeURLError) as e:
        validate_external_url(f"https://{host}/")
    assert e.value.code == policy.UNSAFE_HOST
    assert dns["__calls__"] == []


def test_full_width_ip_is_read_as_the_ip_it_spells(dns):
    with pytest.raises(UnsafeURLError) as e:
        validate_external_url("https://\uff11\uff12\uff17.\uff10.\uff10.\uff11/")
    assert e.value.code == policy.PRIVATE_IP


def test_international_names_are_requested_in_ascii_form(dns):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "text/html"}, content=HTML)

    with client_for(handler) as client:
        fetch("https://b\u00fccher.example/", client=client)
    assert seen[0].headers["host"] == "xn--bcher-kva.example"
    assert seen[0].extensions["sni_hostname"] == "xn--bcher-kva.example"


def test_hostname_is_judged_by_what_it_resolves_to(dns):
    dns["looks-public.example"] = ["10.0.0.8"]
    dns["mixed.example"] = [PUBLIC_IP, "192.168.0.8"]
    dns["v6-local.example"] = ["fd12::1"]
    for host in dns.keys() - {"__calls__"}:
        with pytest.raises(UnsafeURLError) as e:
            validate_external_url(f"https://{host}/")
        assert e.value.code == policy.PRIVATE_IP


def test_public_https_url_accepted_with_validated_addresses(dns):
    dns["good.example"] = ["2606:4700::6810:84e5", PUBLIC_IP]
    target = validate_external_url("https://good.example/report.pdf?x=1")
    assert target.host == "good.example"
    assert target.addresses == (PUBLIC_IP, "2606:4700::6810:84e5")  # IPv4 first


def test_dns_failure_is_unavailability_not_a_security_rejection(monkeypatch):
    def fail(*_a, **_k):
        raise socket.gaierror("no such host")
    monkeypatch.setattr(socket, "getaddrinfo", fail)
    with pytest.raises(UnsafeURLError) as e:
        validate_external_url("https://nowhere.example/")
    assert e.value.code == policy.DNS_RESOLUTION_FAILED
    assert not policy.is_security_rejection(e.value.code)


def test_domain_allowlist_is_configuration_driven(dns, configure):
    configure(allowed_external_domains="Example.com, sec.gov")
    assert validate_external_url("https://example.com/")
    assert validate_external_url("https://www.sec.gov/")
    for url in ("https://other.com/", "https://notexample.com/", "https://example.com.evil.org/"):
        with pytest.raises(UnsafeURLError) as e:
            validate_external_url(url)
        assert e.value.code == policy.DOMAIN_NOT_ALLOWED
    configure(allowed_domains_include_subdomains=False)
    with pytest.raises(UnsafeURLError):
        validate_external_url("https://www.sec.gov/")
    assert validate_external_url("https://sec.gov/")


def test_rejected_url_never_reaches_the_network(dns):
    with client_for(never_called) as client, pytest.raises(FetchError) as e:
        fetch("https://169.254.169.254/latest/meta-data", client=client)
    assert e.value.code == policy.PRIVATE_IP


# ---------------------------------------------------------------- pinning + redirects
def test_connection_is_pinned_to_the_validated_address(dns):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, headers={"content-type": "text/html"}, content=HTML)

    dns["site.example"] = [PUBLIC_IP]
    with client_for(handler) as client:
        res = fetch("https://site.example/page?a=1", client=client)
    req = seen[0]
    assert req.url.host == PUBLIC_IP and req.url.path == "/page" and req.url.query == b"a=1"
    assert req.headers["host"] == "site.example"
    assert req.extensions["sni_hostname"] == "site.example"  # TLS name check stays on the real hostname
    assert res.final_url == "https://site.example/page?a=1"


def test_ipv6_address_can_be_pinned(dns):
    dns["v6.example"] = ["2606:4700::6810:84e5"]
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, headers={"content-type": "text/html"}, content=HTML)

    with client_for(handler) as client:
        fetch("https://v6.example/", client=client)
    assert seen == ["https://[2606:4700::6810:84e5]/"]


def test_dns_rebinding_cannot_redirect_the_connection(dns, monkeypatch):
    """The name turns private right after validation: the request still goes to the validated address,
    and the next hop's validation catches the new answer."""
    seen = []

    def handler(request):
        seen.append(request.url.host)
        dns["rebind.example"] = ["127.0.0.1"]
        return httpx.Response(302, headers={"location": "https://rebind.example/second"})

    dns["rebind.example"] = [PUBLIC_IP]
    with client_for(handler) as client, pytest.raises(FetchError) as e:
        fetch("https://rebind.example/first", client=client)
    assert seen == [PUBLIC_IP]
    assert e.value.code == policy.PRIVATE_IP and e.value.redirected


@pytest.mark.parametrize("location, code", [
    ("https://10.0.0.1/admin", policy.PRIVATE_IP),
    ("https://[::1]/admin", policy.PRIVATE_IP),
    ("https://169.254.169.254/latest/meta-data", policy.PRIVATE_IP),
    ("https://resolves-private.example/", policy.PRIVATE_IP),
    ("https://localhost/", policy.UNSAFE_HOST),
    ("file:///etc/passwd", policy.UNSAFE_SCHEME),
    ("http://public.example/downgrade", policy.HTTP_NOT_ALLOWED),
])
def test_unsafe_redirect_targets_rejected(location, code, dns):
    dns["resolves-private.example"] = ["192.168.5.5"]
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": location})

    with client_for(handler) as client, pytest.raises(FetchError) as e:
        fetch("https://public.example/start", client=client)
    assert e.value.code == code and e.value.redirected
    assert policy.audit_event_for(e.value) == "redirect_rejected"
    assert len(requests) == 1  # the target itself was never requested


def test_redirect_outside_allowlist_rejected(dns, configure):
    configure(allowed_external_domains="public.example")
    with client_for(lambda r: httpx.Response(302, headers={"location": "https://elsewhere.example/"})) as client, \
            pytest.raises(FetchError) as e:
        fetch("https://public.example/", client=client)
    assert e.value.code == policy.DOMAIN_NOT_ALLOWED


def test_redirect_chain_over_limit_rejected(dns, configure):
    configure(max_redirects=3)
    count = {"n": 0}

    def handler(_request):
        count["n"] += 1
        return httpx.Response(302, headers={"location": f"/next{count['n']}"})

    with client_for(handler) as client, pytest.raises(FetchError) as e:
        fetch("https://public.example/", client=client)
    assert e.value.code == policy.TOO_MANY_REDIRECTS
    assert count["n"] == 4  # the first request + 3 permitted redirects


def test_redirect_within_limit_followed_with_relative_location(dns, configure):
    configure(max_redirects=3)

    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(301, headers={"location": "/report.pdf"})
        assert request.headers["host"] == "public.example"
        return pdf_response()

    with client_for(handler) as client:
        res = fetch("https://public.example/start", client=client)
    assert res.kind == "pdf" and res.final_url == "https://public.example/report.pdf"


def test_redirects_are_never_followed_by_the_http_client():
    with safe_fetch.build_client() as client:
        assert client.follow_redirects is False and client.trust_env is False
        assert client.timeout.connect == get_settings().http_connect_timeout_seconds
        assert client.timeout.read == get_settings().http_read_timeout_seconds


# ---------------------------------------------------------------- content
def test_pdf_url_returning_html_rejected(dns):
    with client_for(lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=HTML)) as client, \
            pytest.raises(FetchError) as e:
        fetch("https://public.example/report.pdf", client=client)
    assert e.value.code == policy.INVALID_CONTENT_TYPE


@pytest.mark.parametrize("ctype", ["application/pdf", "application/octet-stream"])
def test_pdf_claim_with_wrong_bytes_rejected(ctype, dns):
    with client_for(lambda r: httpx.Response(200, headers={"content-type": ctype}, content=b"<html>not a pdf</html>")) as client, \
            pytest.raises(FetchError) as e:
        fetch("https://public.example/report.pdf", client=client)
    assert e.value.code == policy.INVALID_PDF_SIGNATURE


@pytest.mark.parametrize("body", [b"", b"%PD", b"%PDF"])
def test_empty_or_truncated_pdf_rejected(body, dns):
    with client_for(lambda r: pdf_response(body)) as client, pytest.raises(FetchError) as e:
        fetch("https://public.example/report.pdf", client=client)
    assert e.value.code == policy.INVALID_PDF_SIGNATURE


def test_redirect_from_pdf_url_keeps_pdf_expectations(dns):
    def handler(request, body=PDF_BYTES, ctype="application/octet-stream"):
        if request.url.path == "/report.pdf":
            return httpx.Response(302, headers={"location": "/download?id=7"})
        return httpx.Response(200, headers={"content-type": ctype}, content=body)

    with client_for(handler) as client:
        assert fetch("https://public.example/report.pdf", client=client).kind == "pdf"
    with client_for(lambda r: handler(r, HTML, "text/html")) as client, pytest.raises(FetchError) as e:
        fetch("https://public.example/report.pdf", client=client)
    assert e.value.code == policy.INVALID_CONTENT_TYPE


def test_redirect_to_pdf_url_returning_html_rejected(dns):
    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/report.pdf"})
        return httpx.Response(200, headers={"content-type": "text/html"}, content=HTML)

    with client_for(handler) as client, pytest.raises(FetchError) as e:
        fetch("https://public.example/start", client=client)
    assert e.value.code == policy.INVALID_CONTENT_TYPE


def test_url_ending_in_pdf_alone_does_not_make_it_a_pdf(dns):
    with client_for(lambda r: httpx.Response(200, headers={"content-type": "application/zip"}, content=b"PK\x03\x04")) as client, \
            pytest.raises(FetchError) as e:
        fetch("https://public.example/report.pdf", client=client)
    assert e.value.code == policy.INVALID_CONTENT_TYPE


@pytest.mark.parametrize("ctype", ["application/x-msdownload", "application/zip", "image/png", "text/javascript", ""])
def test_unrelated_content_types_rejected_without_reading_the_body(ctype, dns):
    def body():
        raise AssertionError("body must not be read")
        yield b""

    with client_for(lambda r: httpx.Response(200, headers={"content-type": ctype}, content=body())) as client, \
            pytest.raises(FetchError) as e:
        fetch("https://public.example/file", client=client)
    assert e.value.code == policy.INVALID_CONTENT_TYPE


def test_valid_pdf_accepted_and_hashed(dns):
    with client_for(lambda r: pdf_response()) as client:
        res = fetch("https://public.example/report.pdf", client=client)
    assert res.kind == "pdf" and res.content == PDF_BYTES and res.size == len(PDF_BYTES)
    assert res.sha256 == content_validator.sha256_hex(PDF_BYTES)
    assert res.scan_status == "not_configured"


def test_declared_oversize_rejected_before_reading_the_body(dns, configure):
    configure(max_download_mb=1)

    def body():
        raise AssertionError("body must not be read")
        yield b""

    def handler(_r):
        return httpx.Response(200, headers={"content-type": "application/pdf", "content-length": str(2 * 1024 * 1024)}, content=body())

    with client_for(handler) as client, pytest.raises(FetchError) as e:
        fetch("https://public.example/report.pdf", client=client)
    assert e.value.code == policy.DOWNLOAD_SIZE_LIMIT_EXCEEDED


@pytest.mark.parametrize("headers", [{}, {"content-length": "10"}])
def test_stream_aborted_when_size_is_missing_or_wrong(headers, dns, configure):
    configure(max_download_mb=1)
    sent = {"chunks": 0}

    def body():
        yield b"%PDF-1.7\n"
        for _ in range(10_000):
            sent["chunks"] += 1
            yield b"x" * 64 * 1024

    with client_for(lambda r: httpx.Response(200, headers={"content-type": "application/pdf", **headers}, content=body())) as client, \
            pytest.raises(FetchError) as e:
        fetch("https://public.example/report.pdf", client=client)
    assert e.value.code == policy.DOWNLOAD_SIZE_LIMIT_EXCEEDED
    assert sent["chunks"] <= 17  # stopped at the limit, not at the end of the stream


def test_compressed_response_cannot_expand_past_the_limit(dns, configure):
    configure(max_download_mb=1)
    bomb = gzip.compress(b"%PDF-1.7\n" + b"\0" * (5 * 1024 * 1024))

    def handler(_r):
        return httpx.Response(200, headers={"content-type": "application/pdf", "content-encoding": "gzip"}, content=bomb)

    with client_for(handler) as client, pytest.raises(FetchError) as e:
        fetch("https://public.example/report.pdf", client=client)
    assert e.value.code == policy.DOWNLOAD_SIZE_LIMIT_EXCEEDED


def test_html_has_its_own_smaller_limit(dns, configure):
    configure(max_download_mb=20, max_html_size_mb=1)
    big = b"<html>" + b"x" * (2 * 1024 * 1024)
    with client_for(lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=big)) as client, \
            pytest.raises(FetchError) as e:
        fetch("https://public.example/", client=client)
    assert e.value.code == policy.DOWNLOAD_SIZE_LIMIT_EXCEEDED


def test_not_a_pdf_stops_at_the_first_read(dns):
    sent = {"chunks": 0}

    def body():
        for _ in range(1000):
            sent["chunks"] += 1
            yield b"MZ" + b"\0" * 1000

    with client_for(lambda r: httpx.Response(200, headers={"content-type": "application/pdf"}, content=body())) as client, \
            pytest.raises(FetchError) as e:
        fetch("https://public.example/report.pdf", client=client)
    assert e.value.code == policy.INVALID_PDF_SIGNATURE
    assert sent["chunks"] * 1002 <= 2 * safe_fetch.READ_CHUNK_BYTES  # not the whole 20 MB allowance


def test_too_many_pdf_pages_rejected_not_truncated(configure):
    configure(max_pdf_pages=3)
    assert extract_pdf(make_pdf(3)).page_count == 3
    with pytest.raises(content_validator.ContentRejected) as e:
        extract_pdf(make_pdf(4))
    assert e.value.code == policy.PDF_PAGE_LIMIT_EXCEEDED


def test_pdf_extraction_stops_at_the_deadline():
    with pytest.raises(content_validator.ContentRejected) as e:
        extract_pdf(make_pdf(2), deadline=time.monotonic() - 1)
    assert e.value.code == policy.SOURCE_TIMEOUT


def test_other_formats_with_pdf_name_are_not_parsed(tmp_path):
    fake = tmp_path / "report.pdf"
    fake.write_bytes(b"<svg xmlns='http://www.w3.org/2000/svg'/>")
    with pytest.raises(content_validator.ContentRejected) as e:
        extract_pdf(fake)
    assert e.value.code == policy.INVALID_PDF_SIGNATURE


# ---------------------------------------------------------------- timeouts + retries
def test_retries_are_bounded_then_source_is_unavailable(dns, configure):
    configure(max_http_retries=2)
    calls = {"n": 0}

    def handler(_r):
        calls["n"] += 1
        raise httpx.ReadTimeout("slow")

    with client_for(handler) as client, pytest.raises(FetchError) as e:
        fetch("https://public.example/", client=client)
    assert calls["n"] == 3
    assert e.value.code == policy.SOURCE_TIMEOUT and not policy.is_security_rejection(e.value.code)


def test_transient_failure_recovers_on_retry(dns, configure):
    configure(max_http_retries=2)
    calls = {"n": 0}

    def handler(_r):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] == 1 else httpx.Response(200, headers={"content-type": "text/html"}, content=HTML)

    with client_for(handler) as client:
        assert fetch("https://public.example/", client=client).kind == "html"
    assert calls["n"] == 2


def test_permanent_http_and_policy_errors_are_not_retried(dns):
    calls = {"n": 0}

    def handler(_r):
        calls["n"] += 1
        return httpx.Response(404)

    with client_for(handler) as client, pytest.raises(FetchError) as e:
        fetch("https://public.example/", client=client)
    assert e.value.code == policy.HTTP_ERROR and calls["n"] == 1


def test_total_source_time_is_bounded_during_slow_streaming(dns):
    deadline = time.monotonic() + 0.1

    def body():
        yield b"%PDF-1.7\n" + b"x" * safe_fetch.READ_CHUNK_BYTES
        while time.monotonic() < deadline:
            pass
        yield b"x" * safe_fetch.READ_CHUNK_BYTES
        raise AssertionError("must stop at the deadline")

    with client_for(lambda r: httpx.Response(200, headers={"content-type": "application/pdf"}, content=body())) as client, \
            pytest.raises(FetchError) as e:
        fetch("https://public.example/report.pdf", client=client, deadline=deadline)
    assert e.value.code == policy.SOURCE_TIMEOUT


def test_serpapi_failure_does_not_log_the_api_key(monkeypatch, caplog):
    def get(*_a, **_k):
        raise httpx.ConnectError("failed for https://serpapi.com/search.json?api_key=SECRET-KEY")

    monkeypatch.setattr(httpx, "get", get)
    with caplog.at_level(logging.INFO):
        assert SerpAPISearchProvider("SECRET-KEY").search("Company", 3) == []
    assert "SECRET-KEY" not in caplog.text


# ---------------------------------------------------------------- malware scan hook
def test_scan_defaults_to_not_configured_and_can_fail_closed(dns, configure):
    assert content_validator.scan_downloaded_content(b"x", kind="pdf", sha256="0").status == "not_configured"
    configure(malware_scan_required=True)
    with client_for(lambda r: pdf_response()) as client, pytest.raises(FetchError) as e:
        fetch("https://public.example/report.pdf", client=client)
    assert e.value.code == policy.MALWARE_SCAN_FAILED


@pytest.mark.parametrize("scanner", [
    lambda c, k, h: ScanResult("infected"),
    lambda c, k, h: ScanResult("error"),
    lambda c, k, h: 1 / 0,
])
def test_only_a_clean_scan_passes_a_configured_scanner(scanner, dns):
    set_content_scanner(scanner)
    with client_for(lambda r: pdf_response()) as client, pytest.raises(FetchError) as e:
        fetch("https://public.example/report.pdf", client=client)
    assert e.value.code == policy.MALWARE_SCAN_FAILED


def test_scan_runs_before_parsing_upload(monkeypatch):
    def parser_called(*_a, **_k):
        raise AssertionError("parser must not run on rejected content")

    monkeypatch.setattr(content_validator.pymupdf, "open", parser_called)
    set_content_scanner(lambda c, k, h: ScanResult("infected"))
    with pytest.raises(content_validator.ContentRejected) as e:
        content_validator.validate_uploaded_pdf(PDF_BYTES)
    assert e.value.code == policy.MALWARE_SCAN_FAILED


def test_clean_scan_is_recorded(dns):
    scanned = []
    set_content_scanner(lambda c, k, h: scanned.append((k, h)) or ScanResult("clean"))
    with client_for(lambda r: pdf_response()) as client:
        res = fetch("https://public.example/report.pdf", client=client)
    assert res.scan_status == "clean" and scanned == [("pdf", res.sha256)]


# ---------------------------------------------------------------- configuration
def test_restricted_mode_defaults():
    s = Settings(_env_file=None)
    assert s.bank_safe_mode is True
    assert s.http_allowed is False
    assert s.store_external_files is False
    assert s.enable_playwright is False
    assert s.max_redirects == 3 and s.max_download_mb == 20 and s.max_pdf_pages == 300


def test_http_rejected_in_restricted_mode_unless_explicitly_allowed(dns, configure):
    with pytest.raises(UnsafeURLError) as e:
        validate_external_url("http://public.example/")
    assert e.value.code == policy.HTTP_NOT_ALLOWED
    configure(allow_http=True)
    assert validate_external_url("http://public.example/")
    configure(allow_http=False, bank_safe_mode=False)
    assert validate_external_url("http://public.example/")


def test_browser_automation_needs_explicit_enabling(configure, monkeypatch):
    assert browser_automation_allowed() is False
    configure(enable_playwright=True)
    assert browser_automation_allowed() is True
    monkeypatch.setenv("USE_PLAYWRIGHT", "true")  # the earlier name of the setting still counts
    assert Settings(_env_file=None).enable_playwright is True


def test_remote_llm_endpoint_never_used_in_restricted_mode(configure):
    set_llm_provider(None)
    configure(llm_provider="ollama", ollama_base_url="https://llm.example.com:11434")
    assert isinstance(get_llm_provider(), NullLLMProvider)
    set_llm_provider(None)
    configure(ollama_base_url="http://localhost:11434")
    assert isinstance(get_llm_provider(), OllamaProvider)
    set_llm_provider(None)
    configure(ollama_base_url="http://127.0.0.1:11434")
    assert isinstance(get_llm_provider(), OllamaProvider)
    set_llm_provider(None)
    configure(ollama_base_url="https://llm.example.com:11434", bank_safe_mode=False)
    assert isinstance(get_llm_provider(), OllamaProvider)
    set_llm_provider(None)


def test_search_provider_is_off_when_search_is_disabled(configure):
    configure(search_provider="ddgs", enable_web_search=False)
    assert isinstance(get_search_provider(), NullSearchProvider)


def test_fetch_refuses_when_external_access_is_off(configure):
    configure(enable_external_fetch=False)
    with client_for(never_called) as client, pytest.raises(FetchError) as e:
        fetch("https://public.example/", client=client)
    assert e.value.code == policy.EXTERNAL_FETCH_DISABLED


# ---------------------------------------------------------------- pipeline
COMPANY = "Nordvik Components AB"
SITE = "https://www.nordvikcomponents.example"


class SpySearch(SearchProvider):
    name = "spy"

    def __init__(self, urls: list[str] | None = None):
        self.urls = urls or []
        self.queries: list[str] = []

    def search(self, query, max_results):
        self.queries.append(query)
        return [SearchResult(title=f"{COMPANY} sustainability report 2025", url=u, snippet="sustainability report", query=query)
                for u in self.urls]


def _assessment(client, **kw) -> int:
    body = {"company_name": COMPANY, "year": 2025, "enable_web_search": True, "confirm_new": True, **kw}
    r = client.post("/api/assessments", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _run(client, aid: int, search: SearchProvider, llm=None) -> dict:
    run_assessment(aid, llm=llm or FakeLLM(), search=search)
    return client.get(f"/api/assessments/{aid}/status").json()


def _use_transport(monkeypatch, handler) -> None:
    monkeypatch.setattr(safe_fetch, "build_client", lambda: client_for(handler))


def _events(aid: int) -> list[AuditEvent]:
    with SessionLocal() as db:
        return list(db.scalars(select(AuditEvent).where(AuditEvent.assessment_id == aid)))


def _sources(aid: int) -> dict[str, Source]:
    with SessionLocal() as db:
        return {s.url: s for s in db.scalars(select(Source).where(Source.assessment_id == aid))}


def _stored_pdfs() -> set[Path]:
    """PDFs anywhere in the application's data directory, plus report files directly in the OS temp dir."""
    temp = {p for p in Path(tempfile.gettempdir()).glob("*.pdf")}
    return temp | set(Path(get_settings().data_dir).rglob("*.pdf"))


def test_full_pipeline_with_blocked_sources_continues_and_scores(client, dns, monkeypatch, sample_pdf, caplog):
    """A report is retrieved through the safe path and scored; the blocked sources change nothing else."""
    pdf = sample_pdf.read_bytes()
    dns["private-host.nordvikcomponents.example"] = ["10.1.2.3"]
    good, redirecting, internal, plain_http = (
        f"{SITE}/sustainability-report-2025.pdf", f"{SITE}/moved-report-2025.pdf",
        "https://private-host.nordvikcomponents.example/report-2025.pdf", "http://www.nordvikcomponents.example/report-2025.pdf",
    )

    def handler(request):
        if request.url.path == "/moved-report-2025.pdf":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data"})
        assert request.url.path == "/sustainability-report-2025.pdf"
        return pdf_response(pdf)

    _use_transport(monkeypatch, handler)
    before = _stored_pdfs()
    aid = _assessment(client)
    llm = FakeLLM()
    with caplog.at_level(logging.INFO):
        st = _run(client, aid, SpySearch([good, redirecting, internal, plain_http]), llm)

    assert st["status"] == "COMPLETED"
    assert llm.calls, "the local model path was used"
    sources = _sources(aid)
    assert sources[good].status == "PROCESSED" and sources[good].fetch_outcome == "ok"
    assert sources[good].content_sha256 == content_validator.sha256_hex(pdf) and sources[good].bytes_downloaded == len(pdf)
    assert sources[good].file_path is None, "external PDFs are not stored"
    assert sources[redirecting].status == "REJECTED" and sources[redirecting].fetch_outcome == policy.HTTP_NOT_ALLOWED
    assert internal not in sources and plain_http not in sources  # refused before becoming a source
    assert _stored_pdfs() == before

    # evidence and deterministic scoring still work, from the extracted text
    with SessionLocal() as db:
        evidence = list(db.scalars(select(Evidence).where(Evidence.assessment_id == aid)))
    assert any(e.source_url == good and e.page_number for e in evidence)
    recs = client.get(f"/api/assessments/{aid}/recommendations").json()
    assert recs and any((r["recommended_level"] or 0) > 0 for r in recs)

    # audit: technical reasons recorded
    events = _events(aid)
    types = {e.event_type for e in events}
    assert {"redirect_rejected", "private_ip_rejected", "unsafe_url_rejected", "temporary_pdf_processed", "temporary_pdf_deleted"} <= types
    done = next(e for e in events if e.event_type == "web_discovery_completed")
    assert done.details["sources_rejected"] == 1 and done.details["candidates_rejected"] == 2

    # analyst view: simple wording, no technical detail
    assert COVERAGE_LIMITED_MESSAGE in st["notices"]
    rows = client.get(f"/api/assessments/{aid}/sources").json()
    blocked = next(r for r in rows if r["url"] == redirecting)
    assert blocked["status_label"] == "Source could not be processed"
    assert blocked["has_file"] is False
    shown = str(rows) + str(st)
    for word in ("169.254", "private_ip", "http_not_allowed", "SSRF", "fetch_outcome", tempfile.gettempdir()):
        assert word not in shown

    # logs: codes and hostnames, never report content
    assert "source_fetch_rejected" in caplog.text
    assert "reduce energy consumption" not in caplog.text


def test_rejected_sources_are_not_reported_as_absence_of_activity(client, dns, monkeypatch):
    """Every source blocked: the assessment completes and says that sources were unavailable."""
    _use_transport(monkeypatch, lambda r: httpx.Response(302, headers={"location": "file:///etc/passwd"}))
    aid = _assessment(client)
    st = _run(client, aid, SpySearch([f"{SITE}/sustainability-report-2025.pdf"]))
    assert st["status"] == "COMPLETED"
    assert COVERAGE_LIMITED_MESSAGE in st["notices"]


def _failing_fetch_handler(kind: str):
    def handler(_r):
        if kind == "timeout":
            raise httpx.ReadTimeout("slow")
        if kind == "invalid":
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"not a pdf at all")
        return pdf_response(b"%PDF-1.7\nthis is not really a pdf")
    return handler


@pytest.mark.parametrize("kind, status", [("timeout", "FAILED"), ("invalid", "REJECTED"), ("corrupt", "FAILED")])
def test_no_report_file_remains_after_failure(kind, status, client, dns, monkeypatch):
    _use_transport(monkeypatch, _failing_fetch_handler(kind))
    before = _stored_pdfs()
    aid = _assessment(client)
    url = f"{SITE}/sustainability-report-2025.pdf"
    st = _run(client, aid, SpySearch([url]))
    assert st["status"] == "COMPLETED"
    assert _sources(aid)[url].status == status
    assert _stored_pdfs() == before


def test_no_report_file_remains_and_bytes_released_after_unexpected_exception(client, dns, monkeypatch, sample_pdf):
    results = []
    real_fetch = discovery.fetch

    def recording_fetch(*a, **k):
        results.append(real_fetch(*a, **k))
        return results[-1]

    def boom(*_a, **_k):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(discovery, "fetch", recording_fetch)
    monkeypatch.setattr(discovery, "ingest_pdf", boom)
    _use_transport(monkeypatch, lambda r: pdf_response(sample_pdf.read_bytes()))
    before = _stored_pdfs()
    aid = _assessment(client)
    url = f"{SITE}/sustainability-report-2025.pdf"
    st = _run(client, aid, SpySearch([url]))
    assert st["status"] == "COMPLETED"
    assert _sources(aid)[url].fetch_outcome == policy.PDF_PARSE_FAILED
    assert results and all(r.content == b"" for r in results)
    assert _stored_pdfs() == before
    assert "temporary_pdf_deleted" in {e.event_type for e in _events(aid)}


def test_bytes_released_after_success(client, dns, monkeypatch, sample_pdf):
    results = []
    real_fetch = discovery.fetch
    monkeypatch.setattr(discovery, "fetch", lambda *a, **k: results.append(real_fetch(*a, **k)) or results[-1])
    _use_transport(monkeypatch, lambda r: pdf_response(sample_pdf.read_bytes()))
    aid = _assessment(client)
    _run(client, aid, SpySearch([f"{SITE}/sustainability-report-2025.pdf"]))
    assert results and all(r.content == b"" for r in results)


def test_store_setting_cannot_override_restricted_mode(client, dns, monkeypatch, configure, sample_pdf):
    configure(store_downloaded_files=True)
    _use_transport(monkeypatch, lambda r: pdf_response(sample_pdf.read_bytes()))
    before = _stored_pdfs()
    aid = _assessment(client)
    url = f"{SITE}/sustainability-report-2025.pdf"
    _run(client, aid, SpySearch([url]))
    assert _sources(aid)[url].file_path is None
    assert _stored_pdfs() == before


def test_storage_is_opt_in_outside_restricted_mode(client, dns, monkeypatch, configure, sample_pdf):
    configure(bank_safe_mode=False, store_downloaded_files=True)
    _use_transport(monkeypatch, lambda r: pdf_response(sample_pdf.read_bytes()))
    aid = _assessment(client)
    url = f"{SITE}/sustainability-report-2025.pdf"
    _run(client, aid, SpySearch([url]))
    stored = Path(_sources(aid)[url].file_path)
    assert stored.parent == get_settings().download_dir and stored.stem != "report"
    stored.unlink()


def test_malformed_links_in_a_page_do_not_stop_the_assessment(client, dns, monkeypatch):
    page = ('<html><title>Nordvik</title><body><a href="https://[bad/sustainability-report">x</a>'
            '<a href="javascript:alert(1)">sustainability</a><p>' + "Our sustainability work continues. " * 20 + "</p></body></html>")
    _use_transport(monkeypatch, lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=page))
    aid = _assessment(client, website=SITE)
    st = _run(client, aid, SpySearch())
    assert st["status"] == "COMPLETED"
    assert _sources(aid)[SITE].status == "PROCESSED"


# ---------------------------------------------------------------- manual-upload-only mode
@pytest.fixture()
def manual_only(configure, monkeypatch):
    configure(enable_web_search=False, enable_external_fetch=False)

    def no_network(*_a, **_k):
        raise AssertionError("manual-upload-only mode must not touch the network")

    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(safe_fetch, "build_client", no_network)


def test_manual_upload_only_mode_completes_without_any_external_request(client, manual_only, sample_pdf):
    aid = _assessment(client, website=SITE)
    assert client.get(f"/api/assessments/{aid}").json()["assessment"]["enable_web_search"] is False
    with open(sample_pdf, "rb") as f:
        assert client.post(f"/api/assessments/{aid}/documents", files={"file": ("report.pdf", f, "application/pdf")}).status_code == 201
    search, llm = SpySearch([f"{SITE}/x.pdf"]), FakeLLM()
    st = _run(client, aid, search, llm)
    assert st["status"] == "COMPLETED"
    assert search.queries == [] and llm.calls
    recs = client.get(f"/api/assessments/{aid}/recommendations").json()
    assert any((r["recommended_level"] or 0) > 0 for r in recs)
    assert [s["source_type"] for s in client.get(f"/api/assessments/{aid}/sources").json()] == ["uploaded_report"]


def test_manual_mode_without_upload_explains_itself(client, manual_only):
    aid = _assessment(client)
    search = SpySearch([f"{SITE}/x.pdf"])
    st = _run(client, aid, search)
    assert st["status"] == "COMPLETED" and search.queries == []
    assert EXTERNAL_ACCESS_OFF_MESSAGE in st["notices"]


def test_search_off_with_fetch_on_still_reads_a_given_website_but_never_searches(client, dns, monkeypatch, configure):
    configure(enable_web_search=False)
    page = "<html><title>Nordvik</title><body><p>" + "Our sustainability work continues. " * 20 + "</p></body></html>"
    _use_transport(monkeypatch, lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=page))
    aid = _assessment(client, website=SITE)
    search = SpySearch([f"{SITE}/x.pdf"])
    st = _run(client, aid, search)
    assert st["status"] == "COMPLETED" and search.queries == []
    assert any(s.status == "PROCESSED" for s in _sources(aid).values())


def test_fetch_off_with_search_on_sends_no_search_either(client, manual_only, configure):
    configure(enable_web_search=True)
    aid = _assessment(client)
    search = SpySearch()
    _run(client, aid, search)
    assert search.queries == []


# ---------------------------------------------------------------- upload validation
def _upload(client, aid, name, data, ctype="application/pdf"):
    return client.post(f"/api/assessments/{aid}/documents", files={"file": (name, data, ctype)})


@pytest.mark.parametrize("name, data, ctype, status", [
    ("malware.exe", b"MZ\x90\x00", "application/octet-stream", 415),
    ("report.pdf", b"MZ\x90\x00", "application/pdf", 415),
    ("report.pdf", make_pdf(1), "text/html", 415),
    ("report.exe", make_pdf(1), "application/pdf", 415),
    ("report.pdf", b"%PDF", "application/pdf", 415),
    ("report.pdf", b"%PDF-1.7\nnot actually a pdf", "application/pdf", 422),
])
def test_invalid_uploads_rejected(name, data, ctype, status, client):
    aid = _assessment(client)
    assert _upload(client, aid, name, data, ctype).status_code == status
    assert client.get(f"/api/assessments/{aid}/sources").json() == []


def test_upload_page_limit(client, configure):
    configure(max_pdf_pages=2)
    aid = _assessment(client)
    r = _upload(client, aid, "report.pdf", make_pdf(3))
    assert r.status_code == 413
    assert "pdf_page_limit_exceeded" in {e.event_type for e in _events(aid)}
    assert _upload(client, aid, "report.pdf", make_pdf(2)).status_code == 201


def test_password_protected_upload_rejected(client, tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner")
    doc.close()
    aid = _assessment(client)
    r = _upload(client, aid, "report.pdf", data)
    assert r.status_code == 422 and "Password" in r.json()["detail"]


def test_oversized_upload_refused_before_the_body_is_parsed(client, configure):
    configure(max_pdf_size_mb=1)
    aid = _assessment(client)
    assert _upload(client, aid, "report.pdf", b"%PDF-1.7\n" + b"x" * (3 * 1024 * 1024)).status_code == 413


def test_upload_filename_never_reaches_the_filesystem(client):
    aid = _assessment(client)
    r = _upload(client, aid, "../../../evil?.exe.pdf", make_pdf(1))
    assert r.status_code == 201
    with SessionLocal() as db:
        source = db.get(Source, r.json()["id"])
    stored = Path(source.file_path)
    assert stored.parent == get_settings().upload_dir and "evil" not in stored.name
    assert "/" not in source.original_filename and ".." not in source.original_filename.replace(".exe.pdf", "")
    stored.unlink()


def test_upload_is_scanned_when_a_scanner_is_configured(client):
    set_content_scanner(lambda c, k, h: ScanResult("infected"))
    aid = _assessment(client)
    r = _upload(client, aid, "report.pdf", make_pdf(1))
    assert r.status_code == 422 and "infected" not in r.text


# ---------------------------------------------------------------- local web server + UI wording
def test_security_headers_and_host_check(client):
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    assert client.get("/api/health", headers={"host": "attacker.example"}).status_code == 400


def test_api_docs_with_external_cdn_are_off_in_restricted_mode():
    from app.main import app

    assert app.docs_url is None and app.redoc_url is None and app.openapi_url is None


def test_invalid_website_message_does_not_explain_the_rules(client):
    r = client.post("/api/assessments", json={"company_name": "X", "year": 2025, "website": "https://10.0.0.1"})
    assert r.status_code == 422
    assert "10.0.0.1" not in r.text and "private" not in r.text.lower()


@pytest.mark.parametrize("error", ["ValueError: secret path C:/x", "OperationalError: no such table", "[runner crash] KeyError: 'a'"])
def test_exception_text_is_not_shown_to_analysts(error):
    assert "secret" not in public_error(error) and "table" not in public_error(error) and "crash" not in public_error(error)
    assert public_error("Server restarted while the analysis was running.") == "Server restarted while the analysis was running."
