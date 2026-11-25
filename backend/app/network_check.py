"""Network diagnostics for this application, run on the computer where web access fails:

    cd backend
    python -m app.network_check                       # checks https://www.example.com/
    python -m app.network_check https://www.company.com/

It makes the same requests the application makes, through the same proxy and certificate settings, and says
what stopped them. It changes nothing, stores nothing and prints no credentials. It does not try to get around
a proxy, web filter or sign-in requirement: if the network refuses a request, that is the answer.

For a proxy it also shows which sign-in methods the proxy asks for and whether Windows can provide a Kerberos
sign-in for it from the current session. A sign-in is only sent to the proxy when WEB_PROXY_AUTH=integrated.
"""
from __future__ import annotations

import socket
import sys
import time
from urllib.parse import urlsplit

from app.core.config import get_settings
from app.services.security import network, policy, windows_proxy
from app.services.security.policy import SourceRejection, run_bounded

EXPLANATIONS = {
    policy.PROXY_AUTH_REQUIRED: "The network proxy requires sign-in (HTTP 407 / 'authentication_failed'). With IT's approval, "
                                "WEB_PROXY_AUTH=integrated signs in with this Windows account (Kerberos; see 'Proxy sign-in' "
                                "above). Otherwise ask IT for an approved proxy setting (WEB_PROXY_URL), or upload reports.",
    policy.PROXY_SIGNIN_UNAVAILABLE: "Integrated sign-in is on, but Windows could not provide a Kerberos sign-in for the proxy "
                                     "(see 'Proxy sign-in' above for the reason). Ask IT whether this computer and account "
                                     "can use Kerberos for the proxy.",
    policy.PROXY_SIGNIN_REFUSED: "The proxy refused this Windows account's Kerberos sign-in. The account may not be allowed "
                                 "internet access through this proxy, or IT may need to allow this application.",
    policy.BLOCKED_BY_WEB_FILTER: "The organisation's web filter blocked this site. Use manual report upload, or ask IT "
                                  "whether the site can be allowed.",
    policy.TLS_VERIFICATION_FAILED: "The server certificate could not be verified. If the network inspects HTTPS, ask IT for "
                                    "the organisation's certificate file (PEM) and set WEB_CA_BUNDLE to its path.",
    policy.PROXY_CONNECTION_FAILED: "The configured proxy could not be reached or refused the connection. Check WEB_PROXY_MODE, "
                                    "WEB_PROXY_URL or the Windows proxy settings.",
    policy.DNS_RESOLUTION_FAILED: "The site's name could not be looked up on this network. A proxy is probably required: "
                                  "set WEB_PROXY_URL (or the Windows proxy settings).",
    policy.NETWORK_ERROR: "No connection could be made. If the network requires a proxy, set WEB_PROXY_URL.",
    policy.SOURCE_TIMEOUT: "The request timed out. Direct connections may be blocked: a proxy may be required.",
    policy.HTTP_ERROR: "The site answered with an error status. The network itself works.",
}


def _line(label: str, value: str) -> None:
    print(f"  {label:<26} {value}")


def check_configuration() -> None:
    s = get_settings()
    net = network.describe()
    print("Configuration")
    _line("Web search enabled", str(s.enable_web_search))
    _line("External retrieval enabled", str(s.enable_external_fetch))
    _line("Proxy mode", f"{net['proxy_mode']} (setting from: {net['proxy_source']})")
    _line("HTTPS proxy", net["https_proxy"])
    _line("HTTP proxy", net["http_proxy"])
    pac = network.windows_autoconfig_url()
    cfg = network.windows_config()
    pac_url, auto_detect = network.pac_configuration()
    _line("Windows PAC script", f"configured (host {_host(pac)})" if pac else "none")
    _line("Windows auto-detect (WPAD)", ("on" if cfg.auto_detect else "off") if cfg else "not applicable")
    _line("Evaluated by this app", f"script at host {_host(pac_url)}" if pac_url else "WPAD auto-detection" if auto_detect
          else "nothing (WEB_PROXY_MODE is not 'system', or proxy environment variables take precedence)" if pac
          else "nothing to evaluate")
    _line("Proxy sign-in", "integrated Windows sign-in (Kerberos) when asked" if network.proxy_auth_mode() == "integrated"
          else "none (WEB_PROXY_AUTH=none)")
    _line("Certificate authorities", net["certificates"])
    _line("Limits", f"MAX_DOWNLOAD_MB={s.max_download_mb} MAX_PDF_SIZE_MB={s.max_pdf_size_mb} MAX_PDF_PAGES={s.max_pdf_pages}")
    if pac and not pac_url:
        print("  ! Windows finds its proxy through an automatic configuration script, which is not used with the")
        print("    current settings. Set WEB_PROXY_MODE=system (and no HTTPS_PROXY/HTTP_PROXY variables) to use it.")
    try:
        network.ssl_context()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! The certificate configuration cannot be used: {type(exc).__name__}. Check WEB_CA_BUNDLE.")


def _host(url: str | None) -> str:
    try:
        return urlsplit(url or "").hostname or "?"
    except ValueError:
        return "?"


def check_route(url: str) -> str | None:
    print(f"\nRoute for {url}")
    route = network.pac_route(url)
    if route is not None:
        if route.not_found:
            _line("Proxy script answer", "none found by auto-detection (normal where WPAD is not set up)")
        elif route.ok:
            _line("Proxy script answer", "DIRECT" if route.direct else
                  "; ".join(network.mask_proxy("http://" + p) for p in route.proxies))
        else:
            _line("Proxy script answer", f"FAILED: {route.error}")
            print("  ! The automatic proxy configuration could not be evaluated, so the static Windows proxy setting")
            print("    (if any) or a direct connection is used, as Windows itself does.")
    proxy = network.proxy_for(url)
    _line("Route", f"via proxy {network.mask_proxy(proxy)}" if proxy else "direct")
    return proxy


def _read_head(sock: socket.socket, deadline: float) -> bytes:
    data = b""
    while b"\r\n\r\n" not in data and len(data) < 16384 and time.monotonic() < deadline:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    return data


def probe_proxy_signin(proxy: str, url: str, header: dict[str, str] | None = None) -> tuple[int | None, list[str], str]:
    """One CONNECT to the proxy for the site's host: (status, sign-in methods the proxy offers, error)."""
    p, target = urlsplit(proxy), urlsplit(url)
    if p.scheme != "http" or not p.hostname:
        return None, [], f"{p.scheme} proxies are not probed"
    host, port = target.hostname or "", target.port or (443 if target.scheme == "https" else 80)
    lines = [f"CONNECT {host}:{port} HTTP/1.1", f"Host: {host}:{port}", "User-Agent: ESG-Compass-NetworkCheck"]
    lines += [f"{k}: {v}" for k, v in (header or {}).items()]
    timeout = get_settings().http_connect_timeout_seconds
    try:
        with socket.create_connection((p.hostname, p.port or 8080), timeout=timeout) as sock:
            sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))
            head = _read_head(sock, time.monotonic() + timeout)
    except OSError as exc:
        return None, [], f"the proxy could not be reached ({type(exc).__name__})"
    text = head.decode("latin-1", "replace").split("\r\n")
    try:
        status = int(text[0].split()[1])
    except (IndexError, ValueError):
        return None, [], "the proxy gave an unreadable answer"
    schemes = [ln.split(":", 1)[1].strip().split(" ")[0] for ln in text[1:] if ln.lower().startswith("proxy-authenticate:")]
    return status, list(dict.fromkeys(s for s in schemes if s)), ""


def check_proxy_signin(proxy: str | None, url: str) -> None:
    if not proxy:
        return
    print("\nProxy sign-in")
    status, schemes, error = probe_proxy_signin(proxy, url)
    if status is None:
        _line("Probe", error)
        return
    if status != 407:
        _line("Without sign-in", f"HTTP {status}" + (" (no sign-in needed)" if 200 <= status < 300 else ""))
        return
    _line("Without sign-in", "HTTP 407: the proxy requires sign-in")
    _line("Methods offered", ", ".join(schemes) or "none named")
    if not any(s.lower() in ("negotiate", "kerberos") for s in schemes):
        print("  ! The proxy does not offer Negotiate/Kerberos. Integrated sign-in uses Kerberos only (NTLM and passwords")
        print("    are not used), so it cannot sign in to this proxy. Ask IT for an approved alternative.")
        return
    try:
        header = network.proxy_signin_header(proxy)
        _line("Kerberos for this proxy", f"available from this Windows session (HTTP/{urlsplit(proxy).hostname})")
    except windows_proxy.IntegratedAuthUnavailable as exc:
        _line("Kerberos for this proxy", f"NOT available: {exc.reason}")
        return
    if network.proxy_auth_mode() != "integrated":
        print("  Integrated sign-in is off. With IT's approval, set WEB_PROXY_AUTH=integrated in backend/.env.")
        return
    status, _, error = probe_proxy_signin(proxy, url, header)
    _line("With Windows sign-in", error or (f"HTTP {status}: accepted" if status and 200 <= status < 300 else
                                            f"HTTP {status}: refused" if status == 407 else f"HTTP {status}"))


def check_fetch(url: str) -> bool:
    from app.services.security.safe_fetch import fetch

    print(f"\nRetrieval of {url}")
    started = time.monotonic()
    try:
        res = fetch(url, expect="any")
    except SourceRejection as exc:
        _line("Result", f"FAILED ({exc.code}) after {time.monotonic() - started:.1f}s")
        print(f"  {EXPLANATIONS.get(exc.code, 'The application refused or could not read this source.')}")
        return False
    _line("Result", f"OK: {res.kind}, {res.size} bytes in {time.monotonic() - started:.1f}s")
    return True


def check_search() -> bool:
    from app.services.search.providers import get_search_provider

    provider = get_search_provider()
    print(f"\nWeb search ({provider.name})")
    if provider.name == "none":
        _line("Result", "web search is turned off or not configured")
        return True
    started = time.monotonic()
    results = provider.search("sustainability report", max_results=3)
    if provider.last_error:
        _line("Result", f"FAILED ({provider.last_error}) after {time.monotonic() - started:.1f}s")
        print(f"  {EXPLANATIONS.get(provider.last_error, 'The search service could not be reached.')}")
        return False
    _line("Result", f"OK: {len(results)} result(s) in {time.monotonic() - started:.1f}s")
    return True


def check_local_model() -> None:
    from app.services.llm.factory import get_llm_provider

    print("\nLocal analysis component")
    try:
        st = run_bounded(get_llm_provider().status, 60)
        _line("Result", ("ready: " if st.available else "not available: ") + st.message)
    except TimeoutError:
        _line("Result", "did not answer within 60s (a blocked or very slow library load); the analysis would continue without it")
    except Exception as exc:  # noqa: BLE001
        _line("Result", f"could not be started ({type(exc).__name__}); the analysis would continue without it")


def main(argv: list[str]) -> int:
    url = argv[1] if len(argv) > 1 else "https://www.example.com/"
    print("Swedbank ESG Compass - network check (no credentials are shown; nothing is stored)\n")
    check_configuration()
    check_proxy_signin(check_route(url), url)
    fetched = check_fetch(url)
    searched = check_search()
    check_local_model()
    print()
    if fetched and searched:
        print("Automatic source discovery should work on this network.")
        return 0
    print("Automatic source discovery is limited on this network. Assessments still run: upload the company's report")
    print("on the New assessment page, and it is used as the primary source.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
