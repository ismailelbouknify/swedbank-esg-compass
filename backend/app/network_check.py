"""Network diagnostics for this application, run on the computer where web access fails:

    cd backend
    python -m app.network_check                       # checks https://www.example.com/
    python -m app.network_check https://www.company.com/

It makes the same requests the application makes, through the same proxy and certificate settings, and says
what stopped them. It changes nothing, stores nothing and prints no credentials. It does not try to get around
a proxy, web filter or sign-in requirement: if the network refuses a request, that is the answer.
"""
from __future__ import annotations

import sys
import time

from app.core.config import get_settings
from app.services.security import network, policy
from app.services.security.policy import SourceRejection, run_bounded

EXPLANATIONS = {
    policy.PROXY_AUTH_REQUIRED: "The network proxy requires sign-in (HTTP 407 / 'authentication_failed'). This application "
                                "cannot sign in to it on its own. Ask IT for an approved proxy setting for this application "
                                "(WEB_PROXY_URL), or use manual report upload.",
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
    _line("Windows PAC script", "configured (not evaluated by this application)" if pac else "none")
    _line("Certificate authorities", net["certificates"])
    _line("Limits", f"MAX_DOWNLOAD_MB={s.max_download_mb} MAX_PDF_SIZE_MB={s.max_pdf_size_mb} MAX_PDF_PAGES={s.max_pdf_pages}")
    if pac and net["proxy_source"] == "none":
        print("  ! Windows finds its proxy through an automatic configuration script. Ask IT for the proxy address and")
        print("    set WEB_PROXY_URL in backend/.env (WEB_PROXY_MODE=explicit).")
    try:
        network.ssl_context()
    except Exception as exc:  # noqa: BLE001
        print(f"  ! The certificate configuration cannot be used: {type(exc).__name__}. Check WEB_CA_BUNDLE.")


def check_fetch(url: str) -> bool:
    from app.services.security.safe_fetch import fetch

    print(f"\nRetrieval of {url}")
    _line("Route", f"via proxy {network.mask_proxy(network.proxy_for(url))}" if network.proxy_for(url) else "direct")
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
