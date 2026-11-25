"""Plain-language reasons why automatic source discovery was limited by the network or by configured limits.

They tell the analyst what happened and what to do; none of them suggests getting around a control. The
technical detail (codes, counts, masked proxy setting) goes to the audit log."""
from __future__ import annotations

from app.services.pipeline.discovery import WebAccessState
from app.services.security import network, policy

MANUAL_UPLOAD_HINT = "Upload the company's report to continue the assessment with it."

PROXY_AUTH_MESSAGE = ("Automatic web access was refused by the organisation's network proxy because it requires "
                      "sign-in. This application does not bypass network authentication. Ask IT for an approved "
                      "proxy setting for this application (WEB_PROXY_URL). " + MANUAL_UPLOAD_HINT)
WEB_FILTER_MESSAGE = ("The organisation's web filter blocked one or more websites, so they were not read. "
                      + MANUAL_UPLOAD_HINT)
TLS_MESSAGE = ("Secure connections to websites could not be verified on this network. If the network inspects "
               "HTTPS traffic, ask IT for the organisation's certificate file and set WEB_CA_BUNDLE; certificate "
               "checks are never switched off. " + MANUAL_UPLOAD_HINT)
PROXY_UNREACHABLE_MESSAGE = ("The configured network proxy could not be reached. Check the proxy setting "
                             "(WEB_PROXY_MODE, WEB_PROXY_URL or the Windows proxy settings). " + MANUAL_UPLOAD_HINT)
DIRECT_UNREACHABLE_MESSAGE = ("Websites could not be reached directly from this computer and no network proxy is "
                              "configured for this application. If the network requires a proxy, set WEB_PROXY_URL "
                              "or use the Windows proxy settings. " + MANUAL_UPLOAD_HINT)
PAC_HINT = (" Windows uses an automatic proxy configuration script here, which this application does not read: "
            "ask IT which proxy address to set as WEB_PROXY_URL.")
CONNECTION_MESSAGE = "Some websites could not be reached over the network connection, so they were not read."
SEARCH_MESSAGE = ("The web search service could not be reached from this network, so no public sources were found "
                  "automatically. " + MANUAL_UPLOAD_HINT)
STOPPED_MESSAGE = ("Automatic web access was stopped for this analysis after repeated network failures, and the "
                   "analysis continued with the sources available.")
SIZE_LIMIT_MESSAGE = ("A document was larger than the configured size limit (MAX_DOWNLOAD_MB) and was not read. "
                      + MANUAL_UPLOAD_HINT)
PAGE_LIMIT_MESSAGE = ("A document had more pages than the configured limit (MAX_PDF_PAGES) and was not read. "
                      + MANUAL_UPLOAD_HINT)


def network_notices(web: WebAccessState) -> list[str]:
    """Most specific cause first; each message at most once."""
    issues = web.issues + web.search_issues
    out: list[str] = []
    if issues[policy.PROXY_AUTH_REQUIRED]:
        out.append(PROXY_AUTH_MESSAGE)
    if issues[policy.TLS_VERIFICATION_FAILED]:
        out.append(TLS_MESSAGE)
    if issues[policy.PROXY_CONNECTION_FAILED]:
        out.append(PROXY_UNREACHABLE_MESSAGE)
    if issues[policy.BLOCKED_BY_WEB_FILTER]:
        out.append(WEB_FILTER_MESSAGE)
    unreachable = sum(issues[c] for c in (policy.NETWORK_ERROR, policy.DNS_RESOLUTION_FAILED, policy.SOURCE_TIMEOUT))
    nothing_got_through = not web.fetch_successes and not web.search_successes
    if unreachable and not out:
        if nothing_got_through and network.proxy_source() == "none":
            out.append(DIRECT_UNREACHABLE_MESSAGE + (PAC_HINT if network.windows_autoconfig_url() else ""))
        elif web.fetch_failures or unreachable > 1:
            out.append(CONNECTION_MESSAGE)
    if web.search_issues and not web.search_successes and not any(m in out for m in (PROXY_AUTH_MESSAGE, DIRECT_UNREACHABLE_MESSAGE)):
        out.append(SEARCH_MESSAGE)
    if web.fetch_stopped or web.search_stopped:
        out.append(STOPPED_MESSAGE)
    if web.issues[policy.DOWNLOAD_SIZE_LIMIT_EXCEEDED]:
        out.append(SIZE_LIMIT_MESSAGE)
    if web.issues[policy.PDF_PAGE_LIMIT_EXCEEDED]:
        out.append(PAGE_LIMIT_MESSAGE)
    return out


def network_audit_details(web: WebAccessState) -> dict:
    return {
        "network": network.describe(),
        "fetch_issues": dict(web.issues),
        "search_issues": dict(web.search_issues),
        "fetch_successes": web.fetch_successes,
        "search_successes": web.search_successes,
        "fetch_stopped": web.fetch_stopped,
        "search_stopped": web.search_stopped,
        "searches_skipped": web.skipped_searches,
    }
