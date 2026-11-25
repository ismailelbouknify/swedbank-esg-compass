"""How this application reaches the public web from the network it runs on: which proxy, which certificates,
and how to tell a refusal by the organisation's network apart from an unavailable website.

Nothing here weakens a check or works around a control:
- Certificates are always verified. The only choice is which trusted authorities are used: the organisation's
  bundle (WEB_CA_BUNDLE), else the operating system's store (so an authority deployed by IT is trusted the way
  the browser trusts it), else the certifi bundle.
- Only a proxy the user or IT already configured is used. The application does not perform Windows integrated
  (NTLM/Kerberos) proxy sign-in, never supplies credentials of its own and never routes around a proxy or filter.
  When the network refuses a request, that is reported, and the analyst can upload the report instead.

WEB_PROXY_MODE
  system    (default) HTTPS_PROXY / HTTP_PROXY / NO_PROXY, otherwise the Windows Internet Options proxy.
            Automatic configuration scripts (PAC/WPAD) are not evaluated: set WEB_PROXY_URL instead.
  explicit  WEB_PROXY_URL for every request.
  none      direct connections only.
"""
from __future__ import annotations

import re
import ssl
import urllib.request
from functools import lru_cache
from urllib.parse import urlsplit

from app.core.config import get_settings
from app.services.security import policy
from app.services.security.policy import run_bounded

PROXY_MODES = ("system", "explicit", "none")
SUPPORTED_PROXY_SCHEMES = ("http", "https", "socks5", "socks5h")
BLOCK_PAGE_SCAN_BYTES = 64 * 1024

# Text that web filters and authenticating proxies put on the pages they return instead of the site.
_FILTER_MARKERS = (
    "blue coat", "bluecoat", "proxysg", "symantec web", "web filter", "webfilter", "zscaler", "forcepoint", "websense",
    "fortiguard", "mcafee web gateway", "skyhigh", "netskope", "url filtering", "web gateway",
)
_AUTH_MARKERS = ("authentication_failed", "credentials are missing", "proxy authentication", "authentication required")
# Blue Coat / Symantec ProxySG exception ids, shown in brackets on their pages: specific enough on their own
_EXCEPTION_ID = re.compile(rb"\(\s*(authentication_failed|policy_denied|content_filter_denied|policy_redirect"
                           rb"|dns_unresolved_hostname|tcp_error|ssl_failed|connect_failed)\s*\)")


def proxy_mode() -> str:
    mode = (get_settings().web_proxy_mode or "system").strip().lower()
    return mode if mode in PROXY_MODES else "system"


def _system_proxies() -> dict[str, str]:
    """Environment variables when any are set, otherwise the Windows Internet Options proxy (registry)."""
    try:
        return urllib.request.getproxies()
    except Exception:  # noqa: BLE001 — an unreadable proxy setting means no proxy, never a crash
        return {}


def _system_bypass(host: str) -> bool:
    """NO_PROXY, or the Windows bypass list (which can look the name up, so it is bounded)."""
    try:
        return bool(run_bounded(lambda: urllib.request.proxy_bypass(host), get_settings().http_connect_timeout_seconds))
    except Exception:  # noqa: BLE001
        return False


def proxy_source() -> str:
    """Where the proxy setting comes from: explicit | environment | windows_settings | none."""
    mode = proxy_mode()
    if mode == "none":
        return "none"
    if mode == "explicit":
        return "explicit" if get_settings().web_proxy_url.strip() else "none"
    try:
        if urllib.request.getproxies_environment():
            return "environment"
    except Exception:  # noqa: BLE001
        pass
    return "windows_settings" if _system_proxies() else "none"


def configured_proxy(scheme: str = "https") -> str | None:
    """The proxy for a scheme, before per-host bypass rules."""
    mode = proxy_mode()
    if mode == "none":
        return None
    if mode == "explicit":
        return get_settings().web_proxy_url.strip() or None
    proxies = _system_proxies()
    return proxies.get(scheme) or proxies.get("all") or None


def proxy_for(url: str) -> str | None:
    """The proxy a request to `url` goes through, or None for a direct connection."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    proxy = configured_proxy(parts.scheme.lower() or "https")
    if not proxy or not is_supported_proxy(proxy):
        return None
    if proxy_mode() == "system" and parts.hostname and _system_bypass(parts.hostname):
        return None
    return proxy


def is_supported_proxy(proxy: str) -> bool:
    try:
        return urlsplit(proxy).scheme.lower() in SUPPORTED_PROXY_SCHEMES
    except ValueError:
        return False


def mask_proxy(proxy: str | None) -> str:
    """A proxy address safe to log or show: never its credentials."""
    if not proxy:
        return "none"
    try:
        p = urlsplit(proxy)
        port = f":{p.port}" if p.port else ""
        return f"{p.scheme}://{'***@' if (p.username or p.password) else ''}{p.hostname or ''}{port}"
    except ValueError:
        return "<unparseable proxy setting>"


def windows_autoconfig_url() -> str | None:
    """A PAC script configured in Windows Internet Options, which this application does not evaluate."""
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
            return str(winreg.QueryValueEx(key, "AutoConfigURL")[0]) or None
    except OSError:
        return None


# ---------------------------------------------------------------- certificates
def trust_source() -> str:
    """ca_bundle | system | certifi: which authorities verify server certificates."""
    s = get_settings()
    if s.web_ca_bundle.strip():
        return "ca_bundle"
    if s.use_system_trust_store and _truststore_available():
        return "system"
    return "certifi"


def _truststore_available() -> bool:
    try:
        import truststore  # noqa: F401
    except ImportError:
        return False
    return True


def ssl_context() -> ssl.SSLContext:
    """Always verifying (CERT_REQUIRED, hostname checked). Raises OSError/SSLError for an unreadable CA bundle."""
    s = get_settings()
    return _ssl_context(s.web_ca_bundle.strip(), trust_source())


@lru_cache(maxsize=8)
def _ssl_context(ca_bundle: str, source: str) -> ssl.SSLContext:
    if source == "ca_bundle":
        ctx = ssl.create_default_context(cafile=ca_bundle)
    elif source == "system":
        import truststore

        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    else:
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    return ctx


def search_client_verify() -> bool | str:
    """For search clients with their own TLS stack: the CA bundle when one is configured, else default verification."""
    bundle = get_settings().web_ca_bundle.strip()
    return bundle or True


# ---------------------------------------------------------------- telling refusals apart
def is_certificate_failure(exc: BaseException) -> bool:
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(cur):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def classify_block_page(status: int, body: bytes) -> str | None:
    """A page returned by the organisation's proxy or web filter instead of the site, or None.
    Successful responses need a filter-specific exception id, so an article about web filters is still an article."""
    if status == 407:
        return policy.PROXY_AUTH_REQUIRED
    text = body[:BLOCK_PAGE_SCAN_BYTES].lower()
    exception_id = _EXCEPTION_ID.search(text)
    if 200 <= status < 300:
        if not exception_id and not (b"access denied" in text and (b"blue coat" in text or b"bluecoat" in text)):
            return None
    elif not exception_id and not any(m.encode() in text for m in _FILTER_MARKERS):
        return None
    if any(m.encode() in text for m in _AUTH_MARKERS):
        return policy.PROXY_AUTH_REQUIRED
    return policy.BLOCKED_BY_WEB_FILTER


def classify_search_error(exc: BaseException) -> str:
    """Search clients raise their own exception types; their text says what the network did."""
    if is_certificate_failure(exc):
        return policy.TLS_VERIFICATION_FAILED
    text = f"{type(exc).__name__} {exc}".lower()
    if "407" in text or "proxy authentication" in text or "authentication_failed" in text:
        return policy.PROXY_AUTH_REQUIRED
    if "certificate" in text:
        return policy.TLS_VERIFICATION_FAILED
    if "proxy" in text:
        return policy.PROXY_CONNECTION_FAILED
    if isinstance(exc, TimeoutError) or "timed out" in text or "timeout" in text:
        return policy.SOURCE_TIMEOUT
    return policy.NETWORK_ERROR


def describe() -> dict[str, str]:
    """Network configuration as it may be logged: no credentials."""
    return {
        "proxy_mode": proxy_mode(),
        "proxy_source": proxy_source(),
        "https_proxy": mask_proxy(configured_proxy("https")),
        "http_proxy": mask_proxy(configured_proxy("http")),
        "pac_script_configured": "yes" if windows_autoconfig_url() else "no",
        "certificates": trust_source(),
    }
