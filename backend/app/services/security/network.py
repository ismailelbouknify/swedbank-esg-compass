"""How this application reaches the public web from the network it runs on: which proxy, which certificates,
and how to tell a refusal by the organisation's network apart from an unavailable website.

Nothing here weakens a check or works around a control:
- Certificates are always verified. The only choice is which trusted authorities are used: the organisation's
  bundle (WEB_CA_BUNDLE), else the operating system's store (so an authority deployed by IT is trusted the way
  the browser trusts it), else the certifi bundle.
- Only a proxy the user or IT already configured is used, and never routed around. When the network refuses a
  request, that is reported, and the analyst can upload the report instead.
- Integrated proxy sign-in (WEB_PROXY_AUTH=integrated, off by default) uses a Kerberos ticket from the signed-in
  Windows user's session, only when the proxy asks for sign-in. No credentials are read or stored; NTLM is not used.

WEB_PROXY_MODE
  system    (default) HTTPS_PROXY / HTTP_PROXY / NO_PROXY; otherwise Windows Internet Options: the automatic
            configuration script (PAC, or WPAD auto-detection) evaluated by Windows itself, then the static proxy.
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
from app.services.security import policy, windows_proxy
from app.services.security.policy import run_bounded

PROXY_MODES = ("system", "explicit", "none")
PROXY_AUTH_MODES = ("none", "integrated")
_pac_cache = windows_proxy.PacCache()
_pac_state: dict[str, str] = {"last_error": "", "last_proxy": ""}
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


def _environment_proxies() -> dict[str, str]:
    try:
        return urllib.request.getproxies_environment()
    except Exception:  # noqa: BLE001
        return {}


def proxy_auth_mode() -> str:
    mode = (get_settings().web_proxy_auth or "none").strip().lower()
    return mode if mode in PROXY_AUTH_MODES else "none"


def windows_config() -> windows_proxy.WindowsProxyConfig | None:
    try:
        return windows_proxy.read_windows_config()
    except Exception:  # noqa: BLE001 — an unreadable setting means none, never a crash
        return None


def pac_configuration() -> tuple[str | None, bool]:
    """(PAC script address, WPAD auto-detection) that system mode evaluates; (None, False) when there is none.
    WEB_PROXY_PAC_URL replaces the Windows setting, e.g. for a service account without the user's settings."""
    if proxy_mode() != "system" or _environment_proxies():
        return None, False
    explicit = get_settings().web_proxy_pac_url.strip()
    if explicit:
        return explicit, False
    cfg = windows_config()
    if cfg is None:
        return None, False
    return cfg.pac_url, cfg.auto_detect


def pac_route(url: str) -> windows_proxy.PacResult | None:
    """What the automatic proxy configuration says for `url`, or None when there is none to evaluate."""
    pac_url, auto_detect = pac_configuration()
    if not pac_url and not auto_detect:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    key = (pac_url, auto_detect, parts.scheme.lower(), (parts.hostname or "").lower())
    cached = _pac_cache.get(key)
    if cached is not None:
        return cached
    timeout = get_settings().http_connect_timeout_seconds
    try:
        result = run_bounded(lambda: windows_proxy.evaluate_pac(url, pac_url=pac_url, auto_detect=auto_detect,
                                                                timeout_seconds=timeout), timeout + 5)
    except Exception as exc:  # noqa: BLE001
        result = windows_proxy.PacResult(False, error=f"evaluation did not finish ({type(exc).__name__})")
    _pac_cache.put(key, result)
    if result.ok:
        _pac_state["last_error"] = ""
        _pac_state["last_proxy"] = result.proxies[0] if result.proxies else "DIRECT"
    elif not result.not_found:
        _pac_state["last_error"] = result.error
    return result


def pac_status() -> dict[str, str]:
    """What the automatic configuration is and what it last answered; proxy addresses masked like any other."""
    pac_url, auto_detect = pac_configuration()
    last = _pac_state["last_proxy"]
    shown = "not evaluated" if not last else last if last == "DIRECT" else mask_proxy("http://" + last)
    return {"pac": "script" if pac_url else "auto_detect" if auto_detect else "none",
            "pac_last_result": shown, "pac_last_error": _pac_state["last_error"] or "none"}


_signin_state: dict[str, str] = {"last_problem": ""}
_signin_proxies: set[str] = set()  # proxies that asked for sign-in in this process: later connections sign in at once


def integrated_signin_enabled(proxy: str | None) -> bool:
    if not proxy or proxy_auth_mode() != "integrated":
        return False
    try:
        return urlsplit(proxy).scheme.lower() in ("http", "https") and bool(urlsplit(proxy).hostname)
    except ValueError:
        return False


def proxy_signin_header(proxy: str) -> dict[str, str]:
    """A fresh `Proxy-Authorization: Negotiate <Kerberos ticket>` for one connection to `proxy`, from the signed-in
    Windows user's session. Raises windows_proxy.IntegratedAuthUnavailable. Never logged."""
    host = urlsplit(proxy).hostname or ""
    try:
        token = run_bounded(lambda: windows_proxy.kerberos_proxy_token(host), get_settings().http_connect_timeout_seconds)
    except windows_proxy.IntegratedAuthUnavailable as exc:
        note_signin_problem(exc.reason)
        raise
    except Exception as exc:  # noqa: BLE001 — e.g. no domain controller answered in time
        note_signin_problem(f"no Kerberos ticket could be obtained ({type(exc).__name__})")
        raise windows_proxy.IntegratedAuthUnavailable(_signin_state["last_problem"]) from exc
    return {"Proxy-Authorization": f"Negotiate {token}"}


def note_signin_problem(reason: str) -> None:
    _signin_state["last_problem"] = reason


def last_signin_problem() -> str:
    return _signin_state["last_problem"]


def remember_signin_required(proxy: str) -> None:
    _signin_proxies.add(proxy)


def signin_known_required(proxy: str) -> bool:
    return proxy in _signin_proxies


def reset_signin_state() -> None:
    _signin_proxies.clear()
    _signin_state["last_problem"] = ""


def pac_failed() -> bool:
    return bool(_pac_state["last_error"])


def reset_pac_cache() -> None:
    _pac_cache.clear()
    _pac_state.update(last_error="", last_proxy="")


def proxy_source() -> str:
    """Where the proxy setting comes from: explicit | environment | windows_pac | windows_settings |
    windows_auto_detect | none."""
    mode = proxy_mode()
    if mode == "none":
        return "none"
    if mode == "explicit":
        return "explicit" if get_settings().web_proxy_url.strip() else "none"
    if _environment_proxies():
        return "environment"
    pac_url, auto_detect = pac_configuration()
    if pac_url:
        return "windows_pac"
    if _system_proxies():
        return "windows_settings"
    return "windows_auto_detect" if auto_detect else "none"


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
    route = pac_route(url)
    if route is not None and route.ok:  # the script decides, including DIRECT; its own exceptions are part of it
        return f"http://{route.proxies[0]}" if route.proxies else None
    # no script, or it could not be evaluated: the static setting applies, as it does for Windows itself
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
    if isinstance(exc, windows_proxy.IntegratedAuthUnavailable):
        return policy.PROXY_SIGNIN_UNAVAILABLE
    if is_certificate_failure(exc):
        return policy.TLS_VERIFICATION_FAILED
    text = f"{type(exc).__name__} {exc}".lower()
    if "integrated sign-in refused" in text:
        return policy.PROXY_SIGNIN_REFUSED
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
        **pac_status(),
        "proxy_auth": proxy_auth_mode(),
        "certificates": trust_source(),
    }
