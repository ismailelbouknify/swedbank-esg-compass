"""Windows' own proxy machinery, used through the operating system rather than re-implemented:

- PAC / WPAD: WinHTTP evaluates the automatic proxy configuration script exactly as Windows does (the script
  named in Internet Options, or found by "Automatically detect settings"). This application never downloads,
  parses or edits the script itself.
- Integrated proxy sign-in: SSPI issues a Kerberos ticket for the proxy from the signed-in Windows user's
  session. No username, password or hash is read, asked for or stored; the token is valid only for that proxy
  and is used for one connection.

NTLM is deliberately not produced: it needs a challenge/response on a single kept-open connection, and NTLM
tokens can be relayed. If Windows can only offer NTLM for the proxy, that is reported, not worked around.
Everything here is Windows-only and returns "unavailable" elsewhere.
"""
from __future__ import annotations

import base64
import ctypes
import ipaddress
import sys
import threading
import time
from dataclasses import dataclass

IS_WINDOWS = sys.platform == "win32"

# ---------------------------------------------------------------- WinHTTP (PAC / WPAD)
WINHTTP_ACCESS_TYPE_NO_PROXY = 1
WINHTTP_ACCESS_TYPE_NAMED_PROXY = 3
WINHTTP_AUTOPROXY_AUTO_DETECT = 0x1
WINHTTP_AUTOPROXY_CONFIG_URL = 0x2
WINHTTP_AUTO_DETECT_TYPE_DHCP = 0x1
WINHTTP_AUTO_DETECT_TYPE_DNS_A = 0x2
ERROR_WINHTTP_LOGIN_FAILURE = 12015
ERROR_WINHTTP_AUTODETECTION_FAILED = 12180
PAC_ERRORS = {
    12015: "the PAC script server asked for a sign-in that failed",
    12166: "the PAC script has an error",
    12167: "the PAC script could not be downloaded",
    12178: "the Windows proxy auto-configuration service reported an error",
    12180: "no proxy configuration was found by automatic detection (WPAD)",
    12005: "the PAC script address is not valid",
    12006: "the PAC script address is not valid (WinHTTP does not read file:// scripts)",
    12002: "downloading the PAC script timed out",
}


@dataclass(frozen=True)
class WindowsProxyConfig:
    auto_detect: bool
    pac_url: str | None
    static_proxy: str | None
    bypass: str | None


@dataclass(frozen=True)
class PacResult:
    ok: bool
    proxies: tuple[str, ...] = ()  # "host:port" in PAC order; empty with ok=True means DIRECT
    error: str = ""  # plain reason when not ok
    not_found: bool = False  # WPAD found no configuration: not a failure, Windows then uses the static setting

    @property
    def direct(self) -> bool:
        return self.ok and not self.proxies


if IS_WINDOWS:
    from ctypes import wintypes as _w

    class _IEConfig(ctypes.Structure):
        _fields_ = [("fAutoDetect", _w.BOOL), ("lpszAutoConfigUrl", ctypes.c_void_p),
                    ("lpszProxy", ctypes.c_void_p), ("lpszProxyBypass", ctypes.c_void_p)]

    class _AutoProxyOptions(ctypes.Structure):
        _fields_ = [("dwFlags", _w.DWORD), ("dwAutoDetectFlags", _w.DWORD), ("lpszAutoConfigUrl", _w.LPCWSTR),
                    ("lpvReserved", ctypes.c_void_p), ("dwReserved", _w.DWORD), ("fAutoLogonIfChallenged", _w.BOOL)]

    class _ProxyInfo(ctypes.Structure):
        _fields_ = [("dwAccessType", _w.DWORD), ("lpszProxy", ctypes.c_void_p), ("lpszProxyBypass", ctypes.c_void_p)]

    _winhttp = ctypes.WinDLL("winhttp", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _winhttp.WinHttpOpen.restype = ctypes.c_void_p
    _winhttp.WinHttpOpen.argtypes = [_w.LPCWSTR, _w.DWORD, _w.LPCWSTR, _w.LPCWSTR, _w.DWORD]
    _winhttp.WinHttpCloseHandle.argtypes = [ctypes.c_void_p]
    _winhttp.WinHttpSetTimeouts.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _winhttp.WinHttpGetProxyForUrl.argtypes = [ctypes.c_void_p, _w.LPCWSTR, ctypes.POINTER(_AutoProxyOptions),
                                               ctypes.POINTER(_ProxyInfo)]
    _winhttp.WinHttpGetIEProxyConfigForCurrentUser.argtypes = [ctypes.POINTER(_IEConfig)]
    _kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    _kernel32.GlobalFree.restype = ctypes.c_void_p


def _take_string(ptr: int | None) -> str | None:
    """Copy a WinHTTP-allocated string and free it."""
    if not ptr:
        return None
    try:
        return ctypes.wstring_at(ptr) or None
    finally:
        _kernel32.GlobalFree(ptr)


def read_windows_config() -> WindowsProxyConfig | None:
    """The current user's Internet Options proxy settings, or None when not on Windows / unreadable."""
    if not IS_WINDOWS:
        return None
    cfg = _IEConfig()
    if not _winhttp.WinHttpGetIEProxyConfigForCurrentUser(ctypes.byref(cfg)):
        return None
    return WindowsProxyConfig(auto_detect=bool(cfg.fAutoDetect), pac_url=_take_string(cfg.lpszAutoConfigUrl),
                              static_proxy=_take_string(cfg.lpszProxy), bypass=_take_string(cfg.lpszProxyBypass))


def parse_proxy_list(value: str | None, scheme: str) -> tuple[str, ...]:
    """WinHTTP proxy lists: "a:8080;b:8080", "http=a:80;https=b:443", or PAC-style "PROXY a:8080; DIRECT".
    SOCKS entries are skipped; DIRECT entries end the list (everything after them is a fallback of its own)."""
    out: list[str] = []
    for raw in (value or "").replace(",", ";").split(";"):
        entry = raw.strip()
        if not entry:
            continue
        upper = entry.upper()
        if upper == "DIRECT":
            break
        if upper.startswith(("SOCKS", "SOCKS4 ", "SOCKS5 ")):
            continue
        for prefix in ("PROXY ", "HTTPS ", "HTTP "):
            if upper.startswith(prefix):
                entry = entry[len(prefix):].strip()
                break
        if "=" in entry:
            key, _, entry = entry.partition("=")
            if key.strip().lower() != scheme:
                continue
        entry = entry.strip()
        if "://" in entry:
            entry = entry.split("://", 1)[1]
        if entry and entry not in out:
            out.append(entry.rstrip("/"))
    return tuple(out)


def evaluate_pac(url: str, *, pac_url: str | None, auto_detect: bool, timeout_seconds: float) -> PacResult:
    """Ask WinHTTP which proxy Windows would use for `url`. Blocking; callers bound it."""
    if not IS_WINDOWS:
        return PacResult(False, error="automatic proxy scripts are only evaluated on Windows")
    if not pac_url and not auto_detect:
        return PacResult(False, error="no automatic proxy configuration")
    session = _winhttp.WinHttpOpen("ESG-Compass-ProxyResolver", WINHTTP_ACCESS_TYPE_NO_PROXY, None, None, 0)
    if not session:
        return PacResult(False, error=f"WinHTTP could not start (error {ctypes.get_last_error()})")
    try:
        ms = max(1000, int(timeout_seconds * 1000))
        _winhttp.WinHttpSetTimeouts(session, ms, ms, ms, ms)
        opts = _AutoProxyOptions()
        if pac_url:
            opts.dwFlags = WINHTTP_AUTOPROXY_CONFIG_URL
            opts.lpszAutoConfigUrl = pac_url
        else:
            opts.dwFlags = WINHTTP_AUTOPROXY_AUTO_DETECT
            opts.dwAutoDetectFlags = WINHTTP_AUTO_DETECT_TYPE_DHCP | WINHTTP_AUTO_DETECT_TYPE_DNS_A
        info = _ProxyInfo()
        ok = _winhttp.WinHttpGetProxyForUrl(session, url, ctypes.byref(opts), ctypes.byref(info))
        err = ctypes.get_last_error()
        if not ok and err == ERROR_WINHTTP_LOGIN_FAILURE:
            # as Microsoft documents: retry with the user's Windows sign-in for the server that hosts the script
            opts.fAutoLogonIfChallenged = True
            ok = _winhttp.WinHttpGetProxyForUrl(session, url, ctypes.byref(opts), ctypes.byref(info))
            err = ctypes.get_last_error()
        if not ok:
            return PacResult(False, error=PAC_ERRORS.get(err, f"WinHTTP error {err}"),
                             not_found=err == ERROR_WINHTTP_AUTODETECTION_FAILED and not pac_url)
        proxy = _take_string(info.lpszProxy)
        _take_string(info.lpszProxyBypass)
        if info.dwAccessType != WINHTTP_ACCESS_TYPE_NAMED_PROXY:
            return PacResult(True)
        scheme = url.split(":", 1)[0].lower()
        return PacResult(True, parse_proxy_list(proxy, scheme))
    finally:
        _winhttp.WinHttpCloseHandle(session)


class PacCache:
    """PAC answers depend on the host (and scheme), rarely on the path: cache per (scheme, host)."""

    def __init__(self, ttl_seconds: float = 300, failure_ttl_seconds: float = 120):
        self.ttl, self.failure_ttl = ttl_seconds, failure_ttl_seconds
        self._items: dict[tuple, tuple[float, PacResult]] = {}
        self._lock = threading.Lock()

    def get(self, key: tuple) -> PacResult | None:
        with self._lock:
            hit = self._items.get(key)
            if hit and hit[0] > time.monotonic():
                return hit[1]
            return None

    def put(self, key: tuple, result: PacResult) -> None:
        with self._lock:
            self._items[key] = (time.monotonic() + (self.ttl if result.ok else self.failure_ttl), result)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


# ---------------------------------------------------------------- SSPI (integrated proxy sign-in)
SEC_E_OK = 0
SEC_I_CONTINUE_NEEDED = 0x00090312
SECPKG_CRED_OUTBOUND = 2
SECURITY_NATIVE_DREP = 0x10
SECBUFFER_TOKEN = 2
ISC_REQ_ALLOCATE_MEMORY = 0x100
SSPI_ERRORS = {
    0x80090303: "Windows has no Kerberos service name for this proxy (SEC_E_TARGET_UNKNOWN)",
    0x8009030E: "this Windows session has no credentials that can be used (SEC_E_NO_CREDENTIALS)",
    0x80090311: "no domain controller could be reached for a Kerberos ticket (SEC_E_NO_AUTHENTICATING_AUTHORITY)",
    0x8009030C: "the Windows sign-in was refused (SEC_E_LOGON_DENIED)",
    0x80090305: "Kerberos/Negotiate is not available on this computer (SEC_E_SECPKG_NOT_FOUND)",
    0x80090322: "the Kerberos service name for this proxy is wrong (SEC_E_WRONG_PRINCIPAL)",
}


class IntegratedAuthUnavailable(Exception):
    """No Kerberos token for the proxy could be produced from the Windows session. `reason` is plain language."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


if IS_WINDOWS:
    class _SecHandle(ctypes.Structure):
        _fields_ = [("dwLower", ctypes.c_void_p), ("dwUpper", ctypes.c_void_p)]

    class _SecBuffer(ctypes.Structure):
        _fields_ = [("cbBuffer", ctypes.c_ulong), ("BufferType", ctypes.c_ulong), ("pvBuffer", ctypes.c_void_p)]

    class _SecBufferDesc(ctypes.Structure):
        _fields_ = [("ulVersion", ctypes.c_ulong), ("cBuffers", ctypes.c_ulong), ("pBuffers", ctypes.POINTER(_SecBuffer))]

    _secur32 = ctypes.WinDLL("secur32")
    _secur32.AcquireCredentialsHandleW.restype = ctypes.c_long
    _secur32.AcquireCredentialsHandleW.argtypes = [
        ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.POINTER(_SecHandle), ctypes.POINTER(ctypes.c_int64)]
    _secur32.InitializeSecurityContextW.restype = ctypes.c_long
    _secur32.InitializeSecurityContextW.argtypes = [
        ctypes.POINTER(_SecHandle), ctypes.POINTER(_SecHandle), ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong,
        ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(_SecHandle), ctypes.POINTER(_SecBufferDesc),
        ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int64)]
    _secur32.FreeContextBuffer.argtypes = [ctypes.c_void_p]
    _secur32.DeleteSecurityContext.argtypes = [ctypes.POINTER(_SecHandle)]
    _secur32.FreeCredentialsHandle.argtypes = [ctypes.POINTER(_SecHandle)]


def _sspi_error(status: int) -> str:
    code = status & 0xFFFFFFFF
    return SSPI_ERRORS.get(code, f"Windows security error 0x{code:08X}")


def is_ntlm_token(token: bytes) -> bool:
    """SPNEGO falls back to NTLM when Kerberos is unavailable; an NTLM message carries this signature."""
    return b"NTLMSSP\x00" in token


def _sspi_token(spn: str) -> bytes:
    cred, ctx = _SecHandle(), _SecHandle()
    expiry = ctypes.c_int64()
    status = _secur32.AcquireCredentialsHandleW(None, "Negotiate", SECPKG_CRED_OUTBOUND, None, None, None, None,
                                                ctypes.byref(cred), ctypes.byref(expiry))
    if status != SEC_E_OK:
        raise IntegratedAuthUnavailable(_sspi_error(status))
    try:
        out_buf = _SecBuffer(0, SECBUFFER_TOKEN, None)
        out_desc = _SecBufferDesc(0, 1, ctypes.pointer(out_buf))
        attrs = ctypes.c_ulong()
        status = _secur32.InitializeSecurityContextW(
            ctypes.byref(cred), None, spn, ISC_REQ_ALLOCATE_MEMORY, 0, SECURITY_NATIVE_DREP, None, 0,
            ctypes.byref(ctx), ctypes.byref(out_desc), ctypes.byref(attrs), ctypes.byref(expiry))
        try:
            if status not in (SEC_E_OK, SEC_I_CONTINUE_NEEDED) or not out_buf.pvBuffer:
                raise IntegratedAuthUnavailable(_sspi_error(status))
            return ctypes.string_at(out_buf.pvBuffer, out_buf.cbBuffer)
        finally:
            if out_buf.pvBuffer:
                _secur32.FreeContextBuffer(out_buf.pvBuffer)
            if ctx.dwLower or ctx.dwUpper:
                _secur32.DeleteSecurityContext(ctypes.byref(ctx))
    finally:
        _secur32.FreeCredentialsHandle(ctypes.byref(cred))


def kerberos_proxy_token(proxy_host: str) -> str:
    """A base64 Negotiate token carrying a Kerberos ticket for HTTP/<proxy_host>, from the current Windows session.
    Raises IntegratedAuthUnavailable with a plain reason. The token is never logged."""
    if not IS_WINDOWS:
        raise IntegratedAuthUnavailable("integrated proxy sign-in uses Windows and is not available on this system")
    host = (proxy_host or "").strip().lower().rstrip(".")
    if not host:
        raise IntegratedAuthUnavailable("the proxy has no host name")
    try:
        ipaddress.ip_address(host.strip("[]"))
        raise IntegratedAuthUnavailable("the proxy is given as an IP address; Kerberos needs the proxy's host name "
                                        "(ask IT for the name, e.g. in WEB_PROXY_URL or the PAC script)")
    except ValueError:
        pass
    token = _sspi_token(f"HTTP/{host}")
    if is_ntlm_token(token):
        raise IntegratedAuthUnavailable("Windows could only offer NTLM for this proxy, not Kerberos (the computer may "
                                        "not be signed in to the corporate domain, or the proxy has no Kerberos "
                                        "service name); NTLM sign-in is not used by this application")
    return base64.b64encode(token).decode("ascii")
