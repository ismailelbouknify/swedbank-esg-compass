"""Central URL and SSRF policy. Every external URL is validated here before any request is made,
and again for every redirect target."""
from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

from app.core.config import get_settings
from app.services.security import policy
from app.services.security.policy import SourceRejection, run_bounded

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

ALLOWED_PORTS = {None, 80, 443}
MAX_URL_LENGTH = 2000
BLOCKED_HOSTNAMES = {
    "localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback", "broadcasthost",
    "metadata", "metadata.google.internal", "instance-data",
}
BLOCKED_SUFFIXES = (".localhost", ".local", ".internal", ".intranet", ".lan", ".corp", ".home", ".arpa")
# Public-range addresses that still reach cloud or legacy infrastructure.
EXTRA_BLOCKED_NETWORKS = tuple(ipaddress.ip_network(n) for n in (
    "168.63.129.16/32",  # Azure platform endpoint
    "192.88.99.0/24",  # deprecated 6to4 relay anycast
))
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_UNSAFE_URL_CHARS = re.compile(r"[\s\\\x00-\x1f\x7f]")
# URL parsers and resolvers read a host ending in a number as IPv4: 2130706433, 0x7f.1, 127.1
_NUMERIC_LAST_LABEL = re.compile(r"^(?:\d+|0x[0-9a-f]*)$")


class UnsafeURLError(SourceRejection):
    pass


@dataclass(frozen=True)
class ValidatedURL:
    url: str
    scheme: str
    host: str
    port: int | None
    addresses: tuple[str, ...]  # validated public IPs; empty when DNS was not resolved


def is_public_address(ip: IPAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is None and ip in _NAT64:
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if embedded is not None:
            return is_public_address(embedded)
    if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
            or ip.is_unspecified or not ip.is_global):
        return False
    return not any(ip in net for net in EXTRA_BLOCKED_NETWORKS if net.version == ip.version)


def validate_resolved_address(address: str) -> None:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError as exc:
        raise UnsafeURLError(policy.UNSAFE_HOST, "Unparseable resolved address") from exc
    if not is_public_address(ip):
        raise UnsafeURLError(policy.PRIVATE_IP, "Host is or resolves to a non-public address")


def resolve_host(host: str, port: int | None, scheme: str) -> tuple[str, ...]:
    """Resolve a hostname and require every returned address to be public. IPv4 is listed first."""
    default_port = 443 if scheme == "https" else 80
    try:
        infos = run_bounded(lambda: socket.getaddrinfo(host, port or default_port, type=socket.SOCK_STREAM),
                            get_settings().http_connect_timeout_seconds)
    except (OSError, UnicodeError, TimeoutError) as exc:
        raise UnsafeURLError(policy.DNS_RESOLUTION_FAILED, f"Cannot resolve host {host!r}", retryable=True) from exc
    addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    if not addresses:
        raise UnsafeURLError(policy.DNS_RESOLUTION_FAILED, f"No addresses for host {host!r}", retryable=True)
    for address in addresses:
        validate_resolved_address(address)
    return tuple(sorted(addresses, key=lambda a: ":" in a))


def validate_domain_policy(host: str) -> None:
    settings = get_settings()
    allowed = settings.allowed_domains
    if not allowed:
        return
    subdomains = settings.allowed_domains_include_subdomains
    if not any(host == d or (subdomains and host.endswith("." + d)) for d in allowed):
        raise UnsafeURLError(policy.DOMAIN_NOT_ALLOWED, f"Host {host!r} is not on the configured allowlist")


def validate_external_url(url: str, *, resolve_dns: bool = True, enforce_domain_policy: bool = True) -> ValidatedURL:
    """Return the validated target or raise UnsafeURLError.

    Checks scheme, credentials, hostname, port, IP literals, the optional domain allowlist and,
    unless `resolve_dns` is off, every address the hostname resolves to."""
    url = (url or "").strip()
    if not url or len(url) > MAX_URL_LENGTH or _UNSAFE_URL_CHARS.search(url):
        raise UnsafeURLError(policy.UNSAFE_URL, "URL missing, too long or containing forbidden characters")
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
        # canonical form before any check: full-width characters in a name can spell "localhost"
        if not host.isascii():
            host = host.encode("idna").decode("ascii").lower()
    except ValueError as exc:  # includes UnicodeError
        raise UnsafeURLError(policy.UNSAFE_URL, "Malformed URL") from exc
    scheme = parsed.scheme.lower()
    if scheme == "http":
        if not get_settings().http_allowed:
            raise UnsafeURLError(policy.HTTP_NOT_ALLOWED, "Plain HTTP is not allowed")
    elif scheme != "https":
        raise UnsafeURLError(policy.UNSAFE_SCHEME, f"Scheme not allowed: {scheme!r}")
    if parsed.username or parsed.password:
        raise UnsafeURLError(policy.CREDENTIALS_IN_URL, "Credentials in URL are not allowed")
    if port not in ALLOWED_PORTS:
        raise UnsafeURLError(policy.UNSAFE_PORT, f"Port not allowed: {port}")
    if not host:
        raise UnsafeURLError(policy.UNSAFE_HOST, "URL has no host")

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if not is_public_address(literal):
            raise UnsafeURLError(policy.PRIVATE_IP, "Host is or resolves to a non-public address")
    else:
        labels = host.split(".")
        if (host in BLOCKED_HOSTNAMES or host.endswith(BLOCKED_SUFFIXES) or len(labels) < 2
                or _NUMERIC_LAST_LABEL.match(labels[-1])):
            raise UnsafeURLError(policy.UNSAFE_HOST, f"Host not allowed: {host!r}")
    if enforce_domain_policy:
        validate_domain_policy(host)

    if literal is not None:
        addresses: tuple[str, ...] = (str(literal),)
    else:
        addresses = resolve_host(host, port, scheme) if resolve_dns else ()
    return ValidatedURL(url=parsed.geturl(), scheme=scheme, host=host, port=port, addresses=addresses)


def validate_public_url(url: str, *, resolve_dns: bool = True) -> str:
    """Return the normalised URL or raise UnsafeURLError."""
    return validate_external_url(url, resolve_dns=resolve_dns).url


def is_loopback_service_url(url: str) -> bool:
    """True when a configured local service (e.g. Ollama) really is on this machine."""
    try:
        host = (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
