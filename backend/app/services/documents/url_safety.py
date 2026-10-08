"""URL validation to reduce SSRF risk when fetching public web content."""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {None, 80, 443}
BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain", "metadata.google.internal"}


class UnsafeURLError(ValueError):
    pass


def _ip_is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_public_url(url: str, *, resolve_dns: bool = True) -> str:
    """Return the normalised URL or raise UnsafeURLError.

    Rejects non-http(s) schemes, credentials in URLs, non-standard ports and hosts that
    resolve to private/loopback/link-local addresses. (DNS rebinding between this check and
    the request is still theoretically possible; acceptable for a local MVP.)
    """
    if not url or len(url) > 2000:
        raise UnsafeURLError("URL missing or too long")
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeURLError(f"Scheme not allowed: {parsed.scheme!r}")
    if parsed.username or parsed.password:
        raise UnsafeURLError("Credentials in URL are not allowed")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host or host in BLOCKED_HOSTNAMES or host.endswith(".local") or host.endswith(".internal"):
        raise UnsafeURLError(f"Host not allowed: {host!r}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise UnsafeURLError("Invalid port") from exc
    if port not in ALLOWED_PORTS:
        raise UnsafeURLError(f"Port not allowed: {port}")

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if not _ip_is_public(literal):
            raise UnsafeURLError("Private or reserved IP address")
    elif resolve_dns:
        try:
            infos = socket.getaddrinfo(host, port or (443 if parsed.scheme == "https" else 80))
        except socket.gaierror as exc:
            raise UnsafeURLError(f"Cannot resolve host {host!r}") from exc
        for info in infos:
            if not _ip_is_public(ipaddress.ip_address(info[4][0])):
                raise UnsafeURLError(f"Host {host!r} resolves to a non-public address")
    return parsed.geturl()


def normalize_website(website: str | None) -> str | None:
    if not website:
        return None
    website = website.strip()
    if not website:
        return None
    if "://" not in website:
        website = "https://" + website
    return website


def registrable_domain(url_or_host: str | None) -> str | None:
    """Very small heuristic: last two labels (last three for e.g. co.uk)."""
    if not url_or_host:
        return None
    host = urlparse(url_or_host).hostname if "://" in url_or_host else url_or_host
    if not host:
        return None
    labels = host.lower().strip(".").split(".")
    if labels and labels[0] == "www":
        labels = labels[1:]
    if len(labels) >= 3 and labels[-2] in {"co", "com", "org", "net", "ac", "gov"} and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else host
