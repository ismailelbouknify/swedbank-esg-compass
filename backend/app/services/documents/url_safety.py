"""Website and domain helpers. URL security policy lives in app.services.security.url_validator."""
from __future__ import annotations

from urllib.parse import urlparse

from app.services.security.url_validator import UnsafeURLError, validate_public_url

__all__ = ["UnsafeURLError", "normalize_website", "registrable_domain", "validate_public_url"]


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
