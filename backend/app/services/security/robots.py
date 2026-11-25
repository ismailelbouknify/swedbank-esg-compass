"""robots.txt, enforced for every automated request (RFC 9309).

safe_fetch asks `enforce(url)` before each request it sends, including every redirect hop, so search results,
sitemap entries, links found on pages, report PDFs and redirect targets are all covered by one check. The only request
not checked is the first one to an address a person typed (the company website on the New assessment page, or a
network_check URL), which is fetched as a browser would; where it redirects is checked like everything else.

Rules are read once per origin (scheme, host, port) and cached for ROBOTS_CACHE_SECONDS. As RFC 9309 says: no
robots.txt (a 4xx answer) means no restrictions; a server error (5xx) means the site is to be treated as fully
disallowed for now. If robots.txt cannot be fetched because the network failed, that network failure is reported for
the page itself: it is not presented as the site's decision."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

from app.services.security import policy
from app.services.security.policy import SourceRejection

ROBOTS_AGENT = "ESG-Assessment-MVP"  # the product token of safe_fetch.USER_AGENT
ROBOTS_CACHE_SECONDS = 3600.0
FAILURE_CACHE_SECONDS = 60.0  # a network failure fetching robots.txt is re-tried after this


@dataclass
class SiteRules:
    """robots.txt for one origin: what may be fetched, and the sitemaps it names. `disallow_all` after a 5xx."""
    parser: RobotFileParser | None = None
    sitemaps: list[str] = field(default_factory=list)
    disallow_all: bool = False

    def allows(self, url: str) -> bool:
        if self.disallow_all:
            return False
        return self.parser is None or self.parser.can_fetch(ROBOTS_AGENT, url)


def parse_robots(text: str, base_url: str) -> SiteRules:
    parser = RobotFileParser()
    lines = text.splitlines()
    parser.parse(lines)
    sitemaps = []
    for line in lines:
        key, _, value = line.partition(":")
        if key.strip().lower() == "sitemap" and value.strip():
            sitemaps.append(urljoin(base_url, value.strip()))
    return SiteRules(parser, list(dict.fromkeys(sitemaps)))


def origin_of(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme.lower()}://{(parts.netloc or '').lower()}"


class RobotsCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, tuple[float, SiteRules | SourceRejection]] = {}
        self.fetches = 0  # robots.txt requests made (for diagnostics and tests)

    def rules_for(self, url: str, fetch_text: Callable[[str], str]) -> SiteRules:
        """The origin's rules, fetched at most once per cache period. Raises the network failure (SourceRejection)
        when robots.txt could not be fetched for a network reason."""
        origin = origin_of(url)
        with self._lock:
            hit = self._items.get(origin)
            if hit and hit[0] > time.monotonic():
                if isinstance(hit[1], SourceRejection):
                    raise hit[1]
                return hit[1]
        self.fetches += 1
        try:
            rules = parse_robots(fetch_text(origin + "/robots.txt"), origin + "/")
            ttl: float = ROBOTS_CACHE_SECONDS
        except SourceRejection as exc:
            status = _http_status(exc)
            if status is not None and 500 <= status < 600:
                rules, ttl = SiteRules(disallow_all=True), ROBOTS_CACHE_SECONDS / 4
            elif status is not None or exc.code in (policy.INVALID_CONTENT_TYPE, policy.DOWNLOAD_SIZE_LIMIT_EXCEEDED,
                                                    policy.INVALID_PDF_SIGNATURE):
                rules, ttl = SiteRules(), ROBOTS_CACHE_SECONDS  # no usable robots.txt: no restrictions stated
            else:  # the network failed: report that, not a decision of the site
                with self._lock:
                    self._items[origin] = (time.monotonic() + FAILURE_CACHE_SECONDS, exc)
                raise
        with self._lock:
            self._items[origin] = (time.monotonic() + ttl, rules)
        return rules

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
        self.fetches = 0


def _http_status(exc: SourceRejection) -> int | None:
    if exc.code != policy.HTTP_ERROR:
        return None
    try:
        return int(str(exc).split()[1])
    except (IndexError, ValueError):
        return None


cache = RobotsCache()


def check(url: str, fetch_text: Callable[[str], str]) -> None:
    """Raise SourceRejection(ROBOTS_DISALLOWED) when the site's robots.txt asks automated clients not to fetch `url`."""
    rules = cache.rules_for(url, fetch_text)
    if not rules.allows(url):
        detail = "the site is temporarily unavailable to automated clients (robots.txt server error)" \
            if rules.disallow_all else "the site's robots.txt asks automated clients not to fetch this address"
        raise SourceRejection(policy.ROBOTS_DISALLOWED, detail)
