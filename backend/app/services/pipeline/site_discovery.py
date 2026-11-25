"""Finding report and ESG pages on a company's own official website, without a search engine.

Pure helpers (no network, no database): robots.txt rules, sitemap parsing and ranking of candidate addresses. The
discovery stage in discovery.py fetches through safe_fetch and passes every document through the same identity and
evidence gates as any other source. Bounded by design: a few sitemap files, a capped number of addresses parsed and a
small number of pages and PDFs actually read; never a crawl of the whole site.

Sitemaps are read with a plain pattern match for <loc> entries, never with an XML parser, so an external file cannot
trigger entity expansion or external entity loading."""
from __future__ import annotations

import html
import re
from urllib.parse import urlparse

from app.services.extraction.patterns import YEAR_RE
from app.services.security.robots import ROBOTS_AGENT, SiteRules, parse_robots  # noqa: F401 — re-exported
MAX_SITEMAP_LOCS = 50_000  # the sitemap protocol's own limit per file

# Address words that point at the pages and documents an ESG assessment needs, by weight
_STRONG = ("sustainability-report", "sustainability_report", "sustainabilityreport", "annual-report", "annual_report",
           "annualreport", "esg-report", "integrated-report", "annual-and-sustainability", "sustainability-statement",
           "climate-report", "impact-report", "responsibility-report", "csr-report", "gri-index", "esg-data")
_MEDIUM = ("sustainab", "esg", "csr", "responsib", "climate", "environment", "investor", "reports", "report",
           "archive", "publications", "governance", "downloads", "diversity", "energy", "human-rights", "ethics",
           "code-of-conduct", "privacy", "security", "tcfd", "csrd")
# Sections that hardly ever hold an ESG report: they are only taken when nothing better exists
_WEAK = ("career", "jobs", "job-", "vacanc", "press-release", "/news/", "/event", "/webcast", "login", "/cart", "/shop",
         "/product", "/tag/", "/search", "cookie", "/contact", "/media/image", "/blog/", "/stories/", "/article",
         "interim-report", "interim_report", "quarterly", "-q1-", "-q2-", "-q3-", "/q1-", "/q2-", "/q3-")
_SKIP_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".mp4", ".mp3", ".zip", ".gz", ".xlsx", ".xls",
                  ".docx", ".doc", ".pptx", ".ppt", ".css", ".js", ".ico", ".woff", ".woff2", ".json", ".xml", ".txt")


_LOC = re.compile(r"<(?:\w+:)?loc>\s*([^<\s]+)\s*</(?:\w+:)?loc>", re.I)


def parse_sitemap(text: str, limit: int = MAX_SITEMAP_LOCS) -> tuple[bool, list[str]]:
    """(is a sitemap index, the addresses it lists)."""
    is_index = bool(re.search(r"<(?:\w+:)?sitemapindex\b", text[:5000], re.I))
    locs: list[str] = []
    for m in _LOC.finditer(text):
        locs.append(html.unescape(m.group(1)))
        if len(locs) >= limit:
            break
    return is_index, locs


def on_site(url: str, domain: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return bool(domain) and (host == domain or host.endswith("." + domain))


def score_site_url(url: str, year: int | None = None, anchor: str = "") -> int:
    """How likely an address on the company's site leads to report or ESG content; 0 or less means not at all."""
    try:
        path = urlparse(url).path.lower()
    except ValueError:
        return 0
    if path.endswith(_SKIP_SUFFIXES):
        return 0
    hay = f"{url.lower()} {anchor.lower()}"
    score = 4 * sum(term in hay for term in _STRONG) + sum(term in hay for term in _MEDIUM)
    if score == 0:
        return 0
    if path.endswith(".pdf"):
        score += 3
    if any(term in hay for term in _WEAK):
        score -= 4
    if year is not None:
        years = [int(y) for y in YEAR_RE.findall(hay) if 1990 <= int(y) <= year + 1]
        if years:
            newest = max(years)
            score += 3 if newest >= year - 1 else -3 if newest < year - 2 else 0
    return score


def rank_site_urls(candidates: list[tuple[str, str]], domain: str, year: int | None) -> list[tuple[int, str]]:
    """(score, url) best first, for (url, anchor) pairs on the official site; duplicates and non-candidates removed."""
    best: dict[str, int] = {}
    for url, anchor in candidates:
        url = url.split("#", 1)[0]
        if not on_site(url, domain):
            continue
        s = score_site_url(url, year, anchor)
        if s > 0:
            best[url] = max(s, best.get(url, 0))
    return sorted(((s, u) for u, s in best.items()), key=lambda t: (-t[0], len(t[1])))


def rank_sitemaps(urls: list[str]) -> list[str]:
    """Child sitemaps of an index: page and report sitemaps before product, news or image sitemaps."""
    def key(u: str) -> int:
        low = u.lower()
        good = sum(t in low for t in ("page", "report", "investor", "sustainab", "esg", "document", "pdf", "about", "corporate"))
        bad = sum(t in low for t in ("product", "news", "press", "image", "video", "blog", "career", "job", "event", "shop"))
        return good - 2 * bad
    return sorted(dict.fromkeys(urls), key=key, reverse=True)


def official_start_urls(domain: str) -> list[str]:
    """Where an official site is entered when only its domain is known."""
    domain = domain.strip().lower().rstrip(".")
    return [f"https://www.{domain}/", f"https://{domain}/"] if not domain.startswith("www.") else [f"https://{domain}/"]
