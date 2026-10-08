"""Company identification and source classification heuristics."""
from __future__ import annotations

import re
from urllib.parse import urlparse

from app.config.loader import source_priority
from app.services.documents.url_safety import registrable_domain

LEGAL_SUFFIXES = r"\b(ab|ag|as|asa|sa|s\.a\.|spa|s\.p\.a\.|nv|n\.v\.|bv|plc|ltd|limited|inc|inc\.|corp|corporation|co|gmbh|oy|oyj|uab|llc|group|holding|holdings)\b"

REGULATORY_HOST_HINTS = ("europa.eu", "globalreporting.org", "sasb.org", "ifrs.org", "efrag.org", "sciencebasedtargets.org", "cdp.net", ".gov", "sec.gov")
EXTERNAL_CREDIBLE_HINTS = ("reuters.com", "bloomberg.com", "ft.com", "wsj.com", "sustainalytics.com", "msci.com", "ecovadis.com", "wikipedia.org")


def normalize_company_name(name: str) -> str:
    n = name.lower()
    n = re.sub(LEGAL_SUFFIXES, " ", n)
    n = re.sub(r"[^a-z0-9À-ɏ ]+", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def name_tokens(name: str) -> list[str]:
    return [t for t in normalize_company_name(name).split() if len(t) >= 3]


def looks_official(url: str, company_name: str, official_domain: str | None) -> bool:
    host = (urlparse(url).hostname or "").lower()
    if official_domain:
        return host == official_domain or host.endswith("." + official_domain)
    dom = (registrable_domain(host) or "").split(".")[0]
    tokens = name_tokens(company_name)
    if not tokens or not dom:
        return False
    joined = "".join(tokens)
    return dom == joined or dom in joined or any(t == dom or (len(t) >= 4 and t in dom) for t in tokens)


def classify_source(url: str, title: str, snippet: str, company_name: str, official_domain: str | None) -> tuple[str, bool]:
    """Return (source_type, is_official)."""
    host = (urlparse(url).hostname or "").lower()
    path = urlparse(url).path.lower()
    hay = f"{title} {snippet} {path}".lower()
    official = looks_official(url, company_name, official_domain)
    if official:
        is_pdf = path.endswith(".pdf")
        if any(k in hay for k in ("sustainability report", "esg report", "sustainability-report", "responsibility report", "impact report", "csr report")):
            return "sustainability_report", True
        if "annual report" in hay or "annual-report" in hay or "annualreport" in hay:
            return "annual_report", True
        if is_pdf and ("sustainab" in hay or "esg" in hay):
            return "sustainability_report", True
        return "company_website", True
    if any(h in host for h in REGULATORY_HOST_HINTS):
        return "regulatory_source", False
    if any(h in host for h in EXTERNAL_CREDIBLE_HINTS):
        return "external_source", False
    return "other_web", False


def rank_search_result(url: str, title: str, snippet: str, company_name: str, official_domain: str | None, year: int) -> float:
    """Higher is better. Used to choose a controlled number of pages to download."""
    source_type, official = classify_source(url, title, snippet, company_name, official_domain)
    score = 10 - source_priority(source_type)["priority"]
    hay = f"{title} {snippet} {url}".lower()
    if url.lower().split("?")[0].endswith(".pdf"):
        score += 2
    if str(year) in hay or str(year - 1) in hay:
        score += 1.5
    if not official and not any(t in hay for t in name_tokens(company_name)):
        score -= 5  # probably about a different company
    return score


def normalize_org_number(value: str | None) -> str | None:
    """'556012-5790' / '556012 5790' -> '5560125790'. None when nothing usable is left."""
    if not value:
        return None
    n = re.sub(r"[^0-9A-Za-z]", "", value).upper()
    return n or None
