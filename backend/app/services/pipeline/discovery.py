"""Controlled public-web discovery: a few search queries + a few pages from the company site.
Never crawls a whole site."""
from __future__ import annotations

import hashlib
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlparse

from sqlalchemy.orm import Session

from app.config.loader import source_priority
from app.core.config import get_settings
from app.models.entities import Assessment, Source
from app.services.audit import log_event
from app.services.company.identify import classify_source, looks_official, rank_search_result
from app.services.documents.html_extractor import extract_html, relevant_links
from app.services.documents.url_safety import UnsafeURLError, registrable_domain, validate_public_url
from app.services.documents.web_fetcher import FetchError, fetch
from app.services.pipeline.ingest import ingest_html, ingest_pdf
from app.services.search.base import SearchProvider, build_queries

log = logging.getLogger(__name__)


@dataclass
class DiscoveryReport:
    queries: list[str] = field(default_factory=list)
    results_seen: int = 0
    official_domain: str | None = None
    warnings: list[str] = field(default_factory=list)


def _save_pdf(content: bytes, url: str) -> str:
    name = hashlib.sha256(url.encode()).hexdigest()[:24] + ".pdf"
    path = get_settings().download_dir / name
    path.write_bytes(content)
    return str(path)


def _new_source(db: Session, a: Assessment, url: str, title: str, source_type: str, official: bool, via: str) -> Source:
    pr = source_priority(source_type)
    s = Source(assessment_id=a.id, source_type=source_type, priority=pr["priority"], title=title[:500] if title else None,
               url=url, is_public=True, is_official=official, discovered_via=via[:255], status="DISCOVERED")
    db.add(s)
    db.flush()
    log_event(db, "source_discovered", assessment_id=a.id, entity_type="source", entity_id=s.id,
              details={"url": url, "source_type": source_type, "via": via})
    return s


def _download_into(db: Session, s: Source, expect: str = "any") -> None:
    db.commit()  # never hold the SQLite write lock during network I/O
    try:
        res = fetch(s.url, expect=expect)
    except FetchError as exc:
        s.status, s.status_detail = "FAILED", str(exc)[:500]
        return
    s.content_type = res.content_type
    if res.kind == "pdf":
        s.file_path = _save_pdf(res.content, res.final_url)
        try:
            ingest_pdf(db, s, s.file_path)
        except ValueError as exc:
            s.status, s.status_detail = "FAILED", str(exc)
    else:
        ingest_html(db, s, res.content, res.final_url)
    db.commit()


def discover(db: Session, a: Assessment, provider: SearchProvider, do_search: bool,
             on_progress: Callable[[str], None] | None = None) -> DiscoveryReport:
    settings = get_settings()
    company = a.company
    report = DiscoveryReport(official_domain=company.domain)
    seen_urls: set[str] = {s.url for s in a.sources if s.url}
    html_budget = settings.max_pages_per_site
    progress = on_progress or (lambda _msg: None)
    pdf_budget = settings.max_downloaded_pdfs

    # 1) official website (if provided): homepage + a few ESG-relevant links
    if company.website:
        try:
            progress(f"Reading company website {company.website}")
            validate_public_url(company.website)
            db.commit()
            home = fetch(company.website, expect="html")
            content = extract_html(home.content, home.final_url)
            report.official_domain = company.domain = registrable_domain(home.final_url) or company.domain
            s = _new_source(db, a, home.final_url, content.title or company.name, "company_website", True, "company website")
            seen_urls.add(home.final_url)
            ingest_html(db, s, home.content, home.final_url)
            html_budget -= 1
            for link in relevant_links(content, report.official_domain, limit=settings.max_pages_per_site):
                if link in seen_urls or html_budget <= 0:
                    continue
                is_pdf = link.lower().split("?")[0].endswith(".pdf")
                if is_pdf and pdf_budget <= 0:
                    continue
                stype, official = classify_source(link, "", "", company.name, report.official_domain)
                s = _new_source(db, a, link, link.rsplit("/", 1)[-1] or link, stype, official, "link on company website")
                progress(f"Downloading {link}")
                seen_urls.add(link)
                _download_into(db, s)
                if is_pdf:
                    pdf_budget -= 1
                else:
                    html_budget -= 1
        except (UnsafeURLError, FetchError) as exc:
            report.warnings.append(f"Company website could not be fetched: {exc}")

    if not do_search:
        return report

    # 2) search engine
    results = []
    queries = build_queries(company.name, a.year)
    for n, q in enumerate(queries, 1):
        progress(f"Searching the web ({n}/{len(queries)}): {q}")
        report.queries.append(q)
        results.extend(provider.search(q, max_results=settings.max_search_results))
    report.results_seen = len(results)
    if not results:
        report.warnings.append(f"Web search ({provider.name}) returned no results.")
        return report

    if not report.official_domain:
        doms = Counter(registrable_domain(r.url) for r in results if looks_official(r.url, company.name, None))
        if doms:
            report.official_domain = company.domain = doms.most_common(1)[0][0]

    unique = {}
    for r in results:
        if r.url and r.url not in unique and r.url not in seen_urls:
            unique[r.url] = r
    ranked = sorted(unique.values(), key=lambda r: rank_search_result(r.url, r.title, r.snippet, company.name, report.official_domain, a.year), reverse=True)

    taken = 0
    for r in ranked:
        if taken >= settings.max_search_results:
            break
        is_pdf = r.url.lower().split("?")[0].endswith(".pdf")
        if (is_pdf and pdf_budget <= 0) or (not is_pdf and html_budget <= 0):
            continue
        if rank_search_result(r.url, r.title, r.snippet, company.name, report.official_domain, a.year) < 0:
            continue
        stype, official = classify_source(r.url, r.title, r.snippet, company.name, report.official_domain)
        host = urlparse(r.url).hostname or ""
        s = _new_source(db, a, r.url, r.title or host, stype, official, f"search: {r.query}")
        progress(f"Downloading {r.title or r.url}")
        _download_into(db, s)
        taken += 1
        if is_pdf:
            pdf_budget -= 1
        else:
            html_budget -= 1
    return report
