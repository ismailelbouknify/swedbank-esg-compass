"""Controlled public-web discovery: a few search queries + a few pages from the company site.
Never crawls a whole site."""
from __future__ import annotations

import hashlib
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlparse

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, object_session

from app.config.loader import OFFICIAL_SOURCE_TYPES, source_priority
from app.core.config import get_settings
from app.models.entities import Assessment, Source
from app.services.audit import log_event, log_source_rejection
from app.services.company.identify import classify_source, looks_official, rank_search_result
from app.services.company.resolution import AMBIGUOUS, UNVERIFIED, resolve_company
from app.services.company.source_identity import _domain_labels
from app.services.documents.html_extractor import extract_html, relevant_links
from app.services.documents.url_safety import registrable_domain
from app.services.pipeline.ingest import ingest_html, ingest_pdf
from app.services.documents.pdf_extractor import peek_pdf
from app.services.pipeline.source_gate import apply_source_identity, identity_for, record_blocked_pdf_identity
from app.services.extraction.patterns import YEAR_RE
from app.services.search.base import SearchProvider, SearchResult, build_queries, build_recovery_queries
from app.services.security import policy
from app.services.security.policy import SourceRejection
from app.services.security.safe_fetch import FetchResult, fetch, source_deadline
from app.services.security.url_validator import UnsafeURLError, validate_external_url

log = logging.getLogger(__name__)

# Keywords in anchor text / URL that suggest a link leads to a primary report.
_REPORT_LINK_KEYWORDS = (
    "annual report", "sustainability report", "esg report", "integrated report",
    "impact report", "full report", "download report", "climate report",
    "responsibility report", "csr report", "environmental report", "sustainability statement",
    "annual and sustainability", "annual review", "annual-report", "sustainability-report",
    "esg-report", "integrated-report", "environmental-report",
)
MAX_REPORT_LINKS_PER_PAGE = 3
PRIMARY_DOCUMENT_HOPS = 2  # landing page -> reports page -> report


def _find_report_links(content, official_domain: str | None, year: int | None = None) -> list[str]:
    """Up to MAX_REPORT_LINKS_PER_PAGE links that look like primary report documents, best first.

    Report PDFs often sit on another host (a CDN or filing service), so a PDF may be off-domain: the identity
    gate judges the document itself once it is read. Other pages must be on the official domain. The latest
    report wins over an older one."""
    from app.services.documents.html_extractor import HtmlContent

    if not isinstance(content, HtmlContent) or not content.links:
        return []
    candidates: list[tuple[int, str]] = []
    for url, anchor in content.links:
        host = (urlparse(url).hostname or "").lower()
        is_pdf = url.lower().split("?")[0].endswith(".pdf")
        if official_domain and not is_pdf and not (host == official_domain or host.endswith("." + official_domain)):
            continue
        hay = (url + " " + anchor).lower()
        score = sum(1 for kw in _REPORT_LINK_KEYWORDS if kw in hay)
        if score == 0:
            continue
        score += 3 if is_pdf else 0
        years = [int(y) for y in YEAR_RE.findall(hay) if year is None or int(y) <= year + 1]
        if years and year is not None:
            score += 3 if max(years) >= year - 1 else -2
        candidates.append((score, url))
    seen: set[str] = set()
    out: list[str] = []
    for _, url in sorted(candidates, key=lambda t: -t[0]):
        if url not in seen:
            seen.add(url)
            out.append(url)
        if len(out) >= MAX_REPORT_LINKS_PER_PAGE:
            break
    return out


@dataclass
class DiscoveryReport:
    queries: list[str] = field(default_factory=list)
    results_seen: int = 0
    official_domain: str | None = None
    warnings: list[str] = field(default_factory=list)
    sources_processed: int = 0
    sources_rejected: int = 0  # refused by security policy
    sources_unavailable: int = 0  # could not be reached or read
    candidates_rejected: int = 0  # URLs refused before any request; no source was created for them
    candidates_other_entity: int = 0  # on the web address of another company with a similar name
    searches_failed: int = 0

    @property
    def coverage_limited(self) -> bool:
        """Missing evidence may reflect unreadable sources rather than missing company activity."""
        return bool(self.sources_rejected or self.sources_unavailable
                    or (self.candidates_rejected and not self.sources_processed))


def _well_formed(url: str) -> bool:
    """Search results are untrusted: a malformed URL would otherwise break ranking further on."""
    try:
        return bool(urlparse(url).hostname)
    except ValueError:
        return False


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


def _admit(db: Session, a: Assessment, url: str, report: DiscoveryReport) -> bool:
    """Security validation of a candidate URL, before it can become a source or use up a budget."""
    db.commit()  # never hold the SQLite write lock during network I/O (DNS)
    try:
        validate_external_url(url)
        return True
    except UnsafeURLError as exc:
        report.candidates_rejected += 1
        log_source_rejection(db, exc, assessment_id=a.id, url=url, stage="candidate")
        return False


def _mark_unprocessed(db: Session, s: Source, exc: SourceRejection, report: DiscoveryReport) -> None:
    """A rejected or unreadable source is recorded and the assessment continues with the others."""
    if policy.is_security_rejection(exc.code):
        s.status = "REJECTED"
        report.sources_rejected += 1
    else:
        s.status = "FAILED"
        report.sources_unavailable += 1
    s.status_detail = str(exc)[:500]
    s.fetch_outcome = exc.code
    log_source_rejection(db, exc, assessment_id=s.assessment_id, url=s.url, source_id=s.id)


def _process_pdf(db: Session, s: Source, res: FetchResult, report: DiscoveryReport, deadline: float) -> None:
    """Text extraction from the in-memory PDF; afterwards the original bytes are released."""
    started = time.monotonic()
    stored = get_settings().store_external_files
    try:
        if stored:
            s.file_path = _save_pdf(res.content, res.final_url)
        document = ingest_pdf(db, s, res.content, deadline=deadline)
        log_event(db, "temporary_pdf_processed", assessment_id=s.assessment_id, entity_type="source", entity_id=s.id,
                  details={"bytes": res.size, "pages": document.page_count, "sha256": res.sha256, "scan": res.scan_status,
                           "seconds": round(time.monotonic() - started, 2), "stored": stored})
    except SourceRejection as exc:
        _mark_unprocessed(db, s, exc, report)
        if exc.code == policy.PDF_PAGE_LIMIT_EXCEEDED:
            record_blocked_pdf_identity(db, s.assessment, s, *peek_pdf(res.content))
    except SQLAlchemyError:
        raise
    except Exception as exc:  # noqa: BLE001 — one unreadable document must not stop the assessment
        _mark_unprocessed(db, s, SourceRejection(policy.PDF_PARSE_FAILED, f"PDF could not be read: {type(exc).__name__}"), report)
    finally:
        res.content = b""
        if not stored:
            log_event(db, "temporary_pdf_deleted", assessment_id=s.assessment_id, entity_type="source", entity_id=s.id,
                      details={"storage": "memory"})


def _download_into(db: Session, s: Source, report: DiscoveryReport, expect: str = "any") -> FetchResult | None:
    """Fetch and ingest one source. Returns None when it was rejected or unavailable."""
    db.commit()  # never hold the SQLite write lock during network I/O
    deadline = source_deadline()
    try:
        res = fetch(s.url, expect=expect, deadline=deadline)
    except SourceRejection as exc:
        _mark_unprocessed(db, s, exc, report)
        db.commit()
        return None
    s.content_type = res.content_type
    s.content_sha256 = res.sha256
    s.bytes_downloaded = res.size
    s.fetch_outcome = "ok"
    if res.kind == "pdf":
        _process_pdf(db, s, res, report, deadline)
    else:
        try:
            ingest_html(db, s, res.content, res.final_url)
        except SQLAlchemyError:
            raise
        except Exception as exc:  # noqa: BLE001
            _mark_unprocessed(db, s, SourceRejection(policy.PDF_PARSE_FAILED, f"Page could not be read: {type(exc).__name__}"), report)
    # wrong company: stop here, before retrieval and extraction, and follow nothing from this page
    if s.status == "PROCESSED" and apply_source_identity(db, s.assessment, s):
        report.sources_processed += 1
    db.commit()
    return res if s.status != "REJECTED" else None


@dataclass
class _Pass:
    """One bounded pass over sources: what has been seen and how much may still be fetched."""
    db: Session
    a: Assessment
    report: DiscoveryReport
    progress: Callable[[str], None]
    seen_urls: set[str]
    pdf_budget: int
    html_budget: int
    max_taken: int
    deadline: float | None = None

    def out_of_time(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    def follow_report_links(self, content, page_url: str, via: str, hops: int = PRIMARY_DOCUMENT_HOPS) -> None:
        """From a landing page to the report itself: the page listing reports is not the report."""
        db, a = self.db, self.a
        for link in _find_report_links(content, self.report.official_domain, a.year):
            is_pdf = link.lower().split("?")[0].endswith(".pdf")
            if link in self.seen_urls or self.out_of_time() or (self.pdf_budget if is_pdf else self.html_budget) <= 0:
                continue
            self.seen_urls.add(link)
            if not _admit(db, a, link, self.report):
                continue
            stype, official = classify_source(link, "", "", a.company.name, self.report.official_domain)
            source = _new_source(db, a, link, link.rsplit("/", 1)[-1] or link, stype, official, f"report link on {via}"[:255])
            self.progress(f"Downloading primary report: {link}")
            res = _download_into(db, source, self.report)
            if source.rejection_reason:
                continue  # a wrong-company document does not use up the budget
            if is_pdf:
                self.pdf_budget -= 1
            else:
                self.html_budget -= 1
                if res is not None and res.kind == "html" and hops > 1:
                    self.follow_report_links(extract_html(res.content, res.final_url), res.final_url, via, hops - 1)
            log_event(db, "primary_document_discovered", assessment_id=a.id, entity_type="source", entity_id=source.id,
                      details={"url": link, "discovered_from": page_url, "via": via})

    def take_results(self, results: list[SearchResult]) -> None:
        db, a = self.db, self.a
        company, report = a.company, self.report
        identity = identity_for(a)
        other_domains = {e.replace(" ", "") for e in identity.excluded}
        unique: dict[str, SearchResult] = {}
        for r in results:
            if r.url and r.url not in unique and r.url not in self.seen_urls:
                unique[r.url] = r
        rank = lambda r: rank_search_result(r.url, r.title, r.snippet, company.name, report.official_domain, a.year)  # noqa: E731
        taken = 0
        for r in sorted(unique.values(), key=rank, reverse=True):
            if taken >= self.max_taken or self.out_of_time():
                break
            is_pdf = r.url.lower().split("?")[0].endswith(".pdf")
            if (is_pdf and self.pdf_budget <= 0) or (not is_pdf and self.html_budget <= 0) or rank(r) < 0:
                continue
            if _domain_labels(r.url) & other_domains:  # no need to fetch what the address already rules out
                report.candidates_other_entity += 1
                continue
            if not _admit(db, a, r.url, report):
                continue
            self.seen_urls.add(r.url)
            stype, official = classify_source(r.url, r.title, r.snippet, company.name, report.official_domain)
            host = urlparse(r.url).hostname or ""
            s = _new_source(db, a, r.url, r.title or host, stype, official, f"search: {r.query}")
            self.progress(f"Downloading {r.title or r.url}")
            res = _download_into(db, s, report)
            if s.rejection_reason:
                continue  # a wrong-company document does not use up the budget
            taken += 1
            if is_pdf:
                self.pdf_budget -= 1
            else:
                self.html_budget -= 1
                # A page found by search may be a landing page: look for the report behind it.
                if res is not None and res.kind == "html" and (s.is_official or s.source_type in OFFICIAL_SOURCE_TYPES):
                    self.follow_report_links(extract_html(res.content, res.final_url), res.final_url, f"search result: {r.title or r.url}")


def _resolve_identity(a: Assessment, report: DiscoveryReport, results: list[SearchResult], may_choose_entity: bool = True) -> None:
    """Once settled, the identity stays: later searches cannot change which company is being assessed."""
    company = a.company
    current = a.company_identity or {}
    if current.get("status") not in (None, UNVERIFIED):
        return
    website_domain = company.domain if company.website else None
    profile = resolve_company(company.name, company.organisation_number, website_domain, results)
    if not may_choose_entity and profile.get("aliases"):
        # Recovery queries are broad: results that merely share the name's word must not decide which company
        # this is, or the identity gate would then approve that other company's own documents.
        profile = {**profile, "basis": "name_only", "official_domain": None, "aliases": [],
                   "excluded": sorted({*profile.get("excluded", []), *profile["aliases"]})}
    a.company_identity = profile
    if profile.get("official_domain"):
        report.official_domain = company.domain = profile["official_domain"]
    elif profile["status"] != AMBIGUOUS and not report.official_domain:
        doms = Counter(registrable_domain(r.url) for r in results if looks_official(r.url, company.name, None))
        if doms:
            report.official_domain = company.domain = doms.most_common(1)[0][0]
    if profile["status"] == AMBIGUOUS:
        report.official_domain = company.domain = None
        log_event(object_session(a), AMBIGUOUS, assessment_id=a.id, details={"candidates": profile.get("candidates")})


def _run_searches(provider: SearchProvider, queries: list[str], report: DiscoveryReport, progress: Callable[[str], None], *,
                  pause: float = 0.0, deadline: float | None = None) -> list[SearchResult]:
    settings = get_settings()
    results: list[SearchResult] = []
    failed_before = provider.failed_searches
    for n, q in enumerate(queries, 1):
        if deadline is not None and time.monotonic() >= deadline:
            break
        if n > 1 and pause:
            time.sleep(pause)
        progress(f"Searching the web ({n}/{len(queries)}): {q}")
        report.queries.append(q)
        results.extend(r for r in provider.search(q, max_results=settings.max_search_results) if _well_formed(r.url))
    report.searches_failed += provider.failed_searches - failed_before
    report.results_seen += len(results)
    return results


def discover(db: Session, a: Assessment, provider: SearchProvider, do_search: bool,
             on_progress: Callable[[str], None] | None = None) -> DiscoveryReport:
    settings = get_settings()
    company = a.company
    report = DiscoveryReport(official_domain=company.domain)
    progress = on_progress or (lambda _msg: None)
    run = _Pass(db, a, report, progress, {s.url for s in a.sources if s.url}, settings.max_downloaded_pdfs, settings.max_pages_per_site,
                settings.max_search_results)

    # 1) official website (if provided): homepage + a few ESG-relevant links
    if company.website and settings.enable_external_fetch:
        try:
            progress(f"Reading company website {company.website}")
            db.commit()
            home = fetch(company.website, expect="html")
            content = extract_html(home.content, home.final_url)
            report.official_domain = company.domain = registrable_domain(home.final_url) or company.domain
            s = _new_source(db, a, home.final_url, content.title or company.name, "company_website", True, "company website")
            run.seen_urls.add(home.final_url)
            s.content_type, s.content_sha256, s.bytes_downloaded, s.fetch_outcome = home.content_type, home.sha256, home.size, "ok"
            ingest_html(db, s, home.content, home.final_url)
            report.sources_processed += s.status == "PROCESSED" and apply_source_identity(db, a, s)
            run.html_budget -= 1
            # Follow links to primary report PDFs found on the homepage.
            run.follow_report_links(content, home.final_url, "company website homepage")
            for link in relevant_links(content, report.official_domain, limit=settings.max_pages_per_site):
                if link in run.seen_urls or run.html_budget <= 0:
                    continue
                is_pdf = link.lower().split("?")[0].endswith(".pdf")
                if is_pdf and run.pdf_budget <= 0:
                    continue
                run.seen_urls.add(link)
                if not _admit(db, a, link, report):
                    continue
                stype, official = classify_source(link, "", "", company.name, report.official_domain)
                s = _new_source(db, a, link, link.rsplit("/", 1)[-1] or link, stype, official, "link on company website")
                progress(f"Downloading {link}")
                res = _download_into(db, s, report)
                if s.rejection_reason:
                    continue  # a wrong-company document does not use up the budget
                if is_pdf:
                    run.pdf_budget -= 1
                else:
                    run.html_budget -= 1
                    # After downloading a sub-page, also check it for report PDF links.
                    if res is not None and res.kind == "html":
                        run.follow_report_links(extract_html(res.content, res.final_url), res.final_url, "company website sub-page")
        except SourceRejection as exc:
            report.warnings.append(f"Company website could not be fetched: {exc.code}")
            report.sources_rejected += policy.is_security_rejection(exc.code)
            report.sources_unavailable += not policy.is_security_rejection(exc.code)
            log_source_rejection(db, exc, assessment_id=a.id, url=company.website, stage="company_website")
        except SQLAlchemyError:
            raise
        except Exception as exc:  # noqa: BLE001 — malformed untrusted content must not stop the assessment
            report.warnings.append(f"Company website could not be fetched: {type(exc).__name__}")
            report.sources_unavailable += 1

    if not do_search or not settings.enable_web_search or not settings.enable_external_fetch:
        _resolve_identity(a, report, [])
        return report

    # 2) search engine
    # providers receive these queries (company name and search terms), never document content
    results = _run_searches(provider, build_queries(company.name, a.year), report, progress)
    _resolve_identity(a, report, results)
    if not results:
        report.warnings.append(f"Web search ({provider.name}) returned no results.")
        return report
    run.take_results(results)
    return report


def recover_sources(db: Session, a: Assessment, provider: SearchProvider, earlier_queries: list[str],
                    on_progress: Callable[[str], None] | None = None) -> DiscoveryReport:
    """The single bounded extra search when coverage is too thin: broader queries, the same hard gates."""
    settings = get_settings()
    company = a.company
    report = DiscoveryReport(official_domain=company.domain)
    progress = on_progress or (lambda _msg: None)
    deadline = time.monotonic() + settings.recovery_time_budget_seconds
    profile = a.company_identity or {}
    queries = build_recovery_queries(company.name, list(profile.get("aliases") or []), company.domain, a.year, earlier_queries, settings.recovery_max_queries)
    results = _run_searches(provider, queries, report, progress, pause=settings.recovery_search_pause_seconds, deadline=deadline)
    _resolve_identity(a, report, results, may_choose_entity=False)
    if results:
        _Pass(db, a, report, progress, {s.url for s in a.sources if s.url}, settings.recovery_max_pdfs, settings.recovery_max_pages,
              settings.recovery_max_pages + settings.recovery_max_pdfs, deadline).take_results(results)
    db.commit()
    return report
