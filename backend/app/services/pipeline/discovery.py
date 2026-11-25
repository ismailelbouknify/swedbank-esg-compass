"""Controlled public-web discovery: a few search queries + a few pages from the company's official site.
Never crawls a whole site.

Search and website retrieval are independent: a search provider that declines automated searches stops searching
only, and the official website (given by the analyst, or the domain the identity check settled) is still read
through its navigation and sitemaps within fixed limits."""
from __future__ import annotations

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
from app.services.company.resolution import AMBIGUOUS, RESOLVED, UNVERIFIED, resolve_company
from app.services.company.source_identity import _domain_labels
from app.services.documents.html_extractor import HtmlContent, extract_html, relevant_links
from app.services.documents.url_safety import registrable_domain
from app.services.pipeline import site_discovery
from app.services.pipeline.ingest import ingest_html, ingest_pdf
from app.services.documents.pdf_extractor import peek_pdf
from app.services.documents.report_kind import primary_report_kind
from app.services.pipeline.source_gate import (
    REPORT_LINK_VIA, apply_source_identity, identity_for, precheck_pdf_identity, record_blocked_pdf_identity,
)
from app.services.documents.text_utils import YEAR_RE
from app.services.search.base import SearchProvider, SearchResult, build_factor_recovery_queries, build_queries, build_recovery_queries
from app.services.security import network, policy, robots
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
        # the label includes the text around a generic "View report" / "Download" link (see extract_html)
        hay = (url + " " + anchor).lower()
        score = sum(1 for kw in _REPORT_LINK_KEYWORDS if kw in hay)
        if score == 0:
            continue
        score += 3 if is_pdf else 0
        # the company-wide report first; a product report or presentation only after it
        score += 4 if primary_report_kind(anchor, "", url) else -4
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
class WebAccessState:
    """How the network has treated this assessment's requests, across discovery and recovery.

    When nothing has ever got through and `limit` requests in a row have failed at network level (proxy refusal,
    certificate failure, no connection), more attempts would only add waiting time: automatic access stops and the
    assessment continues with what it has. Fetches and searches are judged separately: a search service can be
    blocked while company websites are reachable. A web filter blocking one site says nothing about the next.

    A search provider that declines or rate-limits automated searches stops searching at once (also after earlier
    successes): sending more queries would only disregard the provider's limit. Website retrieval is unaffected."""
    limit: int = 0
    fetch_failures: int = 0
    fetch_successes: int = 0
    search_failures: int = 0
    search_successes: int = 0
    issues: Counter = field(default_factory=Counter)  # policy code -> count, fetches and candidate checks
    search_issues: Counter = field(default_factory=Counter)
    skipped_searches: int = 0
    search_declined: bool = False  # the search provider limited or declined automated searches
    sites_explored: set = field(default_factory=set)  # official domains whose navigation and sitemaps were read
    site_pages_read: int = 0  # sources read in the official-website stage
    robots_skipped: int = 0  # addresses not read because the site's robots.txt asks automated clients not to

    def record_fetch(self, code: str | None) -> None:
        """One fetch outcome: None when the site answered with usable content."""
        if code is None:
            self.fetch_successes += 1
            self.fetch_failures = 0
            return
        if code not in (policy.DOWNLOAD_SIZE_LIMIT_EXCEEDED, policy.PDF_PAGE_LIMIT_EXCEEDED):
            self.issues[code] += 1  # limits are counted where the source is marked, whatever stage refused it
        if code in policy.CONNECTIVITY_FAILURE_CODES:
            self.fetch_failures += 1
        elif code != policy.BLOCKED_BY_WEB_FILTER:  # an answer from the site itself: the network works
            self.fetch_successes += 1
            self.fetch_failures = 0

    def record_search(self, code: str | None) -> None:
        if code is None:
            self.search_successes += 1
            self.search_failures = 0
        else:
            self.search_issues[code] += 1
            self.search_failures += 1
            if code in policy.SEARCH_PROVIDER_RESTRICTION_CODES:
                self.search_declined = True

    @property
    def fetch_stopped(self) -> bool:
        return self.limit > 0 and not self.fetch_successes and self.fetch_failures >= self.limit

    @property
    def search_stopped(self) -> bool:
        return self.search_declined or (self.limit > 0 and not self.search_successes and self.search_failures >= self.limit)

    @property
    def search_unavailable(self) -> bool:
        """Search gave nothing usable: stopped, declined, or every attempt failed."""
        return self.search_stopped or (bool(self.search_issues) and not self.search_successes)


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
    fetch_seconds: float = 0.0  # HTTP time, all sources
    pdf_parse_seconds: float = 0.0  # PDF text extraction time
    duplicate_documents: int = 0  # byte-identical to a document already read in this assessment: not parsed again
    web: WebAccessState = field(default_factory=WebAccessState)

    @property
    def coverage_limited(self) -> bool:
        """Missing evidence may reflect unreadable sources rather than missing company activity."""
        return bool(self.sources_rejected or self.sources_unavailable
                    or (self.candidates_rejected and not self.sources_processed))


def url_key(url: str) -> str:
    """One key for the same page reached as http/https, with or without www, a trailing slash or a fragment."""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return url
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path.rstrip("/") or "/"
    return f"{host}{path}" + (f"?{parsed.query}" if parsed.query else "")


class _SeenUrls(set):
    """A set of URLs compared by url_key: an http and an https copy of one page are fetched once."""

    def __init__(self, urls=()):
        super().__init__(url_key(u) for u in urls)

    def add(self, url: str) -> None:  # type: ignore[override]
        super().add(url_key(url))

    def __contains__(self, url: object) -> bool:
        return isinstance(url, str) and super().__contains__(url_key(url))


def _well_formed(url: str) -> bool:
    """Search results are untrusted: a malformed URL would otherwise break ranking further on."""
    try:
        return bool(urlparse(url).hostname)
    except ValueError:
        return False


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
        validate_external_url(url, allow_unresolved=network.proxy_for(url) is not None)
        return True
    except UnsafeURLError as exc:
        report.candidates_rejected += 1
        if exc.code == policy.DNS_RESOLUTION_FAILED:
            report.web.issues[exc.code] += 1
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
    if exc.code in (policy.DOWNLOAD_SIZE_LIMIT_EXCEEDED, policy.PDF_PAGE_LIMIT_EXCEEDED):
        report.web.issues[exc.code] += 1
    log_source_rejection(db, exc, assessment_id=s.assessment_id, url=s.url, source_id=s.id)


def _process_pdf(db: Session, s: Source, res: FetchResult, report: DiscoveryReport, deadline: float) -> None:
    """Text extraction straight from the in-memory PDF bytes. An automatically retrieved PDF is never
    written to disk or to the database; only its URL, metadata and extracted text are kept, and the bytes
    are released as soon as extraction is over, whatever its outcome."""
    started = time.monotonic()
    try:
        # whose document is it? Its first pages are enough to refuse another company's report without parsing all of it
        if not precheck_pdf_identity(db, s.assessment, s, *peek_pdf(res.content)):
            return
        parse_started = time.monotonic()
        try:
            document = ingest_pdf(db, s, res.content, deadline=deadline)
        finally:
            report.pdf_parse_seconds += time.monotonic() - parse_started
        log_event(db, "temporary_pdf_processed", assessment_id=s.assessment_id, entity_type="source", entity_id=s.id,
                  details={"bytes": res.size, "pages": document.page_count, "sha256": res.sha256, "scan": res.scan_status,
                           "seconds": round(time.monotonic() - started, 2), "stored": False})
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
        log_event(db, "temporary_pdf_deleted", assessment_id=s.assessment_id, entity_type="source", entity_id=s.id,
                  details={"storage": "memory"})


def _download_into(db: Session, s: Source, report: DiscoveryReport, expect: str = "any") -> FetchResult | None:
    """Fetch and ingest one source. Returns None when it was rejected or unavailable."""
    db.commit()  # never hold the SQLite write lock during network I/O
    deadline = source_deadline()
    fetch_started = time.monotonic()
    try:
        res = fetch(s.url, expect=expect, deadline=deadline)
    except SourceRejection as exc:
        report.web.record_fetch(exc.code)
        _mark_unprocessed(db, s, exc, report)
        db.commit()
        return None
    finally:
        report.fetch_seconds += time.monotonic() - fetch_started
    report.web.record_fetch(None)
    twin = _identical_document(db, s, res.sha256) if res.kind == "pdf" else None
    if twin is not None:
        # the same PDF under another address (a mirror, a CDN copy, a tracking link): parsed once. Web pages are
        # cheap to read; their repeated passages are removed before extraction instead.
        s.status, s.status_detail, s.fetch_outcome = "SKIPPED", f"Same document as source {twin.id}", "duplicate_content"
        s.content_sha256, s.bytes_downloaded, s.content_type = res.sha256, res.size, res.content_type
        report.duplicate_documents += 1
        res.content = b""
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


def _not_counted(s: Source) -> bool:
    """Sources that leave the budget for the next candidate: another company's document, or an address the site's
    robots.txt excludes (recorded as REJECTED with that reason, never read)."""
    return bool(s.rejection_reason) or s.fetch_outcome == policy.ROBOTS_DISALLOWED


def _identical_document(db: Session, s: Source, sha256: str) -> Source | None:
    """A source of this assessment already read whose content is byte-identical."""
    from sqlalchemy import select

    return db.scalar(select(Source).where(Source.assessment_id == s.assessment_id, Source.id != s.id,
                                          Source.content_sha256 == sha256, Source.status == "PROCESSED").limit(1))


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
        """Also true once the network has refused every request: no further source is attempted."""
        if self.report.web.fetch_stopped:
            return True
        return self.deadline is not None and time.monotonic() >= self.deadline

    def follow_report_links(self, content, page_url: str, via: str, hops: int = PRIMARY_DOCUMENT_HOPS) -> None:
        """From a landing page to the report itself: the page listing reports is not the report."""
        db, a = self.db, self.a
        labels = dict(getattr(content, "links", None) or [])
        for link in _find_report_links(content, self.report.official_domain, a.year):
            is_pdf = link.lower().split("?")[0].endswith(".pdf")
            if link in self.seen_urls or self.out_of_time() or (self.pdf_budget if is_pdf else self.html_budget) <= 0:
                continue
            self.seen_urls.add(link)
            if not _admit(db, a, link, self.report):
                continue
            label = labels.get(link) or ""
            stype, official = classify_source(link, label, "", a.company.name, self.report.official_domain)
            source = _new_source(db, a, link, label[:300] or link.rsplit("/", 1)[-1] or link, stype, official, f"{REPORT_LINK_VIA} {via}"[:255])
            self.progress(f"Downloading primary report: {link}")
            res = _download_into(db, source, self.report)
            if _not_counted(source):
                continue  # a wrong-company or robots-excluded document does not use up the budget
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
            key = url_key(r.url) if r.url else ""
            if r.url and key not in unique and r.url not in self.seen_urls:
                unique[key] = r
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
            if _not_counted(s):
                continue  # a wrong-company or robots-excluded document does not use up the budget
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
        if report.web.search_stopped:
            report.web.skipped_searches += len(queries) - n + 1
            break
        if n > 1 and pause:
            time.sleep(pause)
        progress(f"Searching the web ({n}/{len(queries)}): {q}")
        report.queries.append(q)
        failed = provider.failed_searches
        results.extend(r for r in provider.search(q, max_results=settings.max_search_results) if _well_formed(r.url))
        report.web.record_search((provider.last_error or policy.NETWORK_ERROR) if provider.failed_searches > failed else None)
    report.searches_failed += provider.failed_searches - failed_before
    report.results_seen += len(results)
    return results


def discover(db: Session, a: Assessment, provider: SearchProvider, do_search: bool,
             on_progress: Callable[[str], None] | None = None, web: WebAccessState | None = None) -> DiscoveryReport:
    settings = get_settings()
    company = a.company
    report = DiscoveryReport(official_domain=company.domain, web=web or WebAccessState(settings.web_access_failure_limit))
    progress = on_progress or (lambda _msg: None)
    run = _Pass(db, a, report, progress, _SeenUrls(s.url for s in a.sources if s.url), settings.max_downloaded_pdfs,
                settings.max_pages_per_site, settings.max_search_results)

    # 1) official website (if provided): homepage + a few ESG-relevant links, then its navigation and sitemaps
    if company.website and settings.enable_external_fetch:
        home = _read_company_website(run, company.website, "company website", typed=True)
        if home is not None:
            _explore_official_site(run, *home)
        else:  # the analyst's address was tried: its domain is not tried again under other addresses
            report.web.sites_explored.add((report.official_domain or registrable_domain(company.website) or "").lower())

    if not do_search or not settings.enable_web_search or not settings.enable_external_fetch:
        _resolve_identity(a, report, [])
        return report

    # 2) search engine
    # providers receive these queries (company name and search terms), never document content
    results = _run_searches(provider, build_queries(company.name, a.year), report, progress)
    _resolve_identity(a, report, results)
    if results:
        run.take_results(results)
    elif not report.web.search_unavailable:
        report.warnings.append(f"Web search ({provider.name}) returned no results.")
    # 3) search unavailable or thin: the official site itself, when the identity check has settled which one it is
    if report.web.search_unavailable or report.sources_processed < settings.coverage_min_valid_sources:
        explore_validated_official_site(run)
    return report


def validated_official_domain(a: Assessment, report: DiscoveryReport) -> str | None:
    """The company's own domain, only when it is known for certain: given by the analyst, or settled by the identity
    check (resolved, not ambiguous). A guess from search results is never enough to read a site as official."""
    if a.company.website:
        return report.official_domain or registrable_domain(a.company.website)
    profile = a.company_identity or {}
    if profile.get("status") == RESOLVED and profile.get("official_domain"):
        return str(profile["official_domain"]).lower()
    return None


def explore_validated_official_site(run: "_Pass") -> None:
    """Read the official site without search: homepage (if not read yet), then navigation and sitemaps."""
    settings = get_settings()
    report = run.report
    domain = validated_official_domain(run.a, report)
    if (not domain or not settings.website_discovery_enabled or not settings.enable_external_fetch
            or domain in report.web.sites_explored or report.web.fetch_stopped):
        return
    for start in site_discovery.official_start_urls(domain):
        if start in run.seen_urls:  # already read as a source (e.g. a search result): only its navigation is needed
            home = _fetch_navigation(run, start)
        else:
            home = _read_company_website(run, start, "official domain (identity check)", quiet=True)
        if home is not None:
            _explore_official_site(run, *home)
            return
    report.web.sites_explored.add(domain)  # unreachable: not tried again in this assessment
    report.warnings.append("Company website could not be fetched: official_domain_unreachable")


def _read_company_website(run: "_Pass", url: str, via: str, quiet: bool = False,
                          typed: bool = False) -> tuple[FetchResult, HtmlContent] | None:
    """The homepage plus a few ESG-relevant links on it. Returns the homepage, or None when it could not be read.
    `typed`: the address the analyst entered, fetched as a browser would (its redirects still honour robots.txt)."""
    db, a, report, progress = run.db, run.a, run.report, run.progress
    settings = get_settings()
    company = a.company
    if url in run.seen_urls:
        return None
    try:
        progress(f"Reading company website {url}")
        db.commit()
        fetch_started = time.monotonic()
        try:
            home = fetch(url, expect="html", robots_txt="typed_address" if typed else "enforce")
        except SourceRejection as exc:
            report.web.record_fetch(exc.code)
            raise
        finally:
            report.fetch_seconds += time.monotonic() - fetch_started
        report.web.record_fetch(None)
        content = extract_html(home.content, home.final_url)
        report.official_domain = company.domain = registrable_domain(home.final_url) or company.domain
        s = _new_source(db, a, home.final_url, content.title or company.name, "company_website", True, via)
        run.seen_urls.add(url)
        run.seen_urls.add(home.final_url)
        s.content_type, s.content_sha256, s.bytes_downloaded, s.fetch_outcome = home.content_type, home.sha256, home.size, "ok"
        ingest_html(db, s, home.content, home.final_url)
        report.sources_processed += s.status == "PROCESSED" and apply_source_identity(db, a, s)
        if s.rejection_reason:
            db.commit()
            return None  # not this company's site: nothing on it is followed
        run.html_budget -= 1
        # Follow links to primary report PDFs found on the homepage.
        run.follow_report_links(content, home.final_url, "company website homepage")
        for link in relevant_links(content, report.official_domain, limit=settings.max_pages_per_site):
            if report.web.fetch_stopped:
                break
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
            if _not_counted(s):
                continue  # a wrong-company or robots-excluded document does not use up the budget
            if is_pdf:
                run.pdf_budget -= 1
            else:
                run.html_budget -= 1
                # After downloading a sub-page, also check it for report PDF links.
                if res is not None and res.kind == "html":
                    run.follow_report_links(extract_html(res.content, res.final_url), res.final_url, "company website sub-page")
        db.commit()
        return home, content
    except SourceRejection as exc:
        if not quiet:
            report.warnings.append(f"Company website could not be fetched: {exc.code}")
        report.sources_rejected += policy.is_security_rejection(exc.code)
        report.sources_unavailable += not policy.is_security_rejection(exc.code)
        log_source_rejection(db, exc, assessment_id=a.id, url=url, stage="company_website")
    except SQLAlchemyError:
        raise
    except Exception as exc:  # noqa: BLE001 — malformed untrusted content must not stop the assessment
        if not quiet:
            report.warnings.append(f"Company website could not be fetched: {type(exc).__name__}")
        report.sources_unavailable += 1
    return None


def _fetch_navigation(run: "_Pass", url: str) -> tuple[FetchResult, HtmlContent] | None:
    """A page already recorded as a source, read again only for its links; no second source is created."""
    run.db.commit()
    try:
        res = fetch(url, expect="html")
    except SourceRejection as exc:
        run.report.web.record_fetch(exc.code)
        return None
    run.report.web.record_fetch(None)
    return res, extract_html(res.content, res.final_url)


def _fetch_site_text(run: "_Pass", url: str) -> tuple[str | None, str]:
    """robots.txt or a sitemap: (text or None, outcome code)."""
    run.db.commit()
    try:
        res = fetch(url, expect="text")
    except SourceRejection as exc:
        run.report.web.record_fetch(exc.code)
        return None, f"{exc.code}:{str(exc)[:12]}"
    run.report.web.record_fetch(None)
    return res.content.decode("utf-8", errors="replace"), "ok"


def _read_robots(run: "_Pass", base: str) -> robots.SiteRules | None:
    """The site's robots.txt rules from the shared per-origin cache (the same rules safe_fetch enforces); None when
    the site is to be left alone for now (robots.txt answered 5xx) or robots.txt could not be fetched."""
    def fetch_text(robots_url: str) -> str:
        return fetch(robots_url, expect="text", robots_txt="off").content.decode("utf-8", errors="replace")

    run.db.commit()
    try:
        rules = robots.cache.rules_for(base, fetch_text)
    except SourceRejection as exc:
        run.report.web.record_fetch(exc.code)
        return None
    return None if rules.disallow_all else rules


def _sitemap_urls(run: "_Pass", base: str, rules: site_discovery.SiteRules, domain: str) -> tuple[list[str], int]:
    """Addresses listed in the site's sitemaps (robots.txt Sitemap: lines, else /sitemap.xml): at most
    SITEMAP_MAX_FILES files, index files followed to their most relevant children. Returns (urls, files read)."""
    queue = [u for u in rules.sitemaps if site_discovery.on_site(u, domain)] or [base + "sitemap.xml"]
    seen: set[str] = set()
    urls: list[str] = []
    files = 0
    while queue and files < get_settings().sitemap_max_files and not run.out_of_time():
        url = queue.pop(0)
        if url in seen or url.lower().split("?")[0].endswith(".gz") or not rules.allows(url):
            continue
        seen.add(url)
        text, _ = _fetch_site_text(run, url)
        files += 1
        if text is None:
            continue
        is_index, locs = site_discovery.parse_sitemap(text)
        if is_index:
            queue = site_discovery.rank_sitemaps([u for u in locs if site_discovery.on_site(u, domain)]) + queue
        else:
            urls.extend(locs[: site_discovery.MAX_SITEMAP_LOCS - len(urls)])
    return urls, files


def _explore_official_site(run: "_Pass", home: FetchResult, content: HtmlContent) -> None:
    """Report, ESG, investor and archive pages and report PDFs found through the site's own navigation and sitemaps,
    best first, within WEBSITE_MAX_PAGES / WEBSITE_MAX_PDFS. robots.txt is respected; every document passes the same
    identity and evidence gates as any other source."""
    settings = get_settings()
    db, a, report = run.db, run.a, run.report
    domain = (registrable_domain(home.final_url) or report.official_domain or "").lower()
    if not settings.website_discovery_enabled or not domain or domain in report.web.sites_explored:
        return
    report.web.sites_explored.add(domain)
    parts = urlparse(home.final_url)
    base = f"{parts.scheme}://{parts.netloc}/"
    run.progress(f"Reading the official website's navigation and sitemaps ({parts.netloc})")
    rules = _read_robots(run, base)
    if rules is None:
        log_event(db, "official_site_explored", assessment_id=a.id, details={"domain": domain, "robots": "temporarily_unavailable"})
        return
    sitemap_urls, sitemap_files = _sitemap_urls(run, base, rules, domain)
    ranked = site_discovery.rank_site_urls(list(content.links) + [(u, "") for u in sitemap_urls], domain, a.year)
    stage = _Pass(db, a, report, run.progress, run.seen_urls, settings.website_max_pdfs, settings.website_max_pages,
                  settings.website_max_pages + settings.website_max_pdfs, run.deadline)
    read = 0
    for _, url in ranked:
        if stage.out_of_time() or (stage.pdf_budget <= 0 and stage.html_budget <= 0):
            break
        is_pdf = url.lower().split("?")[0].endswith(".pdf")
        if url in stage.seen_urls or (stage.pdf_budget if is_pdf else stage.html_budget) <= 0:
            continue
        if not rules.allows(url):
            report.web.robots_skipped += 1
            continue
        stage.seen_urls.add(url)
        if not _admit(db, a, url, report):
            continue
        stype, official = classify_source(url, "", "", a.company.name, domain)
        s = _new_source(db, a, url, url.rstrip("/").rsplit("/", 1)[-1] or url, stype, official, "official website (navigation/sitemap)")
        run.progress(f"Downloading {url}")
        res = _download_into(db, s, report)
        if _not_counted(s):
            continue  # a wrong-company or robots-excluded document does not use up the budget
        read += 1
        if is_pdf:
            stage.pdf_budget -= 1
        else:
            stage.html_budget -= 1
            if res is not None and res.kind == "html":
                stage.follow_report_links(extract_html(res.content, res.final_url), res.final_url, "official website page")
    report.web.site_pages_read += read
    log_event(db, "official_site_explored", assessment_id=a.id,
              details={"domain": domain, "sitemap_files": sitemap_files, "sitemap_urls": len(sitemap_urls),
                       "candidates": len(ranked), "sources_read": read, "robots_skipped": report.web.robots_skipped})
    db.commit()


def recover_sources(db: Session, a: Assessment, provider: SearchProvider, earlier_queries: list[str],
                    on_progress: Callable[[str], None] | None = None, weak_factors: list[str] | None = None,
                    web: WebAccessState | None = None) -> DiscoveryReport:
    """The single bounded extra search when coverage is too thin: queries targeted at the factors with weak
    coverage first, then broader report queries; the same hard gates, and the identity settled earlier is never changed."""
    settings = get_settings()
    company = a.company
    report = DiscoveryReport(official_domain=company.domain, web=web or WebAccessState(settings.web_access_failure_limit))
    progress = on_progress or (lambda _msg: None)
    deadline = time.monotonic() + settings.recovery_time_budget_seconds
    profile = a.company_identity or {}
    aliases = list(profile.get("aliases") or [])
    limit = settings.recovery_max_queries
    targeted = build_factor_recovery_queries(company.name, aliases, company.domain, a.year, weak_factors or [], earlier_queries, limit)
    general = build_recovery_queries(company.name, aliases, company.domain, a.year, earlier_queries + targeted, limit)
    queries = _interleave_queries(targeted, general, limit)
    results = _run_searches(provider, queries, report, progress, pause=settings.recovery_search_pause_seconds, deadline=deadline)
    _resolve_identity(a, report, results, may_choose_entity=False)
    run = _Pass(db, a, report, progress, _SeenUrls(s.url for s in a.sources if s.url), settings.recovery_max_pdfs,
                settings.recovery_max_pages, settings.recovery_max_pages + settings.recovery_max_pdfs, deadline)
    if results:
        run.take_results(results)
    if report.web.search_unavailable:  # search cannot help: the official site still can, if not read already
        explore_validated_official_site(run)
    db.commit()
    return report


def _interleave_queries(targeted: list[str], general: list[str], limit: int) -> list[str]:
    """Factor queries first, but always keep room for report searches: the report answers most factors at once."""
    keep_general = min(len(general), max(2, limit // 3)) if targeted else len(general)
    head = targeted[: max(0, limit - keep_general)]
    return list(dict.fromkeys(head + general))[:limit]
