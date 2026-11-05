"""Assessment pipeline:

documents/web -> retrieval (BM25) -> local LLM extracts structured facts -> verification
-> deterministic Python rules -> recommendations -> analyst review.

Runs in a background thread (FastAPI BackgroundTasks); progress is persisted on the
Assessment row so the UI can poll /status."""
from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.config.loader import FACTOR_DIMENSIONS, OFFICIAL_SOURCE_TYPES, REPORTING_FACTOR, factor_label, load_rules, source_priority
from app.core.config import get_settings
from app.models.entities import (
    AnalystDecision, Assessment, Document, DocumentChunk, DocumentPage, Evidence, Recommendation, Source,
)
from app.services.assessments.registry import supersede_previous
from app.services.audit import log_event, log_source_rejection
from app.services.company.identify import normalize_company_name
from app.services.company.resolution import AMBIGUOUS
from app.services.company.source_identity import SOURCE_COMPANY_MISMATCH
from app.services.documents.url_safety import registrable_domain
from app.services.extraction.extractor import Candidate, FactExtractor, Passage, passage_signal
from app.services.llm.base import LLM_NOT_CONFIGURED_MESSAGE, LLMProvider
from app.services.llm.factory import get_llm_provider
from app.services.pipeline.coverage import (
    AMBIGUOUS_COMPANY_MESSAGE, BLOCKED_REPORT_MESSAGE, BLOCKED_REPORT_REASON, INSUFFICIENT_COVERAGE_MESSAGE, INSUFFICIENT_SOURCE_COVERAGE, NOT_ASSESSED_LEVEL, CoverageCheck, assess_coverage, coverage_state,
)
from app.services.pipeline.discovery import discover, recover_sources
from app.services.pipeline.ingest import ingest_pdf
from app.services.pipeline.source_gate import apply_source_identity, identity_for, source_blocks_evidence
from app.services.retrieval.bm25 import BM25Index, query_for
from app.services.scoring.common import NO_EVIDENCE_FOUND, REQUIRES_REVIEW, EvidenceItem, ScoreResult, assert_scoring_evidence
from app.services.scoring.confidence import compute_confidence, label_for
from app.services.scoring.execution import score_execution
from app.services.scoring.performance import score_performance
from app.services.scoring.planning import score_planning
from app.services.scoring.reporting import score_reporting
from app.services.search.base import SearchProvider
from app.services.search.providers import get_search_provider
from app.services.security.policy import SourceRejection
from app.services.verification.verifier import VERIFIED, verify_candidate

log = logging.getLogger(__name__)

STAGES = [
    "Identifying company",
    "Searching reports",
    "Processing documents",
    "Retrieving ESG evidence",
    "Extracting facts",
    "Applying ESG rules",
    "Preparing assessment",
]
STAGE_PROGRESS = [5, 15, 30, 40, 45, 90, 97]
EXTERNAL_ACCESS_OFF_MESSAGE = ("Automatic search and retrieval of public sources are turned off for this installation. "
                               "Upload the sustainability report to include it in the analysis.")
COMPANY_MISMATCH_MESSAGE = ("One or more documents did not appear to concern the assessed company and were not used. "
                            "Check that the right report and company name were given.")
COVERAGE_LIMITED_MESSAGE = ("Some public sources could not be processed, so a gap in this assessment may reflect an "
                            "unavailable source rather than missing company activity.")



def _elapsed(start: datetime | None, end: datetime) -> float | None:
    if start is None:
        return None
    if start.tzinfo is None:  # SQLite hands back naive UTC datetimes
        start = start.replace(tzinfo=timezone.utc)
    return round(max(0.0, (end - start).total_seconds()), 1)


class AssessmentRunner:
    def __init__(self, db: Session, assessment_id: int, llm: LLMProvider | None = None, search: SearchProvider | None = None):
        self.db = db
        self.a: Assessment = db.get(Assessment, assessment_id)
        if self.a is None:
            raise ValueError(f"Assessment {assessment_id} not found")
        self.llm = llm or get_llm_provider()
        self.search = search or get_search_provider()
        self.settings = get_settings()
        self.rules = load_rules()
        self.warnings: list[str] = []
        self.coverage: CoverageCheck | None = None
        self.recovery_runs = 0
        self.discovery_counts: dict[str, int] = {}
        self.discovery_queries: list[str] = []

    # ------------------------------------------------------------------ progress
    def _stage(self, idx: int, message: str | None = None, progress: float | None = None) -> None:
        self.a.stage = idx + 1
        self.a.stage_message = message or STAGES[idx]
        self.a.progress = progress if progress is not None else STAGE_PROGRESS[idx]
        self.db.commit()

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        a = self.a
        if a.status != "RUNNING" or a.started_at is None:  # the /run endpoint already stamped the start time
            a.started_at = datetime.now(timezone.utc)
        a.status, a.error, a.completed_at, a.duration_seconds = "RUNNING", None, None, None
        self._reset_previous_results()
        self.db.commit()
        try:
            self._identify()
            self._search()
            self._process_documents()
            self._validate_source_identity()
            retrieved = self._retrieve()
            if self._recover_if_needed(retrieved[0]):
                self._validate_source_identity()
                retrieved = self._retrieve()
            passages_by_q, all_passages, report_passages = retrieved
            self.coverage = self._coverage(passages_by_q)
            self._report_final_coverage()
            candidates = self._extract(passages_by_q, all_passages, report_passages)
            evidence = self._persist_evidence(candidates)
            self._score(evidence)
            self._record_coverage_metrics(evidence)
            self._stage(6)
            a.status = "COMPLETED"
            a.progress = 100
            a.stage_message = "Assessment ready for analyst review"
            a.warnings = self.warnings
            a.completed_at = datetime.now(timezone.utc)
            a.duration_seconds = _elapsed(a.started_at, a.completed_at)
            prev = supersede_previous(self.db, a)
            log_event(self.db, "assessment_completed", assessment_id=a.id, entity_type="assessment", entity_id=a.id,
                      details={"warnings": self.warnings, "duration_seconds": a.duration_seconds,
                               "replaces_assessment_id": prev.id if prev else None})
            if prev is not None:
                log_event(self.db, "assessment_replaced", assessment_id=prev.id, entity_type="assessment", entity_id=prev.id,
                          details={"replaced_by": a.id})
            self.db.commit()
        except Exception as exc:  # noqa: BLE001
            log.exception("Assessment %s failed", a.id)
            error_msg = f"{type(exc).__name__}: {exc}"
            trace = traceback.format_exc()[-2000:]
            try:
                self.db.rollback()
                a = self.db.get(Assessment, a.id)
                a.status = "FAILED"
                a.error = error_msg
                a.warnings = self.warnings
                a.duration_seconds = _elapsed(a.started_at, datetime.now(timezone.utc))
                log_event(self.db, "assessment_failed", assessment_id=a.id, details={"error": error_msg, "trace": trace})
                self.db.commit()
            except Exception:  # noqa: BLE001
                log.exception("Assessment %s: failed to persist FAILED status after: %s", a.id, error_msg)

    def _reset_previous_results(self) -> None:
        aid = self.a.id
        rec_ids = [r for (r,) in self.db.execute(select(Recommendation.id).where(Recommendation.assessment_id == aid))]
        if rec_ids:
            log_event(self.db, "assessment_rerun", assessment_id=aid, details={"cleared_recommendations": len(rec_ids)})
        self.db.execute(delete(AnalystDecision).where(AnalystDecision.assessment_id == aid))
        self.db.execute(delete(Recommendation).where(Recommendation.assessment_id == aid))
        self.db.execute(delete(Evidence).where(Evidence.assessment_id == aid))
        for s in list(self.a.sources):
            if s.source_type != "uploaded_report":
                self.db.delete(s)  # web sources are re-discovered
            elif s.document is not None:
                self.db.delete(s.document)  # uploaded files are re-processed
                s.status = "DISCOVERED"
                s.company_match, s.rejection_reason = None, None
        self.db.flush()
        self.db.refresh(self.a)

    # ------------------------------------------------------------------ stages
    def _identify(self) -> None:
        self._stage(0)
        c = self.a.company
        c.normalized_name = normalize_company_name(c.name)
        self.a.normalized_company_name = c.normalized_name
        if c.website and not c.domain:
            c.domain = registrable_domain(c.website)
        st = self.llm.status()
        self.a.llm_provider = st.provider
        self.a.llm_model = st.model if st.available else None
        if not st.available:
            msg = st.message or LLM_NOT_CONFIGURED_MESSAGE
            if self.settings.allow_heuristic_fallback:
                self.warnings.append(msg + " Using deterministic keyword/regex extraction instead (lower confidence; all levels require analyst review).")
            else:
                raise RuntimeError(msg)
        log_event(self.db, "company_identified", assessment_id=self.a.id, entity_type="company", entity_id=c.id,
                  details={"name": c.name, "normalized": c.normalized_name, "domain": c.domain})
        self.db.commit()

    def _search(self) -> None:
        self._stage(1)
        uploaded = [s for s in self.a.sources if s.source_type == "uploaded_report"]
        settings = self.settings
        external_on = settings.enable_external_fetch
        do_search = self.a.enable_web_search and settings.enable_web_search and external_on and not uploaded
        fetch_website = external_on and bool(self.a.company.website)
        if uploaded:
            self._stage(1, "Uploaded report provided \u2014 using it as the primary source" + (" (plus company website)" if self.a.company.website else ""))
        if not uploaded and not do_search and not fetch_website:
            if not (settings.enable_web_search and external_on):  # the installation, not the analyst, turned it off
                self.warnings.append(EXTERNAL_ACCESS_OFF_MESSAGE)
            else:
                self.warnings.append("No report uploaded, no website given and web search disabled: nothing to review.")
        if do_search or fetch_website:
            rep = discover(self.db, self.a, self.search, do_search=do_search,
                           on_progress=lambda msg: self._stage(1, msg[:300]))
            self.warnings.extend(rep.warnings)
            if rep.coverage_limited:
                self.warnings.append(COVERAGE_LIMITED_MESSAGE)
            self._add_discovery_counts(rep)
            log_event(self.db, "web_discovery_completed", assessment_id=self.a.id,
                      details={"queries": rep.queries, "results_seen": rep.results_seen, "official_domain": rep.official_domain,
                               "sources_processed": rep.sources_processed, "sources_rejected": rep.sources_rejected,
                               "sources_unavailable": rep.sources_unavailable, "candidates_rejected": rep.candidates_rejected})
        self.db.commit()
        self.db.refresh(self.a)

    def _process_documents(self) -> None:
        self._stage(2)
        for s in self.a.sources:
            if s.source_type == "uploaded_report" and s.document is None and s.file_path:
                try:
                    ingest_pdf(self.db, s, s.file_path)
                except SourceRejection as exc:
                    s.status, s.status_detail, s.fetch_outcome = "FAILED", str(exc), exc.code
                    log_source_rejection(self.db, exc, assessment_id=self.a.id, url=None, source_id=s.id, stage="upload")
                except ValueError as exc:
                    s.status, s.status_detail = "FAILED", str(exc)
                log_event(self.db, "document_processed", assessment_id=self.a.id, entity_type="source", entity_id=s.id,
                          details={"status": s.status, "detail": s.status_detail})
        for s in self.a.sources:
            if s.status == "OCR_REQUIRED":
                self.warnings.append(f"'{s.title or s.original_filename}' appears to be image-only (OCR_REQUIRED) and could not be analysed.")
        self.db.commit()
        self.db.refresh(self.a)

    def _validate_source_identity(self) -> None:
        """Hard gate before retrieval: a source about another company never reaches extraction or scoring."""
        for s in self.a.sources:
            if s.status == "PROCESSED" and s.company_match is None:
                apply_source_identity(self.db, self.a, s)
        rejected = sum(s.rejection_reason == SOURCE_COMPANY_MISMATCH for s in self.a.sources)
        if rejected:
            self.warnings.append(COMPANY_MISMATCH_MESSAGE)
        log_event(self.db, "source_identity_checked", assessment_id=self.a.id,
                  details={"sources": {str(s.id): s.company_match for s in self.a.sources if s.company_match}, "rejected": rejected})
        self.db.commit()
        self.db.refresh(self.a)

    def _add_discovery_counts(self, rep) -> None:
        self.discovery_queries += rep.queries
        c = self.discovery_counts
        c["candidates_rejected"] = c.get("candidates_rejected", 0) + rep.candidates_rejected + rep.candidates_other_entity
        c["search_results_seen"] = c.get("search_results_seen", 0) + rep.results_seen
        c["searches_failed"] = c.get("searches_failed", 0) + rep.searches_failed

    def _coverage(self, by_q) -> CoverageCheck:
        """Uses only the cheap re-ranking signal on retrieved passages: no model call is needed to see a thin result."""
        factors = [f["key"] for f in self.rules["factors"]]
        without = sum(not any(passage_signal(f, dim, p.text) > 0 for dim in FACTOR_DIMENSIONS for p in by_q.get((f, dim), [])) for f in factors)
        return assess_coverage(self.db, self.a, self.settings, without, len(factors))

    def _recover_if_needed(self, by_q) -> bool:
        """At most one recovery search, and only where a search was allowed in the first place."""
        settings = self.settings
        check = self._coverage(by_q)
        has_valid_upload = any(s.source_type == "uploaded_report" and s.status == "PROCESSED" for s in self.a.sources)
        # more searching cannot tell which company is meant: only the analyst can
        ambiguous = (self.a.company_identity or {}).get("status") == AMBIGUOUS
        allowed = (settings.recovery_search_enabled and self.a.enable_web_search and settings.enable_web_search
                   and settings.enable_external_fetch and not has_valid_upload and not ambiguous)
        if not check.recovery_reasons or not allowed or self.recovery_runs >= 1:
            return False
        self.recovery_runs += 1
        self._stage(1, "Looking for more company sources")
        log_event(self.db, "recovery_search_started", assessment_id=self.a.id, details={"reasons": check.recovery_reasons})
        self.recovery_reasons = list(check.recovery_reasons)
        rep = recover_sources(self.db, self.a, self.search, self.discovery_queries, on_progress=lambda msg: self._stage(1, msg[:300]))
        self._add_discovery_counts(rep)
        log_event(self.db, "recovery_search_completed", assessment_id=self.a.id,
                  details={"queries": rep.queries, "results_seen": rep.results_seen, "sources_processed": rep.sources_processed,
                           "searches_failed": rep.searches_failed})
        self.db.commit()
        self.db.refresh(self.a)
        return True

    def _report_final_coverage(self) -> None:
        if (self.a.company_identity or {}).get("status") == AMBIGUOUS:
            self.warnings.append(AMBIGUOUS_COMPANY_MESSAGE)
        for blocked in self.coverage.blocked_primary_reports:
            log_event(self.db, "primary_report_blocked", assessment_id=self.a.id, details=blocked)
        if self.coverage.blocked_primary_reports:
            self.warnings.append(BLOCKED_REPORT_MESSAGE)
        elif not self.coverage.sufficient:
            self.warnings.append(INSUFFICIENT_COVERAGE_MESSAGE)

    def _record_coverage_metrics(self, evidence: list[Evidence]) -> None:
        """Audit export only. Not shown in the normal UI."""
        check, c = self.coverage, self.discovery_counts
        accepted = [e for e in evidence if e.evidence_lifecycle == "accepted"]
        recs = self.db.scalars(select(Recommendation).where(Recommendation.assessment_id == self.a.id)).all()
        self.a.coverage_metrics = {
            "sources_discovered": len(self.a.sources) + c.get("candidates_rejected", 0),
            "sources_validated": check.valid_sources,
            "sources_rejected_company": check.rejected_company,
            "official_sources_processed": check.official_sources,
            "relevant_official_sources": check.relevant_official_sources,
            "primary_reports_processed": check.primary_reports,
            "candidate_evidence_count": len(evidence),
            "accepted_evidence_count": len(accepted),
            "factors_with_evidence": len({e.factor for e in accepted if e.factor != REPORTING_FACTOR}),
            "factors_with_insufficient_coverage": len({r.factor for r in recs if r.coverage_state == INSUFFICIENT_SOURCE_COVERAGE and r.factor != REPORTING_FACTOR}),
            "source_coverage_sufficient": check.sufficient,
            "primary_reports_blocked": len(check.blocked_primary_reports),
            "blocked_primary_reports": check.blocked_primary_reports,
            "recovery_search_runs": self.recovery_runs,
            "recovery_reasons": getattr(self, "recovery_reasons", []),
            "search_queries": len(self.discovery_queries),
            "search_results_seen": c.get("search_results_seen", 0),
            "searches_failed": c.get("searches_failed", 0),
        }
        self.db.commit()

    def _passages(self) -> list[Passage]:
        rows = self.db.execute(
            select(DocumentChunk, Document, Source)
            .join(Document, DocumentChunk.document_id == Document.id)
            .join(Source, Document.source_id == Source.id)
            .where(Source.assessment_id == self.a.id, Source.status == "PROCESSED")
        ).all()
        out = []
        for ch, doc, src in rows:
            if src.company_match is None or source_blocks_evidence(src):
                continue  # no recorded identity decision means no evidence, whatever the reason
            out.append(Passage(
                chunk_id=ch.id, document_id=doc.id, source_id=src.id, source_title=doc.title or src.title,
                source_url=src.url, source_type=src.source_type, source_quality=source_priority(src.source_type)["quality"],
                is_public=src.is_public, page_number=ch.page_number if doc.doc_kind == "pdf" else None,
                text=ch.text, doc_year=doc.report_year, identity_basis=src.company_match,
            ))
        return out

    def _retrieve(self):
        self._stage(3)
        passages = self._passages()
        index = BM25Index([{"id": i, "text": p.text, "page_number": p.page_number} for i, p in enumerate(passages)])
        llm_ok = self.llm.status().available
        k = self.settings.max_chunks_per_question if llm_ok else self.settings.heuristic_chunks_per_question
        by_q: dict[tuple[str, str], list[Passage]] = {}
        for f in self.rules["factors"]:
            for dim in FACTOR_DIMENSIONS:
                terms, must = query_for(f["key"], dim)
                hits = index.search(terms, top_k=self.settings.retrieval_candidates, must_contain_any=must)
                # re-rank: passages with a dimension signal first, then source priority, then BM25
                ranked = sorted(
                    hits,
                    key=lambda h: (passage_signal(f["key"], dim, h.text) > 0, passages[h.chunk_id].source_quality, h.score),
                    reverse=True,
                )
                by_q[(f["key"], dim)] = [passages[h.chunk_id] for h in ranked[:k]]
        # one representative passage per processed official source (document-level reporting evidence)
        report_passages: dict[int, Passage] = {}
        for p in passages:
            if p.source_type in OFFICIAL_SOURCE_TYPES and p.source_id not in report_passages:
                report_passages[p.source_id] = p
        log_event(self.db, "retrieval_completed", assessment_id=self.a.id,
                  details={"chunks_indexed": len(passages), "questions": len(by_q)})
        self.db.commit()
        return by_q, passages, list(report_passages.values())

    def _extract(self, by_q, all_passages, report_passages) -> list[Candidate]:
        self._stage(4)
        total_q = len(by_q) + 1
        done = 0

        def progress() -> None:
            self.a.progress = STAGE_PROGRESS[4] + (STAGE_PROGRESS[5] - STAGE_PROGRESS[4]) * done / total_q
            self.db.commit()

        ex = FactExtractor(self.llm, self.a.year, self.settings.allow_heuristic_fallback)
        self.a.extraction_mode = ex.mode
        candidates: list[Candidate] = []
        # reporting: regex across all official chunks; LLM only on assurance passages
        candidates += ex.reporting(all_passages, report_passages)
        done += 1
        progress()
        for (fkey, dim), passages in by_q.items():
            self.a.stage_message = f"Extracting facts: {factor_label(fkey)} \u2014 {dim}"
            fn = {"planning": ex.planning, "execution": ex.execution, "performance": ex.performance}[dim]
            candidates += fn(fkey, passages)
            done += 1
            progress()
        self.warnings.extend(ex.warnings)
        if ex.llm_failures:
            self.warnings.append(f"{ex.llm_failures} LLM response(s) could not be parsed; those passages used heuristic extraction.")
        self.a.extraction_mode = ex.mode if ex.llm_calls == 0 else "llm"
        log_event(self.db, "fact_extraction_completed", assessment_id=self.a.id,
                  details={"candidates": len(candidates), "llm_calls": ex.llm_calls, "llm_failures": ex.llm_failures, "mode": ex.mode})
        self.db.commit()
        return candidates

    def _persist_evidence(self, candidates: list[Candidate]) -> list[Evidence]:
        page_cache: dict[tuple[int, int], str | None] = {}

        def page_text(p: Passage) -> str | None:
            key = (p.document_id, p.page_number or 1)
            if key not in page_cache:
                row = self.db.execute(select(DocumentPage.text).where(
                    DocumentPage.document_id == p.document_id, DocumentPage.page_number == (p.page_number or 1))).first()
                page_cache[key] = row[0] if row else None
            return page_cache[key]

        out: list[Evidence] = []
        seen: set[tuple] = set()
        identity = identity_for(self.a)
        sources_by_id = {s.id: s for s in self.a.sources}
        for c in candidates:
            key = (c.factor, c.dimension, c.claim_type, c.passage.source_id, c.evidence_text[:120])
            if key in seen:
                continue
            seen.add(key)
            source = sources_by_id.get(c.passage.source_id)
            blocked = source_blocks_evidence(source)
            vr = verify_candidate(c, page_text(c.passage), source_ok=not blocked, company=identity)
            if blocked:
                vr.rejection_codes = ["source_company_mismatch"]
            ev = Evidence(
                assessment_id=self.a.id, company_id=self.a.company_id, factor=c.factor, dimension=c.dimension,
                claim_type=c.claim_type, claim_text=c.claim_text[:2000], facts=c.facts,
                extracted_value=c.extracted_value, extracted_unit=(str(c.extracted_unit)[:50] if c.extracted_unit else None),
                target_year=c.target_year, baseline_year=c.baseline_year,
                source_id=c.passage.source_id, chunk_id=c.passage.chunk_id, source_title=c.passage.source_title,
                source_url=c.passage.source_url, page_number=c.passage.page_number, evidence_text=vr.quote[:4000],
                extraction_method=c.method, model_name=c.model_name, llm_confidence=c.llm_confidence,
                verification_status=vr.status, verification_notes=vr.notes,
                evidence_lifecycle=vr.lifecycle, rejection_codes=vr.rejection_codes,
                evidence_origin=c.evidence_origin, llm_quote_failed=c.llm_quote_failed,
            )
            self.db.add(ev)
            self.db.flush()
            log_event(self.db, "evidence_extracted", assessment_id=self.a.id, entity_type="evidence", entity_id=ev.id,
                      details={"factor": c.factor, "dimension": c.dimension, "claim_type": c.claim_type,
                               "method": c.method, "model": c.model_name, "verification": vr.status})
            out.append(ev)
        self.db.commit()
        return out

    def _score(self, evidence: list[Evidence]) -> None:
        self._stage(5)
        heuristic_cert = self.rules["confidence"]["heuristic_extraction_certainty"]

        def _is_accepted(e: Evidence) -> bool:
            # Use lifecycle if set (new rows), fall back to verification_status for legacy rows.
            if e.evidence_lifecycle is not None:
                return e.evidence_lifecycle == "accepted"
            return e.verification_status == VERIFIED

        def item(e: Evidence) -> EvidenceItem:
            if e.extraction_method == "llm":
                cert = min(0.9, e.llm_confidence if e.llm_confidence is not None else 0.6)
            elif e.extraction_method == "regex":
                cert = 0.8  # deterministic pattern match on verbatim text
            else:
                cert = heuristic_cert
            return EvidenceItem(e.id, e.claim_type, dict(e.facts or {}), True, e.source_id,
                                source_priority(e.source.source_type)["quality"], cert, e.extraction_method, e.page_number)

        # Only accepted evidence reaches the rule engine. Soft candidates only route the answer to review.
        accepted = [e for e in evidence if _is_accepted(e) and not source_blocks_evidence(e.source)]
        accepted_ids = {e.id for e in accepted}
        fallback_ids = {e.id for e in accepted if e.evidence_origin}
        items = {e.id: item(e) for e in accepted}
        review_ids: dict[tuple[str, str], list[int]] = {}
        for e in evidence:
            if e.id not in accepted_ids and e.evidence_lifecycle != "rejected" and e.verification_status != "REJECTED":
                review_ids.setdefault((e.factor, e.dimension), []).append(e.id)
        processed = [s for s in self.a.sources if s.status == "PROCESSED"]
        coverage = {"sources_reviewed": len(processed),
                    "has_report": any(s.source_type in ("uploaded_report", "sustainability_report", "annual_report") for s in processed)}

        def ev_for(factor: str, dim: str) -> list[EvidenceItem]:
            return [items[e.id] for e in accepted if e.factor == factor and e.dimension == dim]

        source_coverage = self.coverage or assess_coverage(self.db, self.a, self.settings, 0, 0)
        results: list[tuple[str, str, ScoreResult, list[EvidenceItem]]] = []
        rep_items = ev_for(REPORTING_FACTOR, "reporting")
        results.append((REPORTING_FACTOR, "reporting", score_reporting(rep_items), rep_items))
        for f in self.rules["factors"]:
            label = f["label"]
            p_items = ev_for(f["key"], "planning")
            planning = score_planning(p_items, label, factor_key=f["key"])
            e_items = ev_for(f["key"], "execution")
            execution = score_execution(e_items, label, formal_goal_exists=planning.recommended_level >= 3)
            perf_items = ev_for(f["key"], "performance")
            perf = score_performance(perf_items, label, self.a.year)
            results += [(f["key"], "planning", planning, p_items), (f["key"], "execution", execution, e_items),
                        (f["key"], "performance", perf, perf_items)]

        heuristic_mode = self.a.extraction_mode == "heuristic"
        for factor, dim, r, its in results:
            conf, label, breakdown = compute_confidence(r, its, coverage)
            notes = list(r.notes)
            status = r.evidence_status
            if heuristic_mode and r.evidence_ids:
                # without the LLM nobody interpreted the context: never High, always analyst review
                conf = min(conf, 0.7)
                label = label_for(conf)
                breakdown["capped"] = "heuristic extraction (local LLM unavailable)"
                if status == "FOUND":
                    status = "REQUIRES_REVIEW"
                notes.append("Local LLM was not available; facts came from keyword/regex fallback extraction.")
            if fallback_ids & set(r.evidence_ids):
                # a sentence no model confirmed as the goal: it may count, but an analyst must confirm it
                conf = min(conf, 0.7)
                label = label_for(conf)
                breakdown["capped"] = "goal sentence taken directly from the source text"
                if status == "FOUND":
                    status = REQUIRES_REVIEW
                notes.append("This answer rests on a goal sentence taken directly from the source text; please confirm it.")
            to_review = review_ids.get((factor, dim), [])
            if to_review:
                notes.append(f"{len(to_review)} evidence item(s) could not be confirmed against the source and were not used for the level.")
                if status == NO_EVIDENCE_FOUND:
                    status = REQUIRES_REVIEW
            # persist evaluation details computed during scoring (e.g. performance change %)
            for it in its:
                if "evaluation" in it.facts:
                    ev = next(e for e in accepted if e.id == it.id)
                    ev.facts = {**(ev.facts or {}), "evaluation": it.facts["evaluation"]}
            # activities outside the time window explain the answer in its text but did not count towards it
            scoring_ids = assert_scoring_evidence(
                [i for i in r.evidence_ids if not (i in items and items[i].facts.get("in_window") is False)], accepted_ids)
            answer_coverage = coverage_state(scoring_ids, source_coverage)
            if answer_coverage == INSUFFICIENT_SOURCE_COVERAGE:
                # an assessment gap, not an answer: nothing was concluded, so no level is recorded
                r.recommended_level, r.triggered_rule = NOT_ASSESSED_LEVEL, INSUFFICIENT_SOURCE_COVERAGE
                r.reason = (BLOCKED_REPORT_REASON if source_coverage.blocked_primary_reports else
                            "Insufficient source coverage \u2014 the system could not access enough reliable company information.")
                r.next_level_missing = ["Upload the company's sustainability or annual report to continue the assessment."]
                status, conf = INSUFFICIENT_SOURCE_COVERAGE, 0.2
                label = label_for(conf)
            rec = Recommendation(
                assessment_id=self.a.id, factor=factor, dimension=dim, recommended_level=r.recommended_level,
                max_level=r.max_level, evidence_status=status, coverage_state=answer_coverage, triggered_rule=r.triggered_rule, reason=r.reason,
                ai_reason=r.reason,  # preserved separately so analyst overrides don't lose AI rationale
                conditions={k: c.as_dict() for k, c in r.conditions.items()}, next_level_missing=r.next_level_missing,
                notes=notes, evidence_ids=scoring_ids,
                confidence=conf, confidence_label=label,
                confidence_breakdown={**breakdown, "used_evidence_ids": scoring_ids, "review_evidence_ids": to_review},
            )
            self.db.add(rec)
            self.db.flush()
            log_event(self.db, "recommendation_generated", assessment_id=self.a.id, entity_type="recommendation", entity_id=rec.id,
                      details={"factor": factor, "dimension": dim, "level": r.recommended_level, "rule": r.triggered_rule,
                               "status": status, "evidence_ids": r.evidence_ids})
        self.db.commit()


def run_assessment(assessment_id: int, session_factory: Callable[[], Session] | sessionmaker | None = None,
                   llm: LLMProvider | None = None, search: SearchProvider | None = None) -> None:
    from app.db.session import SessionLocal

    db = (session_factory or SessionLocal)()
    try:
        AssessmentRunner(db, assessment_id, llm=llm, search=search).run()
    except BaseException as exc:  # noqa: BLE001 — last-resort: inner handler failed, ensure RUNNING→FAILED
        log.exception("run_assessment %d: unhandled %s escaped run()", assessment_id, type(exc).__name__)
        try:
            db.rollback()
            a = db.get(Assessment, assessment_id)
            if a and a.status == "RUNNING":
                a.status = "FAILED"
                a.error = f"[runner crash] {type(exc).__name__}: {exc}"
                db.commit()
        except Exception:  # noqa: BLE001
            log.exception("run_assessment %d: could not persist FAILED status", assessment_id)
    finally:
        db.close()
