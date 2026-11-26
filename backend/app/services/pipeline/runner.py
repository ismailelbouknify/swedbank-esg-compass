"""Assessment pipeline (simple evidence-grounded RAG):

company -> source discovery -> text extraction -> chunking -> retrieval (BM25) -> one local-model call per
questionnaire question, which reads the questionnaire criteria and recommends a level with exact quotations ->
lightweight verification (JSON, scale, chunk ids, quotes, company sources; one corrective retry) -> analyst review.

No supporting evidence means no generated answer: an abstention is recorded as "no evidence found" or "insufficient
source coverage" (never as a low level), and an answer that fails verification is withheld for the analyst.

Runs in a background thread (FastAPI BackgroundTasks); progress is persisted on the
Assessment row so the UI can poll /status."""
from __future__ import annotations

import logging
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.config.loader import REPORTING_FACTOR, load_rules
from app.core.config import get_settings
from app.models.entities import AnalystDecision, Assessment, Evidence, Recommendation
from app.services.assessments.registry import supersede_previous
from app.services.audit import log_event, log_source_rejection
from app.services.company.identify import normalize_company_name
from app.services.company.resolution import AMBIGUOUS
from app.services.company.source_identity import SOURCE_COMPANY_MISMATCH
from app.services.documents.url_safety import registrable_domain
from app.services.llm.base import LLM_NOT_CONFIGURED_MESSAGE, LLMProvider, LLMStatus, NullLLMProvider
from app.services.llm.factory import get_llm_provider
from app.services.pipeline.coverage import (
    AMBIGUOUS_COMPANY_MESSAGE, BLOCKED_REPORT_MESSAGE, BLOCKED_REPORT_REASON, EVIDENCE_FOUND, INSUFFICIENT_COVERAGE_MESSAGE,
    INSUFFICIENT_SOURCE_COVERAGE, NOT_ASSESSED_LEVEL, CoverageCheck, assess_coverage, coverage_state,
)
from app.services.pipeline.discovery import WebAccessState, discover, recover_sources
from app.services.pipeline.network_notices import network_audit_details, network_notices
from app.services.pipeline.ingest import ingest_pdf
from app.services.pipeline import stop
from app.services.pipeline.source_gate import apply_source_identity, identity_for
from app.services.presentation import EVIDENCE_NOT_VERIFIED, NO_LEVEL_LABEL
from app.services.rag import answer as A
from app.services.rag.answer import Answer, Answerer
from app.services.rag.prompt import prompt_overhead_tokens
from app.services.rag.questions import Question, questionnaire
from app.services.rag.retrieval import Passage, Retriever, accepted_sources, load_passages
from app.services.search.base import SearchProvider
from app.services.search.providers import get_search_provider
from app.services.security.policy import SourceRejection, run_bounded

log = logging.getLogger(__name__)

STAGES = [
    "Identifying company",
    "Searching reports",
    "Processing documents",
    "Retrieving ESG evidence",
    "Answering questionnaire questions",
    "Verifying evidence",
    "Preparing assessment",
]
STAGE_PROGRESS = [5, 15, 30, 40, 45, 90, 97]
LLM_STATUS_TIMEOUT_SECONDS = 60
EXTERNAL_ACCESS_OFF_MESSAGE = ("Automatic search and retrieval of public sources are turned off for this installation. "
                               "Upload the sustainability report to include it in the analysis.")
COMPANY_MISMATCH_MESSAGE = ("One or more documents did not appear to concern the assessed company and were not used. "
                            "Check that the right report and company name were given.")
BUDGET_GAP_REASON = ("Insufficient source coverage \u2014 not every relevant passage could be analysed within this assessment's "
                     "processing limit, so no conclusion was drawn.")
COVERAGE_LIMITED_MESSAGE = ("Some public sources could not be processed, so a gap in this assessment may reflect an "
                            "unavailable source rather than missing company activity.")
NO_PASSAGE_TEXT = "No passage relevant to this question was found in the reviewed sources."
UNVERIFIED_TEXT = ("The suggested answer's quotations could not be found in the cited source text, so no level is "
                   "recommended. Please review this question.")
MODEL_ERROR_TEXT = "The local analysis component could not answer this question, so no level is recommended."
BUDGET_TEXT = ("This question was not analysed within the assessment's processing limit, so no level is recommended.")
NO_LLM_TEXT = ("The local analysis component is not available on this computer, so no level can be recommended. "
               "Install the local model, or set the level after reading the sources.")
AI_NOTE = ("AI-recommended level, based on the quoted passages. The quotations were checked against the sources; whether "
           "they meet the level's criteria is for the analyst to confirm.")
WITHHELD_NOTE = "A suggested answer was withheld because its evidence could not be verified."



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
        self.stage_seconds: dict[str, float] = {}
        self.web = WebAccessState(self.settings.web_access_failure_limit)  # one view of the network per assessment
        self.llm_calls = 0
        self.llm_seconds = 0.0
        self.duplicate_passages = 0
        self.questions: list[Question] = questionnaire()
        self.llm_available = False
        self.outcomes: dict[str, int] = {}

    @contextmanager
    def _timed(self, stage: str):
        """Wall time per pipeline stage, kept in the audit metrics: where an assessment spends its time."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stage_seconds[stage] = round(self.stage_seconds.get(stage, 0.0) + time.perf_counter() - started, 2)

    # ------------------------------------------------------------------ progress
    def _check_stop(self) -> None:
        if stop.requested(self.a.id):
            raise stop.AnalysisStopped()

    def _stage(self, idx: int, message: str | None = None, progress: float | None = None) -> None:
        self._check_stop()  # every stage change and discovery step is a point where the analysis can be stopped
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
        stop.clear(a.id)  # a retry starts afresh
        self._reset_previous_results()
        self.db.commit()
        try:
            self._identify()
            with self._timed("discovery"):
                self._search()
            with self._timed("document_processing"):
                self._process_documents()
            with self._timed("source_validation"):
                self._validate_source_identity()
            with self._timed("retrieval"):
                retrieved = self._retrieve()
            with self._timed("recovery"):
                recovered = self._recover_if_needed(retrieved[0])
                if recovered:
                    self._validate_source_identity()
            if recovered:
                with self._timed("retrieval"):
                    retrieved = self._retrieve()
            plans = retrieved[0]
            with self._timed("coverage"):
                self.coverage = self._coverage(plans)
                self._report_final_coverage()
            with self._timed("answering"):
                answers = self._answer(plans)
            with self._timed("persisting"):
                evidence = self._persist(answers)
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
        except stop.AnalysisStopped:
            log.info("Assessment %s stopped on request", a.id)
            try:
                self.db.rollback()
                a = self.db.get(Assessment, a.id)
                a.status, a.error, a.warnings = "FAILED", stop.STOPPED_MESSAGE, self.warnings
                a.stage_message = stop.STOPPED_MESSAGE
                a.duration_seconds = _elapsed(a.started_at, datetime.now(timezone.utc))
                log_event(self.db, "assessment_stopped", assessment_id=a.id, details={"stage": a.stage})
                self.db.commit()
            except Exception:  # noqa: BLE001
                log.exception("Assessment %s: failed to persist the stopped status", a.id)
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
        st = self._llm_status()
        self.a.llm_provider = st.provider
        self.a.llm_model = st.model if st.available else None
        if not st.available:
            msg = st.message or LLM_NOT_CONFIGURED_MESSAGE
            if not self.settings.allow_heuristic_fallback:
                raise RuntimeError(msg)
            # without the model nothing can be answered: sources are still gathered, every level is left to the analyst
            self.warnings.append(msg + " No levels can be recommended without it; the sources are listed for manual review.")
        self.llm_available = st.available
        log_event(self.db, "company_identified", assessment_id=self.a.id, entity_type="company", entity_id=c.id,
                  details={"name": c.name, "normalized": c.normalized_name, "domain": c.domain})
        self.db.commit()

    def _llm_status(self) -> LLMStatus:
        """status() is meant to be cheap and never raise, but loading a local library can fail (or stall) on a managed
        computer, e.g. when endpoint protection blocks a DLL. Then the analysis continues without the model."""
        try:
            return run_bounded(self.llm.status, LLM_STATUS_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001
            log.warning("llm_status_failed: provider=%s error=%s", getattr(self.llm, "provider_name", "?"), type(exc).__name__)
            msg = f"The local analysis component could not be started on this computer ({type(exc).__name__})."
            self.llm = NullLLMProvider(msg)
            return self.llm.status()

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
                           on_progress=lambda msg: self._stage(1, msg[:300]), web=self.web)
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
        c["fetch_seconds"] = c.get("fetch_seconds", 0.0) + rep.fetch_seconds
        c["pdf_parse_seconds"] = c.get("pdf_parse_seconds", 0.0) + rep.pdf_parse_seconds
        c["duplicate_documents"] = c.get("duplicate_documents", 0) + rep.duplicate_documents

    def _coverage(self, plans) -> CoverageCheck:
        """Source availability plus a retrieval diagnostic (passages retrieved per factor). No model call."""
        factors = [f["key"] for f in self.rules["factors"]]
        candidates = {f: sum(len(plans[q.key][0]) for q in self.questions if q.factor == f) for f in factors}
        without = sum(not n for n in candidates.values())
        return assess_coverage(self.db, self.a, self.settings, without, len(factors), factor_candidates=candidates)

    def _recover_if_needed(self, by_q) -> bool:
        """At most one recovery search, and only where a search was allowed in the first place. It runs when the
        sources as a whole are thin, or when a factor has neither enough coverage to say "no evidence" nor any
        candidate passage; its queries then target those factors first."""
        settings = self.settings
        check = self._coverage(by_q)
        weak = check.weak_factors([f["key"] for f in self.rules["factors"]])
        has_valid_upload = any(s.source_type == "uploaded_report" and s.status == "PROCESSED" for s in self.a.sources)
        # more searching cannot tell which company is meant: only the analyst can
        ambiguous = (self.a.company_identity or {}).get("status") == AMBIGUOUS
        allowed = (settings.recovery_search_enabled and self.a.enable_web_search and settings.enable_web_search
                   and settings.enable_external_fetch and not has_valid_upload and not ambiguous
                   and not (self.web.search_stopped and self.web.fetch_stopped))
        reasons = list(check.recovery_reasons) + [f"weak_factor_coverage:{f}" for f in weak]
        if not reasons or not allowed or self.recovery_runs >= 1:
            return False
        self.recovery_runs += 1
        self._stage(1, "Looking for more company sources")
        log_event(self.db, "recovery_search_started", assessment_id=self.a.id, details={"reasons": reasons, "weak_factors": weak})
        self.recovery_reasons = reasons
        rep = recover_sources(self.db, self.a, self.search, self.discovery_queries, on_progress=lambda msg: self._stage(1, msg[:300]),
                              weak_factors=weak, web=self.web)
        self._add_discovery_counts(rep)
        log_event(self.db, "recovery_search_completed", assessment_id=self.a.id,
                  details={"queries": rep.queries, "results_seen": rep.results_seen, "sources_processed": rep.sources_processed,
                           "searches_failed": rep.searches_failed})
        self.db.commit()
        self.db.refresh(self.a)
        return True

    def _report_web_access(self) -> None:
        """Why the network or a configured limit kept sources out, in plain words; the details go to the audit log."""
        notices = network_notices(self.web)
        if notices or self.web.issues or self.web.search_issues:
            log_event(self.db, "web_access_diagnostics", assessment_id=self.a.id, details=network_audit_details(self.web))
        for msg in notices:
            if msg not in self.warnings:
                self.warnings.append(msg)

    def _report_final_coverage(self) -> None:
        self._report_web_access()
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
        factors = [f["key"] for f in self.rules["factors"]]
        self.a.coverage_metrics = {
            "factor_coverage": check.factor_report(factors),
            "stage_seconds": dict(self.stage_seconds),
            "model_calls": self.llm_calls,
            "model_seconds": round(self.llm_seconds, 2),
            "passages_deduplicated": self.duplicate_passages,
            "model_call_ceiling": self.settings.max_llm_calls_per_assessment,
            "model_runtime": getattr(self, "llm_runtime", {}),  # context size, threads, model loads
            "model_retries": sum(1 for c in getattr(self, "call_log", []) if c.get("attempt") == 2),
            "model_call_log": getattr(self, "call_log", []),  # per call: tokens, prompt/generation time, verdict
            "questions": len(self.questions),
            "answer_outcomes": dict(self.outcomes),
            "questions_answered": self.outcomes.get(A.VERIFIED, 0),
            "fetch_seconds": round(c.get("fetch_seconds", 0.0), 2),
            "pdf_parse_seconds": round(c.get("pdf_parse_seconds", 0.0), 2),
            "duplicate_documents_skipped": c.get("duplicate_documents", 0),
            "total_seconds": round(sum(self.stage_seconds.values()), 2),
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

    # ------------------------------------------------------------------ retrieval -> answers -> records
    def _passage_budget(self, q: Question) -> int:
        """Tokens left for passages in the model's context window after the instructions, question and answer."""
        s = self.settings
        free = s.llm_context_tokens - s.llm_max_output_tokens - prompt_overhead_tokens(q, self.a.company.name, self.a.year) - 64
        return max(300, free)

    def _retrieve(self):
        self._stage(3)
        passages, self.duplicate_passages = load_passages(self.db, self.a)
        retriever = Retriever(passages)
        plans: dict[tuple[str, str], tuple[list[Passage], int]] = {
            q.key: retriever.retrieve(q, self.settings.rag_chunks_per_question, self._passage_budget(q)) for q in self.questions
        }
        log_event(self.db, "retrieval_completed", assessment_id=self.a.id,
                  details={"chunks_indexed": len(passages), "questions": len(plans),
                           "chunks_per_question": {f"{k[0]}:{k[1]}": len(v[0]) for k, v in plans.items()}})
        self.db.commit()
        return (plans,)

    def _answer(self, plans) -> list[Answer]:
        self._stage(4)
        s = self.settings
        answerer = Answerer(self.llm, self.a.company.name, self.a.year, max_output_tokens=s.llm_max_output_tokens,
                            call_budget=s.max_llm_calls_per_assessment or None, names_company=identity_for(self.a).is_named_in)
        accepted = set(accepted_sources(self.a))
        self.llm.abort_check = lambda: stop.requested(self.a.id)  # a running model call stops at its next token
        try:
            answers = self._answer_all(plans, answerer, accepted)
        finally:
            self.llm.abort_check = None
        self.llm_calls, self.llm_seconds = answerer.calls, answerer.seconds
        self.call_log = answerer.call_log
        self.llm_runtime = dict(getattr(self.llm, "runtime", {}) or {})
        return self._finish_answers(answers)

    def _answer_all(self, plans, answerer: Answerer, accepted: set[int]) -> list[Answer]:
        answers: list[Answer] = []
        for n, q in enumerate(self.questions):
            self._check_stop()
            passages, candidates = plans[q.key]
            self.a.stage_message = (f"Answering question {n + 1} of {len(self.questions)}: {q.factor_label}"
                                    + ("" if q.factor == REPORTING_FACTOR else f" \u2014 {q.dimension_label}"))
            self.a.progress = STAGE_PROGRESS[4] + (STAGE_PROGRESS[5] - STAGE_PROGRESS[4]) * n / len(self.questions)
            self.db.commit()
            if passages and not self.llm_available:
                answers.append(Answer(q, A.MODEL_ERROR, passages, errors=["model not available"], candidates=candidates))
                continue
            answers.append(answerer.answer(q, passages, accepted, candidates))
        self._check_stop()  # the last call may have been interrupted by the stop request
        return answers

    def _finish_answers(self, answers: list[Answer]) -> list[Answer]:
        self.a.extraction_mode = "llm" if self.llm_available else "none"
        for ans in answers:
            self.outcomes[ans.outcome] = self.outcomes.get(ans.outcome, 0) + 1
        log.info("rag_answering: questions=%d llm_calls=%d llm_seconds=%.1f outcomes=%s",
                 len(answers), self.llm_calls, self.llm_seconds, self.outcomes)
        log_event(self.db, "questions_answered", assessment_id=self.a.id,
                  details={"llm_calls": self.llm_calls, "llm_seconds": round(self.llm_seconds, 1), "outcomes": self.outcomes})
        self.db.commit()
        return answers

    def _evidence_rows(self, ans: Answer, verified: bool) -> list[Evidence]:
        """Quotations as evidence records. Source, page and title come from the chunk records, never from the model."""
        rows = []
        level = ans.level if verified else ans.proposed_level
        for quote in ans.quotes:
            p = quote.passage
            if p is None:
                continue  # an unknown chunk id has no source to record; the error is in the recommendation's audit data
            ok = verified and quote.verified
            ev = Evidence(
                assessment_id=self.a.id, company_id=self.a.company_id, factor=ans.question.factor, dimension=ans.question.dimension,
                claim_type="rag_quote", claim_text=(ans.explanation or "")[:2000],
                facts={"chunk_ref": p.label, "level": level, "section": p.section, "report_year": p.report_year, "table": p.is_table},
                source_id=p.source_id, chunk_id=p.chunk_id, source_title=p.source_title, source_url=p.source_url,
                page_number=p.page_number, evidence_text=quote.text[:4000], extraction_method="llm", model_name=self.llm.model_name,
                verification_status="VERIFIED" if ok else "REJECTED",
                verification_notes=[f"quotation found in {p.label}"] if quote.verified else [f"quotation not found in {p.label}"],
                evidence_lifecycle="accepted" if ok else "rejected", rejection_codes=[] if ok else list(ans.codes),
            )
            self.db.add(ev)
            self.db.flush()
            rows.append(ev)
        return rows

    def _persist(self, answers: list[Answer]) -> list[Evidence]:
        self._stage(5)
        check = self.coverage or assess_coverage(self.db, self.a, self.settings, 0, 0)
        all_rows: list[Evidence] = []
        for ans in answers:
            q = ans.question
            verified = ans.outcome == A.VERIFIED
            rows = self._evidence_rows(ans, verified) if ans.outcome in (A.VERIFIED, A.UNVERIFIED) else []
            all_rows += rows
            notes: list[str] = []
            next_missing: list[str] = []
            if verified:
                level, status, coverage, reason = ans.level, "FOUND", EVIDENCE_FOUND, ans.explanation
                notes.append(AI_NOTE)
                check_state = ("met", "Every quotation was found in the cited source passage.")
            elif ans.outcome in (A.ABSTAINED, A.NO_PASSAGES):
                coverage = coverage_state([], check, q.factor)
                level, status = NOT_ASSESSED_LEVEL, coverage
                if coverage == INSUFFICIENT_SOURCE_COVERAGE:
                    reason = (BLOCKED_REPORT_REASON if check.blocked_primary_reports else
                              "Insufficient source coverage \u2014 the system could not access enough reliable company information.")
                    next_missing = ["Upload the company's sustainability or annual report to continue the assessment."]
                else:
                    reason = NO_LEVEL_LABEL[coverage] + ". " + (NO_PASSAGE_TEXT if ans.outcome == A.NO_PASSAGES or not ans.explanation
                                                                else ans.explanation)
                check_state = ("unknown", "No level was recommended, so there was nothing to verify.")
            else:
                level, coverage = NOT_ASSESSED_LEVEL, None
                status = EVIDENCE_NOT_VERIFIED if ans.outcome == A.UNVERIFIED else "REQUIRES_REVIEW"
                reason = NO_LEVEL_LABEL[status] + ". " + {
                    A.UNVERIFIED: UNVERIFIED_TEXT, A.BUDGET: BUDGET_TEXT,
                    A.MODEL_ERROR: NO_LLM_TEXT if not self.llm_available else MODEL_ERROR_TEXT}[ans.outcome]
                if ans.outcome == A.UNVERIFIED:
                    notes.append(WITHHELD_NOTE)
                    if ans.explanation:
                        reason += f" Suggested explanation (not verified): {ans.explanation}"
                check_state = ("not_found", "; ".join(ans.errors) or "The answer could not be checked.")
            rec = Recommendation(
                assessment_id=self.a.id, factor=q.factor, dimension=q.dimension, recommended_level=level, max_level=q.max_level,
                evidence_status=status, coverage_state=coverage, triggered_rule=f"rag:{ans.outcome}", reason=reason,
                ai_reason=reason, conditions={"evidence_verification": {
                    "label": "Evidence verification", "status": check_state[0],
                    "evidence_ids": [e.id for e in rows], "detail": check_state[1][:500]}},
                next_level_missing=next_missing, notes=notes, evidence_ids=[e.id for e in rows],
                confidence=0.0, confidence_label="Not rated",  # no uncalibrated confidence is presented
                confidence_breakdown={
                    "outcome": ans.outcome, "model_calls": ans.calls, "model_seconds": round(ans.seconds, 2),
                    "retrieved_chunks": [p.label for p in ans.passages], "retrieval_candidates": ans.candidates,
                    "verification_codes": ans.codes, "proposed_level": ans.proposed_level,
                },
            )
            self.db.add(rec)
            self.db.flush()
            log_event(self.db, "recommendation_generated", assessment_id=self.a.id, entity_type="recommendation", entity_id=rec.id,
                      details={"factor": q.factor, "dimension": q.dimension, "level": level, "outcome": ans.outcome,
                               "status": status, "evidence_ids": rec.evidence_ids, "verification": ans.codes})
        self.db.commit()
        return all_rows


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
