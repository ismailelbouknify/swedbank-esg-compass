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

from app.config.loader import FACTOR_DIMENSIONS, REPORTING_FACTOR, factor_label, load_rules, source_priority
from app.core.config import get_settings
from app.models.entities import (
    AnalystDecision, Assessment, Document, DocumentChunk, DocumentPage, Evidence, Recommendation, Source,
)
from app.services.assessments.registry import supersede_previous
from app.services.audit import log_event
from app.services.company.identify import normalize_company_name
from app.services.documents.url_safety import registrable_domain
from app.services.extraction.extractor import Candidate, FactExtractor, Passage, passage_signal
from app.services.llm.base import LLM_NOT_CONFIGURED_MESSAGE, LLMProvider
from app.services.llm.factory import get_llm_provider
from app.services.pipeline.discovery import discover
from app.services.pipeline.ingest import ingest_pdf
from app.services.retrieval.bm25 import BM25Index, query_for
from app.services.scoring.common import EvidenceItem, ScoreResult
from app.services.scoring.confidence import compute_confidence, label_for
from app.services.scoring.execution import score_execution
from app.services.scoring.performance import score_performance
from app.services.scoring.planning import score_planning
from app.services.scoring.reporting import score_reporting
from app.services.search.base import SearchProvider
from app.services.search.providers import get_search_provider
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
OFFICIAL_TYPES = ("uploaded_report", "sustainability_report", "annual_report", "company_website")


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
            passages_by_q, all_passages, report_passages = self._retrieve()
            candidates = self._extract(passages_by_q, all_passages, report_passages)
            evidence = self._persist_evidence(candidates)
            self._score(evidence)
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
            self.db.rollback()
            a = self.db.get(Assessment, a.id)
            a.status = "FAILED"
            a.error = f"{type(exc).__name__}: {exc}"
            a.warnings = self.warnings
            a.duration_seconds = _elapsed(a.started_at, datetime.now(timezone.utc))
            log_event(self.db, "assessment_failed", assessment_id=a.id, details={"error": a.error, "trace": traceback.format_exc()[-2000:]})
            self.db.commit()

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
        do_search = self.a.enable_web_search and not uploaded
        if uploaded:
            self._stage(1, "Uploaded report provided \u2014 using it as the primary source" + (" (plus company website)" if self.a.company.website else ""))
        if not self.a.enable_web_search and not uploaded and not self.a.company.website:
            self.warnings.append("No report uploaded, no website given and web search disabled: nothing to review.")
        if do_search or self.a.company.website:
            rep = discover(self.db, self.a, self.search, do_search=do_search,
                           on_progress=lambda msg: self._stage(1, msg[:300]))
            self.warnings.extend(rep.warnings)
            log_event(self.db, "web_discovery_completed", assessment_id=self.a.id,
                      details={"queries": rep.queries, "results_seen": rep.results_seen, "official_domain": rep.official_domain})
        self.db.commit()
        self.db.refresh(self.a)

    def _process_documents(self) -> None:
        self._stage(2)
        for s in self.a.sources:
            if s.source_type == "uploaded_report" and s.document is None and s.file_path:
                try:
                    ingest_pdf(self.db, s, s.file_path)
                except ValueError as exc:
                    s.status, s.status_detail = "FAILED", str(exc)
                log_event(self.db, "document_processed", assessment_id=self.a.id, entity_type="source", entity_id=s.id,
                          details={"status": s.status, "detail": s.status_detail})
        for s in self.a.sources:
            if s.status == "OCR_REQUIRED":
                self.warnings.append(f"'{s.title or s.original_filename}' appears to be image-only (OCR_REQUIRED) and could not be analysed.")
        self.db.commit()
        self.db.refresh(self.a)

    def _passages(self) -> list[Passage]:
        rows = self.db.execute(
            select(DocumentChunk, Document, Source)
            .join(Document, DocumentChunk.document_id == Document.id)
            .join(Source, Document.source_id == Source.id)
            .where(Source.assessment_id == self.a.id, Source.status == "PROCESSED")
        ).all()
        out = []
        for ch, doc, src in rows:
            out.append(Passage(
                chunk_id=ch.id, document_id=doc.id, source_id=src.id, source_title=doc.title or src.title,
                source_url=src.url, source_type=src.source_type, source_quality=source_priority(src.source_type)["quality"],
                is_public=src.is_public, page_number=ch.page_number if doc.doc_kind == "pdf" else None,
                text=ch.text, doc_year=doc.report_year,
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
            if p.source_type in OFFICIAL_TYPES and p.source_id not in report_passages:
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
        for c in candidates:
            key = (c.factor, c.dimension, c.claim_type, c.passage.source_id, c.evidence_text[:120])
            if key in seen:
                continue
            seen.add(key)
            vr = verify_candidate(c, page_text(c.passage), source_ok=True)
            ev = Evidence(
                assessment_id=self.a.id, company_id=self.a.company_id, factor=c.factor, dimension=c.dimension,
                claim_type=c.claim_type, claim_text=c.claim_text[:2000], facts=c.facts,
                extracted_value=c.extracted_value, extracted_unit=(str(c.extracted_unit)[:50] if c.extracted_unit else None),
                target_year=c.target_year, baseline_year=c.baseline_year,
                source_id=c.passage.source_id, chunk_id=c.passage.chunk_id, source_title=c.passage.source_title,
                source_url=c.passage.source_url, page_number=c.passage.page_number, evidence_text=vr.quote[:4000],
                extraction_method=c.method, model_name=c.model_name, llm_confidence=c.llm_confidence,
                verification_status=vr.status, verification_notes=vr.notes,
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

        def item(e: Evidence) -> EvidenceItem:
            if e.extraction_method == "llm":
                cert = min(0.9, e.llm_confidence if e.llm_confidence is not None else 0.6)
            elif e.extraction_method == "regex":
                cert = 0.8  # deterministic pattern match on verbatim text
            else:
                cert = heuristic_cert
            return EvidenceItem(e.id, e.claim_type, dict(e.facts or {}), e.verification_status == VERIFIED, e.source_id,
                                source_priority(e.source.source_type)["quality"], cert, e.extraction_method, e.page_number)

        items = {e.id: item(e) for e in evidence}
        processed = [s for s in self.a.sources if s.status == "PROCESSED"]
        coverage = {"sources_reviewed": len(processed),
                    "has_report": any(s.source_type in ("uploaded_report", "sustainability_report", "annual_report") for s in processed)}

        def ev_for(factor: str, dim: str) -> list[EvidenceItem]:
            return [items[e.id] for e in evidence if e.factor == factor and e.dimension == dim]

        results: list[tuple[str, str, ScoreResult, list[EvidenceItem]]] = []
        rep_items = ev_for(REPORTING_FACTOR, "reporting")
        results.append((REPORTING_FACTOR, "reporting", score_reporting(rep_items), rep_items))
        for f in self.rules["factors"]:
            label = f["label"]
            p_items = ev_for(f["key"], "planning")
            planning = score_planning(p_items, label)
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
            # persist evaluation details computed during scoring (e.g. performance change %)
            for it in its:
                if "evaluation" in it.facts:
                    ev = next(e for e in evidence if e.id == it.id)
                    ev.facts = {**(ev.facts or {}), "evaluation": it.facts["evaluation"]}
            rec = Recommendation(
                assessment_id=self.a.id, factor=factor, dimension=dim, recommended_level=r.recommended_level,
                max_level=r.max_level, evidence_status=status, triggered_rule=r.triggered_rule, reason=r.reason,
                conditions={k: c.as_dict() for k, c in r.conditions.items()}, next_level_missing=r.next_level_missing,
                notes=notes, evidence_ids=r.evidence_ids + [i for i in r.unverified_ids if i not in r.evidence_ids],
                confidence=conf, confidence_label=label, confidence_breakdown={**breakdown, "used_evidence_ids": r.evidence_ids},
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
    finally:
        db.close()
