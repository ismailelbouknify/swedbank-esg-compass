"""Turns retrieved passages into structured fact candidates.

LLM = "What facts exist in this passage?" -> structured JSON.
Python then normalises and cross-checks those facts (numbers must appear in the text,
benchmarks/owners must be explicitly worded, 'public' comes from source metadata).
If no local LLM is available, a deterministic keyword/regex extractor produces the same
structure with lower confidence and is labelled extraction_method='heuristic'."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config.loader import OFFICIAL_SOURCE_TYPES, REPORT_SOURCE_TYPES, get_factor, load_rules
from app.services.company.source_identity import PARENT_BOUNDARY
from app.services.extraction import gates
from app.services.extraction import patterns as P
from app.services.extraction import prompts
from app.services.extraction.gates import ACTIVITY_RE, STRONG_GOAL_RE
from app.services.llm.base import LLMProvider, LLMUnavailableError
from app.services.llm.json_utils import LLMOutputError, as_bool, as_confidence, as_float, as_year, parse_llm_json

log = logging.getLogger(__name__)


MAX_UNALIGNED_ROWS_PER_PASSAGE = 5
QUOTE_FALLBACK = "deterministic_quote_fallback"
QUOTE_FALLBACK_METHOD = "quote_fallback"
MAX_FALLBACK_SENTENCE_WORDS = 60


@dataclass
class Passage:
    chunk_id: int | None
    document_id: int | None
    source_id: int
    source_title: str | None
    source_url: str | None
    source_type: str
    source_quality: float
    is_public: bool
    page_number: int | None
    text: str
    doc_year: int | None = None
    retrieval_score: float = 0.0
    identity_basis: str | None = None  # how the source was tied to the company

    @property
    def label(self) -> str:
        page = f", page {self.page_number}" if self.page_number else ""
        return f"{self.source_title or self.source_url or 'Source'}{page}"


@dataclass
class Candidate:
    factor: str
    dimension: str
    claim_type: str
    claim_text: str
    facts: dict[str, Any]
    evidence_text: str
    passage: Passage
    method: str  # llm | heuristic | regex
    model_name: str | None = None
    llm_confidence: float | None = None
    extracted_value: float | None = None
    extracted_unit: str | None = None
    target_year: int | None = None
    baseline_year: int | None = None
    notes: list[str] = field(default_factory=list)
    evidence_origin: str | None = None
    llm_quote_failed: bool = False


class FactExtractor:
    def __init__(self, llm: LLMProvider | None, assessment_year: int, allow_heuristic: bool = True,
                 on_llm_call: Callable[[], None] | None = None, llm_budget: int | None = None):
        self.llm = llm
        self.year = assessment_year
        self.allow_heuristic = allow_heuristic
        self.use_llm = bool(llm and llm.is_available())
        self.warnings: list[str] = []
        self.llm_calls = 0
        self.llm_seconds = 0.0
        self.llm_budget = llm_budget  # None = no limit
        self.budget_exhausted = 0  # passages that went to the deterministic extractor because the budget was spent
        self.skipped_llm = {"overlap": 0, "deadline_passed": 0, "outside_window": 0}
        self._sent: dict[tuple[str, str], set[str]] = {}  # sentences already put to the model, per factor x dimension
        self.current_question: tuple[str, str] | None = None  # set by the caller
        self.budget_cut: set[tuple[str, str]] = set()  # questions with a passage the model did not read
        self.llm_failures = 0
        self._on_llm_call = on_llm_call
        self.rules = load_rules()

    @property
    def mode(self) -> str:
        return "llm" if self.use_llm else "heuristic"

    # ------------------------------------------------------------------ LLM plumbing
    def _ask(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        if not self.use_llm or self.llm is None:
            return None
        if self.llm_budget is not None and self.llm_calls >= self.llm_budget:
            self.budget_exhausted += 1
            if self.current_question:
                self.budget_cut.add(self.current_question)
            return None  # the caller falls back to deterministic extraction, which is verified like everything else
        self.llm_calls += 1
        if self._on_llm_call:
            self._on_llm_call()
        started = time.perf_counter()
        try:
            raw = self.llm.complete_json(prompts.SYSTEM_PROMPT, prompt, schema=schema)
            self.llm_seconds += time.perf_counter() - started
            return parse_llm_json(raw)
        except LLMUnavailableError as exc:
            self.use_llm = False
            self.warnings.append(f"Local LLM became unavailable during extraction ({exc}); remaining passages used heuristic extraction.")
            return None
        except LLMOutputError as exc:
            self.llm_failures += 1
            log.warning("llm_output_unparseable: failures=%d", self.llm_failures)  # never the output: it quotes report text
            return None
        except Exception as exc:  # noqa: BLE001
            self.llm_failures += 1
            log.warning("llm_call_failed: error=%s", type(exc).__name__)
            return None

    def _llm_meta(self) -> tuple[str, str | None]:
        return ("llm", self.llm.model_name if self.llm else None)

    @staticmethod
    def _passage_text(p: Passage, limit: int = 6000) -> str:
        return p.text[:limit]

    # ------------------------------------------------------------------ planning
    def planning(self, factor_key: str, passages: list[Passage]) -> list[Candidate]:
        factor = get_factor(factor_key)
        out: list[Candidate] = []
        for p in passages:
            # past-tense activity sentences ("we installed X to achieve our target") are execution, not goals
            # achievements ("we improved X") are not goals, and short headings ("Waste ambition") are not statements
            goal_sentences = [s for s in P.split_sentences(p.text) if is_goal_sentence(s, factor)]
            if not goal_sentences:
                continue  # prefilter: no goal language about this factor -> no LLM call needed
            if all(_deadline_has_passed(s, self.year) for s in goal_sentences):
                self.skipped_llm["deadline_passed"] += 1
                continue  # every goal here expired before the assessment year: none can be a current goal
            if self._already_sent(factor_key, "planning", goal_sentences):
                continue
            data = self._ask(prompts.planning_prompt(factor["label"], p.label, self._passage_text(p)), prompts.PLANNING_SCHEMA)
            if data is not None:
                c = self._planning_from_llm(factor, p, data)
                if c:
                    out.append(c)
                    fallback = self._planning_quote_fallback(factor, p, c, goal_sentences)
                    if fallback:
                        out.append(fallback)
                continue
            if self.allow_heuristic:
                out.extend(self._planning_heuristic(factor, p, goal_sentences))
        return out

    def _already_sent(self, factor_key: str, dimension: str, sentences: list[str]) -> bool:
        """Overlapping chunks repeat sentences. When every candidate sentence of this passage was already put to the
        model for the same question, asking again can only repeat that answer (and its evidence is deduplicated)."""
        key = (factor_key, dimension)
        seen = self._sent.setdefault(key, set())
        normalized = {P.normalize_ws(s) for s in sentences}
        if normalized <= seen:
            self.skipped_llm["overlap"] += 1
            return True
        seen |= normalized
        return False

    def _all_outside_window(self, sentences: list[str], p: Passage) -> bool:
        start = self.year - self.rules["lookback_years"] + 1
        for s in sentences:
            years = [y for y in P.find_years(s) if y <= self.year]
            effective = max(years) if years else p.doc_year
            if effective is None or effective >= start:
                return False
        return True

    def _planning_common(self, factor: dict[str, Any], p: Passage, sentence: str) -> dict[str, Any]:
        prules = self.rules["planning_rules"]
        bench_types = [k for k, terms in prules["benchmark_terms"].items() if P.matched_terms(sentence, terms)]
        owner_terms = P.matched_terms(p.text, prules["ownership_terms"])
        return {"benchmark_types_in_text": bench_types, "owner_terms_in_text": owner_terms}

    def _planning_from_llm(self, factor: dict[str, Any], p: Passage, d: dict[str, Any]) -> Candidate | None:
        if not as_bool(d.get("goal_found")):
            return None
        quote = str(d.get("evidence_quote") or "").strip()
        method, model = self._llm_meta()
        # Wording checks must run on source text: only trust the quote if it really is in the passage.
        quote_in_source = bool(quote) and P.normalize_ws(quote) in P.normalize_ws(p.text)
        common = self._planning_common(factor, p, quote if quote_in_source else p.text)
        notes: list[str] = []
        target_value = as_float(d.get("target_value"))
        target_year = as_year(d.get("target_year"))
        baseline_year = as_year(d.get("baseline_year"))
        quantitative = bool(as_bool(d.get("quantitative"))) and target_value is not None
        clear_owner = bool(as_bool(d.get("clear_owner")))
        if clear_owner and not common["owner_terms_in_text"]:
            clear_owner = False
            notes.append("LLM reported an owner, but no ownership wording was found in the passage; ignored.")
        benchmark = bool(as_bool(d.get("benchmark_found")))
        if benchmark and not common["benchmark_types_in_text"]:
            benchmark = False
            notes.append("LLM reported a benchmark, but no explicit benchmark wording was found; ignored.")
        facts = {
            "goal_found": True,
            "formalized": bool(as_bool(d.get("formalized"))),
            "public": p.is_public,
            "quantitative": quantitative,
            "time_bound": target_year is not None,
            "target_value": target_value,
            "target_unit": d.get("target_unit"),
            "target_year": target_year,
            "baseline_year": baseline_year,
            "clear_owner": clear_owner,
            "owner": d.get("owner") if clear_owner else None,
            "benchmark_found": benchmark,
            "benchmark_type": (common["benchmark_types_in_text"] or [None])[0] if benchmark else None,
        }
        return Candidate(
            factor["key"], "planning", "goal", _goal_claim(factor["label"], facts), facts, quote, p, method, model,
            as_confidence(d.get("confidence")), target_value, d.get("target_unit"), target_year, baseline_year, notes,
        )

    def _goal_facts(self, factor: dict[str, Any], p: Passage, s: str, t: P.TargetSignal) -> dict[str, Any]:
        common = self._planning_common(factor, p, s)
        return {
            "goal_found": True,
            "formalized": True,
            "public": p.is_public,
            "quantitative": t.quantitative,
            "time_bound": t.target_year is not None,
            "target_value": t.target_value,
            "target_unit": t.target_unit,
            "target_year": t.target_year,
            "baseline_year": t.baseline_year,
            "clear_owner": bool(common["owner_terms_in_text"]),
            "owner": ", ".join(common["owner_terms_in_text"][:2]) or None,
            "benchmark_found": bool(common["benchmark_types_in_text"]),
            "benchmark_type": (common["benchmark_types_in_text"] or [None])[0],
        }

    def _planning_heuristic(self, factor: dict[str, Any], p: Passage, sentences: list[str]) -> list[Candidate]:
        out = []
        for s in sentences[:3]:
            t = P.detect_target(s, p.doc_year or self.year)
            if t is None:
                continue
            facts = self._goal_facts(factor, p, s, t)
            out.append(Candidate(
                factor["key"], "planning", "goal", _goal_claim(factor["label"], facts), facts, s, p, "heuristic", None,
                None, t.target_value, t.target_unit, t.target_year, t.baseline_year,
            ))
        return out

    def _planning_quote_fallback(self, factor: dict[str, Any], p: Passage, llm_candidate: Candidate, sentences: list[str]) -> Candidate | None:
        """The model reported a goal but could not quote it: use the goal sentence the pre-filter already
        found in this passage. No model call, and none of the model's values are carried over. The result
        still has to pass verification like any other evidence before it can affect a score."""
        failure = gates.goal_quote_failure(llm_candidate.evidence_text, p.text)
        if failure is None or not self._factor_relevant_or_unquoted(factor, llm_candidate.evidence_text, failure):
            return None
        eligible: list[tuple[P.TargetSignal, str]] = []
        for s in sentences:
            t = P.detect_target(s, p.doc_year or self.year)
            if (t is not None and gates.FORWARD_PLANNING_RE.search(s) and P.normalize_ws(s) in P.normalize_ws(p.text)
                    and "|" not in s and len(s.split()) <= MAX_FALLBACK_SENTENCE_WORDS
                    and not _deadline_has_passed(s, self.year)):
                eligible.append((t, s))
        if not eligible:
            return None
        t, s = max(eligible, key=lambda pair: (pair[0].quantitative, pair[0].target_year is not None))
        facts = {**self._goal_facts(factor, p, s, t), "llm_quote_failure": failure}
        return Candidate(
            factor["key"], "planning", "goal", _goal_claim(factor["label"], facts), facts, s, p, QUOTE_FALLBACK_METHOD, None,
            None, t.target_value, t.target_unit, t.target_year, t.baseline_year,
            notes=[f"The extraction quote could not be used ({failure}); this goal sentence was taken from the source text instead."],
            evidence_origin=QUOTE_FALLBACK, llm_quote_failed=True,
        )

    @staticmethod
    def _factor_relevant_or_unquoted(factor: dict[str, Any], quote: str, failure: str) -> bool:
        """A located quote about another factor means the answer failed on factor, which is never rescued."""
        return failure != "not_a_goal_statement" or bool(P.matched_terms(quote, factor["keywords"]))

    # ------------------------------------------------------------------ execution
    def execution(self, factor_key: str, passages: list[Passage]) -> list[Candidate]:
        factor = get_factor(factor_key)
        link_terms = self.rules["execution_rules"]["goal_link_terms"]
        out: list[Candidate] = []
        for p in passages:
            sentences = [
                s for s in P.split_sentences(p.text)
                if ACTIVITY_RE.search(s) and P.matched_terms(s, factor["keywords"]) and not re.search(r"\b(will|plan to|plans to|intend)\b", s, re.I)
                and not gates.activity_not_performed(s) and not gates.external_actor_only(s) and not gates.external_purpose_only(s)
            ]
            if not sentences:
                continue
            if self._all_outside_window(sentences, p):
                self.skipped_llm["outside_window"] += 1
                continue  # every activity here is dated before the lookback window: none can count
            if self._already_sent(factor_key, "execution", sentences):
                continue
            data = self._ask(prompts.execution_prompt(factor["label"], p.label, self._passage_text(p)), prompts.EXECUTION_SCHEMA)
            if data is not None:
                if not as_bool(data.get("activity_found")):
                    continue
                quote = str(data.get("evidence_quote") or "").strip()
                description = str(data.get("activity_description") or "").strip()
                linked = bool(as_bool(data.get("linked_to_goal")))
                notes = []
                quote, mismatch = _align_activity_quote(description, quote, p.text, notes)
                quote_in_source = bool(quote) and P.normalize_ws(quote) in P.normalize_ws(p.text)
                if linked and not P.matched_terms(quote if quote_in_source else p.text, link_terms):
                    linked = False
                    notes.append("LLM linked the activity to a goal, but no goal/strategy wording was found; ignored.")
                year = as_year(data.get("activity_year"))
                if year is not None and not _year_in(year, quote if quote_in_source else p.text):
                    # the activity is in the text; the date the model gave it is not. An ungrounded date is dropped
                    # (the document's year is used instead); it does not make the activity itself unverifiable.
                    notes.append(f"The model dated the activity {year}, but that year is not in the source; the document year is used.")
                    year = None
                facts = self._activity_facts(p, year, linked, description or None)
                if mismatch:
                    facts["claim_quote_mismatch"] = True
                method, model = self._llm_meta()
                out.append(Candidate(factor_key, "execution", "activity", _activity_claim(factor["label"], facts, quote),
                                     facts, quote, p, method, model, as_confidence(data.get("confidence")), notes=notes))
                continue
            if self.allow_heuristic:
                for s in sentences[:2]:
                    years = [y for y in P.find_years(s) if y <= self.year]
                    facts = self._activity_facts(p, max(years) if years else None, bool(P.matched_terms(s, link_terms)), None)
                    out.append(Candidate(factor_key, "execution", "activity", _activity_claim(factor["label"], facts, s), facts, s, p, "heuristic"))
        return out

    def _activity_facts(self, p: Passage, year: int | None, linked: bool, description: Any) -> dict[str, Any]:
        lookback = self.rules["lookback_years"]
        effective_year = year or p.doc_year
        in_window = None if effective_year is None else (self.year - lookback + 1 <= effective_year <= self.year)
        return {
            "activity_found": True,
            "activity_description": description,
            "activity_year": year,
            "year_source": "stated" if year else ("document_year" if p.doc_year else "unknown"),
            "effective_year": effective_year,
            "in_window": in_window,
            "linked_to_goal": linked,
        }

    # ------------------------------------------------------------------ performance
    def performance(self, factor_key: str, passages: list[Passage]) -> list[Candidate]:
        factor = get_factor(factor_key)
        prules = self.rules["performance_rules"]
        out: list[Candidate] = []
        for p in passages:
            found_numeric = False
            # 1) deterministic series / relative changes
            aligned, unaligned = P.extract_series_with_rejections(p.text)
            for s in aligned:
                # the row's own label must name the metric: a unit ("MWh") or a table heading is not enough
                metric = _row_metric(s, factor)
                if metric is None:
                    continue
                found_numeric = True
                out.append(self._series_candidate(factor, p, s, metric, "regex", None, None))
            for s in unaligned[:MAX_UNALIGNED_ROWS_PER_PASSAGE]:
                metric = _row_metric(s, factor)
                if metric is None:
                    continue
                found_numeric = True  # the model would only guess the same alignment
                out.append(Candidate(factor_key, "performance", "performance_series", f"{metric['name']}: values could not be aligned with years",
                                     {"metric": metric["name"], "metric_label": s.metric_label[:200], "better": metric["better"], "unit": None,
                                      "points": [], "ambiguous_alignment": s.meta.get("reason", "unaligned")},
                                     s.quote, p, "regex"))
            for s in P.extract_from_to_changes(p.text, p.doc_year or self.year):
                metric = P.match_metric(s.metric_label + " " + s.quote, factor)
                if metric is None:
                    continue
                found_numeric = True
                out.append(self._series_candidate(factor, p, s, metric, "regex", None, None))
            for s in P.extract_relative_changes(p.text, p.doc_year or self.year):
                metric = P.match_metric(s.metric_label + " " + s.quote, factor)
                if metric is None:
                    continue
                found_numeric = True
                facts = {
                    "metric": metric["name"], "better": metric["better"], "unit": "%",
                    "change_pct": s.change_pct, "direction_word": s.direction_word,
                    "base_year": s.meta.get("base_year"), "end_year": s.meta.get("end_year"),
                }
                out.append(Candidate(factor_key, "performance", "relative_change",
                                     f"{metric['name']}: {s.direction_word} by {s.change_pct:g}% since {facts['base_year']}",
                                     facts, s.quote, p, "regex", extracted_value=s.change_pct, extracted_unit="%",
                                     baseline_year=facts["base_year"]))
            # 2) sector comparisons (explicit benchmark wording)
            for sent in P.split_sentences(p.text):
                if (P.matched_terms(sent, prules["sector_leading_terms"]) and P.matched_terms(sent, factor["keywords"])
                        and re.search(r"\d", P.YEAR_RE.sub("", sent))):  # needs a figure, not just a year
                    metric = P.match_metric(sent, factor)
                    favorable = _favorable(sent, metric["better"] if metric else None)
                    out.append(Candidate(factor_key, "performance", "sector_comparison",
                                         f"Sector/benchmark comparison: {sent[:160]}",
                                         {"statement": sent, "favorable": favorable, "terms": P.matched_terms(sent, prules["sector_leading_terms"])},
                                         sent, p, "regex"))
            # 3) LLM for prose the regexes could not structure
            if not found_numeric and P.matched_terms(p.text, factor["keywords"]) and _values_stated_with_years(p.text, factor):
                data = self._ask(prompts.performance_prompt(factor["label"], p.label, self._passage_text(p)), prompts.PERFORMANCE_SCHEMA)
                if data is not None and as_bool(data.get("metric_found")):
                    pts = []
                    for v in data.get("values") or []:
                        if isinstance(v, dict):
                            y, val = as_year(v.get("year")), as_float(v.get("value"))
                            if y is not None and val is not None:
                                pts.append((y, val))
                    label = str(data.get("metric_name") or "")
                    metric = P.match_metric(label + " " + str(data.get("evidence_quote") or ""), factor)
                    if metric and len(pts) >= 2:
                        s = P.Series(label, data.get("unit"), sorted(set(pts)), str(data.get("evidence_quote") or "").strip())
                        method, model = self._llm_meta()
                        out.append(self._series_candidate(factor, p, s, metric, method, model, as_confidence(data.get("confidence"))))
                        found_numeric = True
            # 4) unquantified improvement claims are recorded, never scored
            if not found_numeric:
                for sent in P.split_sentences(p.text):
                    if P.matched_terms(sent, prules["vague_improvement_terms"]) and P.matched_terms(sent, factor["keywords"]) and not P.find_percentages(sent):
                        out.append(Candidate(factor_key, "performance", "unquantified_improvement_claim",
                                             "Improvement claimed without quantitative data", {"statement": sent}, sent, p, "regex"))
                        break
        return out

    @staticmethod
    def _series_candidate(factor, p: Passage, s: P.Series, metric, method, model, conf) -> Candidate:
        facts = {
            "metric": metric["name"], "metric_label": s.metric_label[:200], "better": metric["better"],
            "unit": s.unit, "points": [[y, v] for y, v in sorted(s.points)],
        }
        pts = ", ".join(f"{y}: {v:g}" for y, v in sorted(s.points))
        label = P.label_without_units(s.metric_label)[:90] or metric["name"]
        unit = f" ({s.unit})" if s.unit and s.unit not in label else ""
        return Candidate(factor["key"], "performance", "performance_series", f"{label}{unit} [{metric['name']}] \u2014 {pts}",
                         facts, s.quote, p, method, model, conf, extracted_unit=s.unit)

    # ------------------------------------------------------------------ reporting
    def reporting(self, passages: list[Passage], report_sources: list[Passage]) -> list[Candidate]:
        """passages: chunks to scan for frameworks/assurance/regulation (regex over all of them);
        report_sources: one representative passage per official/uploaded document."""
        rules = self.rules["reporting_rules"]
        out: list[Candidate] = []
        for p in report_sources:
            if p.identity_basis == PARENT_BOUNDARY:
                continue  # the parent's report is the parent's reporting
            statement = P.reporting_statement(p.text)
            if p.source_type in REPORT_SOURCE_TYPES:
                # the source is the company's company-wide report (classified from the document itself)
                regular = p.doc_year is not None or bool(re.search(r"\b(annual|yearly|every year|each year)\b", p.text, re.I))
                out.append(Candidate("reporting", "reporting", "report_published",
                                     f"Sustainability information published in '{p.source_title or 'report'}'"
                                     + (f" ({p.doc_year})" if p.doc_year else ""),
                                     {"public": p.is_public, "regular": regular, "report_type": p.source_type, "report_year": p.doc_year},
                                     p.text, p, "regex"))
            elif p.source_type in OFFICIAL_SOURCE_TYPES and statement:
                # a company page counts only for what it discloses about sustainability, not for being the company's page
                out.append(Candidate("reporting", "reporting", "voluntary_disclosure",
                                     "Sustainability information disclosed on the company website",
                                     {"public": True}, statement, p, "regex"))
        seen: set[tuple[str, str]] = set()
        assurance_llm_budget = 3  # the regex check covers the rest
        for p in passages:
            if p.source_type not in OFFICIAL_SOURCE_TYPES or p.identity_basis == PARENT_BOUNDARY:
                continue  # third-party pages cannot evidence the company's own reporting
            for fm in P.detect_frameworks(p.text):
                key = (fm.framework, fm.sentence[:80])
                if key in seen:
                    continue
                seen.add(key)
                out.append(Candidate("reporting", "reporting", "framework_reference",
                                     f"{'Reports in accordance with' if fm.explicit else 'References' if fm.code_reference or fm.content_index else 'Mentions'} {fm.framework}",
                                     {"framework": fm.framework, "explicit": fm.explicit, "code_reference": fm.code_reference,
                                      "content_index": fm.content_index, "source_id": p.source_id,
                                      "level5_eligible": fm.framework in rules["level5_frameworks"], "public": p.is_public},
                                     fm.sentence, p, "regex"))
            for am in P.detect_assurance(p.text):
                key = ("assurance", am.sentence[:80])
                if key in seen:
                    continue
                seen.add(key)
                facts = {"sustainability_assurance": am.covers_sustainability, "provider": am.provider,
                         "standard": am.standard, "assurance_level": am.level, "third_party": am.third_party}
                method, model, conf, quote, notes = "regex", None, None, am.sentence, []
                data = None
                # the model can only confirm what the keyword check found (both must agree), so a statement the
                # check already refuses (a certified facility, verified credits) is not worth a model call
                if assurance_llm_budget > 0 and am.covers_sustainability:
                    assurance_llm_budget -= 1
                    data = self._ask(prompts.assurance_prompt(p.label, self._passage_text(p)), prompts.ASSURANCE_SCHEMA)
                if data is not None:
                    method, model = self._llm_meta()
                    conf = as_confidence(data.get("confidence"))
                    llm_says = bool(as_bool(data.get("sustainability_assurance")))
                    if llm_says != am.covers_sustainability:
                        notes.append(f"LLM ({llm_says}) and keyword check ({am.covers_sustainability}) disagree on sustainability scope.")
                    facts["sustainability_assurance"] = llm_says and am.covers_sustainability
                    facts["provider"] = am.provider or (data.get("assurance_provider") if data.get("assurance_provider") and P.contains_term(p.text, str(data.get("assurance_provider"))) else None)
                out.append(Candidate("reporting", "reporting", "assurance_statement",
                                     f"Assurance statement{' by ' + facts['provider'] if facts['provider'] else ''}",
                                     facts, quote, p, method, model, conf, notes=notes))
            for sent in P.detect_regulatory_reporting(p.text):
                key = ("reg", sent[:80])
                if key in seen:
                    continue
                seen.add(key)
                out.append(Candidate("reporting", "reporting", "regulatory_reporting",
                                     "Reference to statutory / regulatory sustainability reporting",
                                     {"terms": P.matched_terms(sent, rules["regulation_terms"])}, sent, p, "regex"))
        return out


# explicit forward-looking constructions (a noun like "ambition" alone does not make a sentence a goal)
FORWARD_RE = re.compile(r"\b(will|aims? to|aiming to|plans? to|intends? to|commit(?:s|ted)? to|strives? to|seeks? to|by 20\d\d)\b", re.IGNORECASE)


def is_goal_sentence(s: str, factor: dict[str, Any]) -> bool:
    return (len(s.split()) >= 6 and bool(STRONG_GOAL_RE.search(s)) and bool(P.matched_terms(s, factor["keywords"]))
            and not ACTIVITY_RE.search(s) and not gates.is_inactive_goal(s) and not gates.external_goal_only(s)
            and not gates.external_purpose_only(s)
            and not (P.CHANGE_VERB_RE.search(s) and not FORWARD_RE.search(s)))


def _year_in(year: int, text: str) -> bool:
    return re.search(rf"(?<!\d){year}(?!\d)", text) is not None


def _align_activity_quote(description: str, quote: str, text: str, notes: list[str]) -> tuple[str, bool]:
    """The finding (the model's description) must be what the quote says. If it is not, the verbatim sentence of
    the passage that does state it replaces the quote; failing that, the pair is marked as mismatched (rejected
    in verification). A nearby statement about something else is never used."""
    if not description or not quote or gates.claim_supported_by_quote(description, quote):
        return quote, False
    sentence = gates.supporting_sentence(description, text)
    if sentence and P.normalize_ws(sentence) in P.normalize_ws(text):
        notes.append("The extraction quote did not state the finding; the source sentence that does was used instead.")
        return sentence, False
    notes.append("The extraction quote does not state the reported finding.")
    return quote, True


_INVERTED_LABEL_RE = re.compile(r"\b(?:reduc(?:ed|tion)|avoided|diverted|diversion|saved|savings|recovered|prevented)\b", re.IGNORECASE)
_SHARE_UNITS_RE = re.compile(r"^(?:[kmgt]?wh|[kmgtp]?j|t|kt|mt|tco2e?|tonnes?)$", re.IGNORECASE)


_SHARE_LABEL_RE = re.compile(r"\b(?:share|percentage|proportion|percent)\b|%", re.IGNORECASE)
_INTENSITY_LABEL_RE = re.compile(r"\bper\b|\bintensity\b|/", re.IGNORECASE)
# part of a total, not the total: 'renewable energy consumption' is not 'energy consumption'
_COMPONENT_RE = re.compile(r"\b(?:renewable|non-renewable|fossil|nuclear|self-generated|purchased|on-site|by\s+source|from\s+(?:activities|sources))\b",
                           re.IGNORECASE)


def _is_share_metric(metric: dict[str, Any]) -> bool:
    return bool(re.search(r"\bshare\b|coverage", metric["name"], re.IGNORECASE))


def _row_metric(s: P.Series, factor: dict[str, Any]) -> dict[str, Any] | None:
    """The configured metric a table row measures, judged by what the row's own label MEANS, not by which keyword
    happens to be longest. A share is only a share metric; an absolute quantity is never one; a component of a total
    is not the total; a per-unit figure is an intensity. When the meaning does not fit a configured metric: abstain."""
    label = P.label_without_units(s.metric_label)
    metric = P.match_metric(label, factor)
    if metric is None:
        return None
    unit = (s.unit or "").strip().strip("()")
    metrics = factor.get("metrics", [])
    is_share = bool(_SHARE_LABEL_RE.search(s.metric_label)) or unit == "%"
    if is_share and not _is_share_metric(metric):
        renewable_share = next((m for m in metrics if _is_share_metric(m) and "renewable" in m["name"]), None)
        if re.search(r"\brenewable\b", label, re.IGNORECASE) and not re.search(r"\bnon-renewable\b", label, re.IGNORECASE):
            metric = renewable_share  # 'share of renewable sources in total energy consumption'
        elif re.search(r"consumption|generated|landfill|use\b", metric["name"], re.IGNORECASE):
            return None  # 'share of fossil sources in energy consumption' is not the consumption itself
        if metric is None:
            return None
    elif not is_share and _is_share_metric(metric):
        return None  # 'renewable energy production', 'MWh of renewable electricity': a quantity, not a share
    if _INTENSITY_LABEL_RE.search(label) and "intensity" not in metric["name"]:
        intensity = next((m for m in metrics if "intensity" in m["name"] and re.search(r"\w+", m["name"])
                          and m["name"].split()[0] in metric["name"]), None)
        if intensity is None:
            return None
        metric = intensity  # 'energy consumption per revenue' is an energy intensity
    if re.search(r"consumption", metric["name"], re.IGNORECASE) and _COMPONENT_RE.search(label):
        return None  # 'total fossil / renewable / purchased energy consumption' is a part, not the total
    if metric["better"] == "lower" and _INVERTED_LABEL_RE.search(label):
        return None  # 'waste diverted / reduced / avoided' rises when things improve: the direction is not the metric's
    if _is_share_metric(metric) and _SHARE_UNITS_RE.match(unit) and "%" not in s.metric_label:
        return None
    return metric


def _values_stated_with_years(text: str, factor: dict[str, Any]) -> bool:
    """Could a model-read series ever be accepted here? Verification needs values tied to their years, so a
    sentence must name a metric of this factor next to at least two years and two other numbers."""
    for sentence in P.split_sentences(text):
        years = set(P.find_years(sentence))
        others = [n for n in P.NUM_RE.findall(P.YEAR_RE.sub(" ", sentence))]
        if len(years) >= 2 and len(others) >= 2 and P.match_metric(sentence, factor):
            return True
    return False


def _deadline_has_passed(sentence: str, year: int) -> bool:
    """A goal whose only stated deadline lies before the assessment year is not a current goal."""
    deadlines = [int(m.group(1)) for m in P.TARGET_YEAR_RE.finditer(sentence)]
    return bool(deadlines) and max(deadlines) < year


def _goal_claim(label: str, f: dict[str, Any]) -> str:
    parts = [f"{label} goal"]
    unit = f.get("target_unit") or ""
    if f.get("target_value") == 0 and unit.isalpha():
        parts.append(f"target zero {unit}")
    elif f.get("target_value") is not None:
        parts.append(f"target {f['target_value']:g}{unit}")
    if f.get("target_year"):
        parts.append(f"by {f['target_year']}")
    if f.get("baseline_year"):
        parts.append(f"(baseline {f['baseline_year']})")
    return " ".join(parts)


def _activity_claim(label: str, f: dict[str, Any], quote: str) -> str:
    desc = f.get("activity_description") or quote[:140]
    year = f" ({f['effective_year']})" if f.get("effective_year") else ""
    return f"{label} activity{year}: {desc}"


def _favorable(sentence: str, better: str | None) -> bool | None:
    """Is the stated comparison favourable? 'below average' is good only if lower is better."""
    low = sentence.lower()
    if re.search(r"\b(worse than|behind|lags?|lagging)\b", low):
        return False
    if re.search(r"\b(better than|ahead of|outperform\w*|sector[- ]leading|industry[- ]leading|best[- ]in[- ]class|in line with|aligned with)\b", low):
        return True
    up = re.search(r"\b(above|higher than|exceed\w*)\b", low)
    down = re.search(r"\b(below|lower than|less than)\b", low)
    if better is None or bool(up) == bool(down):
        return None
    return (better == "higher") == bool(up)


FUTURE_RE = re.compile(r"\b(will|plan to|plans to|intend)\b", re.IGNORECASE)


def passage_signal(factor_key: str, dimension: str, text: str) -> int:
    """Cheap regex relevance signal used to re-rank BM25 candidates before extraction."""
    factor = get_factor(factor_key)
    sentences = P.split_sentences(text)
    if dimension == "planning":
        return sum(1 for s in sentences if is_goal_sentence(s, factor))
    if dimension == "execution":
        return sum(1 for s in sentences if ACTIVITY_RE.search(s) and P.matched_terms(s, factor["keywords"]) and not FUTURE_RE.search(s)
                   and not gates.activity_not_performed(s) and not gates.external_actor_only(s) and not gates.external_purpose_only(s))
    if dimension == "performance":
        series = P.extract_series(text) + P.extract_from_to_changes(text, None) + P.extract_relative_changes(text, None)
        return sum(2 for s in series if P.match_metric(s.metric_label + " " + s.quote, factor)) + (1 if len(set(P.find_years(text))) >= 2 else 0)
    return 0
