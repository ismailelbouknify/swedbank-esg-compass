"""Turns retrieved passages into structured fact candidates.

LLM = "What facts exist in this passage?" -> structured JSON.
Python then normalises and cross-checks those facts (numbers must appear in the text,
benchmarks/owners must be explicitly worded, 'public' comes from source metadata).
If no local LLM is available, a deterministic keyword/regex extractor produces the same
structure with lower confidence and is labelled extraction_method='heuristic'."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.config.loader import get_factor, load_rules
from app.services.extraction import patterns as P
from app.services.extraction import prompts
from app.services.llm.base import LLMProvider, LLMUnavailableError
from app.services.llm.json_utils import LLMOutputError, as_bool, as_confidence, as_float, as_year, parse_llm_json

log = logging.getLogger(__name__)

ACTIVITY_RE = re.compile(
    r"\b(implemented|installed|introduced|launched|invested|rolled out|completed|conducted|upgraded|replaced|"
    r"carried out|established|deployed|trained|certified|organi[sz]ed|held|started|began|opened|renewed|signed|"
    r"performed|achieved|appointed|adopted|updated|expanded|piloted|retrofitted|converted|switched|audited|assessed)\b",
    re.IGNORECASE,
)
STRONG_GOAL_RE = re.compile(
    r"\b(targets?|goals?|aims?|aiming|ambitions?|commit(?:s|ted|ment|ments)?|pledge[sd]?|policy|policies|strategy|objectives?|roadmap|strives?)\b",
    re.IGNORECASE,
)


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


class FactExtractor:
    def __init__(self, llm: LLMProvider | None, assessment_year: int, allow_heuristic: bool = True,
                 on_llm_call: Callable[[], None] | None = None):
        self.llm = llm
        self.year = assessment_year
        self.allow_heuristic = allow_heuristic
        self.use_llm = bool(llm and llm.is_available())
        self.warnings: list[str] = []
        self.llm_calls = 0
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
        self.llm_calls += 1
        if self._on_llm_call:
            self._on_llm_call()
        try:
            raw = self.llm.complete_json(prompts.SYSTEM_PROMPT, prompt, schema=schema)
            return parse_llm_json(raw)
        except LLMUnavailableError as exc:
            self.use_llm = False
            self.warnings.append(f"Local LLM became unavailable during extraction ({exc}); remaining passages used heuristic extraction.")
            return None
        except LLMOutputError as exc:
            self.llm_failures += 1
            log.warning("Unparseable LLM output: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            self.llm_failures += 1
            log.warning("LLM call failed: %s", exc)
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
            data = self._ask(prompts.planning_prompt(factor["label"], p.label, self._passage_text(p)), prompts.PLANNING_SCHEMA)
            if data is not None:
                c = self._planning_from_llm(factor, p, data)
                if c:
                    out.append(c)
                continue
            if self.allow_heuristic:
                out.extend(self._planning_heuristic(factor, p, goal_sentences))
        return out

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

    def _planning_heuristic(self, factor: dict[str, Any], p: Passage, sentences: list[str]) -> list[Candidate]:
        out = []
        for s in sentences[:3]:
            t = P.detect_target(s, p.doc_year or self.year)
            if t is None:
                continue
            common = self._planning_common(factor, p, s)
            facts = {
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
            out.append(Candidate(
                factor["key"], "planning", "goal", _goal_claim(factor["label"], facts), facts, s, p, "heuristic", None,
                None, t.target_value, t.target_unit, t.target_year, t.baseline_year,
            ))
        return out

    # ------------------------------------------------------------------ execution
    def execution(self, factor_key: str, passages: list[Passage]) -> list[Candidate]:
        factor = get_factor(factor_key)
        link_terms = self.rules["execution_rules"]["goal_link_terms"]
        out: list[Candidate] = []
        for p in passages:
            sentences = [
                s for s in P.split_sentences(p.text)
                if ACTIVITY_RE.search(s) and P.matched_terms(s, factor["keywords"]) and not re.search(r"\b(will|plan to|plans to|intend)\b", s, re.I)
            ]
            if not sentences:
                continue
            data = self._ask(prompts.execution_prompt(factor["label"], p.label, self._passage_text(p)), prompts.EXECUTION_SCHEMA)
            if data is not None:
                if not as_bool(data.get("activity_found")):
                    continue
                quote = str(data.get("evidence_quote") or "").strip()
                linked = bool(as_bool(data.get("linked_to_goal")))
                notes = []
                quote_in_source = bool(quote) and P.normalize_ws(quote) in P.normalize_ws(p.text)
                if linked and not P.matched_terms(quote if quote_in_source else p.text, link_terms):
                    linked = False
                    notes.append("LLM linked the activity to a goal, but no goal/strategy wording was found; ignored.")
                year = as_year(data.get("activity_year"))
                facts = self._activity_facts(p, year, linked, data.get("activity_description"))
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
            for s in P.extract_series(p.text):
                metric = P.match_metric(s.metric_label, factor)
                if metric is None:
                    continue
                found_numeric = True
                out.append(self._series_candidate(factor, p, s, metric, "regex", None, None))
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
            if not found_numeric and P.matched_terms(p.text, factor["keywords"]) and len(P.find_years(p.text)) >= 2:
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
        return Candidate(factor["key"], "performance", "performance_series", f"{metric['name']} ({s.unit or 'unit n/a'}) \u2014 {pts}",
                         facts, s.quote, p, method, model, conf, extracted_unit=s.unit)

    # ------------------------------------------------------------------ reporting
    def reporting(self, passages: list[Passage], report_sources: list[Passage]) -> list[Candidate]:
        """passages: chunks to scan for frameworks/assurance/regulation (regex over all of them);
        report_sources: one representative passage per official/uploaded document."""
        rules = self.rules["reporting_rules"]
        out: list[Candidate] = []
        for p in report_sources:
            if p.source_type in ("sustainability_report", "annual_report", "uploaded_report"):
                regular = p.doc_year is not None or bool(re.search(r"\b(annual|yearly|every year|each year)\b", p.text, re.I))
                out.append(Candidate("reporting", "reporting", "report_published",
                                     f"Sustainability information published in '{p.source_title or 'report'}'"
                                     + (f" ({p.doc_year})" if p.doc_year else ""),
                                     {"public": p.is_public, "regular": regular, "report_type": p.source_type, "report_year": p.doc_year},
                                     p.text, p, "regex"))
            elif p.source_type == "company_website":
                out.append(Candidate("reporting", "reporting", "voluntary_disclosure",
                                     "Sustainability information on company website",
                                     {"public": True}, p.text, p, "regex"))
        seen: set[tuple[str, str]] = set()
        assurance_llm_budget = 3  # the regex check covers the rest
        for p in passages:
            if p.source_type not in ("uploaded_report", "sustainability_report", "annual_report", "company_website"):
                continue  # third-party pages cannot evidence the company's own reporting
            for fm in P.detect_frameworks(p.text):
                key = (fm.framework, fm.sentence[:80])
                if key in seen:
                    continue
                seen.add(key)
                out.append(Candidate("reporting", "reporting", "framework_reference",
                                     f"{'Reports in accordance with / references' if fm.explicit else 'Mentions'} {fm.framework}",
                                     {"framework": fm.framework, "explicit": fm.explicit,
                                      "level5_eligible": fm.framework in rules["level5_frameworks"], "public": p.is_public},
                                     fm.sentence, p, "regex"))
            for am in P.detect_assurance(p.text):
                key = ("assurance", am.sentence[:80])
                if key in seen:
                    continue
                seen.add(key)
                facts = {"sustainability_assurance": am.covers_sustainability, "provider": am.provider,
                         "standard": am.standard, "assurance_level": am.level}
                method, model, conf, quote, notes = "regex", None, None, am.sentence, []
                data = None
                if assurance_llm_budget > 0:
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
            and not ACTIVITY_RE.search(s)
            and not (P.CHANGE_VERB_RE.search(s) and not FORWARD_RE.search(s)))


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
        return sum(1 for s in sentences if ACTIVITY_RE.search(s) and P.matched_terms(s, factor["keywords"]) and not FUTURE_RE.search(s))
    if dimension == "performance":
        series = P.extract_series(text) + P.extract_from_to_changes(text, None) + P.extract_relative_changes(text, None)
        return sum(2 for s in series if P.match_metric(s.metric_label + " " + s.quote, factor)) + (1 if len(set(P.find_years(text))) >= 2 else 0)
    return 0
