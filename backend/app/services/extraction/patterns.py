"""Conventional (non-LLM) extraction: years, percentages, frameworks, assurance, targets,
time series. The LLM is used for context interpretation, not for trivial pattern matching."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.config.loader import load_rules

YEAR_RE = re.compile(r"\b(20\d{2})\b")
PCT_RE = re.compile(r"(?<![\d.,])(-?\d+(?:[.,]\d+)?)\s?(?:%|per\s?cent\b|percent\b)", re.IGNORECASE)
# Thousands separators: comma or (narrow) no-break space. A plain space is NOT treated as a
# separator because PDF tables put space-separated cells on one line ("124 116 108").
NUM_PATTERN = r"-?\d{1,3}(?:[,\u00a0\u202f]\d{3})+(?:\.\d+)?|-?\d+(?:[.,]\d+)?"
NUM_RE = re.compile(rf"(?<![\w.])({NUM_PATTERN})(?![\w])")
UNIT_PATTERN = r"%|[A-Za-z€$£][A-Za-z0-9€$£²³₂]*(?:\s?/\s?[A-Za-z0-9²³]+|\s+per\s+[A-Za-z0-9²³]+)?"
UNIT_STOPWORDS = {"and", "in", "to", "from", "the", "a", "of", "vs", "versus", "compared", "with", "while", "which", "or", "for", "by", "at", "on"}

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\u2022-])|\n+")

TARGET_WORDS_RE = re.compile(
    r"\b(target(?:s|ed|ing)?|goals?|aims?|aiming|ambitions?|commit(?:s|ted|ment|ments)?|pledge[sd]?|"
    r"plans? to|will|strives?|strategy|strategic|objectives?|seeks? to|intends? to|roadmap|policy)\b",
    re.IGNORECASE,
)
FUTURE_WORDS_RE = re.compile(r"\b(will|aim|aims|aiming|target|targets|goal|goals|plan to|plans to|commit|committed|intend|intends|ambition)\b", re.IGNORECASE)
TARGET_YEAR_RE = re.compile(r"\b(?:by|until|before|through|in|towards?|no later than)\s+(?:the\s+end\s+of\s+|end[- ]of[- ]|year[- ]end\s+)?(20\d{2})\b", re.IGNORECASE)
BASELINE_RE = re.compile(
    r"(?:baseline|base[- ]year|compared\s+(?:to|with)|relative\s+to|versus|vs\.?|against)\s+(?:a\s+|the\s+|our\s+)?(?:baseline\s+(?:of\s+|year\s+)?)?(20\d{2})"
    r"|(20\d{2})\s*(?:baseline|base[- ]year|levels?\b)"
    r"|\bfrom\s+(?:a\s+|the\s+)?(20\d{2})\s+(?:baseline|base|level)",
    re.IGNORECASE,
)
ZERO_TARGET_RE = re.compile(r"\b(zero|net[- ]zero)\s+([a-z][a-z -]{2,40})", re.IGNORECASE)
CHANGE_VERB_RE = re.compile(
    r"\b(reduced|decreased|lowered|cut|fell|declined|dropped|went down|increased|improved|rose|grew|raised|went up)\b",
    re.IGNORECASE,
)
DOWN_VERBS = {"reduced", "decreased", "lowered", "cut", "fell", "declined", "dropped", "went down"}


def parse_number(s: str) -> float | None:
    s = s.strip().replace("\u2212", "-")
    if re.fullmatch(r"-?\d{1,3}(?:[ ,\u00a0\u202f]\d{3})+(?:\.\d+)?", s):
        return float(re.sub(r"[ ,\u00a0\u202f]", "", s))
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def find_years(text: str) -> list[int]:
    return [int(y) for y in YEAR_RE.findall(text)]


def find_percentages(text: str) -> list[float]:
    out = []
    for m in PCT_RE.finditer(text):
        v = parse_number(m.group(1))
        if v is not None:
            out.append(v)
    return out


def unwrap_lines(text: str) -> str:
    """Re-join sentences that PDF layout wrapped over several lines. A line is joined to the
    previous one when that line does not end a sentence and either is long (a wrapped body
    line) or the next line starts in lower case. Short lines (headings, table rows) stay."""
    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if out and stripped and out[-1].strip():
            prev = out[-1].rstrip()
            if prev[-1] not in ".!?:;|" and (len(prev) >= 60 or stripped[0].islower()):
                out[-1] = prev + " " + stripped
                continue
        out.append(stripped)
    return "\n".join(out)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(unwrap_lines(text)) if s and len(s.strip()) > 3]


def normalize_ws(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_term(text: str, term: str) -> bool:
    if term.startswith("\\b") or any(c in term for c in "[]()?*+"):
        return re.search(term, text, re.IGNORECASE) is not None
    # An acronym only counts in capitals ("LED" is not the verb "led"), unless the text has no capitals at all.
    if term.isupper() and len(term) <= 4 and any(c.isupper() for c in text):
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    t = term.lower()
    low = text.lower()
    if len(t) <= 4 and re.fullmatch(r"[a-z0-9]+", t):
        return re.search(rf"\b{re.escape(t)}\b", low) is not None
    # A whole word, inflections allowed: "labor" is not in "collaboration" or "laboratory", nor "wage" in "sewage".
    return re.search(rf"(?<![a-z0-9]){re.escape(t)}(?:s|es|d|ed|ing|ers?)?(?![a-z0-9])", low) is not None


def matched_terms(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms if contains_term(text, t)]


# ---------------------------------------------------------------- reporting signals

@dataclass
class FrameworkMention:
    framework: str
    sentence: str
    explicit: bool  # the company's reporting is prepared in accordance with / with reference to the framework
    code_reference: bool = False  # cites the framework's disclosure codes ("GRI 302-1", "ESRS E1-5"): a reference
    content_index: bool = False  # a content index / index of disclosure requirements for the framework: a reference


# A. Reporting-framework USE: the report, statement or disclosures are prepared / reported under the framework.
_FRAMEWORK_USE_RE = re.compile(
    r"\b(?:in\s+accordance\s+with|with\s+reference\s+to|prepared\s+(?:according\s+to|in\s+line\s+with|following|under|using)|"
    r"in\s+compliance\s+with|compl(?:y|ies)\s+with|report(?:s|ed)?\s+(?:according\s+to|in\s+line\s+with|under))\b",
    re.IGNORECASE,
)
# The auditor's description of its own work ("evaluate whether ... is prepared in accordance with ESRS") is about
# the auditor's procedures, not the company stating how it reports.
_AUDITOR_PROCEDURE_RE = re.compile(
    r"\b(?:evaluate\s+whether|obtain\s+an\s+understanding|by\s+inquir(?:y|ies)|our\s+(?:responsibility|procedures|conclusion|opinion)|"
    r"we\s+(?:have\s+)?(?:conducted|performed|planned|evaluated|examined)|nothing\s+has\s+come\s+to\s+our\s+attention|"
    r"auditor'?s?\s+(?:responsibility|report)|assurance\s+engagement|perform(?:ed)?\s+procedures)\b",
    re.IGNORECASE,
)
FRAMEWORK_USE_DISTANCE = 60  # characters between "in accordance with" and the framework's name
_REPORTING_OBJECT_RE = re.compile(r"\b(?:report(?:s|ed|ing)?|statements?|disclosures?|prepared)\b", re.IGNORECASE)
# B. Mere references: a metric definition, disclosure codes, an index, last year's report, someone else's reporting.
_DISCLOSURE_CODE_RE = re.compile(
    r"\bGRI\s*(?:[Dd]isclosures?\s*)?\d{3}(?:-\d+[a-z]?)?\b|\bGRI\s+\d{1,3}\s*:\s*20\d{2}\b"
    r"|\bESRS\s*(?:[ESG]\d{1,2}(?:-\d+)?|[12])\b|\b[A-Z]{2}-[A-Z]{2}-\d{3}[a-z]\.\d+\b",
)
_CONTENT_INDEX_RE = re.compile(r"\bcontent\s+index\b|\bindex\s+of\b[^.]{0,40}\bdisclosure", re.IGNORECASE)
_HISTORICAL_RE = re.compile(r"\b(?:last|previous|prior)\s+year(?:'s)?\b|\bpreviously\b|\b(?:was|were)\s+based\s+on\b|\bformerly\b", re.IGNORECASE)
_OTHERS_REPORTING_RE = re.compile(r"\b(?:suppliers?|participants?|investees?|portfolio\s+companies|customers?|clients?)\b", re.IGNORECASE)


def detect_frameworks(text: str) -> list[FrameworkMention]:
    """explicit = the company states that its reporting follows the framework (A). Disclosure codes, metric
    definitions, content indexes, last year's practice and others' reporting are references (B): they never make a
    mention explicit, though several of them together can show adoption (see scoring.reporting)."""
    rules = load_rules()["reporting_rules"]
    out: list[FrameworkMention] = []
    for sentence in split_sentences(text):
        for name, patterns in rules["frameworks"].items():
            hits = [m for pat in patterns for m in re.finditer(pat if pat.startswith("\\b") else re.escape(pat), sentence, re.IGNORECASE)]
            if not hits:
                continue
            # the use phrase must govern this framework: "prepared in accordance with [the] ESRS", not "reported under fuel"
            governed = any(0 <= h.start() - u.end() <= FRAMEWORK_USE_DISTANCE for u in _FRAMEWORK_USE_RE.finditer(sentence) for h in hits)
            explicit = (governed and bool(_REPORTING_OBJECT_RE.search(sentence)) and not _AUDITOR_PROCEDURE_RE.search(sentence)
                        and not _HISTORICAL_RE.search(sentence) and not _OTHERS_REPORTING_RE.search(sentence))
            out.append(FrameworkMention(name, sentence, explicit, bool(_DISCLOSURE_CODE_RE.search(sentence)),
                                        bool(_CONTENT_INDEX_RE.search(sentence))))
    return out


@dataclass
class AssuranceMention:
    sentence: str
    provider: str | None
    standard: str | None
    covers_sustainability: bool
    level: str | None  # limited | reasonable | None
    third_party: bool = False  # performed by an independent / third-party assurance provider (named or not)


def detect_assurance(text: str) -> list[AssuranceMention]:
    rules = load_rules()["reporting_rules"]
    out: list[AssuranceMention] = []
    sentences = split_sentences(text)
    providers_in_text = [p for p in rules["assurance_providers"] if contains_term(text, p)]
    for s in sentences:
        low = s.lower()
        if len(s.split()) < 6 or not any(p.lower() in low for p in rules["assurance_patterns"]):
            continue  # headings such as "Independent assurance" are not statements
        provider = next((p for p in rules["assurance_providers"] if contains_term(s, p)), None)
        if provider is None and providers_in_text:
            provider = providers_in_text[0]
        standard = None
        for std in ("ISAE 3000", "ISAE3000", "AA1000", "ISAE 3410", "ISSA 5000"):
            if std.lower() in low:
                standard = std
                break
        # a table row or footnote running on for many words says nothing reliable about what was assured
        scope = low if len(s.split()) <= MAX_ASSURANCE_SENTENCE_WORDS else ""
        # Level 6 needs assurance of the sustainability report / statement / disclosures / metrics themselves.
        # A certification scheme, the ESEF/XBRL format of the annual report, the financial statements, or the
        # verification of a product, site, project, asset or credit is assurance of something else.
        covers = (bool(_ASSURANCE_SCOPE_RE.search(scope)) and not _ASSURANCE_NOT_DISCLOSURE_RE.search(scope)
                  and not _ASSURANCE_NEGATED_RE.search(scope))
        if covers and _ASSURANCE_OTHER_OBJECT_RE.search(scope) and not _SUSTAINABILITY_REPORT_SCOPE_RE.search(scope):
            covers = False
        level = "reasonable" if "reasonable assurance" in low else "limited" if "limited assurance" in low else None
        third_party = bool(provider) or bool(_THIRD_PARTY_RE.search(scope))
        out.append(AssuranceMention(s, provider, standard, covers, level, third_party))
    return out


# What a sustainability assurance statement is about: the sustainability report / statement, or its data and metrics.
MAX_ASSURANCE_SENTENCE_WORDS = 60
_SUSTAINABILITY_REPORT_SCOPE_RE = re.compile(
    r"\b(?:statutory\s+)?(?:sustainability|esg|non-financial|csr|climate|corporate\s+responsibility)\s+"
    r"(?:report(?:ing)?|statements?|information|disclosures?|data|metrics|kpis|indicators)\b"
)
_ASSURANCE_SCOPE_RE = re.compile(
    _SUSTAINABILITY_REPORT_SCOPE_RE.pattern
    + r"|\b(?:ghg|greenhouse\s+gas|emissions?)\s+(?:emissions\s+)?(?:data|inventory|statement|figures|metrics)\b"
    r"|\b(?:selected|reported|key|these|the|our|\d{4})?\s*(?:metrics|kpis|indicators)\b"
)
# says that something was NOT assured: "has not been subject to limited assurance", "no external assurance"
_ASSURANCE_NEGATED_RE = re.compile(
    r"\b(?:not|never)\s+(?:been\s+|yet\s+been\s+)?(?:subject(?:ed)?\s+to|covered\s+by|assured|verified|reviewed|audited)\b"
    r"|\b(?:has|have|had|was|were|is|are)\s+not\b[^.;]{0,40}\bassur|\bno\s+(?:independent\s+|external\s+|third[- ]party\s+)?"
    r"(?:assurance|verification)\b|\bwithout\s+(?:independent\s+|external\s+)?(?:assurance|verification)\b|\bunassured\b"
)
# never the sustainability disclosures, whatever else the sentence says
_ASSURANCE_NOT_DISCLOSURE_RE = re.compile(
    r"\besef\b|\bi?xbrl\b|\bdigital(?:ly)?\s+(?:format|tag\w*)|\bformat\s+that\b|\bcertif(?:y|ies|ied|ication|icate)\w*"
    r"|\bassurance\s+(?:framework|scheme|standard\s+for\s+(?:mines|producers))\b"
)
# objects that are not the company's disclosures; they count only when the sustainability report itself is also named
_ASSURANCE_OTHER_OBJECT_RE = re.compile(
    r"\bproduction\b|\bproducts?\b|facilit(?:y|ies)|\bsites?\b|data\s+cent(?:er|re)s?|\bplants?\b|\bbuildings?\b|\bassets?\b"
    r"|\bprojects?\b|\bcredits?\b|\boffsets?\b|\bgreen\s+(?:bond|financ\w+)|\bfinancial\s+statements\b|\bstatutory\s+accounts\b"
    r"|\bconsolidated\s+(?:financial|accounts)\b|\bannual\s+accounts\b"
)
_THIRD_PARTY_RE = re.compile(r"third[- ]party|independent|external(?:ly)?\s+(?:auditor|assur|verif)|\bauditors?\b|assurance\s+provider|accredited")


def detect_regulatory_reporting(text: str) -> list[str]:
    terms = load_rules()["reporting_rules"]["regulation_terms"]
    return [s for s in split_sentences(text) if matched_terms(s, terms)]


# Reporting evidence must DISCLOSE sustainability information, not merely mention sustainability. Being on the
# company's own website is not enough, and neither is a sustainability adjective in marketing text: "our products
# are affordable, sustainable and offer striking results" discloses nothing. A sentence qualifies when it:
#   (a) names a sustainability reporting artefact (sustainability / ESG / non-financial report or statement ...),
#   (b) says the company reports, discloses or publishes a sustainability subject (emissions, ESG data, ...),
#   (c) refers to sustainability reporting under a reporting framework or assurance, or
#   (d) states a quantified sustainability metric for a period ("Scope 1 emissions fell 12% compared with 2024").
_ESG_NOUN = (r"(?:sustainability|esg|climate|greenhouse\s+gas(?:es)?|ghg|(?:co2|carbon)\s+emissions|emissions|carbon\s+footprint|"
             r"scope\s+[123]|non-financial|csr|corporate\s+(?:social\s+)?responsibility|human\s+rights|diversity|"
             r"environmental\s+(?:impact|performance|data|information|footprint)|social\s+(?:impact|performance|data|information)|"
             r"energy\s+(?:use|consumption)|water\s+(?:use|consumption|withdrawal)|waste\s+(?:generated|data)|health\s+and\s+safety)")
_REPORT_ARTEFACT_RE = re.compile(
    r"\b(?:sustainability|esg|csr|non-financial|environmental|climate|impact|responsibility|corporate\s+responsibility|"
    r"integrated|annual\s+and\s+sustainability)\s+(?:report(?:s|ing)?|statements?|disclosures?|data|accounts|index|performance\s+report)\b",
    re.IGNORECASE,
)
_REPORT_VERB = r"(?:report(?:s|ed|ing)?|disclos(?:e|es|ed|ing|ures?)|publish(?:es|ed|ing)?|communicat(?:e|es|ed)|account(?:s|ed)?\s+for)"
_DISCLOSES_ESG_RE = re.compile(
    rf"\b{_REPORT_VERB}\b(?:\s+[\w-]+){{0,6}}?\s+(?:on\s+|about\s+)?(?:its\s+|our\s+|the\s+company's\s+)?(?:[\w-]+\s+)?{_ESG_NOUN}\b"
    rf"|\b{_ESG_NOUN}\b(?:\s+[\w-]+){{0,6}}?\s+(?:is|are|was|were|has\s+been|have\s+been)\s+(?:\w+\s+)?(?:reported|disclosed|published)\b",
    re.IGNORECASE,
)
_FRAMEWORK_OR_ASSURANCE_RE = re.compile(
    r"\b(?:gri|esrs|sasb|tcfd|issb|csrd|cdp)\b[^.]{0,80}\b(?:report\w*|disclos\w+|statement)\b"
    r"|\b(?:report\w*|disclos\w+|statement)\b[^.]{0,80}\b(?:gri|esrs|sasb|tcfd|issb|csrd)\b"
    r"|\b(?:limited|reasonable)\s+assurance\b[^.]{0,80}\b(?:sustainability|esg|non-financial|emissions?)\b",
    re.IGNORECASE,
)
_ESG_METRIC_RE = re.compile(
    r"\b(?:emissions?|ghg|greenhouse\s+gas|carbon\s+footprint|energy\s+(?:use|consumption|intensity)|renewable\s+(?:energy|electricity)|"
    r"water\s+(?:use|consumption|withdrawal)|waste|recycl\w+|women|gender|injur\w+|lost[- ]time|ltifr|employee\s+turnover)\b",
    re.IGNORECASE,
)
_ESG_FIGURE_RE = re.compile(r"\d(?:[\d.,\s]*)\s?(?:%|per\s?cent|percent|t\b|tonnes?|tco2e?|mwh|gwh|kwh)", re.IGNORECASE)
_MARKETING_RE = re.compile(
    r"\b(?:our|these|the)\s+(?:products?|solutions?|machines?|services?|offerings?)\s+(?:offer|provide|deliver|give|bring|are|is|help)\b"
    r"|\b(?:buyers?|marketers?|customers?)\s+who\b|\bdecision-making\s+process(?:es)?\s+of\b|\bstriking\s+results\b",
    re.IGNORECASE,
)
MIN_REPORTING_SENTENCE_WORDS = 6


def reporting_statement(text: str) -> str | None:
    """The first sentence that discloses sustainability information (rules (a)-(d) above). None otherwise."""
    for sentence in split_sentences(text):
        if len(sentence.split()) < MIN_REPORTING_SENTENCE_WORDS:
            continue
        if _REPORT_ARTEFACT_RE.search(sentence) or _DISCLOSES_ESG_RE.search(sentence) or _FRAMEWORK_OR_ASSURANCE_RE.search(sentence):
            return sentence
        quantified = (_ESG_METRIC_RE.search(sentence) and _ESG_FIGURE_RE.search(sentence)
                      and (YEAR_RE.search(sentence) or CHANGE_VERB_RE.search(sentence)))
        # a target says what the company wants to reach, not what it reports having reached
        if quantified and not _MARKETING_RE.search(sentence) and not TARGET_WORDS_RE.search(sentence):
            return sentence
    return None


# ---------------------------------------------------------------- targets (planning)

@dataclass
class TargetSignal:
    sentence: str
    quantitative: bool
    target_value: float | None
    target_unit: str | None
    target_year: int | None
    baseline_year: int | None


_TARGET_CLAUSE_SPLIT_RE = re.compile(r";|,\s*(?:which|leading|while|whereas|but|however|although)\b|\b(?:but|however|whereas)\b", re.IGNORECASE)
_STATISTIC_VERB_RE = re.compile(r"\b(?:represents?|representing|accounts?\s+for|accounting\s+for|according\s+to|surveyed|surveys?|study|studies|research|respondents?)\b", re.IGNORECASE)
_POPULATION_AFTER_VALUE_RE = re.compile(
    r"^\s*(?:%|per\s?cent|percent)?\s*of\s+(?:the\s+)?(?:all\s+|global\s+|world\s+|worldwide\s+|surveyed\s+)*"
    r"(?:organi[sz]ations|companies|businesses|firms|respondents|banks|banking|executives|consumers|people|assets|investors|ceos|leaders|participants|households|market)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_RE = re.compile(r"\b(?:we|our)\b", re.IGNORECASE)
_TARGET_LEAD_IN_RE = re.compile(
    r"(?:\b(?:by|to|of|reach|reaching|achieve|achieving|least|most|maximum|minimum|below|under|above|over|than|targets?|goals?|is|are|be|cut|reduce|maintain|keep)|[:=])\s*$",
    re.IGNORECASE,
)
_CHANGE_AFTER_VALUE_RE = re.compile(r"^\s*[A-Za-z]*\s*(?:reduction|decrease|increase|cut|lower|less|fewer|more)\b", re.IGNORECASE)


def target_clause(sentence: str, position: int) -> tuple[str, int]:
    """(the clause holding this position, the position within it)."""
    start = 0
    for m in _TARGET_CLAUSE_SPLIT_RE.finditer(sentence):
        if m.start() >= position:
            return sentence[start:m.start()], position - start
        start = m.end()
    return sentence[start:], position - start


def value_is_statistic(clause: str, start: int, end: int) -> bool:
    """'represents 40% of global assets', '86% of organisations': a measurement of someone else, not a target."""
    if _STATISTIC_VERB_RE.search(clause):
        return True
    return bool(_POPULATION_AFTER_VALUE_RE.match(clause[end:])) and not _FIRST_PERSON_RE.search(clause)


def value_is_tied_to_target(clause: str, start: int, end: int, is_percentage: bool) -> bool:
    """A figure that merely stands nearby ('With 5,000 employees, we aim to ...') is not the target."""
    if value_is_statistic(clause, start, end):
        return False
    return is_percentage or bool(_TARGET_LEAD_IN_RE.search(clause[:start]) or _CHANGE_AFTER_VALUE_RE.match(clause[end:]))


def detect_target(sentence: str, reference_year: int | None = None) -> TargetSignal | None:
    """Detect a goal/target statement in one sentence. Returns None if no goal language."""
    if not TARGET_WORDS_RE.search(sentence):
        return None
    target_year = None
    for m in TARGET_YEAR_RE.finditer(sentence):
        y = int(m.group(1))
        # "by 2025" in a 2024 report is a deadline; "in 2024" in a 2024 report is not
        is_in = m.group(0).lower().startswith("in")
        if reference_year is None or (y > reference_year if is_in else y >= reference_year - 1):
            target_year = y
            break
    baseline_year = None
    bm = BASELINE_RE.search(sentence)
    if bm:
        baseline_year = int(next(g for g in bm.groups() if g))
        if target_year == baseline_year:
            target_year = None
    value, unit = None, None
    pct = None
    for m in PCT_RE.finditer(sentence):
        clause, at = target_clause(sentence, m.start())
        if value_is_tied_to_target(clause, at, at + len(m.group(1)), True):
            pct = m
            break
    if pct:
        value, unit = parse_number(pct.group(1)), "%"
    else:
        z = ZERO_TARGET_RE.search(sentence)
        if z:
            value, unit = 0.0, z.group(2).strip().split()[0]  # "zero fatalities", "net-zero emissions"
        else:
            for m in re.finditer(rf"(?<![\w/.,])({NUM_PATTERN})\s*({UNIT_PATTERN})", sentence):
                n = parse_number(m.group(1))
                u = m.group(2)
                if n is None or YEAR_RE.fullmatch(m.group(1).strip()) or u.lower() in UNIT_STOPWORDS:
                    continue
                clause, at = target_clause(sentence, m.start())
                if not value_is_tied_to_target(clause, at, at + len(m.group(1)), False):
                    continue
                value, unit = n, u
                break
    return TargetSignal(sentence, value is not None, value, unit, target_year, baseline_year)


# ---------------------------------------------------------------- performance series

@dataclass
class Series:
    metric_label: str
    unit: str | None
    points: list[tuple[int, float]]
    quote: str
    kind: str = "series"  # series | relative_change
    change_pct: float | None = None
    direction_word: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = sorted(self.points)  # reports list years in either direction


def _clean_unit(u: str | None) -> str | None:
    if not u:
        return None
    u = u.strip().strip(".,;")
    return None if not u or u.lower() in UNIT_STOPWORDS else u


def _unit_from_label(label: str) -> str | None:
    """'Energy use (MWh)' -> 'MWh'. With several brackets, the one that is a unit: 'Renewable electricity (PPAs) (MWh)'."""
    groups = [g.strip() for g in re.findall(r"\(([^)]{1,25})\)", label)]
    return next((g for g in groups if is_unit_only(g)), groups[0] if groups else None)


def extract_series(text: str) -> list[Series]:
    return extract_series_with_rejections(text)[0]


def extract_series_with_rejections(text: str) -> tuple[list[Series], list[Series]]:
    """(clearly aligned series, rows refused because values cannot be tied to years).
    A false negative is preferred over a fabricated time series."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rejected: list[Series] = []
    series = _inline_year_value(text, rejected) + _from_to(text) + _tables(lines, rejected)
    return _dedupe(series), rejected


def _inline_year_value(text: str, rejected: list[Series]) -> list[Series]:
    out = []
    pair = re.compile(rf"\b(20\d{{2}})\s*[:=]\s*({NUM_PATTERN})\s*({UNIT_PATTERN})?")
    for sentence in split_sentences(text):
        pts, units = [], []
        for m in pair.finditer(sentence):
            v = parse_number(m.group(2))
            if v is not None:
                pts.append((int(m.group(1)), v))
                units.append(_clean_unit(m.group(3)))
        if len(pts) >= 2:
            first = pair.search(sentence)
            label = sentence[: first.start()].strip(" :-\u2013|") if first else sentence
            unit = next((u for u in units if u), None) or _unit_from_label(label)
            if len(dict(pts)) != len({y for y, _ in pts}) or len({y for y, _ in pts}) != len(pts):
                # one year with several values: the sentence holds more than one metric or scope
                rejected.append(Series(label or sentence[:80], unit, [], sentence, meta={"reason": "repeated_year"}))
                continue
            out.append(Series(label or sentence[:80], unit, pts, sentence))
    return out


def _from_to(text: str) -> list[Series]:
    out = []
    pat = re.compile(
        rf"(?P<label>[^.]*?)\bfrom\s+(?P<v1>{NUM_PATTERN})\s*(?P<u1>{UNIT_PATTERN})?\s+in\s+(?P<y1>20\d{{2}})\s+to\s+(?P<v2>{NUM_PATTERN})\s*(?P<u2>{UNIT_PATTERN})?\s+in\s+(?P<y2>20\d{{2}})",
        re.IGNORECASE,
    )
    for sentence in split_sentences(text):
        m = pat.search(sentence)
        if not m:
            continue
        v1, v2 = parse_number(m.group("v1")), parse_number(m.group("v2"))
        if v1 is None or v2 is None:
            continue
        unit = _clean_unit(m.group("u1")) or _clean_unit(m.group("u2"))
        out.append(Series(m.group("label").strip() or sentence[:80], unit, [(int(m.group("y1")), v1), (int(m.group("y2")), v2)], sentence))
    return out


def _is_year_line(line: str) -> list[int] | None:
    tokens = re.split(r"[\s|]+", line.strip())
    years = [t for t in tokens if re.fullmatch(r"(?:FY)?20\d{2}", t)]
    non_years = [t for t in tokens if t and t not in years]
    if years and len(non_years) <= 2 and all(not re.fullmatch(r"-?\d+(?:[.,]\d+)?", t) for t in non_years):
        return [int(y[-4:]) for y in years]
    return None


CELL_RE = re.compile(rf"\s*({NUM_PATTERN})\s*%?\s*")


def _numeric_cells(lines: list[str], start: int, n: int) -> tuple[list[float], int]:
    """Read up to n one-number-per-line cells from lines[start:]. A bare year ends the run
    (it is a header, not a value)."""
    cells: list[float] = []
    k = start
    while k < len(lines) and len(cells) < n:
        m = CELL_RE.fullmatch(lines[k])
        if not m or re.fullmatch(r"(?:19|20)\d{2}", m.group(1)):
            break
        v = parse_number(m.group(1))
        if v is None:
            break
        cells.append(v)
        k += 1
    return cells, k


def _has_text(line: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", line))


# Header words that head a column without numbers in it.
_TEXT_COLUMN_WORDS = {"unit", "units", "note", "notes", "gri", "source", "ref", "reference", "indicator", "metric", "kpi"}
_NON_ACTUAL_QUALIFIERS = {"target", "targets", "goal", "goals", "ambition", "plan", "budget", "forecast", "by"}
_CELL_TOKEN_RE = re.compile(rf"(?:{NUM_PATTERN})%?")
MAX_ROW_LABEL_WORDS = 12


def _header_columns(line: str) -> list[int | None]:
    """Numeric columns named by a year header. None = a column that is not an actual year's value
    ('2023 2024 Change', '2023 2024 Target 2030')."""
    tokens = [t for t in re.split(r"[ \t|]+", line.strip()) if t]
    positions = [i for i, t in enumerate(tokens) if re.fullmatch(r"(?:FY)?20\d{2}", t)]
    columns: list[int | None] = []
    for i in positions:
        qualified = i > 0 and tokens[i - 1].lower() in _NON_ACTUAL_QUALIFIERS
        columns.append(None if qualified else int(tokens[i][-4:]))
    columns += [None for t in tokens[positions[-1] + 1:] if t.lower() not in _TEXT_COLUMN_WORDS]
    return columns


def _looks_like_axis(values: list[float]) -> bool:
    """An evenly spaced run of round numbers (100, 75, 50, 25): a chart's scale. Reported figures are not."""
    if len(values) < 3:
        return False
    steps = {round(b - a, 6) for a, b in zip(values, values[1:])}
    return len(steps) == 1 and abs(next(iter(steps))) >= 5 and all(v % 5 == 0 for v in values)


def _clean_label(line: str) -> str:
    """'Purchased electricity209' -> 'Purchased electricity': a footnote number printed against the last word."""
    return re.sub(r"(?<=[A-Za-z)])\d{1,3}$", "", line.strip()).strip(" |:-–")


def _is_row_label(line: str) -> bool:
    """A table row's own name: words, no figures, not a unit, not a year header, not a sentence."""
    return (bool(line) and _has_text(line) and not NUM_RE.search(line) and not is_unit_only(line) and not _is_year_line(line)
            and not _is_prose(line) and len(line.split()) <= 2 * MAX_ROW_LABEL_WORDS)


def _is_prose(line: str) -> bool:
    return len(line.split()) >= 10 and line.rstrip()[-1:] in ".!?:"


def _row_cells(line: str, width: int) -> tuple[list[str] | None, str, str | None, int]:
    """(cell values if the row fills exactly `width` columns, label, unit printed in the cells, values seen)."""
    tokens = [t for t in re.split(r"[ \t|]+", line.strip()) if t]
    trailing = 0
    while trailing < len(tokens) and _CELL_TOKEN_RE.fullmatch(tokens[-1 - trailing]):
        trailing += 1
    if trailing == width:
        return [t.rstrip("%") for t in tokens[-width:]], " ".join(tokens[:-width]), None, trailing
    # '124 kWh 116 kWh': the same unit printed in every cell
    numeric = [i for i, t in enumerate(tokens) if _CELL_TOKEN_RE.fullmatch(t)]
    if len(numeric) >= width and trailing <= 1:
        cells = numeric[-width:]
        fillers = [tokens[a + 1:b] for a, b in zip(cells, cells[1:])] + [tokens[cells[-1] + 1:]]
        if all(len(f) == 1 for f in fillers) and len({f[0].lower() for f in fillers}) == 1:
            return [tokens[i].rstrip("%") for i in cells], " ".join(tokens[:cells[0]]), fillers[0][0], width
    return None, " ".join(tokens[:len(tokens) - trailing]), None, trailing


def _tables(lines: list[str], rejected: list[Series] | None = None) -> list[Series]:
    """Handle 'Indicator 2023 2024 2025' headers followed by 'Label 124 116 108' rows, including PyMuPDF's
    one-cell-per-line layout and labels printed above the year header. A row is only read when its cells
    fill the header's columns exactly; anything else cannot be aligned with the years."""
    rejected = [] if rejected is None else rejected
    out: list[Series] = []

    def points(columns: list[int | None], values: list[float]) -> list[tuple[int, float]]:
        return [(y, v) for y, v in zip(columns, values) if y is not None]

    i = 0
    while i < len(lines):
        years = _is_year_line(lines[i])
        j = i + 1
        if years is not None and len(years) == 1:  # one-year-per-line header
            while j < len(lines) and (y := _is_year_line(lines[j])) and len(y) == 1:
                years += y
                j += 1
        if not years or len(years) < 2 or len(set(years)) != len(years):
            i += 1
            continue
        columns: list[int | None] = list(years) if j > i + 1 else _header_columns(lines[i])
        n = len(columns)
        if sum(c is not None for c in columns) < 2:
            i = j
            continue
        k = j
        rows_seen = 0
        # Label printed above a one-year-per-line header, values below it: the label precedes the header
        cells, after = _numeric_cells(lines, k, n)
        if len(cells) == n and _looks_like_axis(cells):
            # '100 / 75 / 50 / 25' under a chart's year labels: the figure's scale, not reported values
            rejected.append(Series(lines[i - 1] if i > 0 else "", None, [], " ".join(lines[max(0, i - 1):after]), meta={"reason": "chart_axis"}))
            i = after
            continue
        if len(cells) == n and i > 0 and _has_text(lines[i - 1]) and not NUM_RE.search(lines[i - 1]):
            out.append(Series(lines[i - 1], _unit_from_label(lines[i - 1]), points(columns, cells),
                              " ".join(lines[i - 1:after])))
            rows_seen += 1
            k = after
        while k < len(lines) and rows_seen < 25:
            line = lines[k]
            if _is_year_line(line) or _is_prose(line):
                break  # the header above says nothing about what follows
            nums = [m.group(1) for m in NUM_RE.finditer(line)]
            label_part = NUM_RE.sub("", line).strip(" |:-–")
            if _has_text(label_part) and nums:
                raw, label, cell_unit, seen = _row_cells(line, n)
                label = label.strip(" |:-–")
                if raw is not None and is_unit_only(label):
                    # 'Renewable electricity procured' on one line, 'MWh 14,313,200 16,565,400 ...' on the next
                    above = _clean_label(lines[k - 1]) if k - 1 >= j else ""
                    if _is_row_label(above):
                        label, line = f"{above} ({label})", f"{above} {line}"
                    else:
                        rejected.append(Series(label, label, [], line, meta={"reason": "metric_label_missing"}))
                        k += 1
                        continue
                if raw is not None and _has_text(label) and len(label.split()) <= 3 * MAX_ROW_LABEL_WORDS:
                    vals = [parse_number(x) for x in raw]
                    if all(v is not None for v in vals) and _looks_like_axis(vals):
                        rejected.append(Series(label, None, [], line, meta={"reason": "chart_axis"}))
                    elif all(v is not None for v in vals) and not all(re.fullmatch(r"(?:19|20)\d{2}", x) for x in raw):
                        out.append(Series(label, _unit_from_label(label) or cell_unit, points(columns, vals), line))
                        rows_seen += 1
                    k += 1
                    continue
                if seen >= 2 and _has_text(label) and len(label.split()) <= MAX_ROW_LABEL_WORDS:
                    rejected.append(Series(label, None, [], line, meta={"reason": f"{seen}_values_for_{n}_columns"}))
                    k += 1
                    continue
            if _has_text(line) and not nums:
                cells, after = _numeric_cells(lines, k + 1, n)
                if len(cells) == n and _looks_like_axis(cells):
                    # '100 / 75 / 50 / 25' under a chart legend: the scale of a figure, not reported values
                    rejected.append(Series(_clean_label(line), None, [], " ".join(lines[k:after]), meta={"reason": "chart_axis"}))
                    k = after
                    continue
                if len(cells) == n:
                    label, quote = _clean_label(line), " ".join(lines[k:after])
                    if is_unit_only(label):
                        # one cell per line: 'Purchased electricity' / 'MWh' / five values. The unit line belongs to
                        # the label line right above it; without one, nobody can say what the values measure.
                        above = _clean_label(lines[k - 1]) if k - 1 >= j else ""
                        if not _is_row_label(above):
                            rejected.append(Series(label, label, [], quote, meta={"reason": "metric_label_missing"}))
                            k = after
                            continue
                        label, quote = f"{above} ({label})", f"{above} {quote}"
                    out.append(Series(label, _unit_from_label(label), points(columns, cells), quote))
                    rows_seen += 1
                    k = after
                    continue
                if after < len(lines) and _is_year_line(lines[after]):
                    break  # label of the next table
            k += 1
        i = max(k, i + 1)
    return out


FROM_TO_RE = re.compile(
    rf"(?P<label>[^.]*?)\b(?P<verb>{CHANGE_VERB_RE.pattern[3:-3]})\s+from\s+(?P<v1>{NUM_PATTERN})\s*(?P<u1>%|per\s?cent)?\s+to\s+(?P<v2>{NUM_PATTERN})\s*(?P<u2>%|per\s?cent|{UNIT_PATTERN})?",
    re.IGNORECASE,
)


def extract_from_to_changes(text: str, reference_year: int | None) -> list[Series]:
    """'The share of renewable energy increased from 60 to 64 per cent year over year' or
    '... between 2023 and 2024'. Years come from the sentence, or the reporting year."""
    out = []
    for sentence in split_sentences(text):
        if FUTURE_WORDS_RE.search(sentence):
            continue
        m = FROM_TO_RE.search(sentence)
        if not m:
            continue
        v1, v2 = parse_number(m.group("v1")), parse_number(m.group("v2"))
        if v1 is None or v2 is None or YEAR_RE.fullmatch(m.group("v1")) or YEAR_RE.fullmatch(m.group("v2")):
            continue
        between = re.search(r"between\s+(20\d{2})\s+and\s+(20\d{2})", sentence, re.I)
        compared = re.search(r"(?:compared\s+(?:to|with)|since|vs\.?)\s+(20\d{2})", sentence, re.I)
        if between:
            y1, y2 = int(between.group(1)), int(between.group(2))
        elif compared and reference_year:
            y1, y2 = int(compared.group(1)), reference_year
        elif re.search(r"year[- ]over[- ]year|year[- ]on[- ]year|previous year|prior year", sentence, re.I) and reference_year:
            y1, y2 = reference_year - 1, reference_year
        else:
            continue
        unit = "%" if (m.group("u1") or m.group("u2") or "").lower().replace(" ", "") in ("%", "percent") else _clean_unit(m.group("u2"))
        out.append(Series(m.group("label").strip() or sentence[:80], unit, [(y1, v1), (y2, v2)], sentence))
    return out


def extract_relative_changes(text: str, reference_year: int | None = None) -> list[Series]:
    """'Energy intensity decreased by 13% compared with 2022' (past-tense, not a target)."""
    out = []
    pat = re.compile(
        rf"(?P<label>[^.]*?)\b(?P<verb>{CHANGE_VERB_RE.pattern[3:-3]})\b[^.%]{{0,60}}?\bby\s+(?P<pct>{NUM_PATTERN})\s?(?:%|per\s?cent|percent)",
        re.IGNORECASE,
    )
    for sentence in split_sentences(text):
        if FUTURE_WORDS_RE.search(sentence):
            continue
        m = pat.search(sentence)
        if not m:
            continue
        pct = parse_number(m.group("pct"))
        if pct is None:
            continue
        since = re.search(r"(?:compared\s+(?:to|with)|since|from|relative to|vs\.?|versus)\s+(?:the\s+)?(20\d{2})", sentence, re.IGNORECASE)
        base_year = int(since.group(1)) if since else None
        years = [y for y in find_years(sentence) if y != base_year]
        end_year = max(years) if years else reference_year
        if base_year is None:
            continue  # a change without a comparison period is not time-bound evidence
        out.append(
            Series(
                m.group("label").strip() or sentence[:80], "%", [], sentence, kind="relative_change",
                change_pct=pct, direction_word=m.group("verb").lower(),
                meta={"base_year": base_year, "end_year": end_year},
            )
        )
    return out


def _dedupe(series: list[Series]) -> list[Series]:
    seen = set()
    out = []
    for s in series:
        key = (s.metric_label.lower()[:60], tuple(s.points))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


_UNIT_TOKEN_RE = re.compile(
    r"^(?:[kmgt]?wh|[kmgtp]?j|%|t|kt|mt|tco2e?|tco₂e?|tonnes?|tons?|m3|m³|kg|g|l|ml|units?|number|no\.?|#|fte|hours?|h|"
    r"msek|meur|musd|bsek|mnok|mdkk|ktco2e|mtco2e)$",
    re.IGNORECASE,
)
_CURRENCIES = {"sek", "eur", "usd", "nok", "dkk", "gbp", "chf", "jpy", "cny"}


def _is_unit_token(token: str) -> bool:
    t = token.strip("()[],:;").lower()
    return not t or t in _CURRENCIES or bool(_UNIT_TOKEN_RE.match(t))


def is_unit_only(label: str) -> bool:
    """'MWh', '(GWh)', '%', 'tCO2e': a unit says how a row is measured, never what it measures."""
    tokens = [t for t in re.split(r"[\s/|]+", label) if t]
    return bool(tokens) and all(_is_unit_token(t) for t in tokens)


def label_without_units(label: str) -> str:
    """The words of a row label that name the metric: units in brackets or on their own are dropped."""
    return " ".join(t for t in re.split(r"\s+", re.sub(r"\([^)]{0,25}\)", " ", label)) if t and not _is_unit_token(t))


def match_metric(label_and_context: str, factor: dict[str, Any]) -> dict[str, Any] | None:
    """Map a series label to a configured metric (with its 'better' direction)."""
    low = label_and_context.lower()
    best, best_len = None, 0
    for metric in factor.get("metrics", []):
        for kw in metric["keywords"]:
            if contains_term(low, kw) and len(kw) > best_len:
                best, best_len = metric, len(kw)
    return best
