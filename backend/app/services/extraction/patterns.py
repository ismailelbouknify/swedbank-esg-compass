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
    t = term.lower()
    low = text.lower()
    if len(t) <= 4 and re.fullmatch(r"[a-z0-9]+", t):
        return re.search(rf"\b{re.escape(t)}\b", low) is not None
    return t in low


def matched_terms(text: str, terms: list[str]) -> list[str]:
    return [t for t in terms if contains_term(text, t)]


# ---------------------------------------------------------------- reporting signals

@dataclass
class FrameworkMention:
    framework: str
    sentence: str
    explicit: bool  # "in accordance with GRI", "GRI content index", ...


def detect_frameworks(text: str) -> list[FrameworkMention]:
    rules = load_rules()["reporting_rules"]
    adherence = rules["framework_adherence_patterns"]
    out: list[FrameworkMention] = []
    for sentence in split_sentences(text):
        for name, patterns in rules["frameworks"].items():
            if any(re.search(p if p.startswith("\\b") else re.escape(p), sentence, re.IGNORECASE) for p in patterns):
                explicit = any(a in sentence.lower() for a in adherence)
                out.append(FrameworkMention(name, sentence, explicit))
    return out


@dataclass
class AssuranceMention:
    sentence: str
    provider: str | None
    standard: str | None
    covers_sustainability: bool
    level: str | None  # limited | reasonable | None


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
        covers = any(k in low for k in ("sustainability", "esg", "non-financial", "gri", "esrs", "csrd", "report", "kpi", "indicators", "sasb"))
        level = "reasonable" if "reasonable assurance" in low else "limited" if "limited assurance" in low else None
        out.append(AssuranceMention(s, provider, standard, covers, level))
    return out


def detect_regulatory_reporting(text: str) -> list[str]:
    terms = load_rules()["reporting_rules"]["regulation_terms"]
    return [s for s in split_sentences(text) if matched_terms(s, terms)]


# ---------------------------------------------------------------- targets (planning)

@dataclass
class TargetSignal:
    sentence: str
    quantitative: bool
    target_value: float | None
    target_unit: str | None
    target_year: int | None
    baseline_year: int | None


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
    pct = PCT_RE.search(sentence)
    if pct:
        value, unit = parse_number(pct.group(1)), "%"
    else:
        z = ZERO_TARGET_RE.search(sentence)
        if z:
            value, unit = 0.0, z.group(2).strip().split()[0]  # "zero fatalities", "net-zero emissions"
        else:
            for m in re.finditer(rf"({NUM_PATTERN})\s*({UNIT_PATTERN})", sentence):
                n = parse_number(m.group(1))
                u = m.group(2)
                if n is None or YEAR_RE.fullmatch(m.group(1).strip()) or u.lower() in UNIT_STOPWORDS:
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
    m = re.search(r"\(([^)]{1,25})\)", label)
    return m.group(1).strip() if m else None


def extract_series(text: str) -> list[Series]:
    series: list[Series] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    series += _inline_year_value(text)
    series += _from_to(text)
    series += _tables(lines)
    return _dedupe(series)


def _inline_year_value(text: str) -> list[Series]:
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


def _tables(lines: list[str]) -> list[Series]:
    """Handle 'Indicator 2023 2024 2025' headers followed by 'Label 124 116 108' rows,
    including PyMuPDF's one-cell-per-line layout and labels printed above the year header."""
    out: list[Series] = []
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
        n = len(years)
        k = j
        rows_seen = 0
        # Label printed above a one-year-per-line header, values below it: the label precedes the header
        cells, after = _numeric_cells(lines, k, n)
        if len(cells) == n and i > 0 and _has_text(lines[i - 1]) and not NUM_RE.search(lines[i - 1]):
            out.append(Series(lines[i - 1], _unit_from_label(lines[i - 1]), list(zip(years, cells)),
                              " ".join(lines[i - 1:after])))
            rows_seen += 1
            k = after
        while k < len(lines) and rows_seen < 25:
            line = lines[k]
            if _is_year_line(line):
                break
            nums = [m.group(1) for m in NUM_RE.finditer(line)]
            label_part = NUM_RE.sub("", line).strip(" |:-–")
            if len(nums) >= n and _has_text(label_part):
                vals = [parse_number(x) for x in nums[-n:]]
                if all(v is not None for v in vals) and not all(re.fullmatch(r"(?:19|20)\d{2}", x) for x in nums[-n:]):
                    out.append(Series(label_part, _unit_from_label(label_part), list(zip(years, vals)), line))
                    rows_seen += 1
                k += 1
                continue
            if _has_text(line) and not nums:
                cells, after = _numeric_cells(lines, k + 1, n)
                if len(cells) == n:
                    out.append(Series(line, _unit_from_label(line), list(zip(years, cells)), " ".join(lines[k:after])))
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


def match_metric(label_and_context: str, factor: dict[str, Any]) -> dict[str, Any] | None:
    """Map a series label to a configured metric (with its 'better' direction)."""
    low = label_and_context.lower()
    best, best_len = None, 0
    for metric in factor.get("metrics", []):
        for kw in metric["keywords"]:
            if contains_term(low, kw) and len(kw) > best_len:
                best, best_len = metric, len(kw)
    return best
