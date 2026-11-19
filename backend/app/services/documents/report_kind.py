"""Is a document the company's PRIMARY report: its company-wide annual, sustainability, ESG or integrated report?

Deterministic and conservative. A primary report needs a strong, company-wide report title. A product carbon
footprint report, LCA, EPD, brochure, position paper, policy, presentation, certificate, interim report or
"sustainability update" stays what it is, even when its file name contains "report" and it sits on the
company's own website. Only a primary report can make a "no evidence" conclusion honest (see coverage)."""
from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

ANNUAL_REPORT = "annual_report"
SUSTAINABILITY_REPORT = "sustainability_report"

# Company-wide annual disclosures. "Annual and Sustainability Report" counts as a sustainability report.
_SUSTAINABILITY_TITLE_RE = re.compile(
    r"\b(?:annual\s+and\s+sustainability|sustainability|esg|integrated|corporate\s+(?:social\s+)?responsibility|csr|"
    r"responsibility|environmental|impact|non-?financial)\s+(?:report|statement)\b"
    r"|\bsustainability\s+(?:and|&)\s+annual\s+report\b"
    r"|\bstatutory\s+sustainability\s+report\b",
    re.IGNORECASE,
)
_ANNUAL_TITLE_RE = re.compile(r"\bannual\s+report\b|\bannual\s+review\b|\bform\s+10-?k\b|\b20-?f\b", re.IGNORECASE)

# Documents that are never the company-wide primary report, whatever else their title says.
_NOT_PRIMARY_RE = re.compile(
    r"\bcarbon\s*footprint\b|\bproduct\s+(?:carbon|environmental|sustainability|footprint)\b|\blife[\s-]*cycle\s+assessment\b"
    r"|\blca\b|\benvironmental\s+product\s+declaration\b|\bepd\b|\bbrochures?\b|\bdata\s*sheets?\b|\bfact\s*sheets?\b"
    r"|\bposition\s+(?:paper|statement)\b|\bposition\s+on\b|\bpolicy\b|\bcode\s+of\s+conduct\b|\bpresentations?\b|\bslides?\b"
    r"|\bcertificates?\b|\bsustainability\s+update\b|\bstrategy\s+update\b|\bquarterly\b|\binterim\b|\bhalf[\s-]?year\b"
    r"|\bq[1-4]\b|\bpress\s+release\b|\bproxy\s+statement\b|\bmodern\s+slavery\b|\btax\s+(?:policy|report)\b"
    r"|\bgovernance\s+report\b|\bremuneration\s+report\b|\bcase\s+stud(?:y|ies)\b|\bwhite\s*paper\b",
    re.IGNORECASE,
)
TITLE_CHARS = 300  # cover page / first lines: where a report names itself
LEADING_CHARS = 2000


def _url_words(url: str | None) -> str:
    if not url:
        return ""
    try:
        path = unquote(urlparse(url).path)
    except ValueError:
        return ""
    # "acme_model_x_carbonfootprintreport.pdf" -> "acme model x carbon footprint report pdf"
    words = re.sub(r"[-_./+]+", " ", path)
    return re.sub(r"carbonfootprint", "carbon footprint", words, flags=re.IGNORECASE)


def _kind(text: str) -> str | None:
    if _SUSTAINABILITY_TITLE_RE.search(text):
        return SUSTAINABILITY_REPORT
    if _ANNUAL_TITLE_RE.search(text):
        return ANNUAL_REPORT
    return None


def primary_report_kind(title: str | None, leading_text: str = "", url: str | None = None, hint: str = "") -> str | None:
    """'sustainability_report', 'annual_report' or None (not a primary report).

    title: document title (metadata or page title); leading_text: start of the document; hint: text that
    pointed to the document (link label and its surrounding text on an official page)."""
    head = " ".join(p for p in (title or "", _url_words(url), hint) if p)
    cover = (leading_text or "")[:TITLE_CHARS]
    if _NOT_PRIMARY_RE.search(head) or _NOT_PRIMARY_RE.search(cover):
        # a strong title of its own can still say "Annual and Sustainability Report"; a URL or link label cannot
        named = _kind(title or "") or _kind(cover)
        if not named or _NOT_PRIMARY_RE.search(title or "") or _NOT_PRIMARY_RE.search(cover):
            return None
        return named
    # the document's own title and cover first; its address or the link that led to it only when they are silent
    return (_kind(title or "") or _kind(cover) or _kind(" ".join((_url_words(url), hint)))
            or _kind((leading_text or "")[:LEADING_CHARS]))


def is_primary_report_title(text: str | None) -> bool:
    return bool(text) and primary_report_kind(text) is not None
