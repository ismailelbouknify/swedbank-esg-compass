"""Source identity gate: does a processed document belong to, or explicitly concern, the assessed company?

Runs once per source before retrieval, using document metadata and text only (no model call). A source
that fails never reaches passage extraction or scoring."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import cached_property

from app.config.loader import load_rules
from app.services.company.identify import looks_official, normalize_company_name
from app.services.documents.url_safety import registrable_domain

SOURCE_COMPANY_MISMATCH = "source_company_mismatch"
MISMATCH = "mismatch"
PARENT_BOUNDARY = "parent_boundary"
# Tied to the company by its own web address or supplied website: passages inherit that identity.
FIRST_PARTY_BASES = {"official_domain", "name_domain"}
MAX_TEXT_CHARS = 400_000
LEADING_TEXT_CHARS = 3000
HEADER_CHARS = 300
FOLLOWER_SHARE = 0.6
MIN_FOLLOWER_MENTIONS = 3

# First words too common to identify a company on their own: "Nordic Steel" is not "Nordic Paper".
GENERIC_FIRST_TOKENS = {
    "nordic", "global", "international", "national", "first", "united", "general", "european", "american", "swedish",
    "svenska", "norsk", "dansk", "suomen", "new", "the", "green", "royal", "central", "north", "south", "east", "west",
}
# Words that may follow a company name without making it a different company.
GENERIC_FOLLOWERS = {
    "sustainability", "annual", "report", "esg", "group", "board", "ceo", "code", "climate", "environmental", "integrated",
    "impact", "responsibility", "corporate", "policy", "strategy", "statement", "review", "website", "and", "in", "the",
}
_TITLE_STOPWORDS = {
    "our", "the", "a", "an", "towards", "toward", "for", "of", "on", "and", "in", "to", "with", "group", "company",
    "microsoft", "word", "draft", "final", "full", "summary", "untitled",
}
_LEGAL_WORDS = {"ab", "as", "asa", "a", "s", "oy", "oyj", "ltd", "limited", "plc", "inc", "llc", "gmbh", "ag", "corp", "corporation", "publ", "co"}
_LEGAL_ENTITY_RE = re.compile(
    r"((?:[A-Z][\w&'-]*\s+){1,4}?(?:AB|ASA|AS|Oyj|Oy|Ltd\.?|Limited|PLC|plc|Inc\.?|LLC|GmbH|AG|S\.A\.|N\.V\.|Corp\.?|Corporation))(?![\w])"
)
_TITLE_COMPANY_RE = re.compile(
    r"^\s*(.+?)\s+(?:20\d{2}\s+)?(?:sustainability|annual|esg|csr|environmental|integrated|impact|responsibility|climate|non-financial)\b",
    re.IGNORECASE,
)
REPORT_DOCUMENT_RE = re.compile(
    r"\b(?:sustainability|esg|csr|environmental|responsibility|non-financial|impact|climate|integrated|annual)"
    r"(?:\s+and\s+[\w-]+)?\s+(?:report|statement|review)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_RE = re.compile(r"\b(?:we|our)\b", re.IGNORECASE)


def _phrase_pattern(tokens: list[str]) -> re.Pattern[str]:
    joined = r"[\s\-_.&,/+%20]*(?:and\s+)?".join(re.escape(t) for t in tokens)
    return re.compile(rf"(?<![\w]){joined}s?(?![\w])", re.IGNORECASE)


@dataclass(frozen=True)
class CompanyIdentity:
    name: str
    organisation_number: str | None = None
    aliases: tuple[str, ...] = ()  # names the resolution step established for this company
    excluded: tuple[str, ...] = ()  # other entities carrying a similar name

    @cached_property
    def normalized(self) -> str:
        return normalize_company_name(self.name)

    @cached_property
    def tokens(self) -> list[str]:
        return self.normalized.split()

    @cached_property
    def full_re(self) -> re.Pattern[str] | None:
        return _phrase_pattern(self.tokens) if self.tokens else None

    @cached_property
    def short_re(self) -> re.Pattern[str] | None:
        """The distinctive first word used alone ("Nordvik" for "Nordvik Components AB")."""
        if len(self.tokens) < 2 or len(self.tokens[0]) < 4 or self.tokens[0] in GENERIC_FIRST_TOKENS:
            return None
        return _phrase_pattern(self.tokens[:1])

    @cached_property
    def alias_res(self) -> list[re.Pattern[str]]:
        return [_phrase_pattern(a.split()) for a in self.aliases]

    @cached_property
    def name_words(self) -> set[str]:
        return set(self.tokens) | {w for a in self.aliases for w in a.split()}

    def is_named_in(self, text: str) -> bool:
        return bool(text and ((self.full_re and self.full_re.search(text)) or (self.short_re and self.short_re.search(text))
                              or any(r.search(text) for r in self.alias_res)))

    def org_number_in(self, text: str) -> bool:
        org = self.organisation_number
        if not org or len(org) < 6:
            return False
        pattern = r"[\s-]?".join(re.escape(ch) for ch in org)
        return re.search(rf"(?<![\w]){pattern}(?![\w])", text) is not None


@dataclass(frozen=True)
class IdentityResult:
    matched: bool
    basis: str  # organisation_number | official_domain | name_in_title | name_in_text | alias | analyst_upload | mismatch
    detail: str


def _other_company_mentions(identity: CompanyIdentity, title: str, text: str, company_mentions: int) -> tuple[str | None, int, int]:
    """Most-mentioned other company, its mention count, and the company mentions left after removing
    those that were really part of a longer, different name ("Acme Foods" when assessing "Acme")."""
    providers = {normalize_company_name(p) for p in load_rules()["reporting_rules"]["assurance_providers"]}
    candidates: set[str] = set()
    for match in _LEGAL_ENTITY_RE.finditer(text):
        norm = normalize_company_name(match.group(1))
        if norm and norm != identity.normalized and norm not in providers and not (identity.full_re and identity.full_re.search(norm)):
            candidates.add(norm)
    title_match = _TITLE_COMPANY_RE.match(title or "")
    if title_match:
        words = title_match.group(1).split()
        norm = normalize_company_name(title_match.group(1))
        if (norm and len(words) <= 4 and words[0][0].isupper() and not any(ch.isdigit() for ch in title_match.group(1))
                and not {w.lower() for w in words} & _TITLE_STOPWORDS and not identity.is_named_in(title_match.group(1))):
            candidates.add(norm)

    best_name, best_count = None, 0
    for norm in candidates:
        count = len(_phrase_pattern(norm.split()).findall(text)) + (1 if title_match and norm == normalize_company_name(title_match.group(1)) else 0)
        if count > best_count:
            best_name, best_count = norm, count

    # the company's name keeps being continued by the same word: that is another company's name
    base = identity.full_re if len(identity.tokens) == 1 else identity.short_re
    if base is not None:
        followers: Counter[str] = Counter()
        hits = 0
        for match in base.finditer(text):
            hits += 1
            following = re.match(r"\s+([A-Z][\w-]+)", text[match.end():match.end() + 40])
            if following and following.group(1).lower() not in GENERIC_FOLLOWERS and following.group(1).lower() not in identity.name_words:
                followers[following.group(1)] += 1
        if followers and hits >= MIN_FOLLOWER_MENTIONS:
            word, count = followers.most_common(1)[0]
            if count / hits >= FOLLOWER_SHARE:
                company_mentions = max(0, company_mentions - count)
                if count > best_count:
                    best_name, best_count = f"{identity.tokens[0]} {word.lower()}", count
    return best_name, best_count, company_mentions


def _domain_labels(url: str | None) -> set[str]:
    """The registrable name and, for sites such as blog.brand, the brand top-level domain."""
    if not url or "://" not in url:
        return set()
    domain = registrable_domain(url) or ""
    parts = domain.split(".")
    return {parts[0], parts[-1]} if len(parts) >= 2 and len(parts[-1]) > 3 else {parts[0]}


def _domain_is_company_name(identity: CompanyIdentity, url: str | None) -> bool:
    names = {"".join(identity.tokens)} | {a.replace(" ", "") for a in identity.aliases}
    return bool(_domain_labels(url) & names)


_BOUNDARY_TERMS = r"(?:subsidiar(?:y|ies)|reporting\s+boundary|scope\s+of\s+(?:this|the)\s+report|included\s+in\s+(?:this|the)\s+report|consolidated\s+entit)"


def _in_reporting_boundary(identity: CompanyIdentity, text: str) -> bool:
    """A group's report may cover the assessed company only if it says so, next to the company's full name."""
    if not identity.full_re:
        return False
    for match in identity.full_re.finditer(text):
        if re.search(_BOUNDARY_TERMS, text[max(0, match.start() - 150):match.end() + 150], re.IGNORECASE):
            return True
    return False


def _rejected(identity: CompanyIdentity, text: str, detail: str) -> IdentityResult:
    if len(identity.tokens) >= 2 and _in_reporting_boundary(identity, text):
        return IdentityResult(True, PARENT_BOUNDARY, f"{detail} It states that the assessed company is within its reporting boundary.")
    return IdentityResult(False, MISMATCH, detail)


def _other_entity_dominates(identity: CompanyIdentity, title: str, url: str | None, text: str, mentions: int) -> str | None:
    """Resolution found other entities with this name: a source about one of them is not about this company."""
    named_here = any(r.search(title) for r in identity.alias_res) or bool(identity.alias_res and _domain_is_company_name(identity, url))
    for entity in identity.excluded:
        words = entity.split()
        if "".join(words) in _domain_labels(url):
            return f"Web address belongs to '{entity}'."
        pattern = _phrase_pattern(words)
        count = len(pattern.findall(text))
        if not named_here and (pattern.search(title) or (count >= 2 and count >= max(mentions - count, 1))):
            return f"Document is about '{entity}' ({count} mentions)."
    return None


def _similar_name_in_title(identity: CompanyIdentity, title: str) -> str | None:
    """'Nordvik Global AS' or 'Volvo Group' as the name in a title when assessing a different company with
    the same first word. Words after the shared one must all belong to the assessed name."""
    if len(identity.tokens) < 2 or not title:
        return None
    for segment in re.split(r"\s*[|–—]\s*|\s+-\s+", title):
        words = re.findall(r"[^\W\d_]+", segment)
        if not 2 <= len(words) <= 4 or words[0].lower() != identity.tokens[0] or not all(w[0].isupper() for w in words):
            continue
        rest = [w.lower() for w in words[1:] if w.lower() not in _LEGAL_WORDS and (w.lower() == "group" or w.lower() not in GENERIC_FOLLOWERS)]
        if rest and not {r.rstrip("s") for r in rest} <= {t.rstrip("s") for t in identity.tokens}:
            return segment.strip()
    return None


def assess_source_identity(identity: CompanyIdentity, *, title: str | None, url: str | None, text: str, uploaded: bool = False,
                           on_official_domain: bool = False) -> IdentityResult:
    """on_official_domain must only be true for a domain the analyst supplied, never for a guessed one."""
    if not identity.tokens:
        return IdentityResult(True, "unverifiable", "Company name has no usable words to match.")
    text = text[:MAX_TEXT_CHARS]
    title = title or ""
    if identity.org_number_in(text):
        return IdentityResult(True, "organisation_number", "Organisation number found in the document.")

    full = len(identity.full_re.findall(text))
    short = len(identity.short_re.findall(text)) if identity.short_re else 0
    short = max(0, short - full)
    other, other_count, mentions = _other_company_mentions(identity, title, text, full + short)

    aliased = sum(len(r.findall(text)) for r in identity.alias_res)
    if on_official_domain:
        if mentions == 0 and other_count >= 3:
            return IdentityResult(False, MISMATCH, f"On the company domain, but names only '{other}'.")
        return IdentityResult(True, "official_domain", "Hosted on the company website given for this assessment.")
    entity_problem = _other_entity_dominates(identity, title, url, text, max(mentions, aliased))
    if entity_problem:
        return _rejected(identity, text, entity_problem)
    if identity.aliases and aliased == 0:
        return _rejected(identity, text, "Does not name the company this assessment was resolved to.")
    if _domain_is_company_name(identity, url) and (full >= 1 or aliased >= 1):
        return IdentityResult(True, "name_domain", "The web address is the company's exact name, and the document names the company.")
    if other_count >= 2 and other_count > max(mentions, aliased):
        return _rejected(identity, text, f"Document is about '{other}' ({other_count} mentions, assessed company {mentions}).")

    similar = _similar_name_in_title(identity, title)
    if similar and full < 2:
        return _rejected(identity, text, f"Document title names a similar but different company: '{similar}'.")
    # an address alone proves nothing: it must be backed by the document's own text
    if identity.full_re.search(title) or (full >= 1 and identity.full_re.search(url or "")) or any(r.search(title) for r in identity.alias_res):
        return IdentityResult(True, "name_in_title", "Company name found in the document title or address.")
    leading = text[:LEADING_TEXT_CHARS]
    if full >= 2 or identity.full_re.search(text[:HEADER_CHARS]):  # one mention further on is only in passing
        return IdentityResult(True, "name_in_text", f"Company name found {full} time(s) in the document.")
    if identity.short_re and short >= 2 and full >= 1 and (identity.is_named_in(title) or identity.is_named_in(url or "") or identity.short_re.search(leading)):
        return IdentityResult(True, "alias", f"Short company name found {short} time(s) in the document.")
    if uploaded:
        return IdentityResult(True, "analyst_upload", "Attached by the analyst; no other company is named in it.")
    return IdentityResult(False, MISMATCH, "Assessed company is not named, or only in passing.")


def looks_like_company_report(title: str | None, text: str) -> bool:
    """A published report of the company itself, judged from the document rather than from its web address."""
    leading = text[:LEADING_TEXT_CHARS]
    return bool(REPORT_DOCUMENT_RE.search(title or "") or REPORT_DOCUMENT_RE.search(leading)) and bool(_FIRST_PERSON_RE.search(text[:MAX_TEXT_CHARS]))


def on_supplied_domain(url: str | None, company_name: str, website: str | None, domain: str | None) -> bool:
    return bool(url and website and domain and looks_official(url, company_name, domain))
