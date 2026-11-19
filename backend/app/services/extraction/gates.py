"""Deterministic state and actor checks on evidence sentences. They run before the LLM call (to skip
passages with nothing usable) and after it (on the returned quote), and never call a model.

A sentence is only refused when every activity or goal in it is negated or belongs to someone else, so
"We audited five sites but did not complete the sixth" stays valid evidence."""
from __future__ import annotations

import difflib
import re
from collections.abc import Callable

from app.services.extraction import patterns as P

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

FUZZY_THRESHOLD = 0.9

# Forward-looking language required for valid planning evidence.
# "is to / are to" covers "strategy is to become", "goal is to achieve", etc.
FORWARD_PLANNING_RE = re.compile(
    r"\b(will|aims?\s+to|aiming\s+to|plans?\s+to|intends?\s+to|commit(?:s|ted)?\s+to|"
    r"strives?\s+to|seeks?\s+to|by\s+20\d\d|target\s+of|goal\s+of|ambition\s+to|"
    r"(?:is|are)\s+to\b)\b",
    re.IGNORECASE,
)

_GOAL_NOUN = r"(?:targets?|goals?|ambitions?|commitments?|strateg(?:y|ies)|plans?|polic(?:y|ies)|objectives?|pledges?|roadmaps?)"
_ENDED = r"(?:discontinued|abandoned|withdrawn|withdrew|retired|scrapped|dropped|cancell?ed|terminated|revoked|ceased|ended|removed)"
INACTIVE_GOAL_RE = re.compile(
    rf"\b{_GOAL_NOUN}\b[^.;]{{0,60}}\b{_ENDED}\b"
    rf"|\b{_ENDED}\b[^.;]{{0,40}}\b{_GOAL_NOUN}\b"
    rf"|\bno\s+longer\b[^.;]{{0,40}}\b{_GOAL_NOUN}\b"
    rf"|\b{_GOAL_NOUN}\b[^.;]{{0,40}}\bno\s+longer\b"
    rf"|\b(?:do|does|did|have|has|had)(?:\s+not|n't)\s+(?:yet\s+)?(?:have|set|adopt(?:ed)?|establish(?:ed)?|define[d]?|formali[sz]ed?)\b[^.;]{{0,40}}\b{_GOAL_NOUN}\b"
    rf"|\b(?:has|have|had)\s+no\b[^.;]{{0,30}}\b{_GOAL_NOUN}\b"
    rf"|\b(?:there\s+(?:is|are)\s+)?no\s+(?:formal\s+|specific\s+|such\s+|current\s+)*{_GOAL_NOUN}\b"
    rf"|\bwithout\s+(?:a\s+|any\s+)?(?:formal\s+)?{_GOAL_NOUN}\b"
    rf"|\b(?:did\s+not|didn't|failed\s+to)\s+(?:meet|achieve|reach)\b[^.;]{{0,40}}\b{_GOAL_NOUN}\b"
    rf"|\bmissed\b[^.;]{{0,30}}\b{_GOAL_NOUN}\b"
    rf"|\b(?:consider(?:s|ing)?|evaluat(?:es|ing)|explor(?:es|ing))\s+(?:whether\s+to\s+|setting\s+|adopting\s+|establishing\s+)",
    re.IGNORECASE,
)

_NEGATION_BEFORE_VERB_RE = re.compile(
    r"(?:\bnot\b(?!\s+(?:only|just|all|every))|n't\b|\bnever\b|\bno\s+longer\b|\byet\s+to\b|\bfailed\s+to\b|\bunable\s+to\b|\brather\s+than\b)"
    r"(?:\s+[\w-]+){0,5}\s*$",
    re.IGNORECASE,
)
_NOTHING_DONE_RE = re.compile(r"\bno\b(?:\s+[\w-]+){1,5}?\s+(?:was|were|has\s+been|have\s+been)\s*$", re.IGNORECASE)
_FUTURE_BEFORE_VERB_RE = re.compile(
    r"\b(?:will|shall|would|could|may|might|should|must|plans?\s+to|planned\s+to|intends?\s+to|aims?\s+to|going\s+to|needs?\s+to|(?<!continues\s)(?<!continue\s)to\s+be)\b"
    r"(?:\s+[\w-]+){0,4}\s*$",
    re.IGNORECASE,
)
_NEGATED_BASE_VERB_RE = re.compile(
    r"\b(?:did|does|do)(?:\s+not|n't)\s+(?:yet\s+)?(?:implement|install|introduce|launch|invest|roll\s+out|complete|conduct|upgrade|"
    r"replace|carry\s+out|establish|deploy|train|certify|organi[sz]e|hold|start|begin|open|renew|sign|perform|achieve|appoint|adopt|"
    r"update|expand|pilot|retrofit|convert|switch|audit|assess)\b",
    re.IGNORECASE,
)
_HALTED_RE = re.compile(r"\b(?:postponed|cancell?ed|suspended|on\s+hold|abandoned|discontinued|delayed)\b", re.IGNORECASE)
_CLAUSE_SPLIT_RE = re.compile(r"[,;:]|\b(?:but|however|although|though|whereas|while)\b", re.IGNORECASE)

_ACTOR = (
    r"(?:governments?|authorit(?:y|ies)|regulators?|supervisory\s+bod(?:y|ies)|ministr(?:y|ies)|parliaments?|"
    r"european\s+commission|european\s+union|eu|un|municipalit(?:y|ies)|the\s+state|customers?|clients?|consumers?|"
    r"suppliers?|competitors?|peers?|industry|sector|ngos?|banks|financial\s+institutions|market\s+participants|sector\s+participants|survey\s+respondents|respondents|associations?|alliances?|coalitions?|federations?|(?<!group\s)(?<!sister\s)(?:companies|organi[sz]ations|businesses|firms))"
)
_ADVERB_AUX = r"(?:\s+(?:also|then|recently|subsequently|have|has|had|[a-z]+ly))*"
_ACTIVE_ACTOR_RE = re.compile(rf"\b(?:(?:our|its|their)\s+)?{_ACTOR}(?!-){_ADVERB_AUX}\s*$", re.IGNORECASE)
_BY_AGENT_RE = re.compile(rf"^(?:\s+[\w-]+){{0,4}}?\s+by\s+(?:the\s+)?(?:[a-z]+\s+)?{_ACTOR}\b", re.IGNORECASE)
_RULE_SUBJECT_RE = re.compile(
    r"\b(?:laws?|regulations?|directives?|legislation|gdpr|nis2?|dora|csrd|esrs|eu\s+taxonomy)\b(?:\s+[\w-]+){0,3}?\s+"
    r"(?:was|were|has\s+been|have\s+been|is|are|entered|came)\s*$",
    re.IGNORECASE,
)
_OBLIGATION_RE = re.compile(
    r"\b(?:companies|organi[sz]ations|businesses|firms|undertakings|operators|institutions|banks|market\s+participants)\s+"
    r"(?:must|shall|should|need\s+to|have\s+to|are\s+required\s+to|are\s+obliged\s+to|are\s+expected\s+to)\b"
    r"|\brequirements?\s+for\s+(?:all\s+)?(?:companies|organi[sz]ations|businesses|firms|undertakings|operators|financial\s+institutions|institutions|banks)\b",
    re.IGNORECASE,
)
_EXTERNAL_GOAL_RE = re.compile(
    rf"\b{_ACTOR}'s\s+(?:[\w-]+\s+){{0,3}}?{_GOAL_NOUN}\b"
    rf"|\b(?:eu|national|government|governmental|federal|state|regional)\s+(?:[\w-]+\s+)?{_GOAL_NOUN}\b"
    rf"|\b{_ACTOR}\s+(?:also\s+)?(?:aims?|plans?|intends?|targets?|commits?|strives?|has\s+set|have\s+set|set)\b",
    re.IGNORECASE,
)
_COMPANY_MARKER_RE = re.compile(r"\b(?:we|our|us|the\s+company|the\s+group|the\s+organi[sz]ation)\b", re.IGNORECASE)
_POSSESSED_ACTOR_RE = re.compile(rf"\b(?:our|its)\s+{_ACTOR}\b", re.IGNORECASE)
# "Other Company AB installed ...": a named legal entity directly before the verb
_NAMED_ENTITY_RE = re.compile(
    r"((?:[A-Z][\w&'-]*\s+){1,4}?(?:AB|ASA|AS|Oyj|Oy|Ltd\.?|Limited|PLC|plc|Inc\.?|LLC|GmbH|AG|Corp\.?|Corporation))"
    + _ADVERB_AUX + r"\s*$"
)
EXTERNAL_SCOPE_RE = re.compile(
    r"\b(?:sector|industry|national|peer|peers|market|country|eu|regional)(?:'s)?\s+(?:average|median|benchmark|mean|total)\b"
    r"|^\s*(?:national|sector|industry|peer|market|eu)\b",
    re.IGNORECASE,
)

IsCompany = Callable[[str], bool]


def is_inactive_goal(sentence: str) -> bool:
    return bool(INACTIVE_GOAL_RE.search(sentence))


def has_no_active_goal(quote: str) -> bool:
    """True when the quote states a goal only to say it ended, is missing or is merely being considered."""
    sentences = P.split_sentences(quote) or [quote]
    goal_sentences = [s for s in sentences if STRONG_GOAL_RE.search(s)] or sentences
    return all(is_inactive_goal(s) for s in goal_sentences)


def _activity_is_performed(clause: str, verb: re.Match[str]) -> bool:
    before = clause[: verb.start()]
    return not (_NEGATION_BEFORE_VERB_RE.search(before) or _NOTHING_DONE_RE.search(before)
                or _FUTURE_BEFORE_VERB_RE.search(before) or _HALTED_RE.search(clause))


def _has_performed_activity(sentence: str) -> bool | None:
    """None when the sentence has no activity verb at all."""
    found = None
    for clause in _CLAUSE_SPLIT_RE.split(sentence):
        for verb in ACTIVITY_RE.finditer(clause):
            if _activity_is_performed(clause, verb):
                return True
            found = False
    if found is None and _NEGATED_BASE_VERB_RE.search(sentence):
        return False
    return found


def activity_not_performed(quote: str) -> bool:
    """True when the quote mentions activities and all of them are negated, future or halted."""
    outcomes = [_has_performed_activity(s) for s in (P.split_sentences(quote) or [quote])]
    outcomes = [o for o in outcomes if o is not None]
    return bool(outcomes) and not any(outcomes)


def _company_named(text: str, is_company: IsCompany | None) -> bool:
    without_possessed = _POSSESSED_ACTOR_RE.sub(" ", text)
    return bool(_COMPANY_MARKER_RE.search(without_possessed)) or bool(is_company and is_company(text))


def _action_is_external(sentence: str, verb: re.Match[str], is_company: IsCompany | None) -> bool:
    before = sentence[: verb.start()]
    named = _NAMED_ENTITY_RE.search(before)
    external = (_ACTIVE_ACTOR_RE.search(before) or _BY_AGENT_RE.match(sentence[verb.end():])
                or _RULE_SUBJECT_RE.search(before) or (named and not (is_company and is_company(named.group(1)))))
    if not external:
        return False
    if named and not _ACTIVE_ACTOR_RE.search(before):
        return True  # the entity itself is the subject, whatever else the sentence names
    return not _company_named(before, is_company)


def external_actor_only(quote: str, verb_re: re.Pattern[str] = ACTIVITY_RE, is_company: IsCompany | None = None) -> bool:
    """True when every action in the quote was carried out by someone other than the company.
    Passive sentences without an agent stay valid: in a company document they describe the company."""
    any_action = False
    for sentence in P.split_sentences(quote) or [quote]:
        if _OBLIGATION_RE.search(sentence) and not _company_named(sentence, is_company):
            any_action = True
            continue
        for verb in verb_re.finditer(sentence):
            any_action = True
            if not _action_is_external(sentence, verb, is_company):
                return False
    return any_action


def external_goal_only(quote: str, is_company: IsCompany | None = None) -> bool:
    """True when the goals in the quote belong to an external body and the company is not part of them."""
    sentences = [s for s in (P.split_sentences(quote) or [quote]) if STRONG_GOAL_RE.search(s) or _OBLIGATION_RE.search(s)]
    return bool(sentences) and all(
        (_EXTERNAL_GOAL_RE.search(s) or _OBLIGATION_RE.search(s)) and not _company_named(s, is_company) for s in sentences
    )


def locate_quote(quote: str, page_text: str) -> tuple[bool, str, str | None]:
    """Return (found, exact_source_text, note). Accepts whitespace/case differences and, at
    >=90% similarity, a lightly paraphrased quote (the exact source sentence is then shown)."""
    if not quote or len(quote.strip()) < 8:
        return False, quote, "Quote missing or too short."
    nq = P.normalize_ws(quote)
    npage = P.normalize_ws(page_text)
    if nq in npage:
        return True, quote, None
    best, best_ratio = None, 0.0
    candidates = P.split_sentences(page_text)
    # also consider windows of 2 consecutive sentences for multi-sentence quotes
    candidates += [a + " " + b for a, b in zip(candidates, candidates[1:])]
    for s in candidates:
        r = difflib.SequenceMatcher(None, nq, P.normalize_ws(s)).ratio()
        if r > best_ratio:
            best, best_ratio = s, r
    if best is not None and best_ratio >= FUZZY_THRESHOLD:
        return True, best, f"Quote matched source text at {best_ratio:.0%} similarity; exact source wording shown."
    return False, quote, f"Quote not found on the stated page (best similarity {best_ratio:.0%})."


def goal_quote_failure(quote: str, passage_text: str, is_company: IsCompany | None = None) -> str | None:
    """Why a quote the model gave for a goal it reported cannot serve as that goal's evidence, or None.

    Only a missing, unlocatable or wrong sentence counts. A quote that is itself an inactive or external
    goal is a substantive answer about the goal, not a quoting failure, so it returns None too."""
    if not quote or len(quote.strip()) < 8:
        return "missing"
    if not locate_quote(quote, passage_text)[0]:
        return "not_in_source"
    if has_no_active_goal(quote) or external_goal_only(quote, is_company):
        return None
    if not (STRONG_GOAL_RE.search(quote) and FORWARD_PLANNING_RE.search(quote)):
        return "not_a_goal_statement"
    return None


NUMERIC_SUBJECT_MISMATCH = "numeric_target_subject_mismatch"
NUMERIC_CONTEXT_INVALID = "numeric_target_context_invalid"
_TARGET_ACTION_RE = re.compile(
    r"\b(?:targets?|goals?|aims?|aiming|ambitions?|commit\w*|pledge\w*|objectives?|strives?|plans?\s+to|intends?\s+to|will|"
    r"reduc\w+|cut\w*|lower\w*|decreas\w+|increas\w+|rais\w+|reach\w*|achiev\w+|maintain\w*|keep\w*|by\s+20\d{2})\b|\bzero\b",
    re.IGNORECASE,
)


def numeric_target_issue(quote: str, value: float, value_forms: set[str], unit: str | None, year: int | None, factor_terms: list[str],
                         is_company: IsCompany | None = None) -> str | None:
    """Does this figure belong to the company's target? Checks the tuple (actor, metric, value, unit, deadline,
    target context) inside the quote itself; appearing somewhere on the page is not enough."""
    if value == 0:
        if not re.search(r"\b(?:zero|no)\b|net[- ]zero|(?<![\w.,])0\s?(?:%|per\s?cent|percent)", quote, re.IGNORECASE):
            return NUMERIC_CONTEXT_INVALID
        return None
    number = "|".join(re.escape(v) for v in sorted(value_forms, key=len, reverse=True))
    found = None
    for sentence in P.split_sentences(quote) or [quote]:
        for m in re.finditer(rf"(?<![\w/.,])(?:{number})(?![\w/]|[.,]\d)", sentence):
            clause, at = P.target_clause(sentence, m.start())
            end = at + len(m.group(0))
            is_percentage = bool(re.match(r"\s*(?:%|per\s?cent|percent)", clause[end:], re.IGNORECASE))
            problem = None
            if P.value_is_statistic(clause, at, end):
                problem = NUMERIC_SUBJECT_MISMATCH
            elif _action_of_someone_else(clause, is_company):
                problem = NUMERIC_SUBJECT_MISMATCH
            elif (not _TARGET_ACTION_RE.search(clause) or not P.value_is_tied_to_target(clause, at, end, is_percentage)
                    or not P.matched_terms(clause, factor_terms) or (str(unit or "").strip() == "%" and not is_percentage)):
                problem = NUMERIC_CONTEXT_INVALID
            if problem is None:
                found = None
                if year is not None and str(year) not in quote:
                    return NUMERIC_CONTEXT_INVALID
                return None
            found = found or problem
    return found or NUMERIC_CONTEXT_INVALID


def _action_of_someone_else(clause: str, is_company: IsCompany | None) -> bool:
    return bool(_EXTERNAL_GOAL_RE.search(clause) or _OBLIGATION_RE.search(clause)) and not _company_named(clause, is_company)


# ---------------------------------------------------------------- purpose: whose factor is being managed?
# A goal or activity whose stated purpose is someone else's benefit (customers, communities, grids, a market,
# policy makers) is not the company's management of its own factor, unless it is also tied to the company's own
# operations. "We aim to help customers cut energy use" is a customer benefit; "we aim to cut the energy use of
# our data centres" is energy management.
_EXTERNAL_PURPOSE_RE = re.compile(
    r"\bbring(?:s|ing)?\b[^.;]{0,60}\bto\s+market\b|\bcommerciali[sz]\w*|\bgo[- ]to[- ]market\b"
    r"|\bhelp(?:s|ed|ing)?\s+(?:our\s+|their\s+)?(?:customers?|clients?|consumers?|users?|communities|cities|utilities|grids?|"
    r"society|others|partners|businesses|organi[sz]ations|countries|governments?|the\s+world|people\s+everywhere)\b"
    r"|\benabl(?:e|es|ed|ing)\s+(?:our\s+)?(?:customers?|clients?|users?|others|communities|partners|cities|businesses|organi[sz]ations|countries|society)\b"
    r"|\bcommunities\s+everywhere\b|\bresilience\s+of\s+(?:the\s+)?grids?\b|\bsupport(?:s|ing)?\s+(?:the\s+)?grids?\b"
    r"|\bpolicy\s+(?:recommendations?|advocacy|makers?|engagement)\b|\bpolicymakers?\b|\badvoca(?:te|tes|ted|cy|ting)\b|\blobby\w*"
    r"|\binvest(?:s|ed|ing|ments?)?\s+in\s+(?:start-?ups?|ventures?|funds?|third[- ]party|external)\b",
    re.IGNORECASE,
)
_OWN_OPERATIONS_RE = re.compile(
    r"\bour\s+(?:own\s+)?(?:global\s+)?(?:operations?|operational|data\s+cent(?:er|re)s?|facilit(?:y|ies)|sites?|offices?|buildings?|"
    r"plants?|factor(?:y|ies)|campus(?:es)?|fleet|vehicles?|manufacturing|production|energy\s+(?:use|consumption)|"
    r"electricity\s+(?:use|consumption)|consumption|footprint|emissions|workforce|employees|staff|people|supply\s+chain|"
    r"products?|packaging|value\s+chain|stores?|warehouses?|business|company|group|organi[sz]ation|it\s+systems?|systems?|networks?)\b"
    r"|\bcompany[- ]wide\b|\bgroup[- ]wide\b|\binternal(?:ly)?\b|\bin[- ]house\b|\bown\s+(?:operations|use|consumption)\b",
    re.IGNORECASE,
)


def external_purpose_only(text: str) -> bool:
    """True when every sentence's stated purpose is an external benefit and none ties it to the company's own operations."""
    sentences = [s for s in (P.split_sentences(text) or [text]) if _EXTERNAL_PURPOSE_RE.search(s)]
    return bool(sentences) and not _OWN_OPERATIONS_RE.search(text)


# ---------------------------------------------------------------- finding / quote alignment
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "at", "from", "as", "our", "we", "its", "their",
    "this", "that", "these", "those", "is", "are", "was", "were", "be", "been", "has", "have", "had", "it", "which", "also",
    "all", "more", "than", "into", "up", "over", "under", "per", "new",
}
MIN_CLAIM_SUPPORT = 0.5


def _content_words(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9][a-z0-9-]+", P.normalize_ws(text)) if w not in _STOP and len(w) > 2}


def claim_supported_by_quote(claim: str, quote: str) -> bool:
    """Does the quote state the finding? Most of the finding's content words, and every number in it, must be in
    the quote. A different statement on the same page (same company, same topic, same year) does not support it."""
    words = _content_words(claim)
    if not words:
        return True
    numbers = set(re.findall(r"\d+(?:[.,]\d+)?", claim)) - set(re.findall(r"\d+(?:[.,]\d+)?", quote))
    if numbers:
        return False
    return len(words & _content_words(quote)) / len(words) >= MIN_CLAIM_SUPPORT


def supporting_sentence(claim: str, text: str) -> str | None:
    """The sentence of the retrieved text that states the finding, verbatim, if exactly one does best."""
    best, best_share = None, 0.0
    words = _content_words(claim)
    if not words:
        return None
    for sentence in P.split_sentences(text):
        if not claim_supported_by_quote(claim, sentence):
            continue
        share = len(words & _content_words(sentence)) / len(words)
        if share > best_share:
            best, best_share = sentence, share
    return best
