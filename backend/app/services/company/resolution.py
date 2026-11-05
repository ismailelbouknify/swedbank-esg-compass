"""Which company does the supplied name refer to? Decided from deterministic signals only, before sources are
read in depth, and kept for the rest of the assessment: "Brand X" must not silently mix Brand X Cars and Brand X Group."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from app.services.company.source_identity import CompanyIdentity
from app.services.documents.url_safety import registrable_domain

RESOLVED = "resolved"
AMBIGUOUS = "company_identity_ambiguous"
UNVERIFIED = "unverified"
DOMINANCE_RATIO = 3
MIN_VARIANT_RESULTS = 2
# Domain endings that describe a site about the name, not a company carrying it.
_FAN_SITE_WORDS = {"forum", "forums", "club", "owners", "parts", "news", "blog", "reviews", "review", "fans", "wiki", "stock", "shop", "store", "dealer"}


def _label(url: str | None) -> str:
    domain = registrable_domain(url) if url and "://" in url else None
    return (domain or "").split(".")[0].lower()


def _variants_in_results(identity: CompanyIdentity, results: Iterable[Any]) -> tuple[dict[str, set[str]], Counter[str], Counter[str]]:
    """(variant name -> its own web domains, results per variant, results per domain carrying exactly the supplied name)."""
    base = identity.tokens[0]
    compact = "".join(identity.tokens)
    domains: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    exact: Counter[str] = Counter()
    for r in results:
        label, domain = _label(r.url), registrable_domain(r.url) or ""
        if label == compact:
            exact[domain] += 1
        elif label.startswith(base) and len(label) > len(base):
            word = label[len(base):].strip("-")
            variant = f"{base} {word}"
            if len(word) >= 3 and word not in _FAN_SITE_WORDS and word not in identity.tokens:
                domains[variant].add(domain)
                counts[variant] += 1
    return domains, counts, exact


def resolve_company(name: str, organisation_number: str | None, website_domain: str | None, results: Iterable[Any] = ()) -> dict[str, Any]:
    """The identity profile stored on the assessment: status, basis, official domain, aliases, other entities."""
    identity = CompanyIdentity(name, organisation_number)
    results = list(results)
    domains, counts, exact = _variants_in_results(identity, results) if identity.tokens else ({}, Counter(), Counter())
    exact_domain = exact.most_common(1)[0][0] if exact else None
    variants = [v for v, n in counts.items() if n >= MIN_VARIANT_RESULTS]
    profile: dict[str, Any] = {"status": UNVERIFIED, "basis": None, "official_domain": website_domain or exact_domain,
                               "aliases": [], "excluded": sorted(variants), "candidates": {v: counts[v] for v in sorted(variants)}}

    if website_domain:  # the analyst said which company is meant
        label = website_domain.split(".")[0].lower()
        alias = next((v for v in variants if v.replace(" ", "") == label), None)
        if alias is None and identity.tokens and label.startswith(identity.tokens[0]) and label != "".join(identity.tokens):
            word = label[len(identity.tokens[0]):].strip("-")
            alias = f"{identity.tokens[0]} {word}" if len(word) >= 3 else None
        return {**profile, "status": RESOLVED, "basis": "supplied_website", "aliases": [alias] if alias else [],
                "excluded": sorted(v for v in variants if v != alias)}
    if organisation_number:
        for r in results:
            if identity.org_number_in(f"{getattr(r, 'title', '')} {getattr(r, 'snippet', '')}"):
                return {**profile, "status": RESOLVED, "basis": "organisation_number", "official_domain": registrable_domain(r.url)}
    # a brand word's own domain settles it, unless several other entities carrying the word outweigh it in the results
    contested = len(identity.tokens) == 1 and len(variants) >= 2 and sum(exact.values()) < sum(counts[v] for v in variants)
    if exact_domain and not contested:
        return {**profile, "status": RESOLVED, "basis": "exact_domain"}
    if len(identity.tokens) > 1:  # a full name that is not just a brand word stands for itself
        return {**profile, "status": RESOLVED if results else UNVERIFIED, "basis": "full_name" if results else None}
    if len(variants) >= 2:
        ranked = sorted(variants, key=lambda v: counts[v], reverse=True)
        if counts[ranked[0]] >= DOMINANCE_RATIO * counts[ranked[1]]:
            return {**profile, "status": RESOLVED, "basis": "dominant_in_results", "aliases": [ranked[0]],
                    "official_domain": sorted(domains[ranked[0]])[0], "excluded": sorted(v for v in variants if v != ranked[0])}
        return {**profile, "status": AMBIGUOUS, "basis": "several_entities_in_results", "official_domain": None}
    if len(variants) == 1:
        return {**profile, "status": RESOLVED, "basis": "single_entity_in_results", "aliases": [variants[0]],
                "official_domain": sorted(domains[variants[0]])[0], "excluded": []}
    return {**profile, "status": RESOLVED if results else UNVERIFIED, "basis": "name_only" if results else None}


def identity_from_profile(name: str, organisation_number: str | None, profile: dict[str, Any] | None) -> CompanyIdentity:
    profile = profile or {}
    return CompanyIdentity(name, organisation_number, aliases=tuple(profile.get("aliases") or ()), excluded=tuple(profile.get("excluded") or ()))
