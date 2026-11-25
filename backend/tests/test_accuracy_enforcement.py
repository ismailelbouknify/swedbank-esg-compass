"""Accuracy-enforcement gates: source identity, accepted-only scoring, actor and state validation, factor
gate, table alignment, reporting aggregation, export fields. All companies here are fictional."""


from __future__ import annotations


import httpx


import pytest


from sqlalchemy import select


from app.db.session import SessionLocal


from app.models.entities import AuditEvent, Evidence, Recommendation, Source


from app.services.company.source_identity import CompanyIdentity, assess_source_identity, looks_like_company_report


from app.services.export import exporter


from app.services.pipeline.runner import COMPANY_MISMATCH_MESSAGE, run_assessment


from app.services.search.base import NullSearchProvider, SearchProvider, SearchResult


from app.services.security import safe_fetch


from tests.conftest import FakeLLM


COMPANY = CompanyIdentity("Northstar Manufacturing AB", "556677-8899".replace("-", ""))


SAMPLE_COMPANY = "Nordvik Components AB"


REPORT_TEXT = "Northstar Manufacturing AB\nSustainability Report 2025\nWe reduced energy use. Northstar Manufacturing owns five plants."


def _assess(text: str, title: str | None = None, url: str | None = None, **kw):
    return assess_source_identity(COMPANY, title=title, url=url, text=text, **kw)


def test_own_report_matches_on_name():
    assert _assess(REPORT_TEXT).matched
    assert _assess("We reduced energy use.", title="Northstar Manufacturing Sustainability Report 2025").basis == "name_in_title"
    # an address alone proves nothing without the document's own text
    assert not _assess("We reduced energy use.", url="https://x.example/northstar-manufacturing-2025.pdf").matched
    assert _assess("Northstar Manufacturing reduced energy use.", url="https://x.example/northstar-manufacturing-2025.pdf").matched


def test_short_name_and_organisation_number_identify_the_company():
    text = "Northstar is a producer of tools. Northstar reduced its energy use. Northstar Manufacturing is its legal name, used in filings."
    assert _assess(text).basis in ("alias", "name_in_text")
    assert not _assess("Northstar is a producer of tools. Northstar reduced its energy use.").matched  # the first word alone is not enough
    assert _assess("Org. nr 556677-8899. Annual figures follow.").basis == "organisation_number"


def test_wrong_company_source_rejected():
    result = _assess("Bluefin Foods Ltd\nSustainability Report 2025\nBluefin Foods Ltd reduced energy use. Bluefin Foods grows.",
                     title="Bluefin Foods Sustainability Report 2025")
    assert not result.matched and result.basis == "mismatch"


def test_source_never_naming_the_company_rejected():
    assert not _assess("We reduced energy use across all of our plants this year.").matched


def test_similar_company_name_rejected():
    text = "Northstar Shipping AB\nNorthstar Shipping AB annual review.\nNorthstar Shipping operates vessels. Northstar Shipping grew."
    assert not _assess(text, title="Northstar Shipping Sustainability Report 2025").matched
    # a name continuing the assessed single-word name is another company too
    acme = CompanyIdentity("Acme")
    words = "Acme Foods sells. Acme Foods grew. Acme Foods hired. Acme Foods invested. Acme Foods expanded."
    assert not assess_source_identity(acme, title=None, url=None, text=words).matched
    # a name that only begins the same way is no match at all
    assert not assess_source_identity(CompanyIdentity("Northstar"), title=None, url=None, text="Northstarlings reduced energy use.").matched


def test_generic_first_word_alone_does_not_identify_a_company():
    identity = CompanyIdentity("Nordic Steel AB")
    assert not assess_source_identity(identity, title=None, url=None, text="Nordic Paper produces paper. Nordic Paper grew.").matched


def test_passing_mention_in_another_companys_report_rejected():
    filler = "Bluefin Foods reduced energy use. " * 10
    text = filler + "Among our suppliers is Northstar Manufacturing AB, and Northstar Manufacturing delivers tools."
    assert not _assess(text, title="Bluefin Foods Sustainability Report 2025").matched


def test_analyst_supplied_domain_establishes_identity_but_a_guess_does_not():
    assert _assess("We reduced energy use.", url="https://own.example/r.pdf", on_official_domain=True).basis == "official_domain"
    assert not _assess("We reduced energy use.", url="https://own.example/r.pdf", on_official_domain=False).matched


def test_upload_without_a_name_is_kept_but_upload_of_another_company_is_not():
    assert _assess("We reduced energy use.", uploaded=True).basis == "analyst_upload"
    other = "Bluefin Foods AB reduced energy use. Bluefin Foods AB grew."
    assert not _assess(other, uploaded=True).matched


def test_auditor_and_title_noise_are_not_treated_as_another_company():
    text = REPORT_TEXT + " KPMG AB assured this report. KPMG AB issued a statement. KPMG AB signed. KPMG AB reviewed."
    assert _assess(text, title="Towards Net Zero Sustainability Report 2025").matched


class _Search(SearchProvider):
    name = "test"

    def __init__(self, urls):
        self.urls = urls

    def search(self, query, max_results):
        return [SearchResult(f"{SAMPLE_COMPANY} sustainability report 2025", u, "sustainability report", query) for u in self.urls]


def _assessment(client, name: str = SAMPLE_COMPANY, **kw) -> int:
    body = {"company_name": name, "year": 2025, "enable_web_search": False, "confirm_new": True, **kw}
    r = client.post("/api/assessments", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _upload(client, aid: int, pdf) -> None:
    with open(pdf, "rb") as f:
        assert client.post(f"/api/assessments/{aid}/documents", files={"file": ("report.pdf", f, "application/pdf")}).status_code == 201


def _recs(aid: int) -> dict[tuple[str, str], Recommendation]:
    with SessionLocal() as db:
        return {(r.factor, r.dimension): r for r in db.scalars(select(Recommendation).where(Recommendation.assessment_id == aid))}


def _evidence(aid: int) -> list[Evidence]:
    with SessionLocal() as db:
        return list(db.scalars(select(Evidence).where(Evidence.assessment_id == aid)))


def test_wrong_company_upload_never_reaches_extraction_or_scoring(client, sample_pdf):
    aid = _assessment(client, name="Bluefin Foods Ltd")
    _upload(client, aid, sample_pdf)
    llm = FakeLLM()
    run_assessment(aid, llm=llm, search=NullSearchProvider())
    assert llm.calls == [], "no model time is spent on the wrong company"
    assert _evidence(aid) == []
    assert all(r.recommended_level == 0 and r.coverage_state == "INSUFFICIENT_SOURCE_COVERAGE" for r in _recs(aid).values())
    with SessionLocal() as db:
        source = db.scalars(select(Source).where(Source.assessment_id == aid)).one()
        events = {e.event_type for e in db.scalars(select(AuditEvent).where(AuditEvent.assessment_id == aid))}
    assert (source.status, source.company_match, source.rejection_reason) == ("REJECTED", "mismatch", "source_company_mismatch")
    assert "source_company_mismatch" in events
    st = client.get(f"/api/assessments/{aid}/status").json()
    assert st["status"] == "COMPLETED" and COMPANY_MISMATCH_MESSAGE in st["notices"]


def _table(*rows: str) -> str:
    return "\n".join(rows)


class _FabricatingLLM(FakeLLM):
    """Answers every question with a quote that is not in any document."""

    def complete_json(self, system, user, schema=None, max_tokens=384) -> str:
        self.calls.append(user)
        if "DIMENSION:\nPlanning" in user:
            return ('{"goal_found": true, "formalized": true, "quantitative": true, "target_value": 90, "target_unit": "%", '
                    '"target_year": 2030, "evidence_quote": "We promise to reach 90% of everything by 2030 without fail.", "confidence": 0.9}')
        if "DIMENSION:\nExecution" in user:
            return ('{"activity_found": true, "activity_description": "x", "activity_year": 2025, "linked_to_goal": true, '
                    '"evidence_quote": "We installed energy systems on the moon in 2025 and audited privacy data security diversity.", "confidence": 0.9}')
        return super().complete_json(system, user, schema, max_tokens)


def test_accepted_valid_evidence_still_reaches_scoring(client, sample_pdf):
    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    run_assessment(aid, llm=FakeLLM(), search=NullSearchProvider())
    recs = _recs(aid)
    assert recs[("energy_management", "planning")].recommended_level >= 4
    assert recs[("energy_management", "execution")].recommended_level >= 2
    assert recs[("energy_management", "performance")].recommended_level == 2
    assert recs[("diversity_inclusion", "planning")].recommended_level >= 4
    assert sum(1 for r in recs.values() if r.evidence_ids) >= 8


def test_processed_report_prevents_false_no_reporting(client, sample_pdf):
    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    run_assessment(aid, llm=FakeLLM(), search=NullSearchProvider())
    rec = _recs(aid)[("reporting", "reporting")]
    assert rec.recommended_level >= 4 and rec.triggered_rule != "R1_NO_REPORTING_EVIDENCE" and rec.evidence_ids


def test_report_found_at_a_neutral_web_address_counts_as_a_report(client, sample_pdf, monkeypatch):
    """Classified from the document: the web address gave no hint that this is a report."""
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))])
    pdf = sample_pdf.read_bytes()
    handler = lambda _r: httpx.Response(200, headers={"content-type": "application/pdf"}, content=pdf)  # noqa: E731
    monkeypatch.setattr(safe_fetch, "build_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    url = "https://files.hosting.example/documents/doc-7781.pdf"

    class Plain(_Search):
        def search(self, query, max_results):
            return [SearchResult("Nordvik Components document", u, "document", query) for u in self.urls]

    aid = _assessment(client, enable_web_search=True)
    run_assessment(aid, llm=FakeLLM(), search=Plain([url]))
    with SessionLocal() as db:
        source = db.scalars(select(Source).where(Source.assessment_id == aid, Source.url == url)).one()
    assert source.source_type == "sustainability_report" and source.status == "PROCESSED" and source.company_match
    rec = _recs(aid)[("reporting", "reporting")]
    assert rec.recommended_level >= 4 and rec.triggered_rule != "R1_NO_REPORTING_EVIDENCE"


def test_only_a_published_report_is_reclassified():
    assert looks_like_company_report("Sustainability Report 2025", "We reduced energy use.")
    assert looks_like_company_report(None, "Environmental Report 2024\nWe reduced energy use.")
    assert not looks_like_company_report("Press release", "We reduced energy use.")
    assert not looks_like_company_report("Sustainability Report 2025", "An analysis of a company by outside researchers.")


def test_analyst_endpoints_do_not_expose_validation_internals(client, sample_pdf):
    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    run_assessment(aid, llm=FakeLLM(), search=NullSearchProvider())
    body = client.get(f"/api/assessments/{aid}/sources").text + client.get(f"/api/assessments/{aid}/recommendations").text
    for word in ("company_match", "rejection_reason", "rejection_codes", "review_evidence_ids", "used_for_scoring"):
        assert word not in body

