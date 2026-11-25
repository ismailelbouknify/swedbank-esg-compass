"""Source identity as a hard gate, actor attribution, and whether a figure really is the company's target.
All companies are fictional."""


from __future__ import annotations


import json


import pytest


from sqlalchemy import select


from app.db.session import SessionLocal


from app.models.entities import Assessment, Evidence, Recommendation, Source


from app.services.company.source_identity import CompanyIdentity, assess_source_identity


from app.services.export import exporter


from app.services.llm.base import NullLLMProvider


from app.services.pipeline.runner import AssessmentRunner, run_assessment


from app.services.pipeline.source_gate import source_blocks_evidence


from app.services.search.base import NullSearchProvider


from tests.conftest import FakeLLM, prompt_passages, rag_answer


COMPANY = CompanyIdentity("Northstar Manufacturing AB")


SAMPLE_COMPANY = "Nordvik Components AB"


def _assess(text, title=None, url=None, company=COMPANY, **kw):
    return assess_source_identity(company, title=title, url=url, text=text, **kw)


def test_document_dominated_by_a_different_company_is_rejected():
    text = "Bluefin Foods annual filing. " * 30 + "Bluefin Foods reported results."
    assert not _assess(text, title="Form 10-K").matched
    # a report with no legal suffix anywhere and a passing mention of the assessed company
    assert not _assess(("Bluefin reduced energy use. " * 40) + "Northstar Manufacturing supplies parts.", title="Annual filing").matched


@pytest.mark.parametrize("title, text", [
    ("Northstar Global AS", "We trade commodities worldwide. Northstar Global trades."),
    ("Sustainability reports | Northstar Group", "Reports of the Northstar Group are published here. Northstar employs many."),
    ("Northstar og miljoet", "Northstar er miljofyrtarnsertifisert. Northstar jobber med miljo."),
    ("Northstars Sustainability Report", "Northstars is a different business. Northstars grows."),
])
def test_similar_company_name_is_rejected(title, text):
    assert not _assess(text, title=title).matched


def test_shared_first_word_or_address_is_not_identity():
    assert not _assess("We trade goods.", url="https://northstar.example/northstar-manufacturing.pdf").matched
    assert not _assess("Northstar sells tools. Northstar grows.", url="https://northstar.example/").matched


def test_same_company_in_title_with_legal_form_or_plural_still_matches():
    assert _assess("We reduced energy use.", title="Northstar Manufacturing AB | Sustainability").matched
    cars = CompanyIdentity("Polar Cars")
    assert assess_source_identity(cars, title="Climate Action | Polar Cars", url=None, text="We reduced emissions.").matched
    assert assess_source_identity(CompanyIdentity("Polar Car AB"), title="Polar Cars", url=None, text="We reduced emissions.").matched


def test_division_on_the_companys_own_exact_domain_is_not_another_company():
    acme = CompanyIdentity("Acme")
    text = "Acme Cloud helps. Acme Cloud grows. Acme Cloud scales. Acme Cloud runs. Acme Cloud is fast."
    assert assess_source_identity(acme, title="Sustainability", url="https://cloud.acme.com/sustainability", text=text).basis == "name_domain"
    assert not assess_source_identity(acme, title="Sustainability", url="https://acmecloud.example/", text=text).matched


def _assessment(client, name=SAMPLE_COMPANY) -> int:
    r = client.post("/api/assessments", json={"company_name": name, "year": 2025, "enable_web_search": False, "confirm_new": True})
    assert r.status_code == 201
    return r.json()["id"]


def _upload(client, aid, pdf):
    with open(pdf, "rb") as f:
        assert client.post(f"/api/assessments/{aid}/documents", files={"file": ("report.pdf", f, "application/pdf")}).status_code == 201


def _rows(aid):
    with SessionLocal() as db:
        return (list(db.scalars(select(Source).where(Source.assessment_id == aid))),
                list(db.scalars(select(Evidence).where(Evidence.assessment_id == aid))),
                list(db.scalars(select(Recommendation).where(Recommendation.assessment_id == aid))))


def test_wrong_company_source_cannot_generate_evidence_or_affect_scoring(client, sample_pdf):
    aid = _assessment(client, "Bluefin Foods Ltd")
    _upload(client, aid, sample_pdf)
    llm = FakeLLM()
    run_assessment(aid, llm=llm, search=NullSearchProvider())
    sources, evidence, recs = _rows(aid)
    assert sources[0].status == "REJECTED" and sources[0].rejection_reason == "source_company_mismatch"
    assert source_blocks_evidence(sources[0]) and llm.calls == [] and evidence == []
    assert all(r.recommended_level == 0 and r.evidence_ids == [] for r in recs)  # a gap, not a Level 1 answer


def test_export_contains_audit_validation_fields(client, sample_pdf):
    """An answer whose quotation is not in the source is withheld; the export still records why, per quotation."""
    invented = rag_answer("EVIDENCE_FOUND", 4, "A goal exists.", [("chunk_1", "This sentence was never in the report at all.")])

    class WrongQuote(FakeLLM):
        def complete_json(self, system, user, schema=None, max_tokens=384):
            if "DIMENSION: Planning" in user:
                self.calls.append(user)
                label = prompt_passages(user)[0][0]
                return invented.replace("chunk_1", label)
            return super().complete_json(system, user, schema, max_tokens)

    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    run_assessment(aid, llm=WrongQuote(), search=NullSearchProvider())
    data = client.get(f"/api/assessments/{aid}/export", params={"format": "json"}).json()
    required = {"evidence_lifecycle", "rejection_codes", "company_match", "used_for_scoring", "scoring_exclusion_reason"}
    assert data["evidence"] and all(required <= set(e) for e in data["evidence"])
    withheld = [e for e in data["evidence"] if e["evidence_lifecycle"] == "rejected"]
    assert withheld and all("quote_not_found" in e["rejection_codes"] and e["used_for_scoring"] is False for e in withheld)
    assert all(not e["quote_confirmed_in_source"] for e in withheld)
    assert set(exporter.EVIDENCE_VALIDATION_COLUMNS) >= required
    planning = [r for r in data["recommendations"] if r["dimension"] == "Planning"]
    assert planning and all(r["ai_recommended_level"] is None and r["evidence_verification"] == "not_verified" for r in planning)


def test_audit_fields_stay_out_of_the_normal_ui_endpoints(client, sample_pdf):
    aid = _assessment(client)
    _upload(client, aid, sample_pdf)
    run_assessment(aid, llm=FakeLLM(), search=NullSearchProvider())
    recs = client.get(f"/api/assessments/{aid}/recommendations").json()
    body = client.get(f"/api/assessments/{aid}/sources").text + json.dumps(recs) + client.get(f"/api/recommendations/{recs[0]['id']}").text
    for word in ("actor_match", "company_match", "rejection_codes", "scoring_exclusion_reason", "evidence_origin"):
        assert word not in body

