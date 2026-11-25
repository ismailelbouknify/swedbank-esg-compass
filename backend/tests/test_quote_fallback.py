"""Deterministic quote fallback: strict conditions, no extra model call, normal verification, nothing rescued."""


from __future__ import annotations


import json


import pytest


from sqlalchemy import select


from app.db.session import SessionLocal


from app.models.entities import Evidence, Recommendation


from app.services.company.source_identity import CompanyIdentity


from app.services.pipeline.runner import run_assessment


from app.services.search.base import NullSearchProvider


from tests.conftest import FakeLLM


COMPANY = CompanyIdentity("Northstar Manufacturing AB")


GOAL = "We aim to reduce energy consumption by 30% by 2030 compared with a 2020 baseline."


ACTIVITY = "In 2025 we completed energy audits at all sites as part of our energy efficiency roadmap."


class _Quoting(FakeLLM):
    """Always reports a goal, with a chosen quote and optionally chosen values."""

    def __init__(self, quote: str, **extra):
        super().__init__()
        self.quote, self.extra = quote, extra

    def complete_json(self, system, user, schema=None, max_tokens=384) -> str:
        self.calls.append(user)
        return json.dumps({"goal_found": True, "formalized": True, "quantitative": False, "evidence_quote": self.quote, "confidence": 0.9, **self.extra})


class _WrongSentenceLLM(FakeLLM):
    """The observed 1B behaviour: says a goal exists, quotes a real sentence that is not the goal."""

    def complete_json(self, system, user, schema=None, max_tokens=384) -> str:
        self.calls.append(user)
        if "DIMENSION:\nPlanning" in user:
            return json.dumps({"goal_found": True, "formalized": True, "quantitative": False,
                               "evidence_quote": "The energy target is owned by our Chief Operating Officer and progress is reviewed quarterly",
                               "confidence": 0.8})
        return super().complete_json(system, user, schema, max_tokens)


def _run(client, sample_pdf, llm, name="Nordvik Components AB") -> int:
    r = client.post("/api/assessments", json={"company_name": name, "year": 2025, "enable_web_search": False, "confirm_new": True})
    aid = r.json()["id"]
    with open(sample_pdf, "rb") as f:
        assert client.post(f"/api/assessments/{aid}/documents", files={"file": ("report.pdf", f, "application/pdf")}).status_code == 201
    run_assessment(aid, llm=llm, search=NullSearchProvider())
    return aid


def _rows(aid):
    with SessionLocal() as db:
        return (list(db.scalars(select(Evidence).where(Evidence.assessment_id == aid))),
                {(r.factor, r.dimension): r for r in db.scalars(select(Recommendation).where(Recommendation.assessment_id == aid))})


def test_wrong_company_source_stays_blocked_with_no_model_call_and_no_fallback(client, sample_pdf):
    llm = _WrongSentenceLLM()
    aid = _run(client, sample_pdf, llm, name="Bluefin Foods Ltd")
    evidence, recs = _rows(aid)
    assert llm.calls == [] and evidence == [] and all(r.recommended_level == 0 and r.evidence_ids == [] for r in recs.values())

