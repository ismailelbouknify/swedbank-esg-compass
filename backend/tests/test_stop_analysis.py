"""Stopping a running analysis from the progress page."""
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.entities import Assessment, AuditEvent, Recommendation
from app.services.llm.base import LLMUnavailableError
from app.services.llm.factory import set_llm_provider
from app.services.pipeline import stop
from app.services.pipeline.runner import run_assessment
from app.services.search.base import NullSearchProvider
from tests.conftest import FakeLLM


def _assessment(client, pdf) -> int:
    r = client.post("/api/assessments", json={"company_name": "Nordvik Components AB", "year": 2025,
                                              "enable_web_search": False, "confirm_new": True})
    aid = r.json()["id"]
    with open(pdf, "rb") as f:
        assert client.post(f"/api/assessments/{aid}/documents", files={"file": ("report.pdf", f, "application/pdf")}).status_code == 201
    return aid


class StopsAfter(FakeLLM):
    """Answers normally, and asks to stop the analysis during its `n`-th call."""

    def __init__(self, aid: int, n: int):
        super().__init__()
        self.aid, self.n = aid, n

    def complete_json(self, system, user, schema=None, max_tokens=384):
        raw = super().complete_json(system, user, schema, max_tokens)
        if len(self.calls) == self.n:
            stop.request(self.aid)  # what POST /stop does while this call is running
        return raw


def test_a_running_analysis_stops_between_questions_and_saves_no_recommendations(client, sample_pdf):
    aid = _assessment(client, sample_pdf)
    llm = StopsAfter(aid, 2)
    run_assessment(aid, llm=llm, search=NullSearchProvider())
    assert len(llm.calls) == 2  # no further question was asked
    with SessionLocal() as db:
        a = db.get(Assessment, aid)
        assert a.status == "FAILED" and a.error == stop.STOPPED_MESSAGE
        assert db.scalars(select(Recommendation).where(Recommendation.assessment_id == aid)).all() == []
        assert "assessment_stopped" in {e.event_type for e in db.scalars(select(AuditEvent).where(AuditEvent.assessment_id == aid))}
    st = client.get(f"/api/assessments/{aid}/status").json()
    assert st["status"] == "FAILED" and st["error"] == stop.STOPPED_MESSAGE  # shown as is to the analyst
    assert any(s["state"] == "failed" for s in st["stages"])


def test_a_stopped_analysis_can_be_started_again(client, sample_pdf):
    aid = _assessment(client, sample_pdf)
    run_assessment(aid, llm=StopsAfter(aid, 1), search=NullSearchProvider())
    set_llm_provider(FakeLLM())
    r = client.post(f"/api/assessments/{aid}/run")  # the progress page's "Start again"
    assert r.status_code == 202
    st = client.get(f"/api/assessments/{aid}/status").json()
    assert st["status"] == "COMPLETED" and st["error"] is None
    assert len(client.get(f"/api/assessments/{aid}/recommendations").json()) == 16


def test_only_a_running_analysis_can_be_stopped(client, sample_pdf):
    aid = _assessment(client, sample_pdf)
    assert client.post(f"/api/assessments/{aid}/stop").status_code == 409  # not started
    with SessionLocal() as db:
        db.get(Assessment, aid).status = "RUNNING"
        db.commit()
    r = client.post(f"/api/assessments/{aid}/stop")
    assert r.status_code == 202 and r.json()["stage_label"] in ("Starting analysis", "Stopping the analysis")
    assert stop.requested(aid)
    with SessionLocal() as db:
        db.get(Assessment, aid).stage = 4
        db.commit()
    assert client.get(f"/api/assessments/{aid}/status").json()["stage_label"] == "Stopping the analysis"
    assert client.post("/api/assessments/999999/stop").status_code == 404
    stop.clear(aid)


def test_a_model_call_in_progress_stops_at_its_next_token():
    from app.services.llm.llama_cpp_provider import LlamaCppProvider

    class SlowLlama:
        def __init__(self):
            self.tokens = 0

        def create_chat_completion(self, **kwargs):
            for _ in range(10_000):
                self.tokens += 1
                yield {"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}

        def tokenize(self, data, add_bos=True):
            return [0]

    provider = LlamaCppProvider(Path("model.gguf"))
    provider._llm = SlowLlama()
    provider.abort_check = lambda: provider._llm.tokens >= 5
    with pytest.raises(LLMUnavailableError, match="Stopped on request"):
        provider.complete_json("s", "u")
    assert provider._llm.tokens == 5 and provider._timed_out is False  # stopping is not a model failure
