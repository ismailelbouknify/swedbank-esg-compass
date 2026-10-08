"""Test fixtures. Tests never need the real model: the LLM is mocked or disabled."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="esg-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{(_TMP / 'test.db').as_posix()}"
os.environ["DATA_DIR"] = str(_TMP / "data")
os.environ["LLM_PROVIDER"] = "none"
os.environ["SEARCH_PROVIDER"] = "none"

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sample_data"))

from app.db.session import init_db  # noqa: E402
from app.services.llm.base import LLMProvider, LLMStatus  # noqa: E402
from app.services.llm.factory import set_llm_provider  # noqa: E402


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    from build_sample_pdf import build_pdf

    return build_pdf(_TMP / "nordvik_sustainability_report_2025.pdf")


@pytest.fixture(scope="session", autouse=True)
def _db():
    init_db()
    yield


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    set_llm_provider(None)
    with TestClient(app) as c:
        yield c
    set_llm_provider(None)


def _text_block(prompt: str) -> str:
    m = re.search(r'TEXT:\n"""\n(.*)\n"""', prompt, re.S)
    return m.group(1) if m else ""


class FakeLLM(LLMProvider):
    """Deterministic stand-in for the 1B model. Answers from the passage text so the LLM code
    path (prompting, JSON parsing, validation, verification) is exercised end to end."""

    provider_name = "fake"

    def __init__(self, overrides: dict[str, str] | None = None):
        self.calls: list[str] = []
        self.overrides = overrides or {}

    @property
    def model_name(self) -> str:
        return "fake:test-1b"

    def status(self) -> LLMStatus:
        return LLMStatus(True, self.provider_name, self.model_name, "fake ready")

    def complete_json(self, system, user, schema=None, max_tokens=384) -> str:
        self.calls.append(user)
        for key, raw in self.overrides.items():
            if key in user:
                return raw
        text = _text_block(user)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]
        if "DIMENSION:\nPlanning" in user:
            goal = next((s for s in sentences if re.search(r"\b(aim|goal|target|strategy|policy|commits?)\b", s, re.I)), None)
            if not goal:
                return json.dumps({"goal_found": False, "evidence_quote": "", "confidence": 0.3})
            pct = re.search(r"(\d+(?:\.\d+)?)%", goal)
            year = re.search(r"by (20\d\d)", goal)
            base = re.search(r"(20\d\d) baseline", goal)
            return "```json\n" + json.dumps({
                "goal_found": True, "formalized": True, "quantitative": bool(pct),
                "target_value": float(pct.group(1)) if pct else None, "target_unit": "%" if pct else None,
                "target_year": int(year.group(1)) if year else None, "baseline_year": int(base.group(1)) if base else None,
                "clear_owner": "owned by" in text, "owner": None, "benchmark_found": False, "benchmark_type": None,
                "evidence_quote": goal, "confidence": 0.8,
            }) + "\n```"
        if "DIMENSION:\nExecution" in user:
            act = next((s for s in sentences if re.search(r"\b(installed|launched|completed|conducted)\b", s)), None)
            if not act:
                return json.dumps({"activity_found": False, "evidence_quote": "", "confidence": 0.3})
            year = re.search(r"\b(20\d\d)\b", act)
            return json.dumps({"activity_found": True, "activity_description": act[:80],
                               "activity_year": int(year.group(1)) if year else None,
                               "linked_to_goal": bool(re.search(r"target|strategy|roadmap", act)),
                               "evidence_quote": act, "confidence": 0.7})
        if "DIMENSION:\nReporting" in user:
            s = next((s for s in sentences if "assurance" in s.lower()), "")
            return json.dumps({"sustainability_assurance": bool(s), "assurance_provider": "KPMG" if "KPMG" in s else None,
                               "assurance_standard": "ISAE 3000" if "ISAE 3000" in s else None,
                               "assurance_level": "limited" if "limited" in s else None, "evidence_quote": s, "confidence": 0.8})
        return json.dumps({"metric_found": False, "values": [], "improvement_stated_without_numbers": False,
                           "sector_comparison": None, "sector_comparison_favorable": None, "evidence_quote": "", "confidence": 0.2})


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()
