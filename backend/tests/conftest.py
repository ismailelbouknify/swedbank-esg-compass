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
os.environ["WEB_PROXY_MODE"] = "none"  # tests never depend on the proxy of the machine running them
os.environ["ALLOWED_HOSTS"] = "localhost,127.0.0.1,testserver"

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


@pytest.fixture(autouse=True)
def _robots_txt(request, monkeypatch):
    """robots.txt is enforced in tests marked `robots`. Elsewhere the mock sites have no robots.txt route, so the check
    is switched off there instead of every handler answering robots.txt; the cache never carries over."""
    from app.services.security import robots

    robots.cache.clear()
    if request.node.get_closest_marker("robots") is None:
        monkeypatch.setattr(robots, "check", lambda url, fetch_text: None)
    yield
    robots.cache.clear()


@pytest.fixture(autouse=True)
def _no_host_proxy_settings(monkeypatch):
    """Tests never read this computer's Windows proxy settings (PAC / WPAD); tests that need them set them."""
    from app.services.search import providers
    from app.services.security import network

    providers.pacer.reset()  # no cool-down or pacing carried between tests, and no real waiting
    monkeypatch.setattr(providers.pacer, "sleep", lambda _s: None)
    monkeypatch.setattr(network, "windows_config", lambda: None)
    monkeypatch.setattr(network, "windows_autoconfig_url", lambda: None)
    network.reset_pac_cache()
    network.reset_signin_state()
    yield
    network.reset_pac_cache()
    network.reset_signin_state()


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    set_llm_provider(None)
    with TestClient(app) as c:
        yield c
    set_llm_provider(None)


PASSAGE_RE = re.compile(r"<<(chunk_\d+)[^>]*>>\n(.*?)\n<<end \1>>", re.S)
# The words a passage must contain for the stand-in model to treat it as being about the factor.
FACTOR_WORDS = {
    "Energy Management": ("energy", "electricity", "solar"),
    "Product Design & Lifecycle Management": ("product", "circular", "recycl", "lifecycle"),
    "Data Security": ("data", "security", "iso 27001", "cyber"),
    "Labor Practices": ("labour", "labor", "working conditions", "wage"),
    "Employee Engagement, Diversity & Inclusion": ("women", "diversity", "inclusion", "engagement", "female"),
}


def prompt_passages(prompt: str) -> list[tuple[str, str]]:
    """(chunk label, text) of every passage in a RAG prompt."""
    return PASSAGE_RE.findall(prompt)


def prompt_field(prompt: str, name: str) -> str:
    m = re.search(rf"^{name}: (.*)$", prompt, re.M)
    return m.group(1).strip() if m else ""


def rag_answer(status: str, level=None, explanation: str = "", evidence=()) -> str:
    return json.dumps({"status": status, "recommended_level": level, "explanation": explanation,
                       "evidence": [{"chunk_id": c, "quote": q} for c, q in evidence]})


INSUFFICIENT = rag_answer("INSUFFICIENT_EVIDENCE", None, "The passages do not support a level.")


class FakeLLM(LLMProvider):
    """Deterministic stand-in for the local model on the RAG prompt: it reads the passages it is given, applies a few
    plain judgements and copies an exact sentence as its quotation, so prompting, JSON parsing, verification and the
    pipeline are exercised end to end. It is a test double, not a scoring method.

    `overrides` maps a text that appears in the prompt to a raw response (or a list of responses, one per call)."""

    provider_name = "fake"

    def __init__(self, overrides: dict[str, str | list[str]] | None = None):
        self.calls: list[str] = []
        self.overrides = {k: (list(v) if isinstance(v, list) else v) for k, v in (overrides or {}).items()}

    @property
    def model_name(self) -> str:
        return "fake:test-1b"

    def status(self) -> LLMStatus:
        return LLMStatus(True, self.provider_name, self.model_name, "fake ready")

    def complete_json(self, system, user, schema=None, max_tokens=384) -> str:
        self.calls.append(user)
        for key, raw in self.overrides.items():
            if key in user:
                return raw.pop(0) if isinstance(raw, list) and len(raw) > 1 else (raw[0] if isinstance(raw, list) else raw)
        return self.judge(user)

    def judge(self, user: str) -> str:
        factor, dimension = prompt_field(user, "ESG FACTOR"), prompt_field(user, "DIMENSION")
        words = FACTOR_WORDS.get(factor, ())
        sentences = []
        for label, text in prompt_passages(user):
            for line in text.split("\n"):
                for sent in re.split(r"(?<=[.!?])\s+", line):
                    if len(sent.split()) >= 4:
                        sentences.append((label, sent.strip(), text))

        def first(pattern: str, about_factor: bool = True):
            for label, sent, text in sentences:
                if re.search(pattern, sent, re.I) and (not about_factor or any(w in sent.lower() for w in words)):
                    return label, sent, text
            return None

        if dimension == "Reporting":
            hit = first(r"assurance", False)
            if hit:
                return rag_answer("EVIDENCE_FOUND", 6, "Disclosures are verified by an independent third party.", [hit[:2]])
            hit = first(r"in accordance with|GRI Standards", False)
            if hit:
                return rag_answer("EVIDENCE_FOUND", 5, "The report follows a recognised reporting standard.", [hit[:2]])
            hit = first(r"publish.*report|sustainability report|report\b.*\b(every year|annual)", False)
            if hit:
                return rag_answer("EVIDENCE_FOUND", 4, "The company reports regularly in a public report.", [hit[:2]])
            return INSUFFICIENT
        if dimension == "Planning":
            hit = first(r"\b(aim|goal|target|strategy|policy|commits?)\b")
            if not hit:
                return INSUFFICIENT
            label, sent, text = hit
            quantitative = re.search(r"\d+(?:\.\d+)?%", sent) and re.search(r"by 20\d\d", sent)
            level = 5 if quantitative and re.search(r"owned by|responsible", text, re.I) else 4
            return rag_answer("EVIDENCE_FOUND", level, "A formal, public goal for the factor is stated.", [(label, sent)])
        if dimension == "Execution":
            hit = first(r"\b(installed|launched|completed|conducted|replaced)\b")
            if not hit:
                return INSUFFICIENT
            level = 3 if re.search(r"target|strategy|roadmap|goal", hit[1], re.I) else 2
            return rag_answer("EVIDENCE_FOUND", level, "Activities to manage the factor were carried out.", [hit[:2]])
        hit = first(r"\b(decreased|reduced|increased|improved)\b.*\d")
        if not hit:
            return INSUFFICIENT
        return rag_answer("EVIDENCE_FOUND", 2, "Performance on the factor improved over recent years.", [hit[:2]])


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()
