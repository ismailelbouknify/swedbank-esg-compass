"""Reporting semantics, assurance scope, framework use, table row association and bounded extraction work.
Every company, domain, report and sentence here is invented: the rules are generic, not tuned to any real company.
No network: DNS and HTTP are mocked."""


from __future__ import annotations


import pymupdf


import pytest


from app.core.config import get_settings


from app.services.company.source_identity import CompanyIdentity


from app.services.pipeline.runner import run_assessment


from tests.conftest import INSUFFICIENT, FakeLLM


from tests.test_source_coverage import Search, _create, _page, _state, web  # noqa: F401


SITE = "https://www.kestreltiles.example"


NAME = "Kestrel Tiles AB"


COMPANY = CompanyIdentity(NAME)


REGULAR = dict(public=True, regular=True)


def test_marketing_only_website_cannot_create_reporting_level_3(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    pages = {
        f"{SITE}/": "Kestrel Tiles AB makes ceramic tiles. Our tiles are sustainable, durable and deliver striking results. " * 3,
        f"{SITE}/products": "Our products offer advantages for buyers who want affordable, sustainable and beautiful tiles. " * 3,
        f"{SITE}/production": "Our sustainable production line delivers great results and an environmentally friendly finish. " * 3,
    }
    for url, body in pages.items():
        web[url] = ("text/html", _page(f"{NAME}", body))
    aid = _create(client, name=NAME)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(u, NAME) for u in pages]))
    _, _, recs, evidence = _state(aid)
    assert not [e for e in evidence if e.factor == "reporting" and e.evidence_lifecycle == "accepted"]
    assert recs[("reporting", "reporting")].recommended_level != 3


STACKED = """Energy
Energy consumption by source
Unit
2022
2023
2024
2025
Fuel
MWh
12,400
11,900
11,300
10,800
Purchased electricity12
MWh
88,100
86,000
83,700
80,200
On-site renewable electricity
MWh
1,200
1,900
2,600
3,400
Total energy consumption
MWh
101,700
99,800
97,600
94,400
"""


INLINE = """Indicator Unit 2022 2023 2024 2025
Total electricity consumption MWh 88,100 86,000 83,700 80,200
Renewable electricity share % 41 48 55 61
Energy intensity MWh/tonne 1.92 1.85 1.80 1.71
"""


GOAL = "We aim to reduce energy consumption at our plants by 20% by 2030 compared with 2024."


def _goal_page(topic_sentence: str) -> str:
    return _page(f"{NAME}", f"Kestrel Tiles AB. {topic_sentence} " * 2)


def test_call_ceiling_is_a_safety_limit_never_a_false_answer(client, web, monkeypatch):
    """Replaces the passage round-robin of the removed extractor: one call now answers a whole question. With a
    ceiling of 3 calls, exactly 3 questions are asked; every other question with passages is withheld with the reason,
    never given a level."""
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    monkeypatch.setattr(get_settings(), "max_llm_calls_per_assessment", 3)
    topics = {
        "energy": "We aim to reduce energy consumption at our plants by 20% by 2030.",
        "security": "Our goal is ISO 27001 certification for our information security management by 2027.",
        "diversity": "Our goal is to reach 40% women in management positions by 2030.",
    }
    urls = []
    for key, sentence in topics.items():
        url = f"{SITE}/{key}"
        web[url] = ("text/html", _page(NAME, f"Kestrel Tiles AB sustainability. {sentence} " + "Background. " * 40))
        urls.append((url, f"{NAME} {key}"))
    llm = FakeLLM()
    aid = _create(client, name=NAME)
    run_assessment(aid, llm=llm, search=Search(first=urls))
    a, _, recs, _ = _state(aid)
    assert len(llm.calls) == 3 and a.coverage_metrics["model_calls"] == 3
    over = [r for r in recs.values() if r.triggered_rule == "rag:budget"]
    assert over and all(r.recommended_level == 0 and r.evidence_status == "REQUIRES_REVIEW" for r in over)
    assert all("processing limit" in r.reason for r in over)
    assert a.coverage_metrics["answer_outcomes"]["budget"] == len(over)


def _pdf(company: str) -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 780), f"{company}\nSustainability Report 2025\n"
                                  "We aim to reduce energy consumption by 30% by 2030 compared with 2020.\n" * 3, fontsize=11)
    doc.set_metadata({"title": f"{company} Sustainability Report 2025"})
    data = doc.tobytes()
    doc.close()
    return data


def test_the_same_pdf_under_two_addresses_is_parsed_once_and_timings_are_recorded(client, web, monkeypatch):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    pdf = _pdf(NAME)
    first, mirror = f"{SITE}/files/sustainability-report-2025.pdf", "https://cdn.tilefiles.example/k/sustainability-report-2025.pdf"
    web[first] = ("application/pdf", pdf)
    web[mirror] = ("application/pdf", pdf)
    aid = _create(client, name=NAME)
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(first, f"{NAME} report"), (mirror, f"{NAME} report")]))
    a, sources, *_ = _state(aid)
    assert sorted(s.status for s in sources) == ["PROCESSED", "SKIPPED"]
    m = a.coverage_metrics
    assert m["duplicate_documents_skipped"] == 1
    for key in ("model_calls", "model_seconds", "fetch_seconds", "pdf_parse_seconds", "total_seconds", "answer_outcomes"):
        assert key in m
    assert {"retrieval", "answering", "persisting"} <= set(m["stage_seconds"])


def _report_with_goals() -> bytes:
    doc = pymupdf.open()
    body = (f"{NAME}\nSustainability Report 2025\n{NAME} publishes this sustainability report every year.\n"
            "We aim to reduce energy consumption at our plants by 20% by 2030.\n"
            "Our goal is ISO 27001 certification for our information security management by 2027.\n")
    doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 780), body, fontsize=11)
    doc.set_metadata({"title": f"{NAME} Sustainability Report 2025"})
    data = doc.tobytes()
    doc.close()
    return data


def _run_with_budget(client, web, monkeypatch, budget: int):
    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    monkeypatch.setattr(get_settings(), "allow_heuristic_fallback", False)
    monkeypatch.setattr(get_settings(), "max_llm_calls_per_assessment", budget)
    url = f"{SITE}/files/sustainability-report-2025.pdf"
    web[url] = ("application/pdf", _report_with_goals())
    nothing = FakeLLM({"DIMENSION: Planning": INSUFFICIENT})
    aid = _create(client, name=NAME)
    run_assessment(aid, llm=nothing, search=Search(first=[(url, f"{NAME} sustainability report")]))
    return _state(aid)


def test_an_unread_question_is_insufficient_coverage_never_a_false_level_1(client, web, monkeypatch):
    a, _, recs, _ = _run_with_budget(client, web, monkeypatch, budget=1)
    assert a.coverage_metrics["primary_reports_processed"] == 1  # coverage itself is sufficient
    cut = [r for r in recs.values() if r.triggered_rule == "rag:budget"]
    assert cut
    for rec in cut:  # not read is not "no evidence": no level, flagged for review, with the reason
        assert rec.evidence_status == "REQUIRES_REVIEW" and rec.recommended_level == 0 and rec.coverage_state is None
        assert "processing limit" in rec.reason


def test_without_the_budget_the_same_question_is_an_honest_no_evidence(client, web, monkeypatch):
    _, _, recs, _ = _run_with_budget(client, web, monkeypatch, budget=0)  # 0 = no limit
    rec = recs[("data_security", "planning")]
    # an honest "no evidence found": recorded as not assessed (0), never as an automatic Level 1
    assert rec.coverage_state == "NO_EVIDENCE_FOUND" and rec.recommended_level == 0
    assert rec.reason.startswith("Insufficient evidence to recommend an assessment level.")

