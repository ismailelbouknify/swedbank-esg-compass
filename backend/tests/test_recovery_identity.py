"""Normal search and recovery search reach ESG processing through one identity gate. All companies are fictional."""
from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.services.pipeline import discovery
from app.services.pipeline.coverage import INSUFFICIENT_SOURCE_COVERAGE, NOT_ASSESSED_LEVEL
from app.services.pipeline.runner import run_assessment
from tests.conftest import FakeLLM
from tests.test_source_coverage import ENERGY_PAGE, NAME, REPORT_URL, SITE, Search, _create, _page, _report_pdf, _state, web  # noqa: F401

OTHER_REPORT = "https://cdn.filings.example/sustainability-report-2025.pdf"
SIMILAR_REPORT = "https://www.northstarglobal.example/sustainability-report-2025.pdf"


def _accepted(evidence, source_ids) -> list:
    return [e for e in evidence if e.source_id in source_ids and e.evidence_lifecycle == "accepted"]


def _only_recovery_finds(*found) -> Search:
    return Search(first=[], later=list(found))


def test_recovery_wrong_company_source_is_rejected_with_no_evidence_or_score(client, web):  # noqa: F811
    web[OTHER_REPORT] = ("application/pdf", _report_pdf("Bluefin Foods Ltd"))
    llm = FakeLLM()
    aid = _create(client)
    run_assessment(aid, llm=llm, search=_only_recovery_finds((OTHER_REPORT, f"{NAME} sustainability report 2025")))
    a, sources, recs, evidence = _state(aid)
    [source] = sources
    assert a.coverage_metrics["recovery_search_runs"] == 1
    assert source.status == "REJECTED" and source.company_match == "mismatch" and source.rejection_reason == "source_company_mismatch"
    assert evidence == [] and llm.calls == []  # no retrieval, extraction or model time
    assert all(r.evidence_ids == [] and r.recommended_level == NOT_ASSESSED_LEVEL for r in recs.values())


@pytest.mark.parametrize("url", [
    "https://cdn.filings.example/nsg/sustainability-report-2025.pdf",  # a neutral address: the document decides
    SIMILAR_REPORT,  # the other company's own address
])
def test_recovery_source_about_a_similarly_named_company_is_rejected(client, web, url):  # noqa: F811
    """The search query and title carry the assessed name; the document is about another company sharing its first word."""
    web[url] = ("application/pdf", _report_pdf("Northstar Global AS"))
    llm = FakeLLM()
    aid = _create(client)
    run_assessment(aid, search=_only_recovery_finds((url, f"{NAME} sustainability report 2025")), llm=llm)
    _, sources, recs, evidence = _state(aid)
    assert not [s for s in sources if s.status == "PROCESSED"]
    assert all(s.rejection_reason == "source_company_mismatch" for s in sources)
    assert evidence == [] and llm.calls == [] and all(r.evidence_ids == [] for r in recs.values())


def test_recovery_results_cannot_turn_a_one_word_name_into_a_similar_company(client, web):  # noqa: F811
    """The initial search failed, so nothing is settled yet. The only results recovery finds are another company
    carrying the word: they must not become the identity that then approves that company's own report."""
    report = "https://www.polarglobal.example/sustainability-report-2025.pdf"
    page = "https://www.polarglobal.example/sustainability"
    web[report] = ("application/pdf", _report_pdf("Polar Global AS"))
    web[page] = ("text/html", _page("Polar Global sustainability", "Polar Global reports its sustainability and climate work here. We aim to reduce energy use by 30% by 2030. " * 8))
    aid = _create(client, name="Polar")
    run_assessment(aid, llm=FakeLLM(), search=_only_recovery_finds((report, "Polar sustainability report 2025"), (page, "Polar sustainability")))
    a, sources, recs, evidence = _state(aid)
    assert a.company_identity["aliases"] == []
    assert not [s for s in sources if s.status == "PROCESSED"]
    assert evidence == [] and all(r.evidence_ids == [] and r.coverage_state == INSUFFICIENT_SOURCE_COVERAGE for r in recs.values())


def test_rejected_recovery_source_does_not_change_the_scores_of_valid_evidence(client, web):  # noqa: F811
    web[REPORT_URL] = ("application/pdf", _report_pdf())
    web[OTHER_REPORT] = ("application/pdf", _report_pdf("Bluefin Foods Ltd"))
    valid_only = _create(client)
    run_assessment(valid_only, llm=FakeLLM(), search=_only_recovery_finds((REPORT_URL, f"{NAME} sustainability report 2025")))
    with_other = _create(client)
    run_assessment(with_other, llm=FakeLLM(), search=_only_recovery_finds((REPORT_URL, f"{NAME} sustainability report 2025"), (OTHER_REPORT, f"{NAME} sustainability report 2024")))
    _, sources, recs, evidence = _state(with_other)
    rejected = {s.id for s in sources if s.url == OTHER_REPORT}
    assert {s.url: s.status for s in sources}[OTHER_REPORT] == "REJECTED"
    assert not [e for e in evidence if e.source_id in rejected]
    cited = {i for r in recs.values() for i in r.evidence_ids}
    assert not cited & {e.id for e in evidence if e.source_id in rejected}
    assert {k: r.recommended_level for k, r in recs.items()} == {k: r.recommended_level for k, r in _state(valid_only)[2].items()}


def test_valid_recovery_source_for_the_assessed_company_passes_and_improves_coverage(client, web, monkeypatch):  # noqa: F811
    web[REPORT_URL] = ("application/pdf", _report_pdf())
    search_finding_it = _only_recovery_finds((REPORT_URL, f"{NAME} sustainability report 2025"))

    monkeypatch.setattr(get_settings(), "recovery_search_enabled", False)
    without = _create(client)
    run_assessment(without, llm=FakeLLM(), search=_only_recovery_finds((REPORT_URL, f"{NAME} report")))
    before = _state(without)

    monkeypatch.setattr(get_settings(), "recovery_search_enabled", True)
    with_recovery = _create(client)
    run_assessment(with_recovery, llm=FakeLLM(), search=search_finding_it)
    a, sources, recs, evidence = _state(with_recovery)

    assert before[1] == [] and before[0].coverage_metrics["sources_validated"] == 0
    assert all(r.coverage_state == INSUFFICIENT_SOURCE_COVERAGE for r in before[2].values())
    assert [s.status for s in sources] == ["PROCESSED"] and sources[0].company_match
    assert a.coverage_metrics["sources_validated"] == 1 and a.coverage_metrics["primary_reports_processed"] == 1
    assert _accepted(evidence, {sources[0].id})
    assert recs[("energy_management", "planning")].recommended_level >= 4


@pytest.mark.parametrize("found_by", ["normal", "recovery"])
def test_both_search_paths_call_the_same_gate_before_any_processing(client, web, monkeypatch, found_by):  # noqa: F811
    good, wrong = REPORT_URL, OTHER_REPORT
    web[good] = ("application/pdf", _report_pdf())
    web[wrong] = ("application/pdf", _report_pdf("Bluefin Foods Ltd"))
    results = [(good, f"{NAME} sustainability report 2025"), (wrong, f"{NAME} sustainability report 2024")]
    checked: list[str] = []
    real_gate = discovery.apply_source_identity

    def recording_gate(db, a, s):
        checked.append(s.url)
        return real_gate(db, a, s)

    monkeypatch.setattr(discovery, "apply_source_identity", recording_gate)
    search = Search(first=results) if found_by == "normal" else Search(first=[], later=results)
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=search)
    a, sources, _, evidence = _state(aid)
    assert set(checked) == {good, wrong} and len(checked) == 2  # each source, once, on either path
    statuses = {s.url: s.status for s in sources}
    assert statuses == {good: "PROCESSED", wrong: "REJECTED"}
    assert not [e for e in evidence if e.source_url == wrong]


def test_normal_search_alone_is_unchanged_and_needs_no_recovery(client, web):  # noqa: F811
    urls = [(f"{SITE}/sustainability/{i}", f"{NAME} sustainability {i}") for i in range(3)]
    for url, _ in urls:
        web[url] = ("text/html", ENERGY_PAGE)
    web[REPORT_URL] = ("application/pdf", _report_pdf())
    search = Search(first=[(REPORT_URL, f"{NAME} sustainability report 2025"), *urls])
    aid = _create(client)
    run_assessment(aid, llm=FakeLLM(), search=search)
    a, sources, recs, _ = _state(aid)
    assert a.coverage_metrics["recovery_search_runs"] == 0 and len(search.queries) == 8
    assert all(s.status == "PROCESSED" for s in sources)
    assert recs[("energy_management", "planning")].recommended_level >= 4


def test_normal_search_can_still_settle_a_brand_name(client, web):  # noqa: F811
    cars = "https://www.polarcars.example/sustainability"
    web[cars] = ("text/html", _page("Polar Cars sustainability", "Polar Cars reports here. Polar Cars aims high. " * 10))
    aid = _create(client, name="Polar")
    run_assessment(aid, llm=FakeLLM(), search=Search(first=[(cars, "Polar Cars"), (cars + "#2", "Polar Cars")]))
    a, *_ = _state(aid)
    assert a.company_identity["aliases"] == ["polar cars"]


def test_recovery_makes_no_model_calls_of_its_own(client, web):  # noqa: F811
    web[REPORT_URL] = ("application/pdf", _report_pdf())
    direct, recovered = FakeLLM(), FakeLLM()
    run_assessment(_create(client), llm=direct, search=Search(first=[(REPORT_URL, f"{NAME} sustainability report 2025")]))
    run_assessment(_create(client), llm=recovered, search=_only_recovery_finds((REPORT_URL, f"{NAME} sustainability report 2025")))
    assert direct.calls and len(recovered.calls) == len(direct.calls)
