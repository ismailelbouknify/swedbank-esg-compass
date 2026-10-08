"""Saved assessments: lookup/reuse, identity normalization, versioned re-runs, duration, list,
analyst-facing output without technical metadata, and the additive schema migration."""
from __future__ import annotations

import json
import sqlite3
import uuid

import pytest

from app.services.company.identify import normalize_company_name, normalize_org_number
from app.services.llm.factory import set_llm_provider
from tests.conftest import FakeLLM

FORBIDDEN = ("llama", "gguf", "fake:test-1b", "llm_provider", "llm_model", "extraction_mode", "triggered_rule",
             "bm25", "regex", "heuristic", "model_name", "llm_confidence")


def _name(base: str = "Testco") -> str:
    return f"{base} {uuid.uuid4().hex[:8]}"  # isolate tests sharing one database


def _analyse(client, name: str, year: int = 2026, **kw) -> dict:
    """Create + run an assessment with no sources (fast; every answer becomes a data gap)."""
    r = client.post("/api/assessments", json={"company_name": name, "year": year, "enable_web_search": False, **kw})
    assert r.status_code == 201, r.text
    a = r.json()
    assert client.post(f"/api/assessments/{a['id']}/run").status_code == 202
    st = client.get(f"/api/assessments/{a['id']}/status").json()
    assert st["status"] == "COMPLETED", st
    return a


def _lookup(client, company: str, year: int = 2026, **params) -> dict:
    r = client.get("/api/assessments/lookup", params={"company": company, "year": year, **params})
    assert r.status_code == 200, r.text
    return r.json()


def test_normalization():
    assert normalize_company_name("Google") == normalize_company_name("google") == normalize_company_name("GOOGLE") \
        == normalize_company_name("Google Inc.") == normalize_company_name("Google LLC") == "google"
    assert normalize_company_name("Nordvik Components AB") == "nordvik components"
    assert normalize_company_name("Google") != normalize_company_name("Googleplex")
    assert normalize_org_number("556012-5790") == normalize_org_number("556012 5790") == "5560125790"
    assert normalize_org_number("  ") is None


def test_existing_company_and_year_returns_saved_assessment(client):
    base = _name("Google")
    a = _analyse(client, base)
    for variant in (base, base.upper(), base.lower(), f"{base} Inc.", f"  {base}  "):
        res = _lookup(client, variant)
        assert res["found"] and not res["ambiguous"] and res["completed"]["id"] == a["id"], variant
    saved = res["completed"]
    assert saved["status"] == "COMPLETED" and saved["completed_at"] and saved["total"] == 16 and saved["reviewed"] == 0
    # a different year is a different assessment
    assert _lookup(client, base, 2025)["found"] is False
    # a similar but different name is not reused
    assert _lookup(client, base + "plex")["found"] is False


def test_loading_saved_assessment_does_not_rerun(client):
    a = _analyse(client, _name())
    events = len(client.get(f"/api/assessments/{a['id']}/audit").json())
    for _ in range(2):
        s = client.get(f"/api/assessments/{a['id']}").json()
        assert s["assessment"]["status"] == "COMPLETED" and s["total"] == 16
        client.get(f"/api/assessments/{a['id']}/recommendations")
    after = client.get(f"/api/assessments/{a['id']}/audit").json()
    assert len(after) == events and not any(e["event_type"] == "analysis_started" for e in after[events:])


def test_duplicate_completed_assessment_requires_confirmation(client):
    name = _name()
    a = _analyse(client, name)
    dup = client.post("/api/assessments", json={"company_name": name.upper(), "year": 2026, "enable_web_search": False})
    assert dup.status_code == 409 and dup.json()["detail"]["assessment_id"] == a["id"]
    # other year: allowed without confirmation
    assert client.post("/api/assessments", json={"company_name": name, "year": 2025, "enable_web_search": False}).status_code == 201
    ok = client.post("/api/assessments", json={"company_name": name, "year": 2026, "enable_web_search": False, "confirm_new": True})
    assert ok.status_code == 201
    new = ok.json()
    assert new["assessment_version"] == 2 and new["supersedes_id"] == a["id"]
    # until the new version completes, the saved one stays current
    assert _lookup(client, name)["completed"]["id"] == a["id"]
    assert _lookup(client, name)["in_progress"]["id"] == new["id"]
    client.post(f"/api/assessments/{new['id']}/run")
    assert _lookup(client, name)["completed"]["id"] == new["id"]


def test_rerun_creates_new_version_and_archives_previous(client):
    name = _name()
    a = _analyse(client, name)
    recs = client.get(f"/api/assessments/{a['id']}/recommendations").json()
    client.post(f"/api/recommendations/{recs[0]['id']}/review", json={"action": "APPROVE"})

    r = client.post(f"/api/assessments/{a['id']}/rerun")
    assert r.status_code == 202, r.text
    new = r.json()
    assert new["assessment_version"] == 2 and new["supersedes_id"] == a["id"]
    assert client.get(f"/api/assessments/{new['id']}/status").json()["status"] == "COMPLETED"

    old = client.get(f"/api/assessments/{a['id']}").json()["assessment"]
    assert old["superseded_by_id"] == new["id"]
    # previous result and its analyst decision are archived, not deleted
    assert client.get(f"/api/recommendations/{recs[0]['id']}").json()["review_status"] == "APPROVED"
    assert _lookup(client, name)["completed"]["id"] == new["id"]
    listed = [x["id"] for x in client.get("/api/assessments", params={"q": name}).json()]
    assert listed == [new["id"]]
    archived = [x["id"] for x in client.get("/api/assessments", params={"q": name, "include_archived": True}).json()]
    assert set(archived) == {a["id"], new["id"]}
    # an archived version cannot be re-run
    assert client.post(f"/api/assessments/{a['id']}/rerun").status_code == 409
    events = {e["event_type"] for e in client.get(f"/api/assessments/{a['id']}/audit").json()}
    assert {"new_analysis_requested", "assessment_replaced"} <= events


def test_duration_and_timestamps_stored(client):
    a = _analyse(client, _name())
    st = client.get(f"/api/assessments/{a['id']}/status").json()
    assert st["started_at"] and st["completed_at"] and st["server_time"]
    assert st["duration_seconds"] is not None and st["duration_seconds"] >= 0
    assert st["stages"][-1]["label"] == "Preparing assessment" and all(s["state"] == "done" for s in st["stages"])
    s = client.get(f"/api/assessments/{a['id']}").json()["assessment"]
    assert s["duration_seconds"] == st["duration_seconds"]


def test_saved_assessment_list_and_search(client):
    name = _name("Searchable")
    a1 = _analyse(client, name, 2026)
    a2 = _analyse(client, name, 2025)
    rows = client.get("/api/assessments", params={"q": name.lower()}).json()
    assert [r["id"] for r in rows] == [a2["id"], a1["id"]]
    assert {r["year"] for r in rows} == {2025, 2026}
    assert all(r["status"] == "COMPLETED" and r["total"] == 16 and "reviewed" in r for r in rows)
    assert client.get("/api/assessments", params={"q": "zz-no-such-company"}).json() == []


def test_org_number_and_country_prevent_wrong_merges(client):
    name = _name("Acme")
    a = _analyse(client, name, organisation_number="556000-0001", country="Sweden")
    b = _analyse(client, name, organisation_number="556000-0002", country="Sweden")
    # same name, two different companies: never merged automatically
    res = _lookup(client, name)
    assert res["ambiguous"] and {c["id"] for c in res["candidates"]} == {a["id"], b["id"]}
    # organisation number picks the right one, regardless of how the name was typed
    assert _lookup(client, name, organisation_number="5560000002")["completed"]["id"] == b["id"]
    assert _lookup(client, "Something Else", organisation_number="556000 0001")["completed"]["id"] == a["id"]
    # a different country is a different company
    assert _lookup(client, name, organisation_number="556000-0001", country="Norway")["found"] is False


def test_analyst_approval_and_override(client):
    a = _analyse(client, _name())
    recs = client.get(f"/api/assessments/{a['id']}/recommendations").json()
    assert all(r["review_state"] == "no_evidence" for r in recs)  # nothing reviewed -> all gaps
    ok = client.post(f"/api/recommendations/{recs[0]['id']}/review", json={"action": "APPROVE"}).json()
    assert ok["review_state"] == "approved" and ok["final_level"] == ok["recommended_level"]
    ch = client.post(f"/api/recommendations/{recs[1]['id']}/review",
                     json={"action": "OVERRIDE", "selected_level": 2, "comment": "Informal goals seen in dialogue."}).json()
    assert ch["review_state"] == "changed" and ch["final_level"] == 2 and ch["recommended_level"] == 1
    assert ch["analyst_comment"] == "Informal goals seen in dialogue."
    s = client.get(f"/api/assessments/{a['id']}").json()
    assert s["reviewed"] == 2


def test_no_technical_metadata_in_analyst_endpoints(client, sample_pdf):
    set_llm_provider(FakeLLM())
    name = _name("Nordvik")
    r = client.post("/api/assessments", json={"company_name": name, "year": 2026, "enable_web_search": False})
    aid = r.json()["id"]
    with open(sample_pdf, "rb") as f:
        client.post(f"/api/assessments/{aid}/documents", files={"file": ("report.pdf", f, "application/pdf")})
    client.post(f"/api/assessments/{aid}/run")
    recs = client.get(f"/api/assessments/{aid}/recommendations").json()
    bodies = [
        client.get(f"/api/assessments/{aid}").text,
        client.get(f"/api/assessments/{aid}/status").text,
        json.dumps(recs),
        client.get(f"/api/assessments/{aid}/sources").text,
        client.get("/api/assessments/lookup", params={"company": name, "year": 2026}).text,
        client.get("/api/assessments", params={"q": name}).text,
        *(client.get(f"/api/recommendations/{x['id']}").text for x in recs),
        *(client.get(f"/api/assessments/{aid}/export", params={"format": f}).content.decode("utf-8", "ignore")
          for f in ("json", "csv")),
    ]
    for body in bodies:
        low = body.lower()
        for term in FORBIDDEN:
            assert term not in low, (term, body[:300])
    # sources: one row each, no extraction statistics
    src = client.get(f"/api/assessments/{aid}/sources").json()
    assert src[0]["status_label"] == "Processed" and "status_detail" not in src[0] and "discovered_via" not in src[0]


def test_questionnaire_config_and_route(client):
    q = client.get("/api/config/questionnaire").json()
    assert list(q["dimensions"]) == ["reporting", "planning", "execution", "performance"]
    assert q["dimensions"]["reporting"]["levels"]["6"].startswith("The publicly available disclosures")
    assert len(q["dimensions"]["execution"]["levels"]) == 3
    from app.main import _dist

    if _dist.is_dir():  # the built UI serves client-side routes such as /questionnaire
        r = client.get("/questionnaire")
        assert r.status_code == 200 and "<div id=\"root\">" in r.text
        assert client.get("/assessments").status_code == 200


def test_migration_adds_columns_to_existing_database(tmp_path):
    from app.db import session

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE companies (id INTEGER PRIMARY KEY, name VARCHAR(255), normalized_name VARCHAR(255), website VARCHAR(500),
            domain VARCHAR(255), country VARCHAR(100), industry VARCHAR(150), created_at DATETIME);
        CREATE TABLE assessments (id INTEGER PRIMARY KEY, company_id INTEGER, year INTEGER, enable_web_search BOOLEAN,
            status VARCHAR(20), stage INTEGER, stage_message TEXT, progress FLOAT, error TEXT, warnings JSON,
            llm_provider VARCHAR(50), llm_model VARCHAR(255), extraction_mode VARCHAR(20), created_at DATETIME,
            started_at DATETIME, completed_at DATETIME);
        INSERT INTO companies (id, name, normalized_name) VALUES (1, 'Google Inc.', 'google');
        INSERT INTO assessments (id, company_id, year, status, stage, progress, warnings, started_at, completed_at)
            VALUES (1, 1, 2026, 'COMPLETED', 7, 100, '[]', '2026-09-26 15:42:52.000000', '2026-09-26 16:01:50.000000');
    """)
    con.commit()
    con.close()
    original = str(session.engine.url)
    try:
        session.configure_engine(f"sqlite:///{path.as_posix()}")
        session.init_db()
        session.init_db()  # idempotent
        con = sqlite3.connect(path)
        row = con.execute("SELECT normalized_company_name, assessment_version, duration_seconds, superseded_by_id "
                          "FROM assessments WHERE id = 1").fetchone()
        cols = {r[1] for r in con.execute("PRAGMA table_info(companies)")}
        con.close()
        assert row[0] == "google" and row[1] == 1 and row[2] == pytest.approx(1138.0) and row[3] is None
        assert "organisation_number" in cols
    finally:
        session.configure_engine(original)


def test_delete_saved_assessment_with_versions_and_files(client, sample_pdf):
    from pathlib import Path

    from app.db.session import SessionLocal
    from app.models.entities import AnalystDecision, AuditEvent, Evidence, Recommendation, Source

    name = _name("Deleteco")
    r = client.post("/api/assessments", json={"company_name": name, "year": 2026, "enable_web_search": False})
    aid = r.json()["id"]
    with open(sample_pdf, "rb") as f:
        client.post(f"/api/assessments/{aid}/documents", files={"file": ("report.pdf", f, "application/pdf")})
    client.post(f"/api/assessments/{aid}/run")
    rec = client.get(f"/api/assessments/{aid}/recommendations").json()[0]
    client.post(f"/api/recommendations/{rec['id']}/review", json={"action": "APPROVE"})
    new_id = client.post(f"/api/assessments/{aid}/rerun").json()["id"]  # v2 reuses the uploaded PDF
    other = _analyse(client, name, 2025)  # same company, other year: must survive
    with SessionLocal() as db:
        pdf = Path(db.scalar(Source.__table__.select().with_only_columns(Source.file_path).where(Source.assessment_id == aid)))
    assert pdf.is_file()

    assert client.delete(f"/api/assessments/{new_id}").status_code == 204
    for i in (aid, new_id):
        assert client.get(f"/api/assessments/{i}").status_code == 404
    assert client.get(f"/api/recommendations/{rec['id']}").status_code == 404
    assert _lookup(client, name)["found"] is False
    assert [x["id"] for x in client.get("/api/assessments", params={"q": name, "include_archived": True}).json()] == [other["id"]]
    assert not pdf.exists()  # no longer referenced by any source
    with SessionLocal() as db:
        for model in (Recommendation, Evidence, AnalystDecision, Source):
            assert db.query(model).filter(model.assessment_id.in_([aid, new_id])).count() == 0
        assert db.query(AuditEvent).filter(AuditEvent.assessment_id.in_([aid, new_id])).count() == 0
        log = db.query(AuditEvent).filter(AuditEvent.event_type == "assessment_deleted", AuditEvent.entity_id == new_id).one()
        assert log.assessment_id is None and set(log.details["deleted_assessment_ids"]) == {aid, new_id}
    assert client.get(f"/api/assessments/{other['id']}").status_code == 200
    assert client.delete(f"/api/assessments/{aid}").status_code == 404


def test_delete_blocked_while_running(client):
    from app.db.session import SessionLocal
    from app.models.entities import Assessment

    a = _analyse(client, _name())
    with SessionLocal() as db:
        db.get(Assessment, a["id"]).status = "RUNNING"
        db.commit()
    assert client.delete(f"/api/assessments/{a['id']}").status_code == 409
    with SessionLocal() as db:
        db.get(Assessment, a["id"]).status = "COMPLETED"
        db.commit()
    assert client.delete(f"/api/assessments/{a['id']}").status_code == 204


def test_export_branding(client):
    import io

    from openpyxl import load_workbook

    a = _analyse(client, _name("Brandco"))
    j = client.get(f"/api/assessments/{a['id']}/export", params={"format": "json"})
    assert j.json()["report"] == {"title": "Swedbank ESG Compass", "subtitle": "Evidence-based sustainability assessment"}
    assert 'filename="swedbank-esg-compass-brandco-' in j.headers["content-disposition"]
    wb = load_workbook(io.BytesIO(client.get(f"/api/assessments/{a['id']}/export", params={"format": "xlsx"}).content))
    assert wb["Assessment"]["A1"].value == "Swedbank ESG Compass" and wb.properties.title.startswith("Swedbank ESG Compass")
    csv_text = client.get(f"/api/assessments/{a['id']}/export", params={"format": "csv"}).content.decode("utf-8-sig")
    assert "Swedbank ESG Compass" not in csv_text  # data rows do not repeat the product name


def test_app_works_with_and_without_logo_asset(client):
    from app.main import _dist

    if not _dist.is_dir():
        pytest.skip("frontend not built")
    logo = _dist / "swedbank-logo.svg"
    existed = logo.exists()
    original = logo.read_bytes() if existed else None
    try:
        if existed:
            logo.unlink()
        # absent: the SPA answers with HTML, which the UI treats as "no logo" (text-only branding)
        r = client.get("/swedbank-logo.svg")
        assert r.status_code == 200 and "image" not in r.headers["content-type"]
        assert "<title>Swedbank ESG Compass</title>" in client.get("/").text
        # present: served as an image
        logo.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>', encoding="utf-8")
        r = client.get("/swedbank-logo.svg")
        assert r.status_code == 200 and r.headers["content-type"].startswith("image/svg")
        assert client.get("/assessments").status_code == 200
    finally:
        if existed:
            logo.write_bytes(original)
        elif logo.exists():
            logo.unlink()
