"""SQLAlchemy engine/session. Only DATABASE_URL needs to change for PostgreSQL."""
from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


def _make_engine(url: str):
    kwargs = {}
    if url.startswith("sqlite"):
        # the analysis runs in a background thread while the UI polls status
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    eng = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(eng, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # WAL: readers do not block on the pipeline's writes
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

    return eng


engine = _make_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def configure_engine(url: str) -> None:
    """Rebind the global engine (used by tests)."""
    global engine
    engine = _make_engine(url)
    SessionLocal.configure(bind=engine)


def init_db() -> None:
    from app.models import entities  # noqa: F401  (register models)

    Base.metadata.create_all(bind=engine)
    _migrate()


# Columns added after the first release. create_all() never alters existing tables, so an existing
# database gets them here (additive only; no data is dropped).
_ADDED_COLUMNS = {
    "companies": {"organisation_number": "VARCHAR(50)"},
    "assessments": {
        "normalized_company_name": "VARCHAR(255)",
        "assessment_version": "INTEGER DEFAULT 1",
        "supersedes_id": "INTEGER",
        "superseded_by_id": "INTEGER",
        "duration_seconds": "FLOAT",
    },
}


def _migrate() -> None:
    from datetime import datetime

    from sqlalchemy import inspect, text

    from app.services.company.identify import normalize_company_name

    existing = {t: {c["name"] for c in inspect(engine).get_columns(t)} for t in _ADDED_COLUMNS}
    with engine.begin() as conn:
        for table, cols in _ADDED_COLUMNS.items():
            for name, ddl in cols.items():
                if name not in existing[table]:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_assessments_identity_year ON assessments (normalized_company_name, year)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_assessments_superseded_by_id ON assessments (superseded_by_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_companies_organisation_number ON companies (organisation_number)"))
        # backfill identity + duration for assessments created before these columns existed
        rows = conn.execute(text(
            "SELECT a.id, c.name FROM assessments a JOIN companies c ON c.id = a.company_id WHERE a.normalized_company_name IS NULL"
        )).all()
        for aid, name in rows:
            conn.execute(text("UPDATE assessments SET normalized_company_name = :n WHERE id = :id"),
                         {"n": normalize_company_name(name), "id": aid})
        conn.execute(text("UPDATE assessments SET assessment_version = 1 WHERE assessment_version IS NULL"))
        for aid, started, completed in conn.execute(text(
            "SELECT id, started_at, completed_at FROM assessments "
            "WHERE duration_seconds IS NULL AND status = 'COMPLETED' AND started_at IS NOT NULL AND completed_at IS NOT NULL"
        )).all():
            s, c = (v if isinstance(v, datetime) else datetime.fromisoformat(str(v)) for v in (started, completed))
            conn.execute(text("UPDATE assessments SET duration_seconds = :d WHERE id = :id"),
                         {"d": max(0.0, (c.replace(tzinfo=None) - s.replace(tzinfo=None)).total_seconds()), "id": aid})


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
