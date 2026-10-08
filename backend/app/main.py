"""FastAPI application entry point: `uvicorn app.main:app --reload` (from backend/)."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import assessments, recommendations, sources, system
from app.config.loader import load_rules
from app.core.config import PROJECT_DIR, get_settings
from app.db.session import SessionLocal, init_db
from app.models.entities import Assessment
from app.services.llm.factory import get_llm_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("esg")


def _recover_interrupted_runs() -> None:
    """Background runs do not survive a restart; mark them failed so they can be re-run."""
    with SessionLocal() as db:
        for a in db.query(Assessment).filter(Assessment.status == "RUNNING"):
            a.status = "FAILED"
            a.error = "Server restarted while the analysis was running. Start the analysis again."
        db.commit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_rules()  # fail fast on a broken questionnaire config
    _recover_interrupted_runs()
    st = get_llm_provider().status()  # never raises: a missing model must not stop the app
    (log.info if st.available else log.warning)("LLM status [%s]: %s", st.provider, st.message)
    yield


app = FastAPI(title="ESG AI Assessment Platform (local MVP)", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in get_settings().cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(system.router)
app.include_router(assessments.router)
app.include_router(recommendations.router)
app.include_router(sources.router)

# Serve the built frontend (frontend/dist) if present, so one process can run everything.
_dist = PROJECT_DIR / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(404, "Not found")
        candidate = (_dist / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(_dist.resolve()):
            return FileResponse(candidate)
        return FileResponse(_dist / "index.html")
