"""FastAPI application entry point: `uvicorn app.main:app --reload` (from backend/)."""
from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import assessments, recommendations, sources, system
from app.config.loader import load_rules
from app.core.config import PROJECT_DIR, get_settings
from app.db.session import SessionLocal, init_db
from app.models.entities import Assessment
from app.services.llm.factory import get_llm_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("esg")
# httpx logs every request URL at INFO, which would include query strings and API keys
for _noisy in ("httpx", "httpcore", "primp"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_UPLOAD_PATH = re.compile(r"^/api/assessments/\d+/documents$")
_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
_SECURITY_HEADERS = {"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer"}
_HTML_CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
             "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")


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
    s = get_settings()
    log.info("Security configuration: restricted_mode=%s web_search=%s external_fetch=%s http_allowed=%s "
             "store_external_files=%s domain_allowlist=%d",
             s.bank_safe_mode, s.enable_web_search, s.enable_external_fetch, s.http_allowed,
             s.store_external_files, len(s.allowed_domains))
    if s.enable_playwright:
        log.warning("ENABLE_PLAYWRIGHT is set, but this version contains no browser fetcher: it has no effect.")
    st = get_llm_provider().status()  # never raises: a missing model must not stop the app
    (log.info if st.available else log.warning)("LLM status [%s]: %s", st.provider, st.message)
    yield


_settings = get_settings()
# the interactive API docs load scripts from a public CDN, so restricted mode leaves them off
_api_docs = {"docs_url": None, "redoc_url": None, "openapi_url": None} if _settings.bank_safe_mode else {}
app = FastAPI(title="ESG AI Assessment Platform (local MVP)", version="0.1.0", lifespan=lifespan, **_api_docs)


@app.middleware("http")
async def _security_guards(request: Request, call_next):
    if request.method == "POST" and _UPLOAD_PATH.match(request.url.path):
        # refuse before the multipart parser spools the body to disk
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > get_settings().max_pdf_size_mb * 1024 * 1024 + _MULTIPART_OVERHEAD_BYTES:
            return JSONResponse({"detail": f"File exceeds {get_settings().max_pdf_size_mb} MB limit"}, status_code=413)
    response = await call_next(request)
    for name, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers.setdefault("Content-Security-Policy", _HTML_CSP)
    return response


if _settings.bank_safe_mode:
    # a web page in the analyst's browser must not reach this server through a rebinding hostname
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=_settings.allowed_host_list)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _settings.cors_origins.split(",") if o.strip()],
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
