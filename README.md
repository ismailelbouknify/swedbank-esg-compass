# ESG AI Assessment Platform — local MVP

A local web application that prepares **evidence-backed recommended answers** to an ESG ambition questionnaire
(Reporting, and per material factor Planning / Execution / Performance Improvement) for a company and year.
Analysts review every recommendation, approve or override it, and export the result.

The key design principle:

> **LLM = "What facts exist?"   Python rules = "Which questionnaire answer do those facts correspond to?"**

A small local model (~1B, CPU-only) only reads short passages and returns structured facts. It never picks a level.
Deterministic Python rules map verified facts to levels, and every level is traceable to source, page and quote.

---

## 1. What it does

1. You enter a company name and assessment year (plus optionally a sustainability report PDF, website, country, industry).
2. **Sources**: the uploaded PDF is the primary source. Without one, a controlled web search finds sustainability /
   annual / ESG reports and a few official web pages (not a crawl). A given website is always read (homepage + a few
   ESG-relevant links).
3. **Documents** are extracted page-by-page (PyMuPDF), tables enriched with pdfplumber where useful, split into
   page-aware chunks.
4. **Retrieval**: BM25 + factor/dimension keyword dictionaries select a few relevant passages per question; a cheap
   regex signal re-ranks them. Whole reports are never sent to the model.
5. **Extraction**: regexes handle the trivial (years, percentages, frameworks, assurance wording, KPI tables); the local
   LLM interprets context in 300–1500-token prompts and returns JSON (goal found? formalised? quantitative? target
   year? owner? benchmark?). Python then cross-checks every LLM claim (a benchmark/owner must be explicitly worded in
   the source, "public" comes from source metadata, not from the model).
6. **Verification**: each evidence item is checked — quote exists on the stated page, numbers/years in the facts appear
   in the source text, relevant to the factor and to the dimension. Failures are marked `REQUIRES_REVIEW` and never
   used to raise a level.
7. **Rules** (`backend/app/services/scoring/`) compute the recommended level, triggered rule, supporting evidence IDs,
   conditions met / not found, what is missing for the next level, and a confidence category.
8. **Review**: the analyst approves or overrides (level + mandatory comment). Everything is in an audit trail.
9. **Export**: JSON, CSV, Excel (Assessment / Evidence / Sources / Analyst Decisions sheets).

"Not found" is never presented as "does not exist": gaps read *"No evidence of … was found in the reviewed sources."*
Recommendation statuses: `FOUND`, `NO_EVIDENCE_FOUND`, `CONFLICTING_EVIDENCE`, `REQUIRES_REVIEW`.

The methodology (level definitions) follows the ESG-76398 questionnaire and lives in
`backend/app/config/esg_rules.json`.

## 2. Architecture

```
PDF upload ─┐
Website  ───┼─► Sources ─► Documents/pages ─► chunks ─► BM25 + re-rank ─► passages (per factor × dimension)
Web search ─┘                                                                     │
                                          regex facts ◄──────────────────────────┤
                                          local 1B LLM facts (JSON) ◄────────────┘
                                                      │
                                   Verification (quote/page/numbers/relevance)
                                                      │
                               Deterministic rules (reporting/planning/execution/performance)
                                                      │
                         Recommendation + evidence IDs + confidence ─► Analyst review ─► Export
```

```
ESG/
  backend/
    app/
      main.py                 FastAPI app (also serves frontend/dist if built)
      api/                    REST endpoints
      config/esg_rules.json   questionnaire: factors, levels, keywords, metrics, rule parameters
      core/config.py          settings (env / backend/.env)
      db/, models/            SQLAlchemy (SQLite; PostgreSQL via DATABASE_URL)
      schemas/                Pydantic models
      services/
        company/              name normalisation, official-source classification
        search/               SearchProvider interface: DDGS, SerpAPI
        documents/            PDF (PyMuPDF), tables (pdfplumber), HTML (trafilatura/BS4), safe fetcher, OCR stub
        retrieval/            chunker, BM25
        llm/                  base.py, llama_cpp_provider.py, ollama_provider.py, JSON parsing
        extraction/           regex patterns, prompts, FactExtractor
        verification/         evidence verification
        scoring/              reporting.py, planning.py, execution.py, performance.py, confidence.py
        pipeline/             discovery, ingestion, runner (background job)
        export/               JSON / CSV / XLSX
    tests/
  frontend/                   React + TypeScript + Vite
  models/README.md            where to put the GGUF model
  sample_data/                synthetic (fictional) report + PDF builder
```

Background work uses FastAPI `BackgroundTasks` in-process with progress stored on the assessment row
(no Celery/Redis/Kafka).

## 3. Requirements

- Python 3.11+
- Node.js 20+ (only to develop/build the frontend)
- CPU only. 8 GB RAM works with a 1B Q4 model; 16 GB recommended.
- Optional: `llama-cpp-python` or [Ollama](https://ollama.com) for the local LLM.

## 4. Backend installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional; defaults work
```

## 5. Frontend installation

```bash
cd frontend
npm install
```

## 6. Model installation

See [`models/README.md`](models/README.md). In short: put a Llama 3.2 1B Instruct Q4 GGUF at `models/model.gguf` and
`pip install -r requirements-llm.txt` (or use the prebuilt CPU wheel index), **or** `ollama pull llama3.2:1b` and set
`LLM_PROVIDER=ollama`. Model files are never downloaded automatically.

## 7. Environment variables

All in `backend/.env` (see `backend/.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///backend/esg.db` | SQLAlchemy URL (PostgreSQL works) |
| `LLM_PROVIDER` | `llama_cpp` | `llama_cpp`, `ollama` or `none` |
| `LOCAL_LLM_MODEL_PATH` | `../models/model.gguf` | GGUF path (relative to `backend/`) |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | `http://localhost:11434` / `llama3.2:1b` | Ollama settings |
| `ALLOW_HEURISTIC_FALLBACK` | `true` | analyse with keyword/regex extraction when no model |
| `SEARCH_PROVIDER` | `ddgs` | `ddgs` (no key), `serpapi` (+`SERPAPI_API_KEY`), `none` |
| `MAX_SEARCH_RESULTS` | `10` | results downloaded from search |
| `MAX_PAGES_PER_SITE` | `10` | HTML pages fetched per assessment |
| `MAX_DOWNLOADED_PDFS` | `3` | PDFs downloaded per assessment |
| `MAX_PDF_SIZE_MB` | `50` | upload & download limit |
| `MAX_CHUNKS_PER_QUESTION` | `4` | passages per factor × dimension sent to the LLM |

## 8. How to run

Development (two terminals):

```bash
# terminal 1
cd backend
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000   # http://127.0.0.1:8000

# terminal 2
cd frontend
npm run dev                        # http://localhost:5173  (proxies /api to :8000)
```

Single process: `cd frontend && npm run build`, then run only uvicorn and open http://127.0.0.1:8000.

Bind to `127.0.0.1` as above rather than `0.0.0.0` unless network access is intentionally required. The application has no
authentication, so anyone who can reach the port can use it. The interactive API docs (`/docs`) load scripts from a public
CDN and are only available with `BANK_SAFE_MODE=false`.

Pages: `/` new assessment · `/assessments` saved assessments · `/assessments/{id}` result (tabs: Assessment,
Sources, Audit) · `/assessments/{id}/progress` running analysis with timer · `/assessments/{id}/report` print/PDF
layout · `/questionnaire` questionnaire example.

Saved assessments are reused: before analysing, the UI calls `GET /api/assessments/lookup?company=&year=`
(normalized name, or organisation number when given). A completed match opens instantly; "Run new analysis"
(`POST /api/assessments/{id}/rerun`, or `POST /api/assessments` with `confirm_new: true`) creates a new version
that replaces the saved one only when it completes. `started_at`, `completed_at` and `duration_seconds` are stored
by the backend. The analyst UI and exports omit technical metadata (model, extraction method, rule ids, internal
scores); it stays in the database and in `GET /api/assessments/{id}/evidence`.

Deleting: "Delete" on the Assessments page (`DELETE /api/assessments/{id}`, confirmation required) permanently
removes the assessment with all its versions, evidence, analyst decisions and audit events, plus uploaded PDFs no
other assessment uses. A running analysis cannot be deleted. An `assessment_deleted` audit record (company, year,
ids) is kept.

Branding: the UI is "Swedbank ESG Compass — Evidence-based sustainability assessment" (strings in
`frontend/src/brand.ts`). The header and print report show `frontend/public/swedbank-logo.svg` when that file exists
(add the approved asset and rebuild); otherwise they use text-only branding. The app never downloads or draws the logo.

Demo: build the synthetic PDF (`python sample_data/build_sample_pdf.py` with the backend venv) and upload
`sample_data/nordvik_sustainability_report_2025.pdf` with company "Nordvik Components AB", year 2025.

## 9. Tests

```bash
cd backend
pytest               # 73 tests; the LLM is mocked — no model needed
cd ../frontend
npm run lint         # TypeScript type-check
npm run build
npm test             # branding present, old product name gone, no model/technical terms (UI + bundle)
```

Covered: PDF extraction (incl. OCR_REQUIRED), chunking, BM25, regex extraction (years, %, tables, from/to changes,
frameworks, assurance), all four rule engines, no-evidence wording, confidence, LLM JSON parsing, hallucination
handling with a mocked LLM, URL/SSRF validation, upload validation, API creation, analyst override, full pipeline
runs (fallback and mocked-LLM), exports.

## 10. Limitations

- **Keyword fallback ≠ LLM.** Without a model, facts come from regex/keyword rules: decent on well-structured reports,
  noisy on others (e.g. a table row may be mapped to a loosely matching metric). All such results are capped at Medium
  confidence and flagged *Requires review*.
- The LLM path is covered by tests with a mocked model; quality with a real 1B model depends on the model — small
  models paraphrase and miss nuance, which is why every claim is re-verified in Python.
- CPU speed: roughly 5–20 s per LLM call; an assessment makes up to ~60 calls (prefilters skip most passages).
- Levels 2–3 of Planning (informal / internal goals) normally need dialogue with the company; public documents
  rarely support them. Level 6 benchmarks and Performance level 3 rely on the company's own comparison statements.
- "Last three years" windows use the assessment year; undated activities fall back to the report year.
- Reporting level 2 ("regulatory only") vs. 3 is hard to distinguish from public text.
- No OCR (image-only PDFs are marked `OCR_REQUIRED`); no JavaScript rendering unless Playwright is added.
- Web discovery depends on DuckDuckGo availability and may pick up pages about similarly-named companies (they are
  ranked lower and labelled by source type). Some sites block automated downloads (recorded as `FAILED`).
- The General Section questions of the reference questionnaire (exclusion list, position statements) and its free-text
  comment are not automated.
- SQLite + in-process background tasks: runs interrupted by a server restart are marked failed and can be re-run.
- Single-user MVP: no authentication.

## 11. Privacy / local-AI design

- No cloud LLM integration exists in the code. Inference runs in-process (llama.cpp) or via a local Ollama server.
- Uploaded documents stay in `backend/data/`. The only outbound traffic is web search and retrieval of public pages,
  which you can disable per assessment (checkbox) or for the installation (see below).
- Search providers receive the company name and search terms, never document content.
- Only short passages are sent to the model; every model output is stored with the model name for audit.

### Source coverage

Each answer has one of three coverage states: `EVIDENCE_FOUND`, `NO_EVIDENCE_FOUND` (enough valid company sources were reviewed and
nothing relevant was accepted) or `INSUFFICIENT_SOURCE_COVERAGE` (too few reliable sources to conclude anything; no level is recorded,
and the analyst is asked to upload a report). Coverage counts sources that pass the company identity gate: one report document, or
`COVERAGE_MIN_VALID_SOURCES` official sources that are about sustainability. Third-party pages never make coverage sufficient. When coverage is thin, one bounded recovery search runs (`RECOVERY_*` settings). Its sources pass
through the same gates. Search queries contain the company name and search terms only. `recovery_search_enabled=false` turns it off.

### Security configuration

`BANK_SAFE_MODE=true` (the default) is a restricted security mode: a security-hardened configuration intended to support
controlled environments. **It reduces technical risk but does not replace organisational security approval or the malware
scanning required by the deployment environment.** Final deployment remains subject to organisational IT/security review.
All settings are listed in `backend/.env.example`; the controls live in `backend/app/services/security/`.

External content is always untrusted and is only ever parsed for text, never opened, rendered or executed:

```
Search (if enabled) -> URL + DNS/IP validation -> domain allowlist -> bounded HTTPS request -> validate every redirect
  -> content type + size checks
       HTML: extract text
       PDF:  %PDF- signature -> malware scan hook -> in-memory text extraction -> discard the PDF
  -> company/source validation -> retrieval -> evidence validation -> local LLM -> deterministic scoring
```

| Control | Behaviour | Settings |
| --- | --- | --- |
| Manual upload only | No search and no retrieval requests are made; analyses use uploaded PDFs | `ENABLE_WEB_SEARCH=false`, `ENABLE_EXTERNAL_FETCH=false` |
| URL policy | HTTPS only; no credentials, non-standard ports, local names or numeric host tricks | `ALLOW_HTTP` |
| Network targets | Every address a host resolves to must be public (private, loopback, link-local, metadata, multicast and reserved ranges are refused, for IPv4, IPv6 and IPv4 embedded in IPv6). The connection is pinned to the validated address | |
| Domain allowlist | Optional; applies to every redirect hop too | `ALLOWED_EXTERNAL_DOMAINS`, `ALLOWED_DOMAINS_INCLUDE_SUBDOMAINS` |
| Redirects | Followed one at a time, each validated like a new URL | `MAX_REDIRECTS` |
| Size and pages | Limit checked from `Content-Length` and again while streaming. Over-limit PDFs are rejected, not truncated; such a report can be reviewed, then uploaded manually if it fits the upload limits | `MAX_DOWNLOAD_MB`, `MAX_HTML_SIZE_MB`, `MAX_PDF_PAGES`, `MAX_PDF_SIZE_MB` (uploads) |
| Time | Connect and read timeouts, a total limit per source, bounded retries with backoff | `HTTP_CONNECT_TIMEOUT_SECONDS`, `HTTP_READ_TIMEOUT_SECONDS`, `MAX_SOURCE_PROCESSING_SECONDS`, `MAX_HTTP_RETRIES` |
| External PDFs | Held in memory only, released after text extraction, never written to disk or to the database. Only metadata and extracted text are kept. Storing them needs `BANK_SAFE_MODE=false` and `STORE_DOWNLOADED_FILES=true` | `STORE_DOWNLOADED_FILES` |
| Uploads | PDF only: extension, content type, signature, size, page count, encryption. Stored under generated names | `MAX_PDF_SIZE_MB`, `MAX_PDF_PAGES` |
| Local LLM | All retrieval, extraction, validation and scoring run locally. No cloud LLM exists in the code, and a non-local Ollama endpoint is refused in this mode | `LLM_PROVIDER`, `OLLAMA_BASE_URL` |
| Web server | Answers only the configured Host names, sends security headers, no CDN-backed API docs | `ALLOWED_HOSTS` |

A rejected or unavailable source never stops an assessment. It is recorded (`sources.status`, `sources.fetch_outcome`, audit
events such as `private_ip_rejected`, `redirect_rejected`, `download_size_limit_exceeded`, `invalid_pdf_signature`) and the
analyst sees only "Source could not be processed", plus a notice that a gap may reflect an unavailable source rather than
missing company activity. Logs hold hostnames, codes, durations and sizes, not document text or prompts.

**Malware scanning.** No scanner is included, and PyMuPDF parsing is not a replacement for one. Register an
organisation-approved scanner with `app.services.security.content_validator.set_content_scanner()`; content must then pass
before it is parsed. `MALWARE_SCAN_REQUIRED=true` rejects all content until a scanner is registered. SHA-256 hashes are kept
for duplicate detection and audit and say nothing about whether a file is safe.

**Browser automation.** This version contains no browser fetcher, and `ENABLE_PLAYWRIGHT` (default `false`) only records
permission through `browser_automation_allowed()`. Any future fallback must require that flag and: validate the initial URL
and every navigation with `validate_external_url`, block `file://` and downloads, set strict navigation timeouts and a
redirect limit, block images, media and fonts, use a fresh context without stored profiles or cookies, and always close the
page, context and browser.

**Outbound proxies.** Requests connect directly and ignore proxy environment variables, so traffic cannot be rerouted silently.
An installation that must use an outbound or TLS-inspecting proxy needs integration work, or can run in manual-upload-only mode.

### Dependency checks

`backend/requirements.lock` records the exact versions last tested; install them with
`pip install -r requirements.txt -c requirements.lock`, and regenerate it with `pip freeze` after a deliberate upgrade. The frontend
is locked by `frontend/package-lock.json` (`npm ci`). To check for known vulnerabilities, which does not prove that a package
is secure:

```bash
cd backend && python -m pip install pip-audit && python -m pip_audit -r requirements.lock
cd frontend && npm audit
```

## 12. Adding a new ESG factor

Add an entry to `factors` in `backend/app/config/esg_rules.json`:

```json
{
  "key": "water_management",
  "label": "Water & Wastewater Management",
  "pillar": "E",
  "keywords": ["water", "water withdrawal", "water consumption", "wastewater", "m3", "water stress"],
  "metrics": [
    {"name": "water intensity", "keywords": ["water intensity", "m3 per", "water per unit"], "better": "lower"}
  ]
}
```

That is all: retrieval queries, extraction prompts, scoring, UI table and exports pick it up automatically
(`keywords` drive retrieval and relevance checks; `metrics` map KPI rows to a direction for improvement calculations).

## 13. Modifying scoring rules

- Level texts, search terms and rule parameters: `backend/app/config/esg_rules.json`
  (`dimensions.*.levels`, `reporting_rules`, `planning_rules.benchmark_terms`, `planning_rules.require_clear_ownership_for_level_5`,
  `performance_rules.sector_leading_terms`, `lookback_years`, `confidence.weights`/`thresholds`, source priorities).
- Rule logic: `backend/app/services/scoring/` — each dimension has a pure rule table
  (`decide_reporting_level`, `decide_planning_level`, `decide_execution_level`, `evaluate_series`) plus a scorer that
  aggregates verified evidence. Scorers take structured facts, never LLM prose. Add a test in
  `backend/tests/test_scoring.py` for every rule change.

Methodology choices made for this MVP (documented so they can be changed):

- Planning level 5 does not *require* evidenced ownership by default (the questionnaire mentions it); missing ownership
  is shown as "?" with a note. Set `require_clear_ownership_for_level_5: true` to enforce it.
- Planning level 6 needs explicit benchmark wording (SBTi, Paris, EU Taxonomy, peers/sector average, …) in the goal
  statement or on the same page — a large number alone is never "ambitious".
- Execution level 3 requires both an activity explicitly linked to goals/strategy **and** a formalised goal found by
  Planning.
- Performance improvement is computed in Python from reported values (earliest vs latest within the last three years);
  unquantified claims are recorded but never scored. Performance level 3 is always flagged for review.
- Reporting level 6 requires framework-based reporting (level 5) plus independent assurance of sustainability
  information (a financial audit alone does not count).
