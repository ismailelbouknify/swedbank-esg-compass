# ESG AI Assessment Platform — local MVP

A local web application that prepares **evidence-backed recommended answers** to an ESG ambition questionnaire
(Reporting, and per material factor Planning / Execution / Performance Improvement) for a company and year.
Analysts review every recommendation, approve or change it, and export the result.

The design principle is a simple, evidence-grounded RAG:

> **No supporting evidence = no generated answer.**
> The local model reads the questionnaire criteria and the retrieved passages and recommends a level with exact
> quotations. Python verifies the references and the response's integrity. The analyst makes the final decision.

A small local model (~1B, CPU-only) answers one questionnaire question per call from a handful of retrieved passages.
Every recommended level carries quotations that were found, word for word, in the company's own sources; an answer
whose quotations cannot be verified is withheld. Nothing is ever converted into a low level for lack of evidence.

---

## 1. What it does

1. You enter a company name and assessment year (plus optionally a sustainability report PDF, website, country, industry).
2. **Sources**: the uploaded PDF is the primary source. Without one, a controlled web search finds sustainability /
   annual / ESG reports and a few official web pages (not a crawl). A given website is always read: homepage, a few
   ESG-relevant links, then report, ESG, investor and archive pages and report PDFs found through the site's own
   navigation and sitemaps (robots.txt respected, fixed limits). When search is unavailable or thin, the same is done
   for the official domain the identity check settled. Search and website retrieval are independent: a search service
   that declines automated searches stops searching only (see "Source coverage").
3. **Identity**: every source is checked against the assessed company (name, organisation number, official domain);
   another company's document never reaches the model.
4. **Documents** are extracted page-by-page (PyMuPDF), tables enriched with pdfplumber, and split into overlapping
   chunks of about 350 tokens (60 overlap). Each chunk keeps its source, page, section heading, report year and table
   context; a table split across chunks repeats its header row, so figures stay next to their labels, units and years.
5. **Retrieval**: for each of the 16 questions, BM25 ranks all chunks of the company's accepted sources against a query
   built from the question, the factor and the level definitions. Near-duplicates are removed and the best 6–8
   passages that fit the model's context window are kept. No passage is discarded for lacking a predefined keyword.
6. **Answering**: one local-model call per question with the company, year, factor, dimension, the exact question,
   the complete level definitions from the questionnaire, and the passages with their chunk IDs. The model returns
   JSON: `EVIDENCE_FOUND` with a level, an explanation and exact quotations with chunk IDs, or `INSUFFICIENT_EVIDENCE`.
   With llama.cpp the output is grammar-constrained to exactly two structures: a supported answer (`EVIDENCE_FOUND`,
   an integer level of the question's scale, an explanation, 1–3 quotations citing given chunk IDs) or an unsupported
   answer (`INSUFFICIENT_EVIDENCE`, level `null`, an explanation, no quotations). Status, level and evidence are
   coupled, so "evidence found" without a level, or a level without a quotation, cannot be generated.
7. **Verification** (generic, no ESG rules): valid JSON; level within the question's scale; no level without
   quotations; every chunk ID is one of the passages given; every quotation is in the cited chunk (only whitespace and
   typographic normalisation); the chunk is from an accepted source of the company (a third-party or group source
   must name the company). A failed answer gets one corrective retry; if it fails again, no level is recommended and
   the question is flagged for review with the suggestion shown as unverified.
8. **Review**: the analyst approves the AI-recommended level, or sets a level with a mandatory comment. AI and analyst
   levels are kept and shown separately. Everything is in an audit trail.
9. **Export**: JSON, CSV, Excel (Assessment / Evidence / Data Gaps / Sources / Analyst Decisions sheets).

Each answer is one of: a level with verified quotations (`EVIDENCE_FOUND`); `NO_EVIDENCE_FOUND` (adequate company
sources were reviewed and nothing supported a level); `INSUFFICIENT_SOURCE_COVERAGE` (the sources were missing,
inaccessible or inadequate); or `REQUIRES_REVIEW` (an answer was withheld because it could not be verified or the
model could not run). The last three record no level ("Insufficient evidence to recommend an assessment level.");
a Level 1 is only ever recommended when quoted evidence supports it.

The methodology (questions, level definitions, scales) follows the ESG-76398 questionnaire and lives, unchanged, in
`backend/app/config/esg_rules.json`. The model applies it; Python does not.

## 2. Architecture

```
PDF upload ─┐
Website  ───┼─► Sources ─► identity check ─► text + tables ─► chunks (~350 tokens, page/section/table context)
Web search ─┘                                                        │
                                       per question: BM25 over accepted sources ─► 6–8 passages (fit the context)
                                                        │
                       local 1B model: questionnaire criteria + passages ─► JSON level, explanation, quotations
                                                        │
          verification: JSON · scale · chunk IDs · quotations in chunk · company source  (one corrective retry)
                                                        │
        verified level + quotations │ no level: no evidence / insufficient coverage / withheld ─► Analyst ─► Export
```

```
ESG/
  backend/
    app/
      main.py                 FastAPI app (also serves frontend/dist if built)
      api/                    REST endpoints
      config/esg_rules.json   questionnaire: factors, questions, level definitions, scales, retrieval vocabulary
      core/config.py          settings (env / backend/.env)
      db/, models/            SQLAlchemy (SQLite; PostgreSQL via DATABASE_URL)
      schemas/                Pydantic models
      services/
        company/              name normalisation, company identity, official-source classification
        search/               SearchProvider interface: DDGS, DuckDuckGo lite, SerpAPI
        security/             safe fetcher, PAC/Kerberos proxy, TLS, robots.txt, SSRF checks
        documents/            PDF (PyMuPDF), tables (pdfplumber), HTML (trafilatura/BS4), text helpers
        retrieval/            chunker, BM25
        rag/                  questions, retrieval, prompt, verify, answer   <- the assessment method
        llm/                  base.py, llama_cpp_provider.py, ollama_provider.py, JSON parsing
        pipeline/             discovery, ingestion, identity gate, source coverage, runner (background job)
        export/               JSON / CSV / XLSX
    tests/
  frontend/                   React + TypeScript + Vite
  models/README.md            where to put the GGUF model
  sample_data/                synthetic (fictional) report + PDF builder
```

Background work uses FastAPI `BackgroundTasks` in-process with progress stored on the assessment row
(no Celery/Redis/Kafka).

### Measured comparison

`docs/simple-rag-benchmark.md` compares baseline-v2.4 with this design on two fixed documents (same model, questions and
laptop): coverage, every recommended level, quotations, model calls, time and memory, plus a labelled manual review of
answer correctness. In short: the corrected RAG answers more questions with verified quotations and never invents a
level for missing evidence, but with the 1B model many of those answers rest on a genuine quotation that does not
support them, and each assessment takes longer (larger prompts). More answers is not the same as more correct answers.

### Removed in the simple-RAG redesign

The deterministic scoring engine (`services/scoring/`: reporting, planning, execution, performance rule tables and
confidence), the fact extractor with its pre-filters and per-factor/actor/state/table gates (`services/extraction/`,
785 lines of ESG regex patterns), the multi-stage evidence verifier (`services/verification/`), the deterministic
quote fallback, the keyword/regex "heuristic mode" without a model, rule-based level overrides and the confidence
score. The `*_rules` and `confidence` sections of `esg_rules.json` are no longer read (left in place so the
questionnaire file is unchanged).

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
| `ALLOW_HEURISTIC_FALLBACK` | `true` | without a model, still gather and list sources (no level is recommended) instead of failing |
| `SEARCH_PROVIDER` | `ddgs` | `ddgs` (no key), `duckduckgo_lite` (no key, application transport), `serpapi` (+`SERPAPI_API_KEY`), `none` |
| `MAX_SEARCH_RESULTS` | `10` | results downloaded from search |
| `MAX_PAGES_PER_SITE` | `10` | HTML pages fetched per assessment |
| `WEBSITE_DISCOVERY_ENABLED` | `true` | read the official site's navigation and sitemaps |
| `WEBSITE_MAX_PAGES` / `WEBSITE_MAX_PDFS` | `8` / `3` | pages / report PDFs read in that stage |
| `SITEMAP_MAX_FILES` | `3` | sitemap files read per site (an index counts as one) |
| `SEARCH_MIN_INTERVAL_SECONDS` | `5` | pacing of keyless DuckDuckGo searches, process-wide |
| `SEARCH_REFUSAL_COOLDOWN_SECONDS` | `900` | no searches for this long after the service declined |
| `MAX_DOWNLOADED_PDFS` | `3` | PDFs downloaded per assessment |
| `MAX_PDF_SIZE_MB` | `100` | upload limit (automatic retrieval: `MAX_DOWNLOAD_MB`, `100`) |
| `MAX_PDF_PAGES` | `0` | page-count limit; `0` = none |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `350` / `60` | chunk size and overlap (approximate model tokens) |
| `RAG_CHUNKS_PER_QUESTION` | `7` | passages per question (fewer if they do not fit `LLM_CONTEXT_TOKENS`) |
| `LLM_CONTEXT_TOKENS` / `LLM_MAX_OUTPUT_TOKENS` | `4096` / `400` | model context window / answer length |
| `MAX_LLM_CALLS_PER_ASSESSMENT` | `100` | safety ceiling on model calls (normal use: 16, plus at most one retry each) |

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
pytest               # the model is replaced by a deterministic test double — no model needed
cd ../frontend
npm run lint         # TypeScript type-check
npm run build
npm test             # branding present, old product name gone, no model/technical terms (UI + bundle)
```

Covered: PDF/HTML extraction (incl. OCR_REQUIRED), chunking (pages, sections, table headers), retrieval (no keyword
filter, near-duplicates, context budget), the prompt (verbatim criteria, untrusted-passage delimiters), verification
(invented quotations, invalid chunk IDs, invalid levels, malformed JSON, multiple quotations, table figures, wrong
company and third-party sources), the corrective retry and call ceiling, the three no-level states, company identity,
search/website/PAC/Kerberos/robots.txt behaviour, analyst review and exports. Tests named `LIMITATION` show what
verification does not guarantee: a genuine quotation can still be attached to a wrong conclusion.

## 10. Limitations

- **A verified quotation is not a verified conclusion.** Python checks that the cited words are in the company's
  sources, not that they meet the level's criteria. A 1B model can over-read a target as "ambitious", take a
  government programme for company activity, or ignore the time window; the analyst decides. Measure semantic
  accuracy against reviewed reference scores before relying on it.
- **Small model.** Llama 3.2 1B on CPU follows the JSON format reliably (grammar-constrained with llama.cpp) but its
  reading of nuanced criteria is limited; it may also abstain where a person would see evidence.
- **Instructions inside sources** cannot forge passages or sources (delimiters are stripped), but a model may still
  follow them; any such answer is a suggestion pending analyst review.
- **Without the model** no level is recommended; sources are still gathered and listed.
- CPU speed: one call per question (16 per assessment, plus retries), each with a ~2–3k-token prompt.
- Levels 2–3 of Planning (informal / internal goals) normally need dialogue with the company; public documents
  rarely support them. Level 6 benchmarks and Performance level 3 rely on the company's own comparison statements.
- Reporting level 2 ("regulatory only") vs. 3 is hard to distinguish from public text.
- No OCR (image-only PDFs are marked `OCR_REQUIRED`); no JavaScript rendering unless Playwright is added.
- Web discovery depends on search availability; keyless DuckDuckGo may decline a sequence of searches (then the
  official website is read directly, see "Source coverage").
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

**Search refusals and the official website.** Keyless DuckDuckGo search has no published limit and answers a burst of
queries with a challenge page: a single query may succeed (as in a diagnostic) while an assessment's sequence is
declined. Searches are therefore paced (`SEARCH_MIN_INTERVAL_SECONDS`); at the first refusal or rate limit
(`search_service_refused`, `search_service_rate_limited`) searching stops for that assessment and pauses for
`SEARCH_REFUSAL_COOLDOWN_SECONDS`, and is never retried around the refusal. A search refusal does not stop website or
PDF retrieval, which has its own failure count. The official website is then read directly: the address given on the
New assessment page, or otherwise the domain the identity check settled (resolved, never ambiguous; a domain is never
guessed). From its homepage, navigation links and sitemaps (robots.txt `Sitemap:` lines, else `/sitemap.xml`; index
files followed to their most relevant children; `.gz` sitemaps skipped), the most relevant report, ESG, investor and
archive pages and report PDFs are read within `WEBSITE_MAX_PAGES` / `WEBSITE_MAX_PDFS`, honouring robots.txt as
every automated request does (see the security table; a robots.txt server error means no exploration). Every document passes the same identity and evidence
gates. The audit event `web_access_diagnostics` separates search failures by cause (`provider_restriction`,
`corporate_network`, `connectivity`), and `official_site_explored` records what the site stage read. An API-based
provider (`SEARCH_PROVIDER=serpapi`) remains available and optional.

### Security configuration

`BANK_SAFE_MODE=true` (the default) is a restricted security mode: a security-hardened configuration intended to support
controlled environments. **It reduces technical risk but does not replace organisational security approval or the malware
scanning required by the deployment environment.** Final deployment remains subject to organisational IT/security review.
All settings are listed in `backend/.env.example`; the controls live in `backend/app/services/security/`.

Web access follows one principle: **allow normal public HTTP/HTTPS access, extract text, and do not persist automatically
retrieved external files.** Any public company website, CDN or filing host can be read without an allowlist; only localhost,
private/internal, link-local and cloud-metadata destinations are blocked. External content is untrusted and is only ever
parsed for text, never opened, rendered or executed:

```
Search (if enabled) -> URL + DNS/IP check (no local/private targets) -> bounded request -> re-check every redirect
  -> size checks; the bytes decide the type (%PDF- signature or HTML markup), not the URL or the declared header
     (only images, media, fonts, archives, executables and scripts are refused unread)
       HTML: extract text in memory
       PDF:  malware scan hook -> text extraction from the in-memory bytes -> bytes discarded
  -> company/source validation -> chunking -> retrieval -> local LLM answer -> lightweight verification -> analyst
```

| Control | Behaviour | Settings |
| --- | --- | --- |
| Manual upload only | No search and no retrieval requests are made; analyses use uploaded PDFs | `ENABLE_WEB_SEARCH=false`, `ENABLE_EXTERNAL_FETCH=false` |
| URL policy | HTTP and HTTPS to any public host; no credentials, non-standard ports, local names or numeric host tricks | |
| Network targets | Every address a host resolves to must be public (private, loopback, link-local, metadata, multicast and reserved ranges are refused, for IPv4, IPv6 and IPv4 embedded in IPv6). A direct connection is pinned to the validated address; through a configured proxy, see "Managed networks" below | |
| Proxy and certificates | Only an already-configured proxy is used (environment variables, Windows settings including the PAC script / WPAD as Windows evaluates it, or an explicit setting). Optional Kerberos sign-in from the Windows session, no stored credentials. Certificates are always verified, against the organisation's bundle or the operating-system store | `WEB_PROXY_MODE`, `WEB_PROXY_URL`, `WEB_PROXY_PAC_URL`, `WEB_PROXY_AUTH`, `USE_SYSTEM_TRUST_STORE`, `WEB_CA_BUNDLE` |
| Early stop | After repeated network-level failures with nothing getting through, automatic access stops for that assessment and the reason is shown | `WEB_ACCESS_FAILURE_LIMIT`, `SEARCH_TIMEOUT_SECONDS` |
| Redirects | Up to 10 (the hop cap also ends redirect loops), to any public host (e.g. corporate site to CDN); each target is checked like a new URL, so a redirect cannot reach a local or internal address. Cookies last for one redirect chain | `MAX_REDIRECTS` |
| robots.txt | Honoured before every automated request, redirect hops included: search results, sitemap entries, links found on pages and report PDFs (RFC 9309: no robots.txt = no restrictions, a 5xx = keep out for now). Read once per origin through the same proxy, sign-in and certificate checks, cached for an hour. Only the first request to the website typed on the New assessment page is exempt, as in a browser. An excluded source is recorded as `REJECTED` with `robots_disallowed` (audit event of the same name); a company report excluded this way is a coverage gap (not assessed), never a low level. A network failure while reading robots.txt is reported as that failure | |
| Size and pages | Configurable resource safeguards, not source restrictions: 100 MB per document, 10 MB per page, no page-count limit by default. Size is checked from `Content-Length` and again while streaming. Over-limit PDFs are rejected, not truncated; such a report can be reviewed, then uploaded manually if it fits the upload limits | `MAX_DOWNLOAD_MB`, `MAX_HTML_SIZE_MB`, `MAX_PDF_PAGES`, `MAX_PDF_SIZE_MB` (uploads) |
| Time | Connect and read timeouts, a total limit per source, bounded retries with backoff | `HTTP_CONNECT_TIMEOUT_SECONDS`, `HTTP_READ_TIMEOUT_SECONDS`, `MAX_SOURCE_PROCESSING_SECONDS`, `MAX_HTTP_RETRIES` |
| External PDFs and pages | Held in memory only, released after text extraction, never written to disk (no temporary files either) or to the database, in any mode. Only the URL, metadata, extracted text and evidence are kept | |
| Uploads | PDF only: extension, content type, signature, size, page count (if `MAX_PDF_PAGES` is set), encryption. Stored under generated names | `MAX_PDF_SIZE_MB`, `MAX_PDF_PAGES` |
| Local LLM | All retrieval, extraction, validation and scoring run locally. No cloud LLM exists in the code, and a non-local Ollama endpoint is refused in this mode | `LLM_PROVIDER`, `OLLAMA_BASE_URL` |
| Web server | Answers only the configured Host names, sends security headers, no CDN-backed API docs. `BANK_SAFE_MODE` governs this and the local-LLM rule only, not web access | `ALLOWED_HOSTS` |

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

**Managed networks (corporate proxy, web filter, TLS inspection).** The application uses the network the way the
organisation set it up, and never works around it:

- *Proxy.* `WEB_PROXY_MODE=system` (default) uses `HTTPS_PROXY`/`HTTP_PROXY`/`NO_PROXY` when set. Otherwise it follows
  Windows Internet Options the way Windows does: the automatic configuration script (`AutoConfigURL`, PAC) or
  "Automatically detect settings" (WPAD), evaluated by Windows itself through WinHTTP for each host, then the static
  proxy and its bypass list. The script's answer is followed as given, including `DIRECT`; the first listed proxy is
  used. If the script cannot be evaluated, the static proxy (or a direct connection) is used, as Windows does, and the
  reason is reported. `WEB_PROXY_PAC_URL` evaluates a different script (e.g. when running under a service account).
  `explicit` uses `WEB_PROXY_URL` for every request; `none` connects directly. The proxy is chosen explicitly per
  request (`.netrc` is never read), and a proxy setting's credentials are never logged. Web search uses the same proxy.
- *Sign-in.* Off by default (`WEB_PROXY_AUTH=none`). With `WEB_PROXY_AUTH=integrated`, approved by IT, a proxy that
  answers `407` is signed in to with a Kerberos ticket for `HTTP/<proxy host>` from the signed-in Windows user's
  session (SSPI, the mechanism browsers use). No username, password or hash is read, asked for or stored; the ticket
  is sent only to the proxy, never to websites, never logged, and a fresh one is used for each connection. NTLM is
  never used (Windows falling back to it is reported instead), and a proxy given only as an IP address cannot be
  signed in to with Kerberos. Page and document retrieval and the SerpAPI search provider sign in this way.
  The DDGS search library uses its own HTTP client (primp), which takes a proxy only as an address and cannot sign in;
  when integrated sign-in applies to the search route, `SEARCH_PROVIDER=ddgs` therefore searches through the
  application's own transport instead, reading DuckDuckGo's lite results page (`duckduckgo_lite`, also selectable
  directly) with the same PAC routing, Kerberos sign-in, certificate and address checks as page retrieval. It
  identifies the application honestly and imitates no browser; if DuckDuckGo answers with a challenge page instead of
  results, that is reported (`search_service_refused`) and not worked around. A proxy that refuses the sign-in, or a page such as Blue Coat's
  "Access Denied (authentication_failed)", is reported and not retried; after `WEB_ACCESS_FAILURE_LIMIT` failures the
  assessment continues with uploaded reports.
- *Certificates.* Always verified. With `USE_SYSTEM_TRUST_STORE=true` (default) the operating-system store is used, so an
  authority that IT installed for HTTPS inspection is trusted as it is in the browser. `WEB_CA_BUNDLE` (a PEM file from IT)
  takes precedence and is also passed to the search client, which does not read the Windows store.
- *SSRF protection through a proxy.* Every URL and redirect target passes the same checks. A name that resolves locally
  must resolve to public addresses only; a name that does not resolve locally (common behind a proxy, which resolves it
  instead) is sent to the proxy by hostname. Local-only names (`localhost`, `.local`, `.internal`, `.corp`, ...), IP
  literals in private ranges, credentials in URLs and non-standard ports are refused before any request either way.
- *Reasons, not waiting.* Proxy sign-in, web-filter blocks, certificate failures, an unreachable proxy, no route to the
  internet and configured size/page limits each produce a plain notice naming the setting involved, and an audit event
  `web_access_diagnostics` with the details (`proxy_authentication_required`, `blocked_by_web_filter`,
  `tls_verification_failed`, `proxy_connection_failed`, `proxy_integrated_signin_unavailable`,
  `proxy_integrated_signin_refused`, ...), plus the PAC state and sign-in mode.
- *Diagnostics.* On the computer where access fails, run `python -m app.network_check [url]` from `backend/`. It makes the
  application's own requests through the same settings and reports the route, the result and what to change. It stores
  nothing and prints no credentials. Website retrieval, single-search connectivity (one query) and sustained search
  (a short sequence paced as in an assessment) are reported separately, since a search service may answer one query
  and decline the next; `--website https://www.company.com/` adds a dry run of official-site discovery (homepage,
  robots.txt, sitemaps, best candidate addresses). `--retrieval-only` / `--search-only` check one component.

When automatic access is refused, upload the company's report on the new-assessment page: the analysis uses it as the
primary source. `ENABLE_WEB_SEARCH=false` and `ENABLE_EXTERNAL_FETCH=false` turn automatic access off entirely.

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

That is all: the factor's three questions, retrieval queries, prompts, UI table and exports pick it up automatically
(`keywords` and `metrics` only steer retrieval).

## 13. Changing the questionnaire or the answering method

- Questions, level definitions and scales: `dimensions.*` in `backend/app/config/esg_rules.json`. They are passed to
  the model verbatim, so a wording change changes what the model is asked to apply.
- Retrieval vocabulary: `dimensions.*.query_terms` and each factor's `keywords`.
- The prompt: `backend/app/services/rag/prompt.py` (system rules, answer format, passage layout).
- Verification: `backend/app/services/rag/verify.py` (generic integrity checks only; do not add ESG rules there — the
  analyst decides whether a verified quotation really meets a level).
- Sizes: `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `RAG_CHUNKS_PER_QUESTION`, `LLM_CONTEXT_TOKENS`,
  `LLM_MAX_OUTPUT_TOKENS`, `MAX_LLM_CALLS_PER_ASSESSMENT` (a ceiling: 16 calls plus at most one retry each).
