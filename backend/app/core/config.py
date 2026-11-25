"""Application settings, loaded from environment variables / backend/.env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore", populate_by_name=True)

    database_url: str = f"sqlite:///{(BACKEND_DIR / 'esg.db').as_posix()}"
    data_dir: Path = BACKEND_DIR / "data"

    # --- Security --------------------------------------------------------
    # Web access: normal public HTTP/HTTPS is allowed; localhost, private, link-local and cloud
    # metadata addresses are always blocked (also after redirects). Automatically retrieved documents
    # are held in memory only and never written to disk. BANK_SAFE_MODE only hardens the local web
    # server (Host header check, no API docs, local-only Ollama); it does not restrict web access.
    bank_safe_mode: bool = True
    enable_web_search: bool = True  # False: no request ever reaches a search provider
    enable_external_fetch: bool = True  # False: no external page or document is retrieved
    enable_playwright: bool = Field(False, validation_alias=AliasChoices("enable_playwright", "use_playwright"))
    malware_scan_required: bool = False  # fail closed when no scanner is registered
    allowed_hosts: str = "localhost,127.0.0.1"  # Host headers the web server answers in restricted mode

    # --- Local LLM -------------------------------------------------------
    llm_provider: str = "llama_cpp"  # llama_cpp | ollama | none
    local_llm_model_path: str = str(PROJECT_DIR / "models" / "model.gguf")
    llm_context_tokens: int = 4096
    llm_threads: int = 0  # 0 = let llama.cpp decide
    llm_max_output_tokens: int = 384
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    llm_timeout_seconds: int = 180
    # When no local LLM is configured, fall back to deterministic keyword/regex
    # fact extraction (clearly labelled, lower confidence) instead of failing.
    allow_heuristic_fallback: bool = True

    # --- Search / web ----------------------------------------------------
    search_provider: str = "ddgs"  # ddgs | serpapi | none
    serpapi_api_key: str = ""
    max_search_results: int = 10
    max_pages_per_site: int = 10
    max_pdf_size_mb: int = 50  # manual uploads
    max_download_mb: int = 50  # any automatically retrieved document, held in memory
    max_html_size_mb: int = 5
    max_pdf_pages: int = 500
    max_downloaded_pdfs: int = 3
    max_redirects: int = 5
    max_http_retries: int = 2
    http_connect_timeout_seconds: float = 10
    http_read_timeout_seconds: float = 30
    max_source_processing_seconds: float = 120  # one source, from first request to extracted text
    search_timeout_seconds: float = 60  # one search query, including every engine the provider tries

    # --- Network environment (proxy / certificates) ----------------------
    # Only a proxy that the user or IT already configured is used; certificates are always verified.
    # system: HTTPS_PROXY/HTTP_PROXY/NO_PROXY, else the Windows Internet Options proxy (PAC scripts are not evaluated)
    # explicit: WEB_PROXY_URL | none: direct connections only
    web_proxy_mode: str = "system"
    web_proxy_url: str = ""  # explicit mode; may hold IT-issued proxy credentials, which are never logged
    use_system_trust_store: bool = True  # trust the certificate authorities installed in the operating system
    web_ca_bundle: str = ""  # PEM file with the organisation's certificate authorities (takes precedence)
    # Stop automatic web access for the rest of an assessment after this many network-level failures in a row
    # with nothing ever succeeding (fetches and searches counted separately). 0 = never stop early.
    web_access_failure_limit: int = 3

    # --- Source coverage / recovery search -------------------------------
    coverage_min_valid_sources: int = 3
    coverage_max_factors_without_candidates: float = 0.6
    coverage_max_company_rejection_share: float = 0.5
    coverage_min_report_page_chars: int = 5000  # below this, a "report" web page only lists reports
    recovery_search_enabled: bool = True
    recovery_max_queries: int = 10
    recovery_max_pages: int = 5
    recovery_max_pdfs: int = 2
    recovery_time_budget_seconds: float = 180
    recovery_search_pause_seconds: float = 1.0

    # --- Retrieval / extraction ------------------------------------------
    chunk_size_words: int = 180
    chunk_overlap_words: int = 40
    # Passages sent to extraction per factor x dimension question. Each can cost one LLM call,
    # so keep this small for CPU inference; the keyword fallback can afford more.
    max_chunks_per_question: int = 4
    # Hard limit on model calls per assessment (~24 s each on CPU). Passages are handed out round-robin over all
    # factor x dimension questions, so the best passage of every question is read first; anything beyond the
    # limit goes to the deterministic extractor and is verified like everything else. 0 = no limit.
    max_llm_calls_per_assessment: int = 100
    heuristic_chunks_per_question: int = 10
    retrieval_candidates: int = 30  # BM25 candidates re-ranked by a cheap regex signal
    max_llm_context_chars: int = 6000  # ~1500 tokens of passage text

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def allowed_host_list(self) -> list[str]:
        return [h.strip() for h in self.allowed_hosts.split(",") if h.strip()]

    @property
    def resolved_model_path(self) -> Path:
        p = Path(self.local_llm_model_path)
        if not p.is_absolute():
            p = (BACKEND_DIR / p).resolve()
        return p

    @property
    def upload_dir(self) -> Path:
        d = Path(self.data_dir) / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d


@lru_cache
def get_settings() -> Settings:
    return Settings()
