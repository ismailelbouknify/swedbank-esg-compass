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

    # --- Restricted security mode ----------------------------------------
    # Security-hardened configuration for controlled environments; see the README. It reduces
    # technical risk but does not replace organisational security review or malware scanning.
    bank_safe_mode: bool = True
    enable_web_search: bool = True  # False: no request ever reaches a search provider
    enable_external_fetch: bool = True  # False: no external page or document is retrieved
    enable_playwright: bool = Field(False, validation_alias=AliasChoices("enable_playwright", "use_playwright"))
    allow_http: bool = False  # only read in restricted mode; outside it plain HTTP stays possible
    store_downloaded_files: bool = False  # ignored in restricted mode: external PDFs are never stored
    allowed_external_domains: str = ""  # comma-separated; empty = any host passing the public-web checks
    allowed_domains_include_subdomains: bool = True
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
    max_download_mb: int = 20  # any automatically retrieved document
    max_html_size_mb: int = 5
    max_pdf_pages: int = 300
    max_downloaded_pdfs: int = 3
    max_redirects: int = 3
    max_http_retries: int = 2
    http_connect_timeout_seconds: float = 10
    http_read_timeout_seconds: float = 30
    max_source_processing_seconds: float = 60  # one source, from first request to extracted text

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
    heuristic_chunks_per_question: int = 10
    retrieval_candidates: int = 30  # BM25 candidates re-ranked by a cheap regex signal
    max_llm_context_chars: int = 6000  # ~1500 tokens of passage text

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def http_allowed(self) -> bool:
        return self.allow_http or not self.bank_safe_mode

    @property
    def store_external_files(self) -> bool:
        return self.store_downloaded_files and not self.bank_safe_mode

    @property
    def allowed_domains(self) -> tuple[str, ...]:
        entries = (d.strip().lower().removeprefix("*.").strip(".") for d in self.allowed_external_domains.split(","))
        return tuple(d for d in entries if d)

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

    @property
    def download_dir(self) -> Path:
        d = Path(self.data_dir) / "downloads"
        d.mkdir(parents=True, exist_ok=True)
        return d


@lru_cache
def get_settings() -> Settings:
    return Settings()
