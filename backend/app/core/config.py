"""Application settings, loaded from environment variables / backend/.env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    database_url: str = f"sqlite:///{(BACKEND_DIR / 'esg.db').as_posix()}"
    data_dir: Path = BACKEND_DIR / "data"

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
    max_pdf_size_mb: int = 50
    max_html_size_mb: int = 5
    max_downloaded_pdfs: int = 3
    http_timeout_seconds: int = 20
    max_redirects: int = 5
    use_playwright: bool = False

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
