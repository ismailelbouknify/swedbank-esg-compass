"""In-process CPU inference over a GGUF model via llama-cpp-python (optional dependency)."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from app.services.llm.base import LLM_NOT_CONFIGURED_MESSAGE, LLMProvider, LLMStatus, LLMUnavailableError

log = logging.getLogger(__name__)


class LlamaCppProvider(LLMProvider):
    provider_name = "llama_cpp"

    def __init__(self, model_path: Path, n_ctx: int = 4096, n_threads: int = 0):
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads or None
        self._llm = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    @property
    def model_name(self) -> str | None:
        return f"llama_cpp:{self.model_path.name}"

    def _library_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            return False
        return True

    def status(self) -> LLMStatus:
        if not self.model_path.is_file():
            return LLMStatus(False, self.provider_name, None, f"{LLM_NOT_CONFIGURED_MESSAGE} (expected file: {self.model_path})")
        if not self._library_available():
            return LLMStatus(
                False,
                self.provider_name,
                self.model_name,
                "llama-cpp-python is not installed. Run: pip install -r requirements-llm.txt "
                "(or set LLM_PROVIDER=ollama).",
            )
        if self._load_error:
            return LLMStatus(False, self.provider_name, self.model_name, f"Model failed to load: {self._load_error}")
        return LLMStatus(True, self.provider_name, self.model_name, "Local GGUF model ready (loaded on first use).")

    def _get(self):
        if self._llm is not None:
            return self._llm
        st = self.status()
        if not st.available:
            raise LLMUnavailableError(st.message)
        from llama_cpp import Llama

        try:
            log.info("Loading GGUF model %s", self.model_path)
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=0,  # CPU only
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            raise LLMUnavailableError(f"Model failed to load: {exc}") from exc
        return self._llm

    def complete_json(self, system: str, user: str, schema: dict[str, Any] | None = None, max_tokens: int = 384) -> str:
        with self._lock:  # llama.cpp contexts are not thread-safe
            llm = self._get()
            response_format: dict[str, Any] = {"type": "json_object"}
            if schema:
                response_format["schema"] = schema
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        return out["choices"][0]["message"]["content"] or ""
