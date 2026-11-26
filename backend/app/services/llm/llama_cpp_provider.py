"""In-process CPU inference over a GGUF model via llama-cpp-python (optional dependency)."""
from __future__ import annotations

import logging
import threading
import time
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
        self._timed_out = False  # True after a timeout; prevents concurrent use of stuck context
        # performance diagnostics: model loads and the latest call (tokens and timings)
        self.loads = 0
        self.load_seconds = 0.0
        self.runtime: dict[str, Any] = {}
        self.last_call: dict[str, Any] = {}

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
            started = time.perf_counter()
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=0,  # CPU only
                verbose=False,
            )
            self.loads += 1
            self.load_seconds += time.perf_counter() - started
            cp = getattr(self._llm, "context_params", None)
            self.runtime = {"n_ctx": self._llm.n_ctx(), "n_threads": getattr(cp, "n_threads", None),
                            "n_threads_batch": getattr(cp, "n_threads_batch", None), "n_batch": getattr(cp, "n_batch", None),
                            "load_seconds": round(self.load_seconds, 1), "loads": self.loads}
            log.info("llm_loaded: %s", self.runtime)
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            raise LLMUnavailableError(f"Model failed to load: {exc}") from exc
        return self._llm

    def complete_json(self, system: str, user: str, schema: dict[str, Any] | None = None, max_tokens: int = 384) -> str:
        """Generate a JSON completion.

        Uses streaming (stream=True) so the timeout is checked at every token boundary.
        When the deadline is reached the generator is closed — no further C++ computation
        is queued, so inference truly stops at the next token handoff rather than leaving
        an orphaned thread running to max_tokens.
        """
        from app.core.config import get_settings

        timeout = get_settings().llm_timeout_seconds

        with self._lock:  # llama.cpp contexts are not thread-safe; only one call at a time
            if self._timed_out:
                raise LLMUnavailableError(
                    "LLM timed out on a previous call; using heuristic fallback for this assessment."
                )
            llm = self._get()
            response_format: dict[str, Any] = {"type": "json_object"}
            if schema:
                response_format["schema"] = schema

            # stream=True yields one token per next() call so we can check the deadline
            # between tokens.  gen.close() stops the Python generator; the C++ side has
            # already returned from the previous token decode, so no orphaned inference.
            gen = llm.create_chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format=response_format,
                stream=True,
            )
            deadline = time.monotonic() + timeout
            parts: list[str] = []
            started = time.perf_counter()
            first_token_at: float | None = None
            output_tokens = 0
            finish_reason = None
            try:
                for chunk in gen:
                    if self.abort_check is not None and self.abort_check():
                        gen.close()
                        raise LLMUnavailableError("Stopped on request")
                    if time.monotonic() > deadline:
                        gen.close()
                        self._timed_out = True
                        log.error("LlamaCpp: inference timed out after %ss", timeout)
                        raise LLMUnavailableError(f"LLM call timed out after {timeout}s")
                    choice = chunk["choices"][0]
                    content = (choice.get("delta") or {}).get("content") or ""
                    finish_reason = choice.get("finish_reason") or finish_reason
                    if content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        output_tokens += 1  # one streamed chunk per generated token
                        parts.append(content)
            except LLMUnavailableError:
                raise
            except Exception as exc:
                raise LLMUnavailableError(f"LLM inference error: {exc}") from exc

            self._timed_out = False  # clear so a transient hang doesn't permanently disable the LLM
            ended = time.perf_counter()
            prompt_tokens = len(llm.tokenize((system + "\n" + user).encode("utf-8"), add_bos=True))
            prompt_s = (first_token_at or ended) - started  # prompt processing (+ the first token)
            gen_s = ended - (first_token_at or ended)
            self.last_call = {
                "prompt_tokens": prompt_tokens, "output_tokens": output_tokens, "max_tokens": max_tokens,
                "truncated": finish_reason == "length", "grammar": bool(schema),
                "prompt_seconds": round(prompt_s, 2), "generation_seconds": round(gen_s, 2),
                "prompt_tokens_per_second": round(prompt_tokens / prompt_s, 1) if prompt_s > 0 else None,
                "generation_tokens_per_second": round((output_tokens - 1) / gen_s, 1) if gen_s > 0 and output_tokens > 1 else None,
            }
            log.info("llm_call: %s", self.last_call)
        return "".join(parts)
