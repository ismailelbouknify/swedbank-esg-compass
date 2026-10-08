"""Local inference through an Ollama server running on this machine (e.g. llama3.2:1b)."""
from __future__ import annotations

import time
from typing import Any

import httpx

from app.services.llm.base import LLMProvider, LLMStatus, LLMUnavailableError


class OllamaProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: int = 180, num_ctx: int = 4096):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
        self._status_cache: tuple[float, LLMStatus] | None = None

    @property
    def model_name(self) -> str | None:
        return f"ollama:{self.model}"

    def status(self) -> LLMStatus:
        now = time.monotonic()
        if self._status_cache and now - self._status_cache[0] < 30:
            return self._status_cache[1]
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            resp.raise_for_status()
            names = {m.get("name", "") for m in resp.json().get("models", [])}
            wanted = self.model if ":" in self.model else f"{self.model}:latest"
            if self.model in names or wanted in names:
                st = LLMStatus(True, self.provider_name, self.model_name, f"Ollama model {self.model} ready.")
            else:
                st = LLMStatus(
                    False, self.provider_name, self.model_name,
                    f"Ollama is running but model '{self.model}' is not pulled. Run: ollama pull {self.model}",
                )
        except Exception:  # noqa: BLE001
            st = LLMStatus(
                False, self.provider_name, self.model_name,
                f"Local LLM model is not configured. Ollama is not reachable at {self.base_url}. "
                "Start Ollama or place the GGUF model in models/ and set LLM_PROVIDER=llama_cpp.",
            )
        self._status_cache = (now, st)
        return st

    def complete_json(self, system: str, user: str, schema: dict[str, Any] | None = None, max_tokens: int = 384) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema or "json",
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": self.num_ctx},
        }
        try:
            resp = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            self._status_cache = None
            raise LLMUnavailableError(f"Ollama request failed: {exc}") from exc
        return resp.json().get("message", {}).get("content", "")
