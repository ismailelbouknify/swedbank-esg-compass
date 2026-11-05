"""Selects the configured local LLM provider. Never raises at startup."""
from __future__ import annotations

import logging

from app.core.config import get_settings
from app.services.llm.base import LLMProvider, NullLLMProvider
from app.services.llm.llama_cpp_provider import LlamaCppProvider
from app.services.llm.ollama_provider import OllamaProvider
from app.services.security.url_validator import is_loopback_service_url

log = logging.getLogger(__name__)

_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        s = get_settings()
        name = s.llm_provider.lower()
        if name == "llama_cpp":
            _provider = LlamaCppProvider(s.resolved_model_path, n_ctx=s.llm_context_tokens, n_threads=s.llm_threads)
        elif name == "ollama" and s.bank_safe_mode and not is_loopback_service_url(s.ollama_base_url):
            # restricted mode never sends report text off this machine
            log.warning("llm_provider_blocked: provider=ollama reason=endpoint_not_local")
            _provider = NullLLMProvider()
        elif name == "ollama":
            _provider = OllamaProvider(s.ollama_base_url, s.ollama_model, timeout=s.llm_timeout_seconds, num_ctx=s.llm_context_tokens)
        else:
            _provider = NullLLMProvider()
    return _provider


def set_llm_provider(provider: LLMProvider | None) -> None:
    """Override the provider (tests) or reset it (None) so settings are re-read."""
    global _provider
    _provider = provider
