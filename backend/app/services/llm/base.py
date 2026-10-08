"""Common interface for local LLM providers. The rest of the system only talks to this.

No cloud LLM providers exist in this codebase by design."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

LLM_NOT_CONFIGURED_MESSAGE = (
    "Local LLM model is not configured. Place the GGUF model in models/ or configure an Ollama model."
)


class LLMUnavailableError(RuntimeError):
    pass


@dataclass
class LLMStatus:
    available: bool
    provider: str
    model: str | None
    message: str


class LLMProvider(ABC):
    provider_name: str = "base"

    @property
    @abstractmethod
    def model_name(self) -> str | None:
        """Model identifier stored with every piece of LLM-generated evidence."""

    @abstractmethod
    def status(self) -> LLMStatus:
        """Cheap availability check. Must never raise."""

    def is_available(self) -> bool:
        return self.status().available

    @abstractmethod
    def complete_json(self, system: str, user: str, schema: dict[str, Any] | None = None, max_tokens: int = 384) -> str:
        """Return the raw model output for a JSON-producing prompt (temperature 0).

        Providers should constrain output to JSON where the backend supports it.
        Raises LLMUnavailableError if the model cannot be used."""


class NullLLMProvider(LLMProvider):
    provider_name = "none"

    def __init__(self, reason: str = LLM_NOT_CONFIGURED_MESSAGE):
        self.reason = reason

    @property
    def model_name(self) -> str | None:
        return None

    def status(self) -> LLMStatus:
        return LLMStatus(False, self.provider_name, None, self.reason)

    def complete_json(self, system: str, user: str, schema: dict[str, Any] | None = None, max_tokens: int = 384) -> str:
        raise LLMUnavailableError(self.reason)
