"""Health, local-LLM status and questionnaire configuration."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.config.loader import load_rules
from app.core.config import get_settings
from app.schemas.api import LLMStatusOut
from app.services.llm.factory import get_llm_provider

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/llm/status", response_model=LLMStatusOut)
def llm_status() -> LLMStatusOut:
    st = get_llm_provider().status()
    return LLMStatusOut(available=st.available, provider=st.provider, model=st.model, message=st.message,
                        heuristic_fallback=get_settings().allow_heuristic_fallback)


@router.get("/config/questionnaire")
def questionnaire() -> dict[str, Any]:
    rules = load_rules()
    return {
        "version": rules["version"],
        "lookback_years": rules["lookback_years"],
        "dimensions": {k: {"label": v["label"], "scope": v["scope"], "max_level": v["max_level"], "question": v["question"],
                           "levels": v["levels"]} for k, v in rules["dimensions"].items()},
        "factors": [{"key": f["key"], "label": f["label"], "short_label": f.get("short_label", f["label"]),
                     "pillar": f.get("pillar")} for f in rules["factors"]],
    }
