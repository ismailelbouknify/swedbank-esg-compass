"""Tolerant parsing of small-model JSON output."""
from __future__ import annotations

import json
import re
from typing import Any


class LLMOutputError(ValueError):
    pass


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_llm_json(raw: str) -> dict[str, Any]:
    """Extract the first JSON object from model output, repairing common small-model slips."""
    if not raw or not raw.strip():
        raise LLMOutputError("Empty model output")
    text = re.sub(r"```(?:json)?", "", raw).strip()
    candidate = _first_json_object(text)
    if candidate is None:
        raise LLMOutputError(f"No JSON object in model output: {raw[:200]!r}")
    for attempt in (candidate, _repair(candidate)):
        try:
            data = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise LLMOutputError(f"Invalid JSON in model output: {candidate[:200]!r}")


def _repair(s: str) -> str:
    s = re.sub(r",\s*([}\]])", r"\1", s)  # trailing commas
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    return s


def as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def as_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:[.,]\d+)?", v.replace(" ", ""))
        if m:
            return float(m.group(0).replace(",", "."))
    return None


def as_year(v: Any) -> int | None:
    f = as_float(v)
    if f is None:
        return None
    y = int(f)
    return y if 1990 <= y <= 2100 else None


def as_confidence(v: Any) -> float:
    f = as_float(v)
    if f is None:
        return 0.5
    if f > 1:
        f = f / 100
    return max(0.0, min(1.0, f))
