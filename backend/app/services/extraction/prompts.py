"""Constrained prompts + JSON schemas for the small local model. Each prompt carries one
short passage (a single chunk, ~200-400 tokens), so a full call stays well under 2.5k tokens."""
from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = (
    "You extract facts from short excerpts of company sustainability documents. "
    "Use ONLY the TEXT provided. Do not use outside knowledge. Never guess. "
    "If something is not explicitly stated in the TEXT, use null (or false for yes/no fields). "
    "evidence_quote must be copied word-for-word from the TEXT. Output a single JSON object only."
)

_NUM = {"type": ["number", "null"]}
_INT = {"type": ["integer", "null"]}
_STR = {"type": ["string", "null"]}
_BOOL = {"type": "boolean"}
_CONF = {"type": "number"}


def _schema(props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(props.keys())}


PLANNING_SCHEMA = _schema({
    "goal_found": _BOOL, "formalized": _BOOL, "quantitative": _BOOL,
    "target_value": _NUM, "target_unit": _STR, "target_year": _INT, "baseline_year": _INT,
    "clear_owner": _BOOL, "owner": _STR, "benchmark_found": _BOOL, "benchmark_type": _STR,
    "evidence_quote": {"type": "string"}, "confidence": _CONF,
})

EXECUTION_SCHEMA = _schema({
    "activity_found": _BOOL, "activity_description": _STR, "activity_year": _INT,
    "linked_to_goal": _BOOL, "evidence_quote": {"type": "string"}, "confidence": _CONF,
})

PERFORMANCE_SCHEMA = _schema({
    "metric_found": _BOOL, "metric_name": _STR, "unit": _STR,
    "values": {"type": "array", "items": {"type": "object", "properties": {"year": {"type": "integer"}, "value": {"type": "number"}}, "required": ["year", "value"]}},
    "improvement_stated_without_numbers": _BOOL,
    "sector_comparison": _STR, "sector_comparison_favorable": {"type": ["boolean", "null"]},
    "evidence_quote": {"type": "string"}, "confidence": _CONF,
})

ASSURANCE_SCHEMA = _schema({
    "sustainability_assurance": _BOOL, "assurance_provider": _STR, "assurance_standard": _STR,
    "assurance_level": _STR, "evidence_quote": {"type": "string"}, "confidence": _CONF,
})


def _header(factor_label: str, dimension_label: str, source: str, text: str) -> str:
    return f'FACTOR:\n{factor_label}\n\nDIMENSION:\n{dimension_label}\n\nSOURCE:\n{source}\n\nTEXT:\n"""\n{text}\n"""\n\n'


def planning_prompt(factor_label: str, source: str, text: str) -> str:
    return _header(factor_label, "Planning (goals, targets, strategy, plans)", source, text) + (
        f"Does the TEXT state a goal, target, strategy or plan related to {factor_label}? Extract factual information only.\n"
        "Return JSON with keys:\n"
        "goal_found: true only if the TEXT states a goal/target/strategy/plan for this factor\n"
        "formalized: true if it is a documented goal, target, policy or strategy (not a vague wish)\n"
        "quantitative: true only if the goal contains a number\n"
        "target_value: the number or null; target_unit: e.g. \"%\" or null\n"
        "target_year: the deadline year or null; baseline_year: the baseline/comparison year or null\n"
        "clear_owner: true only if the TEXT names who is responsible (a person, board, committee, function); owner: that name or null\n"
        "benchmark_found: true only if the TEXT explicitly compares the goal with an external benchmark "
        "(science-based targets/SBTi, Paris Agreement, EU Taxonomy, peers or sector average, national targets, best available technology); "
        "a large number alone is NOT a benchmark\n"
        "benchmark_type: which benchmark or null\n"
        "evidence_quote: the exact sentence from the TEXT\n"
        "confidence: 0 to 1"
    )


def execution_prompt(factor_label: str, source: str, text: str) -> str:
    return _header(factor_label, "Execution (activities actually carried out)", source, text) + (
        f"Does the TEXT describe an activity the company has actually carried out to manage {factor_label}? "
        "Plans and intentions for the future do not count.\n"
        "Return JSON with keys:\n"
        "activity_found: true/false\n"
        "activity_description: short description or null\n"
        "activity_year: year the activity took place if stated, else null\n"
        "linked_to_goal: true only if the TEXT says the activity serves a stated goal, target or strategy\n"
        "evidence_quote: the exact sentence from the TEXT\n"
        "confidence: 0 to 1"
    )


def performance_prompt(factor_label: str, source: str, text: str) -> str:
    return _header(factor_label, "Performance improvement (measured results over time)", source, text) + (
        f"Does the TEXT report measured values over several years for a {factor_label} indicator?\n"
        "Return JSON with keys:\n"
        "metric_found: true/false\n"
        "metric_name: the indicator name or null; unit: the unit or null\n"
        "values: list of {\"year\": year, \"value\": number} exactly as stated in the TEXT (empty list if none)\n"
        "improvement_stated_without_numbers: true if the TEXT claims improvement but gives no numbers\n"
        "sector_comparison: exact phrase comparing performance to a sector/industry average, benchmark or standard, else null\n"
        "sector_comparison_favorable: true if that comparison says the company performs better, false if worse, null if none\n"
        "evidence_quote: the exact sentence from the TEXT\n"
        "confidence: 0 to 1"
    )


def assurance_prompt(source: str, text: str) -> str:
    return _header("Sustainability reporting", "Reporting (independent assurance)", source, text) + (
        "Does the TEXT state that an independent third party assured or verified the company's sustainability / "
        "non-financial information? An audit of the financial statements only does NOT count.\n"
        "Return JSON with keys:\n"
        "sustainability_assurance: true/false\n"
        "assurance_provider: firm name or null\n"
        "assurance_standard: e.g. ISAE 3000, AA1000 or null\n"
        "assurance_level: \"limited\", \"reasonable\" or null\n"
        "evidence_quote: the exact sentence from the TEXT\n"
        "confidence: 0 to 1"
    )
