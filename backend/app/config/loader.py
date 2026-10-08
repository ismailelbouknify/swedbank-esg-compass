"""Loads the questionnaire / rule configuration (app/config/esg_rules.json)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

RULES_PATH = Path(__file__).with_name("esg_rules.json")

FACTOR_DIMENSIONS = ("planning", "execution", "performance")
REPORTING_FACTOR = "reporting"


@lru_cache
def load_rules(path: str | None = None) -> dict[str, Any]:
    with open(path or RULES_PATH, encoding="utf-8") as f:
        rules = json.load(f)
    _validate(rules)
    return rules


def _validate(rules: dict[str, Any]) -> None:
    keys = set()
    for factor in rules["factors"]:
        for field in ("key", "label", "keywords"):
            if field not in factor:
                raise ValueError(f"Factor missing '{field}': {factor}")
        if factor["key"] in keys or factor["key"] == REPORTING_FACTOR:
            raise ValueError(f"Duplicate or reserved factor key: {factor['key']}")
        keys.add(factor["key"])
        for metric in factor.get("metrics", []):
            if metric.get("better") not in ("lower", "higher"):
                raise ValueError(f"Metric '{metric.get('name')}' needs better=lower|higher")
    for dim in ("reporting", *FACTOR_DIMENSIONS):
        if dim not in rules["dimensions"]:
            raise ValueError(f"Missing dimension config: {dim}")


def get_factor(key: str) -> dict[str, Any]:
    for factor in load_rules()["factors"]:
        if factor["key"] == key:
            return factor
    raise KeyError(key)


def factor_label(key: str) -> str:
    if key == REPORTING_FACTOR:
        return "Reporting"
    return get_factor(key)["label"]


def dimension_config(dimension: str) -> dict[str, Any]:
    return load_rules()["dimensions"][dimension]


def source_priority(source_type: str) -> dict[str, Any]:
    table = load_rules()["source_priorities"]
    return table.get(source_type, table["other_web"])
