"""The questionnaire questions, read unchanged from app/config/esg_rules.json.

One question per factor x dimension, plus the company-wide Reporting question. Question text, level definitions and
scales are taken verbatim from the configuration: the questionnaire is the methodology, this module only lists it."""
from __future__ import annotations

from dataclasses import dataclass

from app.config.loader import FACTOR_DIMENSIONS, REPORTING_FACTOR, load_rules


@dataclass(frozen=True)
class Question:
    factor: str  # factor key, or "reporting"
    factor_label: str
    dimension: str  # reporting | planning | execution | performance
    dimension_label: str
    text: str  # the questionnaire question, verbatim
    levels: dict[int, str]  # level -> definition, verbatim
    keywords: tuple[str, ...]  # the factor's vocabulary (retrieval only)
    query_terms: tuple[str, ...]  # the dimension's vocabulary (retrieval only)

    @property
    def max_level(self) -> int:
        return max(self.levels)

    @property
    def key(self) -> tuple[str, str]:
        return self.factor, self.dimension


def questionnaire() -> list[Question]:
    rules = load_rules()
    dims = rules["dimensions"]

    def levels(dim: str) -> dict[int, str]:
        return {int(k): v for k, v in sorted(dims[dim]["levels"].items(), key=lambda kv: int(kv[0]))}

    out = [Question(REPORTING_FACTOR, "Reporting", "reporting", dims["reporting"]["label"], dims["reporting"]["question"],
                    levels("reporting"), (), tuple(dims["reporting"].get("query_terms", ())))]
    for factor in rules["factors"]:
        for dim in FACTOR_DIMENSIONS:
            out.append(Question(factor["key"], factor["label"], dim, dims[dim]["label"], dims[dim]["question"], levels(dim),
                                tuple(factor.get("keywords", ())), tuple(dims[dim].get("query_terms", ()))))
    return out
