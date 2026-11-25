"""4. percentage/year extraction (regex)  12. LLM JSON parsing with mocked LLM output."""


from __future__ import annotations


import json


import pytest


from app.services.llm.base import LLM_NOT_CONFIGURED_MESSAGE, NullLLMProvider


from app.services.llm.json_utils import LLMOutputError, as_confidence, parse_llm_json


from app.services.llm.llama_cpp_provider import LlamaCppProvider


from tests.conftest import FakeLLM


@pytest.mark.parametrize("raw,expected", [
    ('{"goal_found": true, "target_value": 30}', {"goal_found": True, "target_value": 30}),
    ('```json\n{"goal_found": true}\n```', {"goal_found": True}),
    ('Sure! Here is the JSON: {"a": 1, "b": [1, 2,],} Hope this helps', {"a": 1, "b": [1, 2]}),
    ('{"goal_found": True, "owner": None}', {"goal_found": True, "owner": None}),
    ('{"quote": "text with } brace", "x": 1}', {"quote": "text with } brace", "x": 1}),
])
def test_parse_llm_json(raw, expected):
    assert parse_llm_json(raw) == expected


@pytest.mark.parametrize("raw", ["", "no json here", "{broken: json"])
def test_parse_llm_json_errors(raw):
    with pytest.raises(LLMOutputError):
        parse_llm_json(raw)


def test_confidence_normalisation():
    assert as_confidence(91) == 0.91 and as_confidence("0.4") == 0.4 and as_confidence(None) == 0.5


ENERGY_TEXT = ("We aim to reduce energy consumption by 30% by 2030 compared with a 2020 baseline. "
               "The target is owned by our Chief Operating Officer.")

