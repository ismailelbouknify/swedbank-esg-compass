"""4. percentage/year extraction (regex)  12. LLM JSON parsing with mocked LLM output."""
from __future__ import annotations

import json

import pytest

from app.services.extraction import patterns as P
from app.services.extraction.extractor import FactExtractor, Passage
from app.services.llm.base import LLM_NOT_CONFIGURED_MESSAGE, NullLLMProvider
from app.services.llm.json_utils import LLMOutputError, as_confidence, parse_llm_json
from app.services.llm.llama_cpp_provider import LlamaCppProvider
from app.services.verification.verifier import REQUIRES_REVIEW, VERIFIED, number_in_text, verify_candidate
from tests.conftest import FakeLLM


def test_years_and_percentages():
    text = "We reduced emissions by 12.5% between 2019 and 2024, and aim for 30 % by 2030 (40 per cent later)."
    assert P.find_years(text) == [2019, 2024, 2030]
    assert P.find_percentages(text) == [12.5, 30.0, 40.0]


def test_framework_and_assurance_detection():
    text = ("This report has been prepared in accordance with the GRI Standards. We also discuss SASB topics. "
            "Deloitte AB has provided limited assurance in accordance with ISAE 3000 on the sustainability report.")
    fws = {f.framework: f.explicit for f in P.detect_frameworks(text)}
    assert fws["GRI"] is True and fws["SASB"] is False
    a = P.detect_assurance(text)[0]
    assert a.provider == "Deloitte" and a.standard == "ISAE 3000" and a.level == "limited" and a.covers_sustainability


def test_target_detection():
    t = P.detect_target("We aim to reduce energy consumption by 30% by 2030 compared with a 2020 baseline.", 2025)
    assert (t.quantitative, t.target_value, t.target_unit, t.target_year, t.baseline_year) == (True, 30.0, "%", 2030, 2020)
    q = P.detect_target("Our strategy is to become a circular business.", 2025)
    assert q is not None and q.quantitative is False and q.target_year is None
    assert P.detect_target("Energy use rose last year.", 2025) is None
    # "in <reporting year>" is not a deadline; "by <next year>" is
    assert P.detect_target("In 2024 our energy strategy was updated.", 2024).target_year is None
    assert P.detect_target("We aim to use only climate-neutral energy by 2025.", 2024).target_year == 2025
    z = P.detect_target("Our goal is zero data breaches by 2026.", 2025)
    assert z.quantitative and z.target_value == 0 and z.target_unit == "data"


def test_goal_sentence_filter():
    from app.config.loader import get_factor
    from app.services.extraction.extractor import is_goal_sentence

    f = get_factor("product_lifecycle")
    assert is_goal_sentence("By 2025 we aim to use 25 per cent recycled materials in new models.", f)
    assert not is_goal_sentence("Recycled content ambition", f)  # heading
    assert not is_goal_sentence("In 2024, we improved reuse procedures for recycled materials in line with our ambition.", f)


def test_series_and_relative_change():
    s = P.extract_series("Indicator 2023 2024 2025\nEnergy intensity (kWh/unit) 124 116 108\n")
    assert s[0].points == [(2023, 124.0), (2024, 116.0), (2025, 108.0)] and s[0].unit == "kWh/unit"
    # one-cell-per-line PDF layout
    s2 = P.extract_series("Indicator\n2023\n2024\nEmployee turnover (%)\n14.2\n11.8\n")
    assert s2[0].points == [(2023, 14.2), (2024, 11.8)]
    s3 = P.extract_series("Energy use fell from 5,400 MWh in 2022 to 4,900 MWh in 2024.")
    assert s3[0].points == [(2022, 5400.0), (2024, 4900.0)]
    rel = P.extract_relative_changes("Energy intensity decreased by 12.9% compared with 2023.", 2025)
    assert rel[0].change_pct == 12.9 and rel[0].meta["base_year"] == 2023
    # targets are not achievements
    assert P.extract_relative_changes("We aim to reduce energy use by 30% compared with 2020.", 2025) == []


def test_real_world_table_layouts():
    # label printed above a one-year-per-line header (seen in a real annual report)
    s = P.extract_series("Energy intensity per revenue (MWh/SEKm)\n2024\n2023\n2022\n1.2\n1.4\n1.5\n")
    assert s[0].metric_label.startswith("Energy intensity") and s[0].points == [(2022, 1.5), (2023, 1.4), (2024, 1.2)]
    # a label followed by the next header's years must not be read as values
    assert all(v < 1990 for ser in P.extract_series("Indicator 2023 2024\nEnergy intensity\n2023\n2024\n") for _, v in ser.points)
    s2 = P.extract_series("MWh\n2024\n2023\n2022\nTotal fossil energy consumption\n458,000\n539,000\n575,000\n")
    assert s2[0].points == [(2022, 575000.0), (2023, 539000.0), (2024, 458000.0)]


def test_from_to_changes_without_inline_years():
    s = P.extract_from_to_changes("The share of renewable energy increased from 60 to 64 per cent year over year.", 2024)
    assert s[0].points == [(2023, 60.0), (2024, 64.0)] and s[0].unit == "%"
    s = P.extract_from_to_changes("Turnover decreased from 14.2% to 11.8% between 2022 and 2024.", 2024)
    assert s[0].points == [(2022, 14.2), (2024, 11.8)]
    assert P.extract_from_to_changes("We will increase the share from 60 to 80 per cent year over year.", 2024) == []


def test_pdf_text_cleanup():
    from app.services.documents.pdf_extractor import clean_text

    assert clean_text("climate ambi­\ntions") == "climate ambitions"
    assert clean_text("climate-\nneutral") == "climate-neutral"
    assert P.split_sentences("We aim to reduce energy use by 30% compared with a 2020 baseline across all of our\nplants by 2030.")[0].endswith("by 2030.")


def test_number_in_text_variants():
    assert number_in_text(1234, "consumed 1,234 MWh")
    assert number_in_text(12.5, "a 12,5 % reduction")
    assert number_in_text(30, "by 30% by 2030")
    assert not number_in_text(3, "by 30% by 2030")


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


def _passage(text: str, page: int = 42) -> Passage:
    return Passage(chunk_id=1, document_id=1, source_id=1, source_title="Sustainability Report 2025", source_url=None,
                   source_type="uploaded_report", source_quality=1.0, is_public=True, page_number=page, text=text, doc_year=2025)


ENERGY_TEXT = ("We aim to reduce energy consumption by 30% by 2030 compared with a 2020 baseline. "
               "The target is owned by our Chief Operating Officer.")


def test_mocked_llm_planning_extraction_and_verification():
    llm = FakeLLM()
    ex = FactExtractor(llm, 2025)
    p = _passage(ENERGY_TEXT)
    [c] = ex.planning("energy_management", [p])
    assert c.method == "llm" and c.model_name == "fake:test-1b" and llm.calls
    assert c.facts["target_value"] == 30 and c.facts["target_year"] == 2030 and c.facts["baseline_year"] == 2020
    assert c.facts["public"] is True and c.facts["clear_owner"] is True
    assert verify_candidate(c, ENERGY_TEXT).status == VERIFIED
    # the prompt contains only the short passage, never a whole report
    assert len(llm.calls[0]) < 4000


def test_hallucinated_llm_facts_are_not_trusted():
    hallucination = json.dumps({
        "goal_found": True, "formalized": True, "quantitative": True, "target_value": 45, "target_unit": "%",
        "target_year": 2035, "baseline_year": None, "clear_owner": False, "owner": None,
        "benchmark_found": True, "benchmark_type": "SBTi",
        "evidence_quote": "We will cut energy use by 45% by 2035 in line with SBTi.", "confidence": 0.99,
    })
    ex = FactExtractor(FakeLLM({"FACTOR:\nEnergy Management\n\nDIMENSION:\nPlanning": hallucination}), 2025)
    [c] = ex.planning("energy_management", [_passage(ENERGY_TEXT)])
    assert c.facts["benchmark_found"] is False  # no benchmark wording in the source
    assert any("benchmark" in n for n in c.notes)
    v = verify_candidate(c, ENERGY_TEXT)
    assert v.status == REQUIRES_REVIEW
    assert any("Quote not found" in n for n in v.notes)
    assert any("target_value=45" in n for n in v.notes)


def test_unparseable_llm_output_falls_back_to_heuristic():
    ex = FactExtractor(FakeLLM({"DIMENSION:\nPlanning": "I cannot answer that."}), 2025)
    [c] = ex.planning("energy_management", [_passage(ENERGY_TEXT)])
    assert c.method == "heuristic" and ex.llm_failures == 1
    assert c.facts["target_value"] == 30


def test_missing_model_reports_clear_message(tmp_path):
    st = LlamaCppProvider(tmp_path / "model.gguf").status()
    assert not st.available and "Local LLM model is not configured" in st.message
    assert NullLLMProvider().status().message == LLM_NOT_CONFIGURED_MESSAGE
    ex = FactExtractor(NullLLMProvider(), 2025)
    assert ex.mode == "heuristic"
