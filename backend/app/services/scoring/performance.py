"""Performance Improvement ambition level (1-3), per factor.

1 no evidence of measurable improvement
2 quantitative data shows improvement over the last three years (computed here in Python)
3 evidence of sector-leading performance / alignment with a recognised benchmark

Vague claims ("we significantly improved") are never accepted as quantitative evidence."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.config.loader import load_rules
from app.services.scoring.common import (
    MET, NOT_FOUND, Condition, EvidenceItem, ScoreResult, finalize_status, no_evidence_reason, split_verified,
)

MAX_LEVEL = 3
CONFLICT_TOLERANCE = 0.01  # 1 % difference between two sources for the same metric/year


def evaluate_series(points: list[tuple[int, float]] | list[list[float]], better: str, assessment_year: int,
                    lookback: int | None = None) -> dict[str, Any] | None:
    """Compare earliest vs latest value within [year - lookback, year]. None if < 2 points."""
    lookback = load_rules()["lookback_years"] if lookback is None else lookback
    window = sorted((int(y), float(v)) for y, v in points if assessment_year - lookback <= int(y) <= assessment_year)
    if len(window) < 2:
        return None
    (y0, v0), (y1, v1) = window[0], window[-1]
    change = v1 - v0
    pct = (change / abs(v0) * 100) if v0 else None
    improved = change < 0 if better == "lower" else change > 0
    if change == 0:
        improved = False
    return {"from_year": y0, "to_year": y1, "from_value": v0, "to_value": v1, "change": change,
            "change_pct": round(pct, 2) if pct is not None else None, "improved": improved, "points_used": len(window)}


def evaluate_relative_change(facts: dict[str, Any], assessment_year: int, lookback: int | None = None) -> dict[str, Any] | None:
    lookback = load_rules()["lookback_years"] if lookback is None else lookback
    base, end = facts.get("base_year"), facts.get("end_year") or assessment_year
    if base is None or end is None or not (assessment_year - lookback - 1 <= base < end <= assessment_year):
        return None
    went_down = facts.get("direction_word") in {"reduced", "decreased", "lowered", "cut", "fell", "declined", "dropped", "went down"}
    if facts.get("direction_word") == "improved":
        improved = True
    else:
        improved = went_down if facts.get("better") == "lower" else not went_down
    return {"from_year": base, "to_year": end, "change_pct": facts.get("change_pct"), "improved": improved}


def describe_change(e: EvidenceItem, ev: dict[str, Any]) -> str:
    label = (e.facts.get("metric_label") or e.facts.get("metric") or "metric").strip()
    unit = e.facts.get("unit")
    pct = ev.get("change_pct")
    if "from_value" in ev:
        vals = f"{ev['from_value']:,g} → {ev['to_value']:,g}{' ' + unit if unit and unit not in label else ''}"
    else:
        vals = f"{e.facts.get('direction_word', 'changed')} by {pct:g}%"
        pct = None
    return f"{label[:90]}: {vals} ({ev['from_year']}→{ev['to_year']}" + (f", {pct:+.1f}%" if pct is not None else "") + ")"


def _find_conflicts(series: list[EvidenceItem]) -> list[str]:
    by_metric_year: dict[tuple[str, str | None, int], list[tuple[float, int]]] = defaultdict(list)
    for e in series:
        for y, v in e.facts.get("points", []):
            by_metric_year[(e.facts.get("metric"), e.facts.get("unit"), int(y))].append((float(v), e.source_id))
    conflicts = []
    for (metric, unit, year), vals in by_metric_year.items():
        values = {v for v, _ in vals}
        sources = {s for _, s in vals}
        if len(values) > 1 and len(sources) > 1:
            lo, hi = min(values), max(values)
            if hi and (hi - lo) / abs(hi) > CONFLICT_TOLERANCE:
                conflicts.append(f"Different values reported for {metric} ({unit or 'unit n/a'}) in {year}: {sorted(values)}")
    return conflicts


def score_performance(evidence: list[EvidenceItem], factor_label: str = "the factor", assessment_year: int = 2025) -> ScoreResult:
    items = [e for e in evidence if e.claim_type in ("performance_series", "relative_change", "sector_comparison", "unquantified_improvement_claim")]
    verified, unverified = split_verified([e for e in items if e.claim_type != "unquantified_improvement_claim"])
    vague = [e for e in items if e.claim_type == "unquantified_improvement_claim"]

    improving: list[tuple[EvidenceItem, dict[str, Any]]] = []
    worsening: list[tuple[EvidenceItem, dict[str, Any]]] = []
    for e in verified:
        if e.claim_type == "performance_series":
            ev = evaluate_series(e.facts.get("points", []), e.facts.get("better", "lower"), assessment_year)
        elif e.claim_type == "relative_change":
            ev = evaluate_relative_change(e.facts, assessment_year)
        else:
            continue
        if ev is None:
            continue
        e.facts["evaluation"] = ev
        (improving if ev["improved"] else worsening).append((e, ev))

    sector = [e for e in verified if e.claim_type == "sector_comparison" and e.facts.get("favorable") is True]
    conflicts = _find_conflicts([e for e in verified if e.claim_type == "performance_series"])
    improving_metrics = {e.facts.get("metric") for e, _ in improving}
    mixed = sorted(improving_metrics & {e.facts.get("metric") for e, _ in worsening})

    if sector:
        level, rule = 3, "PI3_SECTOR_LEADING_OR_BENCHMARK_ALIGNED"
        reason = f"The company reports {factor_label} performance explicitly compared favourably with a sector average or recognised benchmark."
        ids = [e.id for e in sector] + [e.id for e, _ in improving]
        missing: list[str] = []
    elif improving:
        # headline: the best-supported series (most data points, known unit, source quality), not the
        # largest % change, which is usually noise from tiny base values (e.g. 1% -> 0%)
        best_e, best_ev = max(improving, key=lambda t: (t[1].get("points_used", 2), bool(t[0].facts.get("unit")),
                                                        t[0].source_quality, -abs(t[1].get("change_pct") or 0)))
        level, rule = 2, "PI2_QUANTITATIVE_IMPROVEMENT"
        reason = f"Quantitative improvement: {describe_change(best_e, best_ev)}, calculated from reported values."
        if len(improving) > 1:
            reason += f" {len(improving)} improving data series found in total."
        ids = [e.id for e, _ in improving]
        missing = ["No evidence of sector-leading performance or alignment with a recognised benchmark was found."]
    else:
        level, rule = 1, "PI1_NO_IMPROVEMENT_EVIDENCE"
        ids = [e.id for e, _ in worsening]
        if worsening:
            reason = f"Quantitative {factor_label} data was found, but it does not show improvement over the last three years."
        else:
            reason = no_evidence_reason(f"quantitative {factor_label} performance improvement")
        missing = ["No quantitative multi-year data showing improvement over the last three years was found."]

    r = ScoreResult(level, MAX_LEVEL, "", rule, reason, evidence_ids=ids[:10], next_level_missing=missing, unverified_ids=unverified)
    r.conditions = {
        "quantitative_improvement": Condition("Quantitative improvement (last 3 years)", MET if improving else NOT_FOUND, [e.id for e, _ in improving][:5]),
        "sector_leading": Condition("Sector-leading / benchmark-aligned", MET if sector else NOT_FOUND, [e.id for e in sector][:5]),
    }
    if improving:
        r.conditions["quantitative_improvement"].detail = "; ".join(describe_change(e, ev) for e, ev in improving[:3])
    if vague:
        r.notes.append(f"{len(vague)} unquantified improvement claim(s) found; not accepted as quantitative evidence.")
    if mixed:
        r.notes.append(f"Metrics with both improving and worsening data points: {', '.join(mixed)}.")
    if conflicts:
        r.conflicting = True
        r.notes.extend(conflicts)
    if level == 3:
        r.notes.append("Sector-leading status is based on the company's own comparison; confirm against sector data.")
        r.evidence_status = "REQUIRES_REVIEW"
    return finalize_status(r)
