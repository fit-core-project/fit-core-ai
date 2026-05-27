from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any

import pytest

from engines.candidate_ranker import format_candidates_for_prompt, score_candidate_exercises
from engines.schemas import PainAreaEntry, RecentSetRecord

FIXTURES_PATH = Path(__file__).parent / "pool_size_fixtures.json"
REPORT_PATH = Path(__file__).parent / ".artifacts" / "candidate_pool_size_report.json"
POOL_SIZES = (12, 18)
ALLOWED_VERDICTS = {"promote_pool_18_to_default", "keep_pool_12_as_default"}


def load_fixtures() -> list[dict[str, Any]]:
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


def _pain_areas(scenario: dict[str, Any]) -> list[PainAreaEntry]:
    return [PainAreaEntry(**item) for item in scenario.get("current_pain_areas", [])]


def _recent_sets(scenario: dict[str, Any]) -> list[RecentSetRecord]:
    return [
        RecentSetRecord(exercise_id=exercise_id, exercise_name=exercise_id, reps=8)
        for exercise_id in scenario.get("recent_exercises", [])
    ]


def _equipment_tokens(candidate: dict[str, Any]) -> set[str]:
    return {
        token.strip().upper()
        for token in str(candidate.get("equipment_req") or "").split(",")
        if token.strip()
    }


def _has_pain_trigger(candidate: dict[str, Any], pain_areas: list[PainAreaEntry]) -> bool:
    pain_tokens = {pain.body_part.lower() for pain in pain_areas if pain.body_part}
    triggers = str(candidate.get("pain_triggers") or "").lower()
    return any(token and token in triggers for token in pain_tokens)


def _has_blocked_equipment(candidate: dict[str, Any], unavailable_equipment: list[str]) -> bool:
    blocked = {item.strip().upper() for item in unavailable_equipment}
    return bool((_equipment_tokens(candidate) - {"BODYWEIGHT"}) & blocked)


def _is_safe(candidate: dict[str, Any], scenario: dict[str, Any]) -> bool:
    return not _has_blocked_equipment(candidate, scenario.get("unavailable_equipment", [])) and not _has_pain_trigger(
        candidate, _pain_areas(scenario)
    )


def _rank(scenario: dict[str, Any], pool_size: int) -> list[dict[str, Any]]:
    return score_candidate_exercises(
        scenario["candidates"],
        scenario.get("target_muscles", []),
        doms_db=scenario.get("doms", {}),
        blocked_equipment=scenario.get("unavailable_equipment", []),
        pain_areas=_pain_areas(scenario),
        recent_sets=_recent_sets(scenario),
        top_n=pool_size,
    )


def _target_distribution(ranked: list[dict[str, Any]], target_muscles: list[str]) -> dict[str, int]:
    return {
        muscle: sum(1 for candidate in ranked if candidate.get("primary_muscle") == muscle)
        for muscle in target_muscles
    }


def _quality_proxy_score(metrics: dict[str, Any]) -> int:
    coverage_score = round(metrics["coverage_potential"] * 45)
    diversity_score = min(25, metrics["diversity_potential"] * 4)
    balance_score = 20 if 0.25 <= metrics["compound_ratio"] <= 0.75 else 12
    shortage_penalty = round(metrics["candidate_shortage_rate"] * 20)
    bodyweight_penalty = min(10, metrics["bodyweight_fallback_dependency"] * 2)
    return max(0, min(100, coverage_score + diversity_score + balance_score - shortage_penalty - bodyweight_penalty))


def run_single_scenario(scenario: dict[str, Any], pool_size: int) -> dict[str, Any]:
    ranked = _rank(scenario, pool_size)
    target_muscles = scenario.get("target_muscles", [])
    target_dist = _target_distribution(ranked, target_muscles)
    candidate_count = len(ranked)
    compound_count = sum(1 for candidate in ranked if str(candidate.get("movement_type")).upper() == "COMPOUND")
    isolation_count = sum(1 for candidate in ranked if str(candidate.get("movement_type")).upper() == "ISOLATION")
    safe_candidate_count = sum(1 for candidate in ranked if _is_safe(candidate, scenario))
    bodyweight_count = sum(1 for candidate in ranked if _equipment_tokens(candidate) <= {"BODYWEIGHT"})
    recent_ids = set(scenario.get("recent_exercises", []))
    recent_overlap_count = sum(1 for candidate in ranked if candidate.get("id") in recent_ids)

    equipment_distribution: dict[str, int] = {}
    for candidate in ranked:
        equipment = str(candidate.get("equipment_req") or "UNKNOWN")
        equipment_distribution[equipment] = equipment_distribution.get(equipment, 0) + 1

    shortage_muscles = [muscle for muscle, count in target_dist.items() if count == 0]
    prompt_payload = format_candidates_for_prompt(ranked)
    payload_chars = len(prompt_payload)
    unique_movement_types = {str(candidate.get("movement_type") or "UNKNOWN") for candidate in ranked}
    unique_equipment = {str(candidate.get("equipment_req") or "UNKNOWN") for candidate in ranked}
    hard_constraint_violations = [
        candidate["id"]
        for candidate in ranked
        if _has_blocked_equipment(candidate, scenario.get("unavailable_equipment", []))
        or _has_pain_trigger(candidate, _pain_areas(scenario))
    ]

    metrics = {
        "candidate_count": candidate_count,
        "target_muscle_candidate_count": target_dist,
        "safe_candidate_count": safe_candidate_count,
        "compound_count": compound_count,
        "isolation_count": isolation_count,
        "equipment_distribution": equipment_distribution,
        "pain_safe_alternative_count": safe_candidate_count,
        "recent_exercise_overlap_count": recent_overlap_count,
        "recent_exercise_overlap_rate": recent_overlap_count / max(candidate_count, 1),
        "bodyweight_only_count": bodyweight_count,
        "candidate_payload_char_count": payload_chars,
        "rendered_prompt_char_count": payload_chars,
        "approximate_token_count": math.ceil(payload_chars / 4),
        "coverage_potential": sum(1 for count in target_dist.values() if count > 0) / max(len(target_muscles), 1),
        "diversity_potential": len(unique_movement_types) + len(unique_equipment),
        "compound_ratio": compound_count / max(candidate_count, 1),
        "candidate_shortage_rate": len(shortage_muscles) / max(len(target_muscles), 1),
        "bodyweight_fallback_dependency": bodyweight_count / max(candidate_count, 1),
        "hard_constraint_violation_count": len(hard_constraint_violations),
        "hard_constraint_violations": hard_constraint_violations,
        "repair_count": 0,
        "fallback_count": 0,
    }
    metrics["quality_proxy_score"] = _quality_proxy_score(metrics)
    metrics["scenario_passed"] = (
        metrics["candidate_count"] <= pool_size
        and metrics["hard_constraint_violation_count"] == 0
        and metrics["coverage_potential"] > 0
    )
    return metrics


def run_experiment() -> dict[str, Any]:
    scenarios = []
    pool_12_results = []
    pool_18_results = []

    for scenario in load_fixtures():
        pool_12 = run_single_scenario(scenario, 12)
        pool_18 = run_single_scenario(scenario, 18)
        pool_12_results.append(pool_12)
        pool_18_results.append(pool_18)
        payload_growth = (
            (pool_18["candidate_payload_char_count"] - pool_12["candidate_payload_char_count"])
            / max(pool_12["candidate_payload_char_count"], 1)
        )
        hard_delta = pool_18["hard_constraint_violation_count"] - pool_12["hard_constraint_violation_count"]
        delta = {
            "candidate_count": pool_18["candidate_count"] - pool_12["candidate_count"],
            "quality_proxy_score": pool_18["quality_proxy_score"] - pool_12["quality_proxy_score"],
            "candidate_shortage_rate": pool_18["candidate_shortage_rate"] - pool_12["candidate_shortage_rate"],
            "candidate_payload_char_count": (
                pool_18["candidate_payload_char_count"] - pool_12["candidate_payload_char_count"]
            ),
            "payload_growth_ratio": payload_growth,
            "hard_constraint_violation_count": hard_delta,
            "repair_count": pool_18["repair_count"] - pool_12["repair_count"],
            "fallback_count": pool_18["fallback_count"] - pool_12["fallback_count"],
        }
        warnings = []
        if payload_growth > 0.60:
            warnings.append(f"payload growth exceeds 60%: {payload_growth:.3f}")
        if hard_delta > 0:
            warnings.append("hard constraint violations increased")
        scenarios.append({
            "scenario_id": scenario["id"],
            "pool_12_metrics": pool_12,
            "pool_18_metrics": pool_18,
            "delta": delta,
            "passed": not warnings,
            "warnings": warnings,
        })

    aggregate = _aggregate(pool_12_results, pool_18_results)
    verdict = _verdict(aggregate)
    return {
        "scenarios": scenarios,
        "aggregate": aggregate,
        "verdict": verdict,
    }


def _avg(items: list[dict[str, Any]], key: str) -> float:
    return sum(float(item[key]) for item in items) / max(len(items), 1)


def _aggregate(pool_12: list[dict[str, Any]], pool_18: list[dict[str, Any]]) -> dict[str, Any]:
    avg_payload_12 = _avg(pool_12, "candidate_payload_char_count")
    avg_payload_18 = _avg(pool_18, "candidate_payload_char_count")
    return {
        "pass_rate_12": sum(1 for item in pool_12 if item["scenario_passed"]) / max(len(pool_12), 1),
        "pass_rate_18": sum(1 for item in pool_18 if item["scenario_passed"]) / max(len(pool_18), 1),
        "avg_quality_proxy_12": _avg(pool_12, "quality_proxy_score"),
        "avg_quality_proxy_18": _avg(pool_18, "quality_proxy_score"),
        "avg_payload_char_count_12": avg_payload_12,
        "avg_payload_char_count_18": avg_payload_18,
        "avg_payload_growth_ratio": (avg_payload_18 - avg_payload_12) / max(avg_payload_12, 1),
        "fallback_delta": sum(item["fallback_count"] for item in pool_18) - sum(item["fallback_count"] for item in pool_12),
        "repair_delta": sum(item["repair_count"] for item in pool_18) - sum(item["repair_count"] for item in pool_12),
        "hard_violation_delta": (
            sum(item["hard_constraint_violation_count"] for item in pool_18)
            - sum(item["hard_constraint_violation_count"] for item in pool_12)
        ),
        "candidate_shortage_delta": _avg(pool_18, "candidate_shortage_rate") - _avg(pool_12, "candidate_shortage_rate"),
    }


def _verdict(aggregate: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    if aggregate["pass_rate_18"] < aggregate["pass_rate_12"]:
        reasons.append("scenario pass rate regressed")
    if aggregate["avg_quality_proxy_18"] < aggregate["avg_quality_proxy_12"]:
        reasons.append("average quality proxy regressed")
    if aggregate["candidate_shortage_delta"] > 0:
        reasons.append("candidate shortage worsened")
    if aggregate["fallback_delta"] > 0:
        reasons.append("fallback count increased")
    if aggregate["repair_delta"] > 0:
        reasons.append("repair count increased")
    if aggregate["hard_violation_delta"] > 0:
        reasons.append("hard constraint violations increased")
    if aggregate["avg_payload_growth_ratio"] > 0.60:
        reasons.append("candidate payload growth exceeded 60%")

    return {
        "recommendation": "keep_pool_12_as_default" if reasons else "promote_pool_18_to_default",
        "reasons": reasons or ["all promotion criteria met in offline proxy experiment"],
    }


@pytest.fixture(scope="module")
def fixtures() -> list[dict[str, Any]]:
    return load_fixtures()


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    payload = run_experiment()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def test_fixture_contains_at_least_four_scenarios(fixtures):
    assert len(fixtures) >= 4


def test_each_scenario_contains_at_least_twenty_candidates(fixtures):
    for scenario in fixtures:
        assert len(scenario["candidates"]) >= 20, scenario["id"]


def test_top_n_12_returns_at_most_twelve(fixtures):
    for scenario in fixtures:
        assert len(_rank(scenario, 12)) <= 12


def test_top_n_18_returns_at_most_eighteen(fixtures):
    for scenario in fixtures:
        assert len(_rank(scenario, 18)) <= 18


def test_production_default_remains_twelve():
    signature = inspect.signature(score_candidate_exercises)
    assert signature.parameters["top_n"].default == 12


def test_pool_18_payload_growth_is_within_sixty_percent(report):
    for scenario in report["scenarios"]:
        assert scenario["delta"]["payload_growth_ratio"] <= 0.60, scenario["scenario_id"]


def test_pool_18_hard_constraint_violations_do_not_increase(report):
    for scenario in report["scenarios"]:
        assert scenario["delta"]["hard_constraint_violation_count"] <= 0, scenario["scenario_id"]


def test_verdict_is_allowed_value(report):
    assert report["verdict"]["recommendation"] in ALLOWED_VERDICTS


def test_json_report_is_generated(report):
    assert REPORT_PATH.exists()
    payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert payload["verdict"]["recommendation"] in ALLOWED_VERDICTS


def test_routine_pipeline_does_not_force_top_n_18():
    pipeline = Path("engines/routine_pipeline.py").read_text(encoding="utf-8")
    assert "top_n=18" not in pipeline


def test_candidate_ranker_logic_still_slices_by_top_n():
    source = Path("engines/candidate_ranker.py").read_text(encoding="utf-8")
    assert "top_n: int = 12" in source
    assert ")[:top_n]" in source or "[:top_n]" in source
