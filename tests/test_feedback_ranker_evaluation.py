from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from engines.candidate_ranker import score_candidate_exercises
from engines.feedback_aggregation import get_feedback_adjustments
from engines.schemas import PainAreaEntry, RecentSetRecord
from models.routine_feedback import RoutineFeedback
from tests.evaluation.helpers import EvalHarness


@pytest.fixture(scope="module")
def feedback_scenarios() -> dict[str, dict[str, Any]]:
    path = Path(__file__).parent / "evaluation" / "fixtures" / "feedback_ranker_scenarios.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {scenario["id"]: scenario for scenario in data["scenarios"]}


def _seed_feedback_rows(test_db, rows: list[dict]) -> None:
    for row in rows:
        for _ in range(int(row.get("repeat", 1))):
            fb = RoutineFeedback(
                routine_draft_id=row.get("routine_draft_id", "draft-fb"),
                user_id=row.get("user_id"),
                rating=row.get("rating"),
                completed=row.get("completed"),
                accepted_without_edits=row.get("accepted_without_edits"),
                skipped_exercise_ids=row.get("skipped_exercise_ids"),
                edited_exercises=row.get("edited_exercises"),
                user_note=None,
                source="eval",
            )
            test_db.add(fb)
    test_db.commit()


def _pain_areas(scenario: dict[str, Any]) -> list[PainAreaEntry]:
    return [PainAreaEntry(**item) for item in scenario.get("current_pain_areas", [])]


def _recent_sets(scenario: dict[str, Any]) -> list[RecentSetRecord]:
    return [
        RecentSetRecord(exercise_id=exercise_id, exercise_name=exercise_id, reps=8)
        for exercise_id in scenario.get("recent_exercises", [])
    ]


def _rank(scenario: dict[str, Any], feedback_adjustments: dict | None = None) -> list[dict[str, Any]]:
    return score_candidate_exercises(
        scenario["candidates"],
        scenario.get("target_muscles", []),
        doms_db=scenario.get("doms", {}),
        blocked_equipment=scenario.get("unavailable_equipment", []),
        pain_areas=_pain_areas(scenario),
        recent_sets=_recent_sets(scenario),
        feedback_adjustments=feedback_adjustments,
        top_n=12,
    )


def _ids(ranked: list[dict[str, Any]]) -> list[str]:
    return [str(item["id"]) for item in ranked]


def _score_by_id(ranked: list[dict[str, Any]]) -> dict[str, int]:
    return {str(item["id"]): int(item["score"]) for item in ranked}


def _adjustments(test_db, scenario: dict[str, Any], user_id: str | None = "user-a") -> dict:
    return get_feedback_adjustments(
        test_db,
        user_id=user_id,
        exercise_ids=[str(candidate["id"]) for candidate in scenario["candidates"]],
    )


def test_fb01_positive_replacement_promotes(feedback_scenarios, test_db):
    scenario = feedback_scenarios["FB01"]
    _seed_feedback_rows(test_db, scenario["feedback_rows"])
    baseline = _rank(scenario)
    ranked = _rank(scenario, _adjustments(test_db, scenario))

    assert _ids(baseline) == scenario["expected"]["flag_off_order"]
    assert _ids(ranked) == scenario["expected"]["flag_on_order"]
    delta = _score_by_id(ranked)["zeta_press"] - _score_by_id(baseline)["zeta_press"]
    assert delta >= scenario["expected"]["min_score_delta"]


def test_fb02_skipped_demotes(feedback_scenarios, test_db):
    scenario = feedback_scenarios["FB02"]
    _seed_feedback_rows(test_db, scenario["feedback_rows"])
    baseline = _rank(scenario)
    ranked = _rank(scenario, _adjustments(test_db, scenario))

    assert _ids(ranked) == scenario["expected"]["flag_on_order"]
    delta = _score_by_id(ranked)["alpha_press"] - _score_by_id(baseline)["alpha_press"]
    assert delta <= -scenario["expected"]["min_score_delta"]


def test_fb03_pain_reason_strong_negative(feedback_scenarios, test_db):
    scenario = feedback_scenarios["FB03"]
    _seed_feedback_rows(test_db, scenario["feedback_rows"])
    baseline = _rank(scenario)
    ranked = _rank(scenario, _adjustments(test_db, scenario))

    delta = _score_by_id(ranked)["alpha_press"] - _score_by_id(baseline)["alpha_press"]
    assert _ids(ranked) == scenario["expected"]["flag_on_order"]
    assert delta <= -scenario["expected"]["min_score_delta"]


def test_fb04_no_feedback_no_change(feedback_scenarios, test_db):
    scenario = feedback_scenarios["FB04"]
    baseline = _rank(scenario)
    ranked = _rank(scenario, _adjustments(test_db, scenario))

    assert ranked == baseline
    assert all("feedback adj" not in " ".join(item["score_reasons"]) for item in ranked)


def test_fb05_flag_off_no_change(feedback_scenarios, test_db):
    scenario = feedback_scenarios["FB05"]
    _seed_feedback_rows(test_db, scenario["feedback_rows"])
    baseline = _rank(scenario)
    flag_off = _rank(scenario, None)

    assert flag_off == baseline
    assert all("feedback adj" not in " ".join(item["score_reasons"]) for item in flag_off)


def test_fb09_global_feedback_weak_effect(feedback_scenarios, test_db):
    scenario = feedback_scenarios["FB09"]
    _seed_feedback_rows(test_db, scenario["feedback_rows"])
    baseline = _rank(scenario)
    ranked = _rank(scenario, _adjustments(test_db, scenario, user_id=None))

    delta = _score_by_id(ranked)["zeta_press"] - _score_by_id(baseline)["zeta_press"]
    assert _ids(ranked) == scenario["expected"]["flag_on_order"]
    assert scenario["expected"]["min_score_delta"] <= delta <= 3


def test_fb10_user_feedback_dominates_global(feedback_scenarios, test_db):
    scenario = feedback_scenarios["FB10"]
    _seed_feedback_rows(test_db, scenario["feedback_rows"])
    baseline = _rank(scenario)
    ranked = _rank(scenario, _adjustments(test_db, scenario, user_id="user-a"))

    delta = _score_by_id(ranked)["zeta_press"] - _score_by_id(baseline)["zeta_press"]
    assert _ids(ranked) == scenario["expected"]["flag_on_order"]
    assert delta <= -scenario["expected"]["min_score_delta"]


def test_user_note_ignored(feedback_scenarios, test_db):
    scenario = feedback_scenarios["FB01"]
    rows_with_note = [dict(row, user_note="private free text") for row in scenario["feedback_rows"]]
    _seed_feedback_rows(test_db, rows_with_note)
    with_note = _adjustments(test_db, scenario)["zeta_press"].blended_adjustment

    test_db.query(RoutineFeedback).delete()
    test_db.commit()
    _seed_feedback_rows(test_db, scenario["feedback_rows"])
    without_note = _adjustments(test_db, scenario)["zeta_press"].blended_adjustment

    assert with_note == without_note


def test_fb06_blocked_equipment_not_restored(feedback_scenarios, test_db):
    scenario = feedback_scenarios["FB06"]
    _seed_feedback_rows(test_db, scenario["feedback_rows"])
    ranked = _rank(scenario, _adjustments(test_db, scenario))

    assert "barbell_press" not in _ids(ranked)


def test_fb07_pain_trigger_not_restored(feedback_scenarios, test_db):
    scenario = feedback_scenarios["FB07"]
    _seed_feedback_rows(test_db, scenario["feedback_rows"])
    ranked = _rank(scenario, _adjustments(test_db, scenario))

    assert "overhead_press" not in _ids(ranked)


def test_fb08_doms3_not_restored(feedback_scenarios, test_db):
    scenario = feedback_scenarios["FB08"]
    _seed_feedback_rows(test_db, scenario["feedback_rows"])
    ranked = _rank(scenario, _adjustments(test_db, scenario))

    assert "chest_press" not in _ids(ranked)


def test_existing_harness_scenarios_unaffected():
    path = Path(__file__).parent / "evaluation" / "fixtures" / "scenarios.json"
    scenarios = json.loads(path.read_text(encoding="utf-8"))
    harness = EvalHarness()

    results = [harness.run(scenario) for scenario in scenarios]

    assert all(result.hard_violations == 0 for result in results)
    assert all(result.passed for result in results)


def test_harness_no_hard_violation_with_feedback(feedback_scenarios, test_db):
    scenario = feedback_scenarios["FB06"]
    _seed_feedback_rows(test_db, scenario["feedback_rows"])
    result = EvalHarness().run(scenario, feedback_adjustments=_adjustments(test_db, scenario))

    assert result.hard_violations == 0
    assert result.passed is True


def test_feedback_evaluation_fixture_schema(feedback_scenarios):
    required = {
        "id",
        "name",
        "description",
        "candidates",
        "request_context",
        "feedback_rows",
        "expected",
        "mock_llm_output",
    }

    assert len(feedback_scenarios) == 10
    for scenario in feedback_scenarios.values():
        assert required <= set(scenario)
        for row in scenario["feedback_rows"]:
            assert row.get("user_note") is None


def test_flag_on_score_reasons_only_numeric_feedback_summary(feedback_scenarios, test_db):
    scenario = feedback_scenarios["FB01"]
    _seed_feedback_rows(test_db, scenario["feedback_rows"])
    ranked = _rank(scenario, _adjustments(test_db, scenario))
    reason_text = " ".join(reason for item in ranked for reason in item["score_reasons"])

    assert "feedback adj" in reason_text
    assert "user-a" not in reason_text
    assert "other" not in reason_text
    assert "private" not in reason_text


def test_feedback_does_not_introduce_candidate_outside_list(feedback_scenarios, test_db):
    scenario = feedback_scenarios["FB01"]
    _seed_feedback_rows(test_db, scenario["feedback_rows"])
    ranked = _rank(scenario, _adjustments(test_db, scenario))
    candidate_ids = {str(candidate["id"]) for candidate in scenario["candidates"]}

    assert set(_ids(ranked)) <= candidate_ids
